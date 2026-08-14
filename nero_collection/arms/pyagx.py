from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.arms.kinematics import pose6_to_matrix
from nero_collection.coordinates import nero_v120_joint_velocity
from nero_collection.config import ArmEndpointConfig
from nero_collection.socketcan import interface_usb_serial
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)

_LEADER_MODE_SEND_ATTEMPTS = 3
_LEADER_MODE_SEND_INTERVAL_S = 0.15
_NERO_SAFE_MIT_TORQUE_LIMIT_NM = 16.0


@dataclass
class PyAgxArmAdapter:
    config: ArmEndpointConfig
    name: str = field(init=False)
    dof: int = 7
    _robot: Any = field(init=False, default=None)
    _gripper: Any = field(init=False, default=None)
    _configured_role: str | None = None
    _leader_mode_commanded: bool = False
    _leader_feedback_timestamp: float | None = None
    _leader_gripper_feedback_baseline: float | None = None

    def __post_init__(self) -> None:
        self.name = self.config.name

    def connect(self) -> None:
        if self.config.interface == "socketcan" and self.config.usb_serial is not None:
            actual_serial = interface_usb_serial(self.config.channel)
            if actual_serial != self.config.usb_serial:
                raise RuntimeError(
                    f"Refusing to connect arm {self.name}: {self.config.channel} has "
                    f"USB-CAN serial {actual_serial!r}, expected {self.config.usb_serial!r}. "
                    "Resolve the configured USB-CAN binding before connecting."
                )
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        except ImportError as exc:
            try:
                from PyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "pyAgxArm is not installed. Install AgileX pyAgxArm or use teleop.backend=mock."
                ) from exc

        firmware = _enum_value(NeroFW, self.config.firmware)
        kwargs = dict(self.config.config_kwargs)
        if self.config.can_id is not None:
            kwargs.setdefault("can_id", self.config.can_id)

        sdk_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            comm="can",
            firmeware_version=firmware,
            channel=self.config.channel,
            interface=self.config.interface,
            bitrate=self.config.bitrate,
            **kwargs,
        )
        try:
            self._robot = AgxArmFactory.create_arm(sdk_config)
        except KeyError as exc:
            available = [name for name in dir(NeroFW) if not name.startswith("_")]
            raise RuntimeError(
                f"Failed to create Nero arm {self.name} with firmware={self.config.firmware!r}. "
                f"Current pyAgxArm NeroFW values: {available}. "
                "Update configs/master_slave_can.yaml teleop.master_slave.*.firmware."
            ) from exc
        log.info(
            "connecting Nero arm %s on %s id=%s interface=%s bitrate=%s",
            self.name,
            self.config.channel,
            self.config.can_id,
            self.config.interface,
            self.config.bitrate,
        )
        self._robot.connect()

    def disconnect(self) -> None:
        if self._robot is not None:
            _call_if_exists(self._robot, ("disconnect", "close", "shutdown"))

    def enable(self) -> None:
        robot = self._require_robot()
        ret = False
        for attempt in range(60):
            ret = bool(robot.enable())
            if ret:
                log.info("enabled arm %s after %d attempt(s)", self.name, attempt + 1)
                break
            if attempt == 0:
                log.warning("enable failed for %s; sending reset and clear_joint_error before retry", self.name)
                _call_if_exists(robot, ("reset",))
                clear_joint_error = getattr(robot, "clear_joint_error", None)
                if callable(clear_joint_error):
                    clear_joint_error(255)
            elif attempt % 10 == 0:
                log.warning("still waiting for %s enable; attempt=%d/60", self.name, attempt + 1)
            time.sleep(0.05)
        if not ret:
            joint_enable = _safe_call(robot, "get_joints_enable_status_list")
            arm_status = _unwrap_message(_safe_call(robot, "get_arm_status"))
            raise RuntimeError(
                f"Failed to enable arm {self.name}. Check E-stop, arm power, CAN wiring, "
                f"{self.config.channel} state, and whether the controller needs a power cycle. "
                f"joint_enable={joint_enable}, arm_status={arm_status}"
            )
        enabled = getattr(robot, "is_enabled", None)
        if callable(enabled):
            start = time.monotonic()
            while not enabled() and time.monotonic() - start < 5.0:
                time.sleep(0.05)
            if not enabled():
                raise RuntimeError(f"Timed out waiting for arm {self.name} to enable")

    def disable(self) -> None:
        robot = self._require_robot()
        disable = getattr(robot, "disable", None)
        if not callable(disable):
            raise RuntimeError(f"Arm {self.name} does not expose disable()")
        for attempt in range(20):
            if bool(disable()):
                log.info("disabled arm %s after %d attempt(s)", self.name, attempt + 1)
                return
            time.sleep(0.05)
        log.warning("disable did not confirm for arm %s", self.name)

    def set_leader_mode(self) -> None:
        robot = self._require_robot()
        self._leader_feedback_timestamp = _message_timestamp(
            _call_method(robot, "get_leader_joint_angles")
        )
        if self._gripper is not None:
            control_reader = getattr(self._gripper, "get_gripper_ctrl_states", None)
            self._leader_gripper_feedback_baseline = _message_timestamp(
                control_reader() if callable(control_reader) else None
            )
        successful_sends = 0
        last_error: Exception | None = None
        for attempt in range(1, _LEADER_MODE_SEND_ATTEMPTS + 1):
            try:
                ret = robot.set_leader_mode()
                if ret is False:
                    raise RuntimeError("SDK returned False")
                successful_sends += 1
                log.debug(
                    "sent leader mode command arm=%s attempt=%d/%d",
                    self.name,
                    attempt,
                    _LEADER_MODE_SEND_ATTEMPTS,
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "leader mode command send failed arm=%s attempt=%d/%d: %s",
                    self.name,
                    attempt,
                    _LEADER_MODE_SEND_ATTEMPTS,
                    exc,
                )
            if attempt < _LEADER_MODE_SEND_ATTEMPTS:
                time.sleep(_LEADER_MODE_SEND_INTERVAL_S)
        if successful_sends == 0:
            raise RuntimeError(f"Failed to send leader mode command to {self.name}") from last_error
        self._configured_role = "leader"
        self._leader_mode_commanded = True

    def set_follower_mode(self) -> None:
        ret = self._require_robot().set_follower_mode()
        if ret is False:
            raise RuntimeError(f"Failed to set {self.name} to follower mode")
        self._configured_role = "follower"
        self._leader_mode_commanded = False
        self._leader_feedback_timestamp = None
        self._leader_gripper_feedback_baseline = None

    def set_normal_mode(self) -> None:
        robot = self._require_robot()
        if _call_if_exists(robot, ("set_normal_mode", "set_servo_mode", "set_position_mode")) is False:
            log.warning("arm %s does not expose a normal/position mode method", self.name)
        self._configured_role = None
        self._leader_mode_commanded = False
        self._leader_feedback_timestamp = None
        self._leader_gripper_feedback_baseline = None

    def read_control_role(self, refresh: bool = False) -> str | None:
        if self._configured_role is not None and not refresh:
            return self._configured_role
        commanded_role = self._configured_role
        verify_leader_feedback = self._leader_mode_commanded and commanded_role == "leader"
        robot = self._require_robot()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            status = _unwrap_message(_call_method(robot, "get_arm_status"))
            ctrl_mode = _enum_int(getattr(status, "ctrl_mode", None))
            if ctrl_mode == 0x06:
                self._configured_role = "leader"
                return self._configured_role
            if ctrl_mode == 0x01 and not verify_leader_feedback:
                self._configured_role = "follower"
                return self._configured_role
            leader_reader = getattr(robot, "get_leader_joint_angles", None)
            if callable(leader_reader):
                leader_feedback = leader_reader()
                leader_q = _as_float_array(leader_feedback, self.dof)
                feedback_timestamp = _message_timestamp(leader_feedback)
                feedback_is_fresh = (
                    feedback_timestamp is not None
                    and (
                        self._leader_feedback_timestamp is None
                        or feedback_timestamp > self._leader_feedback_timestamp
                    )
                )
                if np.isfinite(leader_q).all() and (feedback_is_fresh or not verify_leader_feedback):
                    self._configured_role = "leader"
                    if feedback_timestamp is not None:
                        self._leader_feedback_timestamp = feedback_timestamp
                    return self._configured_role
            time.sleep(0.02)
        return None

    def read_state(self) -> ArmState:
        robot = self._require_robot()
        acquired_timestamp_us = now_us()
        reader = (
            getattr(robot, "get_leader_joint_angles", None)
            if self._leader_mode_commanded
            else getattr(robot, "get_joint_angles", None)
        )
        q_message = reader() if callable(reader) else None
        q = _as_float_array(q_message, self.dof)
        q_timestamp_us = _message_timestamp_us(q_message)
        q_component_timestamp_us = np.full(self.dof, q_timestamp_us, dtype=np.int64)
        motor_states = _read_motor_states(robot, self.dof, firmware=self.config.firmware)
        dq = motor_states["velocity"]
        ddq = np.full(self.dof, np.nan, dtype=np.float64)
        ee_pose = _read_pose(robot)
        q_acquired_timestamp_us = acquired_timestamp_us
        timestamp_us = q_timestamp_us or acquired_timestamp_us
        state = ArmState(
            q=q,
            dq=dq,
            ddq=ddq,
            ee_pose=ee_pose,
            torque=motor_states["torque"],
            current=motor_states["current"],
            timestamp_us=timestamp_us,
            acquired_timestamp_us=acquired_timestamp_us,
            q_timestamp_us=q_timestamp_us,
            q_acquired_timestamp_us=q_acquired_timestamp_us,
            q_component_timestamp_us=q_component_timestamp_us,
            q_source_before_timestamp_us=q_component_timestamp_us.copy(),
            q_source_after_timestamp_us=q_component_timestamp_us.copy(),
            motor_timestamp_us=motor_states["timestamp_us"],
            motor_acquired_timestamp_us=motor_states["acquired_timestamp_us"],
        )
        return state

    def read_leader_joint_positions(self) -> np.ndarray:
        return self.read_state().q

    def command_joint_positions(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.dof or not np.isfinite(q).all():
            raise RuntimeError(f"Refusing to send invalid joint command to {self.name}: {q}")
        ret = self._require_robot().move_js(q.tolist())
        if ret is False:
            raise RuntimeError(f"move_js failed on arm {self.name}")

    def command_joint_impedance(
        self,
        q: np.ndarray,
        v_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        t_ff: np.ndarray,
    ) -> None:
        vectors = {
            "q": np.asarray(q, dtype=np.float64).reshape(-1),
            "v_des": np.asarray(v_des, dtype=np.float64).reshape(-1),
            "kp": np.asarray(kp, dtype=np.float64).reshape(-1),
            "kd": np.asarray(kd, dtype=np.float64).reshape(-1),
            "t_ff": np.asarray(t_ff, dtype=np.float64).reshape(-1),
        }
        for label, values in vectors.items():
            if values.size != self.dof or not np.isfinite(values).all():
                raise RuntimeError(f"Refusing to send invalid MIT {label} to {self.name}: {values}")
        excessive_torque = np.abs(vectors["t_ff"]) > _NERO_SAFE_MIT_TORQUE_LIMIT_NM
        if np.any(excessive_torque):
            joints = (np.flatnonzero(excessive_torque) + 1).tolist()
            raise RuntimeError(
                f"Refusing unsafe MIT feed-forward torque for arm {self.name}: "
                f"joints={joints} t_ff={vectors['t_ff']} "
                f"limit=+/-{_NERO_SAFE_MIT_TORQUE_LIMIT_NM:.1f} N.m"
            )

        robot = self._require_robot()
        move_mit = getattr(robot, "move_mit", None)
        if not callable(move_mit):
            raise RuntimeError(
                "Installed pyAgxArm does not expose Nero move_mit(); update pyAgxArm "
                "before using software bilateral MIT control"
            )
        for joint_index in range(1, self.dof + 1):
            idx = joint_index - 1
            ret = move_mit(
                joint_index=joint_index,
                p_des=float(vectors["q"][idx]),
                v_des=float(vectors["v_des"][idx]),
                kp=float(vectors["kp"][idx]),
                kd=float(vectors["kd"][idx]),
                t_ff=float(vectors["t_ff"][idx]),
            )
            if ret is False:
                raise RuntimeError(f"move_mit failed on arm {self.name} joint {joint_index}")

    def validate_joint_impedance_support(self) -> None:
        robot = self._require_robot()
        required = ("move_mit", "set_motion_mode", "set_auto_set_motion_mode_enabled")
        missing = [name for name in required if not callable(getattr(robot, name, None))]
        if missing:
            raise RuntimeError(
                "Installed pyAgxArm does not expose the Nero bilateral MIT API: "
                + ", ".join(missing)
            )

    def configure_joint_impedance_mode(self) -> None:
        robot = self._require_robot()
        self.validate_joint_impedance_support()
        robot.set_auto_set_motion_mode_enabled(True)
        robot.set_motion_mode("mit")
        robot.set_auto_set_motion_mode_enabled(False)
        log.info("configured persistent MIT motion mode arm=%s", self.name)

    def move_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.dof or not np.isfinite(q).all():
            raise RuntimeError(f"Refusing to send invalid reset joint command to {self.name}: {q}")
        robot = self._require_robot()
        auto_mode = getattr(robot, "set_auto_set_motion_mode_enabled", None)
        if callable(auto_mode):
            auto_mode(True)
        ret = robot.move_j(q.tolist())
        if ret is False:
            raise RuntimeError(f"move_j failed on arm {self.name}")

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        robot = self._require_robot()
        time.sleep(0.5)
        start_t = time.monotonic()
        while True:
            status = _unwrap_message(_call_method(robot, "get_arm_status"))
            if status is not None and getattr(status, "motion_status", None) == 0:
                return True
            if time.monotonic() - start_t > timeout_s:
                log.warning("wait_motion_done timeout for %s after %.1fs", self.name, timeout_s)
                return False
            time.sleep(poll_interval_s)

    def init_gripper(self, effector: str = "AGX_GRIPPER") -> None:
        robot = self._require_robot()
        options = getattr(robot, "OPTIONS", None)
        effector_options = getattr(options, "EFFECTOR", None) if options is not None else None
        effector_value = getattr(effector_options, effector, effector)
        init_effector = getattr(robot, "init_effector", None)
        if not callable(init_effector):
            raise RuntimeError(f"Arm {self.name} does not expose init_effector()")
        self._gripper = init_effector(effector_value)
        server_version = _component_version(self._gripper)
        log.info(
            "initialized gripper %s on arm=%s arm_firmware=%s "
            "server_type=%s server_version=%s",
            effector,
            self.name,
            self.config.firmware,
            type(self._gripper).__name__,
            server_version if server_version is not None else "unavailable",
        )

    def read_gripper_state(self) -> GripperState:
        if self._gripper is None:
            return GripperState(value=np.nan, force=np.nan, timestamp_us=0, mode="unknown")
        status_reader = getattr(self._gripper, "get_gripper_status", None)
        status_message = status_reader() if callable(status_reader) else None
        status = _unwrap_message(status_message)
        mode = str(getattr(status, "mode", "unknown")).lower()
        if mode not in {"width", "angle"}:
            mode = "unknown"
        return _gripper_state(status_message, mode)

    def read_leader_gripper_state(self) -> GripperState:
        if self._gripper is None:
            return GripperState(value=np.nan, force=np.nan, timestamp_us=0, mode="unknown")
        control_reader = getattr(self._gripper, "get_gripper_ctrl_states", None)
        control_message = control_reader() if callable(control_reader) else None
        control = _unwrap_message(control_message)
        control_timestamp = _message_timestamp(control_message)
        if self._leader_mode_commanded:
            baseline = self._leader_gripper_feedback_baseline
            if control_timestamp is None or (baseline is not None and control_timestamp <= baseline):
                return GripperState(value=np.nan, force=np.nan, timestamp_us=0, mode="unknown")
        status_code = _enum_int(getattr(control, "status_code", None))
        if status_code in {0, 1, 2, 3}:
            return _gripper_state(control_message, "width")
        if status_code in {4, 5, 6, 7}:
            return _gripper_state(control_message, "angle")
        if self._leader_mode_commanded:
            return GripperState(value=np.nan, force=np.nan, timestamp_us=0, mode="unknown")
        return self.read_gripper_state()

    def disable_gripper(self) -> None:
        if self._gripper is None:
            raise RuntimeError(f"Gripper on arm {self.name} has not been initialized")
        disable_gripper = getattr(self._gripper, "disable_gripper", None)
        if not callable(disable_gripper):
            raise RuntimeError(f"Gripper on arm {self.name} does not expose disable_gripper()")
        status_reader = getattr(self._gripper, "get_gripper_status", None)
        for attempt in range(1, 4):
            disable_gripper()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                status = _unwrap_message(status_reader()) if callable(status_reader) else None
                foc_status = getattr(status, "foc_status", None)
                if getattr(foc_status, "driver_enable_status", None) is False:
                    log.info(
                        "disabled leader-input gripper on arm %s after %d attempt(s)",
                        self.name,
                        attempt,
                    )
                    return
                time.sleep(0.02)
            if attempt < 3:
                log.warning(
                    "leader-input gripper still enabled on arm %s; retrying disable (%d/3)",
                    self.name,
                    attempt + 1,
                )
        raise RuntimeError(
            f"Failed to disable leader-input gripper on {self.name}; "
            "manual gripper teleoperation cannot start"
        )

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        if self._gripper is None:
            raise RuntimeError(f"Gripper on arm {self.name} has not been initialized")
        value = float(value)
        force_n = float(force_n)
        if not np.isfinite(value) or not np.isfinite(force_n) or force_n < 0:
            raise RuntimeError(
                f"Refusing invalid gripper command on {self.name}: "
                f"value={value}, force_n={force_n}, mode={mode}"
            )
        if mode == "width":
            method_name = "move_gripper_m"
        elif mode == "angle":
            method_name = "move_gripper_deg"
        else:
            raise RuntimeError(f"Unsupported gripper mode on {self.name}: {mode!r}")
        move_gripper = getattr(self._gripper, method_name, None)
        if not callable(move_gripper):
            raise RuntimeError(f"Gripper on arm {self.name} does not expose {method_name}()")
        move_gripper(value=value, force=force_n)

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError(f"Arm {self.name} has not been connected")
        return self._robot


def _enum_value(enum_cls: Any, name: str) -> Any:
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    upper = name.upper()
    for attr in dir(enum_cls):
        if attr.upper() == upper:
            return getattr(enum_cls, attr)
    return name


def _enum_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(value.value)
        except (AttributeError, TypeError, ValueError):
            return None


def _message_timestamp(value: Any) -> float | None:
    timestamp = getattr(value, "timestamp", None)
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(timestamp) or timestamp <= 0:
        return None
    return timestamp


def _message_timestamp_us(value: Any) -> int:
    timestamp = _message_timestamp(value)
    return int(round(timestamp * 1_000_000)) if timestamp is not None else 0


def _gripper_state(message: Any, mode: str) -> GripperState:
    value = _unwrap_message(message)
    timestamp = _message_timestamp(message)
    return GripperState(
        value=_field_float(value, ("value", "position", "pos")),
        force=_field_float(value, ("force", "torque")),
        timestamp_us=int(round(timestamp * 1_000_000)) if timestamp is not None else 0,
        mode=mode,
    )


def _call_if_exists(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method()
    return None


def _call_method(obj: Any, name: str) -> Any:
    method = getattr(obj, name, None)
    if callable(method):
        return method()
    return None


def _safe_call(obj: Any, name: str) -> Any:
    try:
        return _call_method(obj, name)
    except Exception as exc:  # pragma: no cover - diagnostic guard
        return f"<{name} failed: {exc}>"


def _component_version(component: Any) -> str | None:
    for method_name in (
        "get_server_version",
        "get_firmware_version",
        "get_firmware",
        "get_version",
    ):
        method = getattr(component, method_name, None)
        if not callable(method):
            continue
        try:
            value = _unwrap_message(method())
        except Exception as exc:  # pragma: no cover - hardware diagnostic guard
            log.debug("gripper %s query failed: %s", method_name, exc)
            continue
        if value is not None:
            return str(value)
    for attribute_name in ("server_version", "firmware_version", "version"):
        value = getattr(component, attribute_name, None)
        if value is not None:
            return str(value)
    return None


def _as_float_array(value: Any, expected_size: int) -> np.ndarray:
    value = _unwrap_message(value)
    if value is None:
        return np.full(expected_size, np.nan, dtype=np.float64)
    if hasattr(value, "joint_angle"):
        value = value.joint_angle
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == expected_size:
        return array
    if array.size == 0:
        return np.full(expected_size, np.nan, dtype=np.float64)
    if array.size > expected_size:
        return array[:expected_size]
    padded = np.full(expected_size, np.nan, dtype=np.float64)
    padded[: array.size] = array
    return padded


def _read_motor_states(
    robot: Any,
    dof: int,
    *,
    firmware: str | None = None,
) -> dict[str, np.ndarray]:
    velocity = np.full(dof, np.nan, dtype=np.float64)
    torque = np.full(dof, np.nan, dtype=np.float64)
    current = np.full(dof, np.nan, dtype=np.float64)
    timestamp_us = np.zeros(dof, dtype=np.int64)
    acquired_timestamp_us = np.zeros(dof, dtype=np.int64)
    reader = getattr(robot, "get_motor_states", None) or getattr(robot, "get_motor_state", None)
    if not callable(reader):
        return {
            "velocity": velocity,
            "torque": torque,
            "current": current,
            "timestamp_us": timestamp_us,
            "acquired_timestamp_us": acquired_timestamp_us,
        }
    for idx in range(dof):
        try:
            state = reader(idx + 1)
        except TypeError:
            state = reader(idx)
        acquired_timestamp_us[idx] = now_us()
        state_message = state
        state = _unwrap_message(state_message)
        if state is None:
            continue
        velocity_value = _field_float(state, ("velocity", "vel", "speed"))
        if str(firmware).strip().upper() == "V120":
            velocity_value = nero_v120_joint_velocity(velocity_value, idx)
        velocity[idx] = velocity_value
        torque[idx] = _field_float(state, ("torque", "tau"))
        current[idx] = _field_float(state, ("current", "iq", "motor_current"))
        timestamp_us[idx] = _message_timestamp_us(state_message)
    return {
        "velocity": velocity,
        "torque": torque,
        "current": current,
        "timestamp_us": timestamp_us,
        "acquired_timestamp_us": acquired_timestamp_us,
    }


def _field_float(obj: Any, names: tuple[str, ...]) -> float:
    obj = _unwrap_message(obj)
    if obj is None:
        return np.nan
    for name in names:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _read_pose(robot: Any) -> np.ndarray:
    for method_name in ("get_tcp_pose", "get_flange_pose", "get_ee_pose", "get_end_pose"):
        method = getattr(robot, method_name, None)
        if callable(method):
            try:
                return pose6_to_matrix(_unwrap_message(method()))
            except Exception as exc:  # pragma: no cover - hardware guard
                log.debug("pose method %s failed: %s", method_name, exc)
    return np.eye(4, dtype=np.float64)


def _unwrap_message(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "msg"):
        return getattr(value, "msg")
    if hasattr(value, "data"):
        return getattr(value, "data")
    return value
