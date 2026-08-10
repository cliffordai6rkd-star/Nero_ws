from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from nero_collection.arms.base import ArmInterface, ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.config import CollectionConfig
from nero_collection.teleop.bilateral import (
    BilateralControlResult,
    BilateralJointController,
    JointMitCommand,
)
from nero_collection.tau_ext_inference import (
    OnlineTauExtInference,
    OnlineTauExtResult,
)
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)


class TeleopSafetyError(RuntimeError):
    pass


@dataclass
class ArmPairRuntime:
    name: str
    leader: ArmInterface
    follower: ArmInterface
    rest_q_leader: np.ndarray
    rest_q_follower: np.ndarray
    controller: BilateralJointController


class MasterSlaveTeleop:
    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        backend = config.teleop.backend
        self.pairs = tuple(
            ArmPairRuntime(
                name=pair.name,
                leader=build_arm(pair.leader, backend),
                follower=build_arm(pair.follower, backend),
                rest_q_leader=_rest_q(pair.leader.rest_q),
                rest_q_follower=_rest_q(pair.follower.rest_q),
                controller=BilateralJointController(
                    config.teleop.command.bilateral_mit,
                    config.realtime_plot.inverse_dynamics,
                ),
            )
            for pair in config.teleop.master_slave
        )
        if config.tau_ext_inference.enabled and len(self.pairs) != 1:
            raise RuntimeError(
                "tau_ext inference currently requires exactly one arm pair"
            )
        self.online_tau_ext = (
            OnlineTauExtInference(
                config.tau_ext_inference,
                config.realtime_plot.inverse_dynamics,
                config.dynamics_processing,
                config.robot_states,
            )
            if config.tau_ext_inference.enabled
            else None
        )
        self.arm_names = tuple(pair.name for pair in self.pairs)
        self._teleop_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._hold_after_reset = False
        self._parked = False
        self._unrecorded_teleop = False
        self._bilateral_active = False
        self._last_bilateral_step_t: float | None = None
        self._last_gripper_command: dict[str, float] = {}
        self._last_gripper_command_mode: dict[str, str] = {}
        self._last_gripper_command_t: dict[str, float] = {}
        self._gripper_command_announced: set[str] = set()
        self._gripper_feedback_warned: set[str] = set()
        self._leader_gripper_feedback_timestamp_us: dict[str, int] = {}
        self._leader_gripper_feedback_change_t: dict[str, float] = {}
        self._leader_gripper_stale_warned: set[str] = set()

    def start(self) -> None:
        log.info("starting Nero master-slave arms over CAN")
        q_state = self.config.robot_states["q"]
        dq_state = self.config.robot_states["velocity"]
        ddq_state = self.config.robot_states["acceleration"]
        for pair in self.pairs:
            log.info("starting pair=%s leader=%s follower=%s", pair.name, pair.leader.name, pair.follower.name)
            pair.leader.connect()
            pair.follower.connect()
            for arm in (pair.leader, pair.follower):
                arm.configure_state_alignment(
                    self.config.teleop.command.state_alignment_delay_s,
                    self.config.teleop.command.sample_rate_hz,
                    q_state.mean_window,
                    q_state.lowpass_cutoff_hz if q_state.lowpass else None,
                    dq_state.lowpass_cutoff_hz if dq_state.lowpass else None,
                    ddq_state.lowpass_cutoff_hz if ddq_state.lowpass else None,
                    self.config.teleop.command.maximum_can_frame_gap_s,
                )
            pair.leader.validate_joint_impedance_support()
            if self.config.teleop.command.control_mode == "mit":
                pair.follower.validate_joint_impedance_support()
            self._log_current_roles(pair)
            self._prepare_pair_for_reset(pair)
            self._init_grippers(pair)
        self.check_input_devices()
        if self.config.teleop.command.reset_on_start:
            log.info("startup reset enabled: both arms reset to follower rest_q")
            self.reset_to_rest()
        else:
            self._set_parked_state()

    def shutdown(self) -> None:
        for pair in self.pairs:
            for arm in (pair.leader, pair.follower):
                try:
                    arm.disconnect()
                except Exception as exc:  # pragma: no cover - shutdown guard
                    log.debug("disconnect failed for %s: %s", arm.name, exc)

    def check_input_devices(self) -> None:
        log.info("checking master-slave teleop input devices")
        timeout_s = self.config.teleop.command.input_ready_timeout_s
        for pair in self.pairs:
            leader_q = self._wait_for_valid_joints(
                lambda: pair.leader.read_state().q,
                timeout_s,
                f"Leader endpoint {pair.leader.name}",
            )
            follower_q = self._wait_for_valid_joints(
                lambda: pair.follower.read_state().q,
                timeout_s,
                f"Follower arm {pair.follower.name}",
            )
            log.info("input ok pair=%s dof=%d", pair.name, leader_q.size)

    def _init_grippers(self, pair: ArmPairRuntime) -> None:
        gripper = self.config.gripper
        if not gripper.enabled:
            return
        if gripper.teleop_enabled or gripper.attach_to in {"leader", "both"}:
            pair.leader.init_gripper(gripper.effector)
            if gripper.teleop_enabled:
                pair.leader.disable_gripper()
        if gripper.teleop_enabled or gripper.attach_to in {"follower", "both"}:
            pair.follower.init_gripper(gripper.effector)

    def _wait_for_valid_joints(self, read_fn, timeout_s: float, label: str) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        last_q = np.empty((0,), dtype=np.float64)
        while time.monotonic() < deadline:
            last_q = np.asarray(read_fn(), dtype=np.float64).reshape(-1)
            if _is_valid_joint_vector(last_q):
                return last_q
            time.sleep(0.05)
        raise RuntimeError(f"{label} did not return valid joint positions within {timeout_s:.1f}s; last={last_q}")

    def idle_step(self) -> None:
        if not self._bilateral_active:
            return
        self._bilateral_step(build_values=False)
        if self._unrecorded_teleop:
            self._update_gripper_teleop({})

    def enter_teleop(self) -> None:
        if self._bilateral_active:
            self._unrecorded_teleop = False
            return
        log.info("entering software bilateral MIT teleoperation")
        for pair in self.pairs:
            self._prepare_pair_for_teleop(pair)
            leader_state = pair.leader.read_state()
            follower_state = pair.follower.read_state()
            pair.controller.activate(leader_state, follower_state)
            self._teleop_reference[pair.name] = (
                leader_state.q.copy(),
                follower_state.q.copy(),
            )
            initial = pair.controller.compute(
                leader_state,
                follower_state,
                timestamp_us=time.monotonic_ns() // 1000,
                **(
                    {"tau_ext_override": np.zeros(7, dtype=np.float64)}
                    if getattr(self, "online_tau_ext", None) is not None
                    else {}
                ),
            )
            self._send_bilateral_commands(pair, initial, follower_state.q)
            log.info(
                "software bilateral reference initialized pair=%s leader_q=%s follower_q=%s",
                pair.name,
                np.array2string(leader_state.q, precision=4),
                np.array2string(follower_state.q, precision=4),
            )
        self._hold_after_reset = False
        self._parked = False
        self._unrecorded_teleop = False
        self._bilateral_active = True
        self._last_bilateral_step_t = time.monotonic()

    def enter_unrecorded_teleop(self) -> None:
        if self._unrecorded_teleop:
            return
        self.enter_teleop()
        log.info("software bilateral teleoperation active without recording")
        self._unrecorded_teleop = True

    @staticmethod
    def _log_current_roles(pair: ArmPairRuntime) -> None:
        leader_role = pair.leader.read_control_role(refresh=True)
        follower_role = pair.follower.read_control_role(refresh=True)
        log.info(
            "current arm roles pair=%s leader_endpoint=%s:%s follower_endpoint=%s:%s",
            pair.name,
            pair.leader.name,
            leader_role or "unknown",
            pair.follower.name,
            follower_role or "unknown",
        )

    @staticmethod
    def _ensure_arm_role(pair_name: str, arm: ArmInterface, expected_role: str) -> bool:
        if expected_role != "follower":
            raise ValueError("Software bilateral control only supports firmware follower mode")
        detected_role = arm.read_control_role()
        if detected_role == expected_role:
            log.info(
                "arm role matches required state pair=%s arm=%s role=%s",
                pair_name,
                arm.name,
                expected_role,
            )
            return False
        log.warning(
            "arm role mismatch pair=%s arm=%s detected=%s required=%s; switching",
            pair_name,
            arm.name,
            detected_role or "unknown",
            expected_role,
        )
        arm.set_follower_mode()
        log.info(
            "arm role switch command sent pair=%s arm=%s required_role=%s; awaiting hardware feedback",
            pair_name,
            arm.name,
            expected_role,
        )
        return True

    def _settle_after_role_switch(self, switched: bool) -> None:
        if switched and self.config.teleop.command.role_switch_settle_s > 0:
            time.sleep(self.config.teleop.command.role_switch_settle_s)

    def _wait_for_arm_role(self, pair_name: str, arm: ArmInterface, expected_role: str) -> None:
        timeout_s = self.config.teleop.command.role_switch_timeout_s
        deadline = time.monotonic() + timeout_s
        last_role: str | None = None
        while time.monotonic() < deadline:
            last_role = arm.read_control_role(refresh=True)
            if last_role == expected_role:
                log.info(
                    "arm role verified from hardware pair=%s arm=%s role=%s",
                    pair_name,
                    arm.name,
                    expected_role,
                )
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Arm role switch not confirmed for {arm.name}: "
            f"expected={expected_role}, detected={last_role or 'unknown'}"
        )

    def _prepare_pair_for_teleop(self, pair: ArmPairRuntime) -> None:
        switched = False
        switched |= self._ensure_arm_role(pair.name, pair.follower, "follower")
        switched |= self._ensure_arm_role(pair.name, pair.leader, "follower")
        self._settle_after_role_switch(switched)
        for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
            try:
                self._wait_for_arm_role(pair.name, arm, "follower")
            except RuntimeError:
                log.warning(
                    "retrying CAN-controlled follower mode before bilateral teleop "
                    "pair=%s logical_role=%s arm=%s",
                    pair.name,
                    role,
                    arm.name,
                )
                arm.set_follower_mode()
                self._settle_after_role_switch(True)
                self._wait_for_arm_role(pair.name, arm, "follower")
            if role == "leader" or self.config.teleop.command.control_mode == "mit":
                arm.configure_joint_impedance_mode()
        log.info(
            "software bilateral control ready pair=%s leader=mit follower=%s",
            pair.name,
            self.config.teleop.command.control_mode,
        )

    def _prepare_pair_for_reset(self, pair: ArmPairRuntime) -> None:
        switched = False
        switched |= self._ensure_arm_role(pair.name, pair.leader, "follower")
        switched |= self._ensure_arm_role(pair.name, pair.follower, "follower")
        self._settle_after_role_switch(switched)
        log.info("confirming both arms enabled before reset pair=%s", pair.name)
        pair.leader.enable()
        pair.follower.enable()
        for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
            try:
                self._wait_for_arm_role(pair.name, arm, "follower")
            except RuntimeError:
                log.warning(
                    "retrying follower mode before reset pair=%s role=%s arm=%s",
                    pair.name,
                    role,
                    arm.name,
                )
                arm.set_follower_mode()
                self._settle_after_role_switch(True)
                arm.enable()
                self._wait_for_arm_role(pair.name, arm, "follower")

    def teleop_step(self) -> tuple[int, dict[str, tuple[str, np.ndarray]]]:
        if not self._bilateral_active:
            raise RuntimeError("Software bilateral teleoperation is not active")
        timestamp_us, values = self._bilateral_step(build_values=True)
        self._update_gripper_teleop(values)
        return timestamp_us, values

    def _bilateral_step(
        self,
        *,
        build_values: bool,
    ) -> tuple[int, dict[str, tuple[str, np.ndarray]]]:
        step_started_t = self._begin_bilateral_step()
        leader_states: list[ArmState] = []
        follower_states: list[ArmState] = []
        q_cmds: list[np.ndarray] = []
        inference_results: list[OnlineTauExtResult] = []
        control_timestamp_us = time.monotonic_ns() // 1000
        for pair in self.pairs:
            leader_state = pair.leader.read_state()
            follower_state = pair.follower.read_state()
            self._check_bilateral_step_deadline(step_started_t, "CAN state read")
            inference_result = None
            q_cmd = _limit_joint_step(
                pair.controller.follower_target(leader_state, follower_state),
                follower_state.q,
                self.config.teleop.command.joint_step_limit_rad,
            )
            online_tau_ext = getattr(self, "online_tau_ext", None)
            if online_tau_ext is not None:
                inference_result = online_tau_ext.estimate_aligned(
                    follower_state.timestamp_us,
                    follower_state.q,
                    follower_state.dq,
                    follower_state.torque,
                    q_cmd,
                )
            result = pair.controller.compute(
                leader_state,
                follower_state,
                timestamp_us=control_timestamp_us,
                **(
                    {
                        "tau_ext_override": (
                            inference_result.tau_ext_pred
                            if inference_result.history_ready
                            else np.zeros(7, dtype=np.float64)
                        )
                    }
                    if inference_result is not None
                    else {}
                ),
            )
            self._check_bilateral_step_deadline(step_started_t, "bilateral compute")
            self._send_bilateral_commands(pair, result, q_cmd)
            leader_states.append(leader_state)
            follower_states.append(follower_state)
            q_cmds.append(q_cmd)
            if inference_result is not None:
                inference_results.append(inference_result)

        timestamp_us = _primary_observation_timestamp_us(follower_states)
        values = (
            self._build_teleop_values(
                leader_states,
                follower_states,
                q_cmds,
                inference_results,
                control_timestamp_us,
            )
            if build_values
            else {}
        )
        return timestamp_us, values

    def _begin_bilateral_step(self) -> float:
        now = time.monotonic()
        previous = getattr(self, "_last_bilateral_step_t", None)
        timeout_s = self.config.teleop.command.control_watchdog_timeout_s
        if previous is not None and now - previous > timeout_s:
            raise TeleopSafetyError(
                f"bilateral control watchdog exceeded: dt={now - previous:.6f}s "
                f"limit={timeout_s:.6f}s"
            )
        self._last_bilateral_step_t = now
        return now

    def _check_bilateral_step_deadline(self, step_started_t: float, stage: str) -> None:
        elapsed_s = time.monotonic() - step_started_t
        timeout_s = self.config.teleop.command.control_watchdog_timeout_s
        if elapsed_s > timeout_s:
            raise TeleopSafetyError(
                f"bilateral control step stalled during {stage}: elapsed={elapsed_s:.6f}s "
                f"limit={timeout_s:.6f}s"
            )

    def _send_bilateral_commands(
        self,
        pair: ArmPairRuntime,
        result: BilateralControlResult,
        follower_q: np.ndarray,
    ) -> None:
        MasterSlaveTeleop._send_mit(pair.leader, result.leader)
        if self.config.teleop.command.control_mode == "position":
            pair.follower.command_joint_positions(np.asarray(follower_q, dtype=np.float64))
            return
        follower_command = JointMitCommand(
            q=np.asarray(follower_q, dtype=np.float64),
            v_des=result.follower.v_des,
            kp=result.follower.kp,
            kd=result.follower.kd,
            t_ff=result.follower.t_ff,
        )
        MasterSlaveTeleop._send_mit(pair.follower, follower_command)

    @staticmethod
    def _send_mit(arm: ArmInterface, command: JointMitCommand) -> None:
        arm.command_joint_impedance(
            command.q,
            command.v_des,
            command.kp,
            command.kd,
            command.t_ff,
        )

    def enter_hold(self) -> None:
        if not self._bilateral_active:
            return
        log.info("leaving bilateral drag mode and holding both arms in MIT")
        for pair in self.pairs:
            leader_state = pair.leader.read_state()
            follower_state = pair.follower.read_state()
            result = pair.controller.hold(leader_state, follower_state)
            self._send_bilateral_commands(pair, result, follower_state.q)
        self._bilateral_active = False
        self._last_bilateral_step_t = None
        self._unrecorded_teleop = False
        self._hold_after_reset = True
        self._parked = True

    def emergency_reset_to_rest(self) -> None:
        """Leave MIT teleoperation, rebuild CAN state streams, and reset both arms."""
        self._bilateral_active = False
        self._unrecorded_teleop = False
        self._last_bilateral_step_t = None
        log.critical("emergency reset requested after collection safety fault")
        q_state = self.config.robot_states["q"]
        dq_state = self.config.robot_states["velocity"]
        ddq_state = self.config.robot_states["acceleration"]
        for pair in self.pairs:
            for arm in (pair.leader, pair.follower):
                arm.set_follower_mode()
                arm.configure_state_alignment(
                    self.config.teleop.command.state_alignment_delay_s,
                    self.config.teleop.command.sample_rate_hz,
                    q_state.mean_window,
                    q_state.lowpass_cutoff_hz if q_state.lowpass else None,
                    dq_state.lowpass_cutoff_hz if dq_state.lowpass else None,
                    ddq_state.lowpass_cutoff_hz if ddq_state.lowpass else None,
                    self.config.teleop.command.maximum_can_frame_gap_s,
                )
        self._settle_after_role_switch(True)
        self.check_input_devices()
        self.reset_to_rest()

    def _update_gripper_teleop(self, values: dict[str, tuple[str, np.ndarray]]) -> None:
        gripper = self.config.gripper
        if not gripper.enabled:
            return
        follower_values: list[np.ndarray] = []
        command_values: list[np.ndarray] = []
        now = time.monotonic()
        command_period = 1.0 / gripper.command_rate_hz
        for pair in self.pairs:
            read_leader = gripper.teleop_enabled or gripper.attach_to in {"leader", "both"}
            read_follower = gripper.teleop_enabled or gripper.attach_to in {"follower", "both"}
            leader_state = pair.leader.read_gripper_state() if read_leader else None
            follower_state = pair.follower.read_gripper_state() if read_follower else None
            if leader_state is not None:
                previous_timestamp = self._leader_gripper_feedback_timestamp_us.get(pair.name)
                if previous_timestamp != leader_state.timestamp_us:
                    self._leader_gripper_feedback_timestamp_us[pair.name] = leader_state.timestamp_us
                    self._leader_gripper_feedback_change_t[pair.name] = now
                    self._leader_gripper_stale_warned.discard(pair.name)
                else:
                    last_change_t = self._leader_gripper_feedback_change_t.get(pair.name, now)
                    if now - last_change_t >= max(1.0, 2.0 * gripper.keepalive_s):
                        if pair.name not in self._leader_gripper_stale_warned:
                            log.warning(
                                "leader gripper feedback is stale pair=%s value=%.6f mode=%s; "
                                "no updated CAN gripper frame received",
                                pair.name,
                                leader_state.value,
                                leader_state.mode,
                            )
                            self._leader_gripper_stale_warned.add(pair.name)
            if follower_state is not None:
                follower_values.append(np.asarray([follower_state.value], dtype=np.float64))

            command_value = np.nan
            valid_leader_state = (
                leader_state is not None
                and np.isfinite(leader_state.value)
                and leader_state.mode == "width"
            )
            if gripper.teleop_enabled and valid_leader_state:
                command_value = float(
                    np.clip(
                        gripper.scale * leader_state.value + gripper.offset_m,
                        gripper.min_width_m,
                        gripper.max_width_m,
                    )
                )
                last_value = self._last_gripper_command.get(pair.name)
                last_mode = self._last_gripper_command_mode.get(pair.name)
                last_t = self._last_gripper_command_t.get(pair.name, float("-inf"))
                changed = (
                    last_value is None
                    or last_mode != "width"
                    or abs(command_value - last_value) >= gripper.deadband_m
                )
                due = now - last_t >= command_period
                keepalive_due = now - last_t >= gripper.keepalive_s
                if due and (changed or keepalive_due):
                    pair.follower.command_gripper(command_value, gripper.force_n, mode="width")
                    if pair.name not in self._gripper_command_announced:
                        log.info(
                            "gripper teleop active pair=%s leader=%.6fm command=%.6fm force=%.3fN",
                            pair.name,
                            leader_state.value,
                            command_value,
                            gripper.force_n,
                        )
                        self._gripper_command_announced.add(pair.name)
                    else:
                        log.debug(
                            "gripper command pair=%s leader=%.6fm command=%.6fm",
                            pair.name,
                            leader_state.value,
                            command_value,
                        )
                    self._last_gripper_command[pair.name] = command_value
                    self._last_gripper_command_mode[pair.name] = "width"
                    self._last_gripper_command_t[pair.name] = now
            elif gripper.teleop_enabled and pair.name not in self._gripper_feedback_warned:
                mode = leader_state.mode if leader_state is not None else "unavailable"
                log.warning(
                    "leader gripper feedback unavailable pair=%s mode=%s; command skipped",
                    pair.name,
                    mode,
                )
                self._gripper_feedback_warned.add(pair.name)
            command_values.append(np.asarray([command_value], dtype=np.float64))

        if follower_values:
            follower_value = _concat(follower_values)
            values["gripper_follower"] = ("gripper", follower_value)
        if command_values:
            values["gripper_cmd"] = ("gripper", _concat(command_values))

    def reset_to_rest(self) -> None:
        command = self.config.teleop.command
        log.info("resetting both arms to each pair's follower rest_q")
        self._hold_after_reset = True
        self._parked = False
        self._unrecorded_teleop = False
        self._last_bilateral_step_t = None
        self._teleop_reference.clear()
        for pair in self.pairs:
            self._prepare_pair_for_reset(pair)
        reset_targets = {
            pair.name: {
                "leader": pair.rest_q_follower.copy(),
                "follower": pair.rest_q_follower.copy(),
            }
            for pair in self.pairs
        }

        self._move_both_arms_to_reset_targets(reset_targets)
        deadline = time.monotonic() + command.reset_timeout_s

        while True:
            time.sleep(command.reset_wait_s)
            reset_errors = self._sample_reset_errors()
            max_error = max(
                float(np.max(np.abs(error)))
                for pair_errors in reset_errors.values()
                for error in pair_errors.values()
            )
            log.info(
                "dual-arm reset check from %d averaged samples: max joint error %.6f rad",
                max(command.reset_test_sample_time, 1),
                max_error,
            )
            if max_error <= command.reset_error_limit_rad:
                log.info("dual-arm reset self-check passed with limit %.6f rad", command.reset_error_limit_rad)
                self._set_parked_state()
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Dual-arm reset self-check failed: max joint error {max_error:.6f} rad "
                    f"> limit {command.reset_error_limit_rad:.6f} rad"
                )
            log.warning(
                "reset error %.6f rad exceeds limit %.6f rad; fine-tuning dual-arm reset targets",
                max_error,
                command.reset_error_limit_rad,
            )
            for pair in self.pairs:
                for role in ("leader", "follower"):
                    error = reset_errors[pair.name][role]
                    current_target = reset_targets[pair.name][role]
                    corrected_target = current_target + error
                    reset_targets[pair.name][role] = _limit_joint_step(
                        corrected_target,
                        current_target,
                        command.joint_step_limit_rad,
                    )
                    log.info(
                        "reset fine-tune pair=%s role=%s mean_error=%s target=%s",
                        pair.name,
                        role,
                        np.array2string(error, precision=6, suppress_small=True),
                        np.array2string(reset_targets[pair.name][role], precision=6, suppress_small=True),
                    )
            self._move_both_arms_to_reset_targets(reset_targets)

    def _move_both_arms_to_reset_targets(
        self,
        reset_targets: dict[str, dict[str, np.ndarray]],
    ) -> None:
        command = self.config.teleop.command
        timeout_s = command.reset_timeout_s
        start_q: dict[str, dict[str, np.ndarray]] = {}
        max_delta = 0.0
        for pair in self.pairs:
            start_q[pair.name] = {}
            for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                q = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
                target = reset_targets[pair.name][role]
                if not _is_valid_joint_vector(q) or q.size != target.size:
                    raise RuntimeError(f"Invalid {role} reset start joints from {arm.name}: {q}")
                start_q[pair.name][role] = q
                max_delta = max(max_delta, float(np.max(np.abs(target - q))))

        if not command.reset_interpolation_enabled or max_delta < 1e-9:
            steps = 1
        else:
            duration_s = max(command.reset_min_duration_s, max_delta / command.reset_joint_speed_rad_s)
            steps = max(
                1,
                math.ceil(duration_s * command.reset_interpolation_rate_hz),
                math.ceil(max_delta / command.reset_max_step_rad),
            )
        effective_duration_s = steps / command.reset_interpolation_rate_hz if steps > 1 else 0.0
        log.info(
            "dual-arm reset interpolation steps=%d duration=%.3fs max_delta=%.4frad",
            steps,
            effective_duration_s,
            max_delta,
        )

        interpolation_start = time.monotonic()
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            # Send both commands before sleeping so both arms advance together.
            for pair in self.pairs:
                for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                    start = start_q[pair.name][role]
                    target = reset_targets[pair.name][role]
                    arm.move_joints(start + alpha * (target - start))
            if steps > 1:
                next_t = interpolation_start + step_index / command.reset_interpolation_rate_hz
                remaining = next_t - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

        timed_out: list[str] = []
        for pair in self.pairs:
            for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                if not arm.wait_motion_done(timeout_s):
                    timed_out.append(f"{pair.name}:{role}:{arm.name}")
        if timed_out:
            raise RuntimeError(f"Timed out waiting for reset arms: {timed_out}")

    def _set_parked_state(self) -> None:
        self._hold_after_reset = True
        self._parked = True
        self._unrecorded_teleop = False
        self._bilateral_active = False
        self._last_bilateral_step_t = None
        self._teleop_reference.clear()
        log.info("both arms parked in CAN-controlled follower mode; waiting for r or t")

    def _sample_reset_errors(self) -> dict[str, dict[str, np.ndarray]]:
        command = self.config.teleop.command
        sample_count = max(command.reset_test_sample_time, 1)
        sample_period = 1.0 / max(command.idle_rate_hz, 1.0)
        samples: dict[str, dict[str, list[np.ndarray]]] = {
            pair.name: {"leader": [], "follower": []} for pair in self.pairs
        }
        for sample_index in range(sample_count):
            for pair in self.pairs:
                for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                    q = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
                    if not _is_valid_joint_vector(q) or q.size != pair.rest_q_follower.size:
                        raise RuntimeError(
                            f"{role.capitalize()} arm {arm.name} returned invalid reset joints: {q}"
                        )
                    samples[pair.name][role].append(q)
            if sample_index + 1 < sample_count:
                time.sleep(sample_period)
        return {
            pair.name: {
                role: pair.rest_q_follower - np.mean(samples[pair.name][role], axis=0)
                for role in ("leader", "follower")
            }
            for pair in self.pairs
        }

    def _build_teleop_values(
        self,
        leader_states: list[ArmState],
        follower_states: list[ArmState],
        q_cmds: list[np.ndarray],
        inference_results: list[OnlineTauExtResult] | None = None,
        control_timestamp_us: int | None = None,
    ) -> dict[str, tuple[str, np.ndarray]]:
        values: dict[str, tuple[str, np.ndarray]] = {}
        states = self.config.robot_states

        if states.get("q") and states["q"].enabled:
            values["q_leader"] = ("q", _concat([state.q for state in leader_states]))
            values["q_follower"] = ("q", _concat([state.q for state in follower_states]))
            values["q_cmd"] = ("q", _concat(q_cmds))

        if states.get("velocity") and states["velocity"].enabled:
            values["dq_leader"] = ("velocity", _concat([state.dq for state in leader_states]))
            values["dq_follower"] = ("velocity", _concat([state.dq for state in follower_states]))

        if states.get("acceleration") and states["acceleration"].enabled:
            values["ddq_leader"] = ("acceleration", _concat([state.ddq for state in leader_states]))
            values["ddq_follower"] = ("acceleration", _concat([state.ddq for state in follower_states]))

        if states.get("ee_pose") and states["ee_pose"].enabled:
            values["ee_pose_follower"] = (
                "ee_pose",
                _pose_stack([state.ee_pose for state in follower_states]),
            )

        if states.get("torque") and states["torque"].enabled:
            values["tau_leader"] = ("torque", _concat([state.torque for state in leader_states]))
            follower_tau = _concat([state.torque for state in follower_states])
            values["tau_follower"] = ("torque", follower_tau)

        if states.get("current") and states["current"].enabled:
            values["current_leader"] = ("current", _concat([state.current for state in leader_states]))
            values["current_follower"] = ("current", _concat([state.current for state in follower_states]))

        if follower_states:
            values["control_timestamp_us"] = (
                "timestamp",
                np.asarray(control_timestamp_us or 0, dtype=np.int64),
            )
            values["state_acquired_timestamp_follower_us"] = (
                "timestamp",
                np.asarray(
                    [state.acquired_timestamp_us for state in follower_states],
                    dtype=np.int64,
                ),
            )
            q_sources = [
                _timestamp_vector(state.q_component_timestamp_us, state.timestamp_us, state.q.size)
                for state in follower_states
            ]
            motor_sources = [
                _timestamp_vector(state.motor_timestamp_us, state.timestamp_us, state.q.size)
                for state in follower_states
            ]
            values["q_source_timestamp_follower_us"] = (
                "timestamp",
                np.concatenate(q_sources),
            )
            values["motor_source_timestamp_follower_us"] = (
                "timestamp",
                np.concatenate(motor_sources),
            )
            values["state_source_skew_follower_us"] = (
                "duration",
                np.asarray(
                    [
                        _source_timestamp_skew_us(q_source, motor_source)
                        for q_source, motor_source in zip(q_sources, motor_sources)
                    ],
                    dtype=np.int64,
                ),
            )

        if inference_results:
            values["ddq_kf_causal"] = (
                "acceleration",
                _concat([result.ddq_kf_causal for result in inference_results]),
            )
            values["tau_id"] = (
                "torque",
                _concat([result.tau_id for result in inference_results]),
            )
            values["tau_id_filtered"] = (
                "torque",
                _concat([result.tau_id_filtered for result in inference_results]),
            )
            values["tau_f_pred"] = (
                "torque",
                _concat([result.tau_f_pred for result in inference_results]),
            )
            values["tau_next_pred"] = (
                "torque",
                _concat([result.tau_next_pred for result in inference_results]),
            )
            values["tau_ext_cal_raw"] = (
                "torque",
                _concat(
                    [
                        result.tau_ext_cal
                        if result.tau_ext_cal_raw is None
                        else result.tau_ext_cal_raw
                        for result in inference_results
                    ]
                ),
            )
            values["tau_ext_pred_raw"] = (
                "torque",
                _concat(
                    [
                        result.tau_ext_pred
                        if result.tau_ext_pred_raw is None
                        else result.tau_ext_pred_raw
                        for result in inference_results
                    ]
                ),
            )
            values["tau_ext_cal"] = (
                "torque",
                _concat([result.tau_ext_cal for result in inference_results]),
            )
            values["tau_ext_pred"] = (
                "torque",
                _concat([result.tau_ext_pred for result in inference_results]),
            )

        return values


def _rest_q(rest_q: tuple[float, ...]) -> np.ndarray:
    if rest_q:
        return np.asarray(rest_q, dtype=np.float64)
    return np.zeros(7, dtype=np.float64)


def _primary_observation_timestamp_us(states: list[ArmState]) -> int:
    if not states:
        return now_us()
    state = states[0]
    return int(
        state.q_timestamp_us
        or state.q_acquired_timestamp_us
        or state.timestamp_us
        or state.acquired_timestamp_us
        or now_us()
    )


def _concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values], axis=0)


def _timestamp_vector(value: np.ndarray, fallback: int, size: int) -> np.ndarray:
    timestamps = np.asarray(value, dtype=np.int64).reshape(-1)
    if timestamps.size == size and np.all(timestamps > 0):
        return timestamps.copy()
    return np.full(size, int(fallback), dtype=np.int64)


def _source_timestamp_skew_us(q_source: np.ndarray, motor_source: np.ndarray) -> int:
    timestamps = np.concatenate((q_source, motor_source))
    timestamps = timestamps[timestamps > 0]
    if timestamps.size == 0:
        return 0
    return int(np.max(timestamps) - np.min(timestamps))


def _pose_stack(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        return np.empty((0, 4, 4), dtype=np.float64)
    normalized = [np.asarray(pose, dtype=np.float64).reshape(4, 4) for pose in poses]
    if len(normalized) == 1:
        return normalized[0]
    return np.stack(normalized, axis=0)


def _limit_joint_step(target_q: np.ndarray, current_q: np.ndarray, max_step_rad: float | None) -> np.ndarray:
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if not _is_valid_joint_vector(target):
        raise RuntimeError(f"Invalid target joint vector: {target}")
    if not _is_valid_joint_vector(current):
        return target
    if max_step_rad is None or current.size != target.size:
        return target
    delta = np.clip(target - current, -max_step_rad, max_step_rad)
    return current + delta


def _max_abs_error(actual_q: np.ndarray, target_q: np.ndarray) -> float:
    actual = np.asarray(actual_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if actual.size != target.size:
        raise RuntimeError(f"Reset q size mismatch: actual={actual.size}, target={target.size}")
    return float(np.max(np.abs(actual - target)))


def _is_valid_joint_vector(q: np.ndarray) -> bool:
    array = np.asarray(q, dtype=np.float64).reshape(-1)
    return array.size > 0 and bool(np.isfinite(array).all())
