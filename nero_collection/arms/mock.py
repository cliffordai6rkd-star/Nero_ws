from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.arms.kinematics import matrix_from_joint_stub
from nero_collection.config import ArmEndpointConfig
from nero_collection.time_utils import now_us


@dataclass
class MockArm:
    config: ArmEndpointConfig
    name: str = field(init=False)
    dof: int = field(init=False)
    _q: np.ndarray = field(init=False)
    _dq: np.ndarray = field(init=False)
    _ddq: np.ndarray = field(init=False)
    _target_q: np.ndarray = field(init=False)
    _last_t: float = field(default_factory=time.monotonic)
    _last_dq: np.ndarray = field(init=False)
    _connected: bool = False
    _mode: str = "normal"
    _gripper_value: float = 0.0
    _gripper_force: float = 0.0
    _gripper_mode: str = "width"

    def __post_init__(self) -> None:
        self.name = self.config.name
        rest_q = np.asarray(self.config.rest_q or (0.0,) * 7, dtype=np.float64)
        self.dof = int(rest_q.size)
        self._q = rest_q.copy()
        self._target_q = rest_q.copy()
        self._dq = np.zeros(self.dof, dtype=np.float64)
        self._ddq = np.zeros(self.dof, dtype=np.float64)
        self._last_dq = np.zeros(self.dof, dtype=np.float64)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def enable(self) -> None:
        self._ensure_connected()

    def disable(self) -> None:
        self._ensure_connected()

    def set_leader_mode(self) -> None:
        self._mode = "leader"

    def set_follower_mode(self) -> None:
        self._mode = "follower"

    def set_normal_mode(self) -> None:
        self._mode = "normal"

    def read_control_role(self, refresh: bool = False) -> str | None:
        if self._mode in {"leader", "follower"}:
            return self._mode
        return None

    def read_state(self) -> ArmState:
        self._ensure_connected()
        self._integrate()
        timestamp_us = now_us()
        return ArmState(
            q=self._q.copy(),
            dq=self._dq.copy(),
            ddq=self._ddq.copy(),
            ee_pose=matrix_from_joint_stub(self._q),
            torque=0.05 * np.sin(self._q),
            current=0.2 * np.cos(self._q),
            timestamp_us=timestamp_us,
            acquired_timestamp_us=timestamp_us,
            q_timestamp_us=timestamp_us,
            q_acquired_timestamp_us=timestamp_us,
            q_component_timestamp_us=np.full(self.dof, timestamp_us, dtype=np.int64),
            q_source_before_timestamp_us=np.full(self.dof, timestamp_us, dtype=np.int64),
            q_source_after_timestamp_us=np.full(self.dof, timestamp_us, dtype=np.int64),
            motor_timestamp_us=np.full(self.dof, timestamp_us, dtype=np.int64),
            motor_acquired_timestamp_us=np.full(self.dof, timestamp_us, dtype=np.int64),
        )

    def configure_state_capture(
        self,
        maximum_source_skew_s: float,
        q_lowpass_cutoff_hz: float | None,
        dq_lowpass_cutoff_hz: float | None,
        ddq_lowpass_cutoff_hz: float | None,
        maximum_input_gap_s: float = 0.03,
    ) -> None:
        self._ensure_connected()

    def read_leader_joint_positions(self) -> np.ndarray:
        self._ensure_connected()
        self._integrate()
        if self._mode == "leader" and self.config.config_kwargs.get("mock_auto_motion", False):
            t = time.monotonic()
            phase = np.linspace(0.0, 1.2, self.dof)
            self._q = np.asarray(self.config.rest_q or (0.0,) * self.dof, dtype=np.float64) + 0.05 * np.sin(t + phase)
        return self._q.copy()

    def command_joint_positions(self, q: np.ndarray) -> None:
        self._ensure_connected()
        self._target_q = np.asarray(q, dtype=np.float64).reshape(self.dof)

    def command_joint_impedance(
        self,
        q: np.ndarray,
        v_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        t_ff: np.ndarray,
    ) -> None:
        self._ensure_connected()
        self._target_q = np.asarray(q, dtype=np.float64).reshape(self.dof)

    def validate_joint_impedance_support(self) -> None:
        self._ensure_connected()

    def configure_joint_impedance_mode(self) -> None:
        self._ensure_connected()

    def move_joints(self, q: np.ndarray) -> None:
        self._ensure_connected()
        self._target_q = np.asarray(q, dtype=np.float64).reshape(self.dof)
        self._q = self._target_q.copy()
        self._dq.fill(0.0)
        self._ddq.fill(0.0)

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        self._ensure_connected()
        return True

    def init_gripper(self, effector: str = "AGX_GRIPPER") -> None:
        self._ensure_connected()

    def read_gripper_state(self) -> GripperState:
        self._ensure_connected()
        return GripperState(
            value=self._gripper_value,
            force=self._gripper_force,
            timestamp_us=now_us(),
            mode=self._gripper_mode,
        )

    def read_leader_gripper_state(self) -> GripperState:
        return self.read_gripper_state()

    def disable_gripper(self) -> None:
        self._ensure_connected()

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        self._ensure_connected()
        self._gripper_value = float(value)
        self._gripper_force = float(force_n)
        self._gripper_mode = mode

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-4)
        self._last_t = now
        error = self._target_q - self._q
        step = np.clip(error, -dt * 2.0, dt * 2.0)
        new_q = self._q + step
        self._dq = (new_q - self._q) / dt
        self._ddq = (self._dq - self._last_dq) / dt
        self._last_dq = self._dq.copy()
        self._q = new_q

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"Mock arm {self.name} is not connected")
