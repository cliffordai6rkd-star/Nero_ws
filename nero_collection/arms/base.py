from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    ee_pose: np.ndarray
    torque: np.ndarray
    current: np.ndarray
    timestamp_us: int
    acquired_timestamp_us: int = 0
    q_timestamp_us: int = 0
    q_acquired_timestamp_us: int = 0
    q_component_timestamp_us: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    q_source_before_timestamp_us: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    q_source_after_timestamp_us: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    motor_timestamp_us: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    motor_acquired_timestamp_us: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )


@dataclass
class GripperState:
    value: float
    force: float
    timestamp_us: int
    mode: str = "width"


class ArmInterface(Protocol):
    name: str
    dof: int

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def enable(self) -> None:
        ...

    def disable(self) -> None:
        ...

    def set_leader_mode(self) -> None:
        ...

    def set_follower_mode(self) -> None:
        ...

    def set_normal_mode(self) -> None:
        ...

    def read_control_role(self, refresh: bool = False) -> str | None:
        ...

    def read_state(self) -> ArmState:
        ...

    def configure_state_alignment(
        self,
        delay_s: float,
        output_rate_hz: float,
        q_mean_window_samples: int,
        q_lowpass_cutoff_hz: float | None,
        dq_lowpass_cutoff_hz: float | None,
        ddq_lowpass_cutoff_hz: float | None,
        maximum_input_gap_s: float = 0.03,
    ) -> None:
        ...

    def read_leader_joint_positions(self) -> np.ndarray:
        ...

    def command_joint_positions(self, q: np.ndarray) -> None:
        ...

    def command_joint_impedance(
        self,
        q: np.ndarray,
        v_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        t_ff: np.ndarray,
    ) -> None:
        ...

    def validate_joint_impedance_support(self) -> None:
        ...

    def configure_joint_impedance_mode(self) -> None:
        ...

    def move_joints(self, q: np.ndarray) -> None:
        ...

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        ...

    def init_gripper(self, effector: str = "AGX_GRIPPER") -> None:
        ...

    def read_gripper_state(self) -> GripperState:
        ...

    def read_leader_gripper_state(self) -> GripperState:
        ...

    def disable_gripper(self) -> None:
        ...

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        ...
