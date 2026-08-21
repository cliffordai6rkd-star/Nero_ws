from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nero_collection.arms.base import ArmState
from nero_collection.config import CollectionConfig, CommandConfig, GripperConfig, OutputConfig, TeleopConfig
from nero_collection.teleop.master_slave import ArmPairRuntime, MasterSlaveTeleop


class StartupArm:
    def __init__(self, name: str, role: str | None, events: list[str]) -> None:
        self.name = name
        self.role = role
        self.events = events
        self.q = np.zeros(7, dtype=np.float64)

    def connect(self) -> None:
        self.events.append(f"{self.name}:connect")

    def read_control_role(self, refresh: bool = False) -> str | None:
        self.events.append(f"{self.name}:read_role={self.role}")
        return self.role

    def set_leader_mode(self) -> None:
        self.events.append(f"{self.name}:set_leader")
        self.role = "leader"

    def set_follower_mode(self) -> None:
        self.events.append(f"{self.name}:set_follower")
        self.role = "follower"

    def enable(self) -> None:
        self.events.append(f"{self.name}:enable")

    def validate_joint_impedance_support(self) -> None:
        pass

    def configure_joint_impedance_mode(self) -> None:
        self.events.append(f"{self.name}:configure_mit")

    def move_joints(self, q: np.ndarray) -> None:
        self.events.append(f"{self.name}:move")
        self.q = np.asarray(q, dtype=np.float64).copy()

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        self.events.append(f"{self.name}:wait")
        return True

    def read_state(self) -> ArmState:
        zeros = np.zeros_like(self.q)
        return ArmState(
            q=self.q.copy(),
            dq=zeros.copy(),
            ddq=zeros.copy(),
            ee_pose=np.eye(4),
            torque=zeros.copy(),
            current=zeros.copy(),
            timestamp_us=0,
        )

    def init_gripper(self, effector: str = "AGX_GRIPPER") -> None:
        self.events.append(f"{self.name}:gripper_init={effector}")

    def reset_gripper(self) -> bool:
        self.events.append(f"{self.name}:gripper_reset")
        return True

    def command_gripper(
        self,
        value: float,
        force_n: float,
        mode: str = "width",
    ) -> None:
        self.events.append(
            f"{self.name}:gripper_command={value:.3f},{force_n:.1f},{mode}"
        )

    def disable_gripper(self) -> None:
        self.events.append(f"{self.name}:gripper_disable")


class StaleRefreshArm(StartupArm):
    def read_control_role(self, refresh: bool = False) -> str | None:
        self.events.append(f"{self.name}:read_role:refresh={refresh}")
        return "follower" if refresh else self.role


def test_start_checks_and_corrects_roles_before_follower_enable() -> None:
    events: list[str] = []
    leader = StartupArm("master", "leader", events)
    follower = StartupArm("slave", "follower", events)
    config = CollectionConfig(
        teleop=TeleopConfig(
            command=CommandConfig(reset_on_start=True, role_switch_settle_s=0.0)
        ),
        output=OutputConfig(directory=Path(".")),
        gripper=GripperConfig(enabled=False),
    )
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = config
    teleop.pairs = (
        ArmPairRuntime(
            name="main",
            leader=leader,
            follower=follower,
            rest_q_leader=np.zeros(7),
            rest_q_follower=np.zeros(7),
            controller=object(),
        ),
    )
    teleop.arm_names = ("main",)
    teleop._teleop_reference = {}
    teleop._hold_after_reset = False
    teleop.check_input_devices = lambda: events.append("check_inputs")
    teleop.reset_to_rest = lambda: events.append("reset")

    teleop.start()

    assert events == [
        "master:connect",
        "slave:connect",
        "master:read_role=leader",
        "slave:read_role=follower",
        "master:read_role=leader",
        "master:set_follower",
        "slave:read_role=follower",
        "master:enable",
        "slave:enable",
        "master:read_role=follower",
        "slave:read_role=follower",
        "check_inputs",
        "reset",
    ]


def test_matching_roles_are_not_switched() -> None:
    events: list[str] = []
    leader = StartupArm("master", "follower", events)
    follower = StartupArm("slave", "follower", events)

    MasterSlaveTeleop._ensure_arm_role("main", leader, "follower")
    MasterSlaveTeleop._ensure_arm_role("main", follower, "follower")

    assert events == ["master:read_role=follower", "slave:read_role=follower"]


def test_gripper_startup_reset_closes_then_opens_before_disabling_leader() -> None:
    events: list[str] = []
    leader = StartupArm("master", "follower", events)
    follower = StartupArm("slave", "follower", events)
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=Path(".")),
        gripper=GripperConfig(
            enabled=True,
            attach_to="both",
            teleop_enabled=True,
            reset_step_wait_s=0.0,
            min_width_m=0.0,
            max_width_m=0.1,
            force_n=1.0,
        ),
    )
    pair = ArmPairRuntime(
        name="main",
        leader=leader,
        follower=follower,
        rest_q_leader=np.zeros(7),
        rest_q_follower=np.zeros(7),
        controller=object(),
    )

    teleop._init_grippers(pair)

    assert events == [
        "master:gripper_init=AGX_GRIPPER",
        "master:gripper_reset",
        "master:gripper_command=0.000,1.0,width",
        "master:gripper_command=0.100,1.0,width",
        "master:gripper_disable",
        "slave:gripper_init=AGX_GRIPPER",
        "slave:gripper_reset",
        "slave:gripper_command=0.000,1.0,width",
        "slave:gripper_command=0.100,1.0,width",
    ]


def test_dual_arm_reset_sends_both_moves_before_waiting() -> None:
    events: list[str] = []
    leader = StartupArm("master", "follower", events)
    follower = StartupArm("slave", "follower", events)
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = CollectionConfig(
        teleop=TeleopConfig(command=CommandConfig(reset_timeout_s=1.0)),
        output=OutputConfig(directory=Path(".")),
    )
    teleop.pairs = (
        ArmPairRuntime(
            name="main",
            leader=leader,
            follower=follower,
            rest_q_leader=np.zeros(7),
            rest_q_follower=np.zeros(7),
            controller=object(),
        ),
    )

    teleop._move_both_arms_to_reset_targets(
        {"main": {"leader": np.zeros(7), "follower": np.zeros(7)}}
    )

    assert events == ["master:move", "slave:move", "master:wait", "slave:wait"]


def test_dual_arm_reset_interpolates_waypoints(monkeypatch) -> None:
    events: list[str] = []
    leader = StartupArm("master", "follower", events)
    follower = StartupArm("slave", "follower", events)
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = CollectionConfig(
        teleop=TeleopConfig(
            command=CommandConfig(
                reset_timeout_s=1.0,
                reset_interpolation_enabled=True,
                reset_interpolation_rate_hz=10.0,
                reset_joint_speed_rad_s=1.0,
                reset_min_duration_s=0.2,
                reset_max_step_rad=0.1,
            )
        ),
        output=OutputConfig(directory=Path(".")),
    )
    teleop.pairs = (
        ArmPairRuntime(
            name="main",
            leader=leader,
            follower=follower,
            rest_q_leader=np.zeros(7),
            rest_q_follower=np.zeros(7),
            controller=object(),
        ),
    )
    monkeypatch.setattr("nero_collection.teleop.master_slave.time.sleep", lambda _seconds: None)
    target = np.full(7, 0.2)

    teleop._move_both_arms_to_reset_targets(
        {"main": {"leader": target, "follower": target}}
    )

    assert events == [
        "master:move",
        "slave:move",
        "master:move",
        "slave:move",
        "master:wait",
        "slave:wait",
    ]
    assert np.allclose(leader.q, target)
    assert np.allclose(follower.q, target)


def test_reset_role_check_prefers_role_commanded_by_this_process() -> None:
    events: list[str] = []
    arm = StaleRefreshArm("master", "leader", events)

    MasterSlaveTeleop._ensure_arm_role("main", arm, "follower")

    assert arm.role == "follower"
    assert events == ["master:read_role:refresh=False", "master:set_follower"]


def test_teleop_keeps_logical_leader_in_follower_mode() -> None:
    events: list[str] = []
    leader = StaleRefreshArm("master", "follower", events)
    follower = StartupArm("slave", "follower", events)
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = CollectionConfig(
        teleop=TeleopConfig(
            command=CommandConfig(role_switch_settle_s=0.0, role_switch_timeout_s=0.01)
        ),
        output=OutputConfig(directory=Path(".")),
    )
    pair = ArmPairRuntime(
        name="main",
        leader=leader,
        follower=follower,
        rest_q_leader=np.zeros(7),
        rest_q_follower=np.zeros(7),
        controller=object(),
    )

    teleop._prepare_pair_for_teleop(pair)

    assert leader.role == "follower"
    assert "master:read_role:refresh=True" in events
    assert "master:configure_mit" in events
    assert "slave:configure_mit" in events
