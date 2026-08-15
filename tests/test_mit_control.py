from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nero_collection.arms.pyagx import (
    PyAgxArmAdapter,
    _component_version,
    _read_motor_states,
)
from nero_collection.arms.base import ArmState
from nero_collection.coordinates import NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
from nero_collection.config import (
    ArmEndpointConfig,
    BilateralMitConfig,
    InverseDynamicsConfig,
    _parse_command,
    load_config,
)
from nero_collection.teleop.bilateral import BilateralJointController


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[dict[str, float | int]] = []

    def move_mit(self, **kwargs):
        self.calls.append(kwargs)
        return True


class StatusRobot:
    def __init__(self, ctrl_mode: int) -> None:
        self.status = SimpleNamespace(msg=SimpleNamespace(ctrl_mode=ctrl_mode))

    def get_arm_status(self):
        return self.status


def test_gripper_server_version_probe_prefers_explicit_server_api() -> None:
    component = SimpleNamespace(
        get_server_version=lambda: "V120",
        version="legacy",
    )

    assert _component_version(component) == "V120"


class LeaderFeedbackRobot(StatusRobot):
    def __init__(self) -> None:
        super().__init__(0x01)
        self.leader_timestamp = 1.0
        self.mode_calls: list[str] = []

    def get_leader_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7, timestamp=self.leader_timestamp)

    def set_normal_mode(self) -> None:
        self.mode_calls.append("normal")

    def set_leader_mode(self) -> None:
        self.mode_calls.append("leader")
        self.leader_timestamp = 2.0


class FakeGripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []
        self.disable_calls = 0
        self.enabled = True

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.calls.append(("width", value, force))

    def move_gripper_deg(self, *, value: float, force: float) -> None:
        self.calls.append(("angle", value, force))

    def disable_gripper(self) -> bool:
        self.disable_calls += 1
        self.enabled = False
        return True

    def get_gripper_status(self):
        return SimpleNamespace(
            timestamp=123.0,
            msg=SimpleNamespace(
                value=25.0,
                force=0.2,
                mode="angle",
                foc_status=SimpleNamespace(driver_enable_status=self.enabled),
            )
        )

    def get_gripper_ctrl_states(self):
        return SimpleNamespace(
            timestamp=124.0,
            msg=SimpleNamespace(value=0.04, force=0.0, status_code=1),
        )


def test_master_slave_config_has_valid_control_parameters() -> None:
    config = load_config("configs/master_slave_can.yaml")

    command = config.teleop.command
    assert command.control_mode == "mit"
    assert len(command.bilateral_mit.leader_kp) == 7
    assert len(command.bilateral_mit.follower_kp) == 7
    assert all(value == 0.0 for value in command.bilateral_mit.leader_kp)
    assert all(0.0 <= value <= 500.0 for value in command.bilateral_mit.follower_kp)
    assert config.robot_states["velocity"].lowpass_cutoff_hz > 0.0
    assert "acceleration" not in config.robot_states
    assert config.gripper.teleop_enabled is True
    assert config.gripper.attach_to == "both"
    if config.realtime_plot.inverse_dynamics.manifest_path is not None:
        assert config.realtime_plot.inverse_dynamics.manifest_path.is_file()
    assert config.dynamics_processing.enabled is False
    assert config.dynamics_processing.state_method == "finite_difference"
    assert config.robot_states["tau_id"].lowpass is False


def test_collection_config_requires_tau_ext_inference_for_realtime_plot(tmp_path) -> None:
    config_path = tmp_path / "master_slave_can.yaml"
    source_path = Path(__file__).resolve().parents[1] / "configs/master_slave_can.yaml"
    text = source_path.read_text(encoding="utf-8")
    text = text.replace(
        "tau_ext_inference:\n  enabled: true",
        "tau_ext_inference:\n  enabled: false",
        1,
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="requires tau_ext_inference"):
        load_config(config_path)


def test_bilateral_controller_uses_preprocessed_learned_external_torque() -> None:
    class Dynamics:
        model = SimpleNamespace(
            lowerPositionLimit=np.full(7, -3.0),
            upperPositionLimit=np.full(7, 3.0),
        )

        def __init__(self) -> None:
            self.estimate_calls = 0

        @staticmethod
        def gravity_torque(_q):
            return np.zeros(7)

        def estimate(self, *_args):
            self.estimate_calls += 1
            raise AssertionError("learned tau_ext must bypass analytical residual")

    def state() -> ArmState:
        zeros = np.zeros(7)
        return ArmState(
            q=zeros.copy(),
            dq=zeros.copy(),
            ddq=zeros.copy(),
            ee_pose=np.eye(4),
            torque=zeros.copy(),
            current=zeros.copy(),
            timestamp_us=1,
        )

    dynamics = Dynamics()
    controller = BilateralJointController(
        BilateralMitConfig(
            force_feedback_gain=(1.0,) * 7,
            force_feedback_deadband_nm=(10.0,) * 7,
            force_feedback_limit_nm=(10.0,) * 7,
            force_feedback_ramp_s=0.0,
        ),
        InverseDynamicsConfig(),
        dynamics=dynamics,
    )
    leader = state()
    follower = state()
    controller.activate(leader, follower)
    controller._rate_limit_feedback = lambda value: value.copy()
    tau_ext = np.linspace(0.1, 0.7, 7)

    result = controller.compute(
        leader,
        follower,
        timestamp_us=1_000_000,
        tau_ext_override=tau_ext,
    )

    assert dynamics.estimate_calls == 0
    np.testing.assert_allclose(result.tau_ext_follower, tau_ext)
    np.testing.assert_allclose(result.tau_feedback_leader, tau_ext)


def test_bilateral_feedback_lowpass_accepts_null_and_bypasses_filter() -> None:
    config = _parse_command(
        {"bilateral_mit": {"force_feedback_lowpass_hz": None}}
    ).bilateral_mit
    assert config.force_feedback_lowpass_hz is None

    class Dynamics:
        model = SimpleNamespace(
            lowerPositionLimit=np.full(7, -3.0),
            upperPositionLimit=np.full(7, 3.0),
        )

        @staticmethod
        def gravity_torque(_q):
            return np.zeros(7)

        @staticmethod
        def estimate(*_args):
            return SimpleNamespace(tau_residual=np.ones(7))

    controller = BilateralJointController(
        config,
        InverseDynamicsConfig(),
        dynamics=Dynamics(),
    )
    assert controller.feedback_filter is None


@pytest.mark.parametrize("cutoff", [0, -1, float("inf")])
def test_bilateral_feedback_lowpass_rejects_invalid_numeric_cutoff(cutoff: float) -> None:
    with pytest.raises(ValueError, match="positive or null"):
        _parse_command({"bilateral_mit": {"force_feedback_lowpass_hz": cutoff}})


@pytest.mark.parametrize(
    "bilateral",
    [
        {"follower_kp": [1.0] * 6},
        {"follower_kp": [501.0] * 7},
        {"leader_kd": [5.1] * 7},
        {"force_feedback_sign": [0.0] * 7},
        {"position_scale": [2.1] * 7},
    ],
)
def test_bilateral_mit_config_rejects_invalid_vectors(
    bilateral: dict[str, list[float]],
) -> None:
    with pytest.raises(ValueError):
        _parse_command({"bilateral_mit": bilateral})


@pytest.mark.parametrize("legacy", ["mit", "teleop_mapping"])
def test_command_config_rejects_legacy_control_fields(legacy: str) -> None:
    with pytest.raises(ValueError, match="legacy"):
        _parse_command({legacy: {}})


@pytest.mark.parametrize("mode", ["mit", "position"])
def test_command_config_accepts_follower_control_modes(mode: str) -> None:
    assert _parse_command({"control_mode": mode}).control_mode == mode


def test_command_config_rejects_unknown_follower_control_mode() -> None:
    with pytest.raises(ValueError, match="must be mit or position"):
        _parse_command({"control_mode": "torque"})


def test_command_config_accepts_fixed_robot_state_sample_rate() -> None:
    assert _parse_command({"sample_rate_hz": 50.0}).sample_rate_hz == 50.0


def test_command_config_rejects_event_alignment_fields() -> None:
    with pytest.raises(ValueError, match="event-alignment"):
        _parse_command({"maximum_can_frame_gap_s": 0.03})


def test_pyagx_adapter_sends_all_seven_mit_commands() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="follower"))
    robot = FakeRobot()
    adapter._robot = robot

    adapter.command_joint_impedance(
        q=np.arange(7, dtype=np.float64) * 0.1,
        v_des=np.zeros(7),
        kp=np.arange(1, 8, dtype=np.float64),
        kd=np.full(7, 0.8),
        t_ff=np.zeros(7),
    )

    assert [call["joint_index"] for call in robot.calls] == list(range(1, 8))
    assert robot.calls[3]["p_des"] == pytest.approx(0.3)
    assert robot.calls[6]["kp"] == pytest.approx(7.0)


def test_pyagx_adapter_rejects_torque_above_v111_limit_before_sdk_call() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="follower", firmware="V112"))
    robot = FakeRobot()
    adapter._robot = robot

    with pytest.raises(RuntimeError, match="unsafe MIT feed-forward torque"):
        adapter.command_joint_impedance(
            q=np.zeros(7),
            v_des=np.zeros(7),
            kp=np.zeros(7),
            kd=np.zeros(7),
            t_ff=np.asarray([0.0, -16.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    assert robot.calls == []


def test_v120_direct_motor_read_uses_the_same_velocity_signs() -> None:
    def get_motor_states(joint_number: int):
        return SimpleNamespace(
            timestamp=0.01,
            msg=SimpleNamespace(
                velocity=0.1 * joint_number,
                torque=10.0 + joint_number,
                current=20.0 + joint_number,
            ),
        )

    states = _read_motor_states(
        SimpleNamespace(get_motor_states=get_motor_states),
        7,
        firmware="V120",
    )

    raw_velocity = np.arange(1.0, 8.0) * 0.1
    np.testing.assert_allclose(
        states["velocity"],
        raw_velocity * np.asarray(NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN),
    )
    np.testing.assert_allclose(states["torque"], np.arange(11.0, 18.0))
    np.testing.assert_allclose(states["current"], np.arange(21.0, 28.0))


def test_pyagx_adapter_rejects_sdk_without_move_mit() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="follower"))
    adapter._robot = object()

    with pytest.raises(RuntimeError, match="does not expose the Nero bilateral MIT API"):
        adapter.validate_joint_impedance_support()


@pytest.mark.parametrize(("ctrl_mode", "expected"), [(0x06, "leader"), (0x01, "follower")])
def test_pyagx_adapter_reads_control_role(ctrl_mode: int, expected: str) -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = StatusRobot(ctrl_mode)

    assert adapter.read_control_role() == expected


def test_pyagx_adapter_refreshes_cached_control_role() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = StatusRobot(0x01)
    adapter._configured_role = "leader"

    assert adapter.read_control_role(refresh=True) == "follower"


def test_pyagx_adapter_verifies_commanded_leader_from_fresh_joint_feedback() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = LeaderFeedbackRobot()

    adapter.set_leader_mode()

    assert adapter._robot.mode_calls == ["leader", "leader", "leader"]
    assert adapter.read_control_role(refresh=True) == "leader"


def test_pyagx_adapter_commands_gripper_in_width_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    adapter.command_gripper(0.035, 2.0)

    assert gripper.calls == [("width", 0.035, 2.0)]


def test_pyagx_adapter_commands_gripper_in_angle_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    adapter.command_gripper(25.0, 2.0, mode="angle")

    assert gripper.calls == [("angle", 25.0, 2.0)]


def test_pyagx_adapter_reads_angle_mode_and_disables_leader_gripper() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    state = adapter.read_gripper_state()
    adapter.disable_gripper()

    assert state.value == pytest.approx(25.0)
    assert state.mode == "angle"
    assert state.timestamp_us == 123_000_000
    assert gripper.disable_calls == 1


def test_pyagx_adapter_reads_leader_gripper_control_frame() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._gripper = FakeGripper()

    state = adapter.read_leader_gripper_state()

    assert state.value == pytest.approx(0.04)
    assert state.mode == "width"
    assert state.timestamp_us == 124_000_000


def test_pyagx_adapter_rejects_gripper_control_frame_from_before_leader_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._gripper = FakeGripper()
    adapter._leader_mode_commanded = True
    adapter._leader_gripper_feedback_baseline = 124.0

    state = adapter.read_leader_gripper_state()

    assert np.isnan(state.value)
    assert state.mode == "unknown"
