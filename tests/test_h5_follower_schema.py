from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nero_collection.config import (
    CollectionConfig,
    DynamicsProcessingConfig,
    OutputConfig,
    StateParamConfig,
    TauFInferenceConfig,
    TeleopConfig,
)
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.tau_f_inference import OnlineTauFResult, TauFCheckpointMetadata


def test_h5_v7_keeps_bilateral_state_and_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")

    class FakeEstimator:
        def __init__(self, _config):
            pass

        def estimate(self, q, dq, ddq, tau):
            zeros = np.zeros(7)
            from nero_collection.contact_wrench import JointTorqueResidualEstimate

            return JointTorqueResidualEstimate(
                tau_id=zeros,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=zeros,
                tau_residual=-np.asarray(tau),
            )

    monkeypatch.setattr(
        "nero_collection.h5_writer.PinocchioJointTorqueResidualEstimator",
        FakeEstimator,
    )
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True),
            "acceleration": StateParamConfig(enabled=True),
            "ee_pose": StateParamConfig(enabled=True),
            "torque": StateParamConfig(enabled=True),
            "current": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",), sample_rate_hz=100.0)
    pose = np.eye(4, dtype=np.float64)
    for index, timestamp_us in enumerate(
        (1_000_000, 1_010_000, 1_020_000, 1_030_000)
    ):
        follower = np.full(7, float(index), dtype=np.float64)
        leader = np.full(7, 100.0 + index, dtype=np.float64)
        buffer.append_teleop(
            timestamp_us,
            {
                "q_follower": ("q", follower),
                "q_cmd": ("q", follower + 0.1),
                "q_timestamp_follower_us": ("timestamp", np.asarray([timestamp_us])),
                "q_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.asarray([timestamp_us + 1]),
                ),
                "dq_follower": ("velocity", follower + 0.2),
                "ddq_follower": ("acceleration", follower + 0.3),
                "ee_pose_follower": ("ee_pose", pose),
                "tau_follower": ("torque", follower + 0.4),
                "motor_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us, dtype=np.int64),
                ),
                "motor_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us + 1, dtype=np.int64),
                ),
                "current_follower": ("current", follower + 0.5),
                "gripper_follower": ("gripper", np.asarray([0.02])),
                "gripper_cmd": ("gripper", np.asarray([0.03])),
                "q_leader": ("q", leader),
                "dq_leader": ("velocity", leader),
                "ddq_leader": ("acceleration", leader),
                "ee_pose_leader": ("ee_pose", pose),
                "cmd_ee_pose": ("ee_pose", pose),
                "tau_leader": ("torque", leader),
                "current_leader": ("current", leader),
                "gripper_leader": ("gripper", np.asarray([0.04])),
                "gripper_state": ("gripper", np.asarray([0.02])),
                "gripper_value": ("gripper", np.asarray([0.02])),
            },
        )

    output = buffer.save(tmp_path / "episode.h5")

    expected = {
        "timestamp_us",
        "q_follower",
        "q_cmd",
        "dq_follower",
        "ddq_follower",
        "ee_pose_follower",
        "tau_follower",
        "current_follower",
        "gripper_follower",
        "gripper_cmd",
        "q_leader",
        "dq_leader",
        "ddq_leader",
        "tau_leader",
        "current_leader",
        "tau_f",
        "tau_f_timestamp_us",
    }
    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert h5.attrs["format"] == "factr_multimodal_episode/v7"
        assert teleop.attrs["data_role"] == "bilateral"
        assert set(teleop.keys()) == expected
        assert teleop["ee_pose_follower"].attrs["frame_name"] == "tcp"


def test_h5_saves_online_tau_f_outputs_on_follower_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")

    class FakeInference:
        def __init__(self, *args, **kwargs):
            self.metadata = TauFCheckpointMetadata(
                checkpoint_path=tmp_path / "model.pt",
                horizon=50,
                input_keys=("q", "dq", "tau"),
                input_dims={"q": 7, "dq": 7, "tau": 7},
                output_key="tau_f",
                output_dim=7,
                architecture="lstm",
                normalize_mode="gaussian",
            )

        def estimate_centered(self, timestamp_us, q, dq, ddq, tau):
            tau = np.asarray(tau, dtype=np.float64)
            tau_f_cal = np.full(7, 2.0)
            tau_f_pred = np.full(7, 0.5)
            return OnlineTauFResult(
                timestamp_us=timestamp_us,
                q=np.asarray(q),
                dq=np.asarray(dq),
                ddq=np.asarray(ddq),
                tau=tau,
                tau_id=tau + tau_f_cal,
                tau_f_cal=tau_f_cal,
                tau_f_pred=tau_f_pred,
                tau_ext=tau_f_cal - tau_f_pred,
            )

    class FakeWrenchEstimator:
        def __init__(self, _config):
            pass

        def map_joint_torque(self, _q, tau_ext):
            return SimpleNamespace(
                wrench=np.full(6, float(np.asarray(tau_ext)[0]))
            )

    monkeypatch.setattr("nero_collection.h5_writer.OnlineTauFInference", FakeInference)
    monkeypatch.setattr(
        "nero_collection.h5_writer.PinocchioContactWrenchEstimator",
        FakeWrenchEstimator,
    )
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        dynamics_processing=DynamicsProcessingConfig(
            enabled=True,
            state_method="finite_difference",
            torque_median_window=1,
            min_samples=3,
        ),
        tau_f_inference=TauFInferenceConfig(
            enabled=True,
            checkpoint_path=tmp_path / "model.pt",
        ),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True),
            "acceleration": StateParamConfig(enabled=True),
            "torque": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(
        config=config,
        arm_names=("main",),
        sample_rate_hz=100.0,
    )
    accepted_count = 0
    for index in range(5):
        value = np.full(7, float(index))
        accepted = buffer.append_teleop(
            1_000_000 + index * 10_000,
            {
                "q_follower": ("q", value),
                "dq_follower": ("velocity", np.zeros(7)),
                "ddq_follower": ("acceleration", np.zeros(7)),
                "tau_follower": ("torque", value + 1.0),
            },
        )
        if accepted is None:
            continue
        accepted_count += 1
        assert set(("tau_f_cal", "tau_f_pred", "tau_ext")) <= set(accepted.values)
    assert accepted_count == 5

    output = buffer.save(tmp_path / "online.h5")
    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        for name in ("q_follower", "dq_follower", "ddq_follower", "tau_follower"):
            assert teleop[name].shape == (5, 7)
            assert teleop[name].attrs["timestamp_path"] == "teleop/timestamp_us"
        assert h5.attrs["format"] == "factr_multimodal_episode/v7"
        assert "tau_f" not in teleop
        assert "tau_f_timestamp_us" not in teleop
        assert teleop["tau_f_cal"][:] == pytest.approx(np.full((5, 7), 2.0))
        assert teleop["tau_f_cal"].attrs["timestamp_path"] == "teleop/timestamp_us"
        assert teleop["tau_ext"].attrs["definition"] == "tau_f_cal - tau_f_pred"
        assert teleop["tau_f_pred"].shape == (5, 7)
        assert teleop["tau_ext"].shape == (5, 7)
        assert teleop["tau_ext"][:] == pytest.approx(np.full((5, 7), 1.5))
        assert teleop["wrench_ext"].shape == (5, 6)
        assert teleop["wrench_ext"][:] == pytest.approx(np.full((5, 6), 1.5))
        assert teleop["wrench_ext"].attrs["state_name"] == "wrench"
        assert teleop["wrench_ext"].attrs["frame_name"] == "gripper_base"
        assert teleop["wrench_ext"].attrs["reference_frame"] == "local"
        assert teleop["wrench_ext"].attrs["wrench_convention"] == "environment_on_tool"
        assert teleop["tau_f_pred"].attrs["model_horizon"] == 50
        assert teleop["tau_f_pred"].attrs["model_training_horizon"] == 50
        assert (
            teleop["tau_f_pred"].attrs["model_inference_mode"]
            == "stateful_recurrent_step"
        )
        assert (
            teleop["tau_f_pred"].attrs["processing_method"]
            == "online_stateful_recurrent_single_frame"
        )
        assert teleop["tau_f_pred"].attrs["model_output_key"] == "tau_f"


def test_h5_stores_precomputed_tau_bg_without_running_inference_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")

    class FakeInference:
        metadata = TauFCheckpointMetadata(
            checkpoint_path=tmp_path / "tau_bg.pt",
            horizon=50,
            input_keys=("q", "tau"),
            input_dims={"q": 7, "tau": 7},
            output_key="tau_bg",
            output_dim=7,
            architecture="gru",
            normalize_mode="gaussian",
        )

        @staticmethod
        def estimate_centered(*_args):
            raise AssertionError("precomputed control result must not be inferred again")

        @staticmethod
        def reset_episode():
            pass

    class FakeWrenchEstimator:
        def __init__(self, _config):
            pass

        @staticmethod
        def map_joint_torque(_q, tau_ext):
            return SimpleNamespace(wrench=np.asarray(tau_ext)[:6])

    monkeypatch.setattr(
        "nero_collection.h5_writer.PinocchioContactWrenchEstimator",
        FakeWrenchEstimator,
    )
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        tau_f_inference=TauFInferenceConfig(
            enabled=True,
            mode="tau_bg",
            checkpoint_path=tmp_path / "tau_bg.pt",
            tau_ext_lowpass_hz=10.0,
            tau_ext_gate_threshold_nm=(0.5,) * 7,
        ),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True),
            "acceleration": StateParamConfig(enabled=True),
            "torque": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(
        config=config,
        arm_names=("main",),
        sample_rate_hz=100.0,
        online_tau_f=FakeInference(),
    )
    zeros = np.zeros(7)
    accepted = buffer.append_teleop(
        1_000_000,
        {
            "q_follower": ("q", zeros),
            "dq_follower": ("velocity", zeros),
            "ddq_follower": ("acceleration", zeros),
            "tau_follower": ("torque", np.full(7, 2.0)),
            "tau_f_cal": ("torque", zeros),
            "tau_bg_pred": ("torque", np.full(7, 0.5)),
            "tau_ext_raw": ("torque", np.full(7, 1.5)),
            "tau_ext_filtered": ("torque", np.full(7, 1.2)),
            "tau_ext": ("torque", np.full(7, 1.2)),
        },
    )

    assert accepted is not None
    np.testing.assert_allclose(accepted.values["tau_ext"][1], np.full(7, 1.2))
    output = buffer.save(tmp_path / "tau_bg.h5")
    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert "tau_bg_pred" in teleop
        assert "tau_f_pred" not in teleop
        np.testing.assert_allclose(teleop["tau_ext"][:], np.full((1, 7), 1.2))
        assert teleop["tau_ext"].attrs["definition"] == (
            "tau_follower - tau_bg_pred"
        )
        assert teleop["tau_ext"].attrs["external_torque_mode"] == "tau_bg"
        assert teleop["tau_ext"].attrs["gate_applied"]
        assert teleop["tau_bg_pred"].attrs["model_output_key"] == "tau_bg"
