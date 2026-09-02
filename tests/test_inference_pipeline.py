from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.config import (
    CheckpointConfig,
    DPSamplingConfig,
    InferenceConfig,
    PredictorConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    TorqueFilterConfig,
    load_inference_config,
)
from inference.pipeline import (
    InferenceInput,
    NeroInferencePipeline,
    _dp_execution_action_chunk,
    _dp_model_overrides,
    _predict_dp_action,
    _minimum_jerk_action_plan,
    _relative_action_pose_torch,
    _select_action_chunk,
    _uses_link7_target_gripper_tcp_current_contract,
)
from nero_collection.control import DynamicsSnapshot


torch = pytest.importorskip("torch")


class _DP(torch.nn.Module):
    n_obs_steps = 2
    image_key = "wrist"
    wrench_key = "wrench_ext"

    class _Encoder:
        wrench_history_steps = 3

    obs_encoder = _Encoder()

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def predict_action(self, obs):
        self.calls += 1
        assert obs["wrist"].shape == (1, 2, 3, 8, 10)
        assert obs["wrench_ext"].shape == (1, 2, 3, 6)
        return {
            "action_target": torch.tensor(
                [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
            )
        }


class _ImageOnlyDP(torch.nn.Module):
    n_obs_steps = 1
    image_key = "wrist"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observation_keys = None

    def predict_action(self, obs):
        self.observation_keys = tuple(obs)
        return {
            "action_target": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
            )
        }


class _PINN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def predict_force(self, inputs):
        self.calls += 1
        assert inputs["action_target"].shape == (1, 7)
        return {"f_ext": torch.tensor([[100.0, 2.0, 3.0, 9.0, 0.0, 0.0]])}


class _ActionDP(torch.nn.Module):
    n_obs_steps = 1
    image_key = "wrist"
    wrench_key = "wrench_ext"

    class _Encoder:
        wrench_history_steps = 1

    obs_encoder = _Encoder()

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def predict_action(self, obs):
        self.calls += 1
        chunk = torch.tensor(
            [
                [
                    [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )
        return {
            "action": chunk,
            "action_target": chunk.mean(dim=1),
        }


class _JointActionDP(_ActionDP):
    def predict_action(self, obs):
        del obs
        self.calls += 1
        chunk = torch.tensor(
            [[[0.4, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3]]]
        )
        return {"action": chunk, "action_target": chunk[:, 0]}


class _JointDiffusionDP(torch.nn.Module):
    """Minimal policy-shaped fixture for the joint-action output contract."""

    horizon = 2
    n_obs_steps = 1
    n_action_steps = 2
    action_start_index = 0
    action_dim = 7

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.conditional_sample_calls = 0
        self.predict_action_calls = 0

        class _Normalizer:
            def __getitem__(self, key):
                assert key == "action"
                return self

            @staticmethod
            def unnormalize(value):
                return value * 2.0 + 0.5

        self.normalizer = _Normalizer()

    def _encode_observation(self, obs):
        assert tuple(obs) == ("wrist",)
        return torch.zeros((1, 1, 4))

    def conditional_sample(self, shape, condition):
        assert shape == (1, 2, 7)
        assert tuple(condition.shape) == (1, 1, 4)
        self.conditional_sample_calls += 1
        return torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
              [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]]
        )

    def predict_action(self, obs):
        del obs
        self.predict_action_calls += 1
        raise AssertionError("joint diffusion path must bypass pose normalization")


class _TwoObservationActionDP(_ActionDP):
    n_obs_steps = 2

    def __init__(self) -> None:
        super().__init__()
        self.observation_markers = []

    def predict_action(self, obs):
        self.observation_markers.append(
            obs["wrist"][0, :, 0, 0, 0].detach().cpu().numpy().copy()
        )
        return super().predict_action(obs)


class _TwoCameraDP(_DP):
    image_keys = ("side", "wrist")
    wrench_key = None

    class _Encoder:
        pass

    obs_encoder = _Encoder()


class _ActionPINN(torch.nn.Module):
    active_inputs = ("q", "v", "a", "tau")
    future_horizon = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self._inference_checkpoint_config = {
            "model": {
                "action_condition_mode": "relative_pose",
                "action_key": "action_relative_future",
                "action_current_frame_name": "link7",
            }
        }
        self.inputs = []

    def predict_force(self, inputs):
        self.inputs.append(inputs)
        action_key = self._inference_checkpoint_config["model"]["action_key"]
        assert inputs[action_key].shape == (1, 2, 7)
        return {"wrench_pred": torch.zeros(1, 2, 6)}


class _WorldModel(torch.nn.Module):
    history_horizon = 3
    future_horizon = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self._inference_checkpoint_config = {
            "dataloader": {
                "normalize_mode": "gaussian",
                "normalize_lowdim_keys": ["q", "v", "a", "tau", "wrench"],
            },
            "model": {
                "action_condition_mode": "absolute_pose",
                "action_key": "action_condition_future",
            },
        }
        stats = {}
        for index, key in enumerate(("q", "v", "a", "tau"), start=1):
            stats[key] = {"mean": [float(index)] * 7, "std": [1.0] * 7}
        stats["wrench"] = {"mean": [0.0] * 6, "std": [1.0] * 6}
        self._inference_normalizer = {
            "normalize_mode": "gaussian",
            "normalize_lowdim_keys": ["q", "v", "a", "tau", "wrench"],
            "eps": 0.0,
            "stats": stats,
        }
        self.inputs = []

    def predict(self, inputs):
        self.inputs.append(inputs)
        return {
            "state_pred": {
                key: torch.zeros(1, self.future_horizon, 7)
                for key in ("q", "v", "a", "tau")
            }
        }


class _WorldModelV4(_WorldModel):
    sampling_dt = 0.01

    def __init__(self) -> None:
        super().__init__()
        self._inference_checkpoint_config["contact_gate"] = {
            "enabled": True,
            "probability_threshold": 0.5,
        }

    def predict(self, inputs):
        self.inputs.append(inputs)
        return {
            "state_pred": {
                key: torch.zeros(1, self.future_horizon, 7)
                for key in ("q", "tau")
            },
            "contact_probability": torch.tensor([[[0.2], [0.8]]]),
        }

    def reconstruct_future_state(self, q_history, q_future):
        assert q_history.shape == (1, self.history_horizon, 7)
        assert q_future.shape == (1, self.future_horizon, 7)
        return {
            "q": q_future,
            "v": torch.full_like(q_future, 5.0),
            "a": torch.full_like(q_future, 6.0),
        }


class _WorldModelWrenchAdapter:
    def __init__(self) -> None:
        self.calls = []

    def states_to_wrenches(self, history, future):
        self.calls.append((history, future))
        return np.array([[1, 2, 3, 1, 2, 3], [2, 3, 4, 2, 3, 4]], dtype=float)


class _Model:
    dof = 7

    def snapshot(self, q, dq):
        pose = np.eye(4)
        return DynamicsSnapshot(
            np.eye(7), np.zeros(7), np.zeros((6, 7)), np.zeros(6), pose
        )


class _IKModel(_Model):
    position_lower = np.full(7, -2.0)
    position_upper = np.full(7, 2.0)

    def snapshot(self, q, dq):
        del dq
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=np.float64)
        jacobian = np.zeros((6, 7), dtype=np.float64)
        jacobian[:, :6] = np.eye(6)
        return DynamicsSnapshot(
            np.eye(7), np.zeros(7), jacobian, np.zeros(6), pose
        )


class _FrameModel(_Model):
    def snapshot(self, q, dq):
        snapshot = super().snapshot(q, dq)
        pose = np.eye(4)
        pose[2, 3] = 1.0
        return replace(snapshot, pose=pose)

    def frame_pose(self, q, frame_name):
        assert frame_name == "link7"
        return np.eye(4)


class _Controller:
    def __init__(self) -> None:
        self.model = _Model()
        self.config = SimpleNamespace(horizon_steps=2, dt_s=0.01)
        self.targets = []

    def optimize_mpc(self, q, dq, target, measured_wrench=None, previous_tau=None):
        self.targets.append(target)
        tau = np.full((2, 7), 50.0)
        return SimpleNamespace(
            tau=tau,
            joint_accelerations=np.zeros((2, 7)),
            predicted_q=np.zeros((3, 7)),
            predicted_dq=np.zeros((3, 7)),
            predicted_wrenches=target.wrenches,
            status="solved",
            iterations=1,
            solve_time_s=0.0,
            objective=0.0,
            max_constraint_violation=0.0,
        )


def _config(tmp_path: Path) -> InferenceConfig:
    checkpoint = CheckpointConfig(tmp_path / "unused.ckpt", device="cpu")
    return InferenceConfig(
        dp_checkpoint=checkpoint,
        pinn_checkpoint=checkpoint,
        robot=RobotConfig(tmp_path / "unused.urdf"),
        runtime=RuntimeConfig(tmp_path / "collection.yaml"),
        safety=SafetyConfig(
            maximum_action_translation_step_m=0.05,
            maximum_target_force_n=40.0,
            maximum_target_moment_nm=5.0,
            maximum_command_torque_nm=20.0,
        ),
        torque_filter=TorqueFilterConfig(enabled=False),
    )


def _sample(timestamp: float) -> InferenceInput:
    return InferenceInput(
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        image=np.zeros((8, 10, 3), dtype=np.uint8),
        wrench_ext=np.zeros(6),
        timestamp_s=timestamp,
    )


def _marked_sample(timestamp: float, marker: int) -> InferenceInput:
    return replace(
        _sample(timestamp),
        image=np.full((8, 10, 3), marker, dtype=np.uint8),
    )


def test_multirate_pipeline_tracks_clipped_action_force_and_torque(tmp_path: Path) -> None:
    dp, pinn, controller = _DP(), _PINN(), _Controller()
    pipeline = NeroInferencePipeline(
        _config(tmp_path), dp_model=dp, pinn_model=pinn, controller=controller
    )

    initial = pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    second = pipeline.step(_sample(0.01))

    assert dp.calls >= 1
    assert pinn.calls == 2
    np.testing.assert_allclose(initial.action_target[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(second.action_target[:3], [0.05, 0.0, 0.0])
    assert second.dp_action_chunk is not None
    np.testing.assert_allclose(second.dp_action_chunk[:, 0], [0.2])
    np.testing.assert_allclose(second.target_wrench, [40.0, 2.0, 3.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(second.tau_command, np.full(7, 20.0))
    assert second.dp_updated
    np.testing.assert_allclose(controller.targets[1].wrenches[0], second.target_wrench)
    pipeline.close()


def test_dp_tau_ext_switch_allows_image_only_checkpoint(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        dp_sampling=DPSamplingConfig(use_tau_ext_observation=False),
    )
    dp = _ImageOnlyDP()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=dp,
        pinn_model=_PINN(),
        controller=_Controller(),
    )

    pipeline._predict_action(
        np.zeros((1, 3, 8, 10), dtype=np.float32),
        np.ones((1, 1, 6), dtype=np.float32),
    )

    assert dp.observation_keys == ("wrist",)
    assert not pipeline._uses_wrench_observation
    pipeline.close()


def test_dp_tau_ext_switch_rejects_force_aware_checkpoint(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        dp_sampling=DPSamplingConfig(use_tau_ext_observation=False),
    )

    with pytest.raises(ValueError, match="requires an image-only DP checkpoint"):
        NeroInferencePipeline(
            config,
            dp_model=_DP(),
            pinn_model=_PINN(),
            controller=_Controller(),
        )


def test_current_dp_reduces_eight_high_rate_chunks_and_drops_first() -> None:
    actions = np.zeros((64, 7), dtype=np.float64)
    actions[:, 6] = 1.0
    for row in range(8):
        start = row * 8
        actions[start : start + 8, 0] = row + np.arange(8) / 10.0
    # Equivalent quaternion signs must not cancel during row aggregation.
    actions[8:16:2, 6] = -1.0

    reduced = _dp_execution_action_chunk(actions, model_horizon=64)

    assert reduced.shape == (7, 7)
    np.testing.assert_allclose(reduced[:, 0], np.arange(1, 8) + 0.35)
    np.testing.assert_allclose(reduced[:, 1:6], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(np.abs(reduced[:, 6]), 1.0, atol=1.0e-8)


def test_current_dp_rejects_incomplete_64_step_output() -> None:
    actions = np.zeros((8, 7), dtype=np.float64)
    actions[:, 6] = 1.0

    with pytest.raises(ValueError, match="64-step DP"):
        _dp_execution_action_chunk(actions, model_horizon=64)


def test_current_dp_reduced_chunk_preserves_all_first_and_mean_modes() -> None:
    actions = np.zeros((64, 7), dtype=np.float64)
    actions[:, 6] = 1.0
    for row in range(8):
        actions[row * 8 : (row + 1) * 8, 0] = float(row)

    reduced = _dp_execution_action_chunk(actions, model_horizon=64)

    np.testing.assert_allclose(reduced[:, 0], np.arange(1, 8))
    np.testing.assert_allclose(_select_action_chunk(reduced, "first")[0], 1.0)
    np.testing.assert_allclose(_select_action_chunk(reduced, "mean")[0], 4.0)


def test_dp_chunk_logging_only_includes_geometric_delta(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="inference.pipeline")
    pipeline = NeroInferencePipeline(
        _config(tmp_path),
        dp_model=_DP(),
        pinn_model=_PINN(),
        controller=_Controller(),
    )

    pipeline.step(_sample(0.0))
    pipeline._dp_future.result(timeout=2.0)
    pipeline.step(_sample(0.01))
    pipeline.close()

    messages = [record.getMessage() for record in caplog.records]
    assert any("delta_xyz_rotvec=" in message for message in messages)
    assert any("0.200000" in message for message in messages)
    assert not any("action_pred=" in message for message in messages)


def test_timed_dp_observation_matches_training_image_and_wrench_grid(
    tmp_path: Path,
) -> None:
    dp = _DP()
    dp.obs_encoder = type("Encoder", (), {"wrench_history_steps": 8})()
    dp._inference_checkpoint_config = {
        "task": {"dataset": {"timestamp_step_sec": 0.1}}
    }
    pipeline = NeroInferencePipeline(
        _config(tmp_path),
        dp_model=dp,
        pinn_model=_PINN(),
        controller=_Controller(),
    )
    for index in range(17):
        state_time = index * 0.0125
        image_time = np.floor((state_time + 1.0e-9) / 0.05) * 0.05
        pipeline._append_observation(
            np.full((8, 10, 3), image_time, dtype=np.float32),
            np.full(6, state_time, dtype=np.float32),
            image_timestamp_s=image_time,
            state_timestamp_s=state_time,
        )

    images, wrenches = pipeline._timed_observation_snapshot(0.2)

    np.testing.assert_allclose(images[:, 0, 0, 0], [0.1, 0.2], atol=1.0e-6)
    expected = np.stack(
        (
            np.arange(1, 9) * 0.0125,
            np.arange(9, 17) * 0.0125,
        )
    )
    np.testing.assert_allclose(wrenches[:, :, 0], expected, atol=1.0e-6)
    pipeline.close()


def test_open_loop_waits_until_can_timestamp_covers_camera_frame(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            inference_mode="open_loop",
            action_chunk_mode="all",
            action_step_s=0.1,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _ActionDP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)
    image = np.full((8, 10, 3), 20, dtype=np.uint8)

    camera_ahead = pipeline.step(
        replace(_sample(0.05), image=image, image_timestamp_s=0.1)
    )
    can_caught_up = pipeline.step(
        replace(_sample(0.1), image=image, image_timestamp_s=0.1)
    )

    assert not camera_ahead.dp_updated
    assert can_caught_up.dp_updated
    assert dp.calls == 1
    pipeline.close()


def test_open_loop_waits_for_can_sample_within_alignment_limit(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            inference_mode="open_loop",
            action_chunk_mode="all",
            action_step_s=0.1,
        ),
        runtime=replace(
            _config(tmp_path).runtime,
            maximum_observation_alignment_gap_s=0.03,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _ActionDP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)
    image = np.full((8, 10, 3), 20, dtype=np.uint8)

    pipeline.step(replace(_sample(0.06), image=image, image_timestamp_s=0.1))
    gap_too_large = pipeline.step(
        replace(_sample(0.14), image=image, image_timestamp_s=0.1)
    )
    aligned = pipeline.step(
        replace(_sample(0.15), image=image, image_timestamp_s=0.15)
    )

    assert not gap_too_large.dp_updated
    assert aligned.dp_updated
    assert dp.calls == 1
    pipeline.close()


def test_open_loop_builds_two_images_with_eight_distinct_can_samples_each(
    tmp_path: Path,
) -> None:
    dp = _DP()
    dp.obs_encoder = type("Encoder", (), {"wrench_history_steps": 8})()
    dp._inference_checkpoint_config = {
        "task": {"dataset": {"timestamp_step_sec": 0.1}}
    }
    config = replace(
        _config(tmp_path),
        predictor=replace(
            _config(tmp_path).predictor,
            inference_mode="open_loop",
        ),
    )
    pipeline = NeroInferencePipeline(
        config,
        dp_model=dp,
        pinn_model=_PINN(),
        controller=_Controller(),
    )
    for index in range(17):
        can_time = index * 0.0125
        image_time = 0.1 if can_time < 0.2 else 0.2
        pipeline._append_observation(
            np.full((8, 10, 3), image_time, dtype=np.float32),
            np.full(6, index, dtype=np.float32),
            image_timestamp_s=image_time,
            state_timestamp_s=can_time,
            allow_backfill=False,
        )

    snapshot = pipeline._timed_observation_snapshot_if_aligned(0.2)

    assert snapshot is not None
    images, can_values = snapshot
    assert images.shape == (2, 3, 8, 10)
    assert can_values.shape == (2, 8, 6)
    np.testing.assert_allclose(can_values[:, :, 0], np.arange(1, 17).reshape(2, 8))
    assert np.unique(can_values[:, :, 0]).size == 16
    pipeline.close()


def test_multicamera_timestamps_use_wrist_anchor_and_causal_previous_side(
    tmp_path: Path,
) -> None:
    pipeline = NeroInferencePipeline(
        _config(tmp_path),
        dp_model=_TwoCameraDP(),
        pinn_model=_PINN(),
        controller=_Controller(),
    )
    zero_wrench = np.zeros(6, dtype=np.float32)

    observations = (
        (0.026, 0.040, 0.1, 0.3, 0.040),
        (0.066, 0.080, 0.2, 0.4, 0.080),
        # This side frame is newer than the latest wrist frame and must not be
        # paired with it; the training contract is causal, not nearest-neighbor.
        (0.106, 0.080, 0.9, 0.4, 0.081),
    )
    for side_time, wrist_time, side_marker, wrist_marker, state_time in observations:
        pipeline._append_observation(
            {
                "side": np.full((8, 10, 3), side_marker, dtype=np.float32),
                "wrist": np.full((8, 10, 3), wrist_marker, dtype=np.float32),
            },
            zero_wrench,
            image_timestamp_s={"side": side_time, "wrist": wrist_time},
            state_timestamp_s=state_time,
        )

    snapshot = pipeline._timed_observation_snapshot(0.080)

    assert snapshot is not None
    images, _ = snapshot
    assert isinstance(images, dict)
    np.testing.assert_allclose(images["wrist"][:, 0, 0, 0], [0.3, 0.4])
    np.testing.assert_allclose(images["side"][:, 0, 0, 0], [0.1, 0.2])
    assert pipeline._timed_images is pipeline._timed_images_by_key["wrist"]
    pipeline.close()


def test_multicamera_timestamp_mapping_must_match_checkpoint_keys(
    tmp_path: Path,
) -> None:
    pipeline = NeroInferencePipeline(
        _config(tmp_path),
        dp_model=_TwoCameraDP(),
        pinn_model=_PINN(),
        controller=_Controller(),
    )
    images = {
        "side": np.zeros((8, 10, 3), dtype=np.uint8),
        "wrist": np.zeros((8, 10, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="image timestamps do not match"):
        pipeline._append_observation(
            images,
            np.zeros(6),
            image_timestamp_s={"wrist": 0.1},
            state_timestamp_s=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        pipeline.step(
            replace(
                _sample(0.1),
                image=images,
                image_timestamp_s={"side": np.nan, "wrist": 0.1},
            )
        )
    pipeline.close()


def test_absolute_action_condition_is_restored_from_pinn_checkpoint(
    tmp_path: Path,
) -> None:
    dp, pinn, controller = _ActionDP(), _ActionPINN(), _Controller()
    controller.model = _FrameModel()
    pinn._inference_checkpoint_config["model"].update(
        {
            "action_condition_mode": "absolute_pose",
            "action_key": "action_condition_future",
        }
    )
    pipeline = NeroInferencePipeline(
        replace(
            _config(tmp_path),
            predictor=PredictorConfig(action_chunk_mode="all"),
        ),
        dp_model=dp,
        pinn_model=pinn,
        controller=controller,
    )

    pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    pipeline.step(_sample(0.01))

    absolute = pinn.inputs[-1]["action_condition_future"].detach().cpu().numpy()[0]
    np.testing.assert_allclose(
        absolute,
        [
            [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    pipeline.close()


def test_absolute_action_condition_uses_pinn_checkpoint_normalizer(
    tmp_path: Path,
) -> None:
    dp, pinn, controller = _ActionDP(), _ActionPINN(), _Controller()
    controller.model = _FrameModel()
    pinn._inference_checkpoint_config.update(
        {
            "dataloader": {
                "normalize_mode": "gaussian",
                "normalize_lowdim_keys": ["action"],
            }
        }
    )
    pinn._inference_checkpoint_config["model"].update(
        {
            "action_condition_mode": "absolute_pose",
            "action_key": "action_condition_future",
            "action_normalizer_key": "action",
        }
    )
    pinn._inference_normalizer = {
        "normalize_mode": "gaussian",
        "normalize_lowdim_keys": ["action"],
        "eps": 0.0,
        "stats": {
            "action": {
                "mean": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "std": [0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        },
    }
    pipeline = NeroInferencePipeline(
        replace(
            _config(tmp_path),
            predictor=PredictorConfig(action_chunk_mode="all"),
        ),
        dp_model=dp,
        pinn_model=pinn,
        controller=controller,
    )

    pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    pipeline.step(_sample(0.01))

    normalized = (
        pinn.inputs[-1]["action_condition_future"].detach().cpu().numpy()[0]
    )
    np.testing.assert_allclose(
        normalized,
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        atol=1.0e-6,
    )
    pipeline.close()


def test_config_has_no_model_architecture_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: pinn.ckpt, device: cpu}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )
    config = load_inference_config(config_file)
    assert config.dp_checkpoint.path == tmp_path / "dp.ckpt"
    assert config.pinn_checkpoint.path == tmp_path / "pinn.ckpt"
    assert config.predictor.mode == "wrench_gru"

    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, model: transformer}
pinn_checkpoint: {path: pinn.ckpt}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_inference_config(config_file)


def test_config_selects_world_model_v3_runtime_contract(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: wm.ckpt, device: cpu}
predictor: {mode: world_model_v3}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.mode == "world_model_v3"


def test_config_selects_world_model_v4_runtime_contract(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: wm.ckpt, device: cpu}
predictor: {mode: world_model_v4}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.mode == "world_model_v4"


def test_config_selects_world_model_v5_runtime_contract(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: wm.ckpt, device: cpu}
predictor: {mode: world_model_v5}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.mode == "world_model_v5"
    assert config.predictor.action_condition_fill == "auto"


def test_config_validates_action_condition_fill(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt}
pinn_checkpoint: {path: wm.ckpt}
predictor: {mode: world_model_v5, action_condition_fill: stale}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_condition_fill"):
        load_inference_config(config_file)


def test_config_selects_ddpm_sampling_and_inference_steps(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: pinn.ckpt, device: cpu}
dp_sampling: {method: ddpm, num_inference_steps: 25, use_tau_ext_observation: false}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)
    overrides = _dp_model_overrides(config)

    assert config.dp_sampling == DPSamplingConfig("ddpm", 25, False)
    assert overrides["noise_scheduler._target_"].endswith("DDPMScheduler")
    assert overrides["num_inference_steps"] == 25


def test_config_validates_maximum_inference_steps(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt}
pinn_checkpoint: {path: pinn.ckpt}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml, maximum_inference_steps: 12}
""",
        encoding="utf-8",
    )

    assert load_inference_config(config_file).runtime.maximum_inference_steps == 12

    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt}
pinn_checkpoint: {path: pinn.ckpt}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml, maximum_inference_steps: 0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="maximum_inference_steps"):
        load_inference_config(config_file)


def test_config_rejects_unknown_predictor_mode(tmp_path: Path) -> None:
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt}
pinn_checkpoint: {path: wm.ckpt}
predictor: {mode: transformer}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predictor.mode"):
        load_inference_config(config_file)


def test_world_model_v3_uses_full_history_and_physical_state_outputs(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(mode="world_model_v3"),
    )
    model = _WorldModel()
    adapter = _WorldModelWrenchAdapter()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_ActionDP(),
        pinn_model=model,
        controller=_Controller(),
        world_model_wrench_adapter=adapter,
    )

    output = pipeline.step(_sample(0.0))

    assert model.inputs[-1]["q"].shape == (1, 3, 7)
    assert model.inputs[-1]["wrench"].shape == (1, 3, 6)
    assert model.inputs[-1]["action_condition_future"].shape == (1, 2, 7)
    history, future = adapter.calls[-1]
    np.testing.assert_allclose(history["q"], np.zeros((3, 7)))
    for index, key in enumerate(("q", "v", "a", "tau"), start=1):
        np.testing.assert_allclose(future[key], np.full((2, 7), float(index)))
    np.testing.assert_allclose(output.target_wrench, [1, 2, 3, 1, 2, 3])
    pipeline.close()


def test_continuous_can_observation_backfills_world_model_history_without_duplicates(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(mode="world_model_v3", inference_mode="open_loop"),
    )
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_ActionDP(),
        pinn_model=_WorldModel(),
        controller=_Controller(),
        world_model_wrench_adapter=_WorldModelWrenchAdapter(),
    )

    pipeline.append_continuous_can_observation(
        q=np.full(7, 1.0),
        dq=np.full(7, 2.0),
        ddq=np.full(7, 3.0),
        tau=np.full(7, 4.0),
        wrench=np.full(6, 5.0),
        timestamp_s=0.1,
    )
    pipeline.append_continuous_can_observation(
        q=np.full(7, 2.0),
        dq=np.full(7, 3.0),
        ddq=np.full(7, 4.0),
        tau=np.full(7, 5.0),
        wrench=np.full(6, 6.0),
        timestamp_s=0.2,
    )
    pipeline.append_continuous_can_observation(
        q=np.full(7, 2.0),
        dq=np.full(7, 3.0),
        ddq=np.full(7, 4.0),
        tau=np.full(7, 5.0),
        wrench=np.full(6, 6.0),
        timestamp_s=0.2,
    )

    assert len(pipeline._wm_history["q"]) == 3
    np.testing.assert_allclose(pipeline._wm_history["q"][-1], 2.0)
    np.testing.assert_allclose(pipeline._wm_history["wrench"][-1], 6.0)
    pipeline.close()


@pytest.mark.parametrize("mode", ["world_model_v4", "world_model_v5"])
def test_q_tau_world_model_reconstructs_v_a_and_gates_contact(
    tmp_path: Path,
    mode: str,
) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(mode=mode),
    )
    model = _WorldModelV4()
    adapter = _WorldModelWrenchAdapter()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_ActionDP(),
        pinn_model=model,
        controller=_Controller(),
        world_model_wrench_adapter=adapter,
    )

    pipeline.step(_sample(0.0))

    history, future = adapter.calls[-1]
    assert model.inputs[-1]["q"].shape == (1, 3, 7)
    assert set(future) == {"q", "v", "a", "tau"}
    np.testing.assert_allclose(future["q"], np.ones((2, 7)))
    np.testing.assert_allclose(future["tau"], np.full((2, 7), 4.0))
    np.testing.assert_allclose(future["v"], np.full((2, 7), 5.0))
    np.testing.assert_allclose(future["a"], np.full((2, 7), 6.0))
    np.testing.assert_allclose(history["q"], np.zeros((3, 7)))
    np.testing.assert_allclose(pipeline._target_wrenches[0], np.zeros(6))
    np.testing.assert_allclose(
        pipeline._target_wrenches[1],
        np.array([2, 3, 4, 2, 3, 4]),
    )
    pipeline.close()


def test_world_model_v5_holds_latest_dp_action_across_high_rate_steps(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(
            mode="world_model_v5",
            action_chunk_mode="all",
            action_condition_fill="auto",
        ),
    )
    model = _WorldModelV4()
    model._inference_checkpoint_config["model"][
        "action_current_frame_name"
    ] = "link7"
    controller = _Controller()
    controller.model = _FrameModel()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_ActionDP(),
        pinn_model=model,
        controller=controller,
        world_model_wrench_adapter=_WorldModelWrenchAdapter(),
    )

    pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    pipeline.step(_sample(0.01))
    first_condition = model.inputs[-1][
        "action_condition_future"
    ].detach().cpu().numpy()[0]

    pipeline._advance_execution_plan(1.0)
    held_inputs = {}
    pipeline._add_action_condition(held_inputs, torch.device("cpu"), np.eye(4))
    held_condition = held_inputs[
        "action_condition_future"
    ].detach().cpu().numpy()[0]

    np.testing.assert_allclose(first_condition[:, 0], [0.04, 0.04])
    np.testing.assert_allclose(held_condition[:, 0], [0.04, 0.04])
    np.testing.assert_allclose(pipeline._action[0], 0.02)
    pipeline.close()


def test_relative_action_matches_pinn_training_xyzw_convention() -> None:
    root_half = np.sqrt(0.5)
    current = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, root_half, root_half]])
    future = torch.tensor(
        [
            [
                [1.1, 1.8, 3.3, 0.0, 0.0, 1.0, 0.0],
                [1.2, 2.0, 3.0, 0.0, 0.0, -1.0, 0.0],
            ]
        ]
    )

    relative = _relative_action_pose_torch(current, future)

    torch.testing.assert_close(
        relative[0, :, :3],
        torch.tensor(
            [[0.1, -0.2, 0.3], [0.2, 0.0, 0.0]],
            dtype=relative.dtype,
        ),
    )
    expected_quaternion = torch.tensor(
        [0.0, 0.0, root_half, root_half],
        dtype=relative.dtype,
    )
    torch.testing.assert_close(relative[0, 0, 3:], expected_quaternion)
    torch.testing.assert_close(relative[0, 1, 3:], expected_quaternion)
    assert torch.all(relative[..., 6] >= 0.0)


def test_action_safety_preserves_absolute_quaternion_hemisphere(
    tmp_path: Path,
) -> None:
    pipeline = NeroInferencePipeline(
        _config(tmp_path),
        dp_model=_ActionDP(),
        pinn_model=_ActionPINN(),
        controller=_Controller(),
    )
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    safe = pipeline._safe_action(action, np.eye(4))

    assert np.dot(safe[3:], action[3:]) > 0.0
    pipeline.close()


def test_action_chunk_conditions_pinn_and_bridges_link7_to_control_frame(
    tmp_path: Path,
) -> None:
    dp, pinn, controller = _ActionDP(), _ActionPINN(), _Controller()
    controller.model = _FrameModel()
    pipeline = NeroInferencePipeline(
        replace(
            _config(tmp_path),
            predictor=PredictorConfig(action_chunk_mode="all"),
        ),
        dp_model=dp,
        pinn_model=pinn,
        controller=controller,
    )

    pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    output = pipeline.step(_sample(0.01))

    relative = pinn.inputs[-1]["action_relative_future"].detach().cpu().numpy()[0]
    np.testing.assert_allclose(relative[:, :3], [[0.04, 0.0, 0.0], [0.02, 0.0, 0.0]])
    np.testing.assert_allclose(relative[:, 3:], [[0, 0, 0, 1], [0, 0, 0, 1]])
    np.testing.assert_allclose(controller.targets[-1].poses[0][:3, 3], [0.04, 0.0, 1.0])
    np.testing.assert_allclose(controller.targets[-1].poses[1][:3, 3], [0.02, 0.0, 1.0])
    np.testing.assert_allclose(output.action_target[:3], [0.04, 0.0, 0.0])
    pipeline.close()


@pytest.mark.parametrize(
    ("chunk_mode", "expected_x"),
    (("first", 0.04), ("mean", 0.03)),
)
def test_disabled_predictor_executes_selected_dp_action_through_ik(
    tmp_path: Path,
    chunk_mode: str,
    expected_x: float,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            mode="world_model_v3",
            action_chunk_mode=chunk_mode,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_ActionDP(),
        controller=controller,
    )

    initial = pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    output = pipeline.step(_sample(0.01))

    assert initial.qp_result is None
    assert initial.joint_position_command is not None
    assert output.pinn_updated is False
    assert output.qp_result is None
    assert output.ik_result is not None and output.ik_result.converged
    np.testing.assert_allclose(output.action_target[0], expected_x)
    np.testing.assert_allclose(
        output.joint_position_command,
        [expected_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1.0e-6,
    )
    assert controller.targets == []
    pipeline.close()


def test_joint_dp_action_bypasses_ik_and_uses_joint_safety(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        action="joint",
        pinn_checkpoint=None,
        predictor=PredictorConfig(enabled=False, action_chunk_mode="first"),
        safety=replace(
            _config(tmp_path).safety,
            maximum_joint_position_step_rad=0.1,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=_JointActionDP(),
        controller=controller,
    )

    pipeline.step(_sample(0.0))
    assert pipeline._dp_future is not None
    pipeline._dp_future.result(timeout=2.0)
    output = pipeline.step(_sample(0.01))

    assert output.ik_result is None
    np.testing.assert_allclose(
        output.action_target,
        [0.4, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3],
    )
    np.testing.assert_allclose(
        output.joint_position_command,
        [0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1],
    )
    pipeline.close()


def test_joint_diffusion_action_does_not_normalize_q_as_quaternion() -> None:
    model = _JointDiffusionDP()
    output = _predict_dp_action(
        model,
        {"wrist": torch.zeros((1, 1, 3, 8, 10))},
        action_type="joint",
    )

    expected = torch.tensor(
        [[[0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
          [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]]]
    )
    torch.testing.assert_close(output["action"], expected)
    torch.testing.assert_close(output["action_target"], expected.mean(dim=1))
    assert model.conditional_sample_calls == 1
    assert model.predict_action_calls == 0


def test_open_loop_all_executes_complete_chunk_before_reobserving_direct_ik(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            inference_mode="open_loop",
            action_chunk_mode="all",
            action_step_s=0.1,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _ActionDP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)

    first = pipeline.step(_sample(0.0))
    held = pipeline.step(_sample(0.05))
    second = pipeline.step(_sample(0.1))
    replanned = pipeline.step(_sample(0.2))

    assert dp.calls == 2
    assert pipeline._dp_future is None
    np.testing.assert_allclose(first.action_target[0], 0.04, atol=1.0e-6)
    np.testing.assert_allclose(held.action_target[0], 0.04, atol=1.0e-6)
    np.testing.assert_allclose(second.action_target[0], 0.02, atol=1.0e-6)
    np.testing.assert_allclose(replanned.action_target[0], 0.04, atol=1.0e-6)
    assert first.dp_updated and not held.dp_updated and not second.dp_updated
    assert replanned.dp_updated
    pipeline.close()


def test_open_loop_collects_a_fresh_checkpoint_window_after_execution(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            inference_mode="open_loop",
            action_chunk_mode="all",
            action_step_s=0.1,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _TwoObservationActionDP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)

    collecting = pipeline.step(_marked_sample(0.0, 10))
    inferred = pipeline.step(_marked_sample(0.1, 20))
    pipeline.step(_marked_sample(0.15, 99))
    pipeline.step(_marked_sample(0.2, 98))
    next_first = pipeline.step(_marked_sample(0.3, 30))
    next_inferred = pipeline.step(_marked_sample(0.4, 40))

    assert not collecting.dp_updated
    assert inferred.dp_updated
    assert not next_first.dp_updated
    assert next_inferred.dp_updated
    assert dp.calls == 2
    np.testing.assert_allclose(
        dp.observation_markers,
        np.asarray(((10, 20), (30, 40)), dtype=np.float32) / 255.0,
    )
    pipeline.close()


def test_open_loop_all_shifts_predictor_to_remaining_chunk(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(
            enabled=True,
            inference_mode="open_loop",
            action_chunk_mode="all",
            action_step_s=0.1,
        ),
    )
    dp, pinn, controller = _ActionDP(), _ActionPINN(), _Controller()
    controller.model = _FrameModel()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=dp,
        pinn_model=pinn,
        controller=controller,
    )

    first = pipeline.step(_sample(0.0))
    second = pipeline.step(_sample(0.1))
    replanned = pipeline.step(_sample(0.2))

    assert dp.calls == 2
    np.testing.assert_allclose(first.action_target[0], 0.04, atol=1.0e-6)
    np.testing.assert_allclose(second.action_target[0], 0.02, atol=1.0e-6)
    np.testing.assert_allclose(replanned.action_target[0], 0.04, atol=1.0e-6)
    np.testing.assert_allclose(
        controller.targets[1].poses[:, 0, 3],
        [0.02, 0.02],
        atol=1.0e-6,
    )
    remaining = pinn.inputs[1]["action_relative_future"].detach().cpu().numpy()[0]
    np.testing.assert_allclose(remaining[:, 0], [0.02, 0.02], atol=1.0e-6)
    pipeline.close()


@pytest.mark.parametrize(
    ("chunk_mode", "target_x"),
    (("first", 0.04), ("mean", 0.03)),
)
def test_minimum_jerk_target_interpolates_selected_action_before_direct_ik(
    tmp_path: Path,
    chunk_mode: str,
    target_x: float,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(
            enabled=False,
            inference_mode="open_loop",
            action_chunk_mode=chunk_mode,
            action_execution_mode="minimum_jerk_target",
            action_interpolation_duration_s=0.2,
            action_interpolation_steps=2,
        ),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _ActionDP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)

    start = pipeline.step(_sample(0.0))
    midpoint = pipeline.step(_sample(0.1))
    target = pipeline.step(_sample(0.2))

    assert dp.calls == 1
    assert start.dp_updated and not midpoint.dp_updated and not target.dp_updated
    np.testing.assert_allclose(start.action_target[0], 0.0)
    np.testing.assert_allclose(start.joint_position_command[0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(midpoint.action_target[0], target_x / 2.0)
    np.testing.assert_allclose(
        midpoint.joint_position_command[0], target_x / 2.0, atol=1.0e-6
    )
    np.testing.assert_allclose(target.action_target[0], target_x)
    np.testing.assert_allclose(target.joint_position_command[0], target_x, atol=1.0e-6)
    pipeline.close()


def test_minimum_jerk_plan_drives_predictor_horizon(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        predictor=PredictorConfig(
            enabled=True,
            inference_mode="open_loop",
            action_chunk_mode="first",
            action_execution_mode="minimum_jerk_target",
            action_interpolation_duration_s=0.2,
            action_interpolation_steps=2,
            action_condition_fill="chunk",
        ),
    )
    dp, pinn, controller = _ActionDP(), _ActionPINN(), _Controller()
    controller.model = _FrameModel()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=dp,
        pinn_model=pinn,
        controller=controller,
    )

    start = pipeline.step(_sample(0.0))
    output = pipeline.step(_sample(0.1))

    np.testing.assert_allclose(output.action_target[0], 0.02)
    np.testing.assert_allclose(
        controller.targets[-1].poses[:, 0, 3],
        [0.02, 0.04],
    )
    relative = pinn.inputs[-1]["action_relative_future"].detach().cpu().numpy()[0]
    np.testing.assert_allclose(relative[:, 0], [0.02, 0.04])
    np.testing.assert_allclose(start.action_target[0], 0.0)
    pipeline.close()


def test_config_rejects_all_chunk_mode_for_minimum_jerk_target(tmp_path: Path) -> None:
    config_file = tmp_path / "minimum_jerk_target.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
predictor:
  enabled: false
  action_chunk_mode: all
  action_execution_mode: minimum_jerk_target
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_chunk_mode must be 'first'.*'last'"):
        load_inference_config(config_file)


def test_config_loads_minimum_jerk_interpolation_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "minimum_jerk_target.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
predictor:
  enabled: false
  action_chunk_mode: mean
  action_execution_mode: minimum-jerk-target
  action_interpolation_duration_s: 0.25
  action_interpolation_steps: 5
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.action_execution_mode == "minimum_jerk_target"
    assert config.predictor.action_chunk_mode == "mean"
    assert config.predictor.action_interpolation_duration_s == pytest.approx(0.25)
    assert config.predictor.action_interpolation_steps == 5


def test_legacy_linear_target_config_uses_minimum_jerk_mode(tmp_path: Path) -> None:
    config_file = tmp_path / "legacy_linear_target.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
predictor:
  enabled: false
  action_chunk_mode: first
  action_execution_mode: linear_target
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.action_execution_mode == "minimum_jerk_target"


def test_minimum_jerk_plan_uses_c2_time_scaling_and_exact_endpoints() -> None:
    start = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    plan = _minimum_jerk_action_plan(start, target, steps=4)

    expected_progress = np.asarray([0.0, 0.103515625, 0.5, 0.896484375, 1.0])
    np.testing.assert_allclose(plan[:, 0], expected_progress)
    np.testing.assert_allclose(plan[0], start)
    np.testing.assert_allclose(plan[-1], target)
    np.testing.assert_allclose(np.linalg.norm(plan[:, 3:], axis=1), 1.0)


def test_config_allows_no_pinn_checkpoint_when_predictor_is_disabled(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "direct_ik.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
predictor:
  enabled: false
  mode: world_model_v3
  inference_mode: open_loop
  action_chunk_mode: all
  action_step_s: 0.05
robot: {urdf_path: robot.urdf, action_frame_name: link7}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_file)

    assert config.predictor.enabled is False
    assert config.predictor.inference_mode == "open_loop"
    assert config.predictor.action_chunk_mode == "all"
    assert config.predictor.action_step_s == pytest.approx(0.05)
    assert config.pinn_checkpoint is None
    assert config.robot.action_frame_name == "link7"


def test_config_loads_joint_dp_action_semantics(tmp_path: Path) -> None:
    config_file = tmp_path / "joint.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
action: joint
predictor: {enabled: false}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    assert load_inference_config(config_file).action == "joint"


def test_config_rejects_unknown_dp_action_semantics(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid_action.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
action: delta_joint
predictor: {enabled: false}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action must be 'eepose' or 'joint'"):
        load_inference_config(config_file)


def test_legacy_wm_frame_split_is_bound_to_exact_checkpoint(tmp_path: Path) -> None:
    exact = Path(
        "/mnt/code/lcx/PINN/outputs/contact_world_model_opd_sweep/20260819_113728/"
        "teacher/checkpoints/step_00100000.pt"
    )
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=CheckpointConfig(exact, device="cpu"),
    )
    other = replace(
        config,
        pinn_checkpoint=CheckpointConfig(exact.with_name("latest.pt"), device="cpu"),
    )

    assert _uses_link7_target_gripper_tcp_current_contract(config)
    assert not _uses_link7_target_gripper_tcp_current_contract(other)


def test_synchronous_direct_ik_consumes_one_dp_result_per_dataset_frame(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(enabled=False, action_chunk_mode="first"),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _DP()
    pipeline = NeroInferencePipeline(
        config,
        dp_model=dp,
        controller=controller,
    )

    first = pipeline.step_direct_ik_synchronous(_sample(0.0))
    second = pipeline.step_direct_ik_synchronous(_sample(0.01))

    assert dp.calls == 2
    assert pipeline._dp_future is None
    assert first.dp_updated and second.dp_updated
    np.testing.assert_allclose(second.joint_position_command[0], 0.05, atol=1.0e-6)
    pipeline.close()


def test_direct_ik_accepts_pre_aligned_model_observation_history(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        pinn_checkpoint=None,
        predictor=PredictorConfig(enabled=False, action_chunk_mode="first"),
    )
    controller = _Controller()
    controller.model = _IKModel()
    dp = _DP()
    pipeline = NeroInferencePipeline(config, dp_model=dp, controller=controller)
    images = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    wrenches = np.zeros((2, 3, 6), dtype=np.float32)

    output = pipeline.step_direct_ik_observation_history(
        _sample(0.0),
        images,
        wrenches,
    )

    assert dp.calls == 1
    assert output.ik_result is not None and output.ik_result.converged
    np.testing.assert_allclose(output.joint_position_command[0], 0.05, atol=1.0e-6)
    pipeline.close()
