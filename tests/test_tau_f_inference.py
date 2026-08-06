from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import (
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    TauFInferenceConfig,
    _parse_tau_f_inference,
)
from nero_collection.contact_wrench import JointTorqueResidualEstimate
from nero_collection.tau_f_inference import (
    OnlineTauFInference,
    TauFCheckpointMetadata,
    TauFSequencePredictor,
)


def test_tau_f_config_resolves_checkpoint_and_defaults_to_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    config = _parse_tau_f_inference(
        {
            "enabled": True,
            "checkpoint_path": "models/tau_f.pt",
            "device": "cpu",
        },
        tmp_path,
    )

    assert config.checkpoint_path == (tmp_path / "models" / "tau_f.pt").resolve()
    assert config.horizon is None
    assert config.input_keys is None
    assert config.output_key is None


def test_tau_f_config_accepts_checkpoint_relative_to_project_root(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    checkpoint = tmp_path / "models" / "tau_f.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    config = _parse_tau_f_inference(
        {
            "enabled": True,
            "checkpoint_path": "models/tau_f.pt",
        },
        config_dir,
    )

    assert config.checkpoint_path == checkpoint.resolve()


def test_tau_bg_config_accepts_external_torque_processing() -> None:
    config = _parse_tau_f_inference(
        {
            "mode": "tau_bg",
            "tau_ext_lowpass_hz": 10.0,
            "tau_ext_gate_threshold_nm": [0.5] * 7,
            "input_keys": ["q", "tau"],
            "output_key": "tau_bg",
        }
    )

    assert config.mode == "tau_bg"
    assert config.tau_ext_lowpass_hz == pytest.approx(10.0)
    assert config.tau_ext_gate_threshold_nm == pytest.approx((0.5,) * 7)
    assert config.input_keys == ("q", "tau")
    assert config.output_key == "tau_bg"


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True},
        {"horizon": 0},
        {"input_keys": []},
        {"input_keys": ["q", "q"]},
        {"input_keys": ["q", "bad"]},
        {"torch_num_threads": 0},
        {"mode": "unknown"},
        {"tau_ext_lowpass_hz": 0.0},
        {"tau_ext_gate_threshold_nm": [0.0] * 6},
        {"tau_ext_gate_threshold_nm": [-0.1] * 7},
    ],
)
def test_tau_f_config_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        _parse_tau_f_inference(value)


def test_online_inference_computes_external_torque_residual() -> None:
    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("fake.pt"),
        horizon=3,
        input_keys=("q", "dq", "tau"),
        input_dims={"q": 7, "dq": 7, "tau": 7},
        output_key="tau_f",
        output_dim=7,
        architecture="lstm",
        normalize_mode="gaussian",
    )

    class Predictor:
        def __init__(self):
            self.metadata = metadata
            self.warm_up_calls = 0
            self.reset_calls = 0

        def warm_up(self):
            self.warm_up_calls += 1

        def reset_recurrent_state(self):
            self.reset_calls += 1

        def append_and_predict(self, features):
            assert set(features) == {"q", "dq", "ddq", "tau"}
            return np.full(7, 0.5)

    class Estimator:
        def estimate(self, q, dq, ddq, tau):
            tau_id = np.full(7, 3.0)
            zeros = np.zeros(7)
            return JointTorqueResidualEstimate(
                tau_id=tau_id,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=tau_id,
                tau_residual=tau_id - tau,
            )

    inference = OnlineTauFInference(
        TauFInferenceConfig(enabled=True, checkpoint_path=Path("fake.pt")),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(enabled=False),
        {},
        predictor=Predictor(),
        estimator=Estimator(),
    )
    inference.warm_up()
    inference.reset_recurrent_state()
    assert inference.append(1_000_000, np.zeros(7), np.ones(7)) is None
    assert inference.append(1_010_000, np.zeros(7), np.ones(7)) is None
    result = inference.append(1_020_000, np.zeros(7), np.ones(7))

    assert inference.predictor.warm_up_calls == 1
    assert inference.predictor.reset_calls == 1
    assert result is not None
    assert result.timestamp_us == 1_010_000
    assert result.dq == pytest.approx(np.zeros(7))
    assert result.ddq == pytest.approx(np.zeros(7))
    assert result.tau_f_cal == pytest.approx(np.full(7, 2.0))
    assert result.tau_f_pred == pytest.approx(np.full(7, 0.5))
    assert result.tau_ext == pytest.approx(np.full(7, 1.5))


def test_disabled_online_tau_f_inference_uses_zero_prediction() -> None:
    class Estimator:
        @staticmethod
        def estimate(q, dq, ddq, tau):
            tau_id = np.full(7, 3.0)
            zeros = np.zeros(7)
            return JointTorqueResidualEstimate(
                tau_id=tau_id,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=tau_id,
                tau_residual=tau_id - tau,
            )

    inference = OnlineTauFInference(
        TauFInferenceConfig(enabled=False),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(enabled=False),
        {},
        estimator=Estimator(),
    )
    inference.warm_up()
    inference.reset_episode()

    result = inference.estimate_centered(
        1_000_000,
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        np.ones(7),
    )

    np.testing.assert_allclose(result.tau_f_pred, np.zeros(7))
    np.testing.assert_allclose(result.tau_f_cal, np.full(7, 2.0))
    np.testing.assert_allclose(result.tau_ext, np.full(7, 2.0))


def test_tau_bg_mode_filters_gates_and_resets_external_torque() -> None:
    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("fake.pt"),
        horizon=3,
        input_keys=("q", "tau"),
        input_dims={"q": 7, "tau": 7},
        output_key="tau_bg",
        output_dim=7,
        architecture="gru",
        normalize_mode="gaussian",
    )

    class Predictor:
        def __init__(self):
            self.metadata = metadata
            self.reset_calls = 0

        def reset_recurrent_state(self):
            self.reset_calls += 1

        @staticmethod
        def append_and_predict(_features):
            return np.full(7, 0.5)

    class Estimator:
        @staticmethod
        def estimate(q, dq, ddq, tau):
            zeros = np.zeros(7)
            return JointTorqueResidualEstimate(
                tau_id=zeros,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=zeros,
                tau_residual=-np.asarray(tau),
            )

    predictor = Predictor()
    inference = OnlineTauFInference(
        TauFInferenceConfig(
            enabled=True,
            mode="tau_bg",
            checkpoint_path=Path("fake.pt"),
            tau_ext_lowpass_hz=10.0,
            tau_ext_gate_threshold_nm=(0.8,) * 7,
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(enabled=False),
        {},
        predictor=predictor,
        estimator=Estimator(),
    )
    zeros = np.zeros(7)
    first = inference.estimate_centered(
        1_000_000,
        zeros,
        zeros,
        zeros,
        np.ones(7),
    )
    second = inference.estimate_centered(
        1_010_000,
        zeros,
        zeros,
        zeros,
        np.full(7, 2.0),
    )

    alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    expected_filtered = 0.5 + alpha
    np.testing.assert_allclose(first.tau_bg_pred, np.full(7, 0.5))
    np.testing.assert_allclose(first.tau_ext_raw, np.full(7, 0.5))
    np.testing.assert_allclose(first.tau_ext_filtered, np.full(7, 0.5))
    np.testing.assert_allclose(first.tau_ext, np.zeros(7))
    np.testing.assert_allclose(second.tau_ext_raw, np.full(7, 1.5))
    np.testing.assert_allclose(
        second.tau_ext_filtered,
        np.full(7, expected_filtered),
    )
    np.testing.assert_allclose(second.tau_ext, np.full(7, expected_filtered))

    inference.reset_episode()
    replayed = inference.estimate_centered(
        2_000_000,
        zeros,
        zeros,
        zeros,
        np.ones(7),
    )
    assert predictor.reset_calls == 1
    np.testing.assert_allclose(replayed.tau_ext, np.zeros(7))


def test_online_tau_f_batch_preserves_chronological_recurrent_outputs() -> None:
    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("fake.pt"),
        horizon=3,
        input_keys=("q", "dq", "tau"),
        input_dims={"q": 7, "dq": 7, "tau": 7},
        output_key="tau_f",
        output_dim=7,
        architecture="gru",
        normalize_mode="gaussian",
    )

    class Predictor:
        def __init__(self):
            self.metadata = metadata
            self.offset = 0
            self.batch_calls = 0

        def append_and_predict(self, _features):
            value = np.full(7, float(self.offset))
            self.offset += 1
            return value

        def append_sequence_and_predict(self, features):
            self.batch_calls += 1
            count = features["q"].shape[0]
            result = np.stack(
                [np.full(7, float(self.offset + index)) for index in range(count)]
            )
            self.offset += count
            return result

    class Estimator:
        @staticmethod
        def estimate(q, dq, ddq, tau):
            zeros = np.zeros(7)
            tau_id = np.asarray(tau) + 2.0
            return JointTorqueResidualEstimate(
                tau_id=tau_id,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=tau_id,
                tau_residual=tau_id - tau,
            )

    predictor = Predictor()
    inference = OnlineTauFInference(
        TauFInferenceConfig(enabled=True, checkpoint_path=Path("fake.pt")),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(enabled=False),
        {},
        predictor=predictor,
        estimator=Estimator(),
    )
    zeros = np.zeros(7)
    results = inference.estimate_aligned_raw_batch(
        tuple(
            (1_000_000 + index * 10_000, zeros, zeros, zeros, zeros)
            for index in range(3)
        )
    )

    assert predictor.batch_calls == 1
    assert [result.timestamp_us for result in results] == [
        1_000_000,
        1_010_000,
        1_020_000,
    ]
    np.testing.assert_allclose(
        np.stack([result.tau_f_pred for result in results]),
        np.repeat(np.arange(3, dtype=np.float64)[:, None], 7, axis=1),
    )
    np.testing.assert_allclose(
        np.stack([result.tau_ext for result in results]),
        2.0
        - np.repeat(np.arange(3, dtype=np.float64)[:, None], 7, axis=1),
    )


def test_online_inference_episode_reset_clears_all_stream_history() -> None:
    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("fake.pt"),
        horizon=3,
        input_keys=("q", "dq", "tau"),
        input_dims={"q": 7, "dq": 7, "tau": 7},
        output_key="tau_f",
        output_dim=7,
        architecture="gru",
        normalize_mode="gaussian",
    )

    class Predictor:
        def __init__(self):
            self.metadata = metadata
            self.reset_calls = 0

        def reset_recurrent_state(self):
            self.reset_calls += 1

        def append_and_predict(self, _features):
            return np.zeros(7)

    class Estimator:
        def estimate(self, q, dq, ddq, tau):
            zeros = np.zeros(7)
            return JointTorqueResidualEstimate(
                tau_id=zeros,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=zeros,
                tau_residual=-tau,
            )

    predictor = Predictor()
    inference = OnlineTauFInference(
        TauFInferenceConfig(enabled=True, checkpoint_path=Path("fake.pt")),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(enabled=False),
        {},
        predictor=predictor,
        estimator=Estimator(),
    )
    zeros = np.zeros(7)
    inference.append(1_000_000, zeros, zeros)
    inference.append(1_010_000, zeros, zeros)

    inference.reset_episode()

    assert predictor.reset_calls == 1
    assert inference.append(2_000_000, zeros, zeros) is None
    assert inference.append(2_010_000, zeros, zeros) is None
    assert inference.append(2_020_000, zeros, zeros) is not None


def test_aligned_raw_sample_uses_collection_torque_filter() -> None:
    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("fake.pt"),
        horizon=1,
        input_keys=("q", "dq", "tau"),
        input_dims={"q": 7, "dq": 7, "tau": 7},
        output_key="tau_f",
        output_dim=7,
        architecture="gru",
        normalize_mode="gaussian",
    )

    class Predictor:
        def __init__(self):
            self.metadata = metadata

        def append_and_predict(self, _features):
            return np.zeros(7)

    class Estimator:
        def estimate(self, q, dq, ddq, tau):
            zeros = np.zeros(7)
            return JointTorqueResidualEstimate(
                tau_id=zeros,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=zeros,
                tau_residual=-tau,
            )

    inference = OnlineTauFInference(
        TauFInferenceConfig(enabled=True, checkpoint_path=Path("fake.pt")),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(
            enabled=True,
            torque_lowpass_hz=10.0,
            torque_median_window=1,
        ),
        {},
        predictor=Predictor(),
        estimator=Estimator(),
    )
    zeros = np.zeros(7)
    inference.estimate_aligned_raw(
        1_000_000,
        zeros,
        zeros,
        zeros,
        np.zeros(7),
    )
    result = inference.estimate_aligned_raw(
        1_010_000,
        zeros,
        zeros,
        zeros,
        np.ones(7),
    )

    expected_alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    assert result.tau == pytest.approx(np.full(7, expected_alpha))


CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[2] / "PINN" / "outputs" / "tau_f_sequence"
)
CHECKPOINTS = {
    "lstm": CHECKPOINT_ROOT
    / "lstm_h50_plus"
    / "checkpoints"
    / "epoch_187_val_loss_0.006817.pt",
    "gru": CHECKPOINT_ROOT
    / "gru_h50_plus"
    / "checkpoints"
    / "epoch_241_val_loss_0.005748.pt",
}


@pytest.mark.skipif(find_spec("torch") is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("architecture", ("lstm", "gru"))
def test_real_checkpoint_restores_config_and_streams(architecture: str) -> None:
    checkpoint = CHECKPOINTS[architecture]
    if not checkpoint.is_file():
        pytest.skip(f"tau_f {architecture} checkpoint is unavailable")
    predictor = TauFSequencePredictor(
        TauFInferenceConfig(
            enabled=True,
            checkpoint_path=checkpoint,
            device="cpu",
            torch_num_threads=1,
        )
    )
    predictor.warm_up()
    frames = [
        {
            key: np.full(dim, float(step) / 10.0)
            for key, dim in predictor.metadata.input_dims.items()
        }
        for step in range(predictor.metadata.horizon)
    ]
    predictions = [predictor.append_and_predict(frame) for frame in frames]

    chronological = {
        key: np.stack([frame[key] for frame in frames])
        for key in predictor.metadata.input_keys
    }
    full_sequence_prediction = predictor._predict_chronological(chronological)

    predictor.reset_recurrent_state()
    batched_predictions = predictor.append_sequence_and_predict(chronological)
    predictor.reset_recurrent_state()
    replayed_first = predictor.append_and_predict(frames[0])

    assert predictor.metadata.horizon == 50
    assert predictor.metadata.input_keys == ("q", "dq", "tau")
    assert predictor.metadata.output_key == "tau_f"
    assert predictor.metadata.architecture == architecture
    assert predictions[-1] == pytest.approx(full_sequence_prediction, abs=1e-6)
    assert batched_predictions == pytest.approx(np.stack(predictions), abs=1e-6)
    assert replayed_first == pytest.approx(predictions[0], abs=1e-6)
    assert np.isfinite(np.stack(predictions)).all()
