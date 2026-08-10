from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nero_collection.causal_kalman import CausalJointKalmanFilter, CausalJointState
from nero_collection.config import (
    CausalKalmanConfig,
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    SequenceCheckpointConfig,
    StateParamConfig,
    TauExtInferenceConfig,
    _parse_tau_ext_inference,
    load_config,
)
from nero_collection.filters import CausalWindowLowPass, OnePoleLowPass
from nero_collection.tau_ext_inference import (
    OnlineTauExtInference,
    SequenceCheckpointMetadata,
    SequenceTorquePredictor,
    _build_checkpoint_model,
    _resolve_checkpoint_path,
)


def _metadata(output_key: str) -> SequenceCheckpointMetadata:
    return SequenceCheckpointMetadata(
        checkpoint_path=Path(f"{output_key}.pt"),
        horizon=50,
        input_keys=("q", "dq", "delta_q"),
        input_dims={"q": 7, "dq": 7, "delta_q": 7},
        output_key=output_key,
        output_dim=7,
        architecture="lstm",
        normalize_mode="gaussian",
        target_contract=(
            "matched_causal_torque_filter_v1" if output_key == "tau_f" else None
        ),
        target_filter_enabled=True,
        target_filter_cutoff_hz=10.0 if output_key == "tau_f" else None,
        target_filter_median_window=1 if output_key == "tau_f" else 3,
    )


def test_checkpoint_directory_ranks_any_named_loss_metric(tmp_path: Path) -> None:
    worse = tmp_path / "epoch_005_train_eval_loss_0.012000.pt"
    best = tmp_path / "epoch_004_train_eval_loss_0.008000.pt"
    worse.touch()
    best.touch()

    assert _resolve_checkpoint_path(tmp_path, "tau_f") == best.resolve()


class _Predictor:
    def __init__(
        self,
        output_key: str,
        value: np.ndarray,
        *,
        ready_after: int = 1,
    ) -> None:
        self.metadata = _metadata(output_key)
        self.value = np.asarray(value, dtype=np.float64)
        self.ready_after = int(ready_after)
        self.feature_ids: list[int] = []
        self.features: list[dict[str, np.ndarray]] = []
        self.reset_count = 0

    def append_and_predict(self, features):
        self.feature_ids.append(id(features))
        self.features.append({key: value.copy() for key, value in features.items()})
        if len(self.features) < self.ready_after:
            return None
        return self.value.copy()

    def warm_up(self):
        pass

    def reset(self):
        self.reset_count += 1

    def predict_sequence(self, features):
        return np.repeat(self.value[None], len(features["q"]), axis=0)


class _StateEstimator:
    def __init__(self, ddq: np.ndarray) -> None:
        self.ddq = np.asarray(ddq, dtype=np.float64)
        self.calls = []

    def update(self, timestamp_us, q, dq):
        self.calls.append((int(timestamp_us), q.copy(), dq.copy()))
        return CausalJointState(
            timestamp_us=int(timestamp_us),
            q=q.copy(),
            dq=dq.copy(),
            ddq=self.ddq.copy(),
        )

    def reset(self):
        self.calls.clear()


class _InverseDynamics:
    def __init__(self, tau_id: np.ndarray) -> None:
        self.tau_id = np.asarray(tau_id, dtype=np.float64)
        self.calls = []

    def estimate(self, q, dq, ddq, tau):
        self.calls.append((q.copy(), dq.copy(), ddq.copy(), tau.copy()))
        return type("Estimate", (), {"tau_id": self.tau_id.copy()})()


def _inference(
    tau_f_predictor,
    tau_next_predictor,
    estimator,
    state_estimator,
) -> OnlineTauExtInference:
    return OnlineTauExtInference(
        TauExtInferenceConfig(enabled=True),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {
            "torque": StateParamConfig(
                lowpass=True,
                lowpass_cutoff_hz=10.0,
                median_window=1,
            )
        },
        tau_f_predictor=tau_f_predictor,
        tau_next_predictor=tau_next_predictor,
        estimator=estimator,
        state_estimator=state_estimator,
    )


def test_dual_inference_uses_one_aligned_feature_sample_and_exact_formulas() -> None:
    tau_f = _Predictor("tau_f", np.full(7, 0.5))
    tau_next = _Predictor("tau", np.full(7, 4.0))
    inverse_dynamics = _InverseDynamics(np.full(7, 2.0))
    state_estimator = _StateEstimator(np.full(7, 3.0))
    inference = _inference(tau_f, tau_next, inverse_dynamics, state_estimator)
    q = np.linspace(0.0, 0.6, 7)
    dq = np.linspace(0.1, 0.7, 7)
    tau = np.full(7, 10.0)
    q_cmd = q + 0.2

    result = inference.estimate_aligned(1_000_000, q, dq, tau, q_cmd)

    assert tau_f.feature_ids == tau_next.feature_ids
    assert len(tau_f.feature_ids) == 1
    np.testing.assert_allclose(tau_f.features[0]["delta_q"], 0.2)
    np.testing.assert_allclose(tau_next.features[0]["delta_q"], 0.2)
    np.testing.assert_allclose(inverse_dynamics.calls[0][2], 3.0)
    np.testing.assert_allclose(result.ddq_kf_causal, 3.0)
    np.testing.assert_allclose(result.tau_ext_cal, -7.5)
    np.testing.assert_allclose(result.tau_ext_pred, -6.0)


def test_tau_id_uses_the_same_causal_filter_as_measured_torque() -> None:
    inverse_dynamics = _InverseDynamics(np.full(7, 2.0))
    inference = _inference(
        _Predictor("tau_f", np.full(7, 0.5)),
        _Predictor("tau", np.full(7, 4.0)),
        inverse_dynamics,
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)
    tau = np.full(7, 10.0)

    inference.estimate_aligned(1_000_000, zeros, zeros, tau, zeros)
    inverse_dynamics.tau_id = np.full(7, 4.0)
    result = inference.estimate_aligned(1_010_000, zeros, zeros, tau, zeros)

    alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    expected_tau_id_filtered = 2.0 + alpha * (4.0 - 2.0)
    np.testing.assert_allclose(result.tau_id, 4.0)
    np.testing.assert_allclose(result.tau_id_filtered, expected_tau_id_filtered)
    np.testing.assert_allclose(
        result.tau_ext_cal_raw,
        expected_tau_id_filtered + 0.5 - 10.0,
    )
    assert np.all(result.tau_ext_cal < result.tau_ext_cal_raw)

    inference.reset_episode()
    reset_result = inference.estimate_aligned(2_000_000, zeros, zeros, tau, zeros)
    np.testing.assert_allclose(reset_result.tau_id_filtered, 4.0)


def test_tau_next_replays_three_point_median_after_shared_source_filter() -> None:
    inference = _inference(
        _Predictor("tau_f", np.zeros(7)),
        _Predictor("tau", np.zeros(7)),
        _InverseDynamics(np.zeros(7)),
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)

    inference.estimate_aligned(1_000_000, zeros, zeros, zeros, zeros)
    spike = inference.estimate_aligned(
        1_010_000,
        zeros,
        zeros,
        np.full(7, 10.0),
        zeros,
    )

    alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    np.testing.assert_allclose(spike.tau, alpha * 10.0)
    np.testing.assert_allclose(spike.tau_ext_cal_raw, -alpha * 10.0)
    np.testing.assert_allclose(spike.tau_ext_pred_raw, 0.0)

    inference.reset_episode()
    restarted = inference.estimate_aligned(
        2_000_000,
        zeros,
        zeros,
        np.full(7, 5.0),
        zeros,
    )
    np.testing.assert_allclose(restarted.tau_ext_pred_raw, -5.0)


def test_tau_f_checkpoint_rejects_old_or_mismatched_filter_contract() -> None:
    tau_f = _Predictor("tau_f", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inverse_dynamics = _InverseDynamics(np.zeros(7))
    state_estimator = _StateEstimator(np.zeros(7))

    tau_f.metadata = replace(tau_f.metadata, target_contract=None)
    with pytest.raises(RuntimeError, match="rebuild matched-filter labels"):
        _inference(tau_f, tau_next, inverse_dynamics, state_estimator)

    tau_f.metadata = replace(
        _metadata("tau_f"),
        target_filter_cutoff_hz=8.0,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        _inference(tau_f, tau_next, inverse_dynamics, state_estimator)


def test_same_timestamp_returns_cached_dual_result_without_advancing_models() -> None:
    tau_f = _Predictor("tau_f", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inference = _inference(
        tau_f,
        tau_next,
        _InverseDynamics(np.zeros(7)),
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)
    first = inference.estimate_aligned(100, zeros, zeros, zeros, zeros)
    second = inference.estimate_aligned(100, np.ones(7), zeros, zeros, zeros)

    assert second is first
    assert len(tau_f.features) == len(tau_next.features) == 1


def test_dual_inference_returns_zero_until_full_real_history_is_ready() -> None:
    tau_f = _Predictor("tau_f", np.full(7, 0.5), ready_after=50)
    tau_next = _Predictor("tau", np.full(7, 4.0), ready_after=50)
    inference = _inference(
        tau_f,
        tau_next,
        _InverseDynamics(np.full(7, 2.0)),
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)

    for index in range(49):
        result = inference.estimate_aligned(
            1_000_000 + index * 10_000,
            zeros,
            zeros,
            np.full(7, 10.0),
            zeros,
        )
        assert not result.history_ready
        np.testing.assert_allclose(result.tau_f_pred, 0.0)
        np.testing.assert_allclose(result.tau_next_pred, 0.0)
        np.testing.assert_allclose(result.tau_ext_cal, 0.0)
        np.testing.assert_allclose(result.tau_ext_pred, 0.0)

    result = inference.estimate_aligned(
        1_490_000,
        zeros,
        zeros,
        np.full(7, 10.0),
        zeros,
    )

    assert result.history_ready
    np.testing.assert_allclose(result.tau_ext_cal, -7.5)
    np.testing.assert_allclose(result.tau_ext_pred, -6.0)


def test_causal_kalman_matches_pinn_forward_filter() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "PINN/data_process/offline_tau_labels.py"
    )
    if not source.is_file():
        pytest.skip("PINN offline label module is unavailable")
    spec = importlib.util.spec_from_file_location("pinn_offline_tau_labels", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    timestamps_s = np.asarray([0.0, 0.009, 0.020, 0.031, 0.045])
    q = np.stack(
        [np.sin(timestamps_s * (joint + 1)) for joint in range(7)], axis=1
    )
    dq = np.stack(
        [
            (joint + 1) * np.cos(timestamps_s * (joint + 1))
            for joint in range(7)
        ],
        axis=1,
    )
    config = CausalKalmanConfig()
    online = CausalJointKalmanFilter(config)
    online_ddq = np.stack(
        [
            online.update(int(round(timestamp * 1.0e6)), q_value, dq_value).ddq
            for timestamp, q_value, dq_value in zip(timestamps_s, q, dq)
        ]
    )
    reference = module.estimate_joint_states_rts(
        timestamps_s,
        q,
        dq,
        module.KalmanRTSConfig(
            position_std=config.position_std,
            velocity_std=config.velocity_std,
            jerk_std=config.jerk_std,
            initial_position_std=config.initial_position_std,
            initial_velocity_std=config.initial_velocity_std,
            initial_acceleration_std=config.initial_acceleration_std,
            max_gap_s=config.max_gap_s,
        ),
    )
    np.testing.assert_allclose(online_ddq, reference.ddq_filtered, atol=1.0e-12)


def test_online_torque_filter_matches_pinn_label_filter() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "PINN/data_process/offline_tau_labels.py"
    )
    if not source.is_file():
        pytest.skip("PINN offline label module is unavailable")
    spec = importlib.util.spec_from_file_location("pinn_tau_filter", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    timestamp_us = np.asarray([1_000_000, 1_009_000, 1_020_000, 1_034_000])
    values = np.stack(
        [
            np.linspace(0.0, 0.6, 7),
            np.linspace(0.2, 0.8, 7),
            np.linspace(-0.1, 0.5, 7),
            np.linspace(0.4, 1.0, 7),
        ]
    )
    reference = module.causal_median_one_pole_filter(
        timestamp_us * 1.0e-6,
        values,
        cutoff_hz=10.0,
        median_window=1,
    )
    online_filter = OnePoleLowPass(10.0, 1)
    online = np.stack(
        [
            online_filter.apply(value, int(timestamp))
            for timestamp, value in zip(timestamp_us, values)
        ]
    )

    np.testing.assert_allclose(online, reference, atol=1.0e-12)


def test_tau_ext_filter_matches_pinn_causal_moving_average_then_lowpass() -> None:
    timestamp_us = np.asarray([1_000_000, 1_010_000, 1_021_000, 1_030_000])
    values = np.asarray([[0.0], [3.0], [-1.0], [5.0]], dtype=np.float64)
    window = 3
    padded = np.concatenate(
        [np.repeat(values[:1], window - 1, axis=0), values],
        axis=0,
    )
    averaged = np.asarray(
        [padded[index : index + window].mean(axis=0) for index in range(len(values))]
    )
    reference_filter = OnePoleLowPass(20.0, 1)
    reference = np.stack(
        [
            reference_filter.apply(value, int(timestamp))
            for timestamp, value in zip(timestamp_us, averaged)
        ]
    )
    online_filter = CausalWindowLowPass("moving_average", window, 20.0)
    actual = np.stack(
        [
            online_filter.apply(value, int(timestamp))
            for timestamp, value in zip(timestamp_us, values)
        ]
    )

    np.testing.assert_allclose(actual, reference, atol=1.0e-12)

    online_filter.reset()
    np.testing.assert_allclose(
        online_filter.apply(np.asarray([7.0]), 2_000_000),
        7.0,
    )


def test_causal_kalman_resets_acceleration_after_large_gap() -> None:
    kalman = CausalJointKalmanFilter(CausalKalmanConfig(max_gap_s=0.05))
    zeros = np.zeros(7)
    kalman.update(0, zeros, zeros)
    moving = kalman.update(10_000, np.full(7, 0.01), np.ones(7))
    assert np.max(np.abs(moving.ddq)) > 0.0
    reset = kalman.update(100_000, np.full(7, 0.02), np.ones(7))
    np.testing.assert_allclose(reset.ddq, 0.0)


def test_tau_ext_config_requires_both_checkpoints_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both checkpoint paths"):
        _parse_tau_ext_inference(
            {
                "enabled": True,
                "tau_f": {"checkpoint_path": "tau_f.pt"},
            },
            tmp_path,
        )

    config = _parse_tau_ext_inference(
        {
            "enabled": True,
            "tau_f": {
                "checkpoint_path": "tau_f.pt",
                "input_keys": ["q", "dq", "delta_q"],
                "output_key": "tau_f",
            },
            "tau_next": {
                "checkpoint_path": "tau_next.pt",
                "input_keys": ["q", "dq", "delta_q"],
                "output_key": "tau",
            },
        },
        tmp_path,
    )
    assert config.tau_f.output_key == "tau_f"
    assert config.tau_next.output_key == "tau"
    assert config.tau_ext_filter.enabled
    assert config.tau_ext_filter.mode == "moving_average"
    assert config.tau_ext_filter.window == 21
    assert config.tau_ext_filter.cutoff_hz == pytest.approx(20.0)


def test_tau_ext_filter_config_is_explicit_and_validated(tmp_path: Path) -> None:
    common = {
        "enabled": True,
        "tau_f": {"checkpoint_path": "tau_f.pt"},
        "tau_next": {"checkpoint_path": "tau_next.pt"},
    }
    config = _parse_tau_ext_inference(
        {
            **common,
            "tau_ext_filter": {
                "enabled": False,
                "mode": "median",
                "window": 5,
                "cutoff_hz": 8.0,
            },
        },
        tmp_path,
    )
    assert not config.tau_ext_filter.enabled
    assert config.tau_ext_filter.mode == "median"
    assert config.tau_ext_filter.window == 5
    assert config.tau_ext_filter.cutoff_hz == pytest.approx(8.0)

    with pytest.raises(ValueError, match="must be odd"):
        _parse_tau_ext_inference(
            {
                **common,
                "tau_ext_filter": {"mode": "median", "window": 4},
            },
            tmp_path,
        )


def test_collection_config_rejects_removed_tau_f_inference_key(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/master_slave_can.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        source.read_text(encoding="utf-8").replace(
            "tau_ext_inference:",
            "tau_f_inference:",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tau_f_inference was removed"):
        load_config(config_path)


@pytest.mark.parametrize("architecture", ["lstm", "gru"])
def test_sequence_checkpoint_restores_configured_recurrent_type(
    tmp_path: Path,
    architecture: str,
) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model_config = {
        "architecture": architecture,
        "inputs": ["q", "dq", "delta_q"],
        "input_dims": {"q": 7, "dq": 7, "delta_q": 7},
        "target_key": "tau_f",
        "output_dim": 7,
        "hidden_dim": 8,
        "num_layers": 1,
        "head_hidden_dim": 16,
        "head_num_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "target_contract": "matched_causal_torque_filter_v1",
        "target_filter": {"cutoff_hz": 10.0, "median_window": 1},
    }
    model = _build_checkpoint_model(
        torch,
        nn,
        model_config,
        tuple(model_config["inputs"]),
        model_config["input_dims"],
    )
    stats = {
        key: {"mean": np.zeros(7), "std": np.ones(7)}
        for key in ("q", "dq", "delta_q", "tau_f")
    }
    path = tmp_path / f"{architecture}.pt"
    torch.save(
        {
            "config": {
                "dataloader": {
                    "horizon": 4,
                    "normalize_mode": "gaussian",
                    "normalize_lowdim_keys": list(stats),
                },
                "model": model_config,
            },
            "normalizer": {
                "stats": stats,
                "normalize_mode": "gaussian",
                "normalize_lowdim_keys": list(stats),
                "eps": 1.0e-6,
            },
            "model": model.state_dict(),
        },
        path,
    )
    predictor = SequenceTorquePredictor(
        SequenceCheckpointConfig(checkpoint_path=path),
        name="tau_f",
    )
    features = {key: np.zeros(7) for key in ("q", "dq", "delta_q")}
    for _ in range(3):
        assert predictor.append_and_predict(features) is None
    prediction = predictor.append_and_predict(features)
    assert predictor.metadata.architecture == architecture
    assert predictor.metadata.target_filter_enabled
    assert predictor.metadata.target_filter_median_window == 1
    assert not predictor.metadata.target_filter_apply_additional_lowpass
    assert prediction is not None
    assert prediction.shape == (7,)
    predictor.reset()
    assert predictor.append_and_predict(features) is None
