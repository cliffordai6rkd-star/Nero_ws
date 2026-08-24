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
    SourceButterworthFilterConfig,
    StateParamConfig,
    TauExtInferenceConfig,
    _parse_tau_ext_inference,
    load_config,
)
from nero_collection.filters import (
    CausalFilterPipeline,
    CausalHampelButterworth,
    CausalWindowLowPass,
    OnePoleLowPass,
    VariableStepButterworthLowPass,
)
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
            "causal_gravity_residual_v1" if output_key == "tau_other" else None
        ),
        derived_target_config=(
            {
                "enabled": True,
                "method": "causal_gravity_residual_v1",
                "target_key": "tau_other",
                "source_keys": {"q": "q", "dq": "dq", "tau": "tau"},
                "dq_sign": [1.0] * 7,
                "torque_filter_key": "tau",
                "torque_filter_operations": [],
                "ddq_source": "unused",
                "measured_tau_source": "observation.torque",
                "residual_formula": "tau_other=tau_measured-tau_g",
            }
            if output_key == "tau_other"
            else {}
        ),
        target_filter_enabled=output_key != "tau_other",
        target_filter_cutoff_hz=None,
        target_filter_moving_average_window=(3 if output_key != "tau_other" else None),
        target_filter_median_window=None,
    )


def test_checkpoint_directory_ranks_any_named_loss_metric(tmp_path: Path) -> None:
    worse = tmp_path / "epoch_005_train_eval_loss_0.012000.pt"
    best = tmp_path / "epoch_004_train_eval_loss_0.008000.pt"
    worse.touch()
    best.touch()

    assert _resolve_checkpoint_path(tmp_path, "tau_other") == best.resolve()


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
        self.tau_g = self.tau_id.copy()
        self.calls = []

    def estimate(self, q, dq, ddq, tau):
        self.calls.append((q.copy(), dq.copy(), ddq.copy(), tau.copy()))
        return type("Estimate", (), {"tau_id": self.tau_id.copy()})()

    def gravity_torque(self, q):
        return self.tau_g.copy()


def _inference(
    tau_other_predictor,
    tau_next_predictor,
    estimator,
    state_estimator,
    *,
    observation_sample_rate_hz: float = 100.0,
) -> OnlineTauExtInference:
    return OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_other=SequenceCheckpointConfig(
                observation_sample_rate_hz=observation_sample_rate_hz,
            ),
            tau_next=SequenceCheckpointConfig(
                observation_sample_rate_hz=observation_sample_rate_hz,
            ),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {
            "torque": StateParamConfig(
                lowpass=True,
                lowpass_cutoff_hz=10.0,
                median_window=1,
            )
        },
        tau_other_predictor=tau_other_predictor,
        tau_next_predictor=tau_next_predictor,
        estimator=estimator,
        state_estimator=state_estimator,
    )


def test_dual_inference_uses_one_aligned_feature_sample_and_exact_formulas() -> None:
    tau_other = _Predictor("tau_other", np.full(7, 0.5))
    tau_next = _Predictor("tau", np.full(7, 4.0))
    inverse_dynamics = _InverseDynamics(np.full(7, 2.0))
    state_estimator = _StateEstimator(np.full(7, 3.0))
    inference = _inference(tau_other, tau_next, inverse_dynamics, state_estimator)
    q = np.linspace(0.0, 0.6, 7)
    dq = np.linspace(0.1, 0.7, 7)
    tau = np.full(7, 10.0)
    q_cmd = q + 0.2

    result = inference.estimate_aligned(1_000_000, q, dq, tau, q_cmd)

    assert len(tau_other.feature_ids) == len(tau_next.feature_ids)
    assert len(tau_other.feature_ids) == 1
    np.testing.assert_allclose(tau_other.features[0]["delta_q"], 0.2)
    np.testing.assert_allclose(tau_next.features[0]["delta_q"], 0.2)
    np.testing.assert_allclose(inverse_dynamics.calls[0][2], 3.0)
    np.testing.assert_allclose(result.ddq_kf_causal, 3.0)
    np.testing.assert_allclose(result.tau_ext_cal, -7.5)
    np.testing.assert_allclose(result.tau_ext_pred, -6.0)


def test_tau_next_can_use_dynamics_inputs_without_leaking_them_to_tau_other() -> None:
    tau_other = _Predictor("tau_other", np.full(7, 0.5))
    tau_next = _Predictor("tau", np.full(7, 4.0))
    input_keys = ("q", "dq", "ddq", "delta_q", "tau", "tau_id")
    input_dims = {key: 7 for key in input_keys}
    tau_next.metadata = replace(
        tau_next.metadata, input_keys=input_keys, input_dims=input_dims
    )
    inverse_dynamics = _InverseDynamics(np.full(7, 2.0))
    state_estimator = _StateEstimator(np.full(7, 3.0))
    inference = _inference(tau_other, tau_next, inverse_dynamics, state_estimator)

    inference.estimate_aligned(
        1_000_000,
        np.zeros(7),
        np.ones(7),
        np.full(7, 10.0),
        np.full(7, 0.2),
    )
    inverse_dynamics.tau_id = np.full(7, 4.0)
    result = inference.estimate_aligned(
        1_020_000,
        np.zeros(7),
        np.ones(7),
        np.full(7, 10.0),
        np.full(7, 0.2),
    )

    assert set(tau_other.features[-1]) == {"q", "dq", "delta_q"}
    np.testing.assert_allclose(tau_next.features[-1]["ddq"], 3.0)
    np.testing.assert_allclose(tau_next.features[-1]["tau_id"], 4.0)
    np.testing.assert_allclose(result.tau_g, 2.0)


def test_tau_free_dynamics_inputs_require_tau_other_provider() -> None:
    tau_next = _Predictor("tau", np.zeros(7))
    tau_next.metadata = replace(
        tau_next.metadata,
        input_keys=("q", "ddq", "tau_id"),
        input_dims={"q": 7, "ddq": 7, "tau_id": 7},
    )
    with pytest.raises(RuntimeError, match="require the tau_other branch"):
        OnlineTauExtInference(
            TauExtInferenceConfig(
                enabled=True,
                tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
            ),
            InverseDynamicsConfig(),
            DynamicsProcessingConfig(),
            {},
            tau_next_predictor=tau_next,
        )


def test_tau_next_only_skips_tau_other_and_produces_prediction() -> None:
    tau_next = _Predictor("tau", np.full(7, 4.0))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {},
        tau_next_predictor=tau_next,
    )
    zeros = np.zeros(7)

    result = inference.estimate_aligned(
        1_000_000,
        zeros,
        zeros,
        np.full(7, 10.0),
        zeros,
    )

    assert inference.tau_other_predictor is None
    assert result.history_ready
    np.testing.assert_allclose(result.tau_other_pred, 0.0)
    np.testing.assert_allclose(result.tau_ext_cal, 0.0)
    np.testing.assert_allclose(result.tau_next_pred, 4.0)
    np.testing.assert_allclose(result.tau_ext_pred, -6.0)


def test_each_active_checkpoint_uses_its_own_observation_rate() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_other=SequenceCheckpointConfig(observation_sample_rate_hz=100.0),
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {"torque": StateParamConfig(lowpass=True, lowpass_cutoff_hz=10.0)},
        tau_other_predictor=tau_other,
        tau_next_predictor=tau_next,
        estimator=_InverseDynamics(np.zeros(7)),
        state_estimator=_StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)

    for index in range(11):
        inference.estimate_aligned(
            1_000_000 + index * 10_000,
            zeros,
            zeros,
            zeros,
            zeros,
        )

    assert len(tau_other.features) == 11
    assert len(tau_next.features) == 6


def test_source_butterworth_filters_all_features_once_before_both_samplers() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inverse_dynamics = _InverseDynamics(np.zeros(7))
    state_estimator = _StateEstimator(np.zeros(7))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_other=SequenceCheckpointConfig(observation_sample_rate_hz=100.0),
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=100.0),
            source_butterworth_filter=SourceButterworthFilterConfig(
                enabled=True,
                cutoff_hz=15.0,
                order=2,
            ),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {"torque": StateParamConfig(lowpass=True, lowpass_cutoff_hz=10.0)},
        tau_other_predictor=tau_other,
        tau_next_predictor=tau_next,
        estimator=inverse_dynamics,
        state_estimator=state_estimator,
    )
    zeros = np.zeros(7)
    ones = np.ones(7)

    inference.estimate_aligned(1_000_000, zeros, zeros, zeros, zeros)
    inference.estimate_aligned(
        1_010_000,
        ones,
        2.0 * ones,
        3.0 * ones,
        4.0 * ones,
    )

    q_reference = VariableStepButterworthLowPass(15.0)
    dq_reference = VariableStepButterworthLowPass(15.0)
    tau_reference = VariableStepButterworthLowPass(15.0)
    q_cmd_reference = VariableStepButterworthLowPass(15.0)
    q_reference.apply(zeros, 1_000_000)
    dq_reference.apply(zeros, 1_000_000)
    tau_reference.apply(zeros, 1_000_000)
    q_cmd_reference.apply(zeros, 1_000_000)
    expected_q = q_reference.apply(ones, 1_010_000)
    expected_dq = dq_reference.apply(2.0 * ones, 1_010_000)
    expected_tau = tau_reference.apply(3.0 * ones, 1_010_000)
    expected_q_cmd = q_cmd_reference.apply(4.0 * ones, 1_010_000)

    np.testing.assert_allclose(tau_other.features[1]["q"], expected_q)
    np.testing.assert_allclose(tau_next.features[1]["q"], expected_q)
    np.testing.assert_allclose(tau_other.features[1]["dq"], expected_dq)
    np.testing.assert_allclose(tau_next.features[1]["dq"], expected_dq)
    np.testing.assert_allclose(
        tau_other.features[1]["delta_q"], expected_q_cmd - expected_q
    )
    np.testing.assert_allclose(
        tau_next.features[1]["delta_q"], expected_q_cmd - expected_q
    )
    np.testing.assert_allclose(inverse_dynamics.calls[1][3], 3.0)

    inference.reset_episode()
    inference.estimate_aligned(2_000_000, 7.0 * ones, zeros, zeros, 8.0 * ones)
    np.testing.assert_allclose(tau_other.features[-1]["q"], 7.0)
    np.testing.assert_allclose(tau_next.features[-1]["q"], 7.0)


def test_source_butterworth_updates_on_skipped_frames_before_stride_sampling() -> None:
    tau_next = _Predictor("tau", np.zeros(7))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
            source_butterworth_filter=SourceButterworthFilterConfig(
                enabled=True,
                cutoff_hz=15.0,
                order=2,
            ),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {},
        tau_next_predictor=tau_next,
        source_sample_rate_hz=100.0,
    )
    zeros = np.zeros(7)
    ones = np.ones(7)

    inference.estimate_aligned(1_000_000, zeros, zeros, zeros, zeros)
    inference.estimate_aligned(
        1_010_000, ones, 2.0 * ones, 3.0 * ones, 4.0 * ones
    )
    inference.estimate_aligned(
        1_020_000, ones, 2.0 * ones, 3.0 * ones, 4.0 * ones
    )

    q_reference = VariableStepButterworthLowPass(15.0)
    q_cmd_reference = VariableStepButterworthLowPass(15.0)
    q_reference.apply(zeros, 1_000_000)
    q_cmd_reference.apply(zeros, 1_000_000)
    q_reference.apply(ones, 1_010_000)
    q_cmd_reference.apply(4.0 * ones, 1_010_000)
    expected_q = q_reference.apply(ones, 1_020_000)
    expected_q_cmd = q_cmd_reference.apply(4.0 * ones, 1_020_000)

    assert len(tau_next.features) == 2
    np.testing.assert_allclose(tau_next.features[-1]["q"], expected_q)
    np.testing.assert_allclose(
        tau_next.features[-1]["delta_q"], expected_q_cmd - expected_q
    )


def test_tau_id_is_full_rnea_and_tau_other_residual_uses_gravity_only() -> None:
    inverse_dynamics = _InverseDynamics(np.full(7, 2.0))
    inference = _inference(
        _Predictor("tau_other", np.full(7, 0.5)),
        _Predictor("tau", np.full(7, 4.0)),
        inverse_dynamics,
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)
    tau = np.full(7, 10.0)

    inference.estimate_aligned(1_000_000, zeros, zeros, tau, zeros)
    inverse_dynamics.tau_id = np.full(7, 4.0)
    result = inference.estimate_aligned(1_010_000, zeros, zeros, tau, zeros)

    np.testing.assert_allclose(result.tau_id, 4.0)
    np.testing.assert_allclose(result.tau_id_filtered, 4.0)
    np.testing.assert_allclose(result.tau_g, 2.0)
    np.testing.assert_allclose(
        result.tau_ext_cal_raw,
        2.0 + 0.5 - 10.0,
    )
    np.testing.assert_allclose(result.tau_ext_cal, result.tau_ext_cal_raw)

    inference.reset_episode()
    reset_result = inference.estimate_aligned(2_000_000, zeros, zeros, tau, zeros)
    np.testing.assert_allclose(reset_result.tau_id_filtered, 4.0)


def test_tau_next_replays_three_point_moving_average_after_shared_source_filter() -> None:
    inference = _inference(
        _Predictor("tau_other", np.zeros(7)),
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

    # The event-aligned result keeps the collected source torque raw; residuals
    # use tau_g and the checkpoint-restored tau_next target filter internally.
    np.testing.assert_allclose(spike.tau, 10.0)
    np.testing.assert_allclose(spike.tau_ext_cal_raw, -10.0)
    np.testing.assert_allclose(spike.tau_ext_pred_raw, -10.0 / 3.0)

    inference.reset_episode()
    restarted = inference.estimate_aligned(
        2_000_000,
        zeros,
        zeros,
        np.full(7, 5.0),
        zeros,
    )
    np.testing.assert_allclose(restarted.tau_ext_pred_raw, -5.0)


def test_tau_other_checkpoint_rejects_missing_gravity_contract() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inverse_dynamics = _InverseDynamics(np.zeros(7))
    state_estimator = _StateEstimator(np.zeros(7))

    tau_other.metadata = replace(tau_other.metadata, target_contract=None)
    with pytest.raises(RuntimeError, match="target_contract"):
        _inference(tau_other, tau_next, inverse_dynamics, state_estimator)


def test_tau_other_rejects_the_legacy_rnea_residual_contract() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    operations = [{"type": "lowpass", "cutoff_hz": 10.0}]
    tau_other.metadata = replace(
        tau_other.metadata,
        input_keys=("q", "dq", "delta_q", "tau"),
        input_dims={"q": 7, "dq": 7, "delta_q": 7, "tau": 7},
        target_contract="causal_rnea_residual_v1",
        dataloader_filters={
            "q": {"enabled": False, "operations": []},
            "dq": {"enabled": False, "operations": []},
            "tau": {"enabled": True, "operations": operations},
        },
        derived_target_config={
            "enabled": True,
            "method": "causal_rnea_residual_v1",
            "target_key": "tau_other",
            "source_keys": {"q": "q", "dq": "dq", "tau": "tau"},
            "state_estimator": {
                "position_std": 5.0e-4,
                "velocity_std": 3.0e-2,
                "jerk_std": 2.0,
                "initial_position_std": 1.0e-2,
                "initial_velocity_std": 2.0e-1,
                "initial_acceleration_std": 5.0,
                "max_gap_s": 0.1,
            },
            "dq_sign": [1.0] * 7,
            "rnea_state_source": "measured",
            "torque_filter_key": "tau",
            "torque_filter_operations": operations,
            "ddq_source": "variable_dt_kalman_forward_filter",
            "residual_formula": "tau_other=tau_filtered-tau_id_filtered",
        },
    )
    with pytest.raises(RuntimeError, match="v3 checkpoint inputs"):
        OnlineTauExtInference(
            TauExtInferenceConfig(
                enabled=True,
                tau_other=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
            ),
            InverseDynamicsConfig(),
            DynamicsProcessingConfig(),
            {},
            tau_other_predictor=tau_other,
            estimator=_InverseDynamics(np.zeros(7)),
            state_estimator=_StateEstimator(np.zeros(7)),
        )


def test_same_timestamp_returns_cached_dual_result_without_advancing_models() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inference = _inference(
        tau_other,
        tau_next,
        _InverseDynamics(np.zeros(7)),
        _StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)
    first = inference.estimate_aligned(100, zeros, zeros, zeros, zeros)
    second = inference.estimate_aligned(100, np.ones(7), zeros, zeros, zeros)

    assert not second.observation_updated
    assert second.observation_timestamp_us == first.observation_timestamp_us
    assert len(tau_other.features) == 1
    assert len(tau_next.features) == 1


def test_both_models_select_first_then_every_second_source_frame() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_other=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {"torque": StateParamConfig(lowpass=True, lowpass_cutoff_hz=10.0)},
        tau_other_predictor=tau_other,
        tau_next_predictor=tau_next,
        estimator=_InverseDynamics(np.zeros(7)),
        state_estimator=_StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)

    timestamps_us = [100_000, 108_000, 117_000, 125_000, 134_000, 142_000]
    results = [
        inference.estimate_aligned(
            timestamp_us,
            np.full(7, index, dtype=np.float64),
            zeros,
            zeros,
            zeros,
        )
        for index, timestamp_us in enumerate(timestamps_us)
    ]

    assert [result.observation_updated for result in results] == [
        True, False, True, False, True, False
    ]
    assert [features["q"][0] for features in tau_other.features] == [0.0, 2.0, 4.0]
    assert [features["q"][0] for features in tau_next.features] == [0.0, 2.0, 4.0]
    assert len(tau_next.features) == 3
    assert results[1].observation_timestamp_us == 100_000
    assert results[2].observation_timestamp_us == 117_000
    assert results[4].observation_timestamp_us == 134_000

    inference.reset_episode()
    restarted = inference.estimate_aligned(1_000_000, zeros, zeros, zeros, zeros)
    assert restarted.observation_updated
    assert len(tau_other.features) == 4
    assert len(tau_next.features) == 4


def test_stride_selection_uses_source_frame_count_not_timestamp_phase() -> None:
    tau_next = _Predictor("tau", np.zeros(7))
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(
            enabled=True,
            tau_next=SequenceCheckpointConfig(observation_sample_rate_hz=50.0),
        ),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {},
        tau_next_predictor=tau_next,
    )
    zeros = np.zeros(7)

    for index, timestamp_us in enumerate([100_007, 111_000, 125_000, 143_000]):
        inference.estimate_aligned(
            timestamp_us,
            np.full(7, index, dtype=np.float64),
            zeros,
            zeros,
            zeros,
        )

    assert [features["q"][0] for features in tau_next.features] == [0.0, 2.0]
    assert inference._tau_next_timestamp_us == 125_000


def test_stride_selects_current_frame_without_nearest_timestamp_lookup() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    inference = _inference(
        tau_other,
        _Predictor("tau", np.zeros(7)),
        _InverseDynamics(np.zeros(7)),
        _StateEstimator(np.zeros(7)),
        observation_sample_rate_hz=50.0,
    )
    zeros = np.zeros(7)

    inference.estimate_aligned(100_000, np.zeros(7), zeros, zeros, zeros)
    inference.estimate_aligned(110_000, np.ones(7), zeros, zeros, zeros)
    inference.estimate_aligned(130_000, np.full(7, 2.0), zeros, zeros, zeros)

    assert [features["q"][0] for features in tau_other.features] == [0.0, 2.0]


def test_stride_ignores_source_timestamp_jitter() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    inference = _inference(
        tau_other,
        _Predictor("tau", np.zeros(7)),
        _InverseDynamics(np.zeros(7)),
        _StateEstimator(np.zeros(7)),
        observation_sample_rate_hz=50.0,
    )
    zeros = np.zeros(7)
    timestamps_us = np.rint(
        1_000_000 + np.arange(118, dtype=np.float64) * 1.0e6 / 117.0
    ).astype(np.int64)

    for index, timestamp_us in enumerate(timestamps_us):
        inference.estimate_aligned(
            int(timestamp_us),
            np.full(7, index, dtype=np.float64),
            zeros,
            zeros,
            zeros,
        )

    assert len(tau_other.features) == 59


def test_observation_rate_must_divide_source_rate_exactly() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        OnlineTauExtInference(
            TauExtInferenceConfig(
                enabled=True,
                tau_next=SequenceCheckpointConfig(
                    observation_sample_rate_hz=50.0
                ),
            ),
            InverseDynamicsConfig(),
            DynamicsProcessingConfig(),
            {},
            tau_next_predictor=_Predictor("tau", np.zeros(7)),
            source_sample_rate_hz=117.0,
        )


def test_dual_inference_returns_zero_until_full_real_history_is_ready() -> None:
    tau_other = _Predictor("tau_other", np.full(7, 0.5), ready_after=50)
    tau_next = _Predictor("tau", np.full(7, 4.0), ready_after=50)
    inference = _inference(
        tau_other,
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
        np.testing.assert_allclose(result.tau_other_pred, 0.0)
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


def test_large_source_gap_warns_without_filling_or_resetting_history(caplog) -> None:
    tau_other = _Predictor("tau_other", np.full(7, 0.5), ready_after=50)
    tau_next = _Predictor("tau", np.full(7, 4.0), ready_after=50)
    inference = OnlineTauExtInference(
        TauExtInferenceConfig(enabled=True, observation_gap_warning_s=0.06),
        InverseDynamicsConfig(),
        DynamicsProcessingConfig(),
        {"torque": StateParamConfig(lowpass=True, lowpass_cutoff_hz=10.0)},
        tau_other_predictor=tau_other,
        tau_next_predictor=tau_next,
        estimator=_InverseDynamics(np.full(7, 2.0)),
        state_estimator=_StateEstimator(np.zeros(7)),
    )
    zeros = np.zeros(7)
    for index in range(49):
        inference.estimate_aligned(
            1_000_000 + index * 20_000,
            zeros,
            zeros,
            np.full(7, 10.0),
            zeros,
        )

    reset_counts = (tau_other.reset_count, tau_next.reset_count)
    with caplog.at_level("WARNING"):
        result = inference.estimate_aligned(
            2_500_000,
            zeros,
            zeros,
            np.full(7, 10.0),
            zeros,
        )

    assert not result.history_ready
    assert len(tau_other.features) == len(tau_next.features) == 25
    assert (tau_other.reset_count, tau_next.reset_count) == reset_counts
    assert "retaining existing observations" in caplog.text

    inference.estimate_aligned(
        2_510_000,
        zeros,
        zeros,
        np.full(7, 10.0),
        zeros,
    )
    assert len(tau_other.features) == len(tau_next.features) == 26


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


def test_checkpoint_filter_pipeline_matches_pinn_unified_filter_exactly() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "PINN/data_process/causal_data_filter.py"
    )
    if not source.is_file():
        pytest.skip("PINN causal filter module is unavailable")
    spec = importlib.util.spec_from_file_location("pinn_causal_filter", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    timestamp_us = np.asarray([1_000_000, 1_010_000, 1_021_000, 1_031_000])
    values = np.asarray([[0.0], [9.0], [0.0], [4.0]])
    operations = [
        {"type": "median", "window": 3},
        {"type": "moving_average", "window": 2},
        {"type": "lowpass", "cutoff_hz": 10.0},
    ]
    reference = module.filter_episode_values(
        timestamp_us * 1.0e-6,
        values,
        operations,
    )
    pipeline = CausalFilterPipeline(operations)
    actual = np.stack(
        [pipeline.apply(value, timestamp) for timestamp, value in zip(timestamp_us, values)]
    )

    np.testing.assert_allclose(actual, reference, atol=1.0e-12)
    pipeline.reset()
    np.testing.assert_allclose(pipeline.apply(np.asarray([7.0]), 2_000_000), 7.0)


def test_checkpoint_sample_rate_must_match_online_observation_rate() -> None:
    tau_other = _Predictor("tau_other", np.zeros(7))
    tau_next = _Predictor("tau", np.zeros(7))
    tau_other.metadata = replace(tau_other.metadata, sample_rate_hz=100.0)
    tau_next.metadata = replace(tau_next.metadata, sample_rate_hz=100.0)

    with pytest.raises(RuntimeError, match="sample_rate_hz=100"):
        OnlineTauExtInference(
            TauExtInferenceConfig(
                enabled=True,
                tau_other=SequenceCheckpointConfig(
                    observation_sample_rate_hz=50.0,
                ),
                tau_next=SequenceCheckpointConfig(
                    observation_sample_rate_hz=50.0,
                ),
            ),
            InverseDynamicsConfig(),
            DynamicsProcessingConfig(),
            {"torque": StateParamConfig(lowpass=True)},
            tau_other_predictor=tau_other,
            tau_next_predictor=tau_next,
            estimator=_InverseDynamics(np.zeros(7)),
            state_estimator=_StateEstimator(np.zeros(7)),
        )


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


def test_tau_ext_hampel_butterworth_rejects_spike_and_attenuates_25_hz() -> None:
    sample_rate_hz = 100.0
    timestamp_us = np.arange(500, dtype=np.int64) * 10_000
    values = 0.2 * np.sin(2.0 * np.pi * 25.0 * np.arange(500) / sample_rate_hz)
    values[250] = 5.0
    online_filter = CausalHampelButterworth(
        window=5,
        n_sigma=3.0,
        cutoff_hz=8.0,
        sample_rate_hz=sample_rate_hz,
        order=4,
    )
    actual = np.asarray(
        [
            online_filter.apply(np.asarray([value]), int(timestamp))[0]
            for timestamp, value in zip(timestamp_us, values)
        ]
    )

    assert np.max(np.abs(actual[100:])) < 0.1
    online_filter.reset()
    assert online_filter.apply(np.asarray([7.0]), 6_000_000)[0] == pytest.approx(7.0)


def test_causal_kalman_resets_acceleration_after_large_gap() -> None:
    kalman = CausalJointKalmanFilter(CausalKalmanConfig(max_gap_s=0.05))
    zeros = np.zeros(7)
    kalman.update(0, zeros, zeros)
    moving = kalman.update(10_000, np.full(7, 0.01), np.ones(7))
    assert np.max(np.abs(moving.ddq)) > 0.0
    reset = kalman.update(100_000, np.full(7, 0.02), np.ones(7))
    np.testing.assert_allclose(reset.ddq, 0.0)


def test_tau_ext_config_allows_independent_or_empty_checkpoints(tmp_path: Path) -> None:
    tau_other_only = _parse_tau_ext_inference(
        {
            "enabled": True,
            "tau_other": {
                "checkpoint_path": "tau_other.pt",
                "observation_sample_rate_hz": 100.0,
            },
        },
        tmp_path,
    )
    assert tau_other_only.tau_other.checkpoint_path is not None
    assert tau_other_only.tau_next.checkpoint_path is None
    assert tau_other_only.tau_other.observation_sample_rate_hz == pytest.approx(100.0)

    empty = _parse_tau_ext_inference({"enabled": True}, tmp_path)
    assert empty.tau_other.checkpoint_path is None
    assert empty.tau_next.checkpoint_path is None

    config = _parse_tau_ext_inference(
        {
            "enabled": True,
            "tau_other": {
                "checkpoint_path": "tau_other.pt",
                "input_keys": ["q", "dq", "delta_q", "tau"],
                "output_key": "tau_other",
            },
            "tau_next": {
                "checkpoint_path": "tau_next.pt",
                "input_keys": ["q", "dq", "delta_q"],
                "output_key": "tau",
            },
        },
        tmp_path,
    )
    assert config.tau_other.output_key == "tau_other"
    assert config.tau_other.input_keys == ("q", "dq", "delta_q", "tau")
    assert config.tau_next.output_key == "tau"
    assert config.tau_ext_filter.enabled
    assert config.tau_ext_filter.mode == "hampel_butterworth"
    assert config.tau_ext_filter.window == 5
    assert config.tau_ext_filter.cutoff_hz == pytest.approx(8.0)
    assert config.tau_ext_filter.hampel_n_sigma == pytest.approx(3.0)
    assert config.tau_ext_filter.order == 4
    assert config.tau_ext_filter.sample_rate_hz == pytest.approx(100.0)


def test_source_butterworth_config_is_explicit_and_validated(tmp_path: Path) -> None:
    config = _parse_tau_ext_inference(
        {
            "enabled": True,
            "source_butterworth_filter": {
                "enabled": True,
                "cutoff_hz": 15.0,
                "order": 2,
            },
        },
        tmp_path,
    )
    assert config.source_butterworth_filter.enabled
    assert config.source_butterworth_filter.cutoff_hz == pytest.approx(15.0)
    assert config.source_butterworth_filter.order == 2

    with pytest.raises(ValueError, match="positive and finite"):
        _parse_tau_ext_inference(
            {
                "source_butterworth_filter": {
                    "enabled": True,
                    "cutoff_hz": 0.0,
                    "order": 2,
                }
            },
            tmp_path,
        )

    with pytest.raises(ValueError, match="unknown options"):
        _parse_tau_ext_inference(
            {"source_butterworth_filter": {"sample_rate_hz": 117.0}},
            tmp_path,
        )

    with pytest.raises(ValueError, match="order must be 2"):
        _parse_tau_ext_inference(
            {"source_butterworth_filter": {"order": 4}},
            tmp_path,
        )


@pytest.mark.parametrize("value", (0, -1, float("nan"), float("inf"), True))
def test_tau_ext_config_rejects_invalid_observation_sample_rate(
    value: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="observation_sample_rate_hz"):
        _parse_tau_ext_inference(
            {"observation_sample_rate_hz": value},
            tmp_path,
        )


def test_tau_ext_config_rejects_removed_observation_stride(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="observation_sample_rate_hz instead"):
        _parse_tau_ext_inference(
            {"observation_stride_frames": 2},
            tmp_path,
        )


def test_tau_ext_filter_config_is_explicit_and_validated(tmp_path: Path) -> None:
    common = {
        "enabled": True,
        "tau_other": {"checkpoint_path": "tau_other.pt"},
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


@pytest.mark.parametrize("source", ("tau_other", "tau_free"))
def test_force_feedback_source_is_configurable(
    source: str,
    tmp_path: Path,
) -> None:
    config = _parse_tau_ext_inference(
        {"enabled": True, "feedback_source": source},
        tmp_path,
    )

    assert config.feedback_source == source


def test_force_feedback_source_rejects_unknown_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be tau_other or tau_free"):
        _parse_tau_ext_inference(
            {"enabled": True, "feedback_source": "tau_next"},
            tmp_path,
        )


def test_collection_config_rejects_removed_tau_other_inference_key(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/master_slave_can.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        source.read_text(encoding="utf-8").replace(
            "tau_ext_inference:",
            "tau_other_inference:",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tau_other_inference was removed"):
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
        "target_key": "tau_other",
        "output_dim": 7,
        "hidden_dim": 8,
        "num_layers": 1,
        "head_hidden_dim": 16,
        "head_num_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "target_contract": "causal_gravity_residual_v1",
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
        for key in ("q", "dq", "delta_q", "tau_other")
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
            "derived_target_config": {
                "enabled": True,
                "method": "causal_gravity_residual_v1",
                "target_key": "tau_other",
                "source_keys": {"q": "q", "dq": "dq", "tau": "tau"},
                "dq_sign": [1.0] * 7,
                "torque_filter_key": "tau",
                "torque_filter_operations": [],
                "ddq_source": "unused",
                "measured_tau_source": "observation.torque",
                "residual_formula": "tau_other=tau_measured-tau_g",
            },
        },
        path,
    )
    predictor = SequenceTorquePredictor(
        SequenceCheckpointConfig(checkpoint_path=path),
        name="tau_other",
    )
    features = {
        key: np.zeros(7) for key in ("q", "dq", "delta_q")
    }
    for _ in range(3):
        assert predictor.append_and_predict(features) is None
    prediction = predictor.append_and_predict(features)
    assert predictor.metadata.architecture == architecture
    assert predictor.metadata.input_keys == ("q", "dq", "delta_q")
    assert not predictor.metadata.target_filter_enabled
    assert not predictor.metadata.target_filter_apply_additional_lowpass
    assert prediction is not None
    assert prediction.shape == (7,)
    predictor.reset()
    assert predictor.append_and_predict(features) is None
