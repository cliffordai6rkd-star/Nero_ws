from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from nero_collection.causal_kalman import CausalJointKalmanFilter
from nero_collection.config import (
    CausalKalmanConfig,
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    SequenceCheckpointConfig,
    StateParamConfig,
    TauExtFilterConfig,
    TauExtInferenceConfig,
)
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator
from nero_collection.filters import (
    CausalFilterPipeline,
    CausalHampelButterworth,
    CausalTrailingMovingAverage,
    CausalTrailingMedian,
    CausalWindowLowPass,
    OnePoleLowPass,
    VariableStepButterworthLowPass,
)


log = logging.getLogger(__name__)

_DEFAULT_CPU_TORCH_NUM_THREADS = 1
_MODEL_INPUT_KEYS = frozenset({"q", "dq", "delta_q", "tau"})
_LEGACY_TAU_F_TARGET_CONTRACT = "matched_causal_torque_filter_v1"
_DERIVED_TAU_F_TARGET_CONTRACT = "causal_rnea_residual_v1"
_TAU_F_TARGET_CONTRACTS = frozenset(
    {_LEGACY_TAU_F_TARGET_CONTRACT, _DERIVED_TAU_F_TARGET_CONTRACT}
)


@dataclass(frozen=True)
class SequenceCheckpointMetadata:
    checkpoint_path: Path
    horizon: int
    input_keys: tuple[str, ...]
    input_dims: dict[str, int]
    output_key: str
    output_dim: int
    architecture: str
    normalize_mode: str
    sample_rate_hz: float | None = None
    dataloader_filters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    derived_target_config: Mapping[str, Any] = field(default_factory=dict)
    inference_mode: str = "fixed_window"
    target_contract: str | None = None
    target_filter_enabled: bool = False
    target_filter_cutoff_hz: float | None = None
    target_filter_moving_average_window: int | None = None
    target_filter_median_window: int | None = None
    target_filter_apply_additional_lowpass: bool = False


@dataclass(frozen=True)
class TauExtInferenceMetadata:
    tau_f: SequenceCheckpointMetadata | None
    tau_next: SequenceCheckpointMetadata | None
    input_keys: tuple[str, ...]
    tau_ext_filter: TauExtFilterConfig = field(default_factory=TauExtFilterConfig)


@dataclass(frozen=True)
class OnlineTauExtResult:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq_kf_causal: np.ndarray
    tau: np.ndarray
    tau_id: np.ndarray
    tau_id_filtered: np.ndarray
    tau_f_pred: np.ndarray
    tau_next_pred: np.ndarray
    tau_ext_cal: np.ndarray
    tau_ext_pred: np.ndarray
    history_ready: bool = True
    observation_updated: bool = True
    observation_timestamp_us: int = 0
    prediction_age_us: int = 0
    tau_ext_cal_raw: np.ndarray | None = None
    tau_ext_pred_raw: np.ndarray | None = None
    tau_f_history_ready: bool | None = None
    tau_free_history_ready: bool | None = None

    def force_feedback(self, source: str) -> tuple[np.ndarray, bool]:
        """Return the configured residual and its branch-local readiness."""
        normalized = str(source).strip().lower()
        if normalized == "tau_f":
            ready = (
                self.history_ready
                if self.tau_f_history_ready is None
                else self.tau_f_history_ready
            )
            return np.asarray(self.tau_ext_cal, dtype=np.float64).copy(), bool(ready)
        if normalized == "tau_free":
            ready = (
                self.history_ready
                if self.tau_free_history_ready is None
                else self.tau_free_history_ready
            )
            return np.asarray(self.tau_ext_pred, dtype=np.float64).copy(), bool(ready)
        raise ValueError(f"force-feedback source must be tau_f or tau_free, got {source!r}")


@dataclass(frozen=True)
class _SourceObservation:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    q_cmd: np.ndarray


@dataclass
class _NearestObservationGrid:
    sample_rate_hz: float
    origin_timestamp_us: int | None = None
    next_index: int = 0
    previous_source: _SourceObservation | None = None

    def reset(self) -> None:
        self.origin_timestamp_us = None
        self.next_index = 0
        self.previous_source = None

    def advance(
        self,
        source: _SourceObservation,
    ) -> list[tuple[int, _SourceObservation]]:
        previous = self.previous_source
        if previous is None:
            self.origin_timestamp_us = source.timestamp_us
            self.next_index = 1
            self.previous_source = source
            return [(source.timestamp_us, source)]

        selected: list[tuple[int, _SourceObservation]] = []
        target_timestamp_us = self.timestamp_us(self.next_index)
        while source.timestamp_us >= target_timestamp_us:
            left_gap_us = target_timestamp_us - previous.timestamp_us
            right_gap_us = source.timestamp_us - target_timestamp_us
            nearest = previous if left_gap_us <= right_gap_us else source
            selected.append((target_timestamp_us, nearest))
            self.next_index += 1
            target_timestamp_us = self.timestamp_us(self.next_index)

        self.previous_source = source
        return selected

    def timestamp_us(self, index: int) -> int:
        assert self.origin_timestamp_us is not None
        offset_us = round(int(index) * 1.0e6 / self.sample_rate_hz)
        return self.origin_timestamp_us + int(offset_us)


@dataclass
class _FixedPhaseCausalObservationGrid:
    """Sample latest complete observations on an absolute, history-only grid."""

    sample_rate_hz: float
    next_timestamp_us: int | None = None
    previous_source: _SourceObservation | None = None

    def __post_init__(self) -> None:
        period_us = 1.0e6 / float(self.sample_rate_hz)
        rounded_period_us = round(period_us)
        if (
            not np.isfinite(period_us)
            or rounded_period_us <= 0
            or not np.isclose(period_us, rounded_period_us, rtol=0.0, atol=1.0e-9)
        ):
            raise ValueError(
                "fixed-phase causal observation rate must divide 1 MHz exactly"
            )
        self._period_us = int(rounded_period_us)

    def reset(self) -> None:
        self.next_timestamp_us = None
        self.previous_source = None

    def advance(
        self,
        source: _SourceObservation,
    ) -> list[tuple[int, _SourceObservation]]:
        previous = self.previous_source
        if previous is None:
            self.next_timestamp_us = self._ceil_to_grid(source.timestamp_us)

        selected: list[tuple[int, _SourceObservation]] = []
        assert self.next_timestamp_us is not None
        while source.timestamp_us >= self.next_timestamp_us:
            if source.timestamp_us == self.next_timestamp_us:
                observation = source
            elif previous is not None:
                observation = previous
            else:
                break
            selected.append((self.next_timestamp_us, observation))
            self.next_timestamp_us += self._period_us

        self.previous_source = source
        return selected

    def _ceil_to_grid(self, timestamp_us: int) -> int:
        return -(-int(timestamp_us) // self._period_us) * self._period_us


class SequenceTorquePredictorProtocol(Protocol):
    metadata: SequenceCheckpointMetadata

    def warm_up(self) -> None:
        ...

    def reset(self) -> None:
        ...

    def append_and_predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray | None:
        ...

    def predict_sequence(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        ...


class SequenceTorquePredictor:
    """Restore a PINN NEXT-style checkpoint and replay its training window."""

    def __init__(self, config: SequenceCheckpointConfig, *, name: str) -> None:
        if config.checkpoint_path is None:
            raise ValueError(f"{name} checkpoint path is required")
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "tau_ext inference requires PyTorch; install PyTorch first"
            ) from exc

        try:
            device = torch.device(config.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"invalid {name} inference device {config.device!r}"
            ) from exc
        if device.type == "cpu":
            torch.set_num_threads(_DEFAULT_CPU_TORCH_NUM_THREADS)

        checkpoint_path = _resolve_checkpoint_path(config.checkpoint_path, name)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"{name} checkpoint root must be a mapping")

        checkpoint_config = _mapping(checkpoint.get("config"), "checkpoint.config")
        dataloader_config = _mapping(
            checkpoint_config.get("dataloader"),
            "checkpoint.config.dataloader",
        )
        dataloader_filters = _normalize_checkpoint_filters(
            checkpoint.get(
                "dataloader_filters",
                dataloader_config.get("filters", {}),
            )
        )
        derived_target_value = checkpoint.get("derived_target_config", {})
        if derived_target_value is None:
            derived_target_value = {}
        if not isinstance(derived_target_value, Mapping):
            raise RuntimeError("checkpoint.derived_target_config must be a mapping")
        derived_target_config = dict(derived_target_value)
        sample_rate_value = checkpoint.get(
            "sample_rate_hz",
            dataloader_config.get(
                "filter_sample_rate_hz",
                dataloader_config.get("expected_fps"),
            ),
        )
        sample_rate_hz = (
            None if sample_rate_value is None else float(sample_rate_value)
        )
        if sample_rate_hz is not None and (
            not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0
        ):
            raise RuntimeError("checkpoint sample_rate_hz must be positive and finite")
        model_config = _mapping(
            checkpoint_config.get("model"),
            "checkpoint.config.model",
        )
        horizon = int(dataloader_config.get("horizon", 0))
        input_keys = tuple(str(key) for key in model_config.get("inputs", ()))
        input_dims_config = _mapping(
            model_config.get("input_dims", {}),
            "checkpoint.config.model.input_dims",
        )
        input_dims = {key: int(input_dims_config.get(key, 7)) for key in input_keys}
        output_key = str(model_config.get("target_key", ""))
        output_dim = int(model_config.get("output_dim", 7))
        architecture = str(model_config.get("architecture", "lstm")).lower()
        target_contract_value = model_config.get("target_contract")
        target_contract = (
            None
            if target_contract_value is None
            else str(target_contract_value).strip()
        )
        target_filter_value = model_config.get("target_filter")
        target_filter_config = target_filter_value or {}
        if not isinstance(target_filter_config, Mapping):
            raise RuntimeError("checkpoint.config.model.target_filter must be a mapping")
        target_filter_enabled = target_filter_value is not None and bool(
            target_filter_config.get("enabled", True)
        )
        target_filter_cutoff_hz = target_filter_config.get("cutoff_hz")
        target_filter_moving_average_window = target_filter_config.get(
            "moving_average_window"
        )
        target_filter_median_window = target_filter_config.get("median_window")
        if (
            target_filter_moving_average_window is not None
            and target_filter_median_window is not None
        ):
            raise RuntimeError(
                "checkpoint target_filter must not define both "
                "moving_average_window and median_window"
            )
        target_filter_apply_additional_lowpass = bool(
            target_filter_config.get("apply_additional_lowpass", False)
        )
        if target_filter_cutoff_hz is not None:
            target_filter_cutoff_hz = float(target_filter_cutoff_hz)
        if target_filter_moving_average_window is not None:
            target_filter_moving_average_window = int(
                target_filter_moving_average_window
            )
        if target_filter_median_window is not None:
            target_filter_median_window = int(target_filter_median_window)

        if horizon <= 0 or not input_keys:
            raise RuntimeError(
                f"{name} checkpoint must define a positive horizon and model inputs"
            )
        unknown_inputs = sorted(set(input_keys) - _MODEL_INPUT_KEYS)
        if unknown_inputs:
            raise RuntimeError(
                f"{name} checkpoint requests inputs forbidden by the online contract: "
                f"{unknown_inputs}; supported inputs are "
                f"{sorted(_MODEL_INPUT_KEYS)}"
            )
        if len(set(input_keys)) != len(input_keys):
            raise RuntimeError(f"{name} checkpoint model inputs contain duplicates")
        if any(dim <= 0 for dim in input_dims.values()) or output_dim != 7:
            raise RuntimeError(f"{name} checkpoint must use positive input dims and output_dim=7")
        if config.horizon is not None and config.horizon != horizon:
            raise RuntimeError(
                f"configured {name} horizon={config.horizon} does not match "
                f"checkpoint horizon={horizon}"
            )
        if config.input_keys is not None and config.input_keys != input_keys:
            raise RuntimeError(
                f"configured {name} input_keys={config.input_keys} do not match "
                f"checkpoint inputs={input_keys}"
            )
        if config.output_key is not None and config.output_key != output_key:
            raise RuntimeError(
                f"configured {name} output_key={config.output_key!r} does not match "
                f"checkpoint target={output_key!r}"
            )

        normalizer = _mapping(checkpoint.get("normalizer"), "checkpoint.normalizer")
        self._stats = _mapping(normalizer.get("stats"), "checkpoint.normalizer.stats")
        self._eps = float(normalizer.get("eps", 1.0e-6))
        self._normalize_mode = str(
            normalizer.get(
                "normalize_mode",
                dataloader_config.get("normalize_mode", "gaussian"),
            )
        ).lower()
        self._normalize_keys = tuple(
            str(key)
            for key in normalizer.get(
                "normalize_lowdim_keys",
                dataloader_config.get("normalize_lowdim_keys", ()),
            )
        )
        if self._normalize_mode not in {"gaussian", "limit", "quantile"}:
            raise RuntimeError(
                f"unsupported {name} normalization mode: {self._normalize_mode!r}"
            )
        missing_stats = [
            key
            for key in (*input_keys, output_key)
            if key in self._normalize_keys and key not in self._stats
        ]
        if missing_stats:
            raise RuntimeError(
                f"{name} checkpoint is missing normalizer stats: {missing_stats}"
            )

        self._name = name
        self._torch = torch
        self._device = device
        self._model = _build_checkpoint_model(
            torch,
            nn,
            model_config,
            input_keys,
            input_dims,
        )
        model_state = checkpoint.get("model")
        if not isinstance(model_state, Mapping):
            raise RuntimeError(f"{name} checkpoint.model must contain a state dictionary")
        self._model.load_state_dict(model_state, strict=True)
        try:
            self._model.to(self._device)
        except Exception as exc:
            raise RuntimeError(
                f"cannot place {name} model on device {config.device!r}: {exc}"
            ) from exc
        self._model.eval()
        self.metadata = SequenceCheckpointMetadata(
            checkpoint_path=checkpoint_path,
            horizon=horizon,
            input_keys=input_keys,
            input_dims=input_dims,
            output_key=output_key,
            output_dim=output_dim,
            architecture=architecture,
            normalize_mode=self._normalize_mode,
            sample_rate_hz=sample_rate_hz,
            dataloader_filters=dataloader_filters,
            derived_target_config=derived_target_config,
            target_contract=target_contract,
            target_filter_enabled=target_filter_enabled,
            target_filter_cutoff_hz=target_filter_cutoff_hz,
            target_filter_moving_average_window=(
                target_filter_moving_average_window
            ),
            target_filter_median_window=target_filter_median_window,
            target_filter_apply_additional_lowpass=(
                target_filter_apply_additional_lowpass
            ),
        )
        self._history: dict[str, deque[Any]] = {}
        self.reset()
        log.info(
            "%s checkpoint ready path=%s architecture=%s horizon=%d inputs=%s "
            "output=%s device=%s",
            name,
            checkpoint_path,
            architecture,
            horizon,
            ",".join(input_keys),
            output_key,
            config.device,
        )

    def append_and_predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray | None:
        normalized_step = {}
        for key, dim in self.metadata.input_dims.items():
            value = _finite_vector(key, features[key], dim)
            tensor = self._torch.as_tensor(
                value,
                dtype=self._torch.float32,
                device=self._device,
            )
            normalized_step[key] = self._normalize(key, tensor)

        parts = []
        for key in self.metadata.input_keys:
            value = normalized_step[key]
            history = self._history[key]
            history.append(value)
            if len(history) < self.metadata.horizon:
                continue
            parts.append(self._torch.stack(tuple(history), dim=0))
        if len(parts) != len(self.metadata.input_keys):
            return None
        sequence = self._torch.cat(parts, dim=-1).unsqueeze(0)
        with self._torch.inference_mode():
            prediction = self._model(sequence)
            prediction = self._denormalize(self.metadata.output_key, prediction)
        result = prediction.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return _finite_vector(f"{self._name}_pred", result, 7)

    def predict_sequence(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        tensors: dict[str, Any] = {}
        length: int | None = None
        for key in self.metadata.input_keys:
            if key not in features:
                raise KeyError(f"missing {self._name} sequence input {key!r}")
            tensor = self._torch.as_tensor(
                features[key],
                dtype=self._torch.float32,
                device=self._device,
            )
            expected_dim = self.metadata.input_dims[key]
            if tensor.ndim != 2 or tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"{self._name} sequence input {key!r} must have shape "
                    f"[T, {expected_dim}], got {tuple(tensor.shape)}"
                )
            if length is None:
                length = int(tensor.shape[0])
            elif tensor.shape[0] != length:
                raise ValueError(f"all {self._name} sequence inputs must share length")
            tensors[key] = self._normalize(key, tensor)
        if not length:
            return np.empty((0, 7), dtype=np.float64)

        predictions = []
        with self._torch.inference_mode():
            for index in range(length):
                start = max(0, index - self.metadata.horizon + 1)
                padding = self.metadata.horizon - (index - start + 1)
                parts = []
                for key in self.metadata.input_keys:
                    values = tensors[key][start : index + 1]
                    if padding:
                        values = self._torch.cat(
                            (values.new_zeros((padding, values.shape[-1])), values),
                            dim=0,
                        )
                    parts.append(values)
                predictions.append(
                    self._model(self._torch.cat(parts, dim=-1).unsqueeze(0))
                )
            prediction = self._torch.cat(predictions, dim=0)
            prediction = self._denormalize(self.metadata.output_key, prediction)
        result = prediction.detach().cpu().numpy().astype(np.float64)
        if result.shape != (length, 7) or not np.isfinite(result).all():
            raise RuntimeError(
                f"{self._name} sequence prediction has invalid shape or values: "
                f"{result.shape}"
            )
        return result.copy()

    def warm_up(self) -> None:
        sequence = self._torch.zeros(
            (1, self.metadata.horizon, sum(self.metadata.input_dims.values())),
            dtype=self._torch.float32,
            device=self._device,
        )
        with self._torch.inference_mode():
            self._model(sequence)
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
        self.reset()

    def reset(self) -> None:
        self._history = {
            key: deque(maxlen=self.metadata.horizon)
            for key in self.metadata.input_keys
        }

    def _normalize(self, key: str, value: Any) -> Any:
        if key not in self._normalize_keys:
            return value
        stats = _mapping(self._stats[key], "normalizer statistics")
        if self._normalize_mode == "gaussian":
            return (value - self._stat(stats, "mean")) / (
                self._stat(stats, "std") + self._eps
            )
        if self._normalize_mode == "limit":
            minimum = self._stat(stats, "min")
            maximum = self._stat(stats, "max")
            return 2.0 * (value - minimum) / (maximum - minimum + self._eps) - 1.0
        q01 = self._stat(stats, "q01")
        q99 = self._stat(stats, "q99")
        return self._torch.clamp(
            2.0 * (value - q01) / (q99 - q01 + self._eps) - 1.0,
            -1.0,
            1.0,
        )

    def _denormalize(self, key: str, value: Any) -> Any:
        if key not in self._normalize_keys:
            return value
        stats = _mapping(self._stats[key], "normalizer statistics")
        if self._normalize_mode == "gaussian":
            return value * (self._stat(stats, "std") + self._eps) + self._stat(
                stats, "mean"
            )
        if self._normalize_mode == "limit":
            minimum = self._stat(stats, "min")
            maximum = self._stat(stats, "max")
            return (value + 1.0) * (maximum - minimum + self._eps) / 2.0 + minimum
        q01 = self._stat(stats, "q01")
        q99 = self._stat(stats, "q99")
        return (value + 1.0) * (q99 - q01 + self._eps) / 2.0 + q01

    def _stat(self, stats: Mapping[str, Any], name: str) -> Any:
        if name not in stats:
            raise RuntimeError(
                f"{self._name} checkpoint normalizer is missing statistic {name!r}"
            )
        return self._torch.as_tensor(
            stats[name],
            dtype=self._torch.float32,
            device=self._device,
        )


class OnlineTauExtInference:
    """Run torque models on their configured fixed-rate observation grids."""

    def __init__(
        self,
        config: TauExtInferenceConfig,
        inverse_dynamics: InverseDynamicsConfig,
        dynamics_processing: DynamicsProcessingConfig,
        robot_states: dict[str, StateParamConfig],
        *,
        tau_f_predictor: SequenceTorquePredictorProtocol | None = None,
        tau_next_predictor: SequenceTorquePredictorProtocol | None = None,
        estimator: Any | None = None,
        state_estimator: CausalJointKalmanFilter | None = None,
    ) -> None:
        if not config.enabled and tau_f_predictor is None and tau_next_predictor is None:
            raise ValueError("tau_ext inference is disabled")
        self.config = config
        self.tau_f_predictor = tau_f_predictor
        if self.tau_f_predictor is None and config.tau_f.checkpoint_path is not None:
            self.tau_f_predictor = SequenceTorquePredictor(config.tau_f, name="tau_f")
        self.tau_next_predictor = tau_next_predictor
        if (
            self.tau_next_predictor is None
            and config.tau_next.checkpoint_path is not None
        ):
            self.tau_next_predictor = SequenceTorquePredictor(
                config.tau_next,
                name="tau_next",
            )
        if self.tau_f_predictor is None and self.tau_next_predictor is None:
            raise ValueError("tau_ext inference has no configured checkpoint")

        if self.tau_f_predictor is not None:
            _validate_predictor_contract(
                self.tau_f_predictor.metadata,
                model_name="tau_f",
                output_key="tau_f",
            )
        if self.tau_next_predictor is not None:
            _validate_predictor_contract(
                self.tau_next_predictor.metadata,
                model_name="tau_next",
                output_key="tau",
            )

        self._tau_f_sample_rate_hz = _resolve_observation_sample_rate(
            config.tau_f,
            None if self.tau_f_predictor is None else self.tau_f_predictor.metadata,
            model_name="tau_f",
        )
        self._tau_next_sample_rate_hz = _resolve_observation_sample_rate(
            config.tau_next,
            None
            if self.tau_next_predictor is None
            else self.tau_next_predictor.metadata,
            model_name="tau_next",
        )
        self._tau_f_grid = (
            None
            if self._tau_f_sample_rate_hz is None
            else _NearestObservationGrid(self._tau_f_sample_rate_hz)
        )
        self._tau_next_grid = (
            None
            if self._tau_next_sample_rate_hz is None
            else _FixedPhaseCausalObservationGrid(self._tau_next_sample_rate_hz)
        )
        source_filter = config.source_butterworth_filter
        self.source_butterworth_filters = (
            {
                key: VariableStepButterworthLowPass(
                    cutoff_hz=source_filter.cutoff_hz,
                )
                for key in (
                    # "q",
                    "dq",
                    "tau",
                    # "q_cmd"
                )
            }
            if source_filter.enabled
            else {}
        )

        self.estimator = None
        self.state_estimator = None
        self.tau_filter = None
        self.tau_id_filter = None
        self.tau_f_input_filters: dict[str, CausalFilterPipeline] = {}
        self.tau_f_state_filters: dict[str, CausalFilterPipeline] = {}
        self._tau_f_dq_sign = np.ones(7, dtype=np.float64)
        self._tau_f_rnea_state_source = "measured"
        if self.tau_f_predictor is not None:
            self.estimator = estimator or PinocchioJointTorqueResidualEstimator(
                inverse_dynamics
            )
            target_config = self.tau_f_predictor.metadata.derived_target_config
            checkpoint_kalman_config = _derived_kalman_config(
                self.tau_f_predictor.metadata
            )
            if state_estimator is not None:
                self.state_estimator = state_estimator
            else:
                if (
                    checkpoint_kalman_config is not None
                    and checkpoint_kalman_config != config.state_estimator
                ):
                    raise RuntimeError(
                        "tau_ext_inference.state_estimator does not match the "
                        "tau_f checkpoint derived_target_config.state_estimator"
                    )
                self.state_estimator = CausalJointKalmanFilter(
                    checkpoint_kalman_config or config.state_estimator
                )
            self._tau_f_dq_sign = _derived_dq_sign(target_config)
            self._tau_f_rnea_state_source = str(
                target_config.get("rnea_state_source", "measured")
            ).strip().lower()
            self.tau_f_input_filters = _build_feature_filter_bank(
                self.tau_f_predictor.metadata,
                exclude_keys={"q", "dq", "tau"},
            )
            self.tau_f_state_filters = _build_feature_filter_bank(
                self.tau_f_predictor.metadata,
                include_keys={"q", "dq"},
            )
            self.tau_filter = _build_tau_f_source_filter(
                self.tau_f_predictor.metadata,
                dynamics_processing,
                robot_states,
            )
            self.tau_id_filter = _build_tau_f_source_filter(
                self.tau_f_predictor.metadata,
                dynamics_processing,
                robot_states,
            )

        self.tau_next_input_filters = (
            {}
            if self.tau_next_predictor is None
            else _build_feature_filter_bank(self.tau_next_predictor.metadata)
        )
        self.tau_next_target_filter = (
            None
            if self.tau_next_predictor is None
            else _build_tau_target_filter(self.tau_next_predictor.metadata)
        )
        self.tau_next_target_uses_raw = bool(
            self.tau_next_predictor is not None
            and self.tau_next_predictor.metadata.dataloader_filters
        )
        self.tau_next_source_filter = (
            _build_tau_f_source_filter(
                self.tau_f_predictor.metadata,
                dynamics_processing,
                robot_states,
            )
            if self.tau_next_predictor is not None
            and self.tau_f_predictor is not None
            and not self.tau_next_target_uses_raw
            else None
        )
        tau_ext_filter = config.tau_ext_filter
        self.tau_ext_cal_filter = (
            _build_tau_ext_post_filter(tau_ext_filter)
            if self.tau_f_predictor is not None
            else None
        )
        self.tau_ext_pred_filter = (
            _build_tau_ext_post_filter(tau_ext_filter)
            if self.tau_next_predictor is not None
            else None
        )
        self._last_input_timestamp_us: int | None = None
        self._last_source_timestamp_us: int | None = None
        self._tau_f_timestamp_us: int | None = None
        self._tau_next_timestamp_us: int | None = None
        self._tau_f_ready = False
        self._tau_next_ready = False
        self._tau_f_ready_logged = False
        self._tau_next_ready_logged = False
        self._ddq = np.zeros(7, dtype=np.float64)
        self._tau_id = np.zeros(7, dtype=np.float64)
        self._tau_id_filtered = np.zeros(7, dtype=np.float64)
        self._tau_f_pred = np.zeros(7, dtype=np.float64)
        self._tau_next_pred = np.zeros(7, dtype=np.float64)
        self._tau_ext_cal_raw = np.zeros(7, dtype=np.float64)
        self._tau_ext_pred_raw = np.zeros(7, dtype=np.float64)
        self._tau_ext_cal = np.zeros(7, dtype=np.float64)
        self._tau_ext_pred = np.zeros(7, dtype=np.float64)

    @property
    def metadata(self) -> TauExtInferenceMetadata:
        tau_f_metadata = (
            None if self.tau_f_predictor is None else self.tau_f_predictor.metadata
        )
        tau_next_metadata = (
            None
            if self.tau_next_predictor is None
            else self.tau_next_predictor.metadata
        )
        return TauExtInferenceMetadata(
            tau_f=tau_f_metadata,
            tau_next=tau_next_metadata,
            input_keys=tuple(
                dict.fromkeys(
                    (tau_f_metadata.input_keys if tau_f_metadata is not None else ())
                    + (
                        tau_next_metadata.input_keys
                        if tau_next_metadata is not None
                        else ()
                    )
                )
            ),
            tau_ext_filter=self.config.tau_ext_filter,
        )

    def observation_sample_rate_hz(self, model_name: str) -> float | None:
        if model_name == "tau_f":
            return self._tau_f_sample_rate_hz
        if model_name == "tau_next":
            return self._tau_next_sample_rate_hz
        raise ValueError(f"unknown torque model {model_name!r}")

    def warm_up(self) -> None:
        if self.tau_f_predictor is not None:
            self.tau_f_predictor.warm_up()
        if self.tau_next_predictor is not None:
            self.tau_next_predictor.warm_up()

    def reset_episode(self) -> None:
        if self.tau_f_predictor is not None:
            self.tau_f_predictor.reset()
        if self.tau_next_predictor is not None:
            self.tau_next_predictor.reset()
        if self.state_estimator is not None:
            self.state_estimator.reset()
        _reset_filter_bank(self.tau_f_input_filters)
        _reset_filter_bank(self.tau_f_state_filters)
        _reset_filter_bank(self.tau_next_input_filters)
        for source_filter in self.source_butterworth_filters.values():
            source_filter.reset()
        if self.tau_filter is not None:
            self.tau_filter.reset()
        if self.tau_id_filter is not None:
            self.tau_id_filter.reset()
        if self.tau_next_target_filter is not None:
            self.tau_next_target_filter.reset()
        if self.tau_next_source_filter is not None:
            self.tau_next_source_filter.reset()
        if self.tau_ext_cal_filter is not None:
            self.tau_ext_cal_filter.reset()
        if self.tau_ext_pred_filter is not None:
            self.tau_ext_pred_filter.reset()
        self._last_input_timestamp_us = None
        self._last_source_timestamp_us = None
        if self._tau_f_grid is not None:
            self._tau_f_grid.reset()
        if self._tau_next_grid is not None:
            self._tau_next_grid.reset()
        self._tau_f_timestamp_us = None
        self._tau_next_timestamp_us = None
        self._tau_f_ready = False
        self._tau_next_ready = False
        self._tau_f_ready_logged = False
        self._tau_next_ready_logged = False
        for value in (
            self._ddq,
            self._tau_id,
            self._tau_id_filtered,
            self._tau_f_pred,
            self._tau_next_pred,
            self._tau_ext_cal_raw,
            self._tau_ext_pred_raw,
            self._tau_ext_cal,
            self._tau_ext_pred,
        ):
            value.fill(0.0)

    def estimate_aligned(
        self,
        timestamp_us: int,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        q_cmd: np.ndarray,
    ) -> OnlineTauExtResult:
        timestamp_us = int(timestamp_us)
        q_value = _finite_vector("q", q, 7)
        dq_value = _finite_vector("dq", dq, 7)
        raw_tau_value = _finite_vector("tau", tau, 7)
        q_cmd_value = _finite_vector("q_cmd", q_cmd, 7)
        if self._last_input_timestamp_us is not None:
            if timestamp_us < self._last_input_timestamp_us:
                raise ValueError(
                    "tau_ext source timestamp moved backwards: "
                    f"{timestamp_us} < {self._last_input_timestamp_us}"
                )
            if timestamp_us == self._last_input_timestamp_us:
                return self._result_for_source(
                    timestamp_us, q_value, dq_value, raw_tau_value, False
                )
        self._last_input_timestamp_us = timestamp_us

        filtered_source = {
            key: (
                self.source_butterworth_filters[key].apply(value, timestamp_us)
                if key in self.source_butterworth_filters
                else value.copy()
            )
            for key, value in (
                ("q", q_value),
                ("dq", dq_value),
                ("tau", raw_tau_value),
            )
        }
        source = _SourceObservation(
            timestamp_us=timestamp_us,
            q=filtered_source["q"],
            dq=filtered_source["dq"],
            tau=filtered_source["tau"],
            q_cmd=q_cmd_value.copy(),
        )
        if self._last_source_timestamp_us is not None:
            source_gap_us = timestamp_us - self._last_source_timestamp_us
            warning_us = int(round(self.config.observation_gap_warning_s * 1e6))
            if source_gap_us > warning_us:
                log.warning(
                    "tau_ext source observation gap exceeded warning threshold: "
                    "gap=%.3fms threshold=%.3fms; retaining the previous 49 "
                    "observations in each model history",
                    source_gap_us * 1.0e-3,
                    warning_us * 1.0e-3,
                )
        self._last_source_timestamp_us = timestamp_us

        tau_f_updated = False
        if self._tau_f_grid is not None:
            selected = self._tau_f_grid.advance(source)
            tau_f_updated = bool(selected)
            for model_timestamp_us, observation in selected:
                self._estimate_tau_f(model_timestamp_us, observation)

        tau_next_updated = False
        if self._tau_next_grid is not None:
            selected = self._tau_next_grid.advance(source)
            tau_next_updated = bool(selected)
            for model_timestamp_us, observation in selected:
                self._estimate_tau_next(model_timestamp_us, observation)

        updated = (
            tau_next_updated
            if self.tau_next_predictor is not None
            else tau_f_updated
        )
        return self._result_for_source(
            timestamp_us, q_value, dq_value, raw_tau_value, updated
        )

    def _estimate_tau_f(
        self,
        timestamp_us: int,
        source: _SourceObservation,
    ) -> None:
        assert self.tau_f_predictor is not None
        assert self.state_estimator is not None
        assert self.estimator is not None
        assert self.tau_filter is not None
        assert self.tau_id_filter is not None
        tau_value = self.tau_filter.apply(source.tau, timestamp_us)
        state_features = _apply_feature_filter_bank(
            {
                "q": source.q,
                "dq": source.dq * self._tau_f_dq_sign,
            },
            self.tau_f_state_filters,
            timestamp_us,
        )
        q_value = state_features["q"]
        dq_value = state_features["dq"]
        kalman_state = self.state_estimator.update(timestamp_us, q_value, dq_value)
        if self._tau_f_rnea_state_source == "filtered":
            q_rnea = kalman_state.q
            dq_rnea = kalman_state.dq
        else:
            q_rnea = q_value
            dq_rnea = dq_value
        inverse_dynamics = self.estimator.estimate(
            q_rnea,
            dq_rnea,
            kalman_state.ddq,
            tau_value,
        )
        tau_id = _finite_vector("tau_id", inverse_dynamics.tau_id, 7)
        tau_id_filtered = self.tau_id_filter.apply(tau_id, timestamp_us)
        raw_features = _observation_features(
            source,
            q=q_value,
            dq=dq_value,
            tau=tau_value,
        )
        _require_features(raw_features, self.tau_f_predictor.metadata.input_keys, "tau_f")
        features = _apply_feature_filter_bank(
            raw_features,
            self.tau_f_input_filters,
            timestamp_us,
        )
        prediction = self.tau_f_predictor.append_and_predict(features)
        self._ddq = kalman_state.ddq.copy()
        self._tau_id = tau_id.copy()
        self._tau_id_filtered = tau_id_filtered.copy()
        self._tau_f_timestamp_us = timestamp_us
        self._tau_f_ready = prediction is not None
        if prediction is None:
            self._tau_f_pred.fill(0.0)
            self._tau_ext_cal_raw.fill(0.0)
            self._tau_ext_cal.fill(0.0)
            return
        self._tau_f_pred = _finite_vector("tau_f_pred", prediction, 7)
        self._tau_ext_cal_raw = tau_id_filtered + self._tau_f_pred - tau_value
        self._tau_ext_cal = (
            self.tau_ext_cal_filter.apply(self._tau_ext_cal_raw, timestamp_us)
            if self.tau_ext_cal_filter is not None
            else self._tau_ext_cal_raw.copy()
        )
        if not self._tau_f_ready_logged:
            log.info(
                "tau_f inference enabled after full history=%d samples",
                self.tau_f_predictor.metadata.horizon,
            )
            self._tau_f_ready_logged = True

    def _estimate_tau_next(
        self,
        timestamp_us: int,
        source: _SourceObservation,
    ) -> None:
        assert self.tau_next_predictor is not None
        tau_next_source = (
            self.tau_next_source_filter.apply(source.tau, timestamp_us)
            if self.tau_next_source_filter is not None
            else source.tau
        )
        tau_next_target = (
            self.tau_next_target_filter.apply(tau_next_source, timestamp_us)
            if self.tau_next_target_filter is not None
            else tau_next_source
        )
        raw_features = _observation_features(source)
        _require_features(
            raw_features, self.tau_next_predictor.metadata.input_keys, "tau_next"
        )
        features = _apply_feature_filter_bank(
            raw_features,
            self.tau_next_input_filters,
            timestamp_us,
        )
        prediction = self.tau_next_predictor.append_and_predict(features)
        self._tau_next_timestamp_us = timestamp_us
        self._tau_next_ready = prediction is not None
        if prediction is None:
            self._tau_next_pred.fill(0.0)
            self._tau_ext_pred_raw.fill(0.0)
            self._tau_ext_pred.fill(0.0)
            return
        self._tau_next_pred = _finite_vector("tau_next_pred", prediction, 7)
        self._tau_ext_pred_raw = self._tau_next_pred - tau_next_target
        self._tau_ext_pred = (
            self.tau_ext_pred_filter.apply(self._tau_ext_pred_raw, timestamp_us)
            if self.tau_ext_pred_filter is not None
            else self._tau_ext_pred_raw.copy()
        )
        if not self._tau_next_ready_logged:
            log.info(
                "tau_next inference enabled after full history=%d samples",
                self.tau_next_predictor.metadata.horizon,
            )
            self._tau_next_ready_logged = True

    def _result_for_source(
        self,
        timestamp_us: int,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        updated: bool,
    ) -> OnlineTauExtResult:
        zeros = np.zeros(7, dtype=np.float64)
        maximum_age_us = int(round(self.config.maximum_prediction_age_s * 1e6))
        tau_f_age_us = (
            0
            if self._tau_f_timestamp_us is None
            else max(0, timestamp_us - self._tau_f_timestamp_us)
        )
        tau_next_age_us = (
            0
            if self._tau_next_timestamp_us is None
            else max(0, timestamp_us - self._tau_next_timestamp_us)
        )
        tau_f_valid = (
            self.tau_f_predictor is not None
            and self._tau_f_ready
            and self._tau_f_timestamp_us is not None
            and tau_f_age_us <= maximum_age_us
        )
        tau_next_valid = (
            self.tau_next_predictor is not None
            and self._tau_next_ready
            and self._tau_next_timestamp_us is not None
            and tau_next_age_us <= maximum_age_us
        )
        history_ready = (
            (self.tau_f_predictor is None or tau_f_valid)
            and (self.tau_next_predictor is None or tau_next_valid)
        )
        primary_timestamp_us = (
            self._tau_next_timestamp_us
            if self.tau_next_predictor is not None
            else self._tau_f_timestamp_us
        )
        observation_timestamp_us = primary_timestamp_us or 0
        age_us = max(0, timestamp_us - observation_timestamp_us)
        return OnlineTauExtResult(
            timestamp_us=timestamp_us, q=q.copy(), dq=dq.copy(),
            ddq_kf_causal=self._ddq.copy(), tau=tau.copy(),
            tau_id=self._tau_id.copy(), tau_id_filtered=self._tau_id_filtered.copy(),
            tau_f_pred=self._tau_f_pred.copy(),
            tau_next_pred=self._tau_next_pred.copy(),
            tau_ext_cal=self._tau_ext_cal.copy() if tau_f_valid else zeros.copy(),
            tau_ext_pred=self._tau_ext_pred.copy() if tau_next_valid else zeros.copy(),
            history_ready=history_ready, observation_updated=updated,
            observation_timestamp_us=observation_timestamp_us,
            prediction_age_us=age_us,
            tau_ext_cal_raw=(self._tau_ext_cal_raw.copy() if tau_f_valid else zeros.copy()),
            tau_ext_pred_raw=(self._tau_ext_pred_raw.copy() if tau_next_valid else zeros.copy()),
            tau_f_history_ready=tau_f_valid,
            tau_free_history_ready=tau_next_valid,
        )


def _require_features(
    features: Mapping[str, np.ndarray],
    input_keys: Sequence[str],
    model_name: str,
) -> None:
    missing = [key for key in input_keys if key not in features]
    if missing:
        raise RuntimeError(f"{model_name} checkpoint is missing model inputs: {missing}")


def _observation_features(
    source: _SourceObservation,
    *,
    q: np.ndarray | None = None,
    dq: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        "q": source.q if q is None else q,
        "dq": source.dq if dq is None else dq,
        "delta_q": source.q_cmd - source.q,
        "tau": source.tau if tau is None else tau,
    }


def _normalize_checkpoint_filters(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint dataloader_filters must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for key, raw_spec in value.items():
        key = str(key)
        if not isinstance(raw_spec, Mapping):
            raise RuntimeError(
                f"checkpoint dataloader_filters.{key} must be a mapping"
            )
        enabled = bool(raw_spec.get("enabled", False))
        raw_operations = raw_spec.get("operations", ())
        if not isinstance(raw_operations, Sequence) or isinstance(
            raw_operations, (str, bytes)
        ):
            raise RuntimeError(
                f"checkpoint dataloader_filters.{key}.operations must be a list"
            )
        operations = []
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, Mapping):
                raise RuntimeError("checkpoint filter operations must be mappings")
            operation_type = str(raw_operation.get("type", "")).strip().lower()
            if operation_type in {"median", "moving_average"}:
                window = int(raw_operation.get("window", 0))
                if window < 1 or (operation_type == "median" and window % 2 == 0):
                    raise RuntimeError(
                        f"invalid checkpoint {operation_type} window {window}"
                    )
                operations.append({"type": operation_type, "window": window})
            elif operation_type == "lowpass":
                cutoff_hz = float(raw_operation.get("cutoff_hz", np.nan))
                if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
                    raise RuntimeError("checkpoint lowpass cutoff_hz must be positive")
                operations.append(
                    {"type": "lowpass", "cutoff_hz": cutoff_hz}
                )
            else:
                raise RuntimeError(
                    f"unsupported checkpoint causal filter {operation_type!r}"
                )
        raw_preprocessed = raw_spec.get("dataset_preprocessed_operations", ())
        if not isinstance(raw_preprocessed, Sequence) or isinstance(
            raw_preprocessed, (str, bytes)
        ):
            raise RuntimeError(
                "checkpoint dataset_preprocessed_operations must be a list"
            )
        preprocessed_operations = []
        for operation in raw_preprocessed:
            if not isinstance(operation, Mapping):
                raise RuntimeError(
                    "checkpoint dataset_preprocessed_operations entries must be mappings"
                )
            operation_type = str(operation.get("type", "")).strip().lower()
            if operation_type in {"median", "moving_average"}:
                preprocessed_operations.append(
                    {"type": operation_type, "window": int(operation["window"])}
                )
            elif operation_type == "lowpass":
                preprocessed_operations.append(
                    {"type": "lowpass", "cutoff_hz": float(operation["cutoff_hz"])}
                )
            else:
                raise RuntimeError(
                    f"unsupported preprocessed checkpoint filter {operation_type!r}"
                )
        if operations[: len(preprocessed_operations)] != preprocessed_operations:
            raise RuntimeError(
                "checkpoint dataset_preprocessed_operations must be an exact "
                "prefix of operations"
            )
        if enabled and not operations:
            raise RuntimeError(
                f"checkpoint dataloader_filters.{key} is enabled without operations"
            )
        normalized[key] = {
            "enabled": enabled,
            "operations": operations,
            "dataset_preprocessed_operations": preprocessed_operations,
        }
    return normalized


def _build_feature_filter_bank(
    metadata: SequenceCheckpointMetadata,
    *,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> dict[str, CausalFilterPipeline]:
    result = {}
    keys = metadata.input_keys if include_keys is None else tuple(include_keys)
    for key in keys:
        if exclude_keys is not None and key in exclude_keys:
            continue
        spec = metadata.dataloader_filters.get(key) or {}
        if bool(spec.get("enabled", False)):
            result[key] = CausalFilterPipeline(spec["operations"])
    return result


def _apply_feature_filter_bank(
    features: Mapping[str, np.ndarray],
    filter_bank: Mapping[str, CausalFilterPipeline],
    timestamp_us: int,
) -> dict[str, np.ndarray]:
    return {
        key: (
            filter_bank[key].apply(value, timestamp_us)
            if key in filter_bank
            else np.asarray(value, dtype=np.float64).copy()
        )
        for key, value in features.items()
    }


def _reset_filter_bank(filter_bank: Mapping[str, CausalFilterPipeline]) -> None:
    for pipeline in filter_bank.values():
        pipeline.reset()


def _validate_predictor_contract(
    metadata: SequenceCheckpointMetadata,
    *,
    model_name: str,
    output_key: str,
) -> None:
    unknown_inputs = sorted(set(metadata.input_keys) - _MODEL_INPUT_KEYS)
    if not metadata.input_keys or unknown_inputs:
        raise RuntimeError(
            f"{model_name} checkpoint inputs must be a non-empty ordered subset "
            f"of {sorted(_MODEL_INPUT_KEYS)}, got {metadata.input_keys}"
        )
    if len(set(metadata.input_keys)) != len(metadata.input_keys):
        raise RuntimeError(f"{model_name} checkpoint inputs contain duplicates")
    invalid_dims = {
        key: metadata.input_dims.get(key)
        for key in metadata.input_keys
        if metadata.input_dims.get(key) != 7
    }
    if invalid_dims:
        raise RuntimeError(
            f"{model_name} online inputs must all have dimension 7, got "
            f"{invalid_dims}"
        )
    if metadata.output_key != output_key:
        raise RuntimeError(
            f"{model_name} checkpoint output must be {output_key!r}, "
            f"got {metadata.output_key!r}"
        )
    if model_name == "tau_f":
        if metadata.target_contract not in _TAU_F_TARGET_CONTRACTS:
            raise RuntimeError(
                "tau_f checkpoint target_contract must be one of "
                f"{sorted(_TAU_F_TARGET_CONTRACTS)}, got "
                f"{metadata.target_contract!r}; rebuild matched-filter labels "
                "or use a causal_rnea_residual_v1 checkpoint"
            )
        if metadata.target_contract == _DERIVED_TAU_F_TARGET_CONTRACT:
            _validate_derived_tau_f_contract(metadata)


def _validate_derived_tau_f_contract(
    metadata: SequenceCheckpointMetadata,
) -> None:
    target = metadata.derived_target_config
    if not target:
        raise RuntimeError(
            "causal_rnea_residual_v1 checkpoint is missing "
            "derived_target_config"
        )
    required_values = {
        "enabled": True,
        "method": _DERIVED_TAU_F_TARGET_CONTRACT,
        "target_key": metadata.output_key,
        "ddq_source": "variable_dt_kalman_forward_filter",
        "residual_formula": "tau_f=tau_filtered-tau_id_filtered",
    }
    for key, expected in required_values.items():
        if target.get(key) != expected:
            raise RuntimeError(
                f"tau_f checkpoint derived_target_config.{key} must be "
                f"{expected!r}, got {target.get(key)!r}"
            )

    source_keys = target.get("source_keys")
    if not isinstance(source_keys, Mapping) or set(source_keys) != {"q", "dq", "tau"}:
        raise RuntimeError(
            "tau_f checkpoint derived_target_config.source_keys must define "
            "q, dq, and tau"
        )
    rnea_state_source = str(target.get("rnea_state_source", "")).strip().lower()
    if rnea_state_source not in {"measured", "filtered"}:
        raise RuntimeError(
            "tau_f checkpoint derived_target_config.rnea_state_source must be "
            "measured or filtered"
        )
    if str(target.get("torque_filter_key", "")) != "tau":
        raise RuntimeError(
            "online tau_f inference requires derived_target_config."
            "torque_filter_key='tau'"
        )

    raw_operations = target.get("torque_filter_operations", ())
    normalized_target_filter = _normalize_checkpoint_filters(
        {
            "tau": {
                "enabled": bool(raw_operations),
                "operations": raw_operations,
            }
        }
    )["tau"]
    checkpoint_tau_filter = metadata.dataloader_filters.get("tau") or {
        "enabled": False,
        "operations": [],
    }
    if (
        bool(checkpoint_tau_filter.get("enabled", False))
        != normalized_target_filter["enabled"]
        or list(checkpoint_tau_filter.get("operations", ()))
        != normalized_target_filter["operations"]
    ):
        raise RuntimeError(
            "tau_f checkpoint derived torque_filter_operations do not match "
            "dataloader_filters.tau"
        )
    _derived_kalman_config(metadata)
    _derived_dq_sign(target)


def _derived_kalman_config(
    metadata: SequenceCheckpointMetadata,
) -> CausalKalmanConfig | None:
    if metadata.target_contract != _DERIVED_TAU_F_TARGET_CONTRACT:
        return None
    raw = metadata.derived_target_config.get("state_estimator")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            "tau_f checkpoint derived_target_config.state_estimator must be a "
            "mapping"
        )
    parameter_names = (
        "position_std",
        "velocity_std",
        "jerk_std",
        "initial_position_std",
        "initial_velocity_std",
        "initial_acceleration_std",
    )
    unknown = sorted(set(raw) - {*parameter_names, "max_gap_s"})
    missing = [key for key in parameter_names if key not in raw]
    if "max_gap_s" not in raw:
        missing.append("max_gap_s")
    if unknown or missing:
        raise RuntimeError(
            "invalid tau_f checkpoint state_estimator; "
            f"missing={missing}, unknown={unknown}"
        )

    def joint_values(name: str) -> tuple[float, ...]:
        value = raw[name]
        if isinstance(value, (int, float)):
            result = (float(value),) * 7
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = tuple(float(item) for item in value)
        else:
            result = ()
        if len(result) != 7 or any(
            not np.isfinite(item) or item <= 0.0 for item in result
        ):
            raise RuntimeError(
                "tau_f checkpoint derived_target_config.state_estimator."
                f"{name} must be a positive scalar or seven positive values"
            )
        return result

    max_gap_s = float(raw["max_gap_s"])
    if not np.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise RuntimeError(
            "tau_f checkpoint state_estimator.max_gap_s must be positive and finite"
        )
    return CausalKalmanConfig(
        **{name: joint_values(name) for name in parameter_names},
        max_gap_s=max_gap_s,
    )


def _derived_dq_sign(target: Mapping[str, Any]) -> np.ndarray:
    raw = target.get("dq_sign")
    if raw is None:
        return np.ones(7, dtype=np.float64)
    sign = np.asarray(raw, dtype=np.float64).reshape(-1)
    if sign.shape != (7,) or not np.all(np.isin(sign, (-1.0, 1.0))):
        raise RuntimeError(
            "tau_f checkpoint derived_target_config.dq_sign must contain seven "
            "values chosen from -1 and 1"
        )
    return sign.copy()


def _validate_predictor_sample_rate(
    metadata: SequenceCheckpointMetadata,
    *,
    model_name: str,
    expected_hz: float,
) -> None:
    checkpoint_hz = metadata.sample_rate_hz
    if checkpoint_hz is None:
        return
    if not np.isclose(checkpoint_hz, expected_hz, rtol=0.0, atol=1.0e-9):
        raise RuntimeError(
            f"{model_name} checkpoint sample_rate_hz={checkpoint_hz:g} does not "
            f"match tau_ext_inference.{model_name}.observation_sample_rate_hz="
            f"{expected_hz:g}"
        )


def _resolve_observation_sample_rate(
    config: SequenceCheckpointConfig,
    metadata: SequenceCheckpointMetadata | None,
    *,
    model_name: str,
) -> float | None:
    if metadata is None:
        return None
    configured_hz = config.observation_sample_rate_hz
    checkpoint_hz = metadata.sample_rate_hz
    if configured_hz is not None:
        _validate_predictor_sample_rate(
            metadata,
            model_name=model_name,
            expected_hz=configured_hz,
        )
        return configured_hz
    if checkpoint_hz is not None:
        return checkpoint_hz
    # Injected and legacy predictors without timing metadata retain the former
    # 50 Hz online observation contract. New checkpoints should carry a rate.
    return 50.0


def _build_tau_f_source_filter(
    metadata: SequenceCheckpointMetadata,
    processing: DynamicsProcessingConfig,
    robot_states: Mapping[str, StateParamConfig],
) -> CausalFilterPipeline | OnePoleLowPass:
    """Restore the matched measured-tau/RNEA chain for tau_f residuals."""

    if metadata.dataloader_filters:
        spec = metadata.dataloader_filters.get("tau") or {}
        if not bool(spec.get("enabled", False)):
            raise RuntimeError(
                "tau_f checkpoint dataloader.filters.tau must be enabled"
            )
        return CausalFilterPipeline(spec["operations"])

    del processing, robot_states
    if not metadata.target_filter_enabled:
        raise RuntimeError(
            "legacy tau_f checkpoint must define model.target_filter"
        )
    cutoff_hz = metadata.target_filter_cutoff_hz
    median_window = metadata.target_filter_median_window
    if cutoff_hz is None or not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise RuntimeError("legacy tau_f checkpoint target filter needs cutoff_hz")
    if median_window is None or median_window < 1 or median_window % 2 == 0:
        raise RuntimeError(
            "legacy tau_f checkpoint target filter needs an odd median_window"
        )
    return OnePoleLowPass(float(cutoff_hz), int(median_window))


def _build_tau_target_filter(
    metadata: SequenceCheckpointMetadata,
) -> CausalFilterPipeline | CausalTrailingMovingAverage | CausalTrailingMedian | CausalWindowLowPass | OnePoleLowPass | None:
    """Restore tau target preprocessing, preferring the unified new contract."""

    if metadata.dataloader_filters:
        spec = metadata.dataloader_filters.get("tau") or {}
        if not bool(spec.get("enabled", False)):
            return None
        return CausalFilterPipeline(spec["operations"])
    return _build_tau_next_target_filter(metadata)


def _build_tau_next_target_filter(
    metadata: SequenceCheckpointMetadata,
) -> (
    CausalTrailingMovingAverage
    | CausalTrailingMedian
    | CausalWindowLowPass
    | OnePoleLowPass
    | None
):
    """Replay tau_free target preprocessing on measured torque only.

    The shared measured-torque filter remains reserved for the tau_f matched
    contract. The tau_free checkpoint target filter is applied afterwards,
    matching the order used by PINN training.
    """

    if not metadata.target_filter_enabled:
        return None
    moving_average_window = metadata.target_filter_moving_average_window
    median_window = metadata.target_filter_median_window
    if moving_average_window is not None and median_window is not None:
        raise RuntimeError(
            "tau_next checkpoint target_filter must not define both "
            "moving_average_window and median_window"
        )
    if moving_average_window is not None:
        if moving_average_window < 1:
            raise RuntimeError(
                "tau_next checkpoint target_filter.moving_average_window "
                "must be a positive integer"
            )
        if not metadata.target_filter_apply_additional_lowpass:
            if moving_average_window == 1:
                return None
            return CausalTrailingMovingAverage(moving_average_window)

        cutoff_hz = metadata.target_filter_cutoff_hz
        if cutoff_hz is None or not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise RuntimeError(
                "tau_next checkpoint must define target_filter.cutoff_hz when "
                "apply_additional_lowpass=true"
            )
        return CausalWindowLowPass(
            "moving_average",
            moving_average_window,
            float(cutoff_hz),
        )

    if median_window is None or median_window < 1 or median_window % 2 == 0:
        raise RuntimeError(
            "tau_next checkpoint target_filter must define either a positive "
            "moving_average_window or a positive odd median_window"
        )
    if not metadata.target_filter_apply_additional_lowpass:
        if median_window == 1:
            return None
        return CausalTrailingMedian(median_window)

    cutoff_hz = metadata.target_filter_cutoff_hz
    if cutoff_hz is None or not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise RuntimeError(
            "tau_next checkpoint must define target_filter.cutoff_hz when "
            "apply_additional_lowpass=true"
        )
    return OnePoleLowPass(float(cutoff_hz), median_window)


def _build_tau_ext_post_filter(
    config: TauExtFilterConfig,
) -> CausalHampelButterworth | CausalWindowLowPass | None:
    if not config.enabled:
        return None
    if config.mode == "hampel_butterworth":
        return CausalHampelButterworth(
            window=config.window,
            n_sigma=config.hampel_n_sigma,
            cutoff_hz=config.cutoff_hz,
            sample_rate_hz=config.sample_rate_hz,
            order=config.order,
        )
    return CausalWindowLowPass(config.mode, config.window, config.cutoff_hz)


def _build_checkpoint_model(
    torch: Any,
    nn: Any,
    config: Mapping[str, Any],
    input_keys: Sequence[str],
    input_dims: Mapping[str, int],
):
    architecture = str(config.get("architecture", "lstm")).lower()
    recurrent_types = {"lstm": nn.LSTM, "gru": nn.GRU}
    if architecture not in recurrent_types:
        raise RuntimeError(f"unsupported recurrent architecture: {architecture!r}")
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 2))
    output_dim = int(config.get("output_dim", 7))
    dropout = float(config.get("dropout", 0.1))
    head_hidden_dim = int(config.get("head_hidden_dim", 256))
    head_num_layers = int(config.get("head_num_layers", 2))
    activation_name = str(config.get("activation", "relu")).lower()
    activations = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    if activation_name not in activations:
        raise RuntimeError(f"unsupported checkpoint activation: {activation_name!r}")

    class CheckpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = recurrent_types[architecture](
                input_size=sum(input_dims[key] for key in input_keys),
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            if head_num_layers == 1:
                self.head = nn.Linear(hidden_dim, output_dim)
                return
            layers = []
            in_dim = hidden_dim
            for _ in range(head_num_layers - 1):
                layers.extend(
                    [
                        nn.Linear(in_dim, head_hidden_dim),
                        activations[activation_name](),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = head_hidden_dim
            layers.append(nn.Linear(in_dim, output_dim))
            self.head = nn.Sequential(*layers)

        def forward(self, sequence):
            recurrent_output, _ = self.recurrent(sequence)
            return self.head(recurrent_output[:, -1])

    return CheckpointModel()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _resolve_checkpoint_path(path: Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise RuntimeError(f"{name} checkpoint does not exist: {resolved}")
    candidates = tuple(resolved.glob("*.pt"))
    if not candidates:
        raise RuntimeError(f"{name} checkpoint directory contains no .pt files: {resolved}")

    def score(candidate: Path) -> tuple[float, float]:
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.pt$", candidate.name)
        loss = float(match.group(1)) if match else float("inf")
        return loss, -candidate.stat().st_mtime

    selected = min(candidates, key=score)
    log.info(
        "%s checkpoint directory selected %s from %s",
        name,
        selected.name,
        resolved,
    )
    return selected.resolve()


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise RuntimeError(
            f"tau_ext inference requires a finite {size}D {name}; got {vector}"
        )
    return vector.copy()
