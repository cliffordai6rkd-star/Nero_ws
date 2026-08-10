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
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    SequenceCheckpointConfig,
    StateParamConfig,
    TauExtFilterConfig,
    TauExtInferenceConfig,
)
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator
from nero_collection.filters import (
    CausalTrailingMedian,
    CausalWindowLowPass,
    OnePoleLowPass,
)


log = logging.getLogger(__name__)

_DEFAULT_CPU_TORCH_NUM_THREADS = 1
_MODEL_INPUT_KEYS = frozenset({"q", "dq", "delta_q"})
_TAU_F_TARGET_CONTRACT = "matched_causal_torque_filter_v1"


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
    inference_mode: str = "fixed_window"
    target_contract: str | None = None
    target_filter_enabled: bool = False
    target_filter_cutoff_hz: float | None = None
    target_filter_median_window: int | None = None
    target_filter_apply_additional_lowpass: bool = False


@dataclass(frozen=True)
class TauExtInferenceMetadata:
    tau_f: SequenceCheckpointMetadata
    tau_next: SequenceCheckpointMetadata
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
    tau_ext_cal_raw: np.ndarray | None = None
    tau_ext_pred_raw: np.ndarray | None = None


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
        target_filter_median_window = target_filter_config.get("median_window")
        target_filter_apply_additional_lowpass = bool(
            target_filter_config.get("apply_additional_lowpass", False)
        )
        if target_filter_cutoff_hz is not None:
            target_filter_cutoff_hz = float(target_filter_cutoff_hz)
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
                f"{unknown_inputs}; expected only q, dq, delta_q"
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
            target_contract=target_contract,
            target_filter_enabled=target_filter_enabled,
            target_filter_cutoff_hz=target_filter_cutoff_hz,
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
    """Compute both external-torque estimates on one aligned online sample."""

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
        if not config.enabled and (
            tau_f_predictor is None or tau_next_predictor is None
        ):
            raise ValueError("tau_ext inference is disabled")
        self.config = config
        self.tau_f_predictor = tau_f_predictor or SequenceTorquePredictor(
            config.tau_f,
            name="tau_f",
        )
        self.tau_next_predictor = tau_next_predictor or SequenceTorquePredictor(
            config.tau_next,
            name="tau_next",
        )
        _validate_predictor_contract(
            self.tau_f_predictor.metadata,
            model_name="tau_f",
            output_key="tau_f",
        )
        _validate_predictor_contract(
            self.tau_next_predictor.metadata,
            model_name="tau_next",
            output_key="tau",
        )
        self.estimator = estimator or PinocchioJointTorqueResidualEstimator(
            inverse_dynamics
        )
        self.state_estimator = state_estimator or CausalJointKalmanFilter(
            config.state_estimator
        )
        torque_filter = _tau_filter_spec(dynamics_processing, robot_states)
        _validate_tau_filter_contract(self.tau_f_predictor.metadata, torque_filter)
        assert torque_filter is not None
        cutoff_hz, median_window = torque_filter
        self.tau_filter = OnePoleLowPass(cutoff_hz, median_window)
        self.tau_id_filter = OnePoleLowPass(cutoff_hz, median_window)
        self.tau_next_target_filter = _build_tau_next_target_filter(
            self.tau_next_predictor.metadata
        )
        tau_ext_filter = config.tau_ext_filter
        self.tau_ext_cal_filter = (
            CausalWindowLowPass(
                tau_ext_filter.mode,
                tau_ext_filter.window,
                tau_ext_filter.cutoff_hz,
            )
            if tau_ext_filter.enabled
            else None
        )
        self.tau_ext_pred_filter = (
            CausalWindowLowPass(
                tau_ext_filter.mode,
                tau_ext_filter.window,
                tau_ext_filter.cutoff_hz,
            )
            if tau_ext_filter.enabled
            else None
        )
        self._last_timestamp_us: int | None = None
        self._last_result: OnlineTauExtResult | None = None
        self._history_ready_logged = False

    @property
    def metadata(self) -> TauExtInferenceMetadata:
        return TauExtInferenceMetadata(
            tau_f=self.tau_f_predictor.metadata,
            tau_next=self.tau_next_predictor.metadata,
            input_keys=tuple(
                dict.fromkeys(
                    self.tau_f_predictor.metadata.input_keys
                    + self.tau_next_predictor.metadata.input_keys
                )
            ),
            tau_ext_filter=self.config.tau_ext_filter,
        )

    def warm_up(self) -> None:
        self.tau_f_predictor.warm_up()
        self.tau_next_predictor.warm_up()

    def reset_episode(self) -> None:
        self.tau_f_predictor.reset()
        self.tau_next_predictor.reset()
        self.state_estimator.reset()
        self.tau_filter.reset()
        self.tau_id_filter.reset()
        if self.tau_next_target_filter is not None:
            self.tau_next_target_filter.reset()
        if self.tau_ext_cal_filter is not None:
            self.tau_ext_cal_filter.reset()
        if self.tau_ext_pred_filter is not None:
            self.tau_ext_pred_filter.reset()
        self._last_timestamp_us = None
        self._last_result = None
        self._history_ready_logged = False

    def estimate_aligned(
        self,
        timestamp_us: int,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        q_cmd: np.ndarray,
    ) -> OnlineTauExtResult:
        timestamp_us = int(timestamp_us)
        if self._last_timestamp_us is not None:
            if timestamp_us < self._last_timestamp_us:
                raise ValueError(
                    "aligned tau_ext inference timestamp moved backwards: "
                    f"{timestamp_us} < {self._last_timestamp_us}"
                )
            if timestamp_us == self._last_timestamp_us:
                assert self._last_result is not None
                return self._last_result

        q_value = _finite_vector("q", q, 7)
        dq_value = _finite_vector("dq", dq, 7)
        tau_value = _finite_vector("tau", tau, 7)
        q_cmd_value = _finite_vector("q_cmd", q_cmd, 7)
        tau_value = self.tau_filter.apply(tau_value, timestamp_us)
        tau_next_target = (
            self.tau_next_target_filter.apply(tau_value, timestamp_us)
            if self.tau_next_target_filter is not None
            else tau_value
        )

        kalman_state = self.state_estimator.update(timestamp_us, q_value, dq_value)
        ddq_value = kalman_state.ddq
        inverse_dynamics = self.estimator.estimate(
            q_value,
            dq_value,
            ddq_value,
            tau_value,
        )
        tau_id = _finite_vector("tau_id", inverse_dynamics.tau_id, 7)
        tau_id_filtered = self.tau_id_filter.apply(tau_id, timestamp_us)
        features = {
            "q": q_value,
            "dq": dq_value,
            "delta_q": q_cmd_value - q_value,
        }
        _require_features(features, self.tau_f_predictor.metadata.input_keys, "tau_f")
        _require_features(
            features,
            self.tau_next_predictor.metadata.input_keys,
            "tau_next",
        )
        tau_f_prediction = self.tau_f_predictor.append_and_predict(features)
        tau_next_prediction = self.tau_next_predictor.append_and_predict(features)
        history_ready = tau_f_prediction is not None and tau_next_prediction is not None
        if history_ready:
            tau_f_pred = _finite_vector("tau_f_pred", tau_f_prediction, 7)
            tau_next_pred = _finite_vector("tau_next_pred", tau_next_prediction, 7)
            tau_ext_cal_raw = tau_id_filtered + tau_f_pred - tau_value
            tau_ext_pred_raw = tau_next_pred - tau_next_target
            tau_ext_cal = (
                self.tau_ext_cal_filter.apply(tau_ext_cal_raw, timestamp_us)
                if self.tau_ext_cal_filter is not None
                else tau_ext_cal_raw.copy()
            )
            tau_ext_pred = (
                self.tau_ext_pred_filter.apply(tau_ext_pred_raw, timestamp_us)
                if self.tau_ext_pred_filter is not None
                else tau_ext_pred_raw.copy()
            )
            if not self._history_ready_logged:
                log.info(
                    "tau_ext inference enabled after full history tau_f=%d tau_next=%d samples",
                    self.tau_f_predictor.metadata.horizon,
                    self.tau_next_predictor.metadata.horizon,
                )
                self._history_ready_logged = True
        else:
            tau_f_pred = np.zeros(7, dtype=np.float64)
            tau_next_pred = np.zeros(7, dtype=np.float64)
            tau_ext_cal = np.zeros(7, dtype=np.float64)
            tau_ext_pred = np.zeros(7, dtype=np.float64)
            tau_ext_cal_raw = np.zeros(7, dtype=np.float64)
            tau_ext_pred_raw = np.zeros(7, dtype=np.float64)
        result = OnlineTauExtResult(
            timestamp_us=timestamp_us,
            q=q_value.copy(),
            dq=dq_value.copy(),
            ddq_kf_causal=ddq_value.copy(),
            tau=tau_value.copy(),
            tau_id=tau_id.copy(),
            tau_id_filtered=tau_id_filtered.copy(),
            tau_f_pred=tau_f_pred.copy(),
            tau_next_pred=tau_next_pred.copy(),
            tau_ext_cal=tau_ext_cal.copy(),
            tau_ext_pred=tau_ext_pred.copy(),
            history_ready=history_ready,
            tau_ext_cal_raw=tau_ext_cal_raw.copy(),
            tau_ext_pred_raw=tau_ext_pred_raw.copy(),
        )
        self._last_timestamp_us = timestamp_us
        self._last_result = result
        return result


def _require_features(
    features: Mapping[str, np.ndarray],
    input_keys: Sequence[str],
    model_name: str,
) -> None:
    missing = [key for key in input_keys if key not in features]
    if missing:
        raise RuntimeError(f"{model_name} checkpoint is missing model inputs: {missing}")


def _validate_predictor_contract(
    metadata: SequenceCheckpointMetadata,
    *,
    model_name: str,
    output_key: str,
) -> None:
    expected_inputs = ("q", "dq", "delta_q")
    if metadata.input_keys != expected_inputs:
        raise RuntimeError(
            f"{model_name} checkpoint inputs must be {expected_inputs}, "
            f"got {metadata.input_keys}"
        )
    if metadata.output_key != output_key:
        raise RuntimeError(
            f"{model_name} checkpoint output must be {output_key!r}, "
            f"got {metadata.output_key!r}"
        )
    if model_name == "tau_f" and metadata.target_contract != _TAU_F_TARGET_CONTRACT:
        raise RuntimeError(
            "tau_f checkpoint target_contract must be "
            f"{_TAU_F_TARGET_CONTRACT!r}, got {metadata.target_contract!r}; "
            "rebuild matched-filter labels and retrain the checkpoint"
        )


def _tau_filter_spec(
    processing: DynamicsProcessingConfig,
    robot_states: Mapping[str, StateParamConfig],
) -> tuple[float, int] | None:
    if processing.enabled:
        return (
            float(processing.torque_lowpass_hz),
            int(processing.torque_median_window),
        )
    param = robot_states.get("torque")
    if param is None or not param.lowpass:
        return None
    return float(param.lowpass_cutoff_hz), int(param.median_window)


def _validate_tau_filter_contract(
    metadata: SequenceCheckpointMetadata,
    online_filter: tuple[float, int] | None,
) -> None:
    if not metadata.target_filter_enabled:
        raise RuntimeError("tau_f checkpoint target_filter must be enabled")
    cutoff_hz = metadata.target_filter_cutoff_hz
    median_window = metadata.target_filter_median_window
    if cutoff_hz is None or not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise RuntimeError("tau_f checkpoint must define target_filter.cutoff_hz")
    if median_window is None or median_window < 1 or median_window % 2 == 0:
        raise RuntimeError(
            "tau_f checkpoint must define a positive odd target_filter.median_window"
        )
    expected = (float(cutoff_hz), int(median_window))
    if online_filter is None:
        raise RuntimeError(
            "online measured torque filtering must be enabled for the matched-filter "
            "tau_f checkpoint"
        )
    if not np.isclose(online_filter[0], expected[0], rtol=0.0, atol=1.0e-12) or (
        online_filter[1] != expected[1]
    ):
        raise RuntimeError(
            f"online torque filter {online_filter} does not match tau_f checkpoint "
            f"target filter {expected}"
        )


def _build_tau_next_target_filter(
    metadata: SequenceCheckpointMetadata,
) -> CausalTrailingMedian | OnePoleLowPass | None:
    """Replay tau_free target preprocessing on measured torque only.

    The shared measured-torque filter remains reserved for the tau_f matched
    contract. The tau_free checkpoint target filter is applied afterwards,
    matching the order used by PINN training.
    """

    if not metadata.target_filter_enabled:
        return None
    median_window = metadata.target_filter_median_window
    if median_window is None or median_window < 1 or median_window % 2 == 0:
        raise RuntimeError(
            "tau_next checkpoint target_filter.median_window must be a positive odd integer"
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
