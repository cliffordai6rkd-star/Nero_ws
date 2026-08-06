from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from nero_collection.config import (
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    StateParamConfig,
    TauFInferenceConfig,
)
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator
from nero_collection.filters import OnePoleLowPass
from nero_collection.realtime_dynamics import (
    CenteredThreePointJointStateStream,
    RealtimeJointState,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TauFCheckpointMetadata:
    checkpoint_path: Path
    horizon: int
    input_keys: tuple[str, ...]
    input_dims: dict[str, int]
    output_key: str
    output_dim: int
    architecture: str
    normalize_mode: str


@dataclass(frozen=True)
class OnlineTauFResult:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    tau_id: np.ndarray
    tau_f_cal: np.ndarray
    tau_f_pred: np.ndarray
    tau_ext: np.ndarray
    tau_bg_pred: np.ndarray | None = None
    tau_ext_raw: np.ndarray | None = None
    tau_ext_filtered: np.ndarray | None = None

    @property
    def model_prediction(self) -> np.ndarray:
        if self.tau_bg_pred is not None:
            return self.tau_bg_pred
        return self.tau_f_pred


class TauFPredictor(Protocol):
    metadata: TauFCheckpointMetadata

    def warm_up(self) -> None:
        ...

    def reset_recurrent_state(self) -> None:
        ...

    def append_and_predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        ...

    def append_sequence_and_predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        ...

    def predict_sequence(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        ...


class ZeroTauFPredictor:
    """Compatibility predictor used when learned tau_f inference is disabled."""

    metadata = TauFCheckpointMetadata(
        checkpoint_path=Path("<tau_f_disabled>"),
        horizon=1,
        input_keys=("q", "dq", "ddq", "tau"),
        input_dims={"q": 7, "dq": 7, "ddq": 7, "tau": 7},
        output_key="tau_f",
        output_dim=7,
        architecture="zeros",
        normalize_mode="none",
    )

    def warm_up(self) -> None:
        return None

    def reset_recurrent_state(self) -> None:
        return None

    def append_and_predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        for key in self.metadata.input_keys:
            _finite_vector(key, features[key], self.metadata.input_dims[key])
        return np.zeros(self.metadata.output_dim, dtype=np.float64)

    def append_sequence_and_predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        return self.predict_sequence(features)

    def predict_sequence(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        lengths = {
            np.asarray(features[key]).shape[0]
            for key in self.metadata.input_keys
        }
        if len(lengths) != 1:
            raise ValueError("All zero tau_f predictor inputs must share time length")
        return np.zeros((lengths.pop(), self.metadata.output_dim), dtype=np.float64)


class TauFSequencePredictor:
    """Restore a PINN tau_f checkpoint and run stateful, single-frame inference."""

    def __init__(self, config: TauFInferenceConfig) -> None:
        if config.checkpoint_path is None:
            raise ValueError("tau_f checkpoint path is required")
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "tau_f inference requires PyTorch; install the tau-f-inference extra"
            ) from exc

        checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(f"tau_f checkpoint does not exist: {checkpoint_path}")
        torch.set_num_threads(config.torch_num_threads)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # PyTorch before weights_only was introduced.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("tau_f checkpoint root must be a mapping")

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
            model_config.get("input_dims"),
            "checkpoint.config.model.input_dims",
        )
        input_dims = {key: int(input_dims_config.get(key, 7)) for key in input_keys}
        output_key = str(model_config.get("target_key", "tau_f"))
        output_dim = int(model_config.get("output_dim", 7))
        architecture = str(model_config.get("architecture", "lstm")).lower()
        if horizon <= 0 or not input_keys:
            raise RuntimeError("checkpoint must define a positive horizon and model inputs")
        allowed_inputs = {"q", "dq", "ddq", "tau"}
        unknown_inputs = sorted(set(input_keys) - allowed_inputs)
        if unknown_inputs:
            raise RuntimeError(f"checkpoint requests unsupported inputs: {unknown_inputs}")
        if len(set(input_keys)) != len(input_keys):
            raise RuntimeError("checkpoint model inputs contain duplicates")
        if any(dim <= 0 for dim in input_dims.values()) or output_dim <= 0:
            raise RuntimeError("checkpoint model dimensions must be positive")
        if config.horizon is not None and config.horizon != horizon:
            raise RuntimeError(
                f"configured horizon={config.horizon} does not match checkpoint horizon={horizon}"
            )
        if config.input_keys is not None and config.input_keys != input_keys:
            raise RuntimeError(
                f"configured input_keys={config.input_keys} do not match checkpoint inputs={input_keys}"
            )
        if config.output_key is not None and config.output_key != output_key:
            raise RuntimeError(
                f"configured output_key={config.output_key!r} does not match "
                f"checkpoint target={output_key!r}"
            )

        normalizer = _mapping(checkpoint.get("normalizer"), "checkpoint.normalizer")
        self._stats = _mapping(normalizer.get("stats"), "checkpoint.normalizer.stats")
        self._eps = float(normalizer.get("eps", 1e-6))
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
                f"unsupported checkpoint normalization mode: {self._normalize_mode!r}"
            )
        missing_stats = [
            key
            for key in (*input_keys, output_key)
            if key in self._normalize_keys and key not in self._stats
        ]
        if missing_stats:
            raise RuntimeError(f"checkpoint is missing normalizer stats: {missing_stats}")

        self._torch = torch
        self._device = torch.device(config.device)
        self._model = _build_checkpoint_model(
            torch,
            nn,
            model_config,
            input_keys,
            input_dims,
        )
        model_state = checkpoint.get("model")
        if not isinstance(model_state, Mapping):
            raise RuntimeError("checkpoint.model must contain a state dictionary")
        self._model.load_state_dict(model_state, strict=True)
        try:
            self._model.to(self._device)
        except Exception as exc:
            raise RuntimeError(
                f"cannot place tau_f model on device {config.device!r}: {exc}"
            ) from exc
        self._model.eval()
        self.metadata = TauFCheckpointMetadata(
            checkpoint_path=checkpoint_path,
            horizon=horizon,
            input_keys=input_keys,
            input_dims=input_dims,
            output_key=output_key,
            output_dim=output_dim,
            architecture=architecture,
            normalize_mode=self._normalize_mode,
        )
        self._recurrent_state = None
        self.reset_recurrent_state()
        log.info(
            "tau_f checkpoint ready path=%s architecture=%s training_horizon=%d "
            "inputs=%s output=%s device=%s inference=stateful_step",
            checkpoint_path,
            architecture,
            horizon,
            ",".join(input_keys),
            output_key,
            config.device,
        )

    def append_and_predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        step = {}
        for key, dim in self.metadata.input_dims.items():
            value = _finite_vector(key, features[key], dim)
            tensor = self._torch.as_tensor(
                value,
                dtype=self._torch.float32,
                device=self._device,
            ).unsqueeze(0)
            step[key] = self._normalize(key, tensor)

        with self._torch.inference_mode():
            out = self._model.forward_step(step, self._recurrent_state)
            self._recurrent_state = self._model.detach_recurrent_state(
                out["recurrent_state"]
            )
            prediction = self._denormalize(
                self.metadata.output_key,
                out["tau_f_pred"],
            )
        result = prediction.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return _finite_vector("tau_f_pred", result, self.metadata.output_dim)

    def append_sequence_and_predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Advance exact single-step recurrence with one host synchronization."""
        tensors = {}
        sequence_length = None
        for key in self.metadata.input_keys:
            value = self._torch.as_tensor(
                features[key],
                dtype=self._torch.float32,
                device=self._device,
            )
            expected_dim = self.metadata.input_dims[key]
            if value.ndim != 2 or value.shape[-1] != expected_dim:
                raise ValueError(
                    f"tau_f batch input {key!r} must have shape [T, "
                    f"{expected_dim}], got {tuple(value.shape)}"
                )
            if sequence_length is None:
                sequence_length = int(value.shape[0])
            elif value.shape[0] != sequence_length:
                raise ValueError("All tau_f batch inputs must share time length")
            tensors[key] = self._normalize(key, value)
        if sequence_length is None or sequence_length <= 0:
            return np.empty((0, self.metadata.output_dim), dtype=np.float64)
        with self._torch.inference_mode():
            predictions = []
            for index in range(sequence_length):
                step = {
                    key: value[index].unsqueeze(0)
                    for key, value in tensors.items()
                }
                out = self._model.forward_step(step, self._recurrent_state)
                self._recurrent_state = self._model.detach_recurrent_state(
                    out["recurrent_state"]
                )
                predictions.append(out["tau_f_pred"])
            prediction = self._torch.cat(predictions, dim=0)
            prediction = self._denormalize(
                self.metadata.output_key,
                prediction,
            )
        result = prediction.detach().cpu().numpy().astype(np.float64)
        expected_shape = (sequence_length, self.metadata.output_dim)
        if result.shape != expected_shape or not np.isfinite(result).all():
            raise RuntimeError(
                "tau_f batch prediction has unexpected values or shape: "
                f"{result.shape}, expected {expected_shape}"
            )
        return result.copy()

    def reset_recurrent_state(self) -> None:
        """Reset state at an episode, controller, or trajectory boundary."""
        self._recurrent_state = self._model.init_recurrent_state(
            batch_size=1,
            device=self._device,
        )

    def warm_up(self) -> None:
        step = {
            key: self._torch.zeros(
                (1, dim),
                dtype=self._torch.float32,
                device=self._device,
            )
            for key, dim in self.metadata.input_dims.items()
        }
        step = {key: self._normalize(key, value) for key, value in step.items()}
        warm_up_state = self._model.init_recurrent_state(
            batch_size=1,
            device=self._device,
        )
        with self._torch.inference_mode():
            self._model.forward_step(step, warm_up_state)
        self.reset_recurrent_state()

    def _predict_chronological(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        return self.predict_sequence(features)[-1]

    def predict_sequence(
        self,
        features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Predict every chronological frame without changing online GRU state."""
        tensors = []
        sequence_length = None
        for key in self.metadata.input_keys:
            if key not in features:
                raise KeyError(f"Missing tau_f sequence input {key!r}")
            tensor = self._torch.as_tensor(
                features[key],
                dtype=self._torch.float32,
                device=self._device,
            )
            expected_dim = self.metadata.input_dims[key]
            if tensor.ndim != 2 or tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"tau_f sequence input {key!r} must have shape [T, "
                    f"{expected_dim}], got {tuple(tensor.shape)}"
                )
            if sequence_length is None:
                sequence_length = tensor.shape[0]
            elif tensor.shape[0] != sequence_length:
                raise ValueError("All tau_f sequence inputs must share time length")
            tensors.append(self._normalize(key, tensor))
        sequence = self._torch.cat(tensors, dim=-1).unsqueeze(0)
        with self._torch.inference_mode():
            recurrent_output, _ = self._model.recurrent(sequence)
            prediction = self._model.head(recurrent_output)
            prediction = self._denormalize(self.metadata.output_key, prediction)
        result = prediction.squeeze(0).detach().cpu().numpy().astype(np.float64)
        if result.ndim != 2 or result.shape != (
            sequence_length,
            self.metadata.output_dim,
        ):
            raise RuntimeError(
                "tau_f sequence prediction has unexpected shape "
                f"{result.shape}"
            )
        if not np.isfinite(result).all():
            raise RuntimeError("tau_f sequence prediction contains non-finite values")
        return result.copy()

    def _normalize(self, key: str, value: Any) -> Any:
        if key not in self._normalize_keys:
            return value
        stats = self._stats[key]
        if self._normalize_mode == "gaussian":
            mean = self._stat_tensor(stats, "mean")
            std = self._stat_tensor(stats, "std")
            return (value - mean) / (std + self._eps)
        if self._normalize_mode == "limit":
            minimum = self._stat_tensor(stats, "min")
            maximum = self._stat_tensor(stats, "max")
            return 2.0 * (value - minimum) / (maximum - minimum + self._eps) - 1.0
        q01 = self._stat_tensor(stats, "q01")
        q99 = self._stat_tensor(stats, "q99")
        return self._torch.clamp(
            2.0 * (value - q01) / (q99 - q01 + self._eps) - 1.0,
            -1.0,
            1.0,
        )

    def _denormalize(self, key: str, value: Any) -> Any:
        if key not in self._normalize_keys:
            return value
        stats = self._stats[key]
        if self._normalize_mode == "gaussian":
            return value * (self._stat_tensor(stats, "std") + self._eps) + self._stat_tensor(
                stats, "mean"
            )
        if self._normalize_mode == "limit":
            minimum = self._stat_tensor(stats, "min")
            maximum = self._stat_tensor(stats, "max")
            return (value + 1.0) * (maximum - minimum + self._eps) / 2.0 + minimum
        q01 = self._stat_tensor(stats, "q01")
        q99 = self._stat_tensor(stats, "q99")
        return (value + 1.0) * (q99 - q01 + self._eps) / 2.0 + q01

    def _stat_tensor(self, stats: Any, name: str) -> Any:
        stats = _mapping(stats, "normalizer statistics")
        if name not in stats:
            raise RuntimeError(f"checkpoint normalizer is missing statistic {name!r}")
        return self._torch.as_tensor(
            stats[name],
            dtype=self._torch.float32,
            device=self._device,
        )


class OnlineTauFInference:
    def __init__(
        self,
        config: TauFInferenceConfig,
        inverse_dynamics: InverseDynamicsConfig,
        dynamics_processing: DynamicsProcessingConfig,
        robot_states: dict[str, StateParamConfig],
        *,
        predictor: TauFPredictor | None = None,
        estimator: Any | None = None,
    ) -> None:
        self.config = config
        self.mode = config.mode
        self.predictor = predictor or (
            TauFSequencePredictor(config) if config.enabled else ZeroTauFPredictor()
        )
        if not config.enabled and predictor is None:
            log.info("tau_f inference disabled; using a zero 7D tau_f prediction")
        self.estimator = estimator or PinocchioJointTorqueResidualEstimator(inverse_dynamics)
        self.state_stream = CenteredThreePointJointStateStream(
            tau_filter=_make_tau_filter(dynamics_processing, robot_states),
        )
        self.tau_id_filter = _make_state_filter(robot_states, "tau_id")
        self.tau_ext_filter = (
            OnePoleLowPass(config.tau_ext_lowpass_hz)
            if config.tau_ext_lowpass_hz is not None
            else None
        )
        self.tau_ext_gate_threshold = np.asarray(
            config.tau_ext_gate_threshold_nm,
            dtype=np.float64,
        )

    @property
    def metadata(self) -> TauFCheckpointMetadata:
        return self.predictor.metadata

    def warm_up(self) -> None:
        self.predictor.warm_up()

    def reset_recurrent_state(self) -> None:
        self.predictor.reset_recurrent_state()

    def reset_episode(self) -> None:
        """Reset every stateful component to match a fresh collection buffer."""
        self.predictor.reset_recurrent_state()
        self.state_stream.reset()
        reset_tau_id = getattr(self.tau_id_filter, "reset", None)
        if callable(reset_tau_id):
            reset_tau_id()
        if self.tau_ext_filter is not None:
            self.tau_ext_filter.reset()

    def append(
        self,
        timestamp_us: int,
        q: np.ndarray,
        tau: np.ndarray,
    ) -> OnlineTauFResult | None:
        state = self.state_stream.append(timestamp_us, q, tau)
        if state is None:
            return None
        return self._estimate_state(state)

    def estimate_centered(
        self,
        timestamp_us: int,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
    ) -> OnlineTauFResult:
        """Evaluate an already aligned center frame without differentiating again."""
        state = RealtimeJointState(
            timestamp_us=int(timestamp_us),
            sample_index=0,
            q=_finite_vector("q", q, 7),
            dq=_finite_vector("dq", dq, 7),
            ddq=_finite_vector("ddq", ddq, 7),
            tau=_finite_vector("tau", tau, 7),
        )
        return self._estimate_state(state)

    def estimate_aligned_raw(
        self,
        timestamp_us: int,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
    ) -> OnlineTauFResult:
        """Apply collection torque preprocessing to an aligned arm sample."""
        tau_value = _finite_vector("tau", tau, 7)
        if self.state_stream.tau_filter is not None:
            tau_value = self.state_stream.tau_filter.apply(
                tau_value,
                int(timestamp_us),
            )
        return self.estimate_centered(
            timestamp_us,
            q,
            dq,
            ddq,
            tau_value,
        )

    def estimate_aligned_raw_batch(
        self,
        samples: Sequence[
            tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> tuple[OnlineTauFResult, ...]:
        """Evaluate chronological aligned states with one recurrent model launch."""
        states: list[RealtimeJointState] = []
        tau_ids: list[np.ndarray] = []
        tau_f_cals: list[np.ndarray] = []
        for timestamp_us, q, dq, ddq, tau in samples:
            tau_value = _finite_vector("tau", tau, 7)
            if self.state_stream.tau_filter is not None:
                tau_value = self.state_stream.tau_filter.apply(
                    tau_value,
                    int(timestamp_us),
                )
            state = RealtimeJointState(
                timestamp_us=int(timestamp_us),
                sample_index=0,
                q=_finite_vector("q", q, 7),
                dq=_finite_vector("dq", dq, 7),
                ddq=_finite_vector("ddq", ddq, 7),
                tau=tau_value,
            )
            estimate = self.estimator.estimate(
                state.q,
                state.dq,
                state.ddq,
                state.tau,
            )
            tau_id = np.asarray(estimate.tau_id, dtype=np.float64)
            if self.tau_id_filter is not None:
                tau_id = self.tau_id_filter.apply(tau_id, state.timestamp_us)
            states.append(state)
            tau_ids.append(tau_id.copy())
            tau_f_cals.append((tau_id - state.tau).copy())
        if not states:
            return ()
        features = {
            key: np.stack([getattr(state, key) for state in states], axis=0)
            for key in ("q", "dq", "ddq", "tau")
        }
        batch_predict = getattr(self.predictor, "append_sequence_and_predict", None)
        if callable(batch_predict):
            tau_f_predictions = np.asarray(
                batch_predict(features),
                dtype=np.float64,
            )
        else:
            tau_f_predictions = np.stack(
                [
                    self.predictor.append_and_predict(
                        {key: value[index] for key, value in features.items()}
                    )
                    for index in range(len(states))
                ],
                axis=0,
            )
        if tau_f_predictions.shape != (len(states), 7):
            raise RuntimeError(
                "tau_f batch predictor must return shape "
                f"({len(states)}, 7); got {tau_f_predictions.shape}"
            )
        results = []
        for state, tau_id, tau_f_cal, prediction in zip(
            states,
            tau_ids,
            tau_f_cals,
            tau_f_predictions,
        ):
            tau_ext_raw = self._external_torque_raw(state.tau, tau_f_cal, prediction)
            tau_ext_filtered, tau_ext = self._filter_and_gate(
                tau_ext_raw,
                state.timestamp_us,
            )
            results.append(
                OnlineTauFResult(
                    timestamp_us=state.timestamp_us,
                    q=state.q.copy(),
                    dq=state.dq.copy(),
                    ddq=state.ddq.copy(),
                    tau=state.tau.copy(),
                    tau_id=tau_id.copy(),
                    tau_f_cal=tau_f_cal.copy(),
                    tau_f_pred=prediction.copy(),
                    tau_ext=tau_ext.copy(),
                    tau_bg_pred=(prediction.copy() if self.mode == "tau_bg" else None),
                    tau_ext_raw=tau_ext_raw.copy(),
                    tau_ext_filtered=tau_ext_filtered.copy(),
                )
            )
        return tuple(results)

    def _estimate_state(self, state: RealtimeJointState) -> OnlineTauFResult:
        estimate = self.estimator.estimate(state.q, state.dq, state.ddq, state.tau)
        tau_id = np.asarray(estimate.tau_id, dtype=np.float64)
        if self.tau_id_filter is not None:
            tau_id = self.tau_id_filter.apply(tau_id, state.timestamp_us)
        features = {
            "q": state.q,
            "dq": state.dq,
            "ddq": state.ddq,
            "tau": state.tau,
        }
        prediction = self.predictor.append_and_predict(features)
        tau_f_cal = tau_id - state.tau
        tau_ext_raw = self._external_torque_raw(state.tau, tau_f_cal, prediction)
        tau_ext_filtered, tau_ext = self._filter_and_gate(
            tau_ext_raw,
            state.timestamp_us,
        )
        return OnlineTauFResult(
            timestamp_us=state.timestamp_us,
            q=state.q.copy(),
            dq=state.dq.copy(),
            ddq=state.ddq.copy(),
            tau=state.tau.copy(),
            tau_id=tau_id.copy(),
            tau_f_cal=tau_f_cal.copy(),
            tau_f_pred=prediction.copy(),
            tau_ext=tau_ext.copy(),
            tau_bg_pred=(prediction.copy() if self.mode == "tau_bg" else None),
            tau_ext_raw=tau_ext_raw.copy(),
            tau_ext_filtered=tau_ext_filtered.copy(),
        )

    def _external_torque_raw(
        self,
        tau: np.ndarray,
        tau_f_cal: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        if self.mode == "tau_bg":
            return np.asarray(tau, dtype=np.float64) - prediction
        return np.asarray(tau_f_cal, dtype=np.float64) - prediction

    def _filter_and_gate(
        self,
        tau_ext_raw: np.ndarray,
        timestamp_us: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered = np.asarray(tau_ext_raw, dtype=np.float64).copy()
        if self.tau_ext_filter is not None:
            filtered = self.tau_ext_filter.apply(filtered, int(timestamp_us))
        gated = np.where(
            np.abs(filtered) >= self.tau_ext_gate_threshold,
            filtered,
            0.0,
        )
        return filtered.copy(), gated.astype(np.float64, copy=True)


def _build_checkpoint_model(
    torch: Any,
    nn: Any,
    config: Mapping[str, Any],
    input_keys,
    input_dims,
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
            else:
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

        def init_recurrent_state(self, batch_size, device=None, dtype=None):
            parameter = next(self.parameters())
            device = parameter.device if device is None else device
            dtype = parameter.dtype if dtype is None else dtype
            shape = (num_layers, int(batch_size), hidden_dim)
            hidden = parameter.new_zeros(shape, device=device, dtype=dtype)
            if architecture == "lstm":
                return hidden, torch.zeros_like(hidden)
            return hidden

        @staticmethod
        def detach_recurrent_state(recurrent_state):
            if recurrent_state is None:
                return None
            if isinstance(recurrent_state, tuple):
                return tuple(state.detach() for state in recurrent_state)
            return recurrent_state.detach()

        def forward_step(self, batch, recurrent_state=None):
            sequence = self._prepare_step_inputs(batch)
            if recurrent_state is None:
                recurrent_output, recurrent_state = self.recurrent(sequence)
            else:
                recurrent_output, recurrent_state = self.recurrent(
                    sequence,
                    recurrent_state,
                )
            return {
                "tau_f_pred": self.head(recurrent_output[:, -1]),
                "sequence_features": recurrent_output,
                "recurrent_state": recurrent_state,
            }

        @staticmethod
        def _prepare_step_inputs(batch):
            step_inputs = []
            batch_size = None
            for key in input_keys:
                if key not in batch:
                    raise KeyError(f"Missing model input {key!r} in batch")
                value = batch[key]
                expected_dim = input_dims[key]
                if value.ndim != 2 or value.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Step input {key!r} must have shape [B, {expected_dim}], "
                        f"got {tuple(value.shape)}"
                    )
                if batch_size is None:
                    batch_size = value.shape[0]
                elif value.shape[0] != batch_size:
                    raise ValueError("All tau_f step inputs must share batch size")
                step_inputs.append(value)
            return torch.cat(step_inputs, dim=-1).unsqueeze(1)

        def forward(self, sequence):
            recurrent_output, _ = self.recurrent(sequence)
            return self.head(recurrent_output[:, -1])

    return CheckpointModel()


def _make_state_filter(
    robot_states: dict[str, StateParamConfig],
    state_name: str,
) -> OnePoleLowPass | None:
    param = robot_states.get(state_name)
    if param is None or not param.lowpass:
        return None
    return OnePoleLowPass(param.lowpass_cutoff_hz, param.median_window)


def _make_tau_filter(
    processing: DynamicsProcessingConfig,
    robot_states: dict[str, StateParamConfig],
) -> OnePoleLowPass | None:
    if processing.enabled:
        return OnePoleLowPass(processing.torque_lowpass_hz, processing.torque_median_window)
    return _make_state_filter(robot_states, "torque")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"tau_f inference requires a finite {size}D {name}; got {vector}")
    return vector.copy()
