from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def _expand_axis_parameter(
    value: Any,
    size: int,
    name: str,
    cast: type[int] | type[float],
) -> np.ndarray:
    """Expand a scalar or validate a per-axis parameter for one feature vector."""

    if np.isscalar(value):
        values = [value] * size
    else:
        values = list(value)
        if len(values) != size:
            raise ValueError(
                f"{name} must be a scalar or contain exactly {size} values"
            )
    result: list[int | float] = []
    for item in values:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} values must be numeric")
        try:
            converted = cast(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values must be numeric") from exc
        if cast is int and float(item) != converted:
            raise ValueError(f"{name} values must be integers")
        result.append(converted)
    return np.asarray(result, dtype=np.int64 if cast is int else np.float64)


def _validate_filter_parameter(value: Any, name: str, cast: type[int] | type[float]) -> None:
    """Validate scalar/list values before the feature dimension is known."""

    if np.isscalar(value):
        values = [value]
    else:
        values = list(value)
        if not values:
            raise ValueError(f"{name} must not be empty")
    _expand_axis_parameter(values if len(values) > 1 else values[0], len(values), name, cast)


def _filter_parameter_values(value: Any) -> list[Any]:
    return [value] if np.isscalar(value) else list(value)


@dataclass
class CausalTrailingMedian:
    """Causal trailing median with first-sample padding at startup."""

    window: int
    history: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("median window must be a positive odd integer")

    def apply(
        self,
        value: np.ndarray,
        timestamp_us: int | None = None,
    ) -> np.ndarray:
        del timestamp_us
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("median filter inputs must be finite")
        if self.history and value.shape != self.history[0].shape:
            raise ValueError(
                f"filter shape changed from {self.history[0].shape} to {value.shape}"
            )
        self.history.append(value.copy())
        while len(self.history) > self.window:
            self.history.popleft()
        samples = list(self.history)
        samples = [samples[0]] * (self.window - len(samples)) + samples
        return np.median(np.stack(samples, axis=0), axis=0)

    def reset(self) -> None:
        self.history.clear()


@dataclass
class CausalTrailingMovingAverage:
    """Causal trailing boxcar average with first-sample startup padding."""

    window: int
    history: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("moving-average window must be positive")

    def apply(
        self,
        value: np.ndarray,
        timestamp_us: int | None = None,
    ) -> np.ndarray:
        del timestamp_us
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("moving-average filter inputs must be finite")
        if self.history and value.shape != self.history[0].shape:
            raise ValueError(
                f"filter shape changed from {self.history[0].shape} to {value.shape}"
            )
        self.history.append(value.copy())
        while len(self.history) > self.window:
            self.history.popleft()
        samples = list(self.history)
        samples = [samples[0]] * (self.window - len(samples)) + samples
        return np.mean(np.stack(samples, axis=0), axis=0)

    def reset(self) -> None:
        self.history.clear()


@dataclass
class CausalTrailingHampel:
    """Causal trailing Hampel outlier replacement using median absolute deviation."""

    window: int
    n_sigma: float = 3.0
    history: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("Hampel window must be a positive odd integer")
        if not np.isfinite(self.n_sigma) or self.n_sigma <= 0.0:
            raise ValueError("Hampel n_sigma must be positive and finite")

    def apply(
        self,
        value: np.ndarray,
        timestamp_us: int | None = None,
    ) -> np.ndarray:
        del timestamp_us
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("Hampel filter inputs must be finite")
        if self.history and value.shape != self.history[0].shape:
            raise ValueError(
                f"filter shape changed from {self.history[0].shape} to {value.shape}"
            )
        self.history.append(value.copy())
        while len(self.history) > self.window:
            self.history.popleft()
        if len(self.history) < 3:
            return value.copy()
        samples = np.stack(tuple(self.history), axis=0)
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        robust_sigma = 1.4826 * mad
        is_outlier = np.abs(value - median) > self.n_sigma * robust_sigma
        return np.where(is_outlier, median, value)

    def reset(self) -> None:
        self.history.clear()


@dataclass
class ButterworthLowPass:
    """Fixed-rate causal Butterworth SOS filter with steady-state startup."""

    cutoff_hz: float
    sample_rate_hz: float = 100.0
    order: int = 4
    state: np.ndarray | None = None
    previous_timestamp_us: int | None = None
    sos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if (
            not np.isfinite(self.cutoff_hz)
            or self.cutoff_hz <= 0.0
            or self.cutoff_hz >= 0.5 * self.sample_rate_hz
        ):
            raise ValueError("cutoff_hz must be positive and below Nyquist")
        if self.order < 1:
            raise ValueError("Butterworth order must be positive")
        self.sos = butter(
            self.order,
            self.cutoff_hz,
            btype="lowpass",
            fs=self.sample_rate_hz,
            output="sos",
        )

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        timestamp_us = int(timestamp_us)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("Butterworth filter expects one finite feature vector")
        if (
            self.previous_timestamp_us is not None
            and timestamp_us <= self.previous_timestamp_us
        ):
            raise ValueError(
                "filter timestamps must be strictly increasing: "
                f"{timestamp_us} <= {self.previous_timestamp_us}"
            )
        if self.state is None:
            self.state = (
                sosfilt_zi(self.sos)[:, :, np.newaxis]
                * value[np.newaxis, np.newaxis, :]
            )
        elif self.state.shape[2:] != value.shape:
            raise ValueError(
                f"filter shape changed from {self.state.shape[2:]} to {value.shape}"
            )
        output, self.state = sosfilt(
            self.sos,
            value[np.newaxis, :],
            axis=0,
            zi=self.state,
        )
        self.previous_timestamp_us = timestamp_us
        return output[0].copy()

    def reset(self) -> None:
        self.state = None
        self.previous_timestamp_us = None


@dataclass
class VariableStepButterworthLowPass:
    """Causal second-order Butterworth filter for irregular timestamps.

    The continuous Butterworth state equation is integrated with a trapezoidal
    (Tustin) step using each measured timestamp interval. This keeps one
    continuous filter state without assuming a fixed source sample rate.
    """

    cutoff_hz: float
    state: np.ndarray | None = None
    previous_value: np.ndarray | None = None
    previous_timestamp_us: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive and finite")
        omega = 2.0 * np.pi * float(self.cutoff_hz)
        self._a = np.asarray(
            [[0.0, 1.0], [-omega * omega, -np.sqrt(2.0) * omega]],
            dtype=np.float64,
        )
        self._b = np.asarray([0.0, omega * omega], dtype=np.float64)
        self._identity = np.eye(2, dtype=np.float64)

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        timestamp_us = int(timestamp_us)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError(
                "variable-step Butterworth filter expects one finite feature vector"
            )
        if self.state is None:
            self.state = np.stack(
                (value.copy(), np.zeros_like(value)),
                axis=0,
            )
            self.previous_value = value.copy()
            self.previous_timestamp_us = timestamp_us
            return value.copy()

        assert self.previous_value is not None
        assert self.previous_timestamp_us is not None
        if value.shape != self.previous_value.shape:
            raise ValueError(
                f"filter shape changed from {self.previous_value.shape} to {value.shape}"
            )
        dt = (timestamp_us - self.previous_timestamp_us) * 1.0e-6
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "filter timestamps must be strictly increasing: "
                f"{timestamp_us} <= {self.previous_timestamp_us}"
            )

        left = self._identity - 0.5 * dt * self._a
        right = (
            (self._identity + 0.5 * dt * self._a) @ self.state
            + 0.5
            * dt
            * self._b[:, np.newaxis]
            * (self.previous_value + value)[np.newaxis, :]
        )
        self.state = np.linalg.solve(left, right)
        self.previous_value = value.copy()
        self.previous_timestamp_us = timestamp_us
        return self.state[0].copy()

    def reset(self) -> None:
        self.state = None
        self.previous_value = None
        self.previous_timestamp_us = None


@dataclass
class CausalHampelButterworth:
    """Hampel rejection followed by Butterworth, optionally per feature axis."""

    window: int | Sequence[int] = 5
    n_sigma: float = 3.0
    cutoff_hz: float | Sequence[float] = 8.0
    sample_rate_hz: float = 100.0
    order: int = 4
    hampel: list[CausalTrailingHampel] | None = field(init=False, default=None)
    lowpass: list[ButterworthLowPass] | None = field(init=False, default=None)
    _window_values: np.ndarray | None = field(init=False, default=None, repr=False)
    _cutoff_values: np.ndarray | None = field(init=False, default=None, repr=False)
    _feature_shape: tuple[int, ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_filter_parameter(self.window, "Hampel window", int)
        _validate_filter_parameter(self.cutoff_hz, "Butterworth cutoff_hz", float)
        window_values = _filter_parameter_values(self.window)
        cutoff_values = _filter_parameter_values(self.cutoff_hz)
        if any(int(window) < 1 or int(window) % 2 == 0 for window in window_values):
            raise ValueError("Hampel window must be a positive odd integer")
        if any(
            not np.isfinite(float(cutoff))
            or float(cutoff) <= 0.0
            or float(cutoff) >= 0.5 * self.sample_rate_hz
            for cutoff in cutoff_values
        ):
            raise ValueError("cutoff_hz must be positive and below Nyquist")
        if not np.isfinite(self.n_sigma) or self.n_sigma <= 0.0:
            raise ValueError("Hampel n_sigma must be positive and finite")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if self.order < 1:
            raise ValueError("Butterworth order must be positive")

    def _initialize(self, size: int) -> None:
        self._window_values = _expand_axis_parameter(
            self.window, size, "Hampel window", int
        )
        self._cutoff_values = _expand_axis_parameter(
            self.cutoff_hz, size, "Butterworth cutoff_hz", float
        )
        if np.any(self._window_values < 1) or np.any(self._window_values % 2 == 0):
            raise ValueError("Hampel window must be a positive odd integer")
        if np.any(
            ~np.isfinite(self._cutoff_values)
            | (self._cutoff_values <= 0.0)
            | (self._cutoff_values >= 0.5 * self.sample_rate_hz)
        ):
            raise ValueError("cutoff_hz must be positive and below Nyquist")
        self.hampel = [
            CausalTrailingHampel(int(window), self.n_sigma)
            for window in self._window_values
        ]
        self.lowpass = [
            ButterworthLowPass(
                cutoff_hz=float(cutoff),
                sample_rate_hz=self.sample_rate_hz,
                order=self.order,
            )
            for cutoff in self._cutoff_values
        ]
        self._feature_shape = (size,)

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("Hampel/Butterworth filter expects one finite feature vector")
        if self._feature_shape is None:
            self._initialize(value.size)
        elif value.shape != self._feature_shape:
            raise ValueError(
                f"filter shape changed from {self._feature_shape} to {value.shape}"
            )
        assert self.hampel is not None and self.lowpass is not None
        hampel_value = np.asarray(
            [
                stage.apply(np.asarray([value[index]]), timestamp_us)[0]
                for index, stage in enumerate(self.hampel)
            ],
            dtype=np.float64,
        )
        return np.asarray(
            [
                stage.apply(np.asarray([hampel_value[index]]), timestamp_us)[0]
                for index, stage in enumerate(self.lowpass)
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self.hampel = None
        self.lowpass = None
        self._window_values = None
        self._cutoff_values = None
        self._feature_shape = None


@dataclass
class OnePoleLowPass:
    cutoff_hz: float
    median_window: int = 1
    state: np.ndarray | None = None
    previous_timestamp_us: int | None = None
    history: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not np.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive and finite")
        if self.median_window < 1 or self.median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        timestamp_us = int(timestamp_us)
        if self.state is not None and value.shape != self.state.shape:
            raise ValueError(f"filter shape changed from {self.state.shape} to {value.shape}")
        self.history.append(value.copy())
        while len(self.history) > self.median_window:
            self.history.popleft()
        samples = list(self.history)
        if samples:
            samples = [samples[0]] * (self.median_window - len(samples)) + samples
        median_value = np.median(np.stack(samples, axis=0), axis=0)
        if self.state is None:
            self.state = median_value.copy()
            self.previous_timestamp_us = timestamp_us
            return self.state.copy()
        assert self.previous_timestamp_us is not None
        dt = (timestamp_us - self.previous_timestamp_us) * 1e-6
        if dt <= 0:
            raise ValueError(
                f"filter timestamps must be strictly increasing: "
                f"{timestamp_us} <= {self.previous_timestamp_us}"
            )
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        self.state = alpha * median_value + (1.0 - alpha) * self.state
        self.previous_timestamp_us = timestamp_us
        return self.state.copy()

    def reset(self) -> None:
        self.state = None
        self.previous_timestamp_us = None
        self.history.clear()


@dataclass
class CausalWindowLowPass:
    """Causal window filter followed by one one-pole low-pass stage.

    ``moving_average`` matches the PINN rollout visualizer: the startup window
    is padded with the first real sample, then a trailing boxcar average is
    passed through a one-pole IIR. ``median`` uses the same startup convention
    through :class:`OnePoleLowPass`. ``window`` and ``cutoff_hz`` may each be a
    scalar or a sequence matching the feature vector length.
    """

    mode: str
    window: int | Sequence[int]
    cutoff_hz: float | Sequence[float]
    history: deque[np.ndarray] = field(default_factory=deque)
    lowpass: list[OnePoleLowPass] | None = field(init=False, default=None)
    _window_values: np.ndarray | None = field(init=False, default=None, repr=False)
    _cutoff_values: np.ndarray | None = field(init=False, default=None, repr=False)
    _feature_shape: tuple[int, ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"moving_average", "median"}:
            raise ValueError("filter mode must be moving_average or median")
        _validate_filter_parameter(self.window, "filter window", int)
        _validate_filter_parameter(self.cutoff_hz, "cutoff_hz", float)
        window_values = _filter_parameter_values(self.window)
        cutoff_values = _filter_parameter_values(self.cutoff_hz)
        if any(int(window) < 1 for window in window_values):
            raise ValueError("filter window must be positive")
        if self.mode == "median" and any(int(window) % 2 == 0 for window in window_values):
            raise ValueError("median filter window must be odd")
        if any(
            not np.isfinite(float(cutoff)) or float(cutoff) <= 0.0
            for cutoff in cutoff_values
        ):
            raise ValueError("cutoff_hz must be positive and finite")

    def _initialize(self, size: int) -> None:
        self._window_values = _expand_axis_parameter(
            self.window, size, "filter window", int
        )
        self._cutoff_values = _expand_axis_parameter(
            self.cutoff_hz, size, "cutoff_hz", float
        )
        if np.any(self._window_values < 1):
            raise ValueError("filter window must be positive")
        if self.mode == "median" and np.any(self._window_values % 2 == 0):
            raise ValueError("median filter window must be odd")
        if np.any(~np.isfinite(self._cutoff_values) | (self._cutoff_values <= 0.0)):
            raise ValueError("cutoff_hz must be positive and finite")
        median_windows = (
            self._window_values
            if self.mode == "median"
            else np.ones(size, dtype=np.int64)
        )
        self.lowpass = [
            OnePoleLowPass(float(cutoff), int(median_window))
            for cutoff, median_window in zip(self._cutoff_values, median_windows)
        ]
        self._feature_shape = (size,)

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("filter expects one finite feature vector")
        if self._feature_shape is None:
            self._initialize(value.size)
        elif value.shape != self._feature_shape:
            raise ValueError(
                f"filter shape changed from {self._feature_shape} to {value.shape}"
            )
        assert self._window_values is not None
        assert self.lowpass is not None
        if self.mode == "median":
            return np.asarray(
                [
                    stage.apply(np.asarray([value[index]]), timestamp_us)[0]
                    for index, stage in enumerate(self.lowpass)
                ],
                dtype=np.float64,
            )

        self.history.append(value.copy())
        max_window = int(np.max(self._window_values))
        while len(self.history) > max_window:
            self.history.popleft()
        history = list(self.history)
        averaged = np.empty_like(value)
        for index, window in enumerate(self._window_values):
            samples = [item[index] for item in history[-int(window) :]]
            samples = [samples[0]] * (int(window) - len(samples)) + samples
            averaged[index] = float(np.mean(samples))
        return np.asarray(
            [
                stage.apply(np.asarray([averaged[index]]), timestamp_us)[0]
                for index, stage in enumerate(self.lowpass)
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self.history.clear()
        self.lowpass = None
        self._window_values = None
        self._cutoff_values = None
        self._feature_shape = None


@dataclass
class CausalFilterPipeline:
    """Ordered, history-only operations restored from a checkpoint."""

    operations: Sequence[Mapping[str, Any]]
    stages: list[Any] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for operation in self.operations:
            if not isinstance(operation, Mapping):
                raise ValueError("causal filter operations must be mappings")
            operation_type = str(operation.get("type", "")).strip().lower()
            if operation_type == "median":
                self.stages.append(CausalTrailingMedian(int(operation["window"])))
            elif operation_type == "moving_average":
                self.stages.append(
                    CausalTrailingMovingAverage(int(operation["window"]))
                )
            elif operation_type == "lowpass":
                self.stages.append(OnePoleLowPass(float(operation["cutoff_hz"]), 1))
            else:
                raise ValueError(
                    "checkpoint causal filter type must be median, "
                    f"moving_average, or lowpass; got {operation_type!r}"
                )

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        output = np.asarray(value, dtype=np.float64).copy()
        for stage in self.stages:
            output = stage.apply(output, timestamp_us)
        return output

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()
