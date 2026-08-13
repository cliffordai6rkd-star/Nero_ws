from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

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
class CausalHampelButterworth:
    """Hampel outlier rejection followed once by a Butterworth low-pass."""

    window: int = 5
    n_sigma: float = 3.0
    cutoff_hz: float = 8.0
    sample_rate_hz: float = 100.0
    order: int = 4
    hampel: CausalTrailingHampel = field(init=False)
    lowpass: ButterworthLowPass = field(init=False)

    def __post_init__(self) -> None:
        self.hampel = CausalTrailingHampel(self.window, self.n_sigma)
        self.lowpass = ButterworthLowPass(
            cutoff_hz=self.cutoff_hz,
            sample_rate_hz=self.sample_rate_hz,
            order=self.order,
        )

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        return self.lowpass.apply(
            self.hampel.apply(value, timestamp_us),
            timestamp_us,
        )

    def reset(self) -> None:
        self.hampel.reset()
        self.lowpass.reset()


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
    through :class:`OnePoleLowPass`.
    """

    mode: str
    window: int
    cutoff_hz: float
    history: deque[np.ndarray] = field(default_factory=deque)
    lowpass: OnePoleLowPass = field(init=False)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"moving_average", "median"}:
            raise ValueError("filter mode must be moving_average or median")
        if self.window < 1:
            raise ValueError("filter window must be positive")
        if self.mode == "median" and self.window % 2 == 0:
            raise ValueError("median filter window must be odd")
        median_window = self.window if self.mode == "median" else 1
        self.lowpass = OnePoleLowPass(self.cutoff_hz, median_window)

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("filter inputs must be finite")
        if self.mode == "median":
            return self.lowpass.apply(value, timestamp_us)

        if self.history and value.shape != self.history[0].shape:
            raise ValueError(
                f"filter shape changed from {self.history[0].shape} to {value.shape}"
            )
        self.history.append(value.copy())
        while len(self.history) > self.window:
            self.history.popleft()
        samples = list(self.history)
        samples = [samples[0]] * (self.window - len(samples)) + samples
        averaged = np.mean(np.stack(samples, axis=0), axis=0)
        return self.lowpass.apply(averaged, timestamp_us)

    def reset(self) -> None:
        self.history.clear()
        self.lowpass.reset()


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
