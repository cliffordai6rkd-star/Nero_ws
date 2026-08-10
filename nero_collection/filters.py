from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nero_collection.config import StateParamConfig


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
class DatasetFilterBank:
    state_params: dict[str, StateParamConfig]
    filters: dict[str, OnePoleLowPass] = field(default_factory=dict)

    def apply(
        self,
        dataset_name: str,
        state_name: str,
        value: np.ndarray,
        timestamp_us: int,
    ) -> np.ndarray:
        param = self.state_params.get(state_name)
        if not param or not param.lowpass:
            return value
        filt = self.filters.get(dataset_name)
        if filt is None:
            filt = OnePoleLowPass(param.lowpass_cutoff_hz, param.median_window)
            self.filters[dataset_name] = filt
        return filt.apply(value, timestamp_us)
