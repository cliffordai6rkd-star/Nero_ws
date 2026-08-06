from __future__ import annotations

from collections import deque

import numpy as np

from inference.config import TorqueFilterConfig


class CausalTorqueCommandFilter:
    """Causal spike rejection, smoothing, and hard slew limiting for 7D torque."""

    def __init__(self, config: TorqueFilterConfig, dof: int = 7) -> None:
        self.config = config
        self.dof = int(dof)
        self._history: deque[np.ndarray] = deque(maxlen=config.median_window)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._state = None

    def apply(
        self,
        command: np.ndarray,
        *,
        dt_s: float,
        initial_tau: np.ndarray,
    ) -> np.ndarray:
        value = self._vector("command", command)
        initial = self._vector("initial_tau", initial_tau)
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("torque filter dt_s must be positive and finite")
        if not self.config.enabled:
            self._state = value.copy()
            return value.copy()

        if self._state is None:
            self._state = initial.copy()
            for _ in range(self.config.median_window - 1):
                self._history.append(initial.copy())
        self._history.append(value.copy())
        median = np.median(np.stack(tuple(self._history), axis=0), axis=0)

        smoothed = median
        cutoff = self.config.lowpass_cutoff_hz
        if cutoff is not None:
            alpha = 1.0 - np.exp(-2.0 * np.pi * float(cutoff) * dt_s)
            smoothed = self._state + alpha * (median - self._state)

        rate_limit = self.config.rate_limit_nm_s
        if rate_limit is not None:
            limit = np.asarray(rate_limit, dtype=np.float64)
            if limit.ndim == 0:
                limit = np.full(self.dof, float(limit))
            else:
                limit = limit.reshape(-1)
            maximum_delta = limit * dt_s
            smoothed = self._state + np.clip(
                smoothed - self._state,
                -maximum_delta,
                maximum_delta,
            )
        self._state = smoothed.copy()
        return smoothed

    def _vector(self, name: str, value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (self.dof,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite {self.dof}-vector")
        return result
