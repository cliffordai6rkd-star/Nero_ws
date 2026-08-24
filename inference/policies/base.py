"""Model adapters used by the generic inference orchestrator."""

from __future__ import annotations

from typing import Callable

import numpy as np

from inference.core.contracts import ActionChunk, Observation


class CallablePolicy:
    """Small policy adapter useful for VLA prototypes and tests."""

    def __init__(
        self,
        predict_fn: Callable[[Observation], ActionChunk | np.ndarray | None],
        *,
        semantic: str = "eepose",
        frame_name: str | None = None,
        step_s: float | None = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.semantic = semantic
        self.frame_name = frame_name
        self.step_s = step_s

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reset_episode(self) -> None:
        return None

    def predict(self, observation: Observation) -> ActionChunk | None:
        result = self.predict_fn(observation)
        if result is None or isinstance(result, ActionChunk):
            return result
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        values = np.asarray(result)
        if values.ndim >= 3:
            values = values[0]
        return ActionChunk(
            values=values,
            semantic=self.semantic,
            frame_name=self.frame_name,
            timestamp_us=observation.timestamp_us,
            step_s=self.step_s,
        )

# Compatibility import: the implementation lives with the DP algorithm.
from inference.policies.dp.adapter import DiffusionPolicyAdapter


__all__ = ["CallablePolicy", "DiffusionPolicyAdapter"]
