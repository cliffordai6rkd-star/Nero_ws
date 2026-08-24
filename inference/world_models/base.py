"""World-model adapters and the explicit no-op WM implementation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from inference.core.base import NullWorldModel
from inference.core.contracts import ActionChunk, ControlTarget, Observation


class CallableWorldModel:
    """Adapt an experimental WM function without coupling it to the runtime."""

    def __init__(
        self,
        infer_fn: Callable[
            [Observation, ActionChunk | None],
            ControlTarget | np.ndarray | None,
        ],
    ) -> None:
        self.infer_fn = infer_fn

    def reset_episode(self) -> None:
        return None

    def infer(
        self,
        observation: Observation,
        action: ActionChunk | None,
    ) -> ControlTarget | None:
        result = self.infer_fn(observation, action)
        if result is None or isinstance(result, ControlTarget):
            return result
        return ControlTarget(
            q=np.asarray(result, dtype=np.float64),
            mode="position",
            metadata={"source": "callable_world_model"},
        )


__all__ = ["CallableWorldModel", "NullWorldModel"]

