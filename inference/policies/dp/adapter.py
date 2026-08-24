"""Small adapter for standard diffusion-policy checkpoint objects."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from inference.core.contracts import ActionChunk, Observation


class DiffusionPolicyAdapter:
    """Adapt a checkpoint exposing ``predict_action`` to ``HighLevelPolicy``.

    This is the lightweight DP adapter for callers that already own their
    observation builder.  The algorithm-level :class:`DiffusionPolicy` adds
    checkpoint-derived image history and shape handling on top of this contract.
    """

    def __init__(
        self,
        model: Any,
        *,
        input_builder: Callable[[Observation], Any] | None = None,
        semantic: str = "eepose",
        frame_name: str | None = None,
        action_key: str = "action",
        device: Any | None = None,
        action_steps: int | None = None,
        step_s: float | None = None,
    ) -> None:
        self.model = model
        self.input_builder = input_builder or self._default_input_builder
        self.semantic = semantic
        self.frame_name = frame_name
        self.action_key = action_key
        self.device = device
        self.action_steps = action_steps
        self.step_s = step_s

    def start(self) -> None:
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        if self.device is not None:
            mover = getattr(self.model, "to", None)
            if callable(mover):
                mover(self.device)

    def close(self) -> None:
        return None

    def reset_episode(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: Observation) -> ActionChunk | None:
        model_input = self.input_builder(observation)
        predict_action = getattr(self.model, "predict_action", None)
        if not callable(predict_action):
            raise TypeError("diffusion policy must expose predict_action()")
        try:
            import torch

            context = torch.no_grad()
        except ImportError:  # pragma: no cover - torch is optional for tests
            context = _NullContext()
        with context:
            output = predict_action(model_input)
        values = output[self.action_key] if isinstance(output, Mapping) else output
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.ndim >= 3:
            values = values[0]
        if self.action_steps is not None:
            values = values[: int(self.action_steps)]
        return ActionChunk(
            values=values,
            semantic=self.semantic,
            frame_name=self.frame_name,
            timestamp_us=observation.timestamp_us,
            step_s=self.step_s,
        )

    @staticmethod
    def _default_input_builder(observation: Observation) -> dict[str, np.ndarray]:
        return {
            **{name: np.asarray(image).copy() for name, image in observation.images.items()},
            "q": observation.q.copy(),
            "dq": observation.dq.copy(),
            "tau_ext": observation.tau_ext.copy(),
        }


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


__all__ = ["DiffusionPolicyAdapter"]
