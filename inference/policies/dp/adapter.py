"""Small adapter for standard diffusion-policy checkpoint objects."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from inference.core.contracts import ActionChunk, Observation


def predict_diffusion_action(
    model: Any,
    model_input: Mapping[str, Any],
    *,
    action_type: str,
) -> Any:
    """Run a DP checkpoint while preserving its action semantic.

    The upstream pose policy normalizes dimensions 3:7 as a quaternion.  That
    operation is invalid for absolute joint actions, so joint checkpoints use
    the lower-level diffusion sampling path and only unnormalize the result.
    """
    action_type = str(action_type).strip().lower()
    if action_type != "joint":
        predict_action = getattr(model, "predict_action", None)
        if not callable(predict_action):
            raise TypeError("diffusion policy must expose predict_action()")
        return predict_action(dict(model_input))

    required = (
        "_encode_observation",
        "conditional_sample",
        "normalizer",
        "horizon",
        "action_dim",
        "n_action_steps",
    )
    if all(hasattr(model, name) for name in required):
        condition = model._encode_observation(dict(model_input))
        batch_size = int(condition.shape[0])
        horizon = int(model.horizon)
        action_dim = int(model.action_dim)
        if condition.ndim >= 2 and action_dim == 7 and horizon >= 1:
            trajectory = model.conditional_sample(
                (batch_size, horizon, action_dim), condition
            )
            action_prediction = model.normalizer["action"].unnormalize(trajectory)
            start = int(
                getattr(
                    model,
                    "action_start_index",
                    int(getattr(model, "n_obs_steps", 1)) - 1,
                )
            )
            end = start + int(model.n_action_steps)
            action = action_prediction[:, start:end]
            if action.ndim != 3 or action.shape[-1] != 7 or action.shape[1] < 1:
                return model.predict_action(dict(model_input))
            return {
                "action": action,
                "action_pred": action_prediction,
                "model_action_pred": action_prediction,
                "action_target": action.mean(dim=1),
            }

    predict_action = getattr(model, "predict_action", None)
    if not callable(predict_action):
        raise TypeError(
            "diffusion policy must expose the standard diffusion methods "
            "or predict_action()"
        )
    return predict_action(dict(model_input))


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
        output = self.predict_raw(model_input)
        if isinstance(output, Mapping):
            values = next(
                (
                    output[key]
                    for key in (
                        self.action_key,
                        "action",
                        "actions",
                        "action_chunk",
                        "action_pred",
                        "action_prediction",
                        "action_target",
                    )
                    if key is not None and key in output
                ),
                output,
            )
        else:
            values = output
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

    def predict_raw(self, model_input: Mapping[str, Any]) -> Any:
        """Return the checkpoint's native mapping for legacy result fields."""
        try:
            import torch

            context = torch.no_grad()
        except ImportError:  # pragma: no cover - torch is optional for tests
            context = _NullContext()
        with context:
            return predict_diffusion_action(
                self.model,
                model_input,
                action_type=self.semantic,
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
