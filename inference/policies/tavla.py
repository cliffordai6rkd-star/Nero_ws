"""TAVLA policy adapter for the modular inference contracts.

TAVLA has had more than one public inference entry point (for example
``predict_action`` and ``get_action``), and the official repository is not a
runtime dependency of this package.  This adapter therefore keeps the model
backend and optional processor injectable while normalising the result to the
same :class:`~inference.core.contracts.ActionChunk` used by DP/VLA policies.

The adapter intentionally does not guess a TAVLA checkpoint architecture.  A
caller can pass an already restored model, or provide ``model_loader`` to
``from_checkpoint``.  This avoids importing an optional official package at
module import time and preserves its own preprocessing/normalisation code.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from inference.core.contracts import ActionChunk, Observation


_ACTION_KEYS = (
    "actions",
    "action",
    "action_chunk",
    "action_pred",
    "action_prediction",
    "trajectory",
    "action_target",
)


class TAVLAObservationBuilder:
    """Build a dependency-free canonical input mapping for TAVLA.

    Official TAVLA processors can be supplied through ``processor`` or
    ``input_builder`` on :class:`TAVLA`.  The fallback mapping exposes both the
    canonical ``state`` field and the individual robot signals, which makes it
    useful for lightweight wrappers and deterministic tests without imposing a
    particular official repository schema.
    """

    def __init__(self, *, instruction: str | None = None) -> None:
        self.instruction = instruction

    def __call__(self, observation: Observation) -> dict[str, Any]:
        metadata = dict(observation.metadata)
        instruction = self.instruction
        if instruction is None:
            for key in ("instruction", "task", "language", "prompt"):
                value = metadata.get(key)
                if value is not None:
                    instruction = str(value)
                    break
        # Preserve image layout/dtype here.  Official processors commonly own
        # resize, normalisation and CHW conversion, so doing that twice would
        # silently change the checkpoint contract.
        images = {
            str(name): np.asarray(image).copy()
            for name, image in observation.images.items()
        }
        result: dict[str, Any] = {
            "images": images,
            "state": observation.q.copy(),
            "q": observation.q.copy(),
            "dq": observation.dq.copy(),
            "ddq": observation.ddq.copy(),
            "tau": observation.tau.copy(),
            "tau_ext": observation.tau_ext.copy(),
            "wrench_ext": observation.wrench_ext.copy(),
            "timestamp_us": int(observation.timestamp_us),
        }
        # Also expose camera names at the top level.  This mirrors the common
        # official VLA input convention while retaining the grouped ``images``
        # mapping for wrappers that prefer it.
        result.update(images)
        if instruction is not None:
            result["instruction"] = instruction
            # ``language`` is used by some official wrappers; retaining the
            # alias is harmless for callers that select a custom input builder.
            result["language"] = instruction
        if "gripper" in metadata:
            result["gripper"] = metadata["gripper"]
        if metadata:
            result["metadata"] = metadata
        return result


class TAVLA:
    """Adapt an official TAVLA model to ``HighLevelPolicy``.

    Parameters
    ----------
    model:
        Restored official model/backend.  It may expose ``predict_action``,
        ``get_action``, ``infer``, ``forward`` or be directly callable.
    processor:
        Optional official preprocessing callable.  If it has ``process`` that
        method is used; otherwise the object itself is called with the input
        mapping produced by ``input_builder``.
    input_builder:
        Callable converting :class:`Observation` into the exact input expected
        by the official model.  The default is :class:`TAVLAObservationBuilder`.
    action_semantic:
        ``joint``, ``eepose``/``pose`` or ``torque``.  This is metadata only;
        conversion to robot control remains an ``ActionResolver`` concern.
    """

    def __init__(
        self,
        model: Any,
        *,
        processor: Any | None = None,
        input_builder: Callable[[Observation], Any] | None = None,
        observation_builder: Callable[[Observation], Any] | None = None,
        semantic: str = "eepose",
        action_semantic: str | None = None,
        frame_name: str | None = None,
        action_frame_name: str | None = None,
        action_key: str | None = None,
        device: Any | None = None,
        action_steps: int | None = None,
        step_s: float | None = None,
        instruction: str | None = None,
    ) -> None:
        if model is None:
            raise ValueError("TAVLA model must not be None")
        if input_builder is not None and observation_builder is not None:
            raise ValueError("pass only one of input_builder/observation_builder")
        self.model = model
        self.processor = processor
        self.input_builder = (
            input_builder
            or observation_builder
            or TAVLAObservationBuilder(instruction=instruction)
        )
        self.semantic = str(action_semantic or semantic).strip().lower()
        self.frame_name = action_frame_name or frame_name
        self.action_key = None if action_key is None else str(action_key)
        self.device = device
        self.action_steps = None if action_steps is None else int(action_steps)
        self.step_s = None if step_s is None else float(step_s)
        self._started = False
        if self.semantic not in {"joint", "eepose", "pose", "torque"}:
            raise ValueError(
                "TAVLA action semantic must be 'joint', 'eepose', 'pose', or 'torque'"
            )
        if self.action_steps is not None and self.action_steps < 1:
            raise ValueError("action_steps must be positive")
        if self.step_s is not None and (
            not np.isfinite(self.step_s) or self.step_s <= 0.0
        ):
            raise ValueError("step_s must be positive and finite")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        model_loader: Callable[[Path], Any],
        **kwargs: Any,
    ) -> "TAVLA":
        """Restore a model through an official-code loader supplied by caller.

        ``model_loader`` receives a resolved path and may call the official
        repository's ``from_pretrained``/checkpoint restoration routine.  A
        callback is required because TAVLA is intentionally an optional
        dependency and its checkpoint layouts are repository-specific.
        """

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"TAVLA checkpoint not found: {path}")
        if not callable(model_loader):
            raise TypeError("model_loader must be callable")
        return cls(model_loader(path), **kwargs)

    def start(self) -> None:
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        mover = getattr(self.model, "to", None)
        if self.device is not None and callable(mover):
            mover(self.device)
        for component in (self.processor, self.model):
            starter = getattr(component, "start", None)
            if callable(starter):
                starter()
        self._started = True

    def close(self) -> None:
        for component in (self.processor, self.model):
            closer = getattr(component, "close", None)
            if callable(closer):
                closer()
            else:
                stopper = getattr(component, "stop", None)
                if callable(stopper):
                    stopper()
        self._started = False

    def reset_episode(self) -> None:
        for component in (self.model, self.processor):
            reset = getattr(component, "reset_episode", None)
            if not callable(reset):
                reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def predict(self, observation: Observation) -> ActionChunk | None:
        model_input = self.input_builder(observation)
        if self.processor is not None:
            process = getattr(self.processor, "process", None)
            model_input = (
                process(model_input)
                if callable(process)
                else self.processor(model_input)
            )
        model_input = self._move_to_device(model_input)
        output = self._run_model(model_input)
        values, output_metadata = self._extract_actions(output)
        if self.action_steps is not None:
            values = values[: self.action_steps]
        if values.shape[0] < 1:
            raise ValueError("TAVLA returned an empty action chunk")
        metadata = {
            "policy": "tavla",
            "algorithm": "tavla",
            **output_metadata,
        }
        return ActionChunk(
            values=values,
            semantic=self.semantic,
            frame_name=self.frame_name,
            timestamp_us=observation.timestamp_us,
            step_s=self.step_s,
            metadata=metadata,
        )

    def _run_model(self, model_input: Any) -> Any:
        methods = (
            "predict_action",
            "get_action",
            "select_action",
            "act",
            "infer",
            "predict",
        )
        try:
            import torch

            context = torch.inference_mode()
        except ImportError:  # pragma: no cover - torch is optional at import time
            context = nullcontext()
        with context:
            for name in methods:
                method = getattr(self.model, name, None)
                if callable(method):
                    return self._call_mapping_or_positional(method, model_input)
            forward = getattr(self.model, "forward", None)
            if callable(forward):
                return self._call_mapping_or_positional(forward, model_input)
            if callable(self.model):
                return self._call_mapping_or_positional(self.model, model_input)
        raise TypeError(
            "TAVLA model must expose predict_action(), get_action(), infer(), "
            "predict(), forward(), or be callable"
        )

    @staticmethod
    def _call_mapping_or_positional(fn: Callable[..., Any], value: Any) -> Any:
        # Official wrappers vary between ``fn(payload)`` and ``fn(**payload)``.
        # Use ``Signature.bind`` (which never executes ``fn``) to select the
        # calling convention.  In particular, do not use a caught TypeError
        # from the real call as a signature probe: model code can legitimately
        # raise TypeError internally, and retrying would execute inference
        # twice and potentially mutate recurrent state twice.
        if not isinstance(value, Mapping):
            return fn(value)
        try:
            import inspect

            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            # Extension/builtin callables may not expose a signature.  The
            # portable convention for those is a single positional payload;
            # callers can provide a small wrapper when keyword expansion is
            # required.
            return fn(value)

        try:
            signature.bind(**value)
        except TypeError as keyword_error:
            try:
                signature.bind(value)
            except TypeError:
                raise keyword_error
            return fn(value)
        return fn(**value)

    def _extract_actions(self, output: Any) -> tuple[np.ndarray, dict[str, Any]]:
        selected_key: str | None = None
        value = output
        if isinstance(output, (tuple, list)) and output:
            # A few official evaluators return ``(actions, auxiliary)``.  The
            # auxiliary item is intentionally ignored; action metadata remains
            # explicit in the resulting ActionChunk.
            value = output[0]
        if isinstance(output, Mapping):
            keys = (self.action_key,) if self.action_key is not None else _ACTION_KEYS
            for key in keys:
                if key is not None and key in output:
                    selected_key = key
                    value = output[key]
                    break
            if selected_key is None:
                # Some wrappers nest the official result under ``output`` or
                # ``prediction``.  Recurse once while retaining clear errors.
                for key in ("output", "prediction", "result"):
                    if key in output:
                        return self._extract_actions(output[key])
                raise KeyError(
                    "TAVLA output mapping has no action field; expected one of "
                    f"{tuple(_ACTION_KEYS)!r}"
                )
        elif hasattr(output, "actions"):
            selected_key = "actions"
            value = getattr(output, "actions")
        elif hasattr(output, "action"):
            selected_key = "action"
            value = getattr(output, "action")
        values = self._to_numpy(value)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim == 3:
            values = values[0]
        if values.ndim != 2:
            raise ValueError(
                "TAVLA must return [D], [H,D], or [B,H,D] actions; "
                f"got {values.shape}"
            )
        if values.shape[-1] < 1:
            raise ValueError("TAVLA action dimension must be positive")
        metadata = {} if selected_key is None else {"action_key": selected_key}
        return np.asarray(values, dtype=np.float64), metadata

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _move_to_device(self, value: Any) -> Any:
        if self.device is None:
            return value
        if hasattr(value, "to") and callable(value.to):
            try:
                return value.to(self.device)
            except (TypeError, RuntimeError):
                return value
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value


TAVLAInferencePolicy = TAVLA
TAVLAAdapter = TAVLA
TAVLAPolicy = TAVLA
TAVLAInference = TAVLA

__all__ = [
    "TAVLA",
    "TAVLAAdapter",
    "TAVLAInferencePolicy",
    "TAVLAPolicy",
    "TAVLAInference",
    "TAVLAObservationBuilder",
]
