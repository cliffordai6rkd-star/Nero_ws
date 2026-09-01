"""Inference adapter for the diffusion-policy (DP) algorithm family.

The adapter follows the standard image diffusion-policy contract: image
observations are converted to ``[B, To, C, H, W]`` tensors and the checkpoint
returns a future action chunk.  Joint-action checkpoints use the same diffusion
sampling path as the upstream policy but skip its pose-quaternion postprocess.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from inference.core.contracts import ActionChunk, Observation


class DiffusionPolicy:
    """High-level policy adapter for checkpoint-restored DP models.

    Model shape metadata is read from the restored policy.  Task-specific
    values such as camera names, horizon and action dimension therefore remain
    in the checkpoint/configuration rather than in a task-named Python class.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: Any | None = None,
        action_steps: int | None = None,
        step_s: float | None = None,
        action_semantic: str = "joint",
        action_frame_name: str | None = None,
        image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
        strict_contract: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.image_keys = self._model_image_keys(model)
        self.n_obs_steps = self._positive_model_int(model, "n_obs_steps")
        self.horizon = self._positive_model_int(model, "horizon")
        self.n_action_steps = self._positive_model_int(model, "n_action_steps")
        self.action_dim = self._positive_model_int(model, "action_dim")
        self.action_semantic = str(action_semantic).strip().lower()
        self.action_frame_name = action_frame_name
        self._image_shapes = self._model_image_shapes(model, image_shapes)
        self.action_steps = (
            self.n_action_steps if action_steps is None else int(action_steps)
        )
        self.step_s = step_s
        self.strict_contract = bool(strict_contract)
        self._history: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.n_obs_steps) for key in self.image_keys
        }
        self._last_timestamp_us: int | None = None
        self._started = False

        if self.action_steps < 1:
            raise ValueError("action_steps must be positive")
        if self.step_s is not None and (
            not np.isfinite(float(self.step_s)) or float(self.step_s) <= 0.0
        ):
            raise ValueError("step_s must be positive and finite")
        if self.strict_contract:
            self._validate_model_contract()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda:0",
        dino_model_path: str | Path | None = None,
        use_ema: bool = True,
        sampling_method: str = "ddim",
        num_inference_steps: int = 8,
        action_steps: int | None = None,
        step_s: float | None = None,
        action_semantic: str = "joint",
        action_frame_name: str | None = None,
        image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
        strict_contract: bool = True,
    ) -> "DiffusionPolicy":
        """Restore a DP checkpoint using its embedded training config.

        Hydra/OmegaConf and the diffusion-policy package are imported lazily so
        importing ``inference.policies`` remains lightweight for tests.
        """

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if int(num_inference_steps) < 1:
            raise ValueError("num_inference_steps must be positive")
        # LeRobot exports a directory checkpoint with config/processor state
        # alongside ``model.safetensors``.  Keep this detection in the generic
        # factory so callers can use the same high-level policy entry point for
        # both legacy Hydra files and native LeRobot artifacts.
        from inference.policies.lerobotdp import is_lerobot_checkpoint

        if is_lerobot_checkpoint(checkpoint):
            from inference.policies.lerobotdp import LeRobotDiffusionPolicy

            return LeRobotDiffusionPolicy.from_pretrained(
                checkpoint,
                device=device,
                num_inference_steps=num_inference_steps,
                action_steps=action_steps,
                step_s=step_s,
                action_semantic=action_semantic,
                action_frame_name=action_frame_name,
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DP checkpoint not found: {checkpoint}")
        sampling_method = str(sampling_method).strip().lower()
        scheduler_targets = {
            "ddim": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "ddpm": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
        }
        if sampling_method not in scheduler_targets:
            raise ValueError("sampling_method must be 'ddim' or 'ddpm'")

        from inference.checkpoints import restore_checkpoint_model

        model_overrides: dict[str, Any] = {
            "noise_scheduler._target_": scheduler_targets[sampling_method],
            "num_inference_steps": int(num_inference_steps),
        }
        if dino_model_path is not None:
            dino = Path(dino_model_path).expanduser().resolve()
            if not dino.is_dir():
                raise FileNotFoundError(f"DINO model not found: {dino}")
            model_overrides["dino_model_name_or_path"] = str(dino)
        model = restore_checkpoint_model(
            checkpoint,
            device,
            use_ema=use_ema,
            kind="DP",
            model_overrides=model_overrides,
        )
        if step_s is None:
            step_s = cls._checkpoint_step_s(model)
        return cls(
            model,
            device=device,
            action_steps=action_steps,
            step_s=step_s,
            action_semantic=action_semantic,
            action_frame_name=action_frame_name,
            image_shapes=image_shapes,
            strict_contract=strict_contract,
        )

    def start(self) -> None:
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        if self.device is not None:
            mover = getattr(self.model, "to", None)
            if callable(mover):
                mover(self.device)
        self._started = True

    def close(self) -> None:
        self._started = False

    def reset_episode(self) -> None:
        for history in self._history.values():
            history.clear()
        self._last_timestamp_us = None
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: Observation) -> ActionChunk:
        """Append one aligned observation and return the next action chunk."""

        self._append_images(observation)
        model_input = self._build_model_input()
        device = self._model_device()
        model_input = {key: value.to(device) for key, value in model_input.items()}

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency is runtime-only
            raise RuntimeError("DP inference requires torch") from exc

        with torch.inference_mode():
            output = self._predict_action(model_input)
        values = output["action"] if isinstance(output, Mapping) else output
        values = self._to_numpy(values)
        if values.ndim == 3:
            values = values[0]
        if values.ndim != 2 or values.shape[-1] != self.action_dim:
            raise ValueError(
                "DP must return action shape [B,H,D] or [H,D], "
                f"got {values.shape}"
            )
        values = values[: self.action_steps]
        return ActionChunk(
            values=values,
            semantic=self.action_semantic,
            frame_name=self.action_frame_name,
            timestamp_us=observation.timestamp_us,
            step_s=self.step_s,
            metadata={
                "policy": "diffusion_policy",
                "algorithm": "dp",
                "n_obs_steps": self.n_obs_steps,
            },
        )

    def _append_images(self, observation: Observation) -> None:
        if self._last_timestamp_us == observation.timestamp_us:
            return
        missing = [key for key in self.image_keys if key not in observation.images]
        if missing:
            raise KeyError(
                "DP requires image observations "
                f"{list(self.image_keys)}, missing={missing}"
            )
        for key in self.image_keys:
            self._history[key].append(self._prepare_image(observation.images[key], key))
        self._last_timestamp_us = observation.timestamp_us

    def _build_model_input(self) -> dict[str, Any]:
        # The diffusion-policy image encoder consumes [B,To,C,H,W].  During the
        # first tick, repeat the first frame to match dataset pad_before=1.
        return {
            key: self._stack_history(history)
            for key, history in self._history.items()
        }

    def _stack_history(self, history: deque[np.ndarray]) -> Any:
        if not history:
            raise RuntimeError("DP image history is empty")
        frames = list(history)
        while len(frames) < self.n_obs_steps:
            frames.insert(0, frames[0].copy())
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency is runtime-only
            raise RuntimeError("DP inference requires torch") from exc
        return torch.from_numpy(np.stack(frames, axis=0)[None]).float()

    def _prepare_image(self, image: Any, key: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"image {key!r} must be HxWx3, got {array.shape}")
        if array.shape[-1] != 3 and array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"image {key!r} must have three channels, got {array.shape}")
        shape = self._image_shapes.get(key)
        expected_hw = None if shape is None else tuple(shape[1:])
        if expected_hw is not None and array.shape[:2] != expected_hw:
            height, width = expected_hw
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - project dependency
                raise RuntimeError(
                    f"image {key!r} has shape {array.shape[:2]}, expected {expected_hw}; "
                    "install opencv for resizing"
                ) from exc
            array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.integer) or float(np.nanmax(array)) > 1.0:
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"image {key!r} contains non-finite values")
        return np.ascontiguousarray(np.moveaxis(array, -1, 0), dtype=np.float32)

    def _predict_action(self, model_input: Mapping[str, Any]) -> Any:
        # The upstream image policy normalizes dimensions 3:7 as a quaternion.
        # That is only valid for pose actions; joint actions must be unnormalized
        # directly from the diffusion trajectory.
        if self.action_semantic != "joint":
            predict_action = getattr(self.model, "predict_action", None)
            if not callable(predict_action):
                raise TypeError("DP model must expose predict_action()")
            return predict_action(dict(model_input))

        required = (
            "_encode_observation",
            "conditional_sample",
            "normalizer",
            "horizon",
            "action_dim",
            "n_action_steps",
        )
        if all(hasattr(self.model, name) for name in required):
            condition = self.model._encode_observation(dict(model_input))
            batch_size = int(condition.shape[0])
            trajectory = self.model.conditional_sample(
                (batch_size, int(self.model.horizon), int(self.model.action_dim)),
                condition,
            )
            action_prediction = self.model.normalizer["action"].unnormalize(trajectory)
            start = int(
                getattr(
                    self.model,
                    "action_start_index",
                    int(getattr(self.model, "n_obs_steps", self.n_obs_steps)) - 1,
                )
            )
            end = start + int(self.model.n_action_steps)
            return {"action": action_prediction[:, start:end]}

        predict_action = getattr(self.model, "predict_action", None)
        if not callable(predict_action):
            raise TypeError(
                "DP model must expose the standard diffusion methods "
                "or predict_action()"
            )
        return predict_action(dict(model_input))

    def _model_device(self) -> Any:
        if self.device is not None:
            try:
                import torch

                return torch.device(self.device)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("DP inference requires torch") from exc
        try:
            import torch

            parameter = next(self.model.parameters())
            return parameter.device
        except (ImportError, AttributeError, StopIteration):
            try:
                import torch

                return torch.device("cpu")
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("DP inference requires torch") from exc

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _validate_model_contract(self) -> None:
        if not self.image_keys:
            raise ValueError("DP model must expose at least one image key")
        if self.action_steps > self.n_action_steps:
            raise ValueError(
                "action_steps cannot exceed checkpoint n_action_steps="
                f"{self.n_action_steps}"
            )
        if self.horizon < self.n_action_steps:
            raise ValueError(
                "DP checkpoint horizon must be >= n_action_steps; "
                f"got horizon={self.horizon}, n_action_steps={self.n_action_steps}"
            )
        if self.action_semantic not in {"joint", "eepose", "pose", "torque"}:
            raise ValueError(f"unsupported DP action semantic {self.action_semantic!r}")
        for key, shape in self._image_shapes.items():
            if len(shape) != 3 or shape[0] != 3 or any(int(value) < 1 for value in shape):
                raise ValueError(f"DP image shape for {key!r} must be [3,H,W], got {shape}")

    @staticmethod
    def _model_image_keys(model: Any) -> tuple[str, ...]:
        keys = getattr(model, "image_keys", None)
        if keys is None:
            keys = getattr(getattr(model, "obs_encoder", None), "rgb_keys", None)
        if not keys:
            raise ValueError("DP model must expose image_keys or obs_encoder.rgb_keys")
        return tuple(sorted(str(key) for key in keys))

    @staticmethod
    def _positive_model_int(model: Any, name: str) -> int:
        value = getattr(model, name, None)
        if value is None or int(value) < 1:
            raise ValueError(f"DP model must expose positive {name}")
        return int(value)

    @staticmethod
    def _model_image_shapes(
        model: Any,
        configured: Mapping[str, tuple[int, int, int]] | None,
    ) -> dict[str, tuple[int, int, int]]:
        shape_map = dict(
            getattr(getattr(model, "obs_encoder", None), "key_shape_map", {})
        )
        result = {}
        for key in DiffusionPolicy._model_image_keys(model):
            shape = (configured or {}).get(key, shape_map.get(key))
            if shape is not None:
                result[key] = tuple(int(value) for value in shape)
        return result

    @staticmethod
    def _checkpoint_step_s(model: Any) -> float | None:
        cfg = getattr(model, "_inference_checkpoint_config", {})
        if isinstance(cfg, Mapping):
            task = cfg.get("task", {})
            dataset = task.get("dataset", {}) if isinstance(task, Mapping) else {}
            value = dataset.get("timestamp_step_sec") if isinstance(dataset, Mapping) else None
            if value is not None:
                value = float(value)
                if np.isfinite(value) and value > 0:
                    return value
        return None


DPPolicy = DiffusionPolicy

__all__ = ["DiffusionPolicy", "DPPolicy"]
