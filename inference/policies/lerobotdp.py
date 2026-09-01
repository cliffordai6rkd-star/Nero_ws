"""LeRobot Diffusion Policy backend for the modular action-expert API.

The checkpoint produced by LeRobot is a directory (``config.json`` plus
``model.safetensors`` and processor state files), unlike the legacy Hydra
``.pt/.ckpt`` files used by the original Nero pipeline.  This module keeps the
optional LeRobot dependency behind a small adapter and exposes the same
``HighLevelPolicy`` contract as :class:`inference.policies.dp.DiffusionPolicy`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from inference.core.contracts import ActionChunk, Observation


_ACTION_KEYS = (
    "action",
    "actions",
    "action_chunk",
    "action_pred",
    "action_prediction",
    "trajectory",
)


def is_lerobot_checkpoint(path: str | Path) -> bool:
    """Return whether *path* has the standard LeRobot policy layout."""

    root = Path(path).expanduser()
    return (
        root.is_dir()
        and (root / "config.json").is_file()
        and (root / "model.safetensors").is_file()
    )


class LeRobotDiffusionPolicy:
    """Adapt a LeRobot ``DiffusionPolicy`` to ``HighLevelPolicy``.

    LeRobot's public online API is ``select_action`` and returns one action at
    a time while maintaining its own action queue.  The Nero high-level
    scheduler consumes chunks, so ``predict`` drains exactly
    ``n_action_steps`` actions from that queue.  If a backend exposes the
    optional ``predict_action_chunk``/``predict_action`` method, its chunk
    result is used directly instead.
    """

    expects_lerobot_contract = True

    def __init__(
        self,
        model: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        device: str | None = None,
        action_steps: int | None = None,
        step_s: float | None = None,
        action_semantic: str = "joint",
        action_frame_name: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.metadata = {} if metadata is None else dict(metadata)
        config = getattr(model, "config", None)
        self.image_keys = self._image_keys(config, self.metadata)
        self.image_shapes = self._image_shapes(config, self.metadata)
        self.n_obs_steps = self._positive_int(
            getattr(config, "n_obs_steps", None), self.metadata.get("n_obs_steps", 1), "n_obs_steps"
        )
        self.horizon = self._positive_int(
            getattr(config, "horizon", None), self.metadata.get("horizon", 1), "horizon"
        )
        self.n_action_steps = self._positive_int(
            getattr(config, "n_action_steps", None),
            self.metadata.get("n_action_steps", 1),
            "n_action_steps",
        )
        self.action_dim = self._positive_int(
            getattr(config, "action_dim", None),
            self.metadata.get("action_dim", 7),
            "action_dim",
        )
        self.action_steps = self.n_action_steps if action_steps is None else int(action_steps)
        self.step_s = self._step_from_metadata() if step_s is None else float(step_s)
        self.action_semantic = str(action_semantic).strip().lower()
        self.action_frame_name = action_frame_name
        # DiffusionPolicy's image-shape discovery uses obs_encoder.  Expose a
        # lightweight equivalent so the generic runtime can resolve cameras.
        self.obs_encoder = SimpleNamespace(key_shape_map=self.image_shapes)
        self._inference_checkpoint_config = self.metadata.get("checkpoint_config", {})
        self._started = False
        if self.action_steps < 1 or self.action_steps > self.n_action_steps:
            raise ValueError(
                "action_steps must be in [1, n_action_steps], got "
                f"{self.action_steps} (n_action_steps={self.n_action_steps})"
            )
        if self.action_semantic not in {"joint", "eepose", "pose", "torque"}:
            raise ValueError(f"unsupported action semantic {self.action_semantic!r}")
        if self.step_s is not None and (
            not np.isfinite(self.step_s) or self.step_s <= 0.0
        ):
            raise ValueError("step_s must be positive and finite")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda:0",
        num_inference_steps: int | None = None,
        action_steps: int | None = None,
        step_s: float | None = None,
        action_semantic: str = "joint",
        action_frame_name: str | None = None,
    ) -> "LeRobotDiffusionPolicy":
        root = Path(checkpoint_path).expanduser().resolve()
        if not is_lerobot_checkpoint(root):
            raise FileNotFoundError(
                "LeRobot DP checkpoint must be a directory containing "
                f"config.json and model.safetensors: {root}"
            )
        try:
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise RuntimeError(
                "this checkpoint uses LeRobot Diffusion Policy; install the "
                "optional dependency with `pip install lerobot==0.4.0`"
            ) from exc

        try:
            model = DiffusionPolicy.from_pretrained(str(root))
        except Exception as exc:
            raise RuntimeError(f"failed to load LeRobot DP checkpoint {root}: {exc}") from exc
        if num_inference_steps is not None:
            steps = int(num_inference_steps)
            if steps < 1:
                raise ValueError("num_inference_steps must be positive")
            for target in (
                model,
                getattr(model, "diffusion", None),
                getattr(model, "config", None),
            ):
                if target is not None and hasattr(target, "num_inference_steps"):
                    setattr(target, "num_inference_steps", steps)
        if device:
            mover = getattr(model, "to", None)
            if callable(mover):
                mover(device)
        metadata = cls._read_metadata(root)
        return cls(
            model,
            metadata=metadata,
            device=device,
            action_steps=action_steps,
            step_s=step_s,
            action_semantic=action_semantic,
            action_frame_name=action_frame_name,
        )

    def start(self) -> None:
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        mover = getattr(self.model, "to", None)
        if self.device is not None and callable(mover):
            mover(self.device)
        self._started = True

    def eval(self) -> "LeRobotDiffusionPolicy":
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        return self

    def to(self, device: Any) -> "LeRobotDiffusionPolicy":
        mover = getattr(self.model, "to", None)
        if callable(mover):
            mover(device)
        self.device = str(device)
        return self

    def parameters(self):
        parameters = getattr(self.model, "parameters", None)
        if not callable(parameters):
            return iter(())
        return parameters()

    def close(self) -> None:
        self._started = False

    def reset_episode(self) -> None:
        reset_episode = getattr(self.model, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()
            return
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: Observation) -> ActionChunk:
        model_input = self._model_input(observation)
        output = self._predict_chunk(model_input)
        values = self._extract_values(output)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim == 3:
            values = values[0]
        if values.ndim != 2 or values.shape[-1] != self.action_dim:
            raise ValueError(
                "LeRobot DP must return action shape [H,D] or [B,H,D], got "
                f"{values.shape}"
            )
        values = values[: self.action_steps]
        if values.shape[0] < 1:
            raise ValueError("LeRobot DP returned an empty action chunk")
        return ActionChunk(
            values=values,
            semantic=self.action_semantic,
            frame_name=self.action_frame_name,
            timestamp_us=observation.timestamp_us,
            step_s=self.step_s,
            metadata={
                "policy": "lerobot_diffusion_policy",
                "algorithm": "diffusion",
                "n_obs_steps": self.n_obs_steps,
                "horizon": self.horizon,
                "n_action_steps": self.n_action_steps,
                "input_keys": tuple(self._input_keys()),
            },
        )

    def predict_action(self, model_input: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compatibility entry point used by legacy DP wrappers.

        The preferred path is :meth:`predict`, which receives a canonical
        ``Observation`` and therefore always supplies ``observation.state``.
        This method accepts an already-built LeRobot mapping for callers that
        use the low-level policy adapter.
        """

        normalized_input = self._normalize_model_input(model_input)
        output = self._predict_chunk(normalized_input)
        return {"action": output}

    def _predict_chunk(self, model_input: Mapping[str, Any]) -> Any:
        for name in ("predict_action_chunk", "predict_action"):
            method = getattr(self.model, name, None)
            if callable(method):
                return self._call_model(method, model_input)

        select_action = getattr(self.model, "select_action", None)
        if not callable(select_action):
            raise TypeError(
                "LeRobot DP model must expose predict_action_chunk(), "
                "predict_action(), or select_action()"
            )
        actions = []
        for _ in range(self.action_steps):
            value = self._call_model(select_action, model_input)
            array = self._extract_values(value)
            if array.ndim == 3 or (array.ndim == 2 and array.shape[0] > 1):
                # Some wrappers return the complete queue on the first call.
                return value
            if array.ndim == 2:
                array = array[0]
            actions.append(array.reshape(-1))
        return np.stack(actions, axis=0)

    @staticmethod
    def _call_model(method: Any, model_input: Mapping[str, Any]) -> Any:
        import inspect

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(dict(model_input))
        try:
            signature.bind(dict(model_input))
        except TypeError as positional_error:
            try:
                signature.bind(**model_input)
            except TypeError:
                raise positional_error
            return method(**dict(model_input))
        return method(dict(model_input))

    def _model_input(self, observation: Observation) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("LeRobot DP inference requires torch") from exc
        result: dict[str, Any] = {
            "observation.state": torch.from_numpy(
                np.asarray(observation.q, dtype=np.float32).copy()
            )[None]
        }
        for key in self.image_keys:
            if key not in observation.images:
                raise KeyError(
                    f"LeRobot DP requires image observations {list(self.image_keys)}, "
                    f"missing={key!r}"
                )
            image = np.asarray(observation.images[key])
            if image.ndim != 3:
                raise ValueError(f"image {key!r} must be HxWx3 or 3xHxW")
            if image.shape[0] == 3 and image.shape[-1] != 3:
                chw = image
            elif image.shape[-1] == 3:
                chw = np.moveaxis(image, -1, 0)
            else:
                raise ValueError(f"image {key!r} must have three channels, got {image.shape}")
            expected = self.image_shapes.get(key)
            if expected is not None and tuple(chw.shape) != tuple(expected):
                try:
                    import cv2
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("opencv is required to resize DP images") from exc
                chw = np.moveaxis(
                    cv2.resize(
                        np.moveaxis(chw, 0, -1),
                        (int(expected[2]), int(expected[1])),
                        interpolation=cv2.INTER_AREA,
                    ),
                    -1,
                    0,
                )
            chw = np.asarray(chw, dtype=np.float32)
            if np.nanmax(chw, initial=0.0) > 1.0:
                chw /= 255.0
            if not np.all(np.isfinite(chw)):
                raise ValueError(f"image {key!r} contains non-finite values")
            # ``select_action`` consumes a one-sample batch.  It owns the
            # temporal queue for ``n_obs_steps`` and therefore receives one
            # current frame here, not a pre-stacked history.
            result[f"observation.images.{key}"] = torch.from_numpy(
                np.ascontiguousarray(chw)
            )[None]
        return result

    def _normalize_model_input(self, model_input: Mapping[str, Any]) -> dict[str, Any]:
        """Accept both canonical LeRobot keys and legacy bare camera names."""

        result = dict(model_input)
        if "observation.state" not in result:
            for alias in ("state", "q"):
                if alias in result:
                    result["observation.state"] = result[alias]
                    break
        for key in self.image_keys:
            canonical = f"observation.images.{key}"
            if canonical not in result and key in result:
                result[canonical] = result[key]
        required = set(self._input_keys())
        missing = sorted(required - set(result))
        if missing:
            raise KeyError(
                "LeRobot DP input is missing checkpoint features: "
                f"{missing}"
            )
        return {key: result[key] for key in self._input_keys()}

    def _input_keys(self) -> tuple[str, ...]:
        return ("observation.state", *(f"observation.images.{key}" for key in self.image_keys))

    def _step_from_metadata(self) -> float | None:
        train = self.metadata.get("train_config", {})
        fps = train.get("fps") if isinstance(train, Mapping) else None
        if fps is None:
            dataset = train.get("dataset", {}) if isinstance(train, Mapping) else {}
            fps = dataset.get("fps") if isinstance(dataset, Mapping) else None
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return None
        return 1.0 / fps if np.isfinite(fps) and fps > 0.0 else None

    @staticmethod
    def _extract_values(output: Any) -> np.ndarray:
        if isinstance(output, Mapping):
            for key in _ACTION_KEYS:
                if key in output:
                    output = output[key]
                    break
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float64)
        return array

    @staticmethod
    def _positive_int(primary: Any, fallback: Any, name: str) -> int:
        value = fallback if primary is None else primary
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LeRobot checkpoint must declare positive {name}") from exc
        if value < 1:
            raise ValueError(f"LeRobot checkpoint must declare positive {name}")
        return value

    @staticmethod
    def _image_keys(config: Any, metadata: Mapping[str, Any]) -> tuple[str, ...]:
        features = getattr(config, "input_features", None)
        if not isinstance(features, Mapping):
            features = metadata.get("input_features", {})
        keys = []
        for raw_key, feature in features.items():
            key = str(raw_key)
            feature_type = getattr(feature, "type", None)
            if feature_type is None and isinstance(feature, Mapping):
                feature_type = feature.get("type")
            if key.startswith("observation.images.") or str(feature_type).upper() == "VISUAL":
                keys.append(key.removeprefix("observation.images."))
        if not keys:
            raise ValueError("LeRobot DP checkpoint declares no visual input features")
        return tuple(sorted(dict.fromkeys(keys)))

    @staticmethod
    def _image_shapes(config: Any, metadata: Mapping[str, Any]) -> dict[str, tuple[int, int, int]]:
        features = getattr(config, "input_features", None)
        if not isinstance(features, Mapping):
            features = metadata.get("input_features", {})
        result = {}
        for raw_key, feature in features.items():
            key = str(raw_key)
            if not key.startswith("observation.images."):
                continue
            shape = getattr(feature, "shape", None)
            if shape is None and isinstance(feature, Mapping):
                shape = feature.get("shape")
            if shape is not None:
                result[key.removeprefix("observation.images.")] = tuple(int(v) for v in shape)
        return result

    @staticmethod
    def _read_metadata(root: Path) -> dict[str, Any]:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        train_path = root / "train_config.json"
        train = (
            json.loads(train_path.read_text(encoding="utf-8"))
            if train_path.is_file()
            else {}
        )
        return {
            **config,
            "input_features": config.get("input_features", {}),
            "train_config": train,
            "checkpoint_config": {
                "type": config.get("type"),
                "input_features": config.get("input_features", {}),
                "output_features": config.get("output_features", {}),
                "horizon": config.get("horizon"),
                "n_obs_steps": config.get("n_obs_steps"),
                "n_action_steps": config.get("n_action_steps"),
                "task": {"dataset": {"timestamp_step_sec": None}},
            },
        }


LeRobotDP = LeRobotDiffusionPolicy

__all__ = [
    "LeRobotDiffusionPolicy",
    "LeRobotDP",
    "is_lerobot_checkpoint",
]
