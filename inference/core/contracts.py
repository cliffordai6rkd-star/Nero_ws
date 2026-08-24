"""Stable data contracts shared by the modular inference components.

The legacy inference implementation passes a mixture of dictionaries and raw
arrays between the runtime and the model pipeline.  These small immutable
records make the boundaries explicit without imposing a dependency on torch,
CAN, or a particular robot SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _vector(name: str, value: Any, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _images(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, frame in value.items():
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"image {name!r} must have HxWx3 shape, got {image.shape}"
            )
        result[str(name)] = np.ascontiguousarray(image).copy()
    return result


@dataclass(frozen=True)
class Observation:
    """Canonical observation consumed by policy and world-model components."""

    timestamp_us: int
    acquired_timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    tau_ext: np.ndarray
    wrench_ext: np.ndarray
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    image_timestamps_us: Mapping[str, int] = field(default_factory=dict)
    q_cmd: np.ndarray | None = None
    wrench_to_control_rotation: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.timestamp_us) < 0:
            raise ValueError("observation timestamp_us must be non-negative")
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))
        object.__setattr__(self, "acquired_timestamp_us", int(self.acquired_timestamp_us))
        for name in ("q", "dq", "ddq", "tau", "tau_ext"):
            object.__setattr__(self, name, _vector(name, getattr(self, name), 7))
        object.__setattr__(self, "wrench_ext", _vector("wrench_ext", self.wrench_ext, 6))
        object.__setattr__(self, "images", _images(self.images))
        object.__setattr__(
            self,
            "image_timestamps_us",
            {str(key): int(value) for key, value in self.image_timestamps_us.items()},
        )
        if self.q_cmd is not None:
            object.__setattr__(self, "q_cmd", _vector("q_cmd", self.q_cmd, 7))
        if self.wrench_to_control_rotation is not None:
            rotation = np.asarray(self.wrench_to_control_rotation, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError(
                    "wrench_to_control_rotation must be a finite 3x3 matrix"
                )
            object.__setattr__(self, "wrench_to_control_rotation", rotation.copy())


@dataclass(frozen=True)
class ActionChunk:
    """A model action with explicit semantic and coordinate-frame metadata."""

    values: np.ndarray
    semantic: str
    frame_name: str | None
    timestamp_us: int
    step_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"action chunk must be [H,D], got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("action chunk must contain only finite values")
        semantic = str(self.semantic).strip().lower()
        if semantic not in {"joint", "eepose", "pose", "torque"}:
            raise ValueError(f"unsupported action semantic {self.semantic!r}")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "semantic", semantic)
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))
        if self.step_s is not None:
            step_s = float(self.step_s)
            if not np.isfinite(step_s) or step_s <= 0.0:
                raise ValueError("action step_s must be positive and finite")
            object.__setattr__(self, "step_s", step_s)

    @property
    def first(self) -> np.ndarray:
        return self.values[0].copy()


@dataclass(frozen=True)
class ControlTarget:
    """Robot-space target produced after optional world-model inference."""

    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    torque: np.ndarray | None = None
    pose: np.ndarray | None = None
    mode: str = "position"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("q", "dq", "torque"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(name, value, 7))
        if self.pose is not None:
            pose = _vector("pose", self.pose)
            if pose.shape not in {(7,), (16,)}:
                raise ValueError(f"pose must be 7D or 4x4 flattened, got {pose.shape}")
            object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "mode", str(self.mode).strip().lower())


@dataclass(frozen=True)
class InferenceCycle:
    """Diagnostics/result record emitted by :class:`InferenceBase`."""

    observation: Observation
    action: ActionChunk | None
    target: ControlTarget | None
    command: Any | None
    policy_updated: bool
    world_model_updated: bool
    timings_s: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
