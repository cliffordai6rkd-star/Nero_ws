"""Action-to-control-target adapters for the modular inference path."""

from __future__ import annotations

from typing import Callable

import numpy as np

from inference.core.contracts import ActionChunk, ControlTarget, Observation


class CallableActionResolver:
    """Adapt an application-specific action conversion function."""

    def __init__(
        self,
        resolve_fn: Callable[
            [Observation, ActionChunk | None, ControlTarget | None],
            ControlTarget | None,
        ],
    ) -> None:
        self.resolve_fn = resolve_fn

    def reset_episode(self) -> None:
        return None

    def resolve(
        self,
        observation: Observation,
        action: ActionChunk | None,
        world_target: ControlTarget | None,
    ) -> ControlTarget | None:
        return self.resolve_fn(observation, action, world_target)


class DirectActionResolver:
    """Resolve joint or torque actions when no pose/IK conversion is needed."""

    def reset_episode(self) -> None:
        return None

    def resolve(
        self,
        observation: Observation,
        action: ActionChunk | None,
        world_target: ControlTarget | None,
    ) -> ControlTarget | None:
        del observation
        if world_target is not None:
            return world_target
        if action is None:
            return None
        values = np.asarray(action.first, dtype=np.float64)
        if action.semantic == "joint":
            if values.shape != (7,):
                raise ValueError("joint action must be a seven-vector")
            return ControlTarget(
                q=values,
                mode="position",
                metadata={"source": "direct_action_resolver"},
            )
        if action.semantic == "torque":
            if values.shape != (7,):
                raise ValueError("torque action must be a seven-vector")
            return ControlTarget(
                torque=values,
                mode="torque",
                metadata={"source": "direct_action_resolver"},
            )
        if action.semantic in {"pose", "eepose"}:
            return ControlTarget(
                pose=values,
                mode="pose",
                metadata={"source": "direct_action_resolver", "frame_name": action.frame_name},
            )
        raise ValueError(f"unsupported action semantic {action.semantic!r}")


__all__ = ["CallableActionResolver", "DirectActionResolver"]
