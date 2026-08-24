"""Thin model-family inference classes built on the common orchestrator."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from inference.core.base import InferenceBase
from inference.core.interfaces import (
    ActionResolver,
    DiagnosticSink,
    ObservationProcessor,
    ObservationSampler,
    RobotController,
    SafetyGuard,
    WorldModel,
)
from inference.core.base import ActionChunkScheduler
from inference.core.contracts import ActionChunk, Observation
from inference.policies.base import CallablePolicy
from inference.policies.dp import DiffusionPolicyAdapter


class DiffusionPolicyInference(InferenceBase):
    """DP-specific shell; all scheduling/control remains in ``InferenceBase``."""

    def __init__(
        self,
        *,
        sampler: ObservationSampler,
        model: Any,
        input_builder: Callable[[Observation], Any] | None = None,
        action_semantic: str = "eepose",
        action_frame_name: str | None = None,
        action_steps: int | None = None,
        action_step_s: float | None = None,
        processor: ObservationProcessor | None = None,
        world_model: WorldModel | None = None,
        action_resolver: ActionResolver | None = None,
        safety_guard: SafetyGuard | None = None,
        controller: RobotController | None = None,
        diagnostics: Iterable[DiagnosticSink] = (),
        scheduler: ActionChunkScheduler | None = None,
    ) -> None:
        policy = DiffusionPolicyAdapter(
            model,
            input_builder=input_builder,
            semantic=action_semantic,
            frame_name=action_frame_name,
            action_steps=action_steps,
            step_s=action_step_s,
        )
        super().__init__(
            sampler=sampler,
            policy=policy,
            processor=processor,
            world_model=world_model,
            action_resolver=action_resolver,
            safety_guard=safety_guard,
            controller=controller,
            diagnostics=diagnostics,
            scheduler=scheduler,
        )


class VLAInference(InferenceBase):
    """VLA shell accepting a callable policy until the VLA contract is fixed."""

    def __init__(
        self,
        *,
        sampler: ObservationSampler,
        predict_fn: Callable[[Observation], ActionChunk | np.ndarray | None],
        action_semantic: str = "eepose",
        action_frame_name: str | None = None,
        action_step_s: float | None = None,
        processor: ObservationProcessor | None = None,
        world_model: WorldModel | None = None,
        action_resolver: ActionResolver | None = None,
        safety_guard: SafetyGuard | None = None,
        controller: RobotController | None = None,
        diagnostics: Iterable[DiagnosticSink] = (),
        scheduler: ActionChunkScheduler | None = None,
    ) -> None:
        policy = CallablePolicy(
            predict_fn,
            semantic=action_semantic,
            frame_name=action_frame_name,
            step_s=action_step_s,
        )
        super().__init__(
            sampler=sampler,
            policy=policy,
            processor=processor,
            world_model=world_model,
            action_resolver=action_resolver,
            safety_guard=safety_guard,
            controller=controller,
            diagnostics=diagnostics,
            scheduler=scheduler,
        )
