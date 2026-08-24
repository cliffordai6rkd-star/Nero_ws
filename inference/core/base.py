"""Generic inference orchestrator.

The orchestrator owns ordering and lifecycle only.  Sensor acquisition,
pre-processing, model execution, world-model inference, action conversion and
hardware control are injected components, so a new policy does not need to
copy the robot loop.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

from inference.core.contracts import ActionChunk, ControlTarget, InferenceCycle, Observation
from inference.core.interfaces import (
    ActionResolver,
    DiagnosticSink,
    HighLevelPolicy,
    ObservationProcessor,
    ObservationSampler,
    RobotController,
    SafetyGuard,
    WorldModel,
)

log = logging.getLogger(__name__)


class ActionChunkScheduler:
    """Advance a policy action chunk using sensor timestamps.

    This class deliberately does not know whether actions are joint or pose
    values.  Frame/semantic conversion belongs to ``ActionResolver``.
    """

    def __init__(self) -> None:
        self._chunk: ActionChunk | None = None
        self._index = 0
        self._next_timestamp_us: int | None = None
        self._completed = False

    @property
    def active(self) -> bool:
        return (
            self._chunk is not None
            and self._index < len(self._chunk.values)
            and not self._completed
        )

    @property
    def chunk(self) -> ActionChunk | None:
        return self._chunk

    def reset_episode(self) -> None:
        self._chunk = None
        self._index = 0
        self._next_timestamp_us = None
        self._completed = False

    def install(self, chunk: ActionChunk) -> None:
        self._chunk = chunk
        self._index = 0
        self._next_timestamp_us = int(chunk.timestamp_us)
        self._completed = False

    def current(self, timestamp_us: int) -> ActionChunk | None:
        chunk = self._chunk
        if chunk is None:
            return None
        if chunk.step_s is not None and self._next_timestamp_us is not None:
            step_us = max(1, int(round(chunk.step_s * 1.0e6)))
            while (
                self._index + 1 < len(chunk.values)
                and int(timestamp_us) >= self._next_timestamp_us + step_us
            ):
                self._index += 1
                self._next_timestamp_us += step_us
            if (
                self._index == len(chunk.values) - 1
                and int(timestamp_us) >= self._next_timestamp_us + step_us
            ):
                self._completed = True
        values = chunk.values[self._index :].copy()
        return ActionChunk(
            values=values,
            semantic=chunk.semantic,
            frame_name=chunk.frame_name,
            timestamp_us=chunk.timestamp_us,
            step_s=chunk.step_s,
            metadata={**chunk.metadata, "index": self._index},
        )


class NullObservationProcessor:
    def reset_episode(self) -> None:
        return None

    def process(self, observation: Observation) -> Observation:
        return observation


class NullWorldModel:
    """No-op WM used while the world-model contract is undecided."""

    def reset_episode(self) -> None:
        return None

    def infer(
        self,
        observation: Observation,
        action: ActionChunk | None,
    ) -> ControlTarget | None:
        del observation, action
        return None


class NullActionResolver:
    def reset_episode(self) -> None:
        return None

    def resolve(
        self,
        observation: Observation,
        action: ActionChunk | None,
        world_target: ControlTarget | None,
    ) -> ControlTarget | None:
        del observation, action
        return world_target


class NullSafetyGuard:
    def reset_episode(self) -> None:
        return None

    def validate(
        self,
        observation: Observation,
        target: ControlTarget | None,
    ) -> ControlTarget | None:
        del observation
        return target


class InferenceBase:
    """Lifecycle and stage orchestration shared by all model families."""

    def __init__(
        self,
        *,
        sampler: ObservationSampler,
        policy: HighLevelPolicy,
        processor: ObservationProcessor | None = None,
        world_model: WorldModel | None = None,
        action_resolver: ActionResolver | None = None,
        safety_guard: SafetyGuard | None = None,
        controller: RobotController | None = None,
        diagnostics: Iterable[DiagnosticSink] = (),
        scheduler: ActionChunkScheduler | None = None,
        policy_update_every_cycle: bool = True,
    ) -> None:
        self.sampler = sampler
        self.policy = policy
        self.processor = processor or NullObservationProcessor()
        self.world_model = world_model or NullWorldModel()
        self.action_resolver = action_resolver or NullActionResolver()
        self.safety_guard = safety_guard or NullSafetyGuard()
        self.controller = controller
        self.diagnostics = tuple(diagnostics)
        self.scheduler = scheduler or ActionChunkScheduler()
        self.policy_update_every_cycle = bool(policy_update_every_cycle)
        self._started = False
        self._last_policy_timestamp_us: int | None = None

    def start(self) -> None:
        if self._started:
            return
        self.sampler.start()
        starter = getattr(self.policy, "start", None)
        if callable(starter):
            starter()
        if self.controller is not None:
            self.controller.start()
        for sink in self.diagnostics:
            sink.start()
        self._started = True

    def reset_episode(self) -> None:
        if not self._started:
            raise RuntimeError("inference must be started before reset_episode()")
        self.scheduler.reset_episode()
        self._last_policy_timestamp_us = None
        for component in (
            self.sampler,
            self.processor,
            self.policy,
            self.world_model,
            self.action_resolver,
            self.safety_guard,
            self.controller,
        ):
            if component is not None:
                reset = getattr(component, "reset_episode", None)
                if callable(reset):
                    reset()
        for sink in self.diagnostics:
            sink.reset_episode()

    def step(self) -> InferenceCycle | None:
        if not self._started:
            raise RuntimeError("inference must be started before step()")
        started = time.perf_counter()
        observation = self.sampler.sample()
        if observation is None:
            return None
        observation = self.processor.process(observation)

        policy_updated = False
        action = self.scheduler.current(observation.timestamp_us)
        should_update = self.policy_update_every_cycle or not self.scheduler.active
        if should_update:
            predicted = self.policy.predict(observation)
            if predicted is not None:
                self.scheduler.install(predicted)
                action = self.scheduler.current(observation.timestamp_us)
                policy_updated = True
                self._last_policy_timestamp_us = observation.timestamp_us

        wm_started = time.perf_counter()
        world_target = self.world_model.infer(observation, action)
        wm_elapsed = time.perf_counter() - wm_started
        target = self.action_resolver.resolve(observation, action, world_target)
        target = self.safety_guard.validate(observation, target)
        command = None
        if self.controller is not None:
            command = self.controller.send(observation, target)

        cycle = InferenceCycle(
            observation=observation,
            action=action,
            target=target,
            command=command,
            policy_updated=policy_updated,
            world_model_updated=world_target is not None,
            timings_s={
                "total": time.perf_counter() - started,
                "world_model": wm_elapsed,
            },
        )
        for sink in self.diagnostics:
            sink.publish(cycle)
        return cycle

    def close(self) -> None:
        if not self._started:
            return
        errors: list[BaseException] = []
        for component in (self.controller, self.policy, self.sampler):
            if component is None:
                continue
            closer = getattr(component, "close", None)
            if not callable(closer):
                closer = getattr(component, "stop", None)
            if callable(closer):
                try:
                    closer()
                except BaseException as exc:  # pragma: no cover - shutdown guard
                    errors.append(exc)
        for sink in self.diagnostics:
            try:
                sink.close()
            except BaseException as exc:  # pragma: no cover - shutdown guard
                errors.append(exc)
        self._started = False
        if errors:
            raise RuntimeError("one or more inference components failed to close") from errors[0]
