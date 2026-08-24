"""Protocols for modular online inference.

Implementations may be backed by real hardware, H5 playback, or a deterministic
test double.  The protocols intentionally contain no torch or SDK types.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from inference.core.contracts import ActionChunk, ControlTarget, Observation


class ObservationSampler(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset_episode(self) -> None: ...

    def sample(self) -> Observation | None: ...


class ObservationProcessor(Protocol):
    def reset_episode(self) -> None: ...

    def process(self, observation: Observation) -> Observation: ...


class HighLevelPolicy(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def reset_episode(self) -> None: ...

    def predict(self, observation: Observation) -> ActionChunk | None: ...


class WorldModel(Protocol):
    def reset_episode(self) -> None: ...

    def infer(
        self,
        observation: Observation,
        action: ActionChunk | None,
    ) -> ControlTarget | None: ...


class ActionResolver(Protocol):
    def reset_episode(self) -> None: ...

    def resolve(
        self,
        observation: Observation,
        action: ActionChunk | None,
        world_target: ControlTarget | None,
    ) -> ControlTarget | None: ...


class SafetyGuard(Protocol):
    def reset_episode(self) -> None: ...

    def validate(
        self,
        observation: Observation,
        target: ControlTarget | None,
    ) -> ControlTarget | None: ...


class RobotController(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset_episode(self) -> None: ...

    def send(
        self,
        observation: Observation,
        target: ControlTarget | None,
    ) -> Any | None: ...


class DiagnosticSink(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def reset_episode(self) -> None: ...

    def publish(self, cycle: Any) -> None: ...


class PolicyUpdateGate(Protocol):
    def __call__(self, observation: Observation) -> bool: ...

