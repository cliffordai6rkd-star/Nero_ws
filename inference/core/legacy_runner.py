"""Compatibility runner that isolates the existing Nero pipeline contract."""

from __future__ import annotations

from typing import Any, Callable

from inference.core.nero_sampler import NeroObservationSampler
from inference.pipeline import InferenceInput


class NeroPipelineRunner:
    """Connect the modular sampler/controller to the legacy pipeline.

    This is intentionally a small bridge.  It allows the runtime to be
    migrated stage-by-stage while the mature DP/WM pipeline keeps its exact
    observation and output semantics.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        sampler: NeroObservationSampler,
        controller: Any,
        image_keys: tuple[str, ...],
        on_observation: Callable[[Any], None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.sampler = sampler
        self.controller = controller
        self.image_keys = tuple(image_keys)
        self.on_observation = on_observation

    def step(self):
        observation = self.sampler.sample()
        if observation is None:
            return None
        if self.on_observation is not None:
            self.on_observation(observation)
        image = (
            observation.images[self.image_keys[0]]
            if len(self.image_keys) == 1
            else observation.images
        )
        image_timestamp_s = (
            observation.image_timestamps_us.get(
                self.image_keys[0], observation.timestamp_us
            )
            * 1.0e-6
            if len(self.image_keys) == 1
            else {
                key: observation.image_timestamps_us.get(key, observation.timestamp_us)
                * 1.0e-6
                for key in self.image_keys
            }
        )
        output = self.pipeline.step(
            InferenceInput(
                q=observation.q,
                dq=observation.dq,
                ddq=observation.ddq,
                tau=observation.tau,
                image=image,
                wrench_ext=observation.wrench_ext,
                timestamp_s=observation.timestamp_us * 1.0e-6,
                wrench_to_control_rotation=observation.wrench_to_control_rotation,
                image_timestamp_s=image_timestamp_s,
            )
        )
        self.controller.send(observation, output)
        return output
