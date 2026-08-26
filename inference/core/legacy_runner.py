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
                q_cmd=getattr(observation, "q_cmd", None),
                timestamp_s=observation.timestamp_us * 1.0e-6,
                wrench_to_control_rotation=observation.wrench_to_control_rotation,
                image_timestamp_s=image_timestamp_s,
            )
        )
        self.controller.send(observation, output)
        return output


class ModularInferenceRunner:
    """Run an :class:`~inference.core.base.InferenceBase` instance.

    ``NeroPipelineRunner`` adapts the old ``pipeline.step(InferenceInput)``
    signature.  Modular policies (including TAVLA) already own the complete
    stage graph and expose a zero-argument ``step()`` method, so passing them
    through the legacy adapter would fail before policy inference.  Keeping a
    separate, deliberately tiny runner makes the two contracts explicit and
    lets :class:`NeroInferenceRuntime` share its sensor loop with either one.
    """

    def __init__(self, inference: Any) -> None:
        if not callable(getattr(inference, "step", None)):
            raise TypeError("modular inference must expose step()")
        self.inference = inference

    def start(self) -> None:
        starter = getattr(self.inference, "start", None)
        if callable(starter):
            starter()

    def step(self):
        return self.inference.step()

    def reset_episode(self) -> None:
        reset = getattr(self.inference, "reset_episode", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        closer = getattr(self.inference, "close", None)
        if callable(closer):
            closer()
        else:
            stopper = getattr(self.inference, "stop", None)
            if callable(stopper):
                stopper()
