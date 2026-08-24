from __future__ import annotations

import numpy as np

from inference.core import (
    ActionChunk,
    ActionChunkScheduler,
    InferenceBase,
    NeroPipelineRunner,
    Observation,
)
from inference.policies import CallablePolicy
from inference.world_models import NullWorldModel


class _Sampler:
    def __init__(self, observations):
        self.observations = list(observations)
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def reset_episode(self):
        return None

    def sample(self):
        return self.observations.pop(0) if self.observations else None


class _Controller:
    def __init__(self):
        self.targets = []

    def start(self):
        return None

    def stop(self):
        return None

    def reset_episode(self):
        return None

    def send(self, _observation, target):
        self.targets.append(target)
        return target


def _observation(timestamp_us: int) -> Observation:
    return Observation(
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us,
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        tau_ext=np.zeros(7),
        wrench_ext=np.zeros(6),
        images={"wrist": np.zeros((4, 5, 3), dtype=np.uint8)},
    )


def test_inference_base_orders_policy_and_controller_stages():
    sampler = _Sampler([_observation(1), _observation(2)])
    controller = _Controller()
    policy_calls = []

    def predict(observation):
        policy_calls.append(observation.timestamp_us)
        return np.ones((2, 7))

    inference = InferenceBase(
        sampler=sampler,
        policy=CallablePolicy(predict, semantic="joint"),
        world_model=NullWorldModel(),
        controller=controller,
    )
    inference.start()
    first = inference.step()
    second = inference.step()

    assert first is not None
    assert second is not None
    assert policy_calls == [1, 2]
    assert first.action is not None
    assert first.action.semantic == "joint"
    assert len(controller.targets) == 2
    inference.close()


def test_action_chunk_accepts_single_action_vector():
    action = ActionChunk(
        values=np.zeros(7),
        semantic="eepose",
        frame_name="gripper_tcp",
        timestamp_us=1,
    )
    assert action.values.shape == (1, 7)


def test_action_scheduler_marks_timed_chunk_complete():
    scheduler = ActionChunkScheduler()
    scheduler.install(
        ActionChunk(
            values=np.zeros((2, 7)),
            semantic="joint",
            frame_name=None,
            timestamp_us=0,
            step_s=0.01,
        )
    )
    assert scheduler.active is True
    assert scheduler.current(10_000).metadata["index"] == 1
    scheduler.current(20_000)
    assert scheduler.active is False


def test_legacy_pipeline_runner_keeps_pipeline_and_controller_separate():
    class Pipeline:
        def step(self, sample):
            assert sample.image.shape == (4, 5, 3)
            return "pipeline-output"

    class Controller:
        def __init__(self):
            self.calls = []

        def send(self, observation, output):
            self.calls.append((observation.timestamp_us, output))

    sampler = _Sampler([_observation(3)])
    controller = Controller()
    runner = NeroPipelineRunner(
        pipeline=Pipeline(),
        sampler=sampler,
        controller=controller,
        image_keys=("wrist",),
    )
    result = runner.step()
    assert result == "pipeline-output"
    assert controller.calls == [(3, "pipeline-output")]
