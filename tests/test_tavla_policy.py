from __future__ import annotations

import numpy as np
import pytest

from inference.core import Observation
from inference.factory import POLICY_REGISTRY
from inference.policies import TAVLA, TAVLAObservationBuilder


def _observation(timestamp_us: int = 123) -> Observation:
    return Observation(
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us,
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        tau_ext=np.zeros(7),
        wrench_ext=np.zeros(6),
        images={"wrist": np.zeros((8, 10, 3), dtype=np.uint8)},
        metadata={"instruction": "insert the USB"},
    )


class _OfficialStyleModel:
    def __init__(self):
        self.inputs = []
        self.reset_count = 0

    def eval(self):
        return self

    def predict_action(self, payload):
        self.inputs.append(payload)
        return {"actions": np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7)}

    def reset(self):
        self.reset_count += 1


def test_tavla_normalizes_official_action_mapping_and_batch():
    model = _OfficialStyleModel()
    policy = TAVLA(
        model,
        semantic="joint",
        frame_name="gripper_tcp",
        action_steps=2,
        step_s=0.02,
    )
    policy.start()
    result = policy.predict(_observation())

    assert result.values.shape == (2, 7)
    np.testing.assert_allclose(result.values[0], np.arange(7))
    assert result.semantic == "joint"
    assert result.frame_name == "gripper_tcp"
    assert result.step_s == 0.02
    assert result.timestamp_us == 123
    assert result.metadata["action_key"] == "actions"
    assert model.inputs[0]["instruction"] == "insert the USB"
    assert model.inputs[0]["state"].shape == (7,)


def test_tavla_supports_custom_processor_and_reset():
    model = _OfficialStyleModel()
    seen = []

    def processor(payload):
        seen.append(payload)
        return {"obs": payload["q"]}

    policy = TAVLA(
        model,
        processor=processor,
        input_builder=lambda observation: {"q": observation.q.copy()},
    )
    policy.predict(_observation(456))
    policy.reset_episode()

    assert seen[0]["q"].shape == (7,)
    assert model.reset_count == 1


def test_tavla_registry_key_and_observation_builder():
    assert POLICY_REGISTRY.get("tavla") is TAVLA
    payload = TAVLAObservationBuilder()(_observation())
    assert set(payload["images"]) == {"wrist"}
    assert payload["instruction"] == "insert the USB"


def test_tavla_does_not_retry_after_internal_type_error():
    class FailingModel:
        def __init__(self):
            self.calls = 0

        def predict_action(self, payload):
            self.calls += 1
            raise TypeError("internal model failure")

    model = FailingModel()
    policy = TAVLA(model, semantic="joint")
    with pytest.raises(TypeError, match="internal model failure"):
        policy.predict(_observation())
    assert model.calls == 1
