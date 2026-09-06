from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.contracts import Observation
from inference.policies.dp.adapter import DiffusionPolicyAdapter
from inference.policies.lerobotdp import (
    LeRobotDiffusionPolicy,
    _compatible_config_payload,
    is_lerobot_checkpoint,
)


class _FakeLeRobotModel:
    def __init__(self):
        self.config = SimpleNamespace(
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            action_dim=7,
            input_features={
                "observation.state": {"type": "STATE", "shape": (7,)},
                "observation.images.wrist": {
                    "type": "VISUAL",
                    "shape": (3, 192, 256),
                },
                "observation.images.side": {
                    "type": "VISUAL",
                    "shape": (3, 192, 256),
                },
            },
        )
        self.seen = []
        self.calls = 0

    def select_action(self, observation):
        self.seen.append(observation)
        value = np.full(7, self.calls, dtype=np.float32)
        self.calls += 1
        return value

    def eval(self):
        return self


class _QueuedLeRobotModel(_FakeLeRobotModel):
    """Match LeRobot's select_action queue behavior closely enough to regress it."""

    def __init__(self):
        super().__init__()
        self._queues = {"action": deque()}

    def select_action(self, observation):
        self.seen.append(observation)
        self._queues["action"].extend(
            np.full(7, index, dtype=np.float32) for index in range(1, 8)
        )
        return np.zeros(7, dtype=np.float32)


def _observation() -> Observation:
    return Observation(
        timestamp_us=100,
        acquired_timestamp_us=100,
        q=np.linspace(0.0, 0.6, 7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        tau_ext=np.zeros(7),
        wrench_ext=np.zeros(6),
        images={
            "wrist": np.zeros((192, 256, 3), dtype=np.uint8),
            "side": np.zeros((192, 256, 3), dtype=np.uint8),
        },
    )


def test_lerobotdp_builds_canonical_state_and_image_contract():
    model = _FakeLeRobotModel()
    policy = LeRobotDiffusionPolicy(
        model,
        metadata={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
        device=None,
        step_s=0.04,
    )

    result = policy.predict(_observation())

    assert result.values.shape == (8, 7)
    assert result.semantic == "joint"
    assert result.step_s == pytest.approx(0.04)
    assert set(model.seen[0]) == {
        "observation.state",
        "observation.images.wrist",
        "observation.images.side",
    }
    np.testing.assert_allclose(
        model.seen[0]["observation.state"].numpy(), np.linspace(0, 0.6, 7)[None]
    )
    assert tuple(model.seen[0]["observation.images.wrist"].shape) == (
        1,
        3,
        192,
        256,
    )


def test_lerobotdp_rejects_missing_checkpoint_feature():
    model = _FakeLeRobotModel()
    policy = LeRobotDiffusionPolicy(
        model,
        metadata={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
    )
    observation = _observation()
    observation = Observation(
        timestamp_us=observation.timestamp_us,
        acquired_timestamp_us=observation.acquired_timestamp_us,
        q=observation.q,
        dq=observation.dq,
        ddq=observation.ddq,
        tau=observation.tau,
        tau_ext=observation.tau_ext,
        wrench_ext=observation.wrench_ext,
        images={"wrist": observation.images["wrist"]},
    )
    with pytest.raises(KeyError, match="side"):
        policy.predict(observation)


def test_legacy_dp_adapter_delegates_native_lerobot_contract():
    model = _FakeLeRobotModel()
    native = LeRobotDiffusionPolicy(
        model,
        metadata={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
        step_s=0.04,
    )
    adapter = DiffusionPolicyAdapter(native, semantic="joint")
    result = adapter.predict(_observation())
    assert result is not None
    assert result.values.shape == (8, 7)


def test_lerobotdp_applies_checkpoint_pre_and_post_processors():
    model = _FakeLeRobotModel()
    seen_raw = []

    def preprocess(value):
        seen_raw.append(value)
        return {key: tensor[None] for key, tensor in value.items()}

    def postprocess(value):
        return np.asarray(value) + 10.0

    policy = LeRobotDiffusionPolicy(
        model,
        metadata={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
        preprocessor=preprocess,
        postprocessor=postprocess,
    )

    result = policy.predict(_observation())

    assert tuple(seen_raw[0]["observation.state"].shape) == (7,)
    assert tuple(seen_raw[0]["observation.images.wrist"].shape) == (3, 192, 256)
    np.testing.assert_allclose(result.values[:, 0], np.arange(8) + 10.0)


def test_lerobotdp_drains_generated_action_queue_without_repeating_observation():
    model = _QueuedLeRobotModel()
    policy = LeRobotDiffusionPolicy(
        model,
        metadata={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
        step_s=0.04,
    )

    result = policy.predict(_observation())

    assert len(model.seen) == 1
    assert len(model._queues["action"]) == 0
    assert policy.action_start_index == 1
    np.testing.assert_allclose(result.values[:, 0], np.arange(8))


def test_lerobotdp_compat_config_drops_only_exact_noop_fields():
    raw = {
        "type": "diffusion",
        "horizon": 16,
        "use_peft": False,
        "resize_shape": None,
        "crop_ratio": 1.0,
        "compile_model": False,
        "compile_mode": "reduce-overhead",
    }
    result = _compatible_config_payload(
        raw,
        supported_fields={"type", "horizon"},
    )
    assert result == {"type": "diffusion", "horizon": 16}


@pytest.mark.parametrize(
    ("key", "value"),
    [("compile_model", True), ("resize_shape", [192, 256]), ("future_field", 1)],
)
def test_lerobotdp_compat_config_rejects_non_noop_fields(key, value):
    with pytest.raises(ValueError, match="unsupported non-noop"):
        _compatible_config_payload(
            {"type": "diffusion", key: value},
            supported_fields={"type"},
        )


def test_is_lerobot_checkpoint_requires_directory_layout(tmp_path):
    assert not is_lerobot_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not is_lerobot_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    assert is_lerobot_checkpoint(tmp_path)
