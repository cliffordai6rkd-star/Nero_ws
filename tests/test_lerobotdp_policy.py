from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.contracts import Observation
from inference.policies.dp.adapter import DiffusionPolicyAdapter
from inference.policies.lerobotdp import LeRobotDiffusionPolicy, is_lerobot_checkpoint


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


def test_is_lerobot_checkpoint_requires_directory_layout(tmp_path):
    assert not is_lerobot_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not is_lerobot_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    assert is_lerobot_checkpoint(tmp_path)
