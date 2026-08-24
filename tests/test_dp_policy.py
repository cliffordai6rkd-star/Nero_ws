from __future__ import annotations

import numpy as np
import pytest

from inference.core.contracts import Observation
from inference.policies import DiffusionPolicy


torch = pytest.importorskip("torch")


class _IdentityNormalizer:
    def unnormalize(self, value):
        return value


class _FakeDPModel(torch.nn.Module):
    image_keys = ("side", "wrist")
    n_obs_steps = 2
    horizon = 9
    n_action_steps = 8
    action_dim = 7
    action_start_index = 1

    class _Encoder:
        key_shape_map = {
            "side": (3, 192, 256),
            "wrist": (3, 192, 256),
        }

    obs_encoder = _Encoder()

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.normalizer = {"action": _IdentityNormalizer()}
        self.seen = []

    def _encode_observation(self, obs):
        self.seen.append(obs)
        assert set(obs) == {"side", "wrist"}
        assert obs["side"].shape == (1, 2, 3, 192, 256)
        assert obs["wrist"].shape == (1, 2, 3, 192, 256)
        return torch.zeros((1, 2, 4), dtype=torch.float32)

    def conditional_sample(self, shape, condition):
        assert shape == (1, 9, 7)
        assert condition.shape == (1, 2, 4)
        # Values in dimensions 3:7 are intentionally not unit quaternions.
        return torch.arange(63, dtype=condition.dtype).reshape(1, 9, 7)


def _observation(timestamp_us: int, marker: int) -> Observation:
    return Observation(
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us,
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        tau_ext=np.zeros(7),
        wrench_ext=np.zeros(6),
        images={
            "side": np.full((192, 256, 3), marker, dtype=np.uint8),
            "wrist": np.full((192, 256, 3), marker + 1, dtype=np.uint8),
        },
    )


def test_dp_policy_builds_padded_two_frame_input_and_joint_chunk():
    model = _FakeDPModel()
    policy = DiffusionPolicy(model, device="cpu", step_s=0.04)
    policy.start()

    result = policy.predict(_observation(100, 64))

    assert result.values.shape == (8, 7)
    assert result.semantic == "joint"
    assert result.frame_name is None
    assert result.step_s == pytest.approx(0.04)
    # The first observation is repeated to satisfy dataset pad_before=1.
    seen = model.seen[0]
    np.testing.assert_allclose(seen["side"][0, 0].numpy(), seen["side"][0, 1].numpy())
    # Joint output must remain raw; dimensions 3:7 are not quaternion-normalized.
    np.testing.assert_allclose(result.values[0], np.arange(7, 14, dtype=np.float64))


def test_dp_policy_converts_chw_and_resizes_input():
    model = _FakeDPModel()
    policy = DiffusionPolicy(model, device="cpu", step_s=0.04)
    policy.start()

    small_hwc = np.zeros((96, 128, 3), dtype=np.uint8)
    observation = _observation(200, 0)
    observation = Observation(
        timestamp_us=observation.timestamp_us,
        acquired_timestamp_us=observation.acquired_timestamp_us,
        q=observation.q,
        dq=observation.dq,
        ddq=observation.ddq,
        tau=observation.tau,
        tau_ext=observation.tau_ext,
        wrench_ext=observation.wrench_ext,
        images={"side": small_hwc, "wrist": small_hwc},
    )

    result = policy.predict(observation)

    assert result.values.shape == (8, 7)
    assert model.seen[0]["wrist"].shape == (1, 2, 3, 192, 256)


def test_dp_policy_requires_all_checkpoint_cameras():
    model = _FakeDPModel()
    policy = DiffusionPolicy(model, device="cpu", step_s=0.04)
    with pytest.raises(KeyError, match="side"):
        policy.predict(
            Observation(
                timestamp_us=1,
                acquired_timestamp_us=1,
                q=np.zeros(7),
                dq=np.zeros(7),
                ddq=np.zeros(7),
                tau=np.zeros(7),
                tau_ext=np.zeros(7),
                wrench_ext=np.zeros(6),
                images={"wrist": np.zeros((192, 256, 3), dtype=np.uint8)},
            )
        )
