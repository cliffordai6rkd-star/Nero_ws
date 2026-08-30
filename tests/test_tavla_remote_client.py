from __future__ import annotations

import numpy as np

from inference.tavla import (
    EffortHistoryBuffer,
    TavlaObservationBuilder,
    TavlaRemotePolicy,
    clip_joint_target,
)
from inference.tavla.tavla_client import EFFORT_OFFSETS


def test_effort_history_uses_training_offsets_at_25_hz():
    history = EffortHistoryBuffer()
    for frame_index in range(51):
        history.append(
            np.full(7, frame_index, dtype=np.float32),
            timestamp=frame_index / 25.0,
        )

    assert history.ready
    sampled = history.sampled()
    expected = np.asarray([50 + offset for offset in EFFORT_OFFSETS])
    np.testing.assert_array_equal(sampled[:, 0], expected)
    assert sampled.shape == (10, 7)


def test_observation_builder_maps_dual_bgr_cameras_and_joint_signals():
    builder = TavlaObservationBuilder("button", camera_color="bgr", resize=0)
    side_bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    wrist_bgr = np.asarray([[[4, 5, 6]]], dtype=np.uint8)
    observation = builder.build(
        side_image=side_bgr,
        wrist_image=wrist_bgr,
        state=np.arange(7),
        effort_history=np.zeros((10, 7)),
    )

    np.testing.assert_array_equal(
        observation["images"]["cam_high"],
        np.asarray([[[3, 2, 1]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        observation["images"]["cam_left_wrist"],
        np.asarray([[[6, 5, 4]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(observation["state"], np.arange(7))
    assert observation["effort"].shape == (10, 7)
    assert observation["prompt"] == "press the button"


def test_remote_policy_limits_every_nero_joint_dimension():
    class FakeWebsocketPolicy:
        def infer(self, observation):
            assert observation == {"snapshot": True}
            return {"actions": np.full((50, 14), 2.0, dtype=np.float32)}

    remote = TavlaRemotePolicy.__new__(TavlaRemotePolicy)
    remote.policy = FakeWebsocketPolicy()
    actions = remote.infer_observation(
        {"snapshot": True},
        current_state=np.zeros(7),
        max_joint_step=0.02,
    )

    assert actions.shape == (50, 7)
    np.testing.assert_allclose(actions, 0.02)


def test_live_target_clipping_uses_latest_state_for_all_seven_joints():
    current = np.arange(7, dtype=np.float32)
    target = current + np.asarray([1.0, -1.0, 0.01, -0.01, 0.5, -0.5, 2.0])
    clipped = clip_joint_target(target, current, 0.02)

    np.testing.assert_allclose(
        clipped - current,
        np.asarray([0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.02]),
        atol=1e-7,
    )
