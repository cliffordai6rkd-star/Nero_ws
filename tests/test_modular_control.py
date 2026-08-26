from __future__ import annotations

import numpy as np

from inference.control import BasicSafetyGuard, DirectActionResolver
from inference.core.contracts import ActionChunk, Observation


def _observation() -> Observation:
    return Observation(
        timestamp_us=1,
        acquired_timestamp_us=1,
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        tau_ext=np.zeros(7),
        wrench_ext=np.zeros(6),
    )


def test_direct_action_resolver_maps_joint_action_to_position_target() -> None:
    target = DirectActionResolver().resolve(
        _observation(),
        ActionChunk(
            values=np.ones((2, 7)),
            semantic="joint",
            frame_name=None,
            timestamp_us=1,
        ),
        None,
    )
    assert target is not None
    assert target.mode == "position"
    np.testing.assert_allclose(target.q, 1.0)


def test_basic_safety_guard_limits_joint_step_and_torque() -> None:
    observation = _observation()
    target = DirectActionResolver().resolve(
        observation,
        ActionChunk(
            values=np.full(7, 2.0),
            semantic="torque",
            frame_name=None,
            timestamp_us=1,
        ),
        None,
    )
    assert target is not None
    target = BasicSafetyGuard(maximum_torque=0.5).validate(observation, target)
    np.testing.assert_allclose(target.torque, 0.5)

