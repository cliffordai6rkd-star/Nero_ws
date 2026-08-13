from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from nero_collection.arms.base import ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.arms.process import IsolatedArmProcess, SharedArmStateRing
from nero_collection.config import ArmEndpointConfig


def _state(timestamp_us: int, value: float = 0.0) -> ArmState:
    vector = np.full(7, value, dtype=np.float64)
    timestamps = np.full(7, timestamp_us, dtype=np.int64)
    return ArmState(
        q=vector.copy(),
        dq=(vector + 1.0),
        ddq=(vector + 2.0),
        ee_pose=np.eye(4, dtype=np.float64),
        torque=(vector + 3.0),
        current=(vector + 4.0),
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us + 1,
        q_timestamp_us=timestamp_us,
        q_acquired_timestamp_us=timestamp_us + 2,
        q_component_timestamp_us=timestamps.copy(),
        q_source_before_timestamp_us=(timestamps - 1),
        q_source_after_timestamp_us=(timestamps + 1),
        motor_timestamp_us=(timestamps + 2),
        motor_acquired_timestamp_us=(timestamps + 3),
    )


def test_shared_arm_state_ring_preserves_order_and_reports_overwrite() -> None:
    ring = SharedArmStateRing(mp.get_context("spawn"), capacity=3)
    for index in range(1, 6):
        ring.append(_state(index * 1_000, float(index)))

    states, sequence, dropped = ring.read_after(0)

    assert sequence == 5
    assert dropped == 2
    assert [state.timestamp_us for state in states] == [3_000, 4_000, 5_000]
    np.testing.assert_allclose(states[-1].q, np.full(7, 5.0))
    np.testing.assert_array_equal(
        states[-1].motor_acquired_timestamp_us,
        np.full(7, 5_003, dtype=np.int64),
    )


def test_isolated_mock_arm_publishes_ordered_states_and_restarts() -> None:
    arm = IsolatedArmProcess(
        ArmEndpointConfig(name="isolated_mock", rest_q=(0.0,) * 7),
        backend="mock",
        history_size=64,
    )
    arm.connect()
    try:
        deadline = time.monotonic() + 2.0
        connected_state = arm.read_state()
        while not np.all(np.isfinite(connected_state.q)) and time.monotonic() < deadline:
            time.sleep(0.01)
            connected_state = arm.read_state()
        assert np.all(np.isfinite(connected_state.q))

        arm.configure_state_capture(0.015, None, None, None, 0.06)
        deadline = time.monotonic() + 2.0
        first = ()
        while time.monotonic() < deadline:
            first = arm.drain_states().states
            if len(first) >= 2:
                break
            time.sleep(0.01)
        assert len(first) >= 2
        assert [state.timestamp_us for state in first] == sorted(
            state.timestamp_us for state in first
        )

        arm.set_follower_mode()
        deadline = time.monotonic() + 2.0
        restarted = ()
        while time.monotonic() < deadline:
            restarted = arm.drain_states().states
            if restarted:
                break
            time.sleep(0.01)
        assert restarted
        assert all(np.all(np.isfinite(state.q)) for state in restarted)
    finally:
        arm.disconnect()


def test_production_factory_uses_isolated_hardware_process() -> None:
    arm = build_arm(ArmEndpointConfig(name="follower"), "pyagxarm")

    assert isinstance(arm, IsolatedArmProcess)
