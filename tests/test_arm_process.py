from __future__ import annotations

import time

import numpy as np

from nero_collection.arms.base import ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.arms.process import IsolatedArmProcess
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


def test_isolated_mock_arm_reads_one_sdk_snapshot_per_request_and_restarts() -> None:
    arm = IsolatedArmProcess(
        ArmEndpointConfig(name="isolated_mock", rest_q=(0.0,) * 7),
        backend="mock",
    )
    arm.connect()
    try:
        deadline = time.monotonic() + 2.0
        connected_state = arm.read_state()
        while not np.all(np.isfinite(connected_state.q)) and time.monotonic() < deadline:
            time.sleep(0.01)
            connected_state = arm.read_state()
        assert np.all(np.isfinite(connected_state.q))

        first = arm.read_state()
        second = arm.read_state()
        assert second.timestamp_us >= first.timestamp_us

        arm.set_follower_mode()
        deadline = time.monotonic() + 2.0
        restarted = None
        while time.monotonic() < deadline:
            restarted = arm.read_state()
            if np.all(np.isfinite(restarted.q)):
                break
            time.sleep(0.01)
        assert restarted is not None
        assert np.all(np.isfinite(restarted.q))
    finally:
        arm.disconnect()


def test_production_factory_uses_isolated_hardware_process() -> None:
    arm = build_arm(ArmEndpointConfig(name="follower"), "pyagxarm")

    assert isinstance(arm, IsolatedArmProcess)
