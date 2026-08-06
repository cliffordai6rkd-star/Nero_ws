from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from calibration.free_space_cli import _execute_trajectory


class _Buffer:
    def __init__(self, discard_initial_s: float) -> None:
        self.config = SimpleNamespace(
            output=SimpleNamespace(discard_initial_s=discard_initial_s)
        )
        self.sample_count = 0
        self.stored_timestamps = []

    def append_teleop(self, timestamp_us, _values, *, store=True):
        if store:
            self.sample_count += 1
            self.stored_timestamps.append(timestamp_us)


class _Arm:
    def __init__(self) -> None:
        self.command_count = 0
        self.read_count = 0
        self.invalid_reads = 0

    def command_joint_positions(self, _q_cmd) -> None:
        self.command_count += 1

    def read_state(self):
        if self.invalid_reads:
            self.invalid_reads -= 1
            values = np.full(7, np.nan)
            return SimpleNamespace(
                q=values,
                dq=values,
                ddq=values,
                torque=values,
                current=values,
                ee_pose=np.eye(4),
                q_timestamp_us=0,
                timestamp_us=0,
            )
        timestamp_us = 1_000_000 + self.read_count * 10_000
        self.read_count += 1
        zeros = np.zeros(7)
        return SimpleNamespace(
            q=zeros,
            dq=zeros,
            ddq=zeros,
            torque=zeros,
            current=zeros,
            ee_pose=np.eye(4),
            q_timestamp_us=timestamp_us,
            timestamp_us=timestamp_us,
        )


def test_execute_trajectory_splits_only_stored_samples_without_trajectory_gaps() -> None:
    sample_count = 10
    trajectory = SimpleNamespace(
        q=np.zeros((sample_count, 7)),
        dq=np.zeros((sample_count, 7)),
        time_s=np.arange(sample_count, dtype=np.float64) * 0.01,
        segment_id=np.zeros(sample_count, dtype=np.int64),
        segment_names=("test",),
    )
    plan = SimpleNamespace(
        excitation=SimpleNamespace(sample_rate_hz=100.0),
        hardware=SimpleNamespace(
            max_timestamp_gap_s=0.1,
            max_tracking_error_rad=np.ones(7),
            max_abs_torque_nm=np.ones(7),
        ),
    )
    arm = _Arm()
    completed = []

    def rotate(buffer, chunk_index, from_index, to_index):
        completed.append(
            (
                chunk_index,
                from_index,
                to_index,
                buffer.sample_count,
                tuple(buffer.stored_timestamps),
            )
        )
        arm.invalid_reads = 2
        return _Buffer(discard_initial_s=0.02)

    final_buffer, final_chunk_index, final_from_index = _execute_trajectory(
        arm,
        SimpleNamespace(poll=lambda: ()),
        _Buffer(discard_initial_s=0.02),
        trajectory,
        plan,
        np.full(7, -1.0),
        np.full(7, 1.0),
        max_episode_samples=3,
        rotate_episode=rotate,
    )

    assert [(item[0], item[1], item[2], item[3]) for item in completed] == [
        (0, 2, 5, 3),
        (1, 5, 8, 3),
    ]
    assert final_chunk_index == 2
    assert final_from_index == 8
    assert final_buffer.sample_count == 2
    all_timestamps = [
        timestamp
        for item in completed
        for timestamp in item[4]
    ] + final_buffer.stored_timestamps
    assert len(all_timestamps) == 8
    assert np.all(np.diff(all_timestamps) > 0)
    assert arm.invalid_reads == 0
    assert arm.command_count == sample_count
