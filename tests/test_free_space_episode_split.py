from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from calibration.free_space_cli import (
    _execute_trajectory,
    _find_free_space_resume,
    _parse_args,
)


class _Buffer:
    def __init__(self, discard_initial_s: float) -> None:
        self.config = SimpleNamespace(
            output=SimpleNamespace(discard_initial_s=discard_initial_s)
        )
        self.sample_count = 0
        self.stored_timestamps = []
        self.stored_values = []

    def append_teleop(self, timestamp_us, _values, *, store=True):
        if store:
            self.sample_count += 1
            self.stored_timestamps.append(timestamp_us)
            self.stored_values.append(_values)


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
            acquired_timestamp_us=timestamp_us,
            q_acquired_timestamp_us=timestamp_us,
            q_component_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
            motor_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
        )

def test_collect_cli_accepts_fresh_native_can_recollection() -> None:
    args = _parse_args(
        [
            "--config",
            "configs/joint_pose_coverage.yaml",
            "collect",
            "--fresh",
        ]
    )

    assert args.command == "collect"
    assert args.fresh


def _state(timestamp_us: int, value: float = 0.0):
    vector = np.full(7, value)
    return SimpleNamespace(
        q=vector.copy(),
        dq=vector.copy(),
        ddq=vector.copy(),
        torque=vector.copy(),
        current=vector.copy(),
        ee_pose=np.eye(4),
        q_timestamp_us=timestamp_us,
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us,
        q_acquired_timestamp_us=timestamp_us,
        q_component_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
        motor_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
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


def test_execute_trajectory_resumes_at_requested_sample_and_chunk() -> None:
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
        completed.append((chunk_index, from_index, to_index, buffer.sample_count))
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
        start_index=6,
        initial_chunk_index=2,
    )

    assert completed == [(2, 6, 9, 3)]
    assert final_chunk_index == 3
    assert final_from_index == 9
    assert final_buffer.sample_count == 1
    assert arm.command_count == 4


def _write_completed_chunk(
    path: Path,
    *,
    chunk_index: int,
    trajectory_from_index: int,
    trajectory_to_index: int,
    trajectory_digest: str = "trajectory-sha",
    coverage_config_digest: str = "config-sha",
) -> None:
    sample_count = trajectory_to_index - trajectory_from_index
    metadata = {
        "source": "free_space_coverage",
        "trajectory_name": "coverage",
        "trajectory_sha256": trajectory_digest,
        "coverage_config_sha256": coverage_config_digest,
        "trajectory_episode_index": chunk_index,
        "trajectory_sample_from_index": trajectory_from_index,
        "trajectory_sample_to_index": trajectory_to_index,
        "trajectory_episode_sample_count": sample_count,
    }
    with h5py.File(path, "w") as h5:
        h5.create_dataset("teleop/timestamp_us", data=np.arange(sample_count))
        h5.create_dataset(
            "metadata/episode_json",
            data=json.dumps(metadata).encode("utf-8"),
        )


def test_resume_uses_latest_contiguous_matching_episode_group(tmp_path: Path) -> None:
    _write_completed_chunk(
        tmp_path / "episode_0001_old.h5",
        chunk_index=0,
        trajectory_from_index=2,
        trajectory_to_index=5,
    )
    _write_completed_chunk(
        tmp_path / "episode_0002_old.h5",
        chunk_index=1,
        trajectory_from_index=5,
        trajectory_to_index=10,
    )
    _write_completed_chunk(
        tmp_path / "episode_0004_latest.h5",
        chunk_index=0,
        trajectory_from_index=2,
        trajectory_to_index=6,
    )
    _write_completed_chunk(
        tmp_path / "episode_0005_latest.h5",
        chunk_index=1,
        trajectory_from_index=6,
        trajectory_to_index=8,
    )

    resume = _find_free_space_resume(
        tmp_path,
        "episode",
        trajectory_name="coverage",
        trajectory_digest="trajectory-sha",
        coverage_config_digest="config-sha",
        first_stored_index=2,
        trajectory_sample_count=10,
    )

    assert resume is not None
    assert [path.name for path in resume.episode_paths] == [
        "episode_0004_latest.h5",
        "episode_0005_latest.h5",
    ]
    assert resume.next_chunk_index == 2
    assert resume.next_trajectory_index == 8


def test_resume_rejects_a_gap_in_chunk_ranges(tmp_path: Path) -> None:
    _write_completed_chunk(
        tmp_path / "episode_0001_first.h5",
        chunk_index=0,
        trajectory_from_index=2,
        trajectory_to_index=5,
    )
    _write_completed_chunk(
        tmp_path / "episode_0002_gap.h5",
        chunk_index=1,
        trajectory_from_index=6,
        trajectory_to_index=8,
    )

    resume = _find_free_space_resume(
        tmp_path,
        "episode",
        trajectory_name="coverage",
        trajectory_digest="trajectory-sha",
        coverage_config_digest="config-sha",
        first_stored_index=2,
        trajectory_sample_count=10,
    )

    assert resume is not None
    assert [path.name for path in resume.episode_paths] == [
        "episode_0001_first.h5"
    ]
    assert resume.next_trajectory_index == 5


def test_each_command_tick_stores_one_latest_sdk_state() -> None:
    trajectory = SimpleNamespace(
        q=np.zeros((3, 7)),
        dq=np.zeros((3, 7)),
        time_s=np.arange(3, dtype=np.float64) * 0.02,
        segment_id=np.zeros(3, dtype=np.int64),
        segment_names=("test",),
    )
    plan = SimpleNamespace(
        excitation=SimpleNamespace(sample_rate_hz=50.0),
        hardware=SimpleNamespace(
            max_timestamp_gap_s=0.1,
            max_tracking_error_rad=np.ones(7),
            max_abs_torque_nm=np.ones(7),
        ),
    )
    buffer, _, _ = _execute_trajectory(
        _Arm(),
        SimpleNamespace(poll=lambda: ()),
        _Buffer(discard_initial_s=0.0),
        trajectory,
        plan,
        np.full(7, -1.0),
        np.full(7, 1.0),
    )

    assert buffer.sample_count == 3
    assert np.all(np.diff(buffer.stored_timestamps) > 0)
    assert all(
        "q_follower" in values
        and "tau_follower" in values
        and "q_source_timestamp_follower_us" not in values
        and "state_source_skew_follower_us" not in values
        for values in buffer.stored_values
    )


def test_repeated_sdk_timestamp_is_stored_on_every_command_tick() -> None:
    class CachedArm(_Arm):
        def read_state(self):
            return _state(1_000_000)

    command_count = 4
    trajectory = SimpleNamespace(
        q=np.zeros((command_count, 7)),
        dq=np.zeros((command_count, 7)),
        time_s=np.arange(command_count, dtype=np.float64) * 0.02,
        segment_id=np.zeros(command_count, dtype=np.int64),
        segment_names=("test",),
    )
    plan = SimpleNamespace(
        excitation=SimpleNamespace(sample_rate_hz=50.0),
        hardware=SimpleNamespace(
            max_timestamp_gap_s=0.1,
            max_tracking_error_rad=np.ones(7),
            max_abs_torque_nm=np.ones(7),
        ),
    )
    arm = CachedArm()

    buffer, _, _ = _execute_trajectory(
        arm,
        SimpleNamespace(poll=lambda: ()),
        _Buffer(discard_initial_s=0.0),
        trajectory,
        plan,
        np.full(7, -1.0),
        np.full(7, 1.0),
    )

    assert arm.command_count == command_count
    assert buffer.sample_count == command_count
    assert np.all(np.diff(buffer.stored_timestamps) > 0)
