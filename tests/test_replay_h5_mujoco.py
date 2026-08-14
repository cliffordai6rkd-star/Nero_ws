from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.replay_h5_mujoco import (
    display_indices,
    load_joint_trajectory,
    resolve_episode,
)


def test_resolve_episode_from_runs_directory(tmp_path: Path) -> None:
    episode = tmp_path / "episode_0003_test.h5"
    episode.touch()

    assert resolve_episode(tmp_path, 3) == episode.resolve()


def test_load_joint_trajectory_selects_arm_and_time_interval(tmp_path: Path) -> None:
    path = tmp_path / "episode.h5"
    timestamp_us = 1_000_000 + np.arange(10, dtype=np.int64) * 10_000
    first = np.arange(70, dtype=np.float64).reshape(10, 7)
    second = first + 100.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("teleop/timestamp_us", data=timestamp_us)
        h5.create_dataset("teleop/q_follower", data=np.concatenate((first, second), axis=1))

    q, time_s = load_joint_trajectory(
        path,
        "teleop/q_follower",
        "teleop/timestamp_us",
        arm_index=1,
        start_s=0.02,
        stop_s=0.06,
    )

    assert np.array_equal(q, second[2:6])
    assert time_s == pytest.approx([0.0, 0.01, 0.02, 0.03])


def test_display_indices_caps_wall_clock_refresh_and_keeps_last_pose() -> None:
    time_s = np.arange(101, dtype=np.float64) / 100.0

    indices = display_indices(time_s, playback_speed=2.0, maximum_display_hz=25.0)

    assert indices[0] == 0
    assert indices[-1] == 100
    assert len(indices) <= 14
