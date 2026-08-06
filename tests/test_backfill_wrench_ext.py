from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.backfill_wrench_ext import (
    compute_episode_wrench,
    discover_h5_files,
    inspect_episode,
    write_wrench_atomic,
)


def _write_episode(path: Path, sample_count: int = 4) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as h5:
        teleop = h5.create_group("teleop")
        teleop.create_dataset(
            "timestamp_us",
            data=np.arange(sample_count, dtype=np.int64) * 10_000 + 1_000_000,
        )
        teleop.create_dataset(
            "q_follower",
            data=np.arange(sample_count * 7, dtype=np.float64).reshape(sample_count, 7),
        )
        teleop.create_dataset(
            "tau_ext",
            data=np.full((sample_count, 7), 2.0, dtype=np.float64),
        )


def test_discover_and_backfill_wrench_ext_atomically(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    episode = tmp_path / "episode_0003_20260728_120000.h5"
    _write_episode(episode)

    class Estimator:
        def map_joint_torque(self, q, tau_ext):
            return SimpleNamespace(wrench=np.asarray(q[:6]) + float(tau_ext[0]))

    assert discover_h5_files([tmp_path], recursive=False) == [episode]
    assert inspect_episode(episode) == (4, False)
    wrench = compute_episode_wrench(episode, Estimator())
    attrs = {"frame_name": "tool", "reference_frame": "local"}

    write_wrench_atomic(episode, wrench, attrs, overwrite=False)

    assert inspect_episode(episode) == (4, True)
    with h5py.File(episode, "r") as h5:
        dataset = h5["teleop/wrench_ext"]
        assert dataset[:] == pytest.approx(wrench)
        assert dataset.attrs["state_name"] == "wrench"
        assert dataset.attrs["frame_name"] == "tool"
        assert dataset.compression == "gzip"


def test_backfill_refuses_existing_dataset_without_overwrite(tmp_path: Path) -> None:
    episode = tmp_path / "episode_0000_20260728_120000.h5"
    _write_episode(episode)
    wrench = np.zeros((4, 6), dtype=np.float64)
    write_wrench_atomic(episode, wrench, {}, overwrite=False)

    with pytest.raises(RuntimeError, match="already exists"):
        write_wrench_atomic(episode, wrench + 1.0, {}, overwrite=False)

