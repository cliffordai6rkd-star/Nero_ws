from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.downsample_h5_camera import downsample_episode, nearest_monotonic_indices


def test_nearest_monotonic_indices_matches_reference_timeline() -> None:
    source = np.arange(0, 1_000_000, 33_333, dtype=np.int64)
    reference = np.arange(0, 1_000_000, 40_000, dtype=np.int64)

    indices = nearest_monotonic_indices(source, reference)

    assert indices.shape == reference.shape
    assert np.all(np.diff(indices) > 0)
    assert np.max(np.abs(source[indices] - reference)) <= 16_667


def test_nearest_monotonic_indices_avoids_frame_reuse() -> None:
    indices = nearest_monotonic_indices(
        np.asarray([0, 100, 200]),
        np.asarray([0, 10, 200]),
    )

    assert np.array_equal(indices, np.asarray([0, 1, 2]))


def test_downsample_episode_overwrites_source_stream_atomically(tmp_path: Path) -> None:
    path = tmp_path / "episode_0000.h5"
    source_timestamps = np.arange(0, 1_000_000, 33_333, dtype=np.int64)
    reference_timestamps = np.arange(0, 1_000_000, 40_000, dtype=np.int64)
    source_frames = np.arange(source_timestamps.size, dtype=np.uint8)[:, None, None, None]
    reference_frames = np.zeros(
        (reference_timestamps.size, 1, 1, 1), dtype=np.uint8
    )
    with h5py.File(path, "w") as h5:
        cameras = h5.create_group("cameras")
        side = cameras.create_group("side")
        side.create_dataset("frames", data=source_frames, compression="gzip")
        side.create_dataset("timestamp_us", data=source_timestamps)
        wrist = cameras.create_group("wrist")
        wrist.create_dataset("frames", data=reference_frames, compression="gzip")
        wrist.create_dataset("timestamp_us", data=reference_timestamps)

    alignment = downsample_episode(path, "side", "wrist", block_frames=4)

    with h5py.File(path, "r") as h5:
        side = h5["cameras/side"]
        assert side["frames"].shape[0] == reference_timestamps.size
        assert np.array_equal(
            side["frames"][:, 0, 0, 0],
            source_frames[alignment.source_indices, 0, 0, 0],
        )
        assert np.array_equal(side["timestamp_us"][:], alignment.source_timestamps_us)
        assert side.attrs["downsample_reference_timeline"] == (
            "cameras/wrist/timestamp_us"
        )
