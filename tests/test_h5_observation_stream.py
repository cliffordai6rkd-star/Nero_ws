from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.h5_observation_stream import (
    H5ObservationEpisode,
    H5ObservationStream,
    load_h5_observation_stream,
    nearest_timestamp_indices,
)
from inference.h5_observation_stream import _derive_ddq, _h5_arm_names


def _episode(
    *,
    state_timestamps: np.ndarray | None = None,
    camera_timestamps: np.ndarray | None = None,
    arm_count: int = 1,
) -> H5ObservationEpisode:
    state_timestamps = (
        np.asarray(state_timestamps, dtype=np.int64)
        if state_timestamps is not None
        else 1_000_000 + np.arange(8, dtype=np.int64) * 10_000
    )
    camera_timestamps = (
        np.asarray(camera_timestamps, dtype=np.int64)
        if camera_timestamps is not None
        else 1_000_000 + np.arange(3, dtype=np.int64) * 40_000
    )
    count = len(state_timestamps)
    q = np.arange(count * 7, dtype=np.float64).reshape(count, 7)
    return H5ObservationEpisode.from_arrays(
        state_timestamp_us=state_timestamps,
        q=q,
        dq=q + 100,
        ddq=q + 200,
        tau=q + 300,
        wrench_ext=np.arange(count * 6, dtype=np.float64).reshape(count, 6),
        camera_timestamp_us=camera_timestamps,
        frames=np.arange(len(camera_timestamps) * 2 * 3 * 3, dtype=np.uint8).reshape(
            len(camera_timestamps), 2, 3, 3
        ),
    )


def test_nearest_timestamp_tie_selects_earlier_sample() -> None:
    source = np.array([1_000, 2_000, 3_000], dtype=np.int64)
    targets = np.array([500, 1_500, 2_600, 4_000], dtype=np.int64)
    np.testing.assert_array_equal(
        nearest_timestamp_indices(source, targets), [0, 0, 2, 2]
    )


def test_stream_uses_regular_100hz_ticks_and_causal_latest_camera() -> None:
    episode = _episode(
        state_timestamps=1_000_000 + np.arange(5, dtype=np.int64) * 10_000,
        camera_timestamps=np.array([1_000_000, 1_040_000], dtype=np.int64),
    )
    stream = H5ObservationStream(
        episode,
        state_rate_hz=100.0,
        history_steps=1,
        camera_history_steps=1,
    )

    np.testing.assert_array_equal(
        stream.timestamps_us,
        1_000_000 + np.arange(5, dtype=np.int64) * 10_000,
    )
    # The second frame is not visible until its timestamp, even though it is
    # the nearest frame to the preceding tick.
    np.testing.assert_array_equal(stream.camera_indices, [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(stream.camera_ages_us, [0, 10_000, 20_000, 30_000, 0])


def test_history_window_left_pads_first_state_and_camera() -> None:
    stream = H5ObservationStream(
        _episode(),
        history_steps=4,
        camera_history_steps=3,
        camera_history_step_s=0.04,
        left_pad=True,
    )
    first = stream[0]
    np.testing.assert_array_equal(first.state_history_indices, [0, 0, 0, 0])
    np.testing.assert_array_equal(first.camera_history_indices, [0, 0, 0])
    np.testing.assert_array_equal(first.q_history, np.repeat(first.q[None], 4, axis=0))
    assert first.camera_is_padded is False


def test_without_left_pad_stream_starts_after_history_and_camera() -> None:
    stream = H5ObservationStream(
        _episode(),
        history_steps=3,
        camera_history_steps=2,
        camera_history_step_s=0.04,
        left_pad=False,
    )
    # Camera frame 1 arrives at 40 ms and the three-state history is complete
    # at 20 ms, so the first usable tick is 40 ms.
    assert stream[0].timestamp_us == 1_040_000
    np.testing.assert_array_equal(stream[0].state_history_indices, [2, 3, 4])
    np.testing.assert_array_equal(stream[0].camera_history_indices, [0, 1])


def test_previous_state_alignment_never_uses_a_future_state() -> None:
    episode = _episode(
        state_timestamps=np.array([1_000_000, 1_015_000, 1_030_000], dtype=np.int64),
        camera_timestamps=np.array([1_000_000], dtype=np.int64),
    )
    stream = H5ObservationStream(
        episode,
        state_rate_hz=100.0,
        history_steps=1,
        camera_history_steps=1,
        state_alignment="previous",
    )
    np.testing.assert_array_equal(stream.state_indices, [0, 0, 1, 2])
    assert np.all(
        episode.state_timestamp_us[stream.state_indices]
        <= stream.timestamps_us
    )


def test_alignment_gap_limit_is_enforced() -> None:
    episode = _episode(
        state_timestamps=np.array([1_000_000, 1_100_000], dtype=np.int64),
        camera_timestamps=np.array([1_000_000], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="state/tick alignment exceeds limit"):
        H5ObservationStream(
            episode,
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            max_state_alignment_gap_us=1,
        )


def test_missing_ddq_fallback_is_causal_and_timestamp_aware() -> None:
    timestamps = np.array([1_000_000, 1_020_000, 1_050_000], dtype=np.int64)
    dq = np.array([[0.0] * 7, [2.0] * 7, [5.0] * 7])
    ddq = _derive_ddq(dq, timestamps)
    np.testing.assert_allclose(ddq[0], 0.0)
    np.testing.assert_allclose(ddq[1], 100.0)
    np.testing.assert_allclose(ddq[2], 100.0)


def test_arm_names_can_be_read_from_legacy_metadata_json() -> None:
    class _Dataset:
        def __getitem__(self, key):
            assert key == ()
            return '["main", "aux"]'

    class _H5:
        def __contains__(self, key):
            return key in {"metadata/arm_names_json"}

        def __getitem__(self, key):
            if key == "teleop":
                class _Teleop:
                    attrs = {}

                return _Teleop()
            return _Dataset()

    assert _h5_arm_names(_H5()) == ("main", "aux")


def test_h5_loader_selects_arm_and_rejects_missing_ddq(tmp_path: Path) -> None:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        pytest.skip(f"h5py unavailable: {exc}")

    path = tmp_path / "episode.h5"
    timestamps = 1_000_000 + np.arange(3, dtype=np.int64) * 10_000
    q = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)
    with h5py.File(path, "w") as h5:
        teleop = h5.create_group("teleop")
        teleop.create_dataset("timestamp_us", data=timestamps)
        for name, value in (
            ("q_follower", q),
            ("dq_follower", q + 1),
            ("tau_follower", q + 3),
            ("wrench_ext", np.arange(3 * 12, dtype=np.float64).reshape(3, 12)),
        ):
            teleop.create_dataset(name, data=value)
        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "arm_names_json",
            data='["left", "right"]',
            dtype=h5py.string_dtype(),
        )
        camera = h5.create_group("cameras/wrist")
        camera.create_dataset("timestamp_us", data=[1_000_000, 1_020_000])
        camera.create_dataset("frames", data=np.zeros((2, 2, 2, 3), dtype=np.uint8))

    stream = load_h5_observation_stream(
        path,
        camera_name="wrist",
        arm_name="right",
        history_steps=1,
        camera_history_steps=1,
    )
    np.testing.assert_array_equal(stream.episode.q[0], q[0, 7:14])
    np.testing.assert_array_equal(stream.episode.wrench_ext[0], np.arange(6, 12))
    assert np.isfinite(stream.episode.ddq).all()
    np.testing.assert_allclose(stream.episode.ddq[0], 0.0)
    np.testing.assert_allclose(
        stream.episode.ddq[1:],
        np.diff(q[:, 7:14], axis=0) / 0.01,
    )
    with pytest.raises(ValueError, match="ddq_follower"):
        H5ObservationEpisode.from_h5(
            path,
            camera_name="wrist",
            arm_name="right",
            derive_ddq_if_missing=False,
        )
