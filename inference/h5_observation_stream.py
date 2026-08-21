"""Timestamped H5 observations for offline inference and simulation.

This module is deliberately independent from the hardware runtime.  It loads a
Nero multimodal episode into a small, immutable data object and exposes a
fixed-rate stream of observations.  The stream has two important properties:

* state samples are selected against an explicit regular tick timeline;
* camera samples are causal: a tick only sees the most recent frame whose
  timestamp is less than or equal to the tick timestamp.

The latter matters for replay/evaluation because selecting a nearest *future*
camera frame would silently give a policy information it did not have online.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


STATE_WIDTH = 7
WRENCH_WIDTH = 6
_DEFAULT_DATASETS = {
    "timestamp": "teleop/timestamp_us",
    "q": "teleop/q_follower",
    "dq": "teleop/dq_follower",
    "ddq": "teleop/ddq_follower",
    "tau": "teleop/tau_follower",
    "wrench": "teleop/wrench_ext",
}


def nearest_timestamp_indices(
    sorted_timestamps_us: np.ndarray,
    target_timestamps_us: np.ndarray,
) -> np.ndarray:
    """Return nearest source indices, resolving equal ties to the earlier one.

    ``sorted_timestamps_us`` must be strictly increasing.  Keeping this helper
    public makes it possible for a simulator or a test harness to use exactly
    the same tie-breaking rule as the stream.
    """

    source = _validate_timestamps(sorted_timestamps_us, "source timestamps")
    targets = np.asarray(target_timestamps_us, dtype=np.int64)
    right = np.searchsorted(source, targets, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - targets) < np.abs(
        source[left] - targets
    )
    return np.where(choose_right, right, left).astype(np.int64)


@dataclass(frozen=True)
class H5ObservationEpisode:
    """In-memory, timestamped source arrays from one H5 episode.

    All state vectors have already been reduced to one arm and share the
    ``state_timestamp_us`` axis.  ``camera_timestamp_us`` is independent and
    normally runs at a lower rate.  Arrays are validated when constructed by
    :meth:`from_h5` or :meth:`from_arrays`.
    """

    path: Path | None
    camera_name: str
    arm_name: str
    state_timestamp_us: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    wrench_ext: np.ndarray
    camera_timestamp_us: np.ndarray
    frames: np.ndarray
    camera_timestamp_us_by_name: Mapping[str, np.ndarray] | None = None
    camera_frames_by_name: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        state_ts = _validate_timestamps(self.state_timestamp_us, "state timestamps")
        camera_ts = _validate_timestamps(
            self.camera_timestamp_us, "camera timestamps"
        )
        object.__setattr__(self, "state_timestamp_us", state_ts)
        object.__setattr__(self, "camera_timestamp_us", camera_ts)

        count = state_ts.size
        for name in ("q", "dq", "ddq", "tau"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, STATE_WIDTH):
                raise ValueError(
                    f"{name} must have shape ({count}, {STATE_WIDTH}), got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value)

        wrench = np.asarray(self.wrench_ext, dtype=np.float64)
        if wrench.shape != (count, WRENCH_WIDTH):
            raise ValueError(
                f"wrench_ext must have shape ({count}, {WRENCH_WIDTH}), got {wrench.shape}"
            )
        if not np.all(np.isfinite(wrench)):
            raise ValueError("wrench_ext contains non-finite values")
        object.__setattr__(self, "wrench_ext", wrench)

        frames = np.asarray(self.frames)
        if (
            frames.ndim != 4
            or frames.shape[0] != camera_ts.size
            or frames.shape[-1] != 3
            or frames.shape[1] < 1
            or frames.shape[2] < 1
        ):
            raise ValueError(
                "frames must have shape [camera_count,H,W,3], "
                f"got {frames.shape} for camera_count={camera_ts.size}"
            )
        if frames.dtype != np.uint8:
            # H5 camera writers use uint8.  Normalizing here avoids surprising
            # float views in downstream DP preprocessing while accepting an
            # integer fixture from a test or an older recording.
            if not np.issubdtype(frames.dtype, np.integer):
                raise ValueError("camera frames must use an integer/uint8 dtype")
            if np.any(frames < 0) or np.any(frames > 255):
                raise ValueError("camera frame values must lie in [0, 255]")
            frames = frames.astype(np.uint8)
        object.__setattr__(self, "frames", frames)

        timestamps_by_name = self.camera_timestamp_us_by_name
        frames_by_name = self.camera_frames_by_name
        if timestamps_by_name is None:
            timestamps_by_name = {str(self.camera_name): camera_ts}
        else:
            timestamps_by_name = {
                str(name): _validate_timestamps(value, f"{name} camera timestamps")
                for name, value in timestamps_by_name.items()
            }
        if frames_by_name is None:
            frames_by_name = {str(self.camera_name): frames}
        else:
            frames_by_name = {
                str(name): np.asarray(value)
                for name, value in frames_by_name.items()
            }
        if set(timestamps_by_name) != set(frames_by_name):
            raise ValueError("camera timestamp/frame mappings must have the same keys")
        if self.camera_name not in timestamps_by_name:
            raise ValueError(
                f"camera mappings must include the primary camera {self.camera_name!r}"
            )
        for name, camera_timestamps in timestamps_by_name.items():
            camera_frames = frames_by_name[name]
            if (
                camera_frames.ndim != 4
                or camera_frames.shape[0] != camera_timestamps.size
                or camera_frames.shape[-1] != 3
                or camera_frames.shape[1] < 1
                or camera_frames.shape[2] < 1
            ):
                raise ValueError(
                    f"camera {name!r} frames must have shape [N,H,W,3] matching "
                    f"timestamps, got {camera_frames.shape}"
                )
            if camera_frames.dtype != np.uint8:
                if not np.issubdtype(camera_frames.dtype, np.integer):
                    raise ValueError(f"camera {name!r} frames must use an integer dtype")
                if np.any(camera_frames < 0) or np.any(camera_frames > 255):
                    raise ValueError(f"camera {name!r} frame values must lie in [0, 255]")
                camera_frames = camera_frames.astype(np.uint8)
            frames_by_name[name] = camera_frames
        object.__setattr__(self, "camera_timestamp_us_by_name", timestamps_by_name)
        object.__setattr__(self, "camera_frames_by_name", frames_by_name)

    @classmethod
    def from_arrays(
        cls,
        *,
        state_timestamp_us: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
        wrench_ext: np.ndarray,
        camera_timestamp_us: np.ndarray,
        frames: np.ndarray,
        camera_name: str = "camera",
        arm_name: str = "arm",
        path: Path | None = None,
        camera_streams: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "H5ObservationEpisode":
        """Build an episode without importing h5py (useful for tests)."""

        timestamp_values = None
        frame_values = None
        if camera_streams is not None:
            timestamp_values = {
                str(name): np.asarray(values[0], dtype=np.int64)
                for name, values in camera_streams.items()
            }
            frame_values = {
                str(name): np.asarray(values[1])
                for name, values in camera_streams.items()
            }
        return cls(
            path=path,
            camera_name=str(camera_name),
            arm_name=str(arm_name),
            state_timestamp_us=np.asarray(state_timestamp_us, dtype=np.int64),
            q=np.asarray(q, dtype=np.float64),
            dq=np.asarray(dq, dtype=np.float64),
            ddq=np.asarray(ddq, dtype=np.float64),
            tau=np.asarray(tau, dtype=np.float64),
            wrench_ext=np.asarray(wrench_ext, dtype=np.float64),
            camera_timestamp_us=np.asarray(camera_timestamp_us, dtype=np.int64),
            frames=np.asarray(frames),
            camera_timestamp_us_by_name=timestamp_values,
            camera_frames_by_name=frame_values,
        )

    @classmethod
    def from_h5(
        cls,
        path: str | Path,
        *,
        camera_name: str,
        camera_names: Sequence[str] | None = None,
        arm_name: str | None = None,
        arm_index: int = 0,
        datasets: Mapping[str, str] | None = None,
        allow_wrench_aliases: bool = True,
        derive_ddq_if_missing: bool = True,
    ) -> "H5ObservationEpisode":
        """Load and validate one camera/arm view from a Nero episode H5.

        ``datasets`` may override any of ``timestamp``, ``q``, ``dq``,
        ``ddq``, ``tau`` and ``wrench``.  The camera group always uses
        ``cameras/<camera_name>/{frames,timestamp_us}``.

        Some recordings produced before the acceleration channel was added
        contain ``q_follower`` and ``dq_follower`` but no ``ddq_follower``.
        By default, ``ddq`` is then derived with a causal, timestamp-aware
        backward difference of ``dq`` (the first sample is zero).  Set
        ``derive_ddq_if_missing=False`` to enforce the strict checkpoint input
        contract instead.
        """

        source = Path(path).expanduser().resolve()
        try:
            import h5py
        except (ImportError, OSError, ValueError) as exc:  # NumPy/h5py ABI is common on servers
            raise RuntimeError(
                "loading H5 observations requires a working h5py installation"
            ) from exc

        names = dict(_DEFAULT_DATASETS)
        if datasets is not None:
            names.update({str(key): str(value) for key, value in datasets.items()})
        with h5py.File(source, "r") as h5:
            requested_cameras = tuple(
                dict.fromkeys(
                    str(name)
                    for name in (camera_names or (camera_name,))
                )
            )
            if camera_name not in requested_cameras:
                requested_cameras = (str(camera_name), *requested_cameras)
            camera_groups = {}
            for requested_camera in requested_cameras:
                camera_path = f"cameras/{requested_camera}"
                if camera_path not in h5:
                    available = (
                        sorted(str(name) for name in h5["cameras"].keys())
                        if "cameras" in h5
                        else []
                    )
                    raise ValueError(
                        f"camera {requested_camera!r} not found in {source.name}; "
                        f"available={available}"
                    )
                camera_group = h5[camera_path]
                for key in ("frames", "timestamp_us"):
                    if key not in camera_group:
                        raise ValueError(f"{camera_path}/{key} is missing from {source.name}")
                camera_groups[requested_camera] = camera_group
            if camera_name not in camera_groups:
                available = (
                    sorted(str(name) for name in h5["cameras"].keys())
                    if "cameras" in h5
                    else []
                )
                raise ValueError(
                    f"camera {camera_name!r} not found in {source.name}; "
                    f"available={available}"
                )

            # A few early episodes used one of the calibrated/predicted wrench
            # names.  Keep the fallback opt-out so a strict evaluator can force
            # the canonical ``wrench_ext`` contract.
            wrench_path = names["wrench"]
            if wrench_path not in h5 and allow_wrench_aliases and wrench_path == _DEFAULT_DATASETS["wrench"]:
                for candidate in ("teleop/wrench_cal", "teleop/wrench_pred"):
                    if candidate in h5:
                        wrench_path = candidate
                        break

            ddq_path = names["ddq"] if names["ddq"] in h5 else None
            required = {
                "timestamp": names["timestamp"],
                "q": names["q"],
                "dq": names["dq"],
                "tau": names["tau"],
                "wrench": wrench_path,
            }
            if ddq_path is None and not derive_ddq_if_missing:
                required["ddq"] = names["ddq"]
            missing = [value for value in required.values() if value not in h5]
            if missing:
                raise ValueError(f"{source.name} is missing inference datasets: {missing}")

            state_timestamps = _timestamp_vector(
                np.asarray(h5[required["timestamp"]][:], dtype=np.int64),
                required["timestamp"],
            )
            count = state_timestamps.size

            arm_names = _h5_arm_names(h5)
            if not arm_names:
                arm_names = (str(arm_name) if arm_name is not None else "arm",)
            if arm_name is not None:
                if arm_name not in arm_names:
                    raise ValueError(
                        f"arm {arm_name!r} not found in teleop.arm_names={list(arm_names)}"
                    )
                selected_arm_index = arm_names.index(arm_name)
                selected_arm_name = str(arm_name)
            else:
                selected_arm_index = int(arm_index)
                if selected_arm_index < 0 or selected_arm_index >= len(arm_names):
                    raise ValueError(
                        f"arm_index={selected_arm_index} outside arm_names={list(arm_names)}"
                    )
                selected_arm_name = str(arm_names[selected_arm_index])

            q = _select_arm_matrix(
                np.asarray(h5[required["q"]][:], dtype=np.float64),
                count,
                STATE_WIDTH,
                arm_names,
                selected_arm_index,
                required["q"],
            )
            dq = _select_arm_matrix(
                np.asarray(h5[required["dq"]][:], dtype=np.float64),
                count,
                STATE_WIDTH,
                arm_names,
                selected_arm_index,
                required["dq"],
            )
            if ddq_path is None:
                ddq = _derive_ddq(dq, state_timestamps)
            else:
                ddq = _select_arm_matrix(
                    np.asarray(h5[ddq_path][:], dtype=np.float64),
                    count,
                    STATE_WIDTH,
                    arm_names,
                    selected_arm_index,
                    ddq_path,
                )
            tau = _select_arm_matrix(
                np.asarray(h5[required["tau"]][:], dtype=np.float64),
                count,
                STATE_WIDTH,
                arm_names,
                selected_arm_index,
                required["tau"],
            )
            wrench = _select_arm_matrix(
                np.asarray(h5[required["wrench"]][:], dtype=np.float64),
                count,
                WRENCH_WIDTH,
                arm_names,
                selected_arm_index,
                required["wrench"],
            )
            camera_timestamps_by_name = {}
            frames_by_name = {}
            for requested_camera, camera_group in camera_groups.items():
                camera_path = f"cameras/{requested_camera}"
                camera_timestamps = _timestamp_vector(
                    np.asarray(camera_group["timestamp_us"][:], dtype=np.int64),
                    f"{camera_path}/timestamp_us",
                )
                camera_frames = np.asarray(camera_group["frames"][:])
                if camera_frames.shape[0] != camera_timestamps.size:
                    raise ValueError(
                        f"{camera_path}/frames and timestamp_us lengths differ: "
                        f"{camera_frames.shape[0]} != {camera_timestamps.size}"
                    )
                camera_timestamps_by_name[requested_camera] = camera_timestamps
                frames_by_name[requested_camera] = camera_frames
            camera_timestamps = camera_timestamps_by_name[str(camera_name)]
            frames = frames_by_name[str(camera_name)]

        return cls.from_arrays(
            path=source,
            camera_name=camera_name,
            arm_name=selected_arm_name,
            state_timestamp_us=state_timestamps,
            q=q,
            dq=dq,
            ddq=ddq,
            tau=tau,
            wrench_ext=wrench,
            camera_timestamp_us=camera_timestamps,
            frames=frames,
            camera_streams={
                name: (camera_timestamps_by_name[name], frames_by_name[name])
                for name in camera_timestamps_by_name
            },
        )

    @property
    def state_count(self) -> int:
        return int(self.state_timestamp_us.size)

    @property
    def camera_count(self) -> int:
        return int(self.camera_timestamp_us.size)

    @property
    def time_s(self) -> np.ndarray:
        return (self.state_timestamp_us - int(self.state_timestamp_us[0])).astype(
            np.float64
        ) * 1.0e-6


@dataclass(frozen=True)
class H5ObservationTick:
    """One fixed-rate observation and its causal history windows."""

    tick_index: int
    timestamp_us: int
    timestamp_s: float
    state_index: int
    state_timestamp_us: int
    state_alignment_gap_us: int
    camera_index: int
    camera_timestamp_us: int
    camera_age_us: int
    camera_is_padded: bool
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    wrench_ext: np.ndarray
    image: np.ndarray
    q_history: np.ndarray
    dq_history: np.ndarray
    ddq_history: np.ndarray
    tau_history: np.ndarray
    wrench_history: np.ndarray
    image_history: np.ndarray
    state_history_indices: np.ndarray
    camera_history_indices: np.ndarray
    image_histories: Mapping[str, np.ndarray] | None = None

    @property
    def wrench(self) -> np.ndarray:
        """Alias used by the runtime's ``InferenceInput`` contract."""

        return self.wrench_ext


class H5ObservationStream(Sequence[H5ObservationTick]):
    """Replay an episode on a regular state clock.

    Parameters
    ----------
    state_rate_hz:
        Regular state/control tick rate.  The default is 100 Hz.
    camera_rate_hz:
        Nominal camera rate used only to choose the default camera history
        spacing.  Current-frame selection always uses recorded timestamps.
    history_steps:
        Number of state history entries returned per tick (e.g. 50 for WM).
    camera_history_steps:
        Number of camera entries returned in ``image_history``.
    camera_history_step_s:
        Time spacing between camera history entries.  ``None`` means
        ``1 / camera_rate_hz``; this may be set to 0.1 for a DP 10 Hz image
        contract while still running state ticks at 100 Hz.
    left_pad:
        Repeat the first available state/frame when a history window reaches
        before the beginning.  If false, the stream starts only once all
        requested histories and a camera frame are available.
    state_alignment:
        ``"nearest"`` (default, tie goes earlier) or causal ``"previous"``.
    """

    def __init__(
        self,
        episode: H5ObservationEpisode,
        *,
        state_rate_hz: float = 100.0,
        camera_rate_hz: float = 25.0,
        history_steps: int = 50,
        camera_history_steps: int = 1,
        camera_history_step_s: float | None = None,
        left_pad: bool = True,
        state_alignment: str = "nearest",
        max_state_alignment_gap_us: int | None = None,
        max_camera_age_us: int | None = None,
        start_timestamp_us: int | None = None,
        stop_timestamp_us: int | None = None,
    ) -> None:
        if not np.isfinite(state_rate_hz) or state_rate_hz <= 0.0:
            raise ValueError("state_rate_hz must be positive and finite")
        if not np.isfinite(camera_rate_hz) or camera_rate_hz <= 0.0:
            raise ValueError("camera_rate_hz must be positive and finite")
        if int(history_steps) < 1 or int(camera_history_steps) < 1:
            raise ValueError("history_steps and camera_history_steps must be positive")
        if state_alignment not in {"nearest", "previous"}:
            raise ValueError("state_alignment must be 'nearest' or 'previous'")
        if max_state_alignment_gap_us is not None and int(max_state_alignment_gap_us) < 0:
            raise ValueError("max_state_alignment_gap_us must be non-negative")
        if max_camera_age_us is not None and int(max_camera_age_us) < 0:
            raise ValueError("max_camera_age_us must be non-negative")
        if camera_history_step_s is None:
            camera_history_step_s = 1.0 / float(camera_rate_hz)
        if not np.isfinite(camera_history_step_s) or camera_history_step_s <= 0.0:
            raise ValueError("camera_history_step_s must be positive and finite")

        self.episode = episode
        self.state_rate_hz = float(state_rate_hz)
        self.camera_rate_hz = float(camera_rate_hz)
        self.history_steps = int(history_steps)
        self.camera_history_steps = int(camera_history_steps)
        self.camera_history_step_s = float(camera_history_step_s)
        self.left_pad = bool(left_pad)
        self.state_alignment = state_alignment
        self.max_state_alignment_gap_us = (
            None if max_state_alignment_gap_us is None else int(max_state_alignment_gap_us)
        )
        self.max_camera_age_us = (
            None if max_camera_age_us is None else int(max_camera_age_us)
        )

        self._tick_period_us = 1.0e6 / self.state_rate_hz
        source_start = int(episode.state_timestamp_us[0])
        source_end = int(episode.state_timestamp_us[-1])
        start_us = source_start if start_timestamp_us is None else int(start_timestamp_us)
        stop_us = source_end if stop_timestamp_us is None else int(stop_timestamp_us)
        if stop_us < start_us:
            raise ValueError("stop_timestamp_us must be >= start_timestamp_us")
        # Round each tick from the origin instead of accumulating a float period;
        # this keeps long episodes on the intended 100 Hz integer grid.
        count = int(np.floor((stop_us - start_us) / self._tick_period_us + 1e-9)) + 1
        all_ticks = start_us + np.rint(
            np.arange(count, dtype=np.float64) * self._tick_period_us
        ).astype(np.int64)
        all_ticks = np.unique(all_ticks)
        if all_ticks.size == 0:
            raise ValueError("selected state tick range is empty")

        state_indices, state_gaps = self._align_states(all_ticks)
        camera_indices, camera_ages, camera_padded = self._align_cameras(all_ticks)

        valid = np.ones(all_ticks.size, dtype=bool)
        if not self.left_pad:
            valid &= state_indices >= 0
            valid &= camera_indices >= 0
            valid &= np.arange(all_ticks.size) >= self.history_steps - 1
            valid &= np.arange(all_ticks.size) >= self.camera_history_steps - 1
            # Do not let a custom start timestamp create a history containing
            # the sentinel ``-1`` from causal state alignment.
            state_valid = state_indices >= 0
            complete_state_history = np.ones(all_ticks.size, dtype=bool)
            for tick_index in range(all_ticks.size):
                history_start = max(0, tick_index - self.history_steps + 1)
                complete_state_history[tick_index] = bool(
                    np.all(state_valid[history_start : tick_index + 1])
                )
            valid &= complete_state_history
            # A camera history is measured in timestamp space, not in state
            # tick count.  At 100 Hz a 25 Hz two-frame history spans 40 ms,
            # hence the first usable tick may be later than index ``1``.
            camera_history_span_us = int(
                round((self.camera_history_steps - 1) * self.camera_history_step_s * 1.0e6)
            )
            valid &= all_ticks >= int(self.episode.camera_timestamp_us[0]) + camera_history_span_us
        if not np.any(valid):
            raise ValueError("selected observation tick range has no complete samples")

        # Keep the positions on the untrimmed regular clock.  In non-padded
        # mode the public stream may begin at tick 40 (after a camera history
        # warm-up), but its first state history still needs ticks 20, 30, 40.
        self._all_timestamps_us = all_ticks
        self._all_state_indices = state_indices
        self._all_state_gaps_us = state_gaps
        self._all_camera_indices = camera_indices
        self._all_camera_ages_us = camera_ages
        self._all_camera_padded = camera_padded
        self._valid_tick_positions = np.flatnonzero(valid).astype(np.int64)
        self._timestamps_us = all_ticks[valid]
        self._state_indices = state_indices[valid]
        self._state_gaps_us = state_gaps[valid]
        self._camera_indices = camera_indices[valid]
        self._camera_ages_us = camera_ages[valid]
        self._camera_padded = camera_padded[valid]
        self._validate_gaps()

    def _align_states(self, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = self.episode.state_timestamp_us
        if self.state_alignment == "nearest":
            indices = nearest_timestamp_indices(source, ticks)
            if not self.left_pad:
                outside = (ticks < source[0]) | (ticks > source[-1])
                indices = np.where(outside, -1, indices)
        else:
            indices = np.searchsorted(source, ticks, side="right").astype(np.int64) - 1
            if self.left_pad:
                indices = np.maximum(indices, 0)
        valid_indices = np.clip(indices, 0, source.size - 1)
        gaps = np.abs(source[valid_indices] - ticks).astype(np.int64)
        # ``-1`` marks ticks preceding the first source sample in non-padded
        # causal mode; retain its gap for diagnostics after filtering.
        if self.state_alignment == "previous" and not self.left_pad:
            gaps = np.where(indices < 0, np.iinfo(np.int64).max, gaps)
        return indices, gaps

    def _align_cameras(self, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.episode.camera_timestamp_us
        indices = np.searchsorted(source, ticks, side="right").astype(np.int64) - 1
        padded = indices < 0
        if self.left_pad:
            indices = np.maximum(indices, 0)
        valid_indices = np.clip(indices, 0, source.size - 1)
        ages = (ticks - source[valid_indices]).astype(np.int64)
        if not self.left_pad:
            ages = np.where(padded, np.iinfo(np.int64).max, ages)
        return indices, ages, padded

    def _validate_gaps(self) -> None:
        if self.max_state_alignment_gap_us is not None:
            bad = self._state_gaps_us > self.max_state_alignment_gap_us
            if np.any(bad):
                index = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    "state/tick alignment exceeds limit: "
                    f"tick={int(self._timestamps_us[index])}, "
                    f"gap={int(self._state_gaps_us[index])} us, "
                    f"limit={self.max_state_alignment_gap_us} us"
                )
        if self.max_camera_age_us is not None:
            # Negative age means the first frame was used as an explicit left
            # pad; it is not a stale camera sample and is exempt from the age
            # limit.
            bad = (self._camera_ages_us >= 0) & (
                self._camera_ages_us > self.max_camera_age_us
            )
            if np.any(bad):
                index = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    "camera frame is too old for state tick: "
                    f"tick={int(self._timestamps_us[index])}, "
                    f"age={int(self._camera_ages_us[index])} us, "
                    f"limit={self.max_camera_age_us} us"
                )

    def __len__(self) -> int:
        return int(self._timestamps_us.size)

    def __iter__(self) -> Iterator[H5ObservationTick]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int | slice) -> H5ObservationTick | tuple[H5ObservationTick, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        state_index = int(self._state_indices[position])
        camera_index = int(self._camera_indices[position])
        global_position = int(self._valid_tick_positions[position])
        history_positions = np.arange(
            global_position - self.history_steps + 1,
            global_position + 1,
            dtype=np.int64,
        )
        state_history_positions = np.clip(
            history_positions, 0, self._all_state_indices.size - 1
        )
        camera_history_targets = self._timestamps_us[position] - np.arange(
            self.camera_history_steps - 1,
            -1,
            -1,
            dtype=np.float64,
        ) * self.camera_history_step_s * 1.0e6
        camera_history_indices = self._camera_indices_for_targets(
            camera_history_targets.astype(np.int64)
        )
        image_histories = {
            name: self.episode.camera_frames_by_name[name][
                self._camera_indices_for_targets(
                    camera_history_targets.astype(np.int64),
                    timestamps=self.episode.camera_timestamp_us_by_name[name],
                )
            ]
            for name in self.episode.camera_timestamp_us_by_name
        }
        tick_us = int(self._timestamps_us[position])
        camera_ts = int(self.episode.camera_timestamp_us[camera_index])
        state_ts = int(self.episode.state_timestamp_us[state_index])
        return H5ObservationTick(
            tick_index=position,
            timestamp_us=tick_us,
            timestamp_s=(tick_us - int(self._timestamps_us[0])) * 1.0e-6,
            state_index=state_index,
            state_timestamp_us=state_ts,
            state_alignment_gap_us=int(self._state_gaps_us[position]),
            camera_index=camera_index,
            camera_timestamp_us=camera_ts,
            camera_age_us=int(self._camera_ages_us[position]),
            camera_is_padded=bool(self._camera_padded[position]),
            q=self.episode.q[state_index],
            dq=self.episode.dq[state_index],
            ddq=self.episode.ddq[state_index],
            tau=self.episode.tau[state_index],
            wrench_ext=self.episode.wrench_ext[state_index],
            image=self.episode.frames[camera_index],
            q_history=self.episode.q[
                self._all_state_indices[state_history_positions]
            ],
            dq_history=self.episode.dq[
                self._all_state_indices[state_history_positions]
            ],
            ddq_history=self.episode.ddq[
                self._all_state_indices[state_history_positions]
            ],
            tau_history=self.episode.tau[
                self._all_state_indices[state_history_positions]
            ],
            wrench_history=self.episode.wrench_ext[
                self._all_state_indices[state_history_positions]
            ],
            image_history=self.episode.frames[camera_history_indices],
            state_history_indices=self._all_state_indices[state_history_positions].copy(),
            camera_history_indices=camera_history_indices.copy(),
            image_histories=image_histories,
        )

    def _camera_indices_for_targets(
        self,
        targets_us: np.ndarray,
        *,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        source = self.episode.camera_timestamp_us if timestamps is None else timestamps
        indices = np.searchsorted(source, targets_us, side="right").astype(np.int64) - 1
        if self.left_pad:
            indices = np.maximum(indices, 0)
        else:
            # The stream itself starts after the first camera frame in this
            # mode, so a negative result should be impossible.  Keep the
            # explicit guard to make malformed custom slices fail loudly.
            if np.any(indices < 0):
                raise RuntimeError("camera history reached before first frame")
        return indices

    @property
    def timestamps_us(self) -> np.ndarray:
        return self._timestamps_us.copy()

    @property
    def state_indices(self) -> np.ndarray:
        return self._state_indices.copy()

    @property
    def camera_indices(self) -> np.ndarray:
        return self._camera_indices.copy()

    @property
    def state_alignment_gaps_us(self) -> np.ndarray:
        return self._state_gaps_us.copy()

    @property
    def camera_ages_us(self) -> np.ndarray:
        return self._camera_ages_us.copy()


def load_h5_observation_stream(
    path: str | Path,
    *,
    camera_name: str,
    arm_name: str | None = None,
    arm_index: int = 0,
    datasets: Mapping[str, str] | None = None,
    allow_wrench_aliases: bool = True,
    derive_ddq_if_missing: bool = True,
    **stream_kwargs: object,
) -> H5ObservationStream:
    """Convenience constructor combining :meth:`from_h5` and the stream."""

    episode = H5ObservationEpisode.from_h5(
        path,
        camera_name=camera_name,
        arm_name=arm_name,
        arm_index=arm_index,
        datasets=datasets,
        allow_wrench_aliases=allow_wrench_aliases,
        derive_ddq_if_missing=derive_ddq_if_missing,
    )
    return H5ObservationStream(episode, **stream_kwargs)


def _validate_timestamps(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size < 1 or np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ValueError(
            f"{name} must contain positive, strictly increasing timestamps"
        )
    return result


def _timestamp_vector(value: np.ndarray, name: str) -> np.ndarray:
    return _validate_timestamps(value, name)


def _decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _h5_arm_names(h5: object) -> tuple[str, ...]:
    """Read current teleop attrs and the legacy metadata JSON field."""

    teleop = h5["teleop"] if "teleop" in h5 else None
    values = () if teleop is None else teleop.attrs.get("arm_names", ())
    if np.asarray(values).size:
        if isinstance(values, (str, bytes, np.str_, np.bytes_)):
            values = (values,)
        return tuple(_decode_h5_string(value) for value in values)
    metadata_path = "metadata/arm_names_json"
    if metadata_path not in h5:
        return ()
    raw = _decode_h5_string(h5[metadata_path][()])
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return ()
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return ()
    return tuple(str(value) for value in parsed if str(value))


def _select_arm_matrix(
    value: np.ndarray,
    count: int,
    width: int,
    arm_names: Sequence[str],
    arm_index: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    arm_count = len(arm_names)
    if result.shape == (count, arm_count * width):
        result = result[:, arm_index * width : (arm_index + 1) * width]
    elif result.shape == (count, arm_count, width):
        result = result[:, arm_index]
    elif arm_count == 1 and result.shape == (count, width):
        pass
    else:
        raise ValueError(
            f"{name} does not match {arm_count} arm(s) of width {width}: {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _derive_ddq(dq: np.ndarray, timestamps_us: np.ndarray) -> np.ndarray:
    """Derive joint acceleration from a timestamped velocity signal.

    H5 state samples are not guaranteed to be exactly periodic, so using a
    fixed ``state_rate_hz`` here would introduce a scale error whenever a
    sample is delayed or dropped.  A backward difference is used instead of a
    centered gradient so a replay tick never consumes a future velocity sample
    that would not have been available to the online state estimator.
    """

    velocity = np.asarray(dq, dtype=np.float64)
    timestamps = _validate_timestamps(timestamps_us, "state timestamps")
    if velocity.ndim != 2 or velocity.shape[0] != timestamps.size:
        raise ValueError(
            "cannot derive ddq: dq must have shape (N, 7), "
            f"got {velocity.shape} for N={timestamps.size}"
        )
    if velocity.shape[1] != STATE_WIDTH or not np.all(np.isfinite(velocity)):
        raise ValueError("cannot derive ddq: dq must be a finite (N, 7) array")
    if velocity.shape[0] == 1:
        return np.zeros_like(velocity)
    time_s = (timestamps - timestamps[0]).astype(np.float64) * 1.0e-6
    derived = np.zeros_like(velocity)
    derived[1:] = np.diff(velocity, axis=0) / np.diff(time_s)[:, None]
    if not np.all(np.isfinite(derived)):
        raise ValueError("derived ddq contains non-finite values")
    return np.asarray(derived, dtype=np.float64)


__all__ = [
    "H5ObservationEpisode",
    "H5ObservationStream",
    "H5ObservationTick",
    "load_h5_observation_stream",
    "nearest_timestamp_indices",
]
