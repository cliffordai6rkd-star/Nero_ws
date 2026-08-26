"""Observation history and timestamp alignment for diffusion-policy inference.

The legacy Nero pipeline mixed camera buffering, CAN history, timestamp
alignment and model execution in one class.  ``DPObservationBuffer`` owns only
the observation contract.  It is deliberately independent of torch, robot
controllers and world-model implementations, so the same stage can be reused
by online and playback runtimes.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _nearest_timestamp_indices(
    sorted_timestamps: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    source = np.asarray(sorted_timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if source.size == 0 or np.any(np.diff(source) <= 0.0):
        raise ValueError("observation timestamps must be non-empty and increasing")
    right = np.searchsorted(source, values, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - values) < np.abs(source[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def _previous_timestamp_indices(
    sorted_timestamps: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Select the latest source sample not newer than each target."""
    source = np.asarray(sorted_timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if source.size == 0 or np.any(np.diff(source) <= 0.0):
        raise ValueError("observation timestamps must be non-empty and increasing")
    # Timestamps are integer microseconds represented as seconds.  Allow one
    # microsecond of roundoff when a target is exactly on a source tick.
    return (np.searchsorted(source, values + 1.0e-9, side="right") - 1).astype(
        np.int64
    )


class DPObservationBuffer:
    """Maintain model-ready image and wrench histories.

    ``observation_step_s`` may be a value or a callback because checkpoint
    metadata is discovered after model restoration.  ``on_continuous_can`` is
    called by ``append_continuous_can_observation`` for optional world-model
    history maintenance; the callback is outside this class's responsibility.
    """

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...],
        anchor_image_key: str,
        n_obs_steps: int,
        wrench_history_steps: int = 1,
        uses_wrench_observation: bool = True,
        observation_step_s: float | Callable[[], float | None] | None = None,
        inference_mode: str = "asynchronous",
        maximum_alignment_gap_s: float = 0.03,
        on_continuous_can: Callable[[float, Mapping[str, np.ndarray]], None]
        | None = None,
    ) -> None:
        keys = tuple(sorted(dict.fromkeys(str(key) for key in image_keys)))
        if not keys:
            raise ValueError("DP observation buffer requires at least one image key")
        if str(anchor_image_key) not in keys:
            raise ValueError(
                f"anchor image key {anchor_image_key!r} is not in image keys {list(keys)}"
            )
        if int(n_obs_steps) < 1 or int(wrench_history_steps) < 1:
            raise ValueError("observation and wrench history steps must be positive")
        gap = float(maximum_alignment_gap_s)
        if not np.isfinite(gap) or gap < 0.0:
            raise ValueError("maximum_alignment_gap_s must be finite and non-negative")
        self.image_keys = keys
        self.anchor_image_key = str(anchor_image_key)
        self.n_obs_steps = int(n_obs_steps)
        self.wrench_history_steps = int(wrench_history_steps)
        self.uses_wrench_observation = bool(uses_wrench_observation)
        self._observation_step_s_source = observation_step_s
        self.inference_mode = str(inference_mode).strip().lower()
        self.maximum_alignment_gap_s = gap
        self.on_continuous_can = on_continuous_can

        self._images_by_key = {
            key: deque(maxlen=self.n_obs_steps) for key in self.image_keys
        }
        self._images = self._images_by_key[self.anchor_image_key]
        self._wrenches: deque[np.ndarray] = deque(
            maxlen=self.n_obs_steps * self.wrench_history_steps
        )
        self._timed_images_by_key = {
            key: deque(maxlen=256) for key in self.image_keys
        }
        self._timed_images = self._timed_images_by_key[self.anchor_image_key]
        self._timed_wrenches: deque[tuple[float, np.ndarray]] = deque(maxlen=2048)
        self._camera_alignment_logged = False
        self._dp_snapshot_pending_reason: str | None = None
        self._latest_policy_anchor_s: float | None = None
        self._last_submitted_policy_anchor_s: float | None = None
        self._open_loop_observation_start_s: float | None = None

    @property
    def observation_step_s(self) -> float | None:
        value = self._observation_step_s_source
        value = value() if callable(value) else value
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value

    @property
    def dp_snapshot_pending_reason(self) -> str | None:
        return self._dp_snapshot_pending_reason

    def clear(self, *, start_s: float | None = None) -> None:
        for images in self._images_by_key.values():
            images.clear()
        self._wrenches.clear()
        for images in self._timed_images_by_key.values():
            images.clear()
        self._timed_wrenches.clear()
        self._dp_snapshot_pending_reason = None
        self._latest_policy_anchor_s = None
        self._last_submitted_policy_anchor_s = None
        self._open_loop_observation_start_s = (
            None if start_s is None else float(start_s)
        )

    def append(
        self,
        image: np.ndarray | Mapping[str, np.ndarray],
        wrench: np.ndarray,
        *,
        image_timestamp_s: float | Mapping[str, float] | None = None,
        state_timestamp_s: float | None = None,
        allow_backfill: bool = True,
    ) -> None:
        chw_images = self._prepare_images(image)
        wrench_value = np.asarray(wrench, dtype=np.float32).reshape(-1).copy()
        if wrench_value.shape != (6,) or not np.all(np.isfinite(wrench_value)):
            raise ValueError("wrench must be a finite 6-vector")

        if image_timestamp_s is not None and state_timestamp_s is not None:
            image_times = self._prepare_image_timestamps(image_timestamp_s)
            state_time = float(state_timestamp_s)
            if not np.isfinite(state_time):
                raise ValueError("observation timestamps must be finite")
            if not self._timed_wrenches or state_time > self._timed_wrenches[-1][0]:
                self._timed_wrenches.append((state_time, wrench_value))
            anchor_time = image_times[self.anchor_image_key]
            image_is_current = (
                self._open_loop_observation_start_s is None
                or anchor_time + 1.0e-9 >= self._open_loop_observation_start_s
            )
            if image_is_current:
                anchor_appended = False
                for key, timed_images in self._timed_images_by_key.items():
                    image_time = image_times[key]
                    if not timed_images or image_time > timed_images[-1][0]:
                        timed_images.append((image_time, chw_images[key]))
                        anchor_appended = anchor_appended or key == self.anchor_image_key
                step_s = self.observation_step_s
                if anchor_appended and (
                    self._latest_policy_anchor_s is None
                    or step_s is None
                    or anchor_time - self._latest_policy_anchor_s >= step_s * 0.999
                ):
                    self._latest_policy_anchor_s = anchor_time
            return

        for key, images in self._images_by_key.items():
            images.append(chw_images[key])
        self._wrenches.append(wrench_value)
        if not allow_backfill:
            return
        for images in self._images_by_key.values():
            while len(images) < self.n_obs_steps:
                images.appendleft(images[0].copy())
        while len(self._wrenches) < self._wrenches.maxlen:
            self._wrenches.appendleft(self._wrenches[0].copy())

    def _prepare_images(
        self, image: np.ndarray | Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if isinstance(image, Mapping):
            image_keys = tuple(sorted(str(key) for key in image))
            if image_keys != self.image_keys:
                raise ValueError(
                    "image observations do not match the DP checkpoint: "
                    f"expected={list(self.image_keys)}, got={list(image_keys)}"
                )
            values = {key: image[key] for key in self.image_keys}
        else:
            if len(self.image_keys) > 1:
                raise ValueError(
                    "multi-camera DP requires image observations keyed by camera name"
                )
            values = {self.image_keys[0]: image}
        result: dict[str, np.ndarray] = {}
        for key, value in values.items():
            array = np.asarray(value)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"image[{key!r}] must have shape [H,W,3]")
            array = array.astype(np.float32)
            if array.max(initial=0.0) > 1.0:
                array /= 255.0
            if not np.all(np.isfinite(array)):
                raise ValueError(f"image[{key!r}] contains non-finite values")
            result[key] = np.ascontiguousarray(np.moveaxis(array, -1, 0))
        return result

    def _prepare_image_timestamps(
        self, timestamps: float | Mapping[str, float]
    ) -> dict[str, float]:
        if isinstance(timestamps, Mapping):
            keys = tuple(sorted(str(key) for key in timestamps))
            if keys != self.image_keys:
                raise ValueError(
                    "image timestamps do not match the DP checkpoint: "
                    f"expected={list(self.image_keys)}, got={list(keys)}"
                )
            result = {key: float(timestamps[key]) for key in self.image_keys}
        else:
            value = float(timestamps)
            result = {key: value for key in self.image_keys}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("observation timestamps must be finite")
        return result

    def snapshot(self) -> tuple[np.ndarray | Mapping[str, np.ndarray], np.ndarray]:
        if len(self.image_keys) == 1:
            images: np.ndarray | Mapping[str, np.ndarray] = np.stack(tuple(self._images), axis=0)
        else:
            images = {
                key: np.stack(tuple(self._images_by_key[key]), axis=0)
                for key in self.image_keys
            }
        return images, np.stack(tuple(self._wrenches), axis=0).reshape(
            self.n_obs_steps, self.wrench_history_steps, 6
        )

    def snapshot_for_dp(
        self,
    ) -> tuple[np.ndarray | Mapping[str, np.ndarray], np.ndarray] | None:
        if self._timed_images:
            anchor_s = self._latest_policy_anchor_s
            if anchor_s is None or (
                self._last_submitted_policy_anchor_s is not None
                and anchor_s <= self._last_submitted_policy_anchor_s
            ):
                self._note_pending("no new wrist image anchor")
                return None
            if self.inference_mode == "open_loop":
                if any(len(images) < self.n_obs_steps for images in self._timed_images_by_key.values()):
                    self._note_pending(
                        "camera history incomplete "
                        f"required={self.n_obs_steps} available="
                        f"{[len(images) for images in self._timed_images_by_key.values()]}"
                    )
                    return None
                step_s = self.observation_step_s or 0.1
                if self.n_obs_steps > 1 and anchor_s - self._timed_images[0][0] < (self.n_obs_steps - 1) * step_s * 0.999:
                    self._note_pending("wrist image history span is shorter than the DP grid")
                    return None
                if self.uses_wrench_observation:
                    wrench_step_s = step_s / max(self.wrench_history_steps, 1)
                    required_span = (self.n_obs_steps - 1) * step_s + (self.wrench_history_steps - 1) * wrench_step_s
                    if not self._timed_wrenches or anchor_s - self._timed_wrenches[0][0] < required_span * 0.999:
                        self._note_pending("wrench history span is shorter than the DP grid")
                        return None
            snapshot = (
                self.timed_snapshot_if_aligned(anchor_s)
                if self.inference_mode == "open_loop"
                else self.timed_snapshot(anchor_s)
            )
            if snapshot is None:
                return None
            self._dp_snapshot_pending_reason = None
            self._last_submitted_policy_anchor_s = anchor_s
            return snapshot
        if self.inference_mode == "open_loop" and (
            any(len(images) < self.n_obs_steps for images in self._images_by_key.values())
            or (self.uses_wrench_observation and len(self._wrenches) < self._wrenches.maxlen)
        ):
            self._note_pending("untimed observation history incomplete")
            return None
        return self.snapshot()

    def timed_snapshot(
        self, anchor_s: float
    ) -> tuple[np.ndarray | Mapping[str, np.ndarray], np.ndarray] | None:
        image_times = np.asarray([timestamp for timestamp, _ in self._timed_images], dtype=np.float64)
        wrench_times = np.asarray([timestamp for timestamp, _ in self._timed_wrenches], dtype=np.float64)
        if image_times.size == 0:
            raise RuntimeError("timed DP observation requires image samples")
        if self.uses_wrench_observation and wrench_times.size == 0:
            self._note_pending("no wrench samples")
            raise RuntimeError("timed force-aware DP observation requires wrench samples")
        image_step_s = self.observation_step_s or 0.1
        image_targets = float(anchor_s) + np.arange(-(self.n_obs_steps - 1), 1, dtype=np.float64) * image_step_s
        image_indices = _nearest_timestamp_indices(image_times, image_targets)
        selected_image_times = image_times[image_indices]
        image_indices_by_key = {self.anchor_image_key: image_indices}
        for key in self.image_keys:
            if key == self.anchor_image_key:
                continue
            key_times = np.asarray([timestamp for timestamp, _ in self._timed_images_by_key[key]], dtype=np.float64)
            if key_times.size == 0:
                self._note_pending(f"no frames received for camera={key}")
                return None
            key_indices = _previous_timestamp_indices(key_times, selected_image_times)
            if np.any(key_indices < 0):
                self._note_pending(f"camera={key} has no frame at or before the wrist anchor")
                return None
            image_indices_by_key[key] = key_indices
        images = self._build_images(image_indices, image_indices_by_key)
        if len(self.image_keys) > 1:
            self._log_camera_alignment_once(selected_image_times, image_indices_by_key)
        if not self.uses_wrench_observation:
            return images, np.zeros((self.n_obs_steps, 1, 6), dtype=np.float32)
        wrench_step_s = image_step_s / max(self.wrench_history_steps, 1)
        wrench_targets = selected_image_times[:, None] + np.arange(-(self.wrench_history_steps - 1), 1, dtype=np.float64)[None, :] * wrench_step_s
        wrench_indices = _previous_timestamp_indices(wrench_times, wrench_targets.reshape(-1)).reshape(self.n_obs_steps, self.wrench_history_steps)
        if np.any(wrench_indices < 0):
            return None
        wrenches = np.stack([self._timed_wrenches[int(index)][1] for index in wrench_indices.reshape(-1)], axis=0).reshape(self.n_obs_steps, self.wrench_history_steps, 6)
        return images, wrenches

    def append_open_loop_can_observation(self, wrench: np.ndarray, timestamp_s: float) -> None:
        if self.inference_mode != "open_loop":
            raise RuntimeError("CAN-only observation append is only valid in open-loop mode")
        value = np.asarray(wrench, dtype=np.float32).reshape(-1)
        timestamp_s = float(timestamp_s)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("CAN-derived wrench must be a finite 6-vector")
        if not np.isfinite(timestamp_s):
            raise ValueError("CAN observation timestamp must be finite")
        if not self._timed_wrenches or timestamp_s > self._timed_wrenches[-1][0]:
            self._timed_wrenches.append((timestamp_s, value.copy()))

    def append_continuous_can_observation(
        self,
        *,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
        wrench: np.ndarray,
        timestamp_s: float,
        q_cmd: np.ndarray | None = None,
    ) -> None:
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("continuous CAN observation timestamp must be finite")
        wrench_value = np.asarray(wrench, dtype=np.float32).reshape(-1)
        if wrench_value.shape != (6,) or not np.all(np.isfinite(wrench_value)):
            raise ValueError("continuous CAN wrench must be a finite 6-vector")
        if not self._timed_wrenches or timestamp_s > self._timed_wrenches[-1][0]:
            self._timed_wrenches.append((timestamp_s, wrench_value.copy()))
        if self.on_continuous_can is not None:
            self.on_continuous_can(timestamp_s, {
                "q": np.asarray(q), "dq": np.asarray(dq), "v": np.asarray(dq),
                "a": np.asarray(ddq), "ddq": np.asarray(ddq),
                "tau": np.asarray(tau), "wrench": wrench_value,
                **({"q_cmd": np.asarray(q_cmd)} if q_cmd is not None else {}),
            })

    def timed_snapshot_if_aligned(
        self, anchor_s: float
    ) -> tuple[np.ndarray | Mapping[str, np.ndarray], np.ndarray] | None:
        image_times = np.asarray([timestamp for timestamp, _ in self._timed_images], dtype=np.float64)
        can_times = np.asarray([timestamp for timestamp, _ in self._timed_wrenches], dtype=np.float64)
        if image_times.size < self.n_obs_steps or (self.uses_wrench_observation and can_times.size == 0):
            self._note_pending(f"history unavailable wrist={image_times.size}/{self.n_obs_steps} can={can_times.size}")
            return None
        image_step_s = self.observation_step_s or 0.1
        nominal_image_times = float(anchor_s) + np.arange(-(self.n_obs_steps - 1), 1, dtype=np.float64) * image_step_s
        image_indices = _nearest_timestamp_indices(image_times, nominal_image_times)
        if np.unique(image_indices).size != self.n_obs_steps:
            self._note_pending("wrist image timestamps collapse to duplicate DP rows")
            return None
        selected_image_times = image_times[image_indices]
        if np.any(np.diff(selected_image_times) <= 0.0):
            self._note_pending("wrist image timestamps are not increasing")
            return None
        image_indices_by_key = {self.anchor_image_key: image_indices}
        for key in self.image_keys:
            if key == self.anchor_image_key:
                continue
            key_times = np.asarray([timestamp for timestamp, _ in self._timed_images_by_key[key]], dtype=np.float64)
            if key_times.size < self.n_obs_steps:
                self._note_pending(f"camera={key} history={key_times.size}/{self.n_obs_steps}")
                return None
            key_indices = _previous_timestamp_indices(key_times, selected_image_times)
            if np.any(key_indices < 0):
                self._note_pending(f"camera={key} has no causal frame for a wrist row")
                return None
            if np.unique(key_indices).size != self.n_obs_steps:
                self._note_pending(f"camera={key} causal rows contain duplicates")
                return None
            if np.any(np.diff(key_times[key_indices]) <= 0.0):
                self._note_pending(f"camera={key} causal timestamps are not increasing")
                return None
            image_indices_by_key[key] = key_indices
        if len(self.image_keys) > 1:
            self._log_camera_alignment_once(selected_image_times, image_indices_by_key)
        images = self._build_images(image_indices, image_indices_by_key)
        if not self.uses_wrench_observation:
            return images, np.zeros((self.n_obs_steps, 1, 6), dtype=np.float32)
        can_step_s = image_step_s / self.wrench_history_steps
        can_targets = selected_image_times[:, None] + np.arange(-(self.wrench_history_steps - 1), 1, dtype=np.float64)[None, :] * can_step_s
        if can_times[0] > float(np.min(can_targets)):
            self._note_pending("wrench history starts after the oldest DP target")
            return None
        can_indices = _previous_timestamp_indices(can_times, can_targets.reshape(-1)).reshape(self.n_obs_steps, self.wrench_history_steps)
        if np.any(can_indices < 0):
            self._note_pending("wrench history has no causal row for a DP target")
            return None
        matched_can_times = can_times[can_indices]
        if np.any(np.diff(matched_can_times, axis=1) < 0.0):
            self._note_pending("wrench causal timestamps move backwards")
            return None
        gaps = can_targets - matched_can_times
        if np.any(gaps > self.maximum_alignment_gap_s):
            self._note_pending(
                "wrench state is too old "
                f"max_age={float(np.max(gaps) * 1.0e3):.2f}ms "
                f"limit={self.maximum_alignment_gap_s * 1.0e3:.2f}ms"
            )
            return None
        can_values = np.stack([self._timed_wrenches[int(index)][1] for index in can_indices.reshape(-1)], axis=0).reshape(self.n_obs_steps, self.wrench_history_steps, 6)
        return images, can_values

    def _build_images(
        self, anchor_indices: np.ndarray, indices_by_key: Mapping[str, np.ndarray]
    ) -> np.ndarray | Mapping[str, np.ndarray]:
        if len(self.image_keys) == 1:
            return np.stack([self._timed_images[int(index)][1] for index in anchor_indices], axis=0)
        return {
            key: np.stack([self._timed_images_by_key[key][int(index)][1] for index in indices_by_key[key]], axis=0)
            for key in self.image_keys
        }

    def _log_camera_alignment_once(
        self, anchor_times: np.ndarray, indices_by_key: Mapping[str, np.ndarray]
    ) -> None:
        if self._camera_alignment_logged:
            return
        ages = []
        for key in self.image_keys:
            if key == self.anchor_image_key:
                continue
            key_times = np.asarray([timestamp for timestamp, _ in self._timed_images_by_key[key]], dtype=np.float64)
            ages.append(f"{key}={float(np.median(anchor_times - key_times[indices_by_key[key]]) * 1.0e3):.3f}ms")
        log.info("DP camera alignment anchor=%s resample=causal_previous median_age=%s", self.anchor_image_key, ",".join(ages))
        self._camera_alignment_logged = True

    def _note_pending(self, reason: str) -> None:
        if reason != self._dp_snapshot_pending_reason:
            log.info("DP observation pending: %s", reason)
        self._dp_snapshot_pending_reason = reason


__all__ = ["DPObservationBuffer"]
