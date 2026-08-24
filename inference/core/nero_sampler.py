"""Online Nero observation sampler used by both legacy and modular runtimes."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from inference.core.contracts import Observation
from nero_collection.cameras import CameraFrame, CameraManager
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)


class NeroObservationSampler:
    """Collect fresh camera/state data without running a model or controller."""

    def __init__(
        self,
        *,
        cameras: CameraManager,
        camera_keys: tuple[str, ...],
        primary_camera: str,
        maximum_state_age_s: float,
        read_state: Callable[[], Any | None],
        drain_state: Callable[[], None],
        observation_ready: Callable[[int], bool],
        open_loop: Callable[[], bool],
        open_loop_active: Callable[[], bool],
        wrench_rotation: Callable[[Any], np.ndarray | None],
    ) -> None:
        self.cameras = cameras
        self.camera_keys = tuple(camera_keys)
        self.primary_camera = str(primary_camera)
        self.maximum_state_age_s = float(maximum_state_age_s)
        self.read_state = read_state
        self.drain_state = drain_state
        self.observation_ready = observation_ready
        self.open_loop = open_loop
        self.open_loop_active = open_loop_active
        self.wrench_rotation = wrench_rotation
        self._latest_frames: dict[str, CameraFrame] = {}
        self._latest_frame: CameraFrame | None = None
        self._last_stale_warning_s = 0.0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset_episode(self) -> None:
        self._latest_frames.clear()
        self._latest_frame = None

    @property
    def latest_frame(self) -> CameraFrame | None:
        return self._latest_frame

    @property
    def latest_frames(self) -> dict[str, CameraFrame]:
        return dict(self._latest_frames)

    def sample(self) -> Observation | None:
        open_loop = bool(self.open_loop())
        executing = bool(self.open_loop_active())
        for frame in self.cameras.poll():
            if frame.camera_name in self.camera_keys:
                self._latest_frames[frame.camera_name] = frame
        primary_frame = self._latest_frames.get(self.primary_camera)
        if primary_frame is None:
            return None
        self._latest_frame = primary_frame
        missing = [key for key in self.camera_keys if key not in self._latest_frames]
        if missing:
            return None

        current_us = now_us()
        camera_age_s = max(
            max(
                0.0,
                (current_us - int(self._latest_frames[key].timestamp_us)) * 1.0e-6,
            )
            for key in self.camera_keys
        )
        camera_stale = camera_age_s > self.maximum_state_age_s
        if camera_stale and not open_loop:
            raise RuntimeError(
                f"camera {self.primary_camera!r} is stale: age={camera_age_s:.3f}s, "
                f"limit={self.maximum_state_age_s:.3f}s"
            )
        if camera_stale and open_loop and not executing:
            monotonic_s = time.monotonic()
            if monotonic_s - self._last_stale_warning_s >= 0.5:
                log.warning(
                    "waiting for a fresh camera frame before the next open-loop "
                    "observation batch camera=%s age=%.3fs limit=%.3fs",
                    self.primary_camera,
                    camera_age_s,
                    self.maximum_state_age_s,
                )
                self._last_stale_warning_s = monotonic_s

        sample = self.read_state()
        if sample is None:
            return None
        self.drain_state()
        if not self.observation_ready(sample.timestamp_us):
            return None
        if open_loop and not executing and camera_stale:
            return None

        images = {key: self._latest_frames[key].frame for key in self.camera_keys}
        image_timestamps = {
            key: int(self._latest_frames[key].timestamp_us) for key in self.camera_keys
        }
        rotation = self.wrench_rotation(sample)
        tau_result = sample.tau_result
        return Observation(
            timestamp_us=int(sample.timestamp_us),
            acquired_timestamp_us=int(sample.acquired_timestamp_us),
            q=sample.q,
            dq=sample.dq,
            ddq=sample.ddq,
            tau=sample.tau,
            tau_ext=tau_result.tau_ext_cal,
            wrench_ext=sample.wrench,
            images=images,
            image_timestamps_us=image_timestamps,
            wrench_to_control_rotation=rotation,
            metadata={
                "processed_wrench": np.asarray(sample.processed_wrench).copy(),
                "tau_result": tau_result,
            },
        )
