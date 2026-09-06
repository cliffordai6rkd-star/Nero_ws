from __future__ import annotations

import numpy as np
import pytest

from inference.core.nero_sampler import NeroObservationSampler
from nero_collection.cameras import CameraFrame


class _Cameras:
    def __init__(self, batches):
        self._batches = iter(batches)
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        return next(self._batches, [])


def _sampler(cameras: _Cameras, *, maximum_age_s: float) -> NeroObservationSampler:
    return NeroObservationSampler(
        cameras=cameras,
        camera_keys=("side", "wrist"),
        primary_camera="wrist",
        maximum_state_age_s=maximum_age_s,
        read_state=lambda: None,
        drain_state=lambda: None,
        observation_ready=lambda _timestamp: True,
        open_loop=lambda: False,
        open_loop_active=lambda: False,
        wrench_rotation=lambda _sample: None,
    )


def _frame(name: str, timestamp_us: int) -> CameraFrame:
    return CameraFrame(name, timestamp_us, np.zeros((8, 8, 3), dtype=np.uint8))


def test_sampler_recovers_latest_frames_after_slow_dp_consumer(monkeypatch) -> None:
    cameras = _Cameras(
        [
            [_frame("side", 1_000_000), _frame("wrist", 1_000_000)],
            [_frame("side", 1_195_000), _frame("wrist", 1_196_000)],
        ]
    )
    sampler = _sampler(cameras, maximum_age_s=0.1)
    monkeypatch.setattr("inference.core.nero_sampler.now_us", lambda: 1_200_000)

    assert sampler.sample() is None
    assert cameras.poll_count == 2
    assert sampler.latest_frames["side"].timestamp_us == 1_195_000
    assert sampler.latest_frames["wrist"].timestamp_us == 1_196_000


def test_sampler_reports_the_camera_that_remains_stale(monkeypatch) -> None:
    cameras = _Cameras(
        [[_frame("side", 1_000_000), _frame("wrist", 1_195_000)]]
    )
    sampler = _sampler(cameras, maximum_age_s=0.005)
    monkeypatch.setattr("inference.core.nero_sampler.now_us", lambda: 1_200_000)
    monkeypatch.setattr("inference.core.nero_sampler.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="camera 'side' is stale"):
        sampler.sample()
