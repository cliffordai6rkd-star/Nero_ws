from __future__ import annotations

import numpy as np
import pytest

from nero_collection.cameras import CameraFrame
from scripts.visualize_cameras import (
    CameraPreviewStats,
    _format_live_stats,
    _validate_frame,
)


def test_camera_preview_stats_uses_frame_timestamps() -> None:
    stats = CameraPreviewStats()
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    for timestamp_us in (1_000_000, 1_033_333, 1_066_666):
        stats.observe(CameraFrame("side", timestamp_us, frame))

    summary = stats.summary()

    assert summary["frames"] == 3
    assert summary["frequency_hz"] == pytest.approx(30.0003, rel=1.0e-4)
    assert summary["p99_gap_ms"] == pytest.approx(33.333)
    assert summary["maximum_gap_ms"] == pytest.approx(33.333)
    assert _format_live_stats({"side": stats}) == "side=30.000 FPS"


def test_camera_preview_rejects_invalid_frame() -> None:
    invalid = CameraFrame(
        "side",
        1_000_000,
        np.zeros((12, 16), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="invalid frame"):
        _validate_frame(invalid)
