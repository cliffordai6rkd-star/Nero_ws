#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.cameras import CameraManager
from nero_collection.config import load_config


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    manager = CameraManager.from_config(config.cameras)
    if not manager.cameras:
        raise RuntimeError("configuration did not create any camera sources")
    counts = {camera.name: 0 for camera in manager.cameras}
    shapes: dict[str, tuple[int, ...]] = {}
    first_timestamp: dict[str, int] = {}
    last_timestamp: dict[str, int] = {}
    manager.start()
    start_t = time.monotonic()
    try:
        while time.monotonic() - start_t < args.duration:
            for frame in manager.poll():
                if frame.frame.dtype != np.uint8 or frame.frame.ndim != 3:
                    raise RuntimeError(
                        f"camera {frame.camera_name} returned invalid frame "
                        f"shape={frame.frame.shape} dtype={frame.frame.dtype}"
                    )
                counts[frame.camera_name] += 1
                shapes[frame.camera_name] = frame.frame.shape
                first_timestamp.setdefault(frame.camera_name, frame.timestamp_us)
                last_timestamp[frame.camera_name] = frame.timestamp_us
            time.sleep(0.002)
    finally:
        manager.stop()

    failed: list[str] = []
    for name in counts:
        count = counts[name]
        elapsed_s = max(
            (last_timestamp.get(name, 0) - first_timestamp.get(name, 0)) * 1e-6,
            args.duration,
        )
        measured_hz = count / max(elapsed_s, 1e-9)
        print(f"{name}: frames={count} shape={shapes.get(name)} measured={measured_hz:.2f} Hz")
        if count < args.min_frames:
            failed.append(f"{name} produced only {count} frames")
    if failed:
        raise RuntimeError("; ".join(failed))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open configured V4L2 cameras without starting Nero arms.")
    parser.add_argument("--config", default="configs/master_slave_can.yaml")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=30)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
