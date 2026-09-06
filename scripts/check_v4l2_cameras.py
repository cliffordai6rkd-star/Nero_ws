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

from nero_collection.cameras import CameraManager, CameraVisualizer
from nero_collection.config import load_config


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    camera_configs = config.cameras
    if args.cameras:
        requested = set(args.cameras)
        available = {camera.name for camera in camera_configs}
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(
                f"requested cameras are not enabled in the configuration: {missing}; "
                f"available={sorted(available)}"
            )
        camera_configs = tuple(
            camera for camera in camera_configs if camera.name in requested
        )
    visualizer = (
        CameraVisualizer.from_config(camera_configs) if args.visualize else None
    )
    manager = CameraManager.from_config(camera_configs, visualizer=visualizer)
    if not manager.cameras:
        raise RuntimeError("configuration did not create any camera sources")
    counts = {camera.name: 0 for camera in manager.cameras}
    shapes: dict[str, tuple[int, ...]] = {}
    first_timestamp: dict[str, int] = {}
    last_timestamp: dict[str, int] = {}
    maximum_interval_us = {camera.name: 0 for camera in manager.cameras}
    latest_frames: dict[str, np.ndarray] = {}
    latest_previews: dict[str, np.ndarray] = {}
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
                latest_frames[frame.camera_name] = frame.frame.copy()
                if frame.preview_frame is not None:
                    latest_previews[frame.camera_name] = frame.preview_frame.copy()
                first_timestamp.setdefault(frame.camera_name, frame.timestamp_us)
                previous_timestamp = last_timestamp.get(frame.camera_name)
                if previous_timestamp is not None:
                    maximum_interval_us[frame.camera_name] = max(
                        maximum_interval_us[frame.camera_name],
                        frame.timestamp_us - previous_timestamp,
                    )
                last_timestamp[frame.camera_name] = frame.timestamp_us
            time.sleep(0.002)
    finally:
        manager.stop()

    failed: list[str] = []
    if args.snapshot_dir is not None:
        import cv2

        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in latest_frames.items():
            output = args.snapshot_dir / f"{name}_policy.png"
            cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"{name}: saved policy snapshot {output}")
        for name, frame in latest_previews.items():
            output = args.snapshot_dir / f"{name}_preview.png"
            cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"{name}: saved preview snapshot {output}")
    for name in counts:
        count = counts[name]
        elapsed_s = max(
            (last_timestamp.get(name, 0) - first_timestamp.get(name, 0)) * 1e-6,
            args.duration,
        )
        measured_hz = count / max(elapsed_s, 1e-9)
        maximum_interval_s = maximum_interval_us[name] * 1e-6
        print(
            f"{name}: frames={count} shape={shapes.get(name)} "
            f"measured={measured_hz:.2f} Hz max_interval={maximum_interval_s:.4f}s"
        )
        if count < args.min_frames:
            failed.append(f"{name} produced only {count} frames")
    if failed:
        raise RuntimeError("; ".join(failed))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open configured V4L2 cameras without starting Nero arms.")
    parser.add_argument("--config", default="configs/master_slave_can.yaml")
    parser.add_argument(
        "--camera",
        dest="cameras",
        action="append",
        help="Only test this configured camera name; repeat for multiple cameras.",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the same isolated camera preview used by collection/inference.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Save the latest policy and preview RGB frames as PNG files.",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
