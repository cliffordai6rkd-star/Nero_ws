#!/usr/bin/env python3
"""Preview every enabled configured camera without starting Nero arms or CAN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.cameras import CameraFrame, CameraManager, CameraVisualizer
from nero_collection.config import load_config
from nero_collection.keyboard import TerminalKeys


log = logging.getLogger(__name__)


@dataclass
class CameraPreviewStats:
    frame_count: int = 0
    first_timestamp_us: int | None = None
    last_timestamp_us: int | None = None
    intervals_ms: list[float] = field(default_factory=list)

    def observe(self, frame: CameraFrame) -> None:
        timestamp_us = int(frame.timestamp_us)
        if self.last_timestamp_us is not None and timestamp_us > self.last_timestamp_us:
            self.intervals_ms.append((timestamp_us - self.last_timestamp_us) * 1.0e-3)
        if self.first_timestamp_us is None:
            self.first_timestamp_us = timestamp_us
        self.last_timestamp_us = timestamp_us
        self.frame_count += 1

    @property
    def frequency_hz(self) -> float | None:
        if (
            self.frame_count < 2
            or self.first_timestamp_us is None
            or self.last_timestamp_us is None
        ):
            return None
        elapsed_s = (self.last_timestamp_us - self.first_timestamp_us) * 1.0e-6
        return (self.frame_count - 1) / elapsed_s if elapsed_s > 0.0 else None

    def summary(self) -> dict[str, float | int | None]:
        intervals = np.asarray(self.intervals_ms, dtype=np.float64)
        return {
            "frames": self.frame_count,
            "frequency_hz": self.frequency_hz,
            "p99_gap_ms": (
                float(np.percentile(intervals, 99.0)) if intervals.size else None
            ),
            "maximum_gap_ms": float(np.max(intervals)) if intervals.size else None,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if not config.cameras:
        raise RuntimeError("configuration has no enabled cameras")

    camera_names = tuple(camera.name for camera in config.cameras)
    visualizer = CameraVisualizer(camera_names)
    manager = CameraManager.from_config(config.cameras, visualizer=visualizer)
    if not manager.cameras:
        raise RuntimeError("configuration did not create any usable camera sources")
    stats = {camera.name: CameraPreviewStats() for camera in manager.cameras}

    manager.start()
    started_s = time.monotonic()
    next_report_s = started_s + args.report_interval
    print(
        f"Previewing {', '.join(stats)} in one window. "
        "Press q or Esc in this terminal to quit; Ctrl-C also works.",
        flush=True,
    )
    try:
        with TerminalKeys() as keys:
            while True:
                now_s = time.monotonic()
                if args.duration is not None and now_s - started_s >= args.duration:
                    break
                key = keys.read_key(0.0)
                if key in {"q", "Q", "\x1b", "\x03"}:
                    break

                for frame in manager.poll():
                    _validate_frame(frame)
                    stats[frame.camera_name].observe(frame)

                if now_s >= next_report_s:
                    print(_format_live_stats(stats), flush=True)
                    next_report_s += args.report_interval
                time.sleep(0.002)
    except KeyboardInterrupt:
        print("\nStopping camera preview.", flush=True)
    finally:
        manager.stop()

    failed = False
    for camera_name, camera_stats in stats.items():
        summary = camera_stats.summary()
        print(
            f"{camera_name}: frames={summary['frames']} "
            f"fps={_format_number(summary['frequency_hz'])} "
            f"p99_gap_ms={_format_number(summary['p99_gap_ms'])} "
            f"max_gap_ms={_format_number(summary['maximum_gap_ms'])}",
            flush=True,
        )
        failed = failed or camera_stats.frame_count == 0
    return 1 if failed else 0


def _validate_frame(frame: CameraFrame) -> None:
    values = np.asarray(frame.frame)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError(
            f"camera {frame.camera_name} returned invalid frame "
            f"shape={values.shape} dtype={values.dtype}"
        )


def _format_live_stats(stats: dict[str, CameraPreviewStats]) -> str:
    values = [
        f"{name}={_format_number(camera_stats.frequency_hz)} FPS"
        for name, camera_stats in stats.items()
    ]
    return " | ".join(values)


def _format_number(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display all enabled cameras from a collection config in one labeled window. "
            "This script never connects to Nero arms or CAN."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/master_slave_can.yaml",
        help="Collection YAML containing camera definitions",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional preview duration in seconds; default runs until q/Esc/Ctrl-C",
    )
    parser.add_argument("--report-interval", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.report_interval <= 0.0:
        parser.error("--report-interval must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
