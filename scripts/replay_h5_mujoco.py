#!/usr/bin/env python3
"""Replay a recorded Nero H5 joint trajectory in the native MuJoCo viewer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from calibration.dynamics_common import load_dynamics_plan
from calibration.simulation import (
    launch_mujoco_live_preview,
    prepare_mujoco_live_preview,
    sync_mujoco_live_preview,
)


log = logging.getLogger("nero.h5_mujoco_replay")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay teleop/q_follower (or another 7D joint dataset) from a Nero "
            "H5 episode in MuJoCo. This command never connects to CAN hardware."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="episode H5 file, or a runs directory used together with --episode",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="episode index when source is a runs directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "calibration/config.yaml",
        help="model and MuJoCo scene configuration",
    )
    parser.add_argument(
        "--dataset",
        default="teleop/q_follower",
        help="joint-position dataset to replay; for example teleop/q_cmd",
    )
    parser.add_argument(
        "--timestamp-dataset",
        default="teleop/timestamp_us",
        help="timestamp dataset corresponding to the joint positions",
    )
    parser.add_argument(
        "--arm-index",
        type=int,
        default=0,
        help="7D arm block to replay when the joint dataset contains multiple arms",
    )
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--stop-s", type=float, help="exclusive episode time in seconds")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument(
        "--max-display-hz",
        type=float,
        default=60.0,
        help="maximum viewer refresh rate; trajectory timing remains unchanged",
    )
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--scene-output",
        type=Path,
        help="keep the generated MJCF scene at this path; default uses a temporary file",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="load every selected pose through MuJoCo without opening a window",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episode is not None and args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.arm_index < 0:
        parser.error("--arm-index must be non-negative")
    if not np.isfinite(args.start_s) or args.start_s < 0:
        parser.error("--start-s must be non-negative and finite")
    if args.stop_s is not None and (
        not np.isfinite(args.stop_s) or args.stop_s <= args.start_s
    ):
        parser.error("--stop-s must be finite and greater than --start-s")
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback-speed must be positive and finite")
    if args.loops < 1:
        parser.error("--loops must be positive")
    if not np.isfinite(args.max_display_hz) or args.max_display_hz <= 0:
        parser.error("--max-display-hz must be positive and finite")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative and finite")


def resolve_episode(source: Path, episode_index: int | None) -> Path:
    path = source.expanduser().resolve()
    if path.is_file():
        if episode_index is not None:
            raise ValueError("--episode cannot be used when source is an H5 file")
        if path.suffix.lower() != ".h5":
            raise ValueError(f"source file must use the .h5 extension: {path}")
        return path
    if not path.is_dir():
        raise ValueError(f"source does not exist: {path}")
    if episode_index is None:
        raise ValueError("--episode is required when source is a runs directory")
    matches = sorted(path.glob(f"episode_{episode_index:04d}_*.h5"))
    if not matches:
        raise ValueError(
            f"episode {episode_index} not found in {path}; expected "
            f"episode_{episode_index:04d}_*.h5"
        )
    if len(matches) > 1:
        raise ValueError(
            f"episode {episode_index} is ambiguous: "
            + ", ".join(item.name for item in matches)
        )
    return matches[0]


def load_joint_trajectory(
    path: Path,
    dataset_name: str,
    timestamp_dataset_name: str,
    *,
    arm_index: int,
    start_s: float,
    stop_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "H5 replay requires a working h5py installation compatible with NumPy"
        ) from exc

    with h5py.File(path, "r") as h5:
        missing = [
            name
            for name in (dataset_name, timestamp_dataset_name)
            if name not in h5
        ]
        if missing:
            raise ValueError(f"{path.name} is missing datasets: {missing}")
        all_q = np.asarray(h5[dataset_name][:], dtype=np.float64)
        timestamp_us = np.asarray(
            h5[timestamp_dataset_name][:], dtype=np.int64
        ).reshape(-1)

    if all_q.ndim != 2 or all_q.shape[0] != timestamp_us.size:
        raise ValueError(
            f"joint/timestamp shape mismatch: {dataset_name}={all_q.shape}, "
            f"{timestamp_dataset_name}={timestamp_us.shape}"
        )
    if all_q.shape[1] < 7 or all_q.shape[1] % 7 != 0:
        raise ValueError(f"{dataset_name} must have shape (N, 7*k), got {all_q.shape}")
    arm_count = all_q.shape[1] // 7
    if arm_index >= arm_count:
        raise ValueError(
            f"--arm-index {arm_index} is outside {dataset_name} arm count {arm_count}"
        )
    if timestamp_us.size < 1 or np.any(np.diff(timestamp_us) <= 0):
        raise ValueError(f"{timestamp_dataset_name} must be non-empty and increasing")
    if not np.isfinite(all_q).all():
        raise ValueError(f"{dataset_name} contains non-finite joint positions")

    episode_time_s = (timestamp_us - int(timestamp_us[0])).astype(np.float64) * 1.0e-6
    start_index = int(np.searchsorted(episode_time_s, start_s, side="left"))
    stop_index = (
        episode_time_s.size
        if stop_s is None
        else int(np.searchsorted(episode_time_s, stop_s, side="left"))
    )
    if start_index >= stop_index:
        duration_s = float(episode_time_s[-1])
        raise ValueError(
            f"selected replay interval is empty; episode duration is {duration_s:.3f}s"
        )
    column = slice(7 * arm_index, 7 * (arm_index + 1))
    q = np.ascontiguousarray(all_q[start_index:stop_index, column])
    time_s = episode_time_s[start_index:stop_index]
    time_s = np.ascontiguousarray(time_s - time_s[0])
    return q, time_s


def display_indices(
    time_s: np.ndarray,
    *,
    playback_speed: float,
    maximum_display_hz: float,
) -> np.ndarray:
    values = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot display an empty trajectory")
    minimum_source_step_s = playback_speed / maximum_display_hz
    selected = [0]
    last_time = float(values[0])
    for index in range(1, values.size - 1):
        if float(values[index]) - last_time >= minimum_source_step_s:
            selected.append(index)
            last_time = float(values[index])
    if values.size > 1 and selected[-1] != values.size - 1:
        selected.append(values.size - 1)
    return np.asarray(selected, dtype=np.int64)


def replay(
    plan,
    q: np.ndarray,
    time_s: np.ndarray,
    scene_path: Path,
    *,
    playback_speed: float,
    loops: int,
    maximum_display_hz: float,
    hold_seconds: float,
    no_viewer: bool,
) -> int:
    preview = prepare_mujoco_live_preview(plan, q[0], scene_path)
    if no_viewer:
        import mujoco

        data = mujoco.MjData(preview.model)
        data.qpos[:] = preview.initial_qpos
        for pose in q:
            data.qpos[preview.joint_qpos_addresses] = pose
            data.qvel[:] = 0.0
            mujoco.mj_forward(preview.model, data)
        return q.shape[0]

    indices = display_indices(
        time_s,
        playback_speed=playback_speed,
        maximum_display_hz=maximum_display_hz,
    )
    rendered = 0
    with launch_mujoco_live_preview(preview) as (data, viewer):
        for loop_index in range(loops):
            started_s = time.monotonic()
            for index in indices:
                if not viewer.is_running():
                    return rendered
                deadline_s = started_s + float(time_s[index]) / playback_speed
                remaining_s = deadline_s - time.monotonic()
                if remaining_s > 0:
                    time.sleep(remaining_s)
                sync_mujoco_live_preview(preview, data, viewer, q[index])
                rendered += 1
            log.info("completed replay loop %d/%d", loop_index + 1, loops)
        deadline_s = time.monotonic() + hold_seconds
        while viewer.is_running() and time.monotonic() < deadline_s:
            viewer.sync()
            time.sleep(0.02)
    return rendered


def _run(args: argparse.Namespace, path: Path, q: np.ndarray, time_s: np.ndarray) -> int:
    plan = load_dynamics_plan(args.config)
    if args.scene_output is not None:
        scene_path = args.scene_output.expanduser().resolve()
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        return replay(
            plan,
            q,
            time_s,
            scene_path,
            playback_speed=args.playback_speed,
            loops=args.loops,
            maximum_display_hz=args.max_display_hz,
            hold_seconds=args.hold_seconds,
            no_viewer=args.no_viewer,
        )
    with tempfile.TemporaryDirectory(prefix="nero_h5_mujoco_replay_") as temporary:
        scene_path = Path(temporary) / f"{path.stem}.scene.xml"
        return replay(
            plan,
            q,
            time_s,
            scene_path,
            playback_speed=args.playback_speed,
            loops=args.loops,
            maximum_display_hz=args.max_display_hz,
            hold_seconds=args.hold_seconds,
            no_viewer=args.no_viewer,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = resolve_episode(args.source, args.episode)
    q, time_s = load_joint_trajectory(
        path,
        args.dataset,
        args.timestamp_dataset,
        arm_index=args.arm_index,
        start_s=args.start_s,
        stop_s=args.stop_s,
    )
    duration_s = float(time_s[-1]) if time_s.size > 1 else 0.0
    print(
        f"H5 replay: {path}\n"
        f"dataset: {args.dataset} arm_index={args.arm_index}\n"
        f"samples: {q.shape[0]} duration={duration_s:.3f}s "
        f"speed={args.playback_speed:.3g}x"
    )
    rendered = _run(args, path, q, time_s)
    mode = "validated" if args.no_viewer else "rendered"
    print(f"MuJoCo {mode} poses: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
