#!/usr/bin/env python3
"""Replay H5 q/dq on one real Nero follower and collect the response."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nero_collection.arms.factory import build_arm
from nero_collection.config import load_config
from nero_collection.episode_output import episode_path, next_episode_index
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.cli import _resolve_can_interfaces, _setup_can_interfaces
from nero_collection.teleop.bilateral import BilateralJointController
from nero_collection.time_utils import now_us

log = logging.getLogger("nero.hardware_h5_replay")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay teleop q/dq through Nero MIT control and collect actual state."
    )
    parser.add_argument("source", type=Path, help="source episode H5")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/master_slave_can.yaml")
    parser.add_argument("--pair", default="main")
    parser.add_argument("--q-dataset", default="teleop/q_follower")
    parser.add_argument("--dq-dataset", default="teleop/dq_follower")
    parser.add_argument("--timestamp-dataset", default="teleop/timestamp_us")
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--stop-s", type=float)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-abs-dq", type=float, default=5.0)
    parser.add_argument("--max-step-rad", type=float, default=0.30)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-can-setup", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate only; never connect to CAN")
    parser.add_argument(
        "--approve-hardware",
        action="store_true",
        help="required acknowledgement that the workspace is clear and E-stop is reachable",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.arm_index < 0 or args.speed <= 0 or args.max_abs_dq <= 0 or args.max_step_rad <= 0:
        parser.error("arm-index must be non-negative and speed/safety limits must be positive")
    if args.start_s < 0 or (args.stop_s is not None and args.stop_s <= args.start_s):
        parser.error("invalid --start-s/--stop-s interval")
    if not args.dry_run and not args.approve_hardware:
        parser.error("real replay requires --approve-hardware")
    return args


def load_replay(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError("h5py is unavailable or ABI-incompatible") from exc
    path = args.source.expanduser().resolve()
    with h5py.File(path, "r") as h5:
        names = (args.q_dataset, args.dq_dataset, args.timestamp_dataset)
        missing = [name for name in names if name not in h5]
        if missing:
            raise ValueError(f"{path} is missing datasets {missing}")
        q_all = np.asarray(h5[args.q_dataset][:], dtype=np.float64)
        dq_all = np.asarray(h5[args.dq_dataset][:], dtype=np.float64)
        timestamp_us = np.asarray(h5[args.timestamp_dataset][:], dtype=np.int64).reshape(-1)
    if q_all.ndim != 2 or dq_all.shape != q_all.shape or q_all.shape[1] % 7:
        raise ValueError(f"q/dq must have matching (N,7*k) shapes, got {q_all.shape}/{dq_all.shape}")
    if timestamp_us.size != q_all.shape[0] or timestamp_us.size < 2 or np.any(np.diff(timestamp_us) <= 0):
        raise ValueError("timestamps must match q/dq and be strictly increasing")
    arm_count = q_all.shape[1] // 7
    if args.arm_index >= arm_count:
        raise ValueError(f"arm index {args.arm_index} outside source arm count {arm_count}")
    column = slice(7 * args.arm_index, 7 * (args.arm_index + 1))
    q, dq = q_all[:, column], dq_all[:, column]
    source_time_s = (timestamp_us - timestamp_us[0]) * 1.0e-6
    selected = (source_time_s >= args.start_s) & (
        True if args.stop_s is None else source_time_s < args.stop_s
    )
    q, dq, source_time_s = q[selected], dq[selected], source_time_s[selected]
    if q.shape[0] < 2 or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("selected q/dq must contain at least two finite samples")
    replay_time_s = (source_time_s - source_time_s[0]) / args.speed
    replay_dq = dq * args.speed
    if np.max(np.abs(replay_dq)) > args.max_abs_dq:
        raise ValueError(f"replay dq exceeds --max-abs-dq: {np.max(np.abs(replay_dq)):.4f}")
    if np.max(np.abs(np.diff(q, axis=0))) > args.max_step_rad:
        raise ValueError(f"adjacent q exceeds --max-step-rad: {np.max(np.abs(np.diff(q, axis=0))):.4f}")
    return np.ascontiguousarray(q), np.ascontiguousarray(replay_dq), replay_time_s


def _pair(config, name: str):
    matches = [pair for pair in config.teleop.master_slave if pair.name == name]
    if len(matches) != 1:
        raise ValueError(f"pair {name!r} not found uniquely in config")
    return matches[0]


def run(args: argparse.Namespace) -> Path | None:
    q, dq, replay_time_s = load_replay(args)
    log.info("validated samples=%d duration=%.3fs max|dq|=%.3f", len(q), replay_time_s[-1], np.max(np.abs(dq)))
    if args.dry_run:
        return None
    config = load_config(args.config)
    config = _resolve_can_interfaces(config)
    if not args.skip_can_setup:
        _setup_can_interfaces(config)
    pair = _pair(config, args.pair)
    arm = build_arm(pair.follower, config.teleop.backend)
    buffer = EpisodeBuffer(
        config=config,
        arm_names=(pair.name,),
        enable_online_tau_ext=False,
        episode_metadata={
            "collection_kind": "hardware_h5_q_dq_replay",
            "source_h5": str(args.source.expanduser().resolve()),
            "source_q_dataset": args.q_dataset,
            "source_dq_dataset": args.dq_dataset,
            "playback_speed": float(args.speed),
        },
    )
    gains = config.teleop.command.bilateral_mit
    kp = np.asarray(gains.follower_kp, dtype=np.float64)
    kd = np.asarray(gains.follower_kd, dtype=np.float64)
    zeros = np.zeros(7, dtype=np.float64)
    dynamics = BilateralJointController(
        gains, config.tau_ext_inference.inverse_dynamics
    ).dynamics
    gravity_scale = np.asarray(gains.follower_gravity_scale, dtype=np.float64)
    torque_limit = np.asarray(gains.follower_torque_limit_nm, dtype=np.float64)

    def gravity_command(position: np.ndarray) -> np.ndarray:
        return np.clip(
            dynamics.gravity_torque(position) * gravity_scale,
            -torque_limit,
            torque_limit,
        )
    output_dir = (args.output_dir or config.output.directory / "hardware_replay").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = episode_path(output_dir, "replay", next_episode_index(output_dir, "replay"))
    connected = False
    try:
        arm.connect()
        connected = True
        arm.set_follower_mode()
        arm.enable()
        arm.move_joints(q[0])
        if not arm.wait_motion_done(20.0):
            raise RuntimeError("timed out moving to replay start pose")
        arm.validate_joint_impedance_support()
        arm.configure_joint_impedance_mode()
        start = time.monotonic()
        for index in range(len(q)):
            deadline = start + float(replay_time_s[index])
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            arm.command_joint_impedance(
                q[index], dq[index], kp, kd, gravity_command(q[index])
            )
            state = arm.read_state()
            error = np.max(np.abs(np.asarray(state.q) - q[index]))
            if not np.isfinite(error) or error > args.max_tracking_error_rad:
                raise RuntimeError(
                    f"tracking safety limit at sample {index}: error={error:.4f} rad"
                )
            buffer.append_teleop(
                now_us(),
                {
                    "q_follower": ("q", np.asarray(state.q)),
                    "dq_follower": ("velocity", np.asarray(state.dq)),
                    "tau_follower": ("torque", np.asarray(state.torque)),
                    "current_follower": ("current", np.asarray(state.current)),
                    "q_cmd": ("q", q[index]),
                    "dq_cmd": ("velocity", dq[index]),
                },
            )
        arm.command_joint_impedance(q[-1], zeros, kp, kd, gravity_command(q[-1]))
        return buffer.save(output)
    finally:
        if connected:
            try:
                arm.command_joint_impedance(
                    q[-1], zeros, kp, kd, gravity_command(q[-1])
                )
            except Exception:
                log.exception("failed to send final hold command")
            arm.disconnect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = run(args)
    if output is not None:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
