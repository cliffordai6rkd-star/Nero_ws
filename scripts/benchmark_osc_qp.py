from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.control import (
    OSCQPConfig,
    OSCQPController,
    OSCTargetTrajectory,
    PinocchioDynamicsModel,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Nero OSC-QP end-to-end latency")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=ROOT / "urdf" / "nero" / "nero_with_gripper.urdf",
    )
    parser.add_argument("--frame", default="gripper_base")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--target-x-offset", type=float, default=0.01)
    parser.add_argument("--target-normal-force", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must be positive")
    model = PinocchioDynamicsModel(args.urdf, frame_name=args.frame)
    controller = OSCQPController(
        model,
        OSCQPConfig(horizon_steps=args.horizon, dt_s=args.dt),
    )
    q = np.zeros(7, dtype=np.float64)
    dq = np.zeros(7, dtype=np.float64)
    pose = model.snapshot(q, dq).pose.copy()
    pose[0, 3] += args.target_x_offset
    wrench = np.zeros(6, dtype=np.float64)
    wrench[2] = args.target_normal_force
    target = OSCTargetTrajectory.constant(pose, wrench, args.horizon)
    measured_wrench = np.zeros(6, dtype=np.float64)

    cold = controller.optimize_mpc(q, dq, target, measured_wrench).solve_time_s
    for _ in range(args.warmup):
        controller.optimize_mpc(q, dq, target, measured_wrench)
    samples = np.empty(args.iterations, dtype=np.float64)
    started = perf_counter()
    for index in range(args.iterations):
        samples[index] = controller.optimize_mpc(
            q, dq, target, measured_wrench
        ).solve_time_s
    wall_time_s = perf_counter() - started

    mean_s = wall_time_s / args.iterations
    print(f"horizon_steps: {args.horizon}")
    print(f"configured_control_frequency_hz: {controller.config.control_frequency_hz:.1f}")
    print(f"target_x_offset_m: {args.target_x_offset:.4f}")
    print(f"target_normal_force_n: {args.target_normal_force:.3f}")
    print(f"cold_start_ms: {cold * 1.0e3:.3f}")
    print(f"steady_mean_ms: {mean_s * 1.0e3:.3f}")
    print(f"steady_median_ms: {np.median(samples) * 1.0e3:.3f}")
    print(f"steady_p95_ms: {np.percentile(samples, 95) * 1.0e3:.3f}")
    print(f"steady_end_to_end_hz: {1.0 / mean_s:.1f}")


if __name__ == "__main__":
    main()
