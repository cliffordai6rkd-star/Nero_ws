from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.dynamics_common import (
    DOF,
    DynamicsDataset,
    DynamicsPlan,
    build_reduced_model,
    load_dynamics_plan,
    save_dynamics_dataset,
    setup_socketcan,
)
from calibration.excitation import FourierTrajectory, load_trajectory, validate_trajectory
from nero_collection.arms.factory import build_arm
from nero_collection.config import ArmEndpointConfig, load_config


log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    plan = load_dynamics_plan(args.config)
    profiles = _selected_profiles(plan.excitation.profiles, args.profile)
    if args.output and len(profiles) != 1:
        raise ValueError("--output is only valid when exactly one --profile is selected")
    trajectories = tuple(
        (profile, load_trajectory(profile.trajectory_path))
        for profile in profiles
    )
    for _, trajectory in trajectories:
        validate_trajectory(trajectory, plan)
    _, model = build_reduced_model(plan.model)
    if args.plan_only:
        for profile, trajectory in trajectories:
            _print_plan(
                plan,
                profile,
                trajectory,
                args.repetitions or profile.repetitions,
            )
        return 0
    if not plan.safety.approved:
        raise RuntimeError(
            "dynamics safety.approved is false; review the generated trajectory on the real workcell"
        )

    collection = load_config(plan.collection_config_path)
    endpoint = _follower_endpoint(collection, args.pair or plan.pair_name)
    backend = args.backend or collection.teleop.backend
    if backend == "pyagxarm" and not args.skip_can_setup:
        setup_socketcan(endpoint.channel, endpoint.bitrate)
    if not args.yes:
        _confirm_motion(endpoint, trajectories, args.repetitions)

    arm = build_arm(endpoint, backend)
    lower = np.asarray(model.lowerPositionLimit) + plan.excitation.joint_limit_margin_rad
    upper = np.asarray(model.upperPositionLimit) - plan.excitation.joint_limit_margin_rad

    log.info(
        "connecting dynamics collector arm=%s channel=%s firmware=%s backend=%s",
        endpoint.name,
        endpoint.channel,
        endpoint.firmware,
        backend,
    )
    arm.connect()
    try:
        arm.set_follower_mode()
        time.sleep(collection.teleop.command.role_switch_settle_s)
        arm.enable()
        _wait_for_follower_role(arm, collection.teleop.command.role_switch_timeout_s)
        for profile, trajectory in trajectories:
            repetitions = args.repetitions or profile.repetitions
            timestamps: list[int] = []
            q_samples: list[np.ndarray] = []
            q_commands: list[np.ndarray] = []
            tau_samples: list[np.ndarray] = []
            current_samples: list[np.ndarray] = []
            motor_timestamps: list[np.ndarray] = []
            motor_acquired_timestamps: list[np.ndarray] = []
            q_can_timestamps: list[int] = []
            q_acquired_timestamps: list[int] = []
            trajectory_ids: list[int] = []
            log.info("moving to excitation profile=%s role=%s", profile.name, profile.role)
            _move_to_start(arm, trajectory.q[0], plan)
            for repetition in range(repetitions):
                log.info(
                    "starting profile=%s repetition %d/%d",
                    profile.name,
                    repetition + 1,
                    repetitions,
                )
                _collect_repetition(
                    arm,
                    trajectory,
                    repetition,
                    plan,
                    lower,
                    upper,
                    timestamps,
                    q_samples,
                    q_commands,
                    tau_samples,
                    current_samples,
                    motor_timestamps,
                    motor_acquired_timestamps,
                    q_can_timestamps,
                    q_acquired_timestamps,
                    trajectory_ids,
                )
            output = (
                Path(args.output).expanduser().resolve()
                if args.output
                else profile.dataset_path
            )
            dataset = DynamicsDataset(
                timestamp_us=np.asarray(timestamps, dtype=np.int64),
                q=np.stack(q_samples),
                q_cmd=np.stack(q_commands),
                tau=np.stack(tau_samples),
                current=np.stack(current_samples),
                motor_timestamp_us=np.stack(motor_timestamps),
                motor_acquired_timestamp_us=np.stack(motor_acquired_timestamps),
                q_can_timestamp_us=np.asarray(q_can_timestamps, dtype=np.int64),
                q_acquired_timestamp_us=np.asarray(q_acquired_timestamps, dtype=np.int64),
                trajectory_id=np.asarray(trajectory_ids, dtype=np.int32),
                metadata={
                    "format_version": 3,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "config_path": str(plan.source_path),
                    "profile_name": profile.name,
                    "profile_role": profile.role,
                    "profile_seed": profile.seed,
                    "trajectory_path": str(profile.trajectory_path),
                    "trajectory_condition_number": trajectory.condition_number,
                    "trajectory_regressor_rank": trajectory.regressor_rank,
                    "arm_name": endpoint.name,
                    "channel": endpoint.channel,
                    "firmware": endpoint.firmware,
                    "backend": backend,
                    "joint_names": list(plan.model.joint_names),
                    "position_unit": "rad",
                    "torque_unit": "N.m",
                    "current_unit": "SDK_native",
                    "velocity_source": "none_v112_not_used",
                    "position_timestamp_source": "sdk_aggregated_joint_can_frame",
                    "torque_timestamp_source": "sdk_per_joint_motor_can_frame",
                    "repetitions": repetitions,
                },
            )
            saved = save_dynamics_dataset(output, dataset)
            print(f"Saved dynamics profile {profile.name}: {saved}")
            print(
                f"samples={dataset.timestamp_us.size}, "
                f"duration={(dataset.timestamp_us[-1] - dataset.timestamp_us[0]) * 1e-6:.3f}s"
            )
    finally:
        # Keep the gravity-loaded arm enabled; disconnecting must not command a drop.
        arm.disconnect()

    return 0


def _collect_repetition(
    arm,
    trajectory: FourierTrajectory,
    trajectory_id: int,
    plan: DynamicsPlan,
    lower: np.ndarray,
    upper: np.ndarray,
    timestamps: list[int],
    q_samples: list[np.ndarray],
    q_commands: list[np.ndarray],
    tau_samples: list[np.ndarray],
    current_samples: list[np.ndarray],
    motor_timestamps: list[np.ndarray],
    motor_acquired_timestamps: list[np.ndarray],
    q_can_timestamps: list[int],
    q_acquired_timestamps: list[int],
    trajectory_ids: list[int],
) -> None:
    period = 1.0 / plan.excitation.sample_rate_hz
    start = time.monotonic()
    previous_timestamp = timestamps[-1] if timestamps else None
    for index, q_cmd in enumerate(trajectory.q):
        deadline = start + index * period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif -remaining > max(2.0 * period, 0.03):
            raise RuntimeError(
                f"excitation command loop missed its deadline by {-remaining:.4f}s at sample {index}"
            )
        arm.command_joint_positions(q_cmd)
        state = arm.read_state()
        q, tau, current, timestamp_us, motor_timestamp_us, motor_acquired_timestamp_us = _checked_measurement(
            state, q_cmd, plan, lower, upper
        )
        if previous_timestamp is not None:
            dt = (timestamp_us - previous_timestamp) * 1e-6
            if dt <= 0 or dt > plan.safety.max_timestamp_gap_s:
                raise RuntimeError(f"invalid measurement timestamp gap {dt:.6f}s")
        timestamps.append(timestamp_us)
        q_samples.append(q)
        q_commands.append(np.asarray(q_cmd, dtype=np.float64).copy())
        tau_samples.append(tau)
        current_samples.append(current)
        motor_timestamps.append(motor_timestamp_us)
        motor_acquired_timestamps.append(motor_acquired_timestamp_us)
        q_can_timestamps.append(int(state.q_timestamp_us))
        q_acquired_timestamps.append(
            int(state.q_acquired_timestamp_us or state.acquired_timestamp_us)
        )
        trajectory_ids.append(trajectory_id)
        previous_timestamp = timestamp_us


def _checked_measurement(state, q_cmd, plan, lower, upper):
    q = np.asarray(state.q, dtype=np.float64).reshape(-1)
    tau = np.asarray(state.torque, dtype=np.float64).reshape(-1)
    if q.size != DOF or not np.isfinite(q).all():
        raise RuntimeError(f"invalid measured joint angles: {q}")
    if tau.size != DOF or not np.isfinite(tau).all():
        raise RuntimeError(f"invalid measured joint torques: {tau}")
    if np.any(q < lower) or np.any(q > upper):
        raise RuntimeError(f"measured position crossed the configured safety limits: {q}")
    tracking_error = np.abs(q - q_cmd)
    if np.any(tracking_error > plan.excitation.max_tracking_error_rad):
        raise RuntimeError(
            "trajectory tracking error exceeded safety limit: "
            f"error={tracking_error}, limit={plan.excitation.max_tracking_error_rad}"
        )
    if np.any(np.abs(tau) > plan.safety.max_abs_torque_nm):
        raise RuntimeError(
            f"measured torque exceeded safety limit: tau={tau}, limit={plan.safety.max_abs_torque_nm}"
        )
    current = np.asarray(state.current, dtype=np.float64).reshape(-1)
    if current.size != DOF:
        current = np.full(DOF, np.nan, dtype=np.float64)
    motor_timestamp_us = np.asarray(state.motor_timestamp_us, dtype=np.int64).reshape(-1)
    if motor_timestamp_us.size != DOF:
        motor_timestamp_us = np.zeros(DOF, dtype=np.int64)
    motor_acquired_timestamp_us = np.asarray(
        state.motor_acquired_timestamp_us,
        dtype=np.int64,
    ).reshape(-1)
    if motor_acquired_timestamp_us.size != DOF:
        motor_acquired_timestamp_us = np.full(
            DOF,
            int(state.acquired_timestamp_us or state.timestamp_us),
            dtype=np.int64,
        )
    return (
        q.copy(),
        tau.copy(),
        current.copy(),
        int(state.timestamp_us),
        motor_timestamp_us.copy(),
        motor_acquired_timestamp_us.copy(),
    )


def _move_to_start(arm, target: np.ndarray, plan: DynamicsPlan) -> None:
    state = arm.read_state()
    start = np.asarray(state.q, dtype=np.float64).reshape(-1)
    if start.size != DOF or not np.isfinite(start).all():
        raise RuntimeError(f"cannot move to trajectory start from invalid q: {start}")
    delta = np.asarray(target) - start
    duration = max(0.5, 1.875 * float(np.max(np.abs(delta))) / plan.excitation.start_move_speed_rad_s)
    steps = max(2, int(np.ceil(duration * 30.0)))
    start_time = time.monotonic()
    for step in range(1, steps + 1):
        phase = step / steps
        scale = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
        arm.command_joint_positions(start + scale * delta)
        remaining = start_time + step / 30.0 - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    time.sleep(0.5)
    final = np.asarray(arm.read_state().q)
    if np.any(np.abs(final - target) > plan.excitation.max_tracking_error_rad):
        raise RuntimeError(f"arm failed to reach excitation start: target={target}, measured={final}")


def _wait_for_follower_role(arm, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = arm.read_control_role(refresh=True)
        if last == "follower":
            return
        time.sleep(0.05)
    raise RuntimeError(f"follower mode was not confirmed; detected={last or 'unknown'}")


def _follower_endpoint(collection, pair_name: str) -> ArmEndpointConfig:
    for pair in collection.teleop.master_slave:
        if pair.name == pair_name:
            return pair.follower
    raise ValueError(f"pair {pair_name!r} not found in collection config")


def _confirm_motion(endpoint, trajectories, repetition_override):
    if not sys.stdin.isatty():
        raise RuntimeError("interactive motion confirmation requires a TTY; use --yes after review")
    total_duration = sum(
        trajectory.time_s[-1] * (repetition_override or profile.repetitions)
        for profile, trajectory in trajectories
    )
    names = ", ".join(profile.name for profile, _ in trajectories)
    print(
        f"WARNING: {endpoint.name} on {endpoint.channel} will execute profiles "
        f"[{names}] for approximately {total_duration:.1f}s.\n"
        "Clear the workcell, remove payloads, keep the gripper fixed, and keep "
        "the emergency stop ready."
    )
    if input("Type MOVE to continue: ").strip() != "MOVE":
        raise RuntimeError("dynamics collection cancelled")


def _print_plan(plan, profile, trajectory, repetitions):
    print(f"\nprofile: {profile.name} role={profile.role}")
    print(f"trajectory samples: {trajectory.time_s.size}")
    print(f"repetitions: {repetitions}")
    print(f"condition number: {trajectory.condition_number:.6g}")
    print("position min: " + np.array2string(np.min(trajectory.q, axis=0), precision=4))
    print("position max: " + np.array2string(np.max(trajectory.q, axis=0), precision=4))
    print("max |velocity|: " + np.array2string(np.max(np.abs(trajectory.dq), axis=0), precision=4))
    print("max |acceleration|: " + np.array2string(np.max(np.abs(trajectory.ddq), axis=0), precision=4))
    print(f"safety.approved={plan.safety.approved}")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect synchronized q and measured torque/current for Nero dynamics identification."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--pair")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-can-setup", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def _selected_profiles(profiles, names):
    if not names:
        return profiles
    by_name = {profile.name: profile for profile in profiles}
    missing = sorted(set(names).difference(by_name))
    if missing:
        raise ValueError(f"unknown excitation profiles: {missing}")
    return tuple(by_name[name] for name in names)


if __name__ == "__main__":
    raise SystemExit(main())
