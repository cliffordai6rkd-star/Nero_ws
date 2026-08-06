from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from calibration.dynamics_common import ExcitationProfile, load_dynamics_plan
from calibration.excitation import (
    FourierTrajectory,
    combined_trajectory_diagnostics,
    generate_optimized_trajectory,
    load_trajectory,
    save_trajectory,
)
from calibration.simulation import (
    play_mujoco_preview,
    prepare_mujoco_preview,
    print_preview_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate, check, and play configured Nero Fourier trajectories."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Process only this configured profile; repeat to select multiple profiles",
    )
    parser.add_argument("--trials", type=int, help="Override optimization_trials")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse configured NPZ files and only rebuild/check/play MuJoCo scenes",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Generate and check MJCF scenes without opening the MuJoCo viewer",
    )
    parser.add_argument("--playback-speed", type=float)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be at least one")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")

    plan = load_dynamics_plan(args.config)
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be positive")
        plan = replace(
            plan,
            excitation=replace(plan.excitation, optimization_trials=args.trials),
        )
    profiles = _selected_profiles(plan.excitation.profiles, args.profile)
    generated: dict[str, FourierTrajectory] = {}
    training_references: list[FourierTrajectory] = []
    for profile in profiles:
        trajectory = _generate_profile(
            plan,
            profile,
            reuse_existing=args.reuse_existing,
            training_references=tuple(training_references),
        )
        generated[profile.name] = trajectory
        if profile.role == "train":
            training_references.append(trajectory)
        preview = prepare_mujoco_preview(
            plan,
            trajectory,
            profile.trajectory_path.with_suffix(".scene.xml"),
        )
        print_preview_report(plan, trajectory, preview)
        if not args.no_visualize:
            play_mujoco_preview(
                plan,
                trajectory,
                preview,
                playback_speed=args.playback_speed,
                loops=args.loops,
                hold_seconds=args.hold_seconds,
            )

    selected_training = tuple(
        generated[profile.name]
        for profile in profiles
        if profile.role == "train"
    )
    if selected_training:
        condition, rank = combined_trajectory_diagnostics(plan, selected_training)
        print(
            f"combined training regressor: rank={rank} "
            f"condition={condition:.6g} profiles={len(selected_training)}"
        )
    return 0


def _generate_profile(
    plan,
    profile,
    *,
    reuse_existing,
    training_references,
):
    print(f"\n=== excitation profile {profile.name} role={profile.role} ===")
    if reuse_existing and profile.trajectory_path.is_file():
        trajectory = load_trajectory(profile.trajectory_path)
        print(f"Loaded existing excitation trajectory: {profile.trajectory_path}")
    else:
        references = training_references if profile.role == "train" else ()
        trajectory = generate_optimized_trajectory(plan, profile, references)
        save_trajectory(profile.trajectory_path, trajectory, plan, profile)
        print(f"Saved excitation trajectory: {profile.trajectory_path}")
    print(
        f"standalone rank={trajectory.regressor_rank} "
        f"condition={trajectory.condition_number:.6g} "
        f"samples={trajectory.time_s.size}"
    )
    print(
        "max |dq| [rad/s]: "
        + np.array2string(np.max(np.abs(trajectory.dq), axis=0), precision=4)
    )
    print(
        "max |ddq| [rad/s^2]: "
        + np.array2string(np.max(np.abs(trajectory.ddq), axis=0), precision=4)
    )
    return trajectory


def _selected_profiles(
    profiles: tuple[ExcitationProfile, ...],
    names: list[str],
) -> tuple[ExcitationProfile, ...]:
    if not names:
        return profiles
    by_name = {profile.name: profile for profile in profiles}
    missing = sorted(set(names).difference(by_name))
    if missing:
        raise ValueError(f"unknown excitation profiles: {missing}")
    return tuple(by_name[name] for name in names)


if __name__ == "__main__":
    raise SystemExit(main())
