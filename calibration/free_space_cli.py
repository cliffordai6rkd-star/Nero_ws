from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibration.dynamics_common import build_reduced_model, setup_socketcan
from calibration.free_space_coverage import (
    CoveragePlan,
    CoverageTrajectory,
    config_sha256,
    empirical_joint_range_statistics,
    effective_joint_position_limits,
    generate_coverage_trajectory,
    hardware_joint_position_limits,
    load_coverage_plan,
    load_coverage_trajectory,
    load_preflight_report,
    save_coverage_trajectory,
    trajectory_sha256,
    validate_coverage_trajectory,
    write_preflight_report,
)
from calibration.simulation import (
    play_mujoco_preview,
    prepare_mujoco_preview,
    print_preview_report,
)
from nero_collection.arms.factory import build_arm
from nero_collection.cameras import CameraManager
from nero_collection.config import ArmEndpointConfig, load_config
from nero_collection.episode_output import episode_path, next_episode_index
from nero_collection.fixed_rate import (
    FixedRateJointCollector,
    FixedRateSampleTiming as _SampleTiming,
)
from nero_collection.h5_writer import EpisodeBuffer

log = logging.getLogger(__name__)

FREE_SPACE_EPISODE_MAX_SAMPLES = 30_000


@dataclass(frozen=True)
class _CollectionResume:
    episode_paths: tuple[Path, ...]
    next_chunk_index: int
    next_trajectory_index: int


@dataclass(frozen=True)
class _CompletedChunk:
    path: Path
    chunk_index: int
    trajectory_from_index: int
    trajectory_to_index: int


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    plan = load_coverage_plan(args.config)
    if args.command == "generate":
        _generate(plan, overwrite=args.overwrite)
        return 0
    if args.command == "summary":
        trajectory = _load_and_validate(plan)
        _print_summary(plan, trajectory)
        return 0
    if args.command == "simulate":
        return _simulate(
            plan,
            reuse_existing=args.reuse_existing,
            headless=args.headless,
            playback_speed=args.playback_speed,
        )
    if args.command == "collect":
        return _collect(
            plan,
            backend_override=args.backend,
            skip_can_setup=args.skip_can_setup,
            assume_yes=args.yes,
            overwrite=args.overwrite,
            fresh=args.fresh,
        )
    if args.command == "run":
        return _run_pipeline(
            plan,
            reuse_existing=args.reuse_existing,
            headless=args.headless,
            playback_speed=args.playback_speed,
            backend_override=args.backend,
            skip_can_setup=args.skip_can_setup,
            assume_yes=args.yes,
            overwrite=args.overwrite,
        )
    raise RuntimeError(f"unsupported command: {args.command}")


def _generate(plan, *, overwrite):
    run = plan.excitation.run
    if run.trajectory_path.exists() and not overwrite:
        raise RuntimeError(
            f"trajectory already exists: {run.trajectory_path}; use --overwrite"
        )
    log.info("generating coverage trajectory=%s duration=%.1fs", run.name, run.duration_s)
    trajectory = generate_coverage_trajectory(plan)
    output = save_coverage_trajectory(run.trajectory_path, trajectory)
    _print_trajectory(trajectory)
    print(f"Saved trajectory: {output}")


def _simulate(plan, *, reuse_existing, headless, playback_speed):
    run = plan.excitation.run
    report = {
        "format": "nero_free_space_preflight/v2",
        "config_path": str(plan.source_path),
        "config_sha256": config_sha256(plan.source_path),
        "sample_rate_hz": plan.excitation.sample_rate_hz,
    }
    if reuse_existing:
        trajectory = load_coverage_trajectory(run.trajectory_path)
        validate_coverage_trajectory(trajectory, plan)
    else:
        trajectory = generate_coverage_trajectory(plan)
        save_coverage_trajectory(run.trajectory_path, trajectory)
    scene_path = run.trajectory_path.with_suffix(".scene.xml")
    print(f"\n=== simulation trajectory={run.name} ===")
    _print_trajectory(trajectory)
    preview = prepare_mujoco_preview(plan, trajectory, scene_path)
    print_preview_report(plan, trajectory, preview)
    passed = preview.workspace_violation_count == 0 and not preview.contact_events
    report["trajectory"] = {
        "name": run.name,
        "planner": plan.excitation.planner,
        "trajectory_path": str(run.trajectory_path),
        "trajectory_sha256": trajectory_sha256(trajectory),
        "reference_h5_path": trajectory.reference_h5_path,
        "reference_h5_sha256": trajectory.reference_h5_sha256,
        "source_h5_paths": list(trajectory.source_h5_paths),
        "workspace_min_m": np.asarray(trajectory.workspace_min_m).tolist(),
        "workspace_max_m": np.asarray(trajectory.workspace_max_m).tolist(),
        "workspace_convex_hull_volume_m3": trajectory.workspace_convex_hull_volume_m3,
        "samples": int(trajectory.time_s.size),
        "duration_s": trajectory.duration_s,
        "segment_names": list(trajectory.segment_names),
        "segment_sample_counts": np.bincount(
            trajectory.segment_id,
            minlength=len(trajectory.segment_names),
        ).astype(int).tolist(),
        "scene_path": str(scene_path),
        "workspace_violation_count": preview.workspace_violation_count,
        "contact_samples_checked": preview.contact_samples_checked,
        "contact_events": [
            {
                "kind": event.kind,
                "geometry_a": event.geometry_a,
                "geometry_b": event.geometry_b,
                "first_sample": event.first_sample,
                "last_sample": event.last_sample,
                "hit_sample_count": event.hit_sample_count,
                "minimum_distance_m": event.minimum_distance_m,
            }
            for event in preview.contact_events
        ],
        "passed": passed,
    }
    if plan.excitation.planner == "tau_refinement":
        report["trajectory"].update(
            {
                "replay_pose_count": plan.excitation.replay_pose_count,
                "static_pose_count": plan.excitation.static_pose_count,
                "target_switch_count": plan.excitation.jump_pose_count,
                "target_switch_delta_rad": [
                    plan.excitation.jump_min_delta_rad,
                    plan.excitation.jump_max_delta_rad,
                ],
            }
        )
    elif plan.excitation.planner == "joint_pose_coverage":
        _, range_model = build_reduced_model(plan.model)
        range_lower, range_upper = effective_joint_position_limits(
            range_model,
            plan.excitation,
        )
        range_stats = empirical_joint_range_statistics(plan.excitation)
        report["trajectory"].update(
            {
                "joint_candidate_count": plan.excitation.joint_candidate_count,
                "joint_pose_count": plan.excitation.joint_pose_count,
                "joint_route_passes": plan.excitation.joint_route_passes,
                "joint_connection_step_rad": (
                    plan.excitation.joint_connection_step_rad
                ),
                "joint_transition_speed_scale": (
                    plan.excitation.joint_transition_speed_scale
                ),
                "static_hold_s": plan.excitation.static_hold_s,
                "static_position_threshold_rad": (
                    plan.excitation.static_position_threshold_rad
                ),
                "static_velocity_threshold_rad_s": (
                    plan.excitation.static_velocity_threshold_rad_s
                ),
                "static_stability_duration_s": (
                    plan.excitation.static_stability_duration_s
                ),
                "static_stability_timeout_s": (
                    plan.excitation.static_stability_timeout_s
                ),
                "joint_range_source_directory": str(
                    range_stats["source_directory"]
                ),
                "joint_range_dataset": range_stats["dataset"],
                "joint_range_sample_count": range_stats["sample_count"],
                "joint_range_quantiles": range_stats[
                    "quantile_levels"
                ].tolist(),
                "joint_range_raw_min_rad": range_stats["minimum"].tolist(),
                "joint_range_raw_max_rad": range_stats["maximum"].tolist(),
                "joint_range_quantile_min_rad": range_stats[
                    "quantile_minimum"
                ].tolist(),
                "joint_range_quantile_max_rad": range_stats[
                    "quantile_maximum"
                ].tolist(),
                "joint_range_effective_min_rad": range_lower.tolist(),
                "joint_range_effective_max_rad": range_upper.tolist(),
                "joint_range_source_files": [
                    str(path) for path in range_stats["included_paths"]
                ],
                "joint_range_excluded_files": [
                    str(path) for path in range_stats["excluded_paths"]
                ],
            }
        )
    elif plan.excitation.planner == "representative_replay":
        report["trajectory"].update(
            {
                "selected_episode_count": len(trajectory.source_h5_paths),
                "source_h5_paths": list(trajectory.source_h5_paths),
                "replay_speed_scale": plan.excitation.replay_speed_scale,
                "replay_path_min_delta_rad": plan.excitation.replay_path_min_delta_rad,
            }
        )
    report["passed"] = passed
    if not headless:
        preview_speed = (
            plan.simulation.playback_speed
            if playback_speed is None
            else float(playback_speed)
        )
        log.info(
            "opening MuJoCo trajectory preview samples=%d duration=%.1fs playback_speed=%.1fx",
            trajectory.time_s.size,
            trajectory.duration_s,
            preview_speed,
        )
        play_mujoco_preview(
            plan,
            trajectory,
            preview,
            playback_speed=preview_speed,
            loops=1,
            hold_seconds=0.5,
        )
        log.info("MuJoCo trajectory preview completed")
    report_path = write_preflight_report(plan.hardware.preflight_report_path, report)
    print(f"Preflight report: {report_path}")
    print(f"Preflight passed: {report['passed']}")
    return 0 if report["passed"] else 2


def _run_pipeline(
    plan,
    *,
    reuse_existing,
    headless,
    playback_speed,
    backend_override,
    skip_can_setup,
    assume_yes,
    overwrite,
):
    if not reuse_existing:
        _generate(plan, overwrite=overwrite)
    simulation_status = _simulate(
        plan,
        reuse_existing=True,
        headless=headless,
        playback_speed=playback_speed,
    )
    if simulation_status != 0:
        log.error("automatic collection stopped because MuJoCo preflight failed")
        return simulation_status
    return _collect(
        plan,
        backend_override=backend_override,
        skip_can_setup=skip_can_setup,
        assume_yes=assume_yes,
        overwrite=overwrite,
        fresh=False,
    )


def _collect(
    plan,
    *,
    backend_override,
    skip_can_setup,
    assume_yes,
    overwrite,
    fresh=False,
):
    run = plan.excitation.run
    trajectory = _load_and_validate(plan)
    _verify_hardware_preflight(plan, trajectory)
    if not plan.hardware.approved:
        raise RuntimeError(
            "hardware.approved is false; review the passing preflight and set it true"
        )
    collection = load_config(plan.collection_config_path)
    trajectory_rate_hz = float(plan.excitation.sample_rate_hz)
    collection_rate_hz = float(collection.teleop.command.sample_rate_hz)
    if not np.isclose(trajectory_rate_hz, collection_rate_hz):
        raise RuntimeError(
            "free-space trajectory and collection sample rates must match: "
            f"trajectory={trajectory_rate_hz:.3f}Hz "
            f"collection={collection_rate_hz:.3f}Hz"
        )
    log.info(
        "online tau_ext inference disabled for free-space collection; "
        "recording raw state, command, and torque datasets only at %.1fHz",
        collection_rate_hz,
    )
    output_dir = collection.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    first_stored_index = int(
        np.searchsorted(
            trajectory.time_s,
            collection.output.discard_initial_s,
            side="left",
        )
    )
    trajectory_digest = trajectory_sha256(trajectory)
    coverage_config_digest = config_sha256(plan.source_path)
    resume = None
    if fresh:
        log.info(
            "fresh free-space collection requested trajectory=%s; existing "
            "episodes remain untouched and will not be used for resume",
            run.name,
        )
    else:
        resume = _find_free_space_resume(
            output_dir,
            collection.output.prefix,
            trajectory_name=run.name,
            trajectory_digest=trajectory_digest,
            coverage_config_digest=coverage_config_digest,
            first_stored_index=first_stored_index,
            trajectory_sample_count=trajectory.time_s.size,
        )
    start_index = 0 if resume is None else resume.next_trajectory_index
    initial_chunk_index = 0 if resume is None else resume.next_chunk_index
    if resume is not None:
        log.info(
            "resuming free-space trajectory=%s completed_episodes=%d "
            "next_chunk=%d next_trajectory_index=%d remaining=%.1fs",
            run.name,
            len(resume.episode_paths),
            initial_chunk_index,
            start_index,
            (trajectory.time_s.size - start_index)
            / plan.excitation.sample_rate_hz,
        )
    if start_index >= trajectory.time_s.size:
        print(
            f"Trajectory {run.name} is already fully collected in "
            f"{len(resume.episode_paths)} episodes; nothing to resume.",
            flush=True,
        )
        return 0
    next_output_episode_index = next_episode_index(
        output_dir,
        collection.output.prefix,
    )
    endpoint = _follower_endpoint(collection, plan.pair_name)
    backend = backend_override or collection.teleop.backend
    if backend == "pyagxarm" and not skip_can_setup:
        setup_socketcan(endpoint.channel, endpoint.bitrate)
    if not assume_yes:
        _confirm_hardware_motion(endpoint, trajectory)

    arm = build_arm(endpoint, backend)
    cameras = CameraManager.from_config(
        collection.cameras if plan.hardware.record_cameras else ()
    )
    _, model = build_reduced_model(plan.model)
    lower, upper = hardware_joint_position_limits(model, plan.excitation)
    from calibration.simulation import MujocoPoseSafetyChecker

    pose_safety_checker = MujocoPoseSafetyChecker(plan)
    arm.connect()
    save_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="free-space-h5-save",
    )
    pending_saves = []
    try:
        arm.set_follower_mode()
        time.sleep(collection.teleop.command.role_switch_settle_s)
        arm.enable()
        _wait_for_follower_role(arm, collection.teleop.command.role_switch_timeout_s)
        cameras.start()
        _move_to_start(
            arm,
            trajectory.q[start_index],
            plan,
            checker=pose_safety_checker,
        )
        base_episode_metadata = {
            "source": "free_space_coverage",
            "coverage_config_path": str(plan.source_path),
            "coverage_config_sha256": coverage_config_digest,
            "trajectory_name": run.name,
            "trajectory_planner": plan.excitation.planner,
            "trajectory_seed": run.seed,
            "trajectory_path": str(run.trajectory_path),
            "trajectory_sha256": trajectory_digest,
            "reference_h5_path": trajectory.reference_h5_path,
            "reference_h5_sha256": trajectory.reference_h5_sha256,
            "source_h5_paths": list(trajectory.source_h5_paths),
            "workspace_min_m": np.asarray(trajectory.workspace_min_m).tolist(),
            "workspace_max_m": np.asarray(trajectory.workspace_max_m).tolist(),
            "workspace_convex_hull_volume_m3": trajectory.workspace_convex_hull_volume_m3,
            "segment_names": list(trajectory.segment_names),
            "segment_sample_counts": np.bincount(
                trajectory.segment_id,
                minlength=len(trajectory.segment_names),
            ).astype(int).tolist(),
            "sample_rate_hz": collection_rate_hz,
            "command_sample_rate_hz": trajectory_rate_hz,
            "trajectory_sample_rate_hz": trajectory_rate_hz,
            "state_capture_policy": "fixed_rate_latest_sdk_cache",
            "state_capture_fixed_rate": True,
            "state_capture_continuous": True,
            "static_stability_samples_recorded": True,
            "episode_save_policy": "background_thread_atomic_h5",
            "online_tau_ext_inference_enabled": False,
            "online_tau_ext_inference_policy": "disabled_for_free_space_collection",
            "collection_resumed": resume is not None,
            "collection_fresh": bool(fresh),
            "collection_resume_from_trajectory_index": start_index,
            "collection_resume_prior_episode_count": initial_chunk_index,
            "trajectory_episode_max_samples": FREE_SPACE_EPISODE_MAX_SAMPLES,
        }
        if plan.excitation.planner == "tau_refinement":
            base_episode_metadata.update(
                {
                    "replay_pose_count": plan.excitation.replay_pose_count,
                    "static_pose_count": plan.excitation.static_pose_count,
                    "target_switch_count": plan.excitation.jump_pose_count,
                    "target_switch_delta_rad": [
                        plan.excitation.jump_min_delta_rad,
                        plan.excitation.jump_max_delta_rad,
                    ],
                }
            )
        elif plan.excitation.planner == "joint_pose_coverage":
            base_episode_metadata.update(
                {
                    "joint_candidate_count": plan.excitation.joint_candidate_count,
                    "joint_pose_count": plan.excitation.joint_pose_count,
                    "joint_route_passes": plan.excitation.joint_route_passes,
                    "joint_connection_step_rad": (
                        plan.excitation.joint_connection_step_rad
                    ),
                    "joint_transition_speed_scale": (
                        plan.excitation.joint_transition_speed_scale
                    ),
                    "joint_range_fraction": plan.excitation.joint_range_fraction,
                    "static_hold_s": plan.excitation.static_hold_s,
                    "static_position_threshold_rad": (
                        plan.excitation.static_position_threshold_rad
                    ),
                    "static_velocity_threshold_rad_s": (
                        plan.excitation.static_velocity_threshold_rad_s
                    ),
                    "static_stability_duration_s": (
                        plan.excitation.static_stability_duration_s
                    ),
                    "static_stability_timeout_s": (
                        plan.excitation.static_stability_timeout_s
                    ),
                }
            )
        elif plan.excitation.planner == "representative_replay":
            base_episode_metadata.update(
                {
                    "selected_episode_count": len(trajectory.source_h5_paths),
                    "source_h5_paths": list(trajectory.source_h5_paths),
                    "replay_speed_scale": plan.excitation.replay_speed_scale,
                    "replay_path_min_delta_rad": plan.excitation.replay_path_min_delta_rad,
                }
            )

        def new_buffer():
            episode_buffer = EpisodeBuffer(
                config=collection,
                arm_names=(plan.pair_name,),
                episode_metadata=base_episode_metadata.copy(),
                enable_online_tau_ext=False,
            )
            return episode_buffer

        def save_episode(
            episode_buffer,
            chunk_index,
            trajectory_from_index,
            trajectory_to_index,
        ):
            nonlocal next_output_episode_index
            episode_buffer.episode_metadata.update(
                {
                    "trajectory_episode_index": int(chunk_index),
                    "trajectory_sample_from_index": int(trajectory_from_index),
                    "trajectory_sample_to_index": int(trajectory_to_index),
                    "trajectory_episode_sample_count": episode_buffer.sample_count,
                }
            )
            output = episode_path(
                output_dir,
                collection.output.prefix,
                next_output_episode_index,
            )
            if output.exists() and not overwrite:
                raise RuntimeError(f"episode already exists: {output}; use --overwrite")
            next_output_episode_index += 1
            sample_count = episode_buffer.sample_count

            def save_in_background():
                saved = episode_buffer.save(output)
                print(
                    f"Saved trajectory {run.name} episode "
                    f"{chunk_index + 1}: {saved} samples={sample_count}",
                    flush=True,
                )
                return saved

            pending_saves.append(
                save_executor.submit(save_in_background)
            )

        def check_background_saves(*, wait):
            remaining = []
            for future in pending_saves:
                if wait or future.done():
                    future.result()
                else:
                    remaining.append(future)
            pending_saves[:] = remaining

        def rotate_episode(
            episode_buffer,
            chunk_index,
            trajectory_from_index,
            trajectory_to_index,
        ):
            check_background_saves(wait=False)
            save_episode(
                episode_buffer,
                chunk_index,
                trajectory_from_index,
                trajectory_to_index,
            )
            return new_buffer()

        buffer = new_buffer()
        log.info(
            "starting hardware trajectory=%s start_index=%d remaining_duration=%.1fs "
            "command_rate=%.1fHz command_samples=%d state_capture=fixed_rate_latest_sdk_cache "
            "state_episode_max_samples=%d",
            run.name,
            start_index,
            (trajectory.time_s.size - start_index)
            / plan.excitation.sample_rate_hz,
            plan.excitation.sample_rate_hz,
            trajectory.time_s.size,
            FREE_SPACE_EPISODE_MAX_SAMPLES,
        )
        buffer, chunk_index, chunk_from_index = _execute_trajectory(
            arm,
            cameras,
            buffer,
            trajectory,
            plan,
            lower,
            upper,
            max_episode_samples=FREE_SPACE_EPISODE_MAX_SAMPLES,
            rotate_episode=rotate_episode,
            start_index=start_index,
            initial_chunk_index=initial_chunk_index,
        )
        if buffer.sample_count:
            save_episode(
                buffer,
                chunk_index,
                chunk_from_index,
                trajectory.time_s.size,
            )
        check_background_saves(wait=True)
    finally:
        save_executor.shutdown(wait=True, cancel_futures=False)
        cameras.stop()
        # Disconnect leaves the gravity-loaded follower enabled at its final hold target.
        arm.disconnect()
    return 0


def _find_free_space_resume(
    output_dir,
    prefix,
    *,
    trajectory_name,
    trajectory_digest,
    coverage_config_digest,
    first_stored_index,
    trajectory_sample_count,
):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "h5py is required to inspect free-space resume files"
        ) from exc

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_.*\.h5$")
    indexed_paths = []
    for path in output_dir.glob(f"{prefix}_*.h5"):
        match = pattern.match(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort(key=lambda item: item[0])

    def read_chunk(path):
        try:
            with h5py.File(path, "r") as h5:
                raw_metadata = h5["metadata/episode_json"][()]
                if isinstance(raw_metadata, bytes):
                    raw_metadata = raw_metadata.decode("utf-8")
                metadata = json.loads(str(raw_metadata))
                sample_count = int(h5["teleop/timestamp_us"].shape[0])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            log.warning(
                "ignoring unreadable episode during resume scan path=%s: %s",
                path,
                exc,
            )
            return None
        if (
            metadata.get("source") != "free_space_coverage"
            or metadata.get("trajectory_name") != trajectory_name
            or metadata.get("trajectory_sha256") != trajectory_digest
            or metadata.get("coverage_config_sha256") != coverage_config_digest
        ):
            return None
        try:
            chunk = _CompletedChunk(
                path=path,
                chunk_index=int(metadata["trajectory_episode_index"]),
                trajectory_from_index=int(metadata["trajectory_sample_from_index"]),
                trajectory_to_index=int(metadata["trajectory_sample_to_index"]),
            )
            metadata_sample_count = int(metadata["trajectory_episode_sample_count"])
        except (KeyError, TypeError, ValueError):
            log.warning("ignoring incomplete free-space metadata path=%s", path)
            return None
        if metadata_sample_count != sample_count or sample_count <= 0:
            log.warning(
                "ignoring inconsistent free-space episode path=%s metadata_samples=%d "
                "stored_samples=%d",
                path,
                metadata_sample_count,
                sample_count,
            )
            return None
        if (
            chunk.chunk_index < 0
            or chunk.trajectory_from_index < first_stored_index
            or chunk.trajectory_to_index <= chunk.trajectory_from_index
            or chunk.trajectory_to_index > trajectory_sample_count
        ):
            log.warning("ignoring invalid free-space episode range path=%s", path)
            return None
        return chunk

    groups = []
    current = []
    for _, path in indexed_paths:
        chunk = read_chunk(path)
        starts_new_group = (
            chunk is not None
            and chunk.chunk_index == 0
            and chunk.trajectory_from_index == first_stored_index
        )
        extends_group = (
            chunk is not None
            and current
            and chunk.chunk_index == current[-1].chunk_index + 1
            and chunk.trajectory_from_index == current[-1].trajectory_to_index
        )
        if starts_new_group:
            if current:
                groups.append(tuple(current))
            current = [chunk]
        elif extends_group:
            current.append(chunk)
        else:
            if current:
                groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    if not groups:
        return None

    latest = groups[-1]
    last = latest[-1]
    return _CollectionResume(
        episode_paths=tuple(chunk.path for chunk in latest),
        next_chunk_index=last.chunk_index + 1,
        next_trajectory_index=last.trajectory_to_index,
    )


def _execute_trajectory(
    arm,
    cameras,
    buffer,
    trajectory,
    plan,
    lower,
    upper,
    *,
    max_episode_samples=FREE_SPACE_EPISODE_MAX_SAMPLES,
    rotate_episode=None,
    start_index=0,
    initial_chunk_index=0,
):
    if max_episode_samples <= 0:
        raise ValueError("max_episode_samples must be positive")
    if start_index < 0 or start_index >= trajectory.time_s.size:
        raise ValueError("start_index must select a trajectory sample")
    if initial_chunk_index < 0:
        raise ValueError("initial_chunk_index must be non-negative")
    period = 1.0 / plan.excitation.sample_rate_hz
    discard_initial_s = buffer.config.output.discard_initial_s
    discard_complete_logged = (
        discard_initial_s <= 0.0
        or float(trajectory.time_s[start_index]) >= discard_initial_s
    )
    if discard_initial_s > 0.0 and start_index == 0:
        log.info(
            "discarding initial trajectory data duration=%.3fs; motion, safety and filters remain active",
            discard_initial_s,
        )
    previous_segment_id = None
    timings = []
    captured_state_count = 0
    first_state_timestamp_us = None
    last_state_timestamp_us = None
    chunk_index = initial_chunk_index
    chunk_from_index = None
    collector = FixedRateJointCollector(
        arm,
        cameras,
        buffer,
        sample_rate_hz=plan.excitation.sample_rate_hz,
        maximum_lateness_s=plan.hardware.max_timestamp_gap_s,
        state_timeout_s=max(2.0 * period, plan.hardware.max_timestamp_gap_s),
    )

    def capture_tick(
        q_cmd,
        *,
        trajectory_index,
        segment_name,
        dq_cmd,
        store,
    ):
        nonlocal captured_state_count
        nonlocal first_state_timestamp_us
        nonlocal last_state_timestamp_us
        nonlocal chunk_from_index

        captured = collector.capture(
            q_cmd,
            store=store,
            context=f"trajectory sample {trajectory_index} segment={segment_name}",
            validate=lambda state, command: _check_measurement(
                state,
                command,
                plan,
                lower,
                upper,
                sample_index=trajectory_index,
                trajectory_time_s=trajectory.time_s[trajectory_index],
                segment_name=segment_name,
                dq_cmd=dq_cmd,
            ),
        )
        captured_state_count += 1
        if first_state_timestamp_us is None:
            first_state_timestamp_us = captured.timestamp_us
        last_state_timestamp_us = captured.timestamp_us
        if store and chunk_from_index is None and collector.buffer.sample_count:
            chunk_from_index = trajectory_index
        timings.append(captured.timing)
        if captured.timing.index == 0:
            log.info(
                "trajectory first-sample timing: %s",
                _format_sample_timing(captured.timing),
            )
        warning_threshold_s = max(2.0 * period, 0.03)
        if collector.last_lateness_s > warning_threshold_s:
            log.warning(
                "trajectory recovering from scheduling jitter lateness=%.4fs "
                "sample=%d segment=%s",
                collector.last_lateness_s,
                trajectory_index,
                segment_name,
            )
        return captured

    def rotate_if_full(trajectory_to_index):
        nonlocal buffer
        nonlocal chunk_index
        nonlocal chunk_from_index
        nonlocal timings
        if collector.buffer.sample_count < max_episode_samples:
            return
        if rotate_episode is None:
            raise RuntimeError("rotate_episode callback is required to split episodes")
        log.info(
            "free-space episode full chunk=%d samples=%d trajectory_indices=[%d,%d); %s",
            chunk_index,
            collector.buffer.sample_count,
            chunk_from_index,
            trajectory_to_index,
            _format_timing_diagnostics(timings),
        )
        buffer = rotate_episode(
            collector.buffer,
            chunk_index,
            chunk_from_index,
            trajectory_to_index,
        )
        collector.replace_buffer(buffer)
        chunk_index += 1
        chunk_from_index = None
        timings = []

    for index in range(start_index, trajectory.time_s.size):
        q_cmd = trajectory.q[index]
        segment_id = int(trajectory.segment_id[index])
        segment_name = trajectory.segment_names[segment_id]
        entering_static_hold = (
            getattr(plan.excitation, "planner", None) == "joint_pose_coverage"
            and segment_name == "joint_pose_hold"
            and (
                index == start_index
                or int(trajectory.segment_id[index - 1]) != segment_id
            )
        )
        store = float(trajectory.time_s[index]) >= discard_initial_s
        if entering_static_hold:
            _wait_for_static_stability(
                lambda: _capture_static_sample(
                    capture_tick,
                    rotate_if_full,
                    q_cmd,
                    index,
                    store,
                ),
                q_cmd,
                plan,
                sample_index=index,
                trajectory_time_s=float(trajectory.time_s[index]),
            )
        if segment_id != previous_segment_id:
            log.info(
                "starting trajectory segment=%s index=%d time=%.3fs",
                segment_name,
                index,
                float(trajectory.time_s[index]),
            )
            previous_segment_id = segment_id
        capture_tick(
            q_cmd,
            trajectory_index=index,
            segment_name=segment_name,
            dq_cmd=trajectory.dq[index],
            store=store,
        )
        if store and not discard_complete_logged:
            log.info(
                "initial trajectory discard complete index=%d time=%.3fs; saving started",
                index,
                float(trajectory.time_s[index]),
            )
            discard_complete_logged = True
        if index + 1 < trajectory.time_s.size:
            rotate_if_full(index + 1)
    state_duration_s = (
        0.0
        if first_state_timestamp_us is None or last_state_timestamp_us is None
        else (last_state_timestamp_us - first_state_timestamp_us) * 1.0e-6
    )
    state_rate_hz = (
        (captured_state_count - 1) / state_duration_s
        if captured_state_count > 1 and state_duration_s > 0.0
        else 0.0
    )
    log.info(
        "trajectory capture summary command_rate=%.1fHz commands=%d "
        "robot_state_samples=%d measured_state_rate=%.2fHz; %s",
        plan.excitation.sample_rate_hz,
        trajectory.time_s.size - start_index,
        captured_state_count,
        state_rate_hz,
        _format_timing_diagnostics(timings),
    )
    if chunk_from_index is None:
        chunk_from_index = trajectory.time_s.size
    return buffer, chunk_index, chunk_from_index


def _wait_for_static_stability(
    capture_state,
    q_target,
    plan,
    *,
    sample_index,
    trajectory_time_s,
) -> None:
    cfg = plan.excitation
    started_s = time.monotonic()
    deadline_s = started_s + cfg.static_stability_timeout_s
    stable_started_s = None
    last_position_error = float("inf")
    last_velocity = float("inf")
    sample_count = 0
    while True:
        state = capture_state().state
        q = np.asarray(state.q, dtype=np.float64).reshape(-1)
        dq = np.asarray(state.dq, dtype=np.float64).reshape(-1)
        last_position_error = float(np.max(np.abs(q - q_target)))
        last_velocity = float(np.max(np.abs(dq)))
        now_s = time.monotonic()
        stable = (
            last_position_error < cfg.static_position_threshold_rad
            and last_velocity < cfg.static_velocity_threshold_rad_s
        )
        if stable:
            if stable_started_s is None:
                stable_started_s = now_s
            if now_s - stable_started_s >= cfg.static_stability_duration_s:
                log.info(
                    "static target stabilized sample=%d reads=%d wait=%.3fs "
                    "position_error=%.6frad max_velocity=%.6frad/s",
                    sample_index,
                    sample_count + 1,
                    now_s - started_s,
                    last_position_error,
                    last_velocity,
                )
                return
        else:
            stable_started_s = None
        sample_count += 1
        if now_s >= deadline_s:
            raise RuntimeError(
                "static target did not stabilize before timeout: "
                f"sample={sample_index} wait={now_s - started_s:.3f}s "
                f"position_error={last_position_error:.6f}rad "
                f"position_limit={cfg.static_position_threshold_rad:.6f}rad "
                f"max_velocity={last_velocity:.6f}rad/s "
                f"velocity_limit={cfg.static_velocity_threshold_rad_s:.6f}rad/s"
            )


def _capture_static_sample(capture_tick, rotate_if_full, q_target, index, store):
    captured = capture_tick(
        q_target,
        trajectory_index=index,
        segment_name="joint_pose_stability_wait",
        dq_cmd=np.zeros_like(q_target),
        store=store,
    )
    rotate_if_full(index)
    return captured


def _format_sample_timing(timing: _SampleTiming) -> str:
    return (
        f"sample={timing.index} command={timing.command_s * 1e3:.3f}ms "
        f"read={timing.read_s * 1e3:.3f}ms safety={timing.safety_s * 1e3:.3f}ms "
        f"append={timing.append_s * 1e3:.3f}ms camera={timing.camera_s * 1e3:.3f}ms "
        f"total={timing.total_s * 1e3:.3f}ms"
    )


def _format_timing_diagnostics(timings: list[_SampleTiming]) -> str:
    if not timings:
        return "no completed sample timing available"
    previous = _format_sample_timing(timings[-1])
    stable = timings[1:]
    if not stable:
        return f"previous [{previous}]; stable samples=0"
    total_ms = np.asarray([timing.total_s for timing in stable]) * 1e3
    append_ms = np.asarray([timing.append_s for timing in stable]) * 1e3
    return (
        f"previous [{previous}]; stable samples={len(stable)} "
        f"total mean={np.mean(total_ms):.3f}ms p95={np.percentile(total_ms, 95):.3f}ms "
        f"max={np.max(total_ms):.3f}ms; append mean={np.mean(append_ms):.3f}ms "
        f"p95={np.percentile(append_ms, 95):.3f}ms max={np.max(append_ms):.3f}ms"
    )


def _check_measurement(
    state,
    q_cmd,
    plan,
    lower,
    upper,
    *,
    sample_index=None,
    trajectory_time_s=None,
    segment_name=None,
    dq_cmd=None,
):
    q = np.asarray(state.q, dtype=np.float64).reshape(-1)
    tau = np.asarray(state.torque, dtype=np.float64).reshape(-1)
    if q.size != 7 or not np.isfinite(q).all():
        raise RuntimeError(f"invalid follower joint measurement: {q}")
    if tau.size != 7 or not np.isfinite(tau).all():
        raise RuntimeError(f"invalid follower torque measurement: {tau}")
    if np.any(q < lower) or np.any(q > upper):
        raise RuntimeError(f"follower crossed URDF safety limits: {q}")
    error = np.abs(q - q_cmd)
    if np.any(error > plan.hardware.max_tracking_error_rad):
        context = ""
        if sample_index is not None:
            context = (
                f" sample={sample_index} time={float(trajectory_time_s):.3f}s "
                f"segment={segment_name} q={q} q_cmd={np.asarray(q_cmd)} "
                f"dq_cmd={np.asarray(dq_cmd)};"
            )
        raise RuntimeError(
            f"tracking error exceeded limit:{context} error={error}, "
            f"limit={plan.hardware.max_tracking_error_rad}"
        )
    if np.any(np.abs(tau) > plan.hardware.max_abs_torque_nm):
        raise RuntimeError(
            f"measured torque exceeded limit: tau={tau}, "
            f"limit={plan.hardware.max_abs_torque_nm}"
        )


def _verify_hardware_preflight(plan, trajectory):
    path = plan.hardware.preflight_report_path
    if not path.is_file():
        raise RuntimeError(f"preflight report does not exist: {path}")
    report = load_preflight_report(path)
    if report.get("passed") is not True:
        raise RuntimeError("preflight report did not pass the coverage trajectory")
    expected_config_hash = config_sha256(plan.source_path)
    if report.get("config_sha256") != expected_config_hash:
        raise RuntimeError("coverage config changed after simulation; rerun simulate")
    item = report.get("trajectory")
    if not isinstance(item, dict) or item.get("passed") is not True:
        raise RuntimeError("coverage trajectory has no passing preflight")
    if item.get("trajectory_sha256") != trajectory_sha256(trajectory):
        raise RuntimeError("coverage trajectory changed after simulation; rerun simulate")


def _move_to_start(arm, target, plan, *, checker=None):
    start = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
    if start.size != 7 or not np.isfinite(start).all():
        raise RuntimeError(f"cannot move from invalid follower state: {start}")
    if checker is not None and (
        not bool(checker.safe_mask(start[None, :])[0])
        or not checker.leg_is_safe(start, target)
    ):
        raise RuntimeError(
            "current follower pose cannot reach the trajectory start through the "
            "configured collision-free joint-space leg"
        )
    delta = np.asarray(target) - start
    duration = max(
        0.5,
        1.875 * float(np.max(np.abs(delta))) / plan.hardware.start_move_speed_rad_s,
    )
    steps = max(2, int(np.ceil(duration * 30.0)))
    start_t = time.monotonic()
    for step in range(1, steps + 1):
        phase = step / steps
        blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
        arm.command_joint_positions(start + blend * delta)
        remaining = start_t + step / 30.0 - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    time.sleep(0.5)
    final = np.asarray(arm.read_state().q, dtype=np.float64)
    if np.any(np.abs(final - target) > plan.hardware.max_tracking_error_rad):
        raise RuntimeError(f"follower did not reach trajectory start: {final}")


def _wait_for_follower_role(arm, timeout_s):
    log.info("waiting for follower role arm=%s timeout=%.2fs", arm.name, timeout_s)
    deadline = time.monotonic() + timeout_s
    last = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        last = arm.read_control_role(refresh=True)
        log.debug(
            "follower role check arm=%s attempt=%d detected=%s",
            arm.name,
            attempt,
            last or "unknown",
        )
        if last == "follower":
            log.info("follower role confirmed arm=%s attempts=%d", arm.name, attempt)
            return
        time.sleep(0.05)
    log.error("follower role confirmation timed out arm=%s detected=%s", arm.name, last or "unknown")
    raise RuntimeError(f"follower role was not confirmed; detected={last or 'unknown'}")


def _load_and_validate(plan):
    run = plan.excitation.run
    trajectory = load_coverage_trajectory(run.trajectory_path)
    if trajectory.name != run.name:
        raise RuntimeError(f"trajectory metadata does not match {run.name!r}")
    validate_coverage_trajectory(trajectory, plan)
    return trajectory


def _print_summary(plan, trajectory):
    print(f"sample rate: {plan.excitation.sample_rate_hz:.1f} Hz")
    print(f"config fingerprint: {config_sha256(plan.source_path)}")
    _print_trajectory(trajectory)


def _print_trajectory(trajectory):
    counts = np.bincount(trajectory.segment_id, minlength=len(trajectory.segment_names))
    rate = trajectory.time_s.size / trajectory.duration_s
    print(
        f"trajectory={trajectory.name} samples={trajectory.time_s.size} "
        f"duration={trajectory.duration_s:.3f}s rate={rate:.3f}Hz"
    )
    for name, count in zip(trajectory.segment_names, counts):
        print(f"  {name}: {count / rate:.3f}s")
    print("  q min: " + np.array2string(np.min(trajectory.q, axis=0), precision=4))
    print("  q max: " + np.array2string(np.max(trajectory.q, axis=0), precision=4))
    print("  max |dq|: " + np.array2string(np.max(np.abs(trajectory.dq), axis=0), precision=4))
    print("  max |ddq|: " + np.array2string(np.max(np.abs(trajectory.ddq), axis=0), precision=4))
    print("  workspace xyz min [m]: " + np.array2string(trajectory.workspace_min_m, precision=4))
    print("  workspace xyz max [m]: " + np.array2string(trajectory.workspace_max_m, precision=4))
    print(
        "  workspace convex hull [m^3]: "
        f"{trajectory.workspace_convex_hull_volume_m3:.6f}"
    )
    print(f"  reference H5: {trajectory.reference_h5_path or 'none'}")
    static = np.linalg.norm(trajectory.dq, axis=1) < 0.01
    print(f"  near-static samples: {np.count_nonzero(static)}/{trajectory.time_s.size}")


def _follower_endpoint(collection, pair_name) -> ArmEndpointConfig:
    for pair in collection.teleop.master_slave:
        if pair.name == pair_name:
            return pair.follower
    raise ValueError(f"pair {pair_name!r} was not found in collection config")


def _confirm_hardware_motion(endpoint, trajectory):
    rate = trajectory.time_s.size / trajectory.duration_s
    message = (
        f"WARNING: {endpoint.name} on {endpoint.channel} will execute "
        f"{trajectory.name!r} for {trajectory.duration_s:.1f}s at {rate:.1f} Hz. "
        "Clear the workcell, remove external "
        "contacts and payloads, and keep the emergency stop ready."
    )
    print(message, flush=True)
    log.info(
        "waiting for operator MOVE confirmation trajectory=%s duration=%.1fs",
        trajectory.name,
        trajectory.duration_s,
    )
    if _read_terminal_line("Type MOVE to continue: ").strip() != "MOVE":
        log.warning("hardware motion confirmation rejected by operator")
        raise RuntimeError("free-space collection cancelled")
    log.info("operator MOVE confirmation accepted")


def _read_terminal_line(prompt: str) -> str:
    if sys.stdin.isatty():
        log.debug("reading hardware confirmation from stdin")
        return input(prompt)
    try:
        log.info("stdin is not a TTY; reading hardware confirmation from /dev/tty")
        with open("/dev/tty", "w", encoding="utf-8", buffering=1) as terminal_out:
            terminal_out.write(prompt)
            terminal_out.flush()
        with open("/dev/tty", "r", encoding="utf-8") as terminal_in:
            return terminal_in.readline()
    except OSError as exc:
        raise RuntimeError(
            "hardware confirmation requires an interactive terminal; "
            "run this Python script from an interactive terminal or use --yes after review"
        ) from exc


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate, preflight and execute Nero free-space coverage trajectories."
    )
    parser.add_argument("--config", default="configs/free_space_coverage.yaml")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--overwrite", action="store_true")
    summary = subparsers.add_parser("summary")
    simulate = subparsers.add_parser(
        "simulate",
        help="run MuJoCo preflight and optionally show a fast trajectory preview",
    )
    simulate.add_argument(
        "--reuse-existing",
        action="store_true",
        help="load the generated NPZ instead of regenerating it",
    )
    simulate.add_argument(
        "--headless",
        action="store_true",
        help="skip the MuJoCo viewer after the full preflight scan",
    )
    simulate.add_argument(
        "--playback-speed",
        type=float,
        help="viewer playback multiplier (default: simulation.playback_speed)",
    )
    collect = subparsers.add_parser("collect")
    collect.add_argument("--backend", choices=("pyagxarm", "mock"))
    collect.add_argument("--skip-can-setup", action="store_true")
    collect.add_argument("--yes", action="store_true")
    collect.add_argument("--overwrite", action="store_true")
    collect.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "start this trajectory from index zero and allocate new episode "
            "numbers instead of resuming existing trajectory episodes"
        ),
    )
    run = subparsers.add_parser(
        "run",
        help="generate, preflight and execute the trajectory with the existing collector",
    )
    run.add_argument("--reuse-existing", action="store_true")
    run.add_argument("--headless", action="store_true")
    run.add_argument("--playback-speed", type=float)
    run.add_argument("--backend", choices=("pyagxarm", "mock"))
    run.add_argument("--skip-can-setup", action="store_true")
    run.add_argument("--yes", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
