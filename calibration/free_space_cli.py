from __future__ import annotations

import argparse
import gc
import json
import logging
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
    generate_coverage_trajectory,
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
from nero_collection.h5_writer import EpisodeBuffer

log = logging.getLogger(__name__)

FREE_SPACE_EPISODE_MAX_SAMPLES = 30_000


@dataclass(frozen=True)
class _SampleTiming:
    index: int
    command_s: float
    read_s: float
    safety_s: float
    append_s: float
    camera_s: float
    total_s: float


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
    )


def _collect(
    plan,
    *,
    backend_override,
    skip_can_setup,
    assume_yes,
    overwrite,
):
    run = plan.excitation.run
    trajectory = _load_and_validate(plan)
    _verify_hardware_preflight(plan, trajectory)
    if not plan.hardware.approved:
        raise RuntimeError(
            "hardware.approved is false; review the passing preflight and set it true"
        )
    collection = load_config(plan.collection_config_path)
    output_dir = collection.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = next_episode_index(output_dir, collection.output.prefix)
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
    lower = np.asarray(model.lowerPositionLimit) + plan.excitation.joint_limit_margin_rad
    upper = np.asarray(model.upperPositionLimit) - plan.excitation.joint_limit_margin_rad
    arm.connect()
    try:
        q_state = collection.robot_states["q"]
        dq_state = collection.robot_states["velocity"]
        ddq_state = collection.robot_states["acceleration"]
        arm.configure_state_alignment(
            collection.teleop.command.state_alignment_delay_s,
            plan.excitation.sample_rate_hz,
            q_state.mean_window,
            q_state.lowpass_cutoff_hz if q_state.lowpass else None,
            dq_state.lowpass_cutoff_hz if dq_state.lowpass else None,
            ddq_state.lowpass_cutoff_hz if ddq_state.lowpass else None,
            collection.teleop.command.maximum_can_frame_gap_s,
        )
        arm.set_follower_mode()
        time.sleep(collection.teleop.command.role_switch_settle_s)
        arm.enable()
        _wait_for_follower_role(arm, collection.teleop.command.role_switch_timeout_s)
        cameras.start()
        _move_to_start(arm, trajectory.q[0], plan)
        first_stored_index = int(
            np.searchsorted(
                trajectory.time_s,
                collection.output.discard_initial_s,
                side="left",
            )
        )
        stored_sample_capacity = max(0, trajectory.time_s.size - first_stored_index)
        episode_count = max(
            1,
            (stored_sample_capacity + FREE_SPACE_EPISODE_MAX_SAMPLES - 1)
            // FREE_SPACE_EPISODE_MAX_SAMPLES,
        )
        base_episode_metadata = {
            "source": "free_space_coverage",
            "coverage_config_path": str(plan.source_path),
            "coverage_config_sha256": config_sha256(plan.source_path),
            "trajectory_name": run.name,
            "trajectory_planner": plan.excitation.planner,
            "trajectory_seed": run.seed,
            "trajectory_path": str(run.trajectory_path),
            "trajectory_sha256": trajectory_sha256(trajectory),
            "reference_h5_path": trajectory.reference_h5_path,
            "reference_h5_sha256": trajectory.reference_h5_sha256,
            "workspace_min_m": np.asarray(trajectory.workspace_min_m).tolist(),
            "workspace_max_m": np.asarray(trajectory.workspace_max_m).tolist(),
            "workspace_convex_hull_volume_m3": trajectory.workspace_convex_hull_volume_m3,
            "segment_names": list(trajectory.segment_names),
            "segment_sample_counts": np.bincount(
                trajectory.segment_id,
                minlength=len(trajectory.segment_names),
            ).astype(int).tolist(),
            "sample_rate_hz": plan.excitation.sample_rate_hz,
            "trajectory_episode_count": episode_count,
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

        def new_buffer():
            episode_buffer = EpisodeBuffer(
                config=collection,
                arm_names=(plan.pair_name,),
                sample_rate_hz=plan.excitation.sample_rate_hz,
                episode_metadata=base_episode_metadata.copy(),
            )
            warm_up_started = time.monotonic()
            if episode_buffer.warm_up_online_inference():
                log.info(
                    "online tau_f warm-up complete elapsed=%.3fms samples=%d",
                    (time.monotonic() - warm_up_started) * 1e3,
                    episode_buffer.sample_count,
                )
            return episode_buffer

        def save_episode(
            episode_buffer,
            chunk_index,
            trajectory_from_index,
            trajectory_to_index,
        ):
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
                episode_index + chunk_index,
            )
            if output.exists() and not overwrite:
                raise RuntimeError(f"episode already exists: {output}; use --overwrite")
            saved = episode_buffer.save(output)
            print(
                f"Saved trajectory {run.name} episode "
                f"{chunk_index + 1}/{episode_count}: {saved} "
                f"samples={episode_buffer.sample_count}",
                flush=True,
            )

        def rotate_episode(
            episode_buffer,
            chunk_index,
            trajectory_from_index,
            trajectory_to_index,
        ):
            save_episode(
                episode_buffer,
                chunk_index,
                trajectory_from_index,
                trajectory_to_index,
            )
            return new_buffer()

        buffer = new_buffer()
        log.info(
            "starting hardware trajectory=%s duration=%.1fs samples=%d episodes=%d "
            "episode_max_samples=%d",
            run.name,
            trajectory.duration_s,
            trajectory.time_s.size,
            episode_count,
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
        )
        if buffer.sample_count:
            save_episode(
                buffer,
                chunk_index,
                chunk_from_index,
                trajectory.time_s.size,
            )
    finally:
        cameras.stop()
        # Disconnect leaves the gravity-loaded follower enabled at its final hold target.
        arm.disconnect()
    return 0


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
):
    if max_episode_samples <= 0:
        raise ValueError("max_episode_samples must be positive")
    period = 1.0 / plan.excitation.sample_rate_hz
    start = time.monotonic()
    discard_initial_s = buffer.config.output.discard_initial_s
    discard_complete_logged = discard_initial_s <= 0.0
    if discard_initial_s > 0.0:
        log.info(
            "discarding initial trajectory data duration=%.3fs; motion, safety and filters remain active",
            discard_initial_s,
        )
    previous_timestamp_us = None
    previous_segment_id = None
    timings = []
    chunk_index = 0
    chunk_from_index = None
    for index, q_cmd in enumerate(trajectory.q):
        deadline = start + index * period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif -remaining > max(2.0 * period, 0.03):
            timing_detail = _format_timing_diagnostics(timings)
            raise RuntimeError(
                f"100 Hz trajectory missed deadline by {-remaining:.4f}s at sample {index}; "
                f"{timing_detail}"
            )

        sample_started = time.monotonic()
        arm.command_joint_positions(q_cmd)
        command_finished = time.monotonic()
        state = _read_finite_arm_state(
            arm,
            timeout_s=max(2.0 * period, 0.03),
            context=f"trajectory sample {index}",
        )
        read_finished = time.monotonic()
        segment_id = int(trajectory.segment_id[index])
        segment_name = trajectory.segment_names[segment_id]
        if segment_id != previous_segment_id:
            log.info(
                "starting trajectory segment=%s index=%d time=%.3fs",
                segment_name,
                index,
                float(trajectory.time_s[index]),
            )
            previous_segment_id = segment_id
        _check_measurement(
            state,
            q_cmd,
            plan,
            lower,
            upper,
            sample_index=index,
            trajectory_time_s=trajectory.time_s[index],
            segment_name=segment_name,
            dq_cmd=trajectory.dq[index],
        )
        timestamp_us = int(state.q_timestamp_us or state.timestamp_us)
        if previous_timestamp_us is not None and timestamp_us > previous_timestamp_us:
            gap_s = (timestamp_us - previous_timestamp_us) * 1e-6
            if gap_s > plan.hardware.max_timestamp_gap_s:
                raise RuntimeError(f"follower timestamp gap exceeded limit: {gap_s:.6f}s")
        safety_finished = time.monotonic()
        store = float(trajectory.time_s[index]) >= discard_initial_s
        if store and not discard_complete_logged:
            log.info(
                "initial trajectory discard complete index=%d time=%.3fs; saving started",
                index,
                float(trajectory.time_s[index]),
            )
            discard_complete_logged = True
        buffer.append_teleop(
            timestamp_us,
            _follower_values(state, q_cmd),
            store=store,
        )
        if store and chunk_from_index is None and buffer.sample_count:
            chunk_from_index = index
        append_finished = time.monotonic()
        previous_timestamp_us = timestamp_us
        for frame in cameras.poll():
            if store:
                buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame)
        sample_finished = time.monotonic()
        timing = _SampleTiming(
            index=index,
            command_s=command_finished - sample_started,
            read_s=read_finished - command_finished,
            safety_s=safety_finished - read_finished,
            append_s=append_finished - safety_finished,
            camera_s=sample_finished - append_finished,
            total_s=sample_finished - sample_started,
        )
        timings.append(timing)
        if index == 0:
            log.info("100 Hz first-sample timing: %s", _format_sample_timing(timing))
        if (
            buffer.sample_count >= max_episode_samples
            and index + 1 < trajectory.time_s.size
        ):
            if rotate_episode is None:
                raise RuntimeError("rotate_episode callback is required to split episodes")
            log.info(
                "free-space episode full chunk=%d samples=%d trajectory_indices=[%d,%d); %s",
                chunk_index,
                buffer.sample_count,
                chunk_from_index,
                index + 1,
                _format_timing_diagnostics(timings),
            )
            pause_started = time.monotonic()
            buffer = rotate_episode(
                buffer,
                chunk_index,
                chunk_from_index,
                index + 1,
            )
            chunk_index += 1
            chunk_from_index = None
            previous_timestamp_us = None
            timings = []
            gc.collect()
            _read_finite_arm_state(
                arm,
                timeout_s=plan.hardware.max_timestamp_gap_s,
                context=f"episode boundary after trajectory sample {index}",
            )
            start += time.monotonic() - pause_started
    log.info("100 Hz trajectory timing summary: %s", _format_timing_diagnostics(timings))
    if chunk_from_index is None:
        chunk_from_index = trajectory.time_s.size
    return buffer, chunk_index, chunk_from_index


def _read_finite_arm_state(arm, *, timeout_s, context):
    deadline = time.monotonic() + float(timeout_s)
    invalid_fields = ()
    while True:
        state = arm.read_state()
        invalid_fields = tuple(
            name
            for name in ("q", "dq", "ddq", "torque")
            if not _is_finite_joint_vector(getattr(state, name, None))
        )
        if not invalid_fields:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"follower state alignment did not recover within {timeout_s:.3f}s "
                f"at {context}; invalid fields={invalid_fields}"
            )
        time.sleep(0.001)


def _is_finite_joint_vector(value) -> bool:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return vector.size == 7 and np.isfinite(vector).all()


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
        raise RuntimeError(f"follower crossed configured soft limits: {q}")
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


def _follower_values(state, q_cmd):
    return {
        "q_follower": ("q", np.asarray(state.q, dtype=np.float64)),
        "q_cmd": ("q", np.asarray(q_cmd, dtype=np.float64)),
        "dq_follower": ("velocity", np.asarray(state.dq, dtype=np.float64)),
        "ddq_follower": ("acceleration", np.asarray(state.ddq, dtype=np.float64)),
        "ee_pose_follower": ("ee_pose", np.asarray(state.ee_pose, dtype=np.float64)),
        "tau_follower": ("torque", np.asarray(state.torque, dtype=np.float64)),
        "current_follower": ("current", np.asarray(state.current, dtype=np.float64)),
    }


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


def _move_to_start(arm, target, plan):
    start = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
    if start.size != 7 or not np.isfinite(start).all():
        raise RuntimeError(f"cannot move from invalid follower state: {start}")
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
    print(f"  reference H5: {trajectory.reference_h5_path}")
    static = np.linalg.norm(trajectory.dq, axis=1) < 0.01
    print(f"  near-static samples: {np.count_nonzero(static)}/{trajectory.time_s.size}")


def _follower_endpoint(collection, pair_name) -> ArmEndpointConfig:
    for pair in collection.teleop.master_slave:
        if pair.name == pair_name:
            return pair.follower
    raise ValueError(f"pair {pair_name!r} was not found in collection config")


def _confirm_hardware_motion(endpoint, trajectory):
    message = (
        f"WARNING: {endpoint.name} on {endpoint.channel} will execute "
        f"{trajectory.name!r} for {trajectory.duration_s:.1f}s at 100 Hz. "
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
