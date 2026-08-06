#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from calibration.dynamics_common import DynamicsPlan, load_dynamics_plan
from calibration.simulation import (
    MujocoPoseSafetyChecker,
    launch_mujoco_live_preview,
    prepare_mujoco_live_preview,
    sync_mujoco_live_preview,
)
from inference.config import InferenceConfig, load_inference_config
from inference.pipeline import InferenceInput, NeroInferencePipeline


log = logging.getLogger("nero.h5_direct_ik")


@dataclass(frozen=True)
class H5InferenceEpisode:
    path: Path
    camera_name: str
    arm_name: str
    camera_indices: np.ndarray
    camera_timestamp_us: np.ndarray
    teleop_indices: np.ndarray
    teleop_timestamp_us: np.ndarray
    alignment_gap_us: np.ndarray
    frames: np.ndarray
    image_history: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    wrench_ext: np.ndarray
    wrench_history: np.ndarray

    @property
    def time_s(self) -> np.ndarray:
        return (
            self.camera_timestamp_us - int(self.camera_timestamp_us[0])
        ).astype(np.float64) * 1.0e-6


@dataclass(frozen=True)
class DirectIKInferenceResult:
    action: np.ndarray
    dp_action_chunk: np.ndarray
    q_ik: np.ndarray
    q_command: np.ndarray
    ik_iterations: np.ndarray
    ik_position_error_m: np.ndarray
    ik_rotation_error_rad: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deployed DP -> direct IK contract on a Nero runs H5 episode "
            "and play the predicted joint trajectory in MuJoCo."
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
        default=ROOT / "inference/configs/nero_direct_ik.yaml",
        help="inference YAML; predictor.enabled must be false",
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "calibration/config.yaml",
        help="MuJoCo scene/model configuration",
    )
    parser.add_argument("--camera", help="camera group; defaults to runtime.camera")
    parser.add_argument("--arm", help="arm in teleop.arm_names; defaults to runtime.arm_pair")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, help="exclusive camera-frame index")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--max-alignment-gap-ms",
        type=float,
        default=100.0,
        help="reject camera/state matches farther apart than this",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output NPZ; defaults next to the H5 with suffix _direct_ik.npz",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        help="generated MJCF scene; defaults next to the output NPZ",
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run MuJoCo safety/forward simulation without opening a window",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episode is not None and args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.stop_frame is not None and args.stop_frame <= args.start_frame:
        parser.error("--stop-frame must be greater than --start-frame")
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.max_alignment_gap_ms < 0 or not np.isfinite(args.max_alignment_gap_ms):
        parser.error("--max-alignment-gap-ms must be non-negative and finite")
    if args.playback_speed <= 0 or not np.isfinite(args.playback_speed):
        parser.error("--playback-speed must be positive and finite")
    if args.loops < 1:
        parser.error("--loops must be positive")
    if args.hold_seconds < 0 or not np.isfinite(args.hold_seconds):
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


def load_h5_episode(
    path: Path,
    *,
    camera_name: str,
    arm_name: str,
    start_frame: int,
    stop_frame: int | None,
    stride: int,
    max_frames: int | None,
    maximum_alignment_gap_us: int,
    observation_steps: int,
    wrench_history_steps: int,
    observation_step_s: float | None,
) -> H5InferenceEpisode:
    h5py = _import_h5py()
    required = {
        "timestamp": "teleop/timestamp_us",
        "q": "teleop/q_follower",
        "dq": "teleop/dq_follower",
        "ddq": "teleop/ddq_follower",
        "tau": "teleop/tau_follower",
        "wrench": "teleop/wrench_ext",
    }
    with h5py.File(path, "r") as h5:
        missing = [dataset for dataset in required.values() if dataset not in h5]
        if missing:
            raise ValueError(f"{path.name} is missing inference datasets: {missing}")
        camera_path = f"cameras/{camera_name}"
        if camera_path not in h5:
            available = sorted(h5.get("cameras", {}).keys())
            raise ValueError(
                f"camera {camera_name!r} not found in {path.name}; available={available}"
            )
        camera_group = h5[camera_path]
        for name in ("frames", "timestamp_us"):
            if name not in camera_group:
                raise ValueError(f"{camera_path}/{name} is missing from {path.name}")

        arm_names = tuple(
            _decode_h5_string(value)
            for value in h5["teleop"].attrs.get("arm_names", ())
        )
        if not arm_names:
            arm_names = (arm_name,)
        if arm_name not in arm_names:
            raise ValueError(
                f"arm {arm_name!r} not found in teleop.arm_names={list(arm_names)}"
            )
        arm_index = arm_names.index(arm_name)
        teleop_timestamp_us = _timestamp_vector(
            np.asarray(h5[required["timestamp"]][:], dtype=np.int64),
            required["timestamp"],
        )
        all_camera_timestamp_us = _timestamp_vector(
            np.asarray(camera_group["timestamp_us"][:], dtype=np.int64),
            f"{camera_path}/timestamp_us",
        )
        camera_indices = _frame_indices(
            len(all_camera_timestamp_us),
            start_frame,
            stop_frame,
            stride,
            max_frames,
        )
        camera_timestamp_us = all_camera_timestamp_us[camera_indices]
        image_indices = _observation_image_indices(
            all_camera_timestamp_us,
            camera_indices,
            observation_steps,
            observation_step_s,
        )
        image_timestamp_us = all_camera_timestamp_us[image_indices]
        observation_teleop_indices = nearest_indices(
            teleop_timestamp_us,
            image_timestamp_us.reshape(-1),
        ).reshape(image_indices.shape)
        teleop_indices = observation_teleop_indices[:, -1]
        matched_timestamp_us = teleop_timestamp_us[teleop_indices]
        alignment_gap_us = np.abs(matched_timestamp_us - camera_timestamp_us)
        if np.any(alignment_gap_us > maximum_alignment_gap_us):
            worst = int(np.argmax(alignment_gap_us))
            raise ValueError(
                "camera/state alignment exceeds limit: "
                f"camera_frame={int(camera_indices[worst])}, "
                f"gap={alignment_gap_us[worst] * 1.0e-3:.3f} ms, "
                f"limit={maximum_alignment_gap_us * 1.0e-3:.3f} ms"
            )

        count = len(teleop_timestamp_us)
        q = _select_arm_matrix(
            np.asarray(h5[required["q"]][:], dtype=np.float64),
            count,
            7,
            arm_names,
            arm_index,
            required["q"],
        )[teleop_indices]
        dq = _select_arm_matrix(
            np.asarray(h5[required["dq"]][:], dtype=np.float64),
            count,
            7,
            arm_names,
            arm_index,
            required["dq"],
        )[teleop_indices]
        ddq = _select_arm_matrix(
            np.asarray(h5[required["ddq"]][:], dtype=np.float64),
            count,
            7,
            arm_names,
            arm_index,
            required["ddq"],
        )[teleop_indices]
        tau = _select_arm_matrix(
            np.asarray(h5[required["tau"]][:], dtype=np.float64),
            count,
            7,
            arm_names,
            arm_index,
            required["tau"],
        )[teleop_indices]
        all_wrench = _select_arm_matrix(
            np.asarray(h5[required["wrench"]][:], dtype=np.float64),
            count,
            6,
            arm_names,
            arm_index,
            required["wrench"],
        )
        history_offsets = np.arange(
            wrench_history_steps - 1,
            -1,
            -1,
            dtype=np.int64,
        )
        wrench_indices = np.clip(
            observation_teleop_indices[:, :, None] - history_offsets,
            0,
            count - 1,
        )
        wrench_history = all_wrench[wrench_indices]
        all_frames = np.asarray(camera_group["frames"][:], dtype=np.uint8)
        image_history = all_frames[image_indices]
        frames = image_history[:, -1]
        wrench = wrench_history[:, -1, -1]

    if frames.ndim != 4 or frames.shape[0] != len(camera_indices) or frames.shape[-1] != 3:
        raise ValueError(
            f"{camera_path}/frames must have shape [N,H,W,3], got {frames.shape}"
        )
    return H5InferenceEpisode(
        path=path,
        camera_name=camera_name,
        arm_name=arm_name,
        camera_indices=camera_indices,
        camera_timestamp_us=camera_timestamp_us,
        teleop_indices=teleop_indices,
        teleop_timestamp_us=matched_timestamp_us,
        alignment_gap_us=alignment_gap_us,
        frames=frames,
        image_history=image_history,
        q=q,
        dq=dq,
        ddq=ddq,
        tau=tau,
        wrench_ext=wrench,
        wrench_history=wrench_history,
    )


def run_direct_ik_inference(
    pipeline: NeroInferencePipeline,
    episode: H5InferenceEpisode,
) -> DirectIKInferenceResult:
    if pipeline.config.predictor.enabled:
        raise ValueError(
            "H5 direct-IK inference requires predictor.enabled: false in the inference config"
        )
    actions: list[np.ndarray] = []
    q_ik: list[np.ndarray] = []
    q_command: list[np.ndarray] = []
    iterations: list[int] = []
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    dp_action_chunks: list[np.ndarray] = []
    pipeline.reset()
    for index in range(len(episode.camera_indices)):
        output = pipeline.step_direct_ik_observation_history(
            InferenceInput(
                q=episode.q[index],
                dq=episode.dq[index],
                ddq=episode.ddq[index],
                tau=episode.tau[index],
                image=episode.frames[index],
                wrench_ext=episode.wrench_ext[index],
                timestamp_s=float(episode.time_s[index]),
            ),
            episode.image_history[index],
            episode.wrench_history[index],
        )
        if output.ik_result is None or output.joint_position_command is None:
            raise RuntimeError("direct-IK pipeline returned no IK joint command")
        actions.append(output.action_target.copy())
        if output.dp_action_chunk is None:
            raise RuntimeError("direct-IK pipeline returned no DP action chunk")
        dp_action_chunks.append(output.dp_action_chunk.copy())
        q_ik.append(output.ik_result.q.copy())
        q_command.append(output.joint_position_command.copy())
        iterations.append(output.ik_result.iterations)
        position_errors.append(output.ik_result.position_error_m)
        rotation_errors.append(output.ik_result.rotation_error_rad)
        if (index + 1) % 25 == 0 or index + 1 == len(episode.camera_indices):
            log.info(
                "inferred %d/%d camera frames",
                index + 1,
                len(episode.camera_indices),
            )
    return DirectIKInferenceResult(
        action=np.stack(actions),
        dp_action_chunk=np.stack(dp_action_chunks),
        q_ik=np.stack(q_ik),
        q_command=np.stack(q_command),
        ik_iterations=np.asarray(iterations, dtype=np.int32),
        ik_position_error_m=np.asarray(position_errors, dtype=np.float64),
        ik_rotation_error_rad=np.asarray(rotation_errors, dtype=np.float64),
    )


def save_result(
    output_path: Path,
    config_path: Path,
    episode: H5InferenceEpisode,
    result: DirectIKInferenceResult,
) -> Path:
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "nero_h5_direct_ik/v2",
        "source_h5": str(episode.path),
        "inference_config": str(config_path.expanduser().resolve()),
        "camera": episode.camera_name,
        "arm": episode.arm_name,
        "samples": len(episode.camera_indices),
    }
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        camera_indices=episode.camera_indices,
        camera_timestamp_us=episode.camera_timestamp_us,
        teleop_indices=episode.teleop_indices,
        teleop_timestamp_us=episode.teleop_timestamp_us,
        alignment_gap_us=episode.alignment_gap_us,
        time_s=episode.time_s,
        q_observed=episode.q,
        action=result.action,
        dp_action_chunk=result.dp_action_chunk,
        q_ik=result.q_ik,
        q_command=result.q_command,
        ik_iterations=result.ik_iterations,
        ik_position_error_m=result.ik_position_error_m,
        ik_rotation_error_rad=result.ik_rotation_error_rad,
    )
    return output


def simulate_and_play(
    plan: DynamicsPlan,
    q: np.ndarray,
    dp_action_chunk: np.ndarray,
    camera_frames: np.ndarray,
    time_s: np.ndarray,
    scene_path: Path,
    *,
    playback_speed: float,
    loops: int,
    hold_seconds: float,
    no_viewer: bool,
) -> np.ndarray:
    _validate_q_trajectory(q, time_s)
    _validate_visualization_inputs(dp_action_chunk, camera_frames, len(q))
    checker = MujocoPoseSafetyChecker(plan)
    safe_mask = checker.safe_mask(q)
    preview = prepare_mujoco_live_preview(plan, q[0], scene_path)
    if no_viewer:
        import mujoco

        data = mujoco.MjData(preview.model)
        data.qpos[:] = preview.initial_qpos
        for pose in q:
            data.qpos[preview.joint_qpos_addresses] = pose
            data.qvel[:] = 0.0
            mujoco.mj_forward(preview.model, data)
        return safe_mask

    with launch_mujoco_live_preview(preview) as (data, viewer):
        for _ in range(loops):
            started_s = time.monotonic()
            for index, pose in enumerate(q):
                if not viewer.is_running():
                    return safe_mask
                deadline_s = started_s + float(time_s[index]) / playback_speed
                remaining_s = deadline_s - time.monotonic()
                if remaining_s > 0:
                    time.sleep(remaining_s)
                sync_mujoco_live_preview(preview, data, viewer, pose)
                _sync_mujoco_action_visualization(
                    preview,
                    data,
                    viewer,
                    dp_action_chunk[index],
                    camera_frames[index],
                    index,
                    plan.simulation.end_effector_body,
                )
        deadline_s = time.monotonic() + hold_seconds
        while viewer.is_running() and time.monotonic() < deadline_s:
            viewer.sync()
            time.sleep(0.02)
    return safe_mask


def _validate_visualization_inputs(
    dp_action_chunk: np.ndarray,
    camera_frames: np.ndarray,
    sample_count: int,
) -> None:
    """Validate the two streams before opening a viewer.

    Keeping this check outside the render loop makes malformed H5 data fail
    before a partially initialized MuJoCo window is left behind.
    """
    chunk = np.asarray(dp_action_chunk, dtype=np.float64)
    if (
        chunk.ndim != 3
        or chunk.shape[0] != sample_count
        or chunk.shape[2] != 7
        or chunk.shape[1] < 1
        or not np.all(np.isfinite(chunk))
    ):
        raise ValueError(
            "dp_action_chunk must have finite shape [N,H,7] with N matching q; "
            f"got {chunk.shape} for N={sample_count}"
        )
    frames = np.asarray(camera_frames)
    if (
        frames.ndim != 4
        or frames.shape[0] != sample_count
        or frames.shape[-1] != 3
        or frames.shape[1] < 1
        or frames.shape[2] < 1
    ):
        raise ValueError(
            "camera_frames must have shape [N,H,W,3] with N matching q; "
            f"got {frames.shape} for N={sample_count}"
        )


def _sync_mujoco_action_visualization(
    preview: Any,
    data: Any,
    viewer: Any,
    dp_action_chunk: np.ndarray,
    camera_frame: np.ndarray,
    frame_index: int,
    end_effector_body: str,
) -> None:
    """Draw raw DP waypoints, the executed endpoint, and the aligned RGB frame."""
    import mujoco

    chunk = np.asarray(dp_action_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"one DP action chunk must have shape [H,7], got {chunk.shape}")

    # User geoms are transient and must be rebuilt after every robot pose.
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        capacity = len(scene.geoms)
        marker_size = np.array([0.009, 0.009, 0.009], dtype=np.float64)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        for waypoint_index, action in enumerate(chunk[: max(capacity - 1, 0)]):
            position = action[:3]
            # Earlier waypoints are intentionally translucent; the ordering
            # remains visible without adding a second trajectory object.
            progress = (
                1.0
                if len(chunk) == 1
                else waypoint_index / (len(chunk) - 1)
            )
            alpha = 0.35 + 0.55 * progress
            rgba = np.array([0.05, 0.85, 0.35, alpha], dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                marker_size,
                position,
                identity,
                rgba,
            )
            scene.ngeom += 1

        body_id = mujoco.mj_name2id(
            preview.model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(end_effector_body),
        )
        if body_id < 0:
            # URDF-to-MJCF conversion keeps link names as body names in the
            # normal Nero scene; retain a useful fallback for older scenes.
            for fallback in ("gripper_base", "link7"):
                body_id = mujoco.mj_name2id(
                    preview.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    fallback,
                )
                if body_id >= 0:
                    break
        if body_id < 0:
            raise ValueError(
                "MuJoCo scene has no end-effector body for command marker: "
                f"{end_effector_body!r}"
            )
        command_position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
        if scene.ngeom < capacity:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([0.016, 0.016, 0.016], dtype=np.float64),
                command_position,
                identity,
                np.array([1.0, 0.45, 0.05, 1.0], dtype=np.float32),
            )
            scene.ngeom += 1

    # MuJoCo >= 3.6 exposes image overlays directly on the native viewer. The
    # fallback keeps playback usable on older installations, while the 3D
    # markers still render normally.
    _set_mujoco_camera_overlay(viewer, camera_frame)
    text_setter = getattr(viewer, "set_texts", None)
    if callable(text_setter):
        text_setter(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f"frame {frame_index}   green: raw DP chunk ({len(chunk)} targets)",
                "orange: executed q_command end-effector",
            )
        )
    viewer.sync()


def _set_mujoco_camera_overlay(viewer: Any, camera_frame: np.ndarray) -> None:
    import mujoco

    setter = getattr(viewer, "set_images", None)
    if not callable(setter):
        return
    viewport = viewer.viewport
    if viewport is None or viewport.width < 2 or viewport.height < 2:
        return
    frame = np.asarray(camera_frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return
    max_width = min(420, max(2, int(viewport.width * 0.34)))
    scale = max_width / float(frame.shape[1])
    image_width = max(2, int(round(frame.shape[1] * scale)))
    image_height = max(2, int(round(frame.shape[0] * scale)))
    max_height = max(2, int(viewport.height * 0.42))
    if image_height > max_height:
        scale = max_height / float(frame.shape[0])
        image_width = max(2, int(round(frame.shape[1] * scale)))
        image_height = max_height
    try:
        import cv2

        image = cv2.resize(
            frame,
            (image_width, image_height),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        # A nearest-neighbour fallback avoids making visualization depend on
        # an optional GUI-enabled OpenCV build.
        y = np.linspace(0, frame.shape[0] - 1, image_height).astype(np.int64)
        x = np.linspace(0, frame.shape[1] - 1, image_width).astype(np.int64)
        image = np.ascontiguousarray(frame[y[:, None], x[None, :]])
    rect = mujoco.MjrRect(
        max(0, int(viewport.width - image_width - 10)),
        max(0, int(viewport.height - image_height - 10)),
        image_width,
        image_height,
    )
    setter((rect, np.ascontiguousarray(image)))


def nearest_indices(sorted_timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    source = np.asarray(sorted_timestamps, dtype=np.int64).reshape(-1)
    values = np.asarray(targets, dtype=np.int64).reshape(-1)
    if source.size == 0:
        raise ValueError("cannot align against an empty timestamp vector")
    right = np.searchsorted(source, values, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - values) < np.abs(source[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def _observation_image_indices(
    camera_timestamp_us: np.ndarray,
    anchor_indices: np.ndarray,
    observation_steps: int,
    observation_step_s: float | None,
) -> np.ndarray:
    if observation_steps < 1:
        raise ValueError("observation_steps must be positive")
    anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    if observation_step_s is None:
        offsets = np.arange(observation_steps - 1, -1, -1, dtype=np.int64)
        return np.clip(anchors[:, None] - offsets[None], 0, len(camera_timestamp_us) - 1)
    step_us = int(round(float(observation_step_s) * 1.0e6))
    if step_us < 1:
        raise ValueError("observation_step_s must resolve to at least one microsecond")
    relative = np.arange(
        -(observation_steps - 1),
        1,
        dtype=np.int64,
    )
    targets = camera_timestamp_us[anchors, None] + relative[None] * step_us
    return nearest_indices(camera_timestamp_us, targets.reshape(-1)).reshape(targets.shape)


def _frame_indices(
    count: int,
    start: int,
    stop: int | None,
    stride: int,
    maximum: int | None,
) -> np.ndarray:
    end = count if stop is None else min(stop, count)
    if start >= end:
        raise ValueError(f"selected camera frame range is empty: [{start}, {end})")
    indices = np.arange(start, end, stride, dtype=np.int64)
    if maximum is not None:
        indices = indices[:maximum]
    if indices.size == 0:
        raise ValueError("selected camera frame range is empty")
    return indices


def _timestamp_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size < 1 or np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must contain positive, strictly increasing timestamps")
    return result


def _select_arm_matrix(
    value: np.ndarray,
    count: int,
    width: int,
    arm_names: tuple[str, ...],
    arm_index: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    arm_count = len(arm_names)
    if result.shape == (count, arm_count * width):
        result = result[:, arm_index * width : (arm_index + 1) * width]
    elif result.shape == (count, arm_count, width):
        result = result[:, arm_index]
    elif arm_count == 1 and result.shape == (count, width):
        pass
    else:
        raise ValueError(
            f"{name} does not match {arm_count} arm(s) of width {width}: {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_q_trajectory(q: np.ndarray, time_s: np.ndarray) -> None:
    poses = np.asarray(q, dtype=np.float64)
    timestamps = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.all(np.isfinite(poses)):
        raise ValueError(f"q trajectory must have finite shape [N,7], got {poses.shape}")
    if timestamps.shape != (len(poses),) or not np.all(np.isfinite(timestamps)):
        raise ValueError("simulation timestamps must match q trajectory length")
    if timestamps[0] != 0.0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("simulation timestamps must start at zero and increase")


def _validate_simulation_contract(
    inference_config: InferenceConfig,
    plan: DynamicsPlan,
) -> None:
    inference_urdf = inference_config.robot.urdf_path.resolve()
    simulation_urdf = plan.model.urdf_path.resolve()
    if inference_urdf != simulation_urdf:
        raise ValueError(
            "inference and simulation URDFs differ: "
            f"{inference_urdf} != {simulation_urdf}"
        )
    if set(inference_config.robot.locked_joint_names) != set(plan.model.locked_joint_names):
        raise ValueError(
            "inference and simulation locked_joint_names differ: "
            f"{inference_config.robot.locked_joint_names} != {plan.model.locked_joint_names}"
        )


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _import_h5py():
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "A working h5py compatible with the installed NumPy is required. "
            "Use the Nero environment or reinstall h5py>=3.11."
        ) from exc
    return h5py


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    episode_path = resolve_episode(args.source, args.episode)
    config = load_inference_config(args.config)
    if config.predictor.enabled:
        raise ValueError(
            f"set predictor.enabled: false in {args.config} before direct-IK inference"
        )
    camera_name = args.camera or config.runtime.camera
    arm_name = args.arm or config.runtime.arm_pair
    pipeline = NeroInferencePipeline(config)
    try:
        episode = load_h5_episode(
            episode_path,
            camera_name=camera_name,
            arm_name=arm_name,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame,
            stride=args.stride,
            max_frames=args.max_frames,
            maximum_alignment_gap_us=int(round(args.max_alignment_gap_ms * 1.0e3)),
            observation_steps=pipeline.observation_steps,
            wrench_history_steps=pipeline.wrench_history_steps,
            observation_step_s=pipeline.observation_step_s,
        )
        log.info(
            "DP observation contract image_steps=%d wrench_history_steps=%d step_s=%s",
            pipeline.observation_steps,
            pipeline.wrench_history_steps,
            pipeline.observation_step_s,
        )
        result = run_direct_ik_inference(pipeline, episode)
    finally:
        pipeline.close()
    log.info(
        "loaded %s frames=%d camera=%s arm=%s max_alignment_gap=%.3f ms",
        episode.path,
        len(episode.camera_indices),
        episode.camera_name,
        episode.arm_name,
        float(np.max(episode.alignment_gap_us)) * 1.0e-3,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else episode.path.with_name(f"{episode.path.stem}_direct_ik.npz")
    )
    output = save_result(output, args.config, episode, result)
    plan = load_dynamics_plan(args.simulation_config)
    _validate_simulation_contract(config, plan)
    scene_path = (
        args.scene_output.expanduser().resolve()
        if args.scene_output is not None
        else output.with_suffix(".scene.xml")
    )
    safe_mask = simulate_and_play(
        plan,
        result.q_command,
        result.dp_action_chunk,
        episode.frames,
        episode.time_s,
        scene_path,
        playback_speed=args.playback_speed,
        loops=args.loops,
        hold_seconds=args.hold_seconds,
        no_viewer=args.no_viewer,
    )
    print(f"source H5: {episode.path}")
    print(f"result NPZ: {output}")
    print(f"MuJoCo scene: {scene_path}")
    print(f"frames inferred: {len(episode.camera_indices)}")
    print(
        "camera/state alignment max [ms]: "
        f"{np.max(episode.alignment_gap_us) * 1.0e-3:.3f}"
    )
    print(
        "IK residual max: "
        f"position={np.max(result.ik_position_error_m):.6g} m, "
        f"rotation={np.max(result.ik_rotation_error_rad):.6g} rad"
    )
    print(f"MuJoCo safe poses: {np.count_nonzero(safe_mask)}/{len(safe_mask)}")


if __name__ == "__main__":
    main()
