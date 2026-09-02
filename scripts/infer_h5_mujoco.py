#!/usr/bin/env python3
"""Run the independent H5 -> inference -> MuJoCo dynamics pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.dynamics_common import load_dynamics_plan
from inference.config import InferenceConfig, load_inference_config
from inference.h5_observation_stream import H5ObservationEpisode
from inference.mujoco_backend import MujocoDynamicsBackend
from inference.simulation_runner import (
    EXECUTION_MODES,
    OBSERVATION_MODES,
    SimulationRunnerConfig,
    backend_config_from_inference,
    build_pipeline,
    make_observation_stream,
    run_h5_simulation,
)


log = logging.getLogger("nero.infer_h5_mujoco")


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


def apply_checkpoint_overrides(
    config: InferenceConfig,
    *,
    dp_checkpoint: Path | None = None,
    pinn_checkpoint: Path | None = None,
    contactworldmodel: Path | None = None,
) -> InferenceConfig:
    """Return ``config`` with optional command-line checkpoint replacements.

    Checkpoint locations are intentionally not inferred from the H5 file.  A
    DP checkpoint can be trained for a different task and a PINN path may
    point to a distilled student, so selecting either one must remain an
    explicit user decision.
    """

    if dp_checkpoint is not None:
        config = replace(
            config,
            dp_checkpoint=replace(
                config.dp_checkpoint,
                path=dp_checkpoint.expanduser().resolve(),
            ),
        )
    if pinn_checkpoint is not None and contactworldmodel is not None:
        raise ValueError("pass only one of pinn_checkpoint/contactworldmodel")
    contact_override = contactworldmodel or pinn_checkpoint
    if contact_override is not None:
        if config.contactworldmodel is None:
            raise ValueError(
                "--contactworldmodel requires predictor.enabled=true and a Contact WM config"
            )
        config = replace(
            config,
            contactworldmodel=replace(
                config.contactworldmodel,
                path=contact_override.expanduser().resolve(),
            ),
            pinn_checkpoint=replace(
                config.pinn_checkpoint,
                path=contact_override.expanduser().resolve(),
            ),
        )
    return config


def apply_execution_mode_override(
    config: InferenceConfig,
    mode: str | None,
) -> InferenceConfig:
    """Keep the policy contract and MuJoCo routing on the same mode.

    The runner can route an output explicitly, but doing the override on the
    config as well keeps contact-WM ``q``, ``mtc`` and ``tau`` contracts aligned.
    """

    if mode is None:
        return config
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in {"mtc", "q", "tau"}:
        raise ValueError(f"unsupported execution mode override: {mode!r}")
    return replace(
        config,
        execution=replace(config.execution, mode=normalized),
    )


def validate_checkpoint_paths(config: InferenceConfig) -> None:
    """Fail before model construction with actionable path diagnostics."""

    paths = [("DP", config.dp_checkpoint.path)]
    if config.predictor.enabled:
        if config.contactworldmodel is None:
            raise ValueError("predictor.enabled=true requires contactworldmodel")
        paths.append(("ContactWorldModel", config.contactworldmodel.path))
    for kind, path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"{kind} checkpoint does not exist: {path}. "
                f"Pass --{'dp' if kind == 'DP' else 'pinn'}-checkpoint or edit --config."
            )
    dino_path = config.dp_checkpoint.dino_model_path
    if dino_path is not None and not dino_path.is_dir():
        raise FileNotFoundError(
            f"DP DINO model directory does not exist: {dino_path}. "
            "Set dp_checkpoint.dino_model_path in the config."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nero DP/contact-WM inference on recorded H5 observations and "
            "execute q, MTC, or tau commands in MuJoCo."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="episode H5 file, or a runs directory used with --episode",
    )
    parser.add_argument("--episode", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "inference/configs/nero_contact_wm.yaml",
        help="inference config containing DP/Contact WM checkpoints and execution.mode",
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "calibration/config.yaml",
        help="calibration dynamics and MuJoCo scene config",
    )
    parser.add_argument(
        "--dp-checkpoint",
        type=Path,
        help="override config.dp_checkpoint.path for this replay",
    )
    parser.add_argument(
        "--pinn-checkpoint",
        "--contactworldmodel",
        type=Path,
        dest="pinn_checkpoint",
        help="override config.contactworldmodel.path for this replay",
    )
    parser.add_argument("--camera", help="H5 camera group; defaults to runtime.camera")
    parser.add_argument(
        "--arm",
        help="H5 teleop arm name; defaults to runtime.arm_pair (use --arm-index when unnamed)",
    )
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument(
        "--strict-ddq",
        action="store_true",
        help="reject legacy H5 files without teleop/ddq_follower",
    )
    parser.add_argument(
        "--mode",
        choices=EXECUTION_MODES,
        help="override inference execution.mode for simulation",
    )
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="recorded",
        help="recorded H5 q/dq/tau, or hybrid_closed_loop simulation feedback",
    )
    parser.add_argument("--state-rate-hz", type=float, default=100.0)
    parser.add_argument("--camera-rate-hz", type=float, default=25.0)
    parser.add_argument(
        "--history-steps",
        type=int,
        help="WM q/tau history length; defaults to checkpoint contract or 50",
    )
    parser.add_argument(
        "--camera-history-steps",
        type=int,
        help="image history length for direct-IK contract; defaults to DP n_obs_steps",
    )
    parser.add_argument(
        "--camera-history-step-s",
        type=float,
        help="image history spacing; defaults to DP checkpoint timestamp_step_sec or 0.1",
    )
    parser.add_argument(
        "--nearest-state",
        action="store_true",
        help="allow nearest H5 state alignment; default is causal previous",
    )
    parser.add_argument(
        "--no-left-pad",
        action="store_true",
        help="discard ticks until requested state/camera histories are complete",
    )
    parser.add_argument("--max-state-gap-ms", type=float)
    parser.add_argument("--max-camera-age-ms", type=float)
    parser.add_argument("--physics-dt-s", type=float, default=0.001)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--stop-s", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="output NPZ; defaults to <episode>_mujoco_inference.npz",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        help="generated actuator-enabled MJCF; defaults next to output",
    )
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo passive viewer")
    parser.add_argument("--realtime", action="store_true", help="pace replay at recorded time")
    parser.add_argument(
        "--allow-asynchronous",
        action="store_true",
        help="allow thread-scheduled predictor inference (results are not deterministic)",
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
    for name in ("state_rate_hz", "camera_rate_hz", "physics_dt_s"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("start_s", "stop_s", "max_state_gap_ms", "max_camera_age_ms"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    if args.stop_s is not None and args.stop_s <= args.start_s:
        parser.error("--stop-s must be greater than --start-s")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.history_steps is not None and args.history_steps < 1:
        parser.error("--history-steps must be positive")
    if args.camera_history_steps is not None and args.camera_history_steps < 1:
        parser.error("--camera-history-steps must be positive")
    if args.camera_history_step_s is not None and (
        not np.isfinite(args.camera_history_step_s) or args.camera_history_step_s <= 0.0
    ):
        parser.error("--camera-history-step-s must be positive and finite")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = resolve_episode(args.source, args.episode)
    inference_config = load_inference_config(args.config)
    try:
        inference_config = apply_checkpoint_overrides(
            inference_config,
            dp_checkpoint=args.dp_checkpoint,
            pinn_checkpoint=args.pinn_checkpoint,
        )
        inference_config = apply_execution_mode_override(
            inference_config,
            args.mode,
        )
        validate_checkpoint_paths(inference_config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    camera_name = args.camera or inference_config.runtime.camera
    arm_name = args.arm or inference_config.runtime.arm_pair

    # Construct the policy before loading H5 cameras so the replay can honor
    # the checkpoint's single- or multi-camera observation contract.
    pipeline = build_pipeline(inference_config)
    checkpoint_image_keys = tuple(
        str(key) for key in getattr(pipeline.dp, "image_keys", ())
    )
    if not checkpoint_image_keys:
        checkpoint_image_keys = (str(getattr(pipeline.dp, "image_key", camera_name)),)
    if camera_name not in checkpoint_image_keys:
        raise SystemExit(
            f"primary camera {camera_name!r} is not in the DP checkpoint cameras "
            f"{list(checkpoint_image_keys)}; pass --camera with one of those names"
        )

    try:
        episode = H5ObservationEpisode.from_h5(
            path,
            camera_name=camera_name,
            camera_names=checkpoint_image_keys,
            arm_name=arm_name,
            arm_index=args.arm_index,
            derive_ddq_if_missing=not args.strict_ddq,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    # The policy also determines the image history spacing used by the stream.
    default_history = int(getattr(pipeline, "_contact_history_horizon", 50))
    history_steps = args.history_steps or default_history
    camera_history_steps = args.camera_history_steps or int(
        max(1, getattr(pipeline, "observation_steps", 1))
    )
    camera_history_step_s = args.camera_history_step_s
    if camera_history_step_s is None:
        camera_history_step_s = getattr(pipeline, "observation_step_s", None) or 0.1
    runner_config = SimulationRunnerConfig(
        observation_mode=args.observation_mode,
        execution_mode=args.mode,
        state_rate_hz=args.state_rate_hz,
        camera_rate_hz=args.camera_rate_hz,
        history_steps=history_steps,
        camera_history_steps=camera_history_steps,
        camera_history_step_s=camera_history_step_s,
        left_pad=not args.no_left_pad,
        state_alignment="nearest" if args.nearest_state else "previous",
        max_state_alignment_gap_s=(
            None if args.max_state_gap_ms is None else args.max_state_gap_ms * 1.0e-3
        ),
        max_camera_age_s=(
            None if args.max_camera_age_ms is None else args.max_camera_age_ms * 1.0e-3
        ),
        physics_dt_s=args.physics_dt_s,
        max_steps=args.max_steps,
        realtime=args.realtime,
        viewer=args.viewer,
        allow_asynchronous=args.allow_asynchronous,
    )
    start_us = int(episode.state_timestamp_us[0] + round(args.start_s * 1.0e6))
    stop_us = (
        None
        if args.stop_s is None
        else int(episode.state_timestamp_us[0] + round(args.stop_s * 1.0e6))
    )
    stream = make_observation_stream(
        episode,
        runner_config,
        start_timestamp_us=start_us,
        stop_timestamp_us=stop_us,
    )

    plan = load_dynamics_plan(args.simulation_config)
    backend = MujocoDynamicsBackend(
        plan,
        stream[0].q,
        config=backend_config_from_inference(inference_config, plan, runner_config),
        scene_path=args.scene_output,
    )
    backend.reset(stream[0].q, stream[0].dq)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else path.with_name(f"{path.stem}_mujoco_inference.npz")
    )
    result = run_h5_simulation(
        stream,
        pipeline,
        backend,
        config=runner_config,
    )
    result.save_npz(
        output_path,
        metadata={
            "source_h5": str(path),
            "inference_config": str(args.config.expanduser().resolve()),
            "simulation_config": str(args.simulation_config.expanduser().resolve()),
            "camera": camera_name,
            "arm": episode.arm_name,
        },
    )
    print(
        f"H5 MuJoCo inference complete: {path}\n"
        f"samples={result.sample_count} duration={result.timestamps_s[-1]:.3f}s "
        f"observation_mode={result.observation_mode} execution_mode={result.execution_mode}\n"
        f"trace={output_path}\n"
        f"scene={backend.scene_path if backend.scene_path is not None else '(not written)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
