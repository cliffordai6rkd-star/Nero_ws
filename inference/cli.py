from __future__ import annotations

import argparse
import json
import logging

from inference.checkpoints import restore_checkpoint_model
from inference.config import load_inference_config
from inference.pipeline import _dp_model_overrides
from inference.runtime import NeroInferenceRuntime
from nero_collection.keyboard import TerminalKeys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nero DP inference with direct-q/tau and MTC control"
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--check",
        action="store_true",
        help="restore configured checkpoint-defined models and validate runtime configuration",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="start the online inference loop (the default when no mode is given)",
    )
    parser.add_argument("--backend", choices=("pyagxarm", "mock"), default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="start paused; press s for exactly one inference/control cycle, c to resume",
    )
    parser.add_argument(
        "--enable-command",
        action="store_true",
        help="send torque or direct-IK joint commands; otherwise inference is read-only",
    )
    parser.add_argument("--skip-can-setup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not args.check and not args.run:
        args.run = True
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_inference_config(args.config)
    if args.check:
        dp = restore_checkpoint_model(
            config.dp_checkpoint.path,
            config.dp_checkpoint.device,
            use_ema=config.dp_checkpoint.use_ema,
            kind="DP",
            model_overrides=_dp_model_overrides(config),
        )
        pinn = None
        if config.predictor.enabled:
            if config.pinn_checkpoint is None:
                raise ValueError(
                    "contactworldmodel is required when predictor.enabled=true"
                )
            pinn = restore_checkpoint_model(
                config.pinn_checkpoint.path,
                config.pinn_checkpoint.device,
                use_ema=config.pinn_checkpoint.use_ema,
                kind="PINN",
                pinn_mode=config.predictor.mode,
            )
        dp_model = getattr(dp, "model", None)
        diffusion = getattr(dp_model, "diffusion", None)
        scheduler = getattr(diffusion, "noise_scheduler", None)
        print(
            json.dumps(
                {
                    "dp": type(dp).__qualname__,
                    "pinn": None if pinn is None else type(pinn).__qualname__,
                    "dp_contract": {
                        "action_semantic": config.action,
                        "action_dim": getattr(dp, "action_dim", None),
                        "horizon": getattr(dp, "horizon", None),
                        "n_obs_steps": getattr(dp, "n_obs_steps", None),
                        "n_action_steps": getattr(dp, "n_action_steps", None),
                        "action_start_index": getattr(dp, "action_start_index", None),
                        "image_keys": list(getattr(dp, "image_keys", ())),
                        "image_shapes": getattr(dp, "image_shapes", {}),
                        "normalizer_restored": bool(
                            getattr(dp, "preprocessor", None) is not None
                            and getattr(dp, "postprocessor", None) is not None
                        ),
                        "scheduler": None if scheduler is None else type(scheduler).__name__,
                        "actual_inference_steps": getattr(diffusion, "num_inference_steps", None),
                    },
                    "wm_contract": None
                    if pinn is None
                    else {
                        "inputs": list(getattr(pinn, "inputs", ())),
                        "history_horizon": getattr(pinn, "history_horizon", None),
                        "prediction_horizon": getattr(pinn, "future_horizon", None),
                        "action_condition_horizon": getattr(
                            pinn, "action_condition_horizon", None
                        ),
                        "joint_dim": getattr(pinn, "joint_dim", None),
                        "action_dim": getattr(pinn, "action_dim", None),
                        "flow_inference_steps": getattr(
                            pinn, "flow_inference_steps", None
                        ),
                        "flow_solver": getattr(pinn, "flow_solver", None),
                        "normalizer_restored": bool(
                            getattr(pinn, "_inference_normalizer", None)
                        ),
                    },
                    "predictor_enabled": config.predictor.enabled,
                    "predictor_mode": config.predictor.mode,
                    "execution_mode": config.execution.mode,
                    "inference_mode": config.predictor.inference_mode,
                    "mujoco_visualization": {
                        "enabled": config.mujoco_visualization.enabled,
                        "num_future_samples": config.mujoco_visualization.num_future_samples,
                        "mujoco_model_path": None if config.mujoco_visualization.mujoco_model_path is None else str(config.mujoco_visualization.mujoco_model_path),
                        "ee_site_name": config.mujoco_visualization.ee_site_name,
                        "ee_body_name": config.mujoco_visualization.ee_body_name,
                        "robot_joint_names": list(config.mujoco_visualization.robot_joint_names),
                        "prediction_visualization_hz": config.mujoco_visualization.prediction_visualization_hz,
                        "render_fps": config.mujoco_visualization.render_fps,
                        "observed_q_update_hz": config.mujoco_visualization.observed_q_update_hz,
                        "latest_only": config.mujoco_visualization.latest_only,
                    },
                    "action_chunk_mode": config.predictor.action_chunk_mode,
                    "action_step_s": config.predictor.action_step_s,
                    "action_execution_mode": config.predictor.action_execution_mode,
                    "action_interpolation_duration_s": (
                        config.predictor.action_interpolation_duration_s
                    ),
                    "action_interpolation_steps": (
                        config.predictor.action_interpolation_steps
                    ),
                    "architecture": {
                        "enabled": config.architecture.enabled,
                        "policy_type": config.architecture.policy_type,
                        "world_model_type": config.architecture.world_model_type,
                    },
                    "dp_sampling_method": config.dp_sampling.method,
                    "dp_inference_steps": config.dp_sampling.num_inference_steps,
                    "maximum_inference_steps": config.runtime.maximum_inference_steps,
                    "collection_config": str(config.runtime.collection_config),
                    "status": "ok",
                },
                ensure_ascii=False,
            )
        )
        return

    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    runtime = NeroInferenceRuntime(
        config,
        backend=args.backend,
        command_enabled=args.enable_command,
    )
    if (
        runtime.collection.teleop.backend == "pyagxarm"
        and not args.skip_can_setup
    ):
        from nero_collection.cli import _setup_can_interfaces

        _setup_can_interfaces(runtime.collection)
    mode_text = "COMMAND ENABLED" if args.enable_command else "read-only"
    print(
        f"Starting Nero inference ({mode_text}). "
        "Press s for one inference cycle (pauses after it), c to resume "
        "continuous inference, i to reset/start the next episode, or q/Ctrl-C "
        "to reset and exit.",
        flush=True,
    )
    try:
        with TerminalKeys() as keys:
            cycles = runtime.run(
                args.duration,
                read_key=keys.read_key,
                single_step=args.single_step,
            )
    except KeyboardInterrupt:
        # Covers interrupts raised before NeroInferenceRuntime.run() takes
        # ownership. Interrupts during the control loop are reset there.
        print("\nCtrl-C received before inference start; shutting down.", flush=True)
        return
    print(f"Inference stopped after {cycles} control cycles.", flush=True)


if __name__ == "__main__":
    main()
