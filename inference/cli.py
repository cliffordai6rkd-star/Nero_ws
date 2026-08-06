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
        description="Nero DP inference with optional predictor/OSC-QP control"
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="restore configured checkpoint-defined models and validate runtime configuration",
    )
    mode.add_argument("--run", action="store_true", help="start the online inference loop")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"), default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--enable-command",
        action="store_true",
        help="send torque or direct-IK joint commands; otherwise inference is read-only",
    )
    parser.add_argument("--skip-can-setup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
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
                    "pinn_checkpoint is required when predictor.enabled=true"
                )
            pinn = restore_checkpoint_model(
                config.pinn_checkpoint.path,
                config.pinn_checkpoint.device,
                use_ema=config.pinn_checkpoint.use_ema,
                kind="PINN",
                pinn_mode=config.predictor.mode,
            )
        if config.predictor.enabled and config.predictor.mode in {
            "world_model_v3",
            "world_model_v4",
            "world_model_v5",
        }:
            from inference.world_model import WorldModelWrenchAdapter

            WorldModelWrenchAdapter.from_collection_config(
                config.runtime.collection_config
            )
        print(
            json.dumps(
                {
                    "dp": type(dp).__qualname__,
                    "pinn": None if pinn is None else type(pinn).__qualname__,
                    "predictor_enabled": config.predictor.enabled,
                    "predictor_mode": config.predictor.mode,
                    "inference_mode": config.predictor.inference_mode,
                    "action_chunk_mode": config.predictor.action_chunk_mode,
                    "action_step_s": config.predictor.action_step_s,
                    "action_execution_mode": config.predictor.action_execution_mode,
                    "action_interpolation_duration_s": (
                        config.predictor.action_interpolation_duration_s
                    ),
                    "action_interpolation_steps": (
                        config.predictor.action_interpolation_steps
                    ),
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
        "Press i to reset/start the next episode; press q or Ctrl-C to "
        "reset and exit.",
        flush=True,
    )
    try:
        with TerminalKeys() as keys:
            cycles = runtime.run(args.duration, read_key=keys.read_key)
    except KeyboardInterrupt:
        # Covers interrupts raised before NeroInferenceRuntime.run() takes
        # ownership. Interrupts during the control loop are reset there.
        print("\nCtrl-C received before inference start; shutting down.", flush=True)
        return
    print(f"Inference stopped after {cycles} control cycles.", flush=True)


if __name__ == "__main__":
    main()
