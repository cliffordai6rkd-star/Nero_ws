#!/usr/bin/env python3
"""Replay LeRobot v3 state/action labels through CARS-WM only.

This command intentionally does not construct the real-robot runtime, DP, CAN,
or a MuJoCo dynamics backend.  New checkpoints can run a strict feedback-free
rollout for visualization: every WM query predicts 32 frames, only a committed
prefix is retained, and the four predicted streams (q/dq/delta_q/tau) become
the next 50-frame condition.  Legacy q/tau checkpoints use the original
independent-window sampler because they do not emit dq and delta_q.  The
MuJoCo process only performs FK and rendering.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
import sys

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        raise SystemExit(
            "visualize_wm_lerobotv3.py requires NumPy, but the selected Python "
            "environment does not provide it. Activate nero_ws/.venv after "
            "installing the inference extra, or run with the sibling PINN "
            "environment: ../PINN/.conda-env/bin/python scripts/visualize_wm_lerobotv3.py ..."
        ) from exc
    raise
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.checkpoints import restore_checkpoint_model
from inference.config import load_inference_config
from inference.mujoco_visualization import MujocoKinematicFK, MujocoKinematicVisualizer
from inference.async_fast_slow import StateHistorySnapshot, ActionTrajectory


def _load_parquet(source: Path, action_key: str) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("LeRobot v3 replay requires pyarrow") from exc
    files = sorted(source.glob("data/chunk-*/file-*.parquet")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"no LeRobot v3 parquet files found under {source}")
    tables = [pq.read_table(path, columns=None) for path in files]
    table = __import__("pyarrow").concat_tables(tables, promote_options="default")
    names = set(table.column_names)
    required = {
        "observation.joint",
        "observation.velocity",
        "observation.delta_q",
        "observation.torque",
        action_key,
        "timestamp",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"dataset is missing required WM fields: {missing}")
    def col(name, width=None):
        value = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if width is not None:
            value = value.reshape(-1, width)
        return value
    tau_name = "observation.tau_f" if "observation.tau_f" in names else "observation.torque"
    return {
        "q": col("observation.joint", 7),
        "dq": col("observation.velocity", 7),
        "tau": col(tau_name, 7),
        "delta_q": col("observation.delta_q", 7),
        "action": col(action_key, 7),
        "timestamp": col("timestamp").reshape(-1),
    }


def _normalize(model, key, value):
    metadata = getattr(model, "_inference_normalizer", None)
    config = getattr(model, "_inference_checkpoint_config", {})
    keys = set((metadata or {}).get("normalize_lowdim_keys") or (config.get("dataloader", {}) if isinstance(config, dict) else {}).get("normalize_lowdim_keys") or ())
    if key not in keys:
        return value
    mode = (metadata or {}).get("normalize_mode") or config.get("dataloader", {}).get("normalize_mode")
    normalizer = getattr(model, "_inference_normalizer_obj", None)
    if normalizer is not None:
        return getattr(normalizer, f"{mode}_normalize")(key, value)
    stats = metadata["stats"][key]
    eps = float(metadata.get("eps", 1e-6))
    if mode == "gaussian":
        return (value - _tensor(stats["mean"], value)) / (_tensor(stats["std"], value) + eps)
    if mode == "limit":
        return 2 * (value - _tensor(stats["min"], value)) / (_tensor(stats["max"], value) - _tensor(stats["min"], value) + eps) - 1
    if mode == "quantile":
        result = 2 * (value - _tensor(stats["q01"], value)) / (_tensor(stats["q99"], value) - _tensor(stats["q01"], value) + eps) - 1
        return result.clamp(-1, 1)
    raise ValueError(f"unsupported normalization mode {mode!r}")


def _tensor(value, reference):
    import torch
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _pad_action(values: np.ndarray, horizon: int) -> np.ndarray:
    """Return exactly ``horizon`` action rows using causal tail hold."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 1:
        raise ValueError(f"action condition must have shape [T,7], got {values.shape}")
    result = np.repeat(values[-1:, :], int(horizon), axis=0)
    result[: min(values.shape[0], int(horizon))] = values[: int(horizon)]
    return result


def _history_window(arrays: dict[str, np.ndarray], start: int, horizon: int) -> dict[str, np.ndarray]:
    result = {}
    begin = int(start) - int(horizon) + 1
    end = int(start) + 1
    for key in ("q", "dq", "delta_q", "tau"):
        value = np.asarray(arrays[key][begin:end], dtype=np.float64)
        if value.shape != (horizon, 7):
            raise ValueError(f"{key} history must have shape [{horizon},7], got {value.shape}")
        result[key] = value
    return result


def _sample_history(
    model,
    history,
    action_values,
    samples,
    noise_bank=None,
    steps=None,
    solver=None,
    *,
    required_outputs=("q", "dq", "delta_q", "tau"),
    action_time=None,
    future_time=None,
):
    import torch
    future = int(model.future_horizon)
    action_horizon = int(model.action_condition_horizon)
    stride = int(getattr(model, "temporal_stride", 1))
    external_history = int(getattr(model, "external_history_horizon", int(model.history_horizon) * stride))
    device = next(model.parameters()).device
    count = int(samples)
    if count < 1:
        raise ValueError("samples must be positive")
    inputs = {}
    for key in model.inputs:
        values = np.asarray(history[key], dtype=np.float64)
        if stride > 1:
            if values.ndim == 2 and values.shape[0] == external_history:
                values = values[::stride]
            elif values.ndim == 3 and values.shape[1] == external_history:
                values = values[:, ::stride]
        if values.ndim == 2:
            values = np.repeat(values[None], count, axis=0)
        if values.shape != (count, int(model.history_horizon), 7):
            raise ValueError(
                f"{key} history must have shape [{count},{int(model.history_horizon)},7], got {values.shape}"
            )
        values = torch.as_tensor(values, dtype=torch.float32, device=device)
        inputs[key] = _normalize(model, key, values)
    actions_np = _pad_action(action_values, action_horizon)
    actions_np = np.repeat(actions_np[None], count, axis=0)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    inputs["action"] = _normalize(model, "action", actions)
    inputs["action_mask"] = torch.ones((count, action_horizon), dtype=torch.bool, device=device)
    if action_time is not None:
        at = torch.as_tensor(np.asarray(action_time, dtype=np.float32), device=device)
        inputs["action_time"] = at.reshape(1, action_horizon).expand(count, -1)
    if future_time is not None:
        ft = torch.as_tensor(np.asarray(future_time, dtype=np.float32), device=device)
        inputs["future_time"] = ft.reshape(1, future).expand(count, -1)
    with torch.inference_mode():
        encode_started = time.perf_counter()
        encoded = model.encode_conditions(inputs)
        encode_ms = (time.perf_counter() - encode_started) * 1e3
        # Some checkpoints include scalar (0-D) condition metadata alongside
        # batched tensors.  Accessing ``value.shape[0]`` on those scalars
        # raises ``IndexError``; only repeat tensors that actually have a
        # batch dimension.
        encoded = {
            key: value.repeat_interleave(count, dim=0)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1
            else value
            for key, value in encoded.items()
        }
        shape = (count, future, int(model.flow_dim))
        if noise_bank is None:
            noise_bank = torch.randn(shape, device=device)
        else:
            noise_bank = torch.as_tensor(noise_bank, dtype=torch.float32, device=device)
            if tuple(noise_bank.shape) != shape:
                raise ValueError(f"noise_bank must have shape {shape}, got {tuple(noise_bank.shape)}")
        flow_started = time.perf_counter()
        generated = model.integrate_flow(noise_bank, encoded, steps=steps, solver=solver)
        flow_ms = (time.perf_counter() - flow_started) * 1e3
        decoded = model._decoded_output(generated, encoded)
    normalizer = getattr(model, "_inference_normalizer_obj", None)
    metadata = getattr(model, "_inference_normalizer", None) or {}
    output = {}
    missing = []
    for key in ("q", "dq", "delta_q", "tau"):
        value = decoded.get(f"{key}_pred")
        if value is None:
            missing.append(key)
            continue
        value = value.reshape(count, future, 7)
        if normalizer is not None and key in (metadata.get("normalize_lowdim_keys") or ()):
            mode = metadata.get("normalize_mode", "gaussian")
            value = getattr(normalizer, f"{mode}_denormalize")(key, value)
        value = value.detach().cpu().numpy().astype(np.float64)
        if stride > 1:
            value = np.repeat(value, stride, axis=1)
        output[key] = value
    required_outputs = tuple(required_outputs)
    missing_required = [key for key in required_outputs if key in missing]
    if missing_required:
        raise RuntimeError(
            "checkpoint does not provide the outputs required by this rollout: "
            f"{missing_required}"
        )
    return output, noise_bank.detach().cpu(), {
        "condition_encoding_ms": encode_ms,
        "flow_integration_N_ms": flow_ms,
    }


def recursive_rollout(
    model,
    arrays: dict[str, np.ndarray],
    start: int,
    samples: int,
    *,
    segment_steps: int = 16,
    segments: int = 4,
    noise_banks: list[np.ndarray | None] | None = None,
    steps: int | None = None,
    solver: str | None = None,
):
    """Run 34+16 strict recursive segments from one recorded anchor.

    Every segment samples the full checkpoint horizon, commits only its first
    ``segment_steps`` rows, and reconditions all four predicted state streams.
    The action condition is re-anchored to the corresponding recorded action
    rows; its tail is held when the episode has no more action rows.
    """

    stride = int(getattr(model, "temporal_stride", 1))
    history_horizon = int(getattr(model, "external_history_horizon", int(model.history_horizon) * stride))
    future_horizon = int(getattr(model, "external_future_horizon", int(model.future_horizon) * stride))
    segment_steps = int(segment_steps)
    segments = int(segments)
    if segment_steps < 1 or segment_steps > future_horizon:
        raise ValueError("segment_steps must be in [1, model.future_horizon]")
    if segments < 1:
        raise ValueError("segments must be positive")
    required_outputs = {"q", "dq", "delta_q", "tau"}
    declared_outputs = getattr(model, "outputs", None)
    available_outputs = required_outputs if declared_outputs is None else set(declared_outputs or ())
    if not required_outputs.issubset(available_outputs):
        missing = sorted(required_outputs - available_outputs)
        raise RuntimeError(
            "strict recursive rollout requires checkpoint outputs for "
            f"q/dq/delta_q/tau; missing {missing}. "
            "Use the legacy q/tau visualization path for this checkpoint."
        )
    if int(start) < history_horizon - 1:
        raise ValueError("start does not have a complete model history")
    history = {
        key: np.repeat(_history_window(arrays, start, history_horizon)[key][None], int(samples), axis=0)
        for key in ("q", "dq", "delta_q", "tau")
    }
    if noise_banks is None:
        noise_banks = [None] * segments
    if len(noise_banks) != segments:
        raise ValueError("noise_banks length must equal segments")
    trajectory_parts = []
    timings = []
    for segment_index in range(segments):
        action_start = int(start) + segment_index * segment_steps
        action = (
            _action_condition(model, arrays["action"], action_start)
            if stride > 1
            else arrays["action"][action_start : action_start + int(model.action_condition_horizon)]
        )
        prediction, noise_bank, timing = _sample_history(
            model,
            history,
            action,
            samples,
            noise_bank=noise_banks[segment_index],
            steps=steps,
            solver=solver,
        )
        noise_banks[segment_index] = noise_bank.numpy()
        committed = {key: prediction[key][:, :segment_steps, :] for key in ("q", "dq", "delta_q", "tau")}
        trajectory_parts.append(committed["q"])
        timings.append(timing)
        for key in ("q", "dq", "delta_q", "tau"):
            history[key] = np.concatenate((history[key][:, segment_steps:, :], committed[key]), axis=1)
    return np.concatenate(trajectory_parts, axis=1), noise_banks, timings


def sample(model, arrays, start, samples, noise_bank=None, steps=None, solver=None):
    """Sample one WM window from the recorded external-rate history."""

    stride = int(getattr(model, "temporal_stride", 1))
    history_horizon = int(getattr(model, "external_history_horizon", int(model.history_horizon) * stride))
    history = _history_window(arrays, start, history_horizon)
    action = _action_condition(model, arrays["action"], start)
    action_time, future_time = _relative_times(model, arrays["timestamp"], start)
    prediction, noise_bank, timing = _sample_history(
        model,
        history,
        action,
        samples,
        noise_bank=noise_bank,
        steps=steps,
        solver=solver,
        required_outputs=("q",),
        action_time=action_time, future_time=future_time,
    )
    return prediction["q"], noise_bank, timing


def _action_condition(model, actions: np.ndarray, start: int) -> np.ndarray:
    """Match PINN expert-rate action tokens from a high-rate data stream."""
    stride = int(getattr(model, "temporal_stride", 1))
    horizon = int(getattr(model, "external_action_condition_horizon", int(model.action_condition_horizon) * stride))
    # PINN training starts the action chunk one expert token after the state
    # anchor (action_start_offset=1), so preserve that causal offset here.
    offset = int(getattr(model, "action_start_offset", 1))
    indices = int(start) + offset + np.arange(horizon, dtype=np.int64)
    indices = np.clip(indices, 0, len(actions) - 1)
    values = np.asarray(actions[indices], dtype=np.float64)
    return values[::stride] if stride > 1 else values


def _relative_times(model, timestamps: np.ndarray, start: int):
    """Build the same timestamp-relative grids used by PINN training."""
    anchor = float(timestamps[int(start)])
    offset = int(getattr(model, "action_start_offset", 1))
    stride = int(getattr(model, "temporal_stride", 1))
    action_horizon = int(getattr(model, "external_action_condition_horizon", int(model.action_condition_horizon) * stride))
    future_horizon = int(getattr(model, "external_future_horizon", int(model.future_horizon) * stride))
    ai = int(start) + offset + np.arange(action_horizon)
    fi = int(start) + 1 + np.arange(future_horizon)
    ai = np.clip(ai, 0, len(timestamps) - 1)
    fi = np.clip(fi, 0, len(timestamps) - 1)
    return ((timestamps[ai] - anchor) * 1e-9)[::stride], ((timestamps[fi] - anchor) * 1e-9)[::stride]


def _execution_step_counts(model, requested_steps: int, available_external: int) -> tuple[int, int]:
    """Convert internal WM execution tokens to external-rate replay frames."""
    stride = int(getattr(model, "temporal_stride", 1))
    if stride < 1:
        raise ValueError(f"model temporal_stride must be positive, got {stride}")
    internal = min(int(requested_steps), int(getattr(model, "future_horizon")))
    external = min(internal * stride, int(available_external))
    return internal, external


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Offline real-time kinematic visualization of sampled futures from LeRobot v3")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--action-key", default="action.ee_pose")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--rollout-segment-steps",
        type=int,
        default=None,
        help="number of predicted high-rate states committed per recursive WM query",
    )
    parser.add_argument(
        "--rollout-segments",
        type=int,
        default=None,
        help="number of strict recursive WM segments to visualize",
    )
    parser.add_argument("--visualization-horizon", type=int, default=None, help="number of predicted trajectory steps to draw (default: full WM horizon)")
    parser.add_argument("--headless", action="store_true", help="run FK without opening a MuJoCo window")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    replay_cfg = raw.get("mujoco_visualization", {}) or {}
    if args.rollout_segment_steps is None:
        args.rollout_segment_steps = int(replay_cfg.get("execute_steps", 16))
    if args.rollout_segments is None:
        args.rollout_segments = 1
    if args.visualization_horizon is None and replay_cfg.get("trajectory_horizon") is not None:
        args.visualization_horizon = int(replay_cfg["trajectory_horizon"])
    if args.rollout_segment_steps < 1 or args.rollout_segments < 1:
        parser.error("rollout segment steps and segments must be positive")
    if args.visualization_horizon is not None and args.visualization_horizon < 1:
        parser.error("--visualization-horizon must be positive")
    contact = raw.get("contactworldmodel") or raw.get("pinn_checkpoint")
    if not contact:
        raise ValueError("config must declare contactworldmodel/pinn_checkpoint")
    ckpt = Path(contact["path"])
    if not ckpt.is_absolute():
        ckpt = (args.config.parent / ckpt).resolve()
    model = restore_checkpoint_model(ckpt, str(contact.get("device", "cpu")), use_ema=bool(contact.get("use_ema", True)), kind="PINN", pinn_mode="contact_world_model")
    viz_cfg = load_inference_config(args.config).mujoco_visualization
    # The offline command is explicitly a visualization entry point; keep the
    # online default disabled while enabling this independent replay path.
    viz_cfg = replace(viz_cfg, enabled=True, headless=bool(args.headless))
    viz = MujocoKinematicVisualizer(viz_cfg)
    viz.start()
    fk = MujocoKinematicFK(viz_cfg)
    arrays = _load_parquet(args.episode, args.action_key)
    temporal_stride = int(getattr(model, "temporal_stride", 1))
    history = int(getattr(model, "external_history_horizon", int(model.history_horizon) * temporal_stride))
    samples = int(viz_cfg.num_future_samples)
    if args.rollout_segment_steps > int(model.future_horizon):
        parser.error(
            "--rollout-segment-steps cannot exceed the checkpoint future horizon "
            f"({int(model.future_horizon)} internal tokens)"
        )
    noise_bank = None
    timings = []
    stage_timings = []
    fk_timings = []
    q = None
    ee_positions = None
    previous_playback_last_q = None
    try:
        external_action_horizon = int(
            getattr(model, "external_action_condition_horizon", int(model.action_condition_horizon) * temporal_stride)
        )
        last_start = arrays["q"].shape[0] - external_action_horizon
        index = history - 1
        while index < min(last_start, history - 1 + args.max_steps):
            started = time.perf_counter()
            q, noise_bank, stage = sample(
                model,
                arrays,
                index,
                samples,
                noise_bank=(noise_bank if viz_cfg.use_fixed_noise_bank else None),
                steps=viz_cfg.flow_steps,
                solver=viz_cfg.flow_solver,
            )
            stages = [stage]
            timings.append((time.perf_counter() - started) * 1e3)
            stage_timings.extend(stages)
            # Limit drawn trajectory independently from rollout/execution horizon.
            draw_q = q if args.visualization_horizon is None else q[:, :args.visualization_horizon, :]
            fk_started = time.perf_counter()
            ee_positions = fk.predicted_ee_positions(draw_q)
            fk_timings.append((time.perf_counter() - fk_started) * 1e3)
            internal_execute_count, execute_count = _execution_step_counts(
                model, args.rollout_segment_steps, q.shape[1]
            )
            current_first_q = np.asarray(q[0, 0], dtype=np.float64)
            current_last_q = np.asarray(q[0, execute_count - 1], dtype=np.float64)
            boundary_jump = (
                None
                if previous_playback_last_q is None
                else float(np.linalg.norm(current_first_q - previous_playback_last_q))
            )
            logging.debug(
                "WM window index=%d execute_internal=%d execute_external=%d physical_ms=%.1f boundary_jump_rad=%s",
                index,
                internal_execute_count,
                execute_count,
                1000.0 * execute_count / float(viz_cfg.render_fps),
                "n/a" if boundary_jump is None else f"{boundary_jump:.6f}",
            )
            # Play the committed WM prefix at the configured data rate before
            # taking the next recorded observation for a fresh prediction.
            period = 1.0 / float(viz_cfg.render_fps)
            # Refresh the visualizer's observed-state cache once per WM loop.
            # publish_prediction intentionally uses that cache (rather than
            # its positional observed_q argument) so online delayed packets
            # cannot overwrite a newer observation.
            viz.publish_observed(
                float(arrays["timestamp"][index]), arrays["q"][index]
            )
            for step in range(execute_count):
                viz.publish_prediction(
                    float(arrays["timestamp"][index]) + step * period,
                    arrays["q"][index], draw_q,
                    prediction_id=index * 10000 + step,
                    predicted_ee_position=ee_positions,
                    playback_q=q[0, step],
                )
                time.sleep(period)
            previous_playback_last_q = current_last_q
            index += execute_count
    finally:
        viz.close()
    print(json.dumps({"samples_shape": None if q is None else list(q.shape), "ee_position_shape": None if ee_positions is None else list(ee_positions.shape), "future_horizon": int(model.future_horizon), "external_future_horizon": int(getattr(model, "external_future_horizon", q.shape[1] if q is not None else model.future_horizon)), "rollout_segment_steps": int(args.rollout_segment_steps), "rollout_segments": int(args.rollout_segments), "visualization_horizon": int(args.visualization_horizon or q.shape[1]) if q is not None else None, "rollout_mode": "independent_recorded_history", "display_q_source": "wm_sample_0_step_0", "batch_sampling_ms_mean": float(np.mean(timings)) if timings else None, "condition_encoding_ms": float(np.mean([item["condition_encoding_ms"] for item in stage_timings])) if stage_timings else None, "flow_integration_N_ms": float(np.mean([item["flow_integration_N_ms"] for item in stage_timings])) if stage_timings else None, "fk_NxH_ms": float(np.mean(fk_timings)) if fk_timings else None, "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
