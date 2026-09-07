#!/usr/bin/env python3
"""Replay LeRobot v3 state/action labels through CARS-WM only.

This command intentionally does not construct the real-robot runtime, DP, CAN,
or a MuJoCo dynamics backend. The displayed robot follows the first step of
the first sampled WM future; all sampled futures remain visible as FK markers.
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
    required = {"observation.joint", "observation.velocity", "observation.torque", action_key, "timestamp"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"dataset is missing required WM fields: {missing}")
    def col(name, width=None):
        value = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if width is not None:
            value = value.reshape(-1, width)
        return value
    delta_name = "observation.delta_q" if "observation.delta_q" in names else None
    tau_name = "observation.tau_f" if "observation.tau_f" in names else "observation.torque"
    return {
        "q": col("observation.joint", 7),
        "dq": col("observation.velocity", 7),
        "tau": col(tau_name, 7),
        "delta_q": col(delta_name, 7) if delta_name else np.zeros_like(col("observation.joint", 7)),
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


def sample(model, arrays, start, samples, noise_bank=None, steps=None, solver=None):
    import torch
    horizon = int(model.history_horizon)
    future = int(model.future_horizon)
    action_horizon = int(model.action_condition_horizon)
    device = next(model.parameters()).device
    inputs = {}
    for key in model.inputs:
        values = torch.as_tensor(arrays[key][start - horizon + 1:start + 1], dtype=torch.float32, device=device)[None]
        inputs[key] = _normalize(model, key, values)
    actions = torch.as_tensor(arrays["action"][start:start + action_horizon], dtype=torch.float32, device=device)[None]
    inputs["action"] = _normalize(model, "action", actions)
    inputs["action_mask"] = torch.ones((1, action_horizon), dtype=torch.bool, device=device)
    encode_started = time.perf_counter()
    encoded = model.encode_conditions(inputs)
    encode_ms = (time.perf_counter() - encode_started) * 1e3
    encoded = {key: value.repeat_interleave(samples, dim=0) if torch.is_tensor(value) and value.shape[0] == 1 else value for key, value in encoded.items()}
    shape = (samples, future, int(model.flow_dim))
    if noise_bank is None:
        noise_bank = torch.randn(shape, device=device)
    flow_started = time.perf_counter()
    generated = model.integrate_flow(noise_bank, encoded, steps=steps, solver=solver)
    flow_ms = (time.perf_counter() - flow_started) * 1e3
    decoded = model._decoded_output(generated, encoded)
    q = decoded["q_pred"].reshape(samples, future, 7)
    normalizer = getattr(model, "_inference_normalizer_obj", None)
    metadata = getattr(model, "_inference_normalizer", None) or {}
    if normalizer is not None and "q" in (metadata.get("normalize_lowdim_keys") or ()):
        mode = metadata.get("normalize_mode", "gaussian")
        q = getattr(normalizer, f"{mode}_denormalize")("q", q)
    return q.detach().cpu().numpy(), noise_bank, {"condition_encoding_ms": encode_ms, "flow_integration_N_ms": flow_ms}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Offline real-time kinematic visualization of sampled futures from LeRobot v3")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--action-key", default="action.ee_pose")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--headless", action="store_true", help="run FK without opening a MuJoCo window")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
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
    history = int(model.history_horizon)
    samples = int(viz_cfg.num_future_samples)
    noise_bank = None
    timings = []
    stage_timings = []
    fk_timings = []
    try:
        for index in range(history - 1, min(arrays["q"].shape[0] - int(model.action_condition_horizon), history - 1 + args.max_steps)):
            started = time.perf_counter()
            q, noise_bank, stage = sample(model, arrays, index, samples, noise_bank if viz_cfg.use_fixed_noise_bank else None, steps=viz_cfg.flow_steps, solver=viz_cfg.flow_solver)
            timings.append((time.perf_counter() - started) * 1e3)
            stage_timings.append(stage)
            fk_started = time.perf_counter()
            ee_positions = fk.predicted_ee_positions(q)
            fk_timings.append((time.perf_counter() - fk_started) * 1e3)
            viz.publish_prediction(
                float(arrays["timestamp"][index]), arrays["q"][index], q,
                prediction_id=index,
                predicted_ee_position=ee_positions,
                playback_q=q[0, 0],
            )
    finally:
        viz.close()
    print(json.dumps({"samples_shape": list(q.shape), "ee_position_shape": list(ee_positions.shape), "future_horizon": int(model.future_horizon), "display_q_source": "wm_sample_0_step_0", "batch_sampling_ms_mean": float(np.mean(timings)) if timings else None, "condition_encoding_ms": float(np.mean([item["condition_encoding_ms"] for item in stage_timings])) if stage_timings else None, "flow_integration_N_ms": float(np.mean([item["flow_integration_N_ms"] for item in stage_timings])) if stage_timings else None, "fk_NxH_ms": float(np.mean(fk_timings)) if fk_timings else None, "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
