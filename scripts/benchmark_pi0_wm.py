"""Offline CaRS-WM latency and numerical equivalence benchmark.

The benchmark uses one checkpoint, one generated condition, one explicit flow
source for every mode, and a fixed dtype.  It never opens cameras, a robot, or
the π0 client.  Compile and warmup calls are excluded from the reported
steady-state samples; their duration is reported separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from inference.pi0_wm.config import load_config
from inference.pi0_wm.wm import WMAdapter


LOG = logging.getLogger("benchmark_pi0_wm")


def _percentile(values, q):
    return None if not values else float(np.percentile(np.asarray(values), q))


def _summary(values):
    return {
        "count": len(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "max_ms": None if not values else float(max(values)),
    }


def _mode_config(wm, mode, steps, solver):
    result = copy.deepcopy(wm)
    result["flow_steps"] = steps
    result["solver"] = solver
    result["acceleration"] = {
        "enabled": mode in ("kv", "compile", "kv_compile"),
        "cache_condition_kv": mode in ("kv", "kv_compile"),
        "compile": mode in ("compile", "kv_compile"),
        "compile_mode": wm.get("acceleration", {}).get("compile_mode", "reduce-overhead"),
    }
    return result


def _error(reference, value):
    result = {}
    for key in ("q", "tau", "contact_phase"):
        if key in reference and key in value:
            result[key] = float(np.max(np.abs(reference[key].astype(np.float64) - value[key].astype(np.float64))))
    return result


def _run_scenario(wm_config, config, modes, *, scenario, steps, solver,
                  iterations, warmup, seed):
    rng = np.random.default_rng(seed)
    adapters = {}
    payloads = {}
    source_noises = {}
    results = {}
    baseline = None
    for mode in modes:
        LOG.info("loading scenario=%s mode=%s steps=%s solver=%s", scenario, mode, steps, solver)
        adapter = WMAdapter(
            _mode_config(wm_config, mode, steps, solver),
            config["control"]["hz"], config["pi0"]["action_hz"],
        )
        history = {
            key: rng.standard_normal((adapter.history_horizon, 7), dtype=np.float32)
            for key in adapter.inputs
        }
        action = rng.standard_normal((adapter.action_horizon, 7), dtype=np.float32)
        # The source is in the model's internal (possibly strided) horizon.
        source = rng.standard_normal(
            (1, adapter.num_samples, adapter.model.future_horizon, adapter.model.flow_dim),
            dtype=np.float32,
        )
        adapters[mode] = adapter
        payloads[mode] = (history, action)
        source_noises[mode] = source

    # Reuse byte-identical conditions and source across modes.
    first_mode = modes[0]
    first_history, first_action = payloads[first_mode]
    first_source = source_noises[first_mode]
    for mode, adapter in adapters.items():
        payloads[mode] = ({key: value.copy() for key, value in first_history.items()}, first_action.copy())
        source_noises[mode] = first_source.copy()
        if (adapter.model.future_horizon != adapters[first_mode].model.future_horizon
                or adapter.model.flow_dim != adapters[first_mode].model.flow_dim):
            raise RuntimeError("modes do not have the same checkpoint shape")

    for mode in modes:
        adapter = adapters[mode]
        LOG.info("warming scenario=%s mode=%s calls=%s", scenario, mode, warmup + 1)
        for _ in range(warmup + 1):
            adapter.infer(payloads[mode], source_noise=source_noises[mode])
        if adapter.device.type == "cuda":
            torch.cuda.synchronize(adapter.device)
            torch.cuda.reset_peak_memory_stats(adapter.device)
        e2e, gpu = [], []
        first_output = None
        for _ in range(iterations):
            started = time.perf_counter()
            output = adapter.infer(payloads[mode], source_noise=source_noises[mode])
            timing = adapter._last_timing or {}
            e2e.append(float(timing.get("adapter_seconds", time.perf_counter() - started)) * 1000)
            if timing.get("gpu_seconds") is not None:
                gpu.append(float(timing["gpu_seconds"]) * 1000)
            if first_output is None:
                first_output = output
        memory = None
        if adapter.device.type == "cuda":
            memory = int(torch.cuda.max_memory_allocated(adapter.device))
        if baseline is None:
            baseline = first_output
        results[mode] = {
            "steps": steps,
            "solver": solver,
            "adapter": _summary(e2e),
            "gpu": _summary(gpu),
            "compile_warmup_ms": None if adapter._compile_warmup_seconds is None else adapter._compile_warmup_seconds * 1000,
            "max_memory_allocated_bytes": memory,
            "max_abs_error_vs_first_mode": _error(baseline, first_output),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--modes", nargs="+", default=["eager", "kv", "compile", "kv_compile"],
                        choices=["eager", "kv", "compile", "kv_compile"])
    parser.add_argument("--scenarios", nargs="+", default=["current", "heun64"],
                        choices=["current", "heun64"])
    parser.add_argument("--checkpoint", type=Path, help="override wm.checkpoint")
    parser.add_argument("--device", help="override wm.device (for example cpu for a CUDA-free check)")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup must be non-negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    wm_config = copy.deepcopy(config["wm"])
    if args.checkpoint is not None:
        wm_config["checkpoint"] = str(args.checkpoint.resolve())
    if args.device is not None:
        wm_config["device"] = args.device
    if not Path(wm_config["checkpoint"]).exists():
        raise SystemExit(
            "checkpoint not found; provide --checkpoint or configure wm.checkpoint. "
            "No mock timing is reported."
        )
    device_name = str(wm_config.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA device {device_name} requested but CUDA is unavailable; "
            "GPU latency and memory are unverified. No mock timing is reported."
        )

    all_scenarios = {
        "current": (wm_config.get("flow_steps"), wm_config.get("solver")),
        "heun64": (64, "heun"),
    }
    scenarios = {name: all_scenarios[name] for name in args.scenarios}
    results = {
        name: _run_scenario(
            wm_config, config, args.modes, scenario=name, steps=steps, solver=solver,
            iterations=args.iterations, warmup=args.warmup, seed=args.seed,
        )
        for name, (steps, solver) in scenarios.items()
    }

    print(json.dumps({"config": str(args.config.resolve()), "seed": args.seed, "results": results}, indent=2))


if __name__ == "__main__":
    main()
