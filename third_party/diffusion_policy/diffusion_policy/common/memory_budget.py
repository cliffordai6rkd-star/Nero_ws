import logging
import math
import os
import resource
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)


def gb_to_bytes(value_gb: Optional[float]) -> Optional[int]:
    if value_gb is None:
        return None
    return int(float(value_gb) * (1024 ** 3))


def bytes_to_gb(value_bytes: Optional[int]) -> Optional[float]:
    if value_bytes is None:
        return None
    return float(value_bytes) / (1024 ** 3)


def format_gb(value_bytes: Optional[int]) -> str:
    if value_bytes is None:
        return "unlimited"
    return f"{bytes_to_gb(value_bytes):.2f} GiB"


def get_system_memory_bytes() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        num_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return int(page_size * num_pages)


def compute_effective_budget_bytes(
    memory_limit_gb: Optional[float],
    memory_reserve_gb: float = 2.0,
) -> Optional[int]:
    limit_bytes = gb_to_bytes(memory_limit_gb)
    if limit_bytes is None:
        return None
    reserve_bytes = gb_to_bytes(memory_reserve_gb) or 0
    min_budget_bytes = 256 * 1024 ** 2
    return max(min_budget_bytes, limit_bytes - reserve_bytes)


def set_process_memory_limit(memory_limit_gb: Optional[float]) -> bool:
    limit_bytes = gb_to_bytes(memory_limit_gb)
    if limit_bytes is None:
        return False

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    new_soft_limit = limit_bytes
    if hard_limit != resource.RLIM_INFINITY:
        new_soft_limit = min(new_soft_limit, hard_limit)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft_limit, hard_limit))
    except (ValueError, OSError) as exc:
        log.warning("Failed to apply process RAM limit %s: %s", format_gb(limit_bytes), exc)
        return False

    log.info("Applied process RLIMIT_AS soft limit: %s", format_gb(new_soft_limit))
    return True


def estimate_array_nbytes(shape: Tuple[int, ...], dtype: Any) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def estimate_obs_sample_nbytes(
    shape_meta: Mapping[str, Any],
    n_obs_steps: int,
    action_horizon: int,
    float_dtype=np.float32,
    safety_multiplier: float = 1.35,
) -> int:
    total_bytes = 0
    obs_meta = shape_meta["obs"]
    float_itemsize = np.dtype(float_dtype).itemsize

    for _, attr in obs_meta.items():
        shape = tuple(attr["shape"])
        total_bytes += n_obs_steps * int(np.prod(shape)) * float_itemsize

    action_shape = tuple(shape_meta["action"]["shape"])
    total_bytes += action_horizon * int(np.prod(action_shape)) * float_itemsize

    return int(math.ceil(total_bytes * safety_multiplier))


def build_memory_limited_dataloader_kwargs(
    dataloader_cfg: Mapping[str, Any],
    shape_meta: Mapping[str, Any],
    n_obs_steps: int,
    action_horizon: int,
    memory_limit_gb: Optional[float],
    memory_reserve_gb: float = 2.0,
    loader_name: str = "train",
) -> Dict[str, Any]:
    kwargs = dict(dataloader_cfg)
    if memory_limit_gb is None:
        return kwargs

    effective_budget = compute_effective_budget_bytes(memory_limit_gb, memory_reserve_gb)
    if effective_budget is None:
        return kwargs

    batch_size = int(kwargs.get("batch_size", 1))
    requested_workers = int(kwargs.get("num_workers", 0) or 0)
    requested_prefetch = int(kwargs.get("prefetch_factor", 2) or 2)
    sample_nbytes = estimate_obs_sample_nbytes(shape_meta, n_obs_steps, action_horizon)
    batch_nbytes = sample_nbytes * batch_size

    # Keep DataLoader memory a bounded fraction of the configured process budget.
    loader_budget = max(batch_nbytes, int(effective_budget * 0.55))
    adjusted_prefetch = 1 if requested_workers > 0 else None
    max_workers = requested_workers
    if requested_workers > 0:
        denom = max(batch_nbytes * adjusted_prefetch, 1)
        max_workers = max(0, int((loader_budget - batch_nbytes) // denom))

    if requested_workers > max_workers:
        log.warning(
            "%s dataloader num_workers reduced from %d to %d to stay within RAM budget %s "
            "(estimated batch footprint %s).",
            loader_name,
            requested_workers,
            max_workers,
            format_gb(gb_to_bytes(memory_limit_gb)),
            format_gb(batch_nbytes),
        )
        kwargs["num_workers"] = max_workers

    num_workers = int(kwargs.get("num_workers", 0) or 0)

    # Pinned memory and persistent workers keep extra unswappable CPU pages alive.
    kwargs["pin_memory"] = False
    kwargs["persistent_workers"] = False

    if num_workers > 0:
        kwargs["prefetch_factor"] = adjusted_prefetch
    else:
        kwargs.pop("prefetch_factor", None)

    if batch_nbytes > effective_budget:
        log.warning(
            "%s dataloader single-batch estimate %s exceeds effective RAM budget %s. "
            "The training may still hit the cap; reduce batch_size, n_obs_steps, or image resolution.",
            loader_name,
            format_gb(batch_nbytes),
            format_gb(effective_budget),
        )

    log.info(
        "%s dataloader RAM plan: batch_size=%d, num_workers=%d, prefetch_factor=%s, "
        "est_batch=%s, effective_budget=%s",
        loader_name,
        batch_size,
        num_workers,
        kwargs.get("prefetch_factor", "n/a"),
        format_gb(batch_nbytes),
        format_gb(effective_budget),
    )
    return kwargs
