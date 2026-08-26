"""Adapters that bridge legacy Nero objects to the core contracts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from inference.core.contracts import Observation


def observation_from_state_sample(
    sample: Any,
    images: Mapping[str, np.ndarray] | None = None,
    image_timestamps_us: Mapping[str, int] | None = None,
    *,
    q_cmd: np.ndarray | None = None,
) -> Observation:
    """Convert ``ContinuousInferenceSample`` into the stable observation type."""

    if q_cmd is None:
        q_cmd = getattr(sample, "q_cmd", None)

    tau_result = getattr(sample, "tau_result", None)
    tau_ext = (
        getattr(tau_result, "tau_ext_cal", None)
        if tau_result is not None
        else getattr(sample, "tau_ext", None)
    )
    if tau_ext is None:
        raise ValueError("state sample does not contain tau_ext_cal")
    metadata = {}
    if tau_result is not None:
        metadata["tau_result"] = tau_result
    return Observation(
        timestamp_us=int(sample.timestamp_us),
        acquired_timestamp_us=int(
            getattr(sample, "acquired_timestamp_us", sample.timestamp_us)
        ),
        q=sample.q,
        dq=sample.dq,
        ddq=sample.ddq,
        tau=sample.tau,
        tau_ext=tau_ext,
        wrench_ext=getattr(sample, "processed_wrench", sample.wrench),
        images={} if images is None else images,
        image_timestamps_us={} if image_timestamps_us is None else image_timestamps_us,
        q_cmd=q_cmd,
        metadata=metadata,
    )
