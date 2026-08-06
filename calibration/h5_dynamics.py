from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.dynamics_common import DOF, DynamicsDataset, save_dynamics_dataset


_REQUIRED_DATASETS = {
    "timestamp_us": "teleop/timestamp_us",
    "q": "teleop/q_follower",
    "q_cmd": "teleop/q_cmd",
    "tau": "teleop/tau_follower",
}


def convert_teleop_h5_to_dynamics_npz(
    h5_path: str | Path,
    output_path: str | Path,
) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 import requires h5py") from exc

    source = Path(h5_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"teleop HDF5 does not exist: {source}")

    with h5py.File(source, "r") as h5:
        missing = [path for path in _REQUIRED_DATASETS.values() if path not in h5]
        if missing:
            raise ValueError(f"teleop HDF5 is missing required datasets: {missing}")

        values = {
            name: np.asarray(h5[path]) for name, path in _REQUIRED_DATASETS.items()
        }
        current = (
            np.asarray(h5["teleop/current_follower"], dtype=np.float64)
            if "teleop/current_follower" in h5
            else np.full_like(values["q"], np.nan, dtype=np.float64)
        )
        h5_format = str(h5.attrs.get("format", ""))
        saved_at_us = int(h5.attrs.get("saved_at_us", 0))

    timestamp_us = _timestamp_vector(values["timestamp_us"], "teleop/timestamp_us")
    count = timestamp_us.size
    q = _joint_matrix(values["q"], count, "teleop/q_follower")
    q_cmd = _joint_matrix(values["q_cmd"], count, "teleop/q_cmd")
    tau = _joint_matrix(values["tau"], count, "teleop/tau_follower")
    current = _joint_matrix(current, count, "teleop/current_follower", finite=False)
    joint_timeline = np.repeat(timestamp_us[:, None], DOF, axis=1)

    dataset = DynamicsDataset(
        timestamp_us=timestamp_us,
        q=q,
        q_cmd=q_cmd,
        tau=tau,
        current=current,
        motor_timestamp_us=joint_timeline.copy(),
        motor_acquired_timestamp_us=joint_timeline.copy(),
        q_can_timestamp_us=timestamp_us.copy(),
        q_acquired_timestamp_us=timestamp_us.copy(),
        trajectory_id=np.zeros(count, dtype=np.int32),
        metadata={
            "format_version": 1,
            "source_type": "teleop_h5",
            "source_h5": str(source),
            "source_h5_format": h5_format,
            "source_saved_at_us": saved_at_us,
            "joint_names": [f"joint{index}" for index in range(1, DOF + 1)],
            "position_unit": "rad",
            "torque_unit": "N.m",
            "contact_assumption": "operator_marked_contact_free_episode",
            "conversion": "clean aligned follower q/tau on teleop timeline",
        },
    )
    return save_dynamics_dataset(output_path, dataset)


def conversion_manifest_entry(npz_path: Path) -> dict[str, object]:
    with np.load(npz_path, allow_pickle=False) as values:
        metadata = json.loads(str(np.asarray(values["metadata_json"]).item()))
        return {
            "npz": str(npz_path.resolve()),
            "source_h5": metadata["source_h5"],
            "samples": int(np.asarray(values["timestamp_us"]).size),
        }


def _timestamp_vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size < 2 or np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must contain at least two positive, increasing timestamps")
    return result


def _joint_matrix(value, count: int, name: str, *, finite: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, DOF):
        raise ValueError(f"{name} must have shape ({count}, {DOF}); got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if not finite and np.isinf(result).any():
        raise ValueError(f"{name} may contain NaN but not infinity")
    return result


def _joint_timestamp_matrix(
    value,
    count: int,
    name: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != (count, DOF):
        raise ValueError(f"{name} must have shape ({count}, {DOF}); got {result.shape}")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must contain positive timestamps")
    return result


def _arm_timestamp_vector(
    value,
    count: int,
    name: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape == (count, 1):
        result = result[:, 0]
    elif result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},) or ({count}, 1); got {result.shape}")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must contain positive timestamps")
    return result
