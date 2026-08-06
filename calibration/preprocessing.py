from __future__ import annotations

from dataclasses import replace

import numpy as np

from calibration.dynamics_common import (
    DOF,
    DynamicsDataset,
    DynamicsPlan,
    ProcessedDynamicsDataset,
    load_dynamics_dataset,
)
from nero_collection.dynamics_processing import (
    filter_torque as filter_dynamics_torque,
    reconstruct_state_from_positions as reconstruct_dynamics_state,
    resample_columns,
)


def preprocess_dataset(
    dataset: DynamicsDataset,
    plan: DynamicsPlan,
) -> ProcessedDynamicsDataset:
    pieces: list[ProcessedDynamicsDataset] = []
    for trajectory_id in np.unique(dataset.trajectory_id):
        indices = np.flatnonzero(dataset.trajectory_id == trajectory_id)
        if indices.size < plan.preprocess.min_samples:
            raise ValueError(
                f"trajectory {trajectory_id} has {indices.size} samples, below "
                f"preprocess.min_samples={plan.preprocess.min_samples}"
            )
        pieces.append(_preprocess_segment(dataset, indices, int(trajectory_id), plan))
    return concatenate_processed(pieces)


def preprocess_files(
    paths: list[str],
    plan: DynamicsPlan,
) -> ProcessedDynamicsDataset:
    pieces: list[ProcessedDynamicsDataset] = []
    next_trajectory_id = 0
    for path in paths:
        processed = preprocess_dataset(load_dynamics_dataset(path), plan)
        unique = np.unique(processed.trajectory_id)
        remap = {int(old): next_trajectory_id + index for index, old in enumerate(unique)}
        mapped = np.asarray([remap[int(value)] for value in processed.trajectory_id], dtype=np.int32)
        pieces.append(replace(processed, trajectory_id=mapped))
        next_trajectory_id += len(unique)
    if not pieces:
        raise ValueError("at least one dynamics dataset is required")
    return concatenate_processed(pieces)


def reconstruct_state_from_positions(
    timestamp_us: np.ndarray,
    q: np.ndarray,
    plan: DynamicsPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return reconstruct_dynamics_state(
        timestamp_us,
        q,
        state_method=plan.preprocess.state_method,
        spline_smoothing_rad2=plan.preprocess.spline_smoothing_rad2,
        fourier_fundamental_hz=plan.excitation.fundamental_hz,
        fourier_harmonics=plan.preprocess.fourier_harmonics,
    )


def split_train_validation(
    dataset: ProcessedDynamicsDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[ProcessedDynamicsDataset, ProcessedDynamicsDataset]:
    trajectory_ids = np.unique(dataset.trajectory_id)
    if trajectory_ids.size >= 2:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(trajectory_ids)
        validation_count = max(1, int(round(trajectory_ids.size * validation_fraction)))
        validation_count = min(validation_count, trajectory_ids.size - 1)
        validation_ids = set(int(value) for value in shuffled[:validation_count])
        validation_mask = np.asarray(
            [int(value) in validation_ids for value in dataset.trajectory_id], dtype=bool
        )
    else:
        # This fallback avoids adjacent-sample random leakage, but a separate
        # --validation-data trajectory is preferred for final model reporting.
        count = dataset.q.shape[0]
        validation_count = max(1, int(round(count * validation_fraction)))
        start = count - validation_count
        validation_mask = np.arange(count) >= start
    return _select(dataset, ~validation_mask), _select(dataset, validation_mask)


def concatenate_processed(
    datasets: list[ProcessedDynamicsDataset],
) -> ProcessedDynamicsDataset:
    if not datasets:
        raise ValueError("cannot concatenate an empty processed dataset list")
    return ProcessedDynamicsDataset(
        timestamp_us=np.concatenate([item.timestamp_us for item in datasets]),
        time_s=np.concatenate([item.time_s for item in datasets]),
        q=np.concatenate([item.q for item in datasets]),
        dq=np.concatenate([item.dq for item in datasets]),
        ddq=np.concatenate([item.ddq for item in datasets]),
        tau=np.concatenate([item.tau for item in datasets]),
        current=np.concatenate([item.current for item in datasets]),
        q_cmd=np.concatenate([item.q_cmd for item in datasets]),
        trajectory_id=np.concatenate([item.trajectory_id for item in datasets]),
        source_indices=np.concatenate([item.source_indices for item in datasets]),
    )


def _preprocess_segment(dataset, indices, trajectory_id, plan):
    timestamp_us = dataset.timestamp_us[indices]
    q_raw = dataset.q[indices]
    q, dq, ddq = reconstruct_state_from_positions(timestamp_us, q_raw, plan)
    time_s = (timestamp_us - timestamp_us[0]).astype(np.float64) * 1e-6
    motor_timestamp_us = (
        dataset.motor_timestamp_us[indices]
        if dataset.motor_timestamp_us is not None
        else np.repeat(timestamp_us[:, None], DOF, axis=1)
    )
    motor_acquired_timestamp_us = (
        dataset.motor_acquired_timestamp_us[indices]
        if dataset.motor_acquired_timestamp_us is not None
        else np.repeat(timestamp_us[:, None], DOF, axis=1)
    )
    tau_aligned = resample_columns(
        motor_timestamp_us,
        dataset.tau[indices],
        timestamp_us,
        fallback_source_timestamp_us=motor_acquired_timestamp_us,
    )
    tau = _filter_torque(timestamp_us, tau_aligned, plan)

    q_residual = q_raw - q
    tau_residual = tau_aligned - tau
    valid = _robust_row_mask(q_residual, plan.preprocess.outlier_z)
    valid &= _robust_row_mask(tau_residual, plan.preprocess.outlier_z)
    trim = plan.preprocess.endpoint_trim_s
    if trim > 0:
        valid &= (time_s >= trim) & (time_s <= time_s[-1] - trim)
    valid &= np.isfinite(q).all(axis=1)
    valid &= np.isfinite(dq).all(axis=1)
    valid &= np.isfinite(ddq).all(axis=1)
    valid &= np.isfinite(tau).all(axis=1)
    if int(np.count_nonzero(valid)) < plan.preprocess.min_samples:
        raise ValueError(
            f"trajectory {trajectory_id} retains only {np.count_nonzero(valid)} samples "
            "after filtering and outlier rejection"
        )
    selected = indices[valid]
    return ProcessedDynamicsDataset(
        timestamp_us=timestamp_us[valid],
        time_s=time_s[valid],
        q=q[valid],
        dq=dq[valid],
        ddq=ddq[valid],
        tau=tau[valid],
        current=dataset.current[selected],
        q_cmd=dataset.q_cmd[selected],
        trajectory_id=np.full(np.count_nonzero(valid), trajectory_id, dtype=np.int32),
        source_indices=selected.astype(np.int64),
    )


def _filter_torque(timestamp_us, tau, plan):
    return filter_dynamics_torque(
        timestamp_us,
        tau,
        median_window=plan.preprocess.torque_median_window,
        lowpass_hz=plan.preprocess.torque_lowpass_hz,
    )


def _robust_row_mask(residual, threshold):
    residual = np.asarray(residual, dtype=np.float64)
    center = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-9)
    z = np.abs(residual - center) / scale
    return np.all(z <= threshold, axis=1)


def _select(dataset, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("dataset split produced an empty partition")
    return ProcessedDynamicsDataset(
        timestamp_us=dataset.timestamp_us[mask],
        time_s=dataset.time_s[mask],
        q=dataset.q[mask],
        dq=dataset.dq[mask],
        ddq=dataset.ddq[mask],
        tau=dataset.tau[mask],
        current=dataset.current[mask],
        q_cmd=dataset.q_cmd[mask],
        trajectory_id=dataset.trajectory_id[mask],
        source_indices=dataset.source_indices[mask],
    )
