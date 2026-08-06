from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.dynamics_common import DOF, ProcessedDynamicsDataset


def torque_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if measured.shape != predicted.shape or measured.ndim != 2 or measured.shape[1] != DOF:
        raise ValueError("torque metrics require matching (N, 7) arrays")
    residual = measured - predicted
    rmse = np.sqrt(np.mean(residual * residual, axis=0))
    torque_range = np.ptp(measured, axis=0)
    fallback = np.maximum(np.std(measured, axis=0), 1e-9)
    normalizer = np.where(torque_range > 1e-9, torque_range, fallback)
    nrmse = rmse / normalizer
    return {
        "rmse_per_joint_nm": rmse.tolist(),
        "overall_rmse_nm": float(np.sqrt(np.mean(residual * residual))),
        "nrmse_per_joint": nrmse.tolist(),
        "mean_nrmse": float(np.mean(nrmse)),
        "max_abs_residual_per_joint_nm": np.max(np.abs(residual), axis=0).tolist(),
    }


def save_residual_plot(
    path: str | Path,
    dataset: ProcessedDynamicsDataset,
    nominal_prediction: np.ndarray,
    identified_prediction: np.ndarray,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nominal_residual = dataset.tau - nominal_prediction
    identified_residual = dataset.tau - identified_prediction
    sample_time = np.arange(dataset.q.shape[0], dtype=np.float64)
    figure, axes = plt.subplots(DOF, 1, figsize=(13, 14), sharex=True)
    for joint, axis in enumerate(axes):
        axis.plot(sample_time, nominal_residual[:, joint], linewidth=0.8, label="original")
        axis.plot(sample_time, identified_residual[:, joint], linewidth=0.8, label="identified")
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_ylabel(f"J{joint + 1}\nN.m")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("validation sample")
    figure.suptitle("Nero torque residuals on held-out trajectory")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
