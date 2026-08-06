from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import yaml

from calibration.dynamics_common import DOF, DynamicsPlan, load_dynamics_plan
from calibration.evaluation import save_residual_plot, torque_metrics
from calibration.preprocessing import preprocess_files
from calibration.regressor import PinocchioDynamicsRegressor, RegressorData


@dataclass(frozen=True)
class FixedInertiaFit:
    parameters: np.ndarray
    coulomb_nm: np.ndarray
    viscous_nm_per_rad_s: np.ndarray
    bias_nm: np.ndarray
    rank: int
    condition_number: float
    singular_values: np.ndarray
    iterations: int


def fit_fixed_inertia_friction_bias(
    regressor: RegressorData,
    plan: DynamicsPlan,
) -> FixedInertiaFit:
    inertial_count = 10 * DOF
    matrix = np.asarray(regressor.matrix[:, inertial_count:], dtype=np.float64)
    inertial_prediction = (
        regressor.matrix[:, :inertial_count] @ regressor.prior_parameters[:inertial_count]
    )
    target = np.asarray(regressor.observation - inertial_prediction, dtype=np.float64)
    joint_index = np.tile(np.arange(DOF), matrix.shape[0] // DOF)
    scale = np.asarray([_robust_scale(target[joint_index == joint]) for joint in range(DOF)])
    sample_weight = 1.0 / scale[joint_index]
    weighted_matrix = matrix * sample_weight[:, None]
    weighted_target = target * sample_weight
    column_scale = np.linalg.norm(weighted_matrix, axis=0)
    column_scale[column_scale <= np.finfo(np.float64).eps] = 1.0
    normalized_matrix = weighted_matrix / column_scale[None, :]

    singular_values = np.linalg.svd(normalized_matrix, compute_uv=False)
    threshold = singular_values[0] * plan.identification.svd_relative_tolerance
    identifiable = singular_values[singular_values > threshold]
    rank = int(identifiable.size)
    condition_number = (
        float(identifiable[0] / identifiable[-1]) if identifiable.size else float("inf")
    )

    lower = np.concatenate(
        (
            np.zeros(DOF),
            np.zeros(DOF),
            -plan.identification.max_abs_bias_nm,
        )
    )
    upper = np.concatenate(
        (
            plan.identification.max_coulomb_nm,
            plan.identification.max_viscous_nm_per_rad_s,
            plan.identification.max_abs_bias_nm,
        )
    )
    lower_scaled = lower * column_scale
    upper_scaled = upper * column_scale
    robust_weights = np.ones_like(weighted_target)
    solution_scaled = np.zeros(3 * DOF, dtype=np.float64)
    iterations = 0
    for iterations in range(1, plan.identification.max_irls_iterations + 1):
        sqrt_weight = np.sqrt(robust_weights)
        result = lsq_linear(
            normalized_matrix * sqrt_weight[:, None],
            weighted_target * sqrt_weight,
            bounds=(lower_scaled, upper_scaled),
            tol=plan.identification.physical_tolerance,
            max_iter=1000,
        )
        candidate = np.asarray(result.x, dtype=np.float64)
        residual = weighted_target - normalized_matrix @ candidate
        updated_weights = _huber_weights(residual, plan.identification.huber_delta)
        if np.linalg.norm(candidate - solution_scaled) <= 1e-9 * (
            1.0 + np.linalg.norm(solution_scaled)
        ):
            solution_scaled = candidate
            robust_weights = updated_weights
            break
        solution_scaled = candidate
        robust_weights = updated_weights

    fitted = solution_scaled / column_scale
    parameters = regressor.prior_parameters.copy()
    parameters[inertial_count:] = fitted
    return FixedInertiaFit(
        parameters=parameters,
        coulomb_nm=fitted[:DOF].copy(),
        viscous_nm_per_rad_s=fitted[DOF : 2 * DOF].copy(),
        bias_nm=fitted[2 * DOF :].copy(),
        rank=rank,
        condition_number=condition_number,
        singular_values=singular_values,
        iterations=iterations,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_dynamics_plan(args.config)
    training_inputs = args.data
    validation_inputs = args.validation_data
    if not training_inputs or not validation_inputs:
        raise ValueError("fixed-inertia identification requires training and validation data")
    overlap = {
        Path(path).expanduser().resolve() for path in training_inputs
    }.intersection(Path(path).expanduser().resolve() for path in validation_inputs)
    if overlap:
        raise ValueError(f"training and validation data overlap: {sorted(overlap)}")

    training = preprocess_files(training_inputs, plan)
    validation = preprocess_files(validation_inputs, plan)
    dynamics = PinocchioDynamicsRegressor(
        plan.model, plan.preprocess.coulomb_velocity_scale_rad_s
    )
    fit = fit_fixed_inertia_friction_bias(dynamics.build(training), plan)
    original_training = dynamics.predict(training, dynamics.prior_parameters)
    fitted_training = dynamics.predict(training, fit.parameters)
    original_validation = dynamics.predict(validation, dynamics.prior_parameters)
    fitted_validation = dynamics.predict(validation, fit.parameters)
    plot_path = save_residual_plot(
        args.residual_plot, validation, original_validation, fitted_validation
    )
    manifest_path = _write_manifest(args.manifest, plan, fit, training_inputs)
    report_path = _write_report(
        args.report,
        plan,
        fit,
        training_inputs,
        validation_inputs,
        training,
        validation,
        original_training,
        fitted_training,
        original_validation,
        fitted_validation,
        plot_path,
        manifest_path,
    )
    original_rmse = torque_metrics(validation.tau, original_validation)["overall_rmse_nm"]
    fitted_rmse = torque_metrics(validation.tau, fitted_validation)["overall_rmse_nm"]
    print(f"Model URDF (fixed): {plan.model.urdf_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(
        f"friction/bias fit: rank={fit.rank}/21 condition={fit.condition_number:.6g} "
        f"IRLS iterations={fit.iterations}"
    )
    print(
        f"validation RMSE original={original_rmse:.6f} N.m "
        f"fixed-inertia+friction/bias={fitted_rmse:.6f} N.m"
    )
    return 0


def _write_manifest(path, plan, fit, training_inputs) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "identification_mode": "fixed_urdf_inertia_friction_bias_only",
        "source_urdf": str(plan.model.urdf_path),
        "identified_urdf": str(plan.model.urdf_path),
        "config_path": str(plan.source_path),
        "training_data": [str(Path(path).expanduser().resolve()) for path in training_inputs],
        "joint_names": list(plan.model.joint_names),
        "locked_joint_names": list(plan.model.locked_joint_names),
        "friction": {
            "coulomb_nm": fit.coulomb_nm.tolist(),
            "viscous_nm_per_rad_s": fit.viscous_nm_per_rad_s.tolist(),
            "coulomb_velocity_scale_rad_s": plan.preprocess.coulomb_velocity_scale_rad_s,
            "velocity_sign_model": (
                f"tanh(dq / {plan.preprocess.coulomb_velocity_scale_rad_s:.12g})"
            ),
        },
        "joint_torque_bias_nm": fit.bias_nm.tolist(),
        "fit": {
            "rank": fit.rank,
            "condition_number": fit.condition_number,
            "singular_values": fit.singular_values.tolist(),
            "irls_iterations": fit.iterations,
        },
    }
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _write_report(
    path,
    plan,
    fit,
    training_inputs,
    validation_inputs,
    training,
    validation,
    original_training,
    fitted_training,
    original_validation,
    fitted_validation,
    plot_path,
    manifest_path,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "accepted": bool(
            torque_metrics(validation.tau, fitted_validation)["overall_rmse_nm"]
            < torque_metrics(validation.tau, original_validation)["overall_rmse_nm"]
        ),
        "identification_mode": "fixed_urdf_inertia_friction_bias_only",
        "model_urdf": str(plan.model.urdf_path),
        "input": {
            "training_data": [str(Path(path).expanduser().resolve()) for path in training_inputs],
            "validation_data": [
                str(Path(path).expanduser().resolve()) for path in validation_inputs
            ],
            "training_samples": int(training.q.shape[0]),
            "validation_samples": int(validation.q.shape[0]),
            "state_method": plan.preprocess.state_method,
        },
        "fit": {
            "parameter_count": 3 * DOF,
            "rank": fit.rank,
            "condition_number": fit.condition_number,
            "singular_values": fit.singular_values.tolist(),
            "irls_iterations": fit.iterations,
            "coulomb_nm": fit.coulomb_nm.tolist(),
            "viscous_nm_per_rad_s": fit.viscous_nm_per_rad_s.tolist(),
            "joint_bias_nm": fit.bias_nm.tolist(),
        },
        "training_metrics": {
            "original": torque_metrics(training.tau, original_training),
            "fitted": torque_metrics(training.tau, fitted_training),
        },
        "validation_metrics": {
            "original": torque_metrics(validation.tau, original_validation),
            "fitted": torque_metrics(validation.tau, fitted_validation),
        },
        "outputs": {
            "manifest": str(manifest_path),
            "residual_plot": str(plot_path),
        },
    }
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _robust_scale(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    return max(float(1.4826 * np.median(np.abs(values - median))), 1e-3)


def _huber_weights(residual, delta) -> np.ndarray:
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    weights = np.ones_like(absolute)
    mask = absolute > delta
    weights[mask] = delta / absolute[mask]
    return weights


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Fit Nero friction and torque bias while keeping URDF inertias fixed."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--validation-data", action="append", default=[])
    parser.add_argument("--manifest", default="calibration/results/fixed_inertia_manifest.yaml")
    parser.add_argument("--report", default="calibration/results/fixed_inertia_identification.yaml")
    parser.add_argument("--residual-plot", default="calibration/results/fixed_inertia_residuals.png")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
