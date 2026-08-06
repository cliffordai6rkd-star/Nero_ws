from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from calibration.dynamics_common import load_dynamics_plan
from calibration.evaluation import save_residual_plot, torque_metrics
from calibration.identification import (
    fit_identifiable_base_parameters,
    recover_physical_parameters,
)
from calibration.preprocessing import preprocess_files, split_train_validation
from calibration.regressor import PinocchioDynamicsRegressor
from calibration.urdf_writer import write_identified_urdf


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_dynamics_plan(args.config)
    training_inputs = args.data or [
        str(profile.dataset_path)
        for profile in plan.excitation.profiles
        if profile.role == "train"
    ]
    validation_inputs = args.validation_data or [
        str(profile.dataset_path)
        for profile in plan.excitation.profiles
        if profile.role == "validation"
    ]
    training_paths = {Path(path).expanduser().resolve() for path in training_inputs}
    validation_paths = {Path(path).expanduser().resolve() for path in validation_inputs}
    overlap = sorted(training_paths.intersection(validation_paths))
    if overlap:
        raise ValueError(
            "validation trajectories must not participate in identification; "
            f"overlapping files={overlap}"
        )
    all_training = preprocess_files(training_inputs, plan)
    if validation_inputs:
        training = all_training
        validation = preprocess_files(validation_inputs, plan)
        validation_kind = "separate_trajectory_files"
    else:
        training, validation = split_train_validation(
            all_training,
            plan.preprocess.validation_fraction,
            plan.preprocess.split_seed,
        )
        validation_kind = "held_out_trajectory_id_or_contiguous_tail"

    dynamics = PinocchioDynamicsRegressor(
        plan.model, plan.preprocess.coulomb_velocity_scale_rad_s
    )
    training_regressor = dynamics.build(training)
    base_fit = fit_identifiable_base_parameters(training_regressor, plan.identification)
    physical_fit = recover_physical_parameters(dynamics, base_fit, plan.identification)
    if not physical_fit.optimizer_success:
        raise RuntimeError(
            "physical parameter recovery did not converge; refusing to write URDF: "
            f"{physical_fit.optimizer_message}; "
            f"nfev={physical_fit.optimizer_nfev}, "
            f"cost={physical_fit.optimizer_cost:.6g}, "
            f"optimality={physical_fit.optimizer_optimality:.6g}, "
            f"backend={physical_fit.optimizer_backend}, "
            f"device={physical_fit.optimizer_device}"
        )

    urdf_path, manifest_path = write_identified_urdf(
        plan,
        physical_fit,
        base_fit,
        output_path=args.output_urdf,
        manifest_path=args.manifest,
        training_data=training_inputs,
    )
    nominal_training = dynamics.predict(training, dynamics.prior_parameters)
    identified_training = dynamics.predict(training, physical_fit.parameters)
    nominal_validation = dynamics.predict(validation, dynamics.prior_parameters)
    identified_validation = dynamics.predict(validation, physical_fit.parameters)
    plot_path = save_residual_plot(
        args.residual_plot, validation, nominal_validation, identified_validation
    )

    report = {
        "format_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "accepted": True,
        "input": {
            "training_data": [str(Path(path).expanduser().resolve()) for path in training_inputs],
            "validation_data": [
                str(Path(path).expanduser().resolve()) for path in validation_inputs
            ],
            "validation_kind": validation_kind,
            "training_samples": int(training.q.shape[0]),
            "validation_samples": int(validation.q.shape[0]),
            "state_method": plan.preprocess.state_method,
            "velocity_firmware_field_used": False,
        },
        "identifiability": {
            "parameter_count": dynamics.parameter_count,
            "base_rank": base_fit.rank,
            "condition_number": base_fit.condition_number,
            "singular_values": base_fit.singular_values.tolist(),
            "svd_relative_tolerance": plan.identification.svd_relative_tolerance,
            "irls_iterations": base_fit.irls_iterations,
        },
        "physical_parameters": {
            "masses_kg": physical_fit.parameters[:70].reshape(7, 10)[:, 0].tolist(),
            "coulomb_nm": physical_fit.coulomb_nm.tolist(),
            "viscous_nm_per_rad_s": physical_fit.viscous_nm_per_rad_s.tolist(),
            "joint_bias_nm": physical_fit.bias_nm.tolist(),
            "inertia_eigenvalues_kg_m2": physical_fit.inertia_eigenvalues.tolist(),
            "optimizer_nfev": physical_fit.optimizer_nfev,
            "optimizer_cost": physical_fit.optimizer_cost,
            "optimizer_optimality": physical_fit.optimizer_optimality,
            "optimizer_backend": physical_fit.optimizer_backend,
            "optimizer_device": physical_fit.optimizer_device,
        },
        "training_metrics": {
            "original": torque_metrics(training.tau, nominal_training),
            "identified": torque_metrics(training.tau, identified_training),
        },
        "validation_metrics": {
            "original": torque_metrics(validation.tau, nominal_validation),
            "identified": torque_metrics(validation.tau, identified_validation),
        },
        "outputs": {
            "identified_urdf": str(urdf_path),
            "manifest": str(manifest_path),
            "residual_plot": str(plot_path),
            "source_urdf_overwritten": False,
        },
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    old_rmse = report["validation_metrics"]["original"]["overall_rmse_nm"]
    new_rmse = report["validation_metrics"]["identified"]["overall_rmse_nm"]
    print(f"Identified URDF: {urdf_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(
        "physical optimizer: "
        f"backend={physical_fit.optimizer_backend} "
        f"device={physical_fit.optimizer_device} "
        f"nfev={physical_fit.optimizer_nfev} "
        f"optimality={physical_fit.optimizer_optimality:.6g}"
    )
    print(f"validation RMSE original={old_rmse:.6f} N.m identified={new_rmse:.6f} N.m")
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Identify full Nero inertial, friction, and torque-bias parameters."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        help="Override configured training NPZ; repeat for multiple files",
    )
    parser.add_argument(
        "--validation-data",
        action="append",
        default=[],
        help="Independent validation NPZ; repeat for multiple files",
    )
    parser.add_argument("--output-urdf")
    parser.add_argument("--manifest", default="calibration/results/dynamics_manifest.yaml")
    parser.add_argument("--report", default="calibration/results/dynamics_identification.yaml")
    parser.add_argument("--residual-plot", default="calibration/results/dynamics_residuals.png")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
