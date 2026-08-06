from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from calibration.dynamics_common import build_reduced_model, load_dynamics_plan
from calibration.evaluation import save_residual_plot, torque_metrics
from calibration.preprocessing import preprocess_files
from calibration.regressor import PinocchioDynamicsRegressor
from calibration.urdf_writer import load_identified_parameters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare original and identified Nero dynamics on an independent trajectory."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--data", action="append", required=True, help="Independent validation NPZ")
    parser.add_argument("--identified-urdf", required=True)
    parser.add_argument("--manifest", default="calibration/results/dynamics_manifest.yaml")
    parser.add_argument("--output", default="calibration/results/dynamics_external_validation.yaml")
    parser.add_argument("--plot", default="calibration/results/dynamics_external_residuals.png")
    args = parser.parse_args(argv)

    plan = load_dynamics_plan(args.config)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    training_paths = {
        Path(path).expanduser().resolve()
        for path in manifest_payload.get("training_data", [])
    }
    validation_paths = {Path(path).expanduser().resolve() for path in args.data}
    overlap = sorted(training_paths.intersection(validation_paths))
    if overlap:
        raise ValueError(
            "external validation data was used for identification; "
            f"overlapping files={overlap}"
        )
    dataset = preprocess_files(args.data, plan)
    dynamics = PinocchioDynamicsRegressor(
        plan.model, plan.preprocess.coulomb_velocity_scale_rad_s
    )
    identified_settings = replace(
        plan.model, urdf_path=Path(args.identified_urdf).expanduser().resolve()
    )
    _, identified_model = build_reduced_model(identified_settings)
    coulomb, viscous, bias = load_identified_parameters(manifest_path)
    identified_parameters = dynamics.parameters_from_model(
        identified_model, coulomb, viscous, bias
    )
    original_prediction = dynamics.predict(dataset, dynamics.prior_parameters)
    identified_prediction = dynamics.predict(dataset, identified_parameters)
    original_metrics = torque_metrics(dataset.tau, original_prediction)
    identified_metrics = torque_metrics(dataset.tau, identified_prediction)
    plot = save_residual_plot(args.plot, dataset, original_prediction, identified_prediction)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "validation_type": "independent_dynamic_trajectory",
        "data": [str(Path(path).expanduser().resolve()) for path in args.data],
        "original_urdf": str(plan.model.urdf_path),
        "identified_urdf": str(identified_settings.urdf_path),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "sample_count": int(dataset.q.shape[0]),
        "original_metrics": original_metrics,
        "identified_metrics": identified_metrics,
        "residual_plot": str(plot),
    }
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"External validation report: {output}")
    print(
        "overall RMSE: "
        f"original={original_metrics['overall_rmse_nm']:.6f} N.m, "
        f"identified={identified_metrics['overall_rmse_nm']:.6f} N.m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
