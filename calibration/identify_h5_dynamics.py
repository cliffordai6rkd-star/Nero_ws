from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.dynamics_common import load_dynamics_plan
from calibration.h5_dynamics import (
    conversion_manifest_entry,
    convert_teleop_h5_to_dynamics_npz,
)
from calibration.identify_dynamics import main as identify_dynamics_main
from calibration.identify_fixed_inertia import main as identify_fixed_inertia_main


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_dynamics_plan(args.config)
    converted_dir = Path(args.converted_dir).expanduser().resolve()
    converted_dir.mkdir(parents=True, exist_ok=True)

    training = _convert_group(args.train_h5, converted_dir, "train")
    validation = _convert_group(args.validation_h5, converted_dir, "validation")
    conversion_manifest = converted_dir / "conversion_manifest.json"
    conversion_manifest.write_text(
        json.dumps(
            {
                "training": [conversion_manifest_entry(path) for path in training],
                "validation": [conversion_manifest_entry(path) for path in validation],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    source = plan.model.urdf_path
    output_urdf = (
        Path(args.output_urdf).expanduser().resolve()
        if args.output_urdf
        else source.with_name(f"{source.stem}_h5_identified{source.suffix}")
    )
    identify_args = ["--config", str(plan.source_path)]
    for path in training:
        identify_args.extend(("--data", str(path)))
    for path in validation:
        identify_args.extend(("--validation-data", str(path)))
    if not args.fixed_inertia:
        identify_args.extend(("--output-urdf", str(output_urdf)))
    default_prefix = "h5_fixed_inertia" if args.fixed_inertia else "h5_dynamics"
    manifest = args.manifest or f"calibration/results/{default_prefix}_manifest.yaml"
    report = args.report or f"calibration/results/{default_prefix}_identification.yaml"
    residual_plot = args.residual_plot or f"calibration/results/{default_prefix}_residuals.png"
    identify_args.extend(
        (
            "--manifest",
            str(Path(manifest).expanduser().resolve()),
            "--report",
            str(Path(report).expanduser().resolve()),
            "--residual-plot",
            str(Path(residual_plot).expanduser().resolve()),
        )
    )
    print(f"Converted dataset manifest: {conversion_manifest}")
    return (
        identify_fixed_inertia_main(identify_args)
        if args.fixed_inertia
        else identify_dynamics_main(identify_args)
    )


def _convert_group(paths: list[str], output_dir: Path, role: str) -> list[Path]:
    converted: list[Path] = []
    for index, value in enumerate(paths):
        source = Path(value).expanduser().resolve()
        output = output_dir / f"{role}_{index:03d}_{source.stem}.npz"
        converted.append(convert_teleop_h5_to_dynamics_npz(source, output))
    return converted


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(
        description="Identify Nero dynamics directly from contact-free teleoperation HDF5 episodes."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument(
        "--train-h5",
        action="append",
        required=True,
        help="Contact-free training HDF5 episode; repeat for multiple episodes",
    )
    parser.add_argument(
        "--validation-h5",
        action="append",
        required=True,
        help="Independent contact-free validation HDF5 episode; repeat for multiple episodes",
    )
    parser.add_argument("--converted-dir", default="calibration/data/h5_imported")
    parser.add_argument(
        "--fixed-inertia",
        action="store_true",
        help="Keep official URDF inertias and fit only friction and torque bias",
    )
    parser.add_argument("--output-urdf")
    parser.add_argument("--manifest")
    parser.add_argument("--report")
    parser.add_argument("--residual-plot")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
