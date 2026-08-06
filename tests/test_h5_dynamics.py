from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from calibration.dynamics_common import load_dynamics_dataset, load_dynamics_plan
from calibration.h5_dynamics import convert_teleop_h5_to_dynamics_npz
from calibration.identify_fixed_inertia import fit_fixed_inertia_friction_bias
from calibration.regressor import RegressorData


def _write_h5(path: Path, *, count: int = 8, omit: str | None = None) -> None:
    h5py = pytest.importorskip("h5py")
    timestamp = np.arange(count, dtype=np.int64) * 10_000 + 1_000_000
    joint_timestamp = np.repeat(timestamp[:, None], 7, axis=1)
    arm_timestamp = timestamp[:, None]
    datasets = {
        "timestamp_us": timestamp,
        "q_follower_raw": np.arange(count * 7, dtype=np.float64).reshape(count, 7) * 1e-3,
        "q_cmd": np.zeros((count, 7)),
        "tau_follower_raw": np.ones((count, 7)),
        "current_follower": np.full((count, 7), 2.0),
        "motor_timestamp_follower_us": joint_timestamp,
        "motor_acquired_timestamp_follower_us": joint_timestamp + 10,
        "q_timestamp_follower_us": arm_timestamp,
        "q_acquired_timestamp_follower_us": arm_timestamp + 10,
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "factr_multimodal_episode/v3"
        teleop = h5.create_group("teleop")
        for name, value in datasets.items():
            if name != omit:
                teleop.create_dataset(name, data=value)


def test_convert_teleop_h5_uses_raw_follower_values_and_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "episode.h5"
    output = tmp_path / "episode.npz"
    _write_h5(source)

    converted = convert_teleop_h5_to_dynamics_npz(source, output)
    dataset = load_dynamics_dataset(converted)

    assert dataset.q.shape == (8, 7)
    assert dataset.q[1, 0] == pytest.approx(0.007)
    assert dataset.tau == pytest.approx(np.ones((8, 7)))
    assert dataset.motor_timestamp_us.shape == (8, 7)
    assert dataset.q_can_timestamp_us.shape == (8,)
    assert np.unique(dataset.trajectory_id).tolist() == [0]
    assert dataset.metadata["source_h5"] == str(source.resolve())


def test_convert_teleop_h5_rejects_missing_raw_torque(tmp_path: Path) -> None:
    source = tmp_path / "episode.h5"
    _write_h5(source, omit="tau_follower_raw")

    with pytest.raises(ValueError, match="tau_follower_raw"):
        convert_teleop_h5_to_dynamics_npz(source, tmp_path / "episode.npz")


def test_conversion_metadata_is_json_serializable(tmp_path: Path) -> None:
    source = tmp_path / "episode.h5"
    output = tmp_path / "episode.npz"
    _write_h5(source)
    convert_teleop_h5_to_dynamics_npz(source, output)

    with np.load(output, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata_json"].item()))
    assert metadata["contact_assumption"] == "operator_marked_contact_free_episode"


def test_fixed_inertia_fit_recovers_only_friction_and_bias() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = load_dynamics_plan(root / "calibration" / "config.yaml")
    rng = np.random.default_rng(7)
    sample_count = 200
    rows = sample_count * 7
    nuisance = rng.normal(size=(rows, 70))
    design = rng.normal(size=(rows, 21))
    matrix = np.hstack((nuisance, design))
    prior = np.zeros(91)
    prior[:70] = rng.normal(size=70)
    fitted_truth = np.concatenate(
        (
            np.linspace(0.1, 0.7, 7),
            np.linspace(0.05, 0.2, 7),
            np.linspace(-0.3, 0.3, 7),
        )
    )
    observation = matrix[:, :70] @ prior[:70] + design @ fitted_truth
    regressor = RegressorData(
        matrix=matrix,
        observation=observation,
        prior_parameters=prior,
        parameter_names=tuple(f"p{index}" for index in range(91)),
    )

    fit = fit_fixed_inertia_friction_bias(regressor, plan)

    assert fit.rank == 21
    assert fit.parameters[:70] == pytest.approx(prior[:70])
    assert fit.parameters[70:] == pytest.approx(fitted_truth, abs=1e-7)
