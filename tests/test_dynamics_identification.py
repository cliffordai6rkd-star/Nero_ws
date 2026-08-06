from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from calibration.dynamics_common import (
    DynamicsDataset,
    ProcessedDynamicsDataset,
    load_dynamics_plan,
    load_dynamics_dataset,
    save_dynamics_dataset,
)
from calibration.excitation import FourierTrajectory, evaluate_fourier, validate_trajectory
from calibration.identification import (
    BaseParameterFit,
    PhysicalParameterFit,
    fit_identifiable_base_parameters,
    recover_physical_parameters,
)
from calibration.preprocessing import reconstruct_state_from_positions, split_train_validation
from calibration.regressor import PinocchioDynamicsRegressor
from calibration.urdf_writer import write_identified_urdf


ROOT = Path(__file__).resolve().parents[1]


def test_primary_calibration_config_is_fourier_not_pose_based() -> None:
    text = (ROOT / "calibration" / "config.yaml").read_text(encoding="utf-8")
    plan = load_dynamics_plan(ROOT / "calibration" / "config.yaml")

    assert "poses:" not in text
    assert plan.excitation.harmonics >= 1
    assert plan.excitation.optimization_trials >= 1


def test_fourier_evaluation_has_analytic_velocity_and_acceleration() -> None:
    time_s = np.linspace(0.0, 2.0, 401)
    center = np.zeros(7)
    sin_coefficients = np.zeros((7, 2))
    cos_coefficients = np.zeros((7, 2))
    sin_coefficients[0, 0] = 0.3
    cos_coefficients[1, 1] = -0.2

    q, dq, ddq = evaluate_fourier(
        time_s, center, sin_coefficients, cos_coefficients, fundamental_hz=0.5
    )

    omega = np.pi
    assert q[:, 0] == pytest.approx(0.3 * np.sin(omega * time_s))
    assert dq[:, 0] == pytest.approx(0.3 * omega * np.cos(omega * time_s))
    assert ddq[:, 0] == pytest.approx(-0.3 * omega**2 * np.sin(omega * time_s))
    assert q[:, 1] == pytest.approx(-0.2 * np.cos(2.0 * omega * time_s))


def test_fourier_position_fit_recovers_derivatives_without_second_difference() -> None:
    plan = load_dynamics_plan(ROOT / "calibration" / "config.yaml")
    plan = replace(
        plan,
        excitation=replace(plan.excitation, fundamental_hz=0.2),
        preprocess=replace(plan.preprocess, state_method="fourier", fourier_harmonics=2),
    )
    time_s = np.arange(1000) / 100.0
    timestamp_us = (1_000_000 + time_s * 1e6).astype(np.int64)
    sin_coefficients = np.zeros((7, 2))
    cos_coefficients = np.zeros((7, 2))
    sin_coefficients[:, 0] = np.linspace(0.05, 0.2, 7)
    cos_coefficients[:, 1] = np.linspace(-0.1, 0.1, 7)
    expected_q, expected_dq, expected_ddq = evaluate_fourier(
        time_s,
        np.asarray(plan.excitation.profiles[0].center_rad),
        sin_coefficients,
        cos_coefficients,
        plan.excitation.fundamental_hz,
    )

    q, dq, ddq = reconstruct_state_from_positions(timestamp_us, expected_q, plan)

    assert q == pytest.approx(expected_q, abs=1e-10)
    assert dq == pytest.approx(expected_dq, abs=1e-9)
    assert ddq == pytest.approx(expected_ddq, abs=1e-8)


def test_dataset_roundtrip_and_whole_trajectory_split(tmp_path: Path) -> None:
    count = 40
    q = np.arange(count * 7, dtype=np.float64).reshape(count, 7) * 1e-3
    raw = DynamicsDataset(
        timestamp_us=np.arange(count, dtype=np.int64) * 10_000 + 1,
        q=q,
        q_cmd=q.copy(),
        tau=q * 2,
        current=np.full_like(q, np.nan),
        trajectory_id=np.repeat([0, 1], count // 2).astype(np.int32),
        metadata={"firmware": "V112"},
    )
    loaded = load_dynamics_dataset(save_dynamics_dataset(tmp_path / "data.npz", raw))
    assert loaded.motor_timestamp_us == pytest.approx(
        np.repeat(raw.timestamp_us[:, None], 7, axis=1)
    )
    processed = ProcessedDynamicsDataset(
        timestamp_us=loaded.timestamp_us,
        time_s=np.arange(count) * 0.01,
        q=loaded.q,
        dq=np.zeros_like(q),
        ddq=np.zeros_like(q),
        tau=loaded.tau,
        current=loaded.current,
        q_cmd=loaded.q_cmd,
        trajectory_id=loaded.trajectory_id,
        source_indices=np.arange(count),
    )

    train, validation = split_train_validation(processed, 0.5, seed=7)

    assert set(np.unique(train.trajectory_id)).isdisjoint(np.unique(validation.trajectory_id))
    assert train.q.shape[0] + validation.q.shape[0] == count


@pytest.mark.skipif(
    pytest.importorskip("pinocchio", reason="Pinocchio is required") is None,
    reason="Pinocchio is required",
)
def test_regressor_base_fit_and_physical_recovery_match_synthetic_torque() -> None:
    plan = load_dynamics_plan(ROOT / "calibration" / "config.yaml")
    count = 500
    time_s = np.arange(count) / 100.0
    sin_coefficients = np.zeros((7, 3))
    cos_coefficients = np.zeros((7, 3))
    for joint in range(7):
        sin_coefficients[joint, joint % 3] = 0.05 + 0.01 * joint
        cos_coefficients[joint, (joint + 1) % 3] = 0.04 + 0.005 * joint
    q, dq, ddq = evaluate_fourier(
        time_s,
        plan.excitation.profiles[0].center_rad,
        sin_coefficients,
        cos_coefficients,
        0.2,
    )
    dataset = ProcessedDynamicsDataset(
        timestamp_us=(time_s * 1e6).astype(np.int64) + 1,
        time_s=time_s,
        q=q,
        dq=dq,
        ddq=ddq,
        tau=np.zeros((count, 7)),
        current=np.full((count, 7), np.nan),
        q_cmd=q,
        trajectory_id=np.zeros(count, dtype=np.int32),
        source_indices=np.arange(count),
    )
    dynamics = PinocchioDynamicsRegressor(
        plan.model, plan.preprocess.coulomb_velocity_scale_rad_s
    )
    initial_regressor = dynamics.build(dataset)
    truth = dynamics.prior_parameters.copy()
    truth[70:77] = np.linspace(0.1, 0.4, 7)
    truth[77:84] = np.linspace(0.02, 0.08, 7)
    truth[84:91] = np.linspace(-0.1, 0.1, 7)
    dataset = replace(
        dataset,
        tau=(initial_regressor.matrix @ truth).reshape(count, 7),
    )
    regressor = dynamics.build(dataset)

    base = fit_identifiable_base_parameters(regressor, plan.identification)
    physical = recover_physical_parameters(dynamics, base, plan.identification)

    base_rmse = np.sqrt(np.mean((regressor.matrix @ base.parameters - regressor.observation) ** 2))
    physical_rmse = np.sqrt(
        np.mean((regressor.matrix @ physical.parameters - regressor.observation) ** 2)
    )
    assert base.rank > 20
    assert base_rmse < 1e-5
    assert physical.optimizer_success
    assert physical_rmse < 1e-4
    assert np.all(physical.inertia_eigenvalues > 0)
    principal = np.sort(physical.inertia_eigenvalues, axis=1)
    assert np.all(principal[:, 2] <= principal[:, 0] + principal[:, 1] + 1e-10)


def test_urdf_writer_refuses_source_path_and_reloads_generated_model(tmp_path: Path) -> None:
    pytest.importorskip("pinocchio")
    plan = load_dynamics_plan(ROOT / "calibration" / "config.yaml")
    source = tmp_path / "nero.urdf"
    shutil.copy2(plan.model.urdf_path, source)
    plan = replace(plan, model=replace(plan.model, urdf_path=source))
    dynamics = PinocchioDynamicsRegressor(
        plan.model, plan.preprocess.coulomb_velocity_scale_rad_s
    )
    eigenvalues = np.stack(
        [
            np.linalg.eigvalsh(np.asarray(dynamics.model.inertias[joint].inertia))
            for joint in range(1, 8)
        ]
    )
    physical = PhysicalParameterFit(
        parameters=dynamics.prior_parameters,
        coulomb_nm=np.zeros(7),
        viscous_nm_per_rad_s=np.zeros(7),
        bias_nm=np.zeros(7),
        optimizer_success=True,
        optimizer_message="test",
        optimizer_cost=0.0,
        optimizer_nfev=1,
        optimizer_optimality=0.0,
        optimizer_backend="test",
        optimizer_device="cpu",
        inertia_eigenvalues=eigenvalues,
    )
    base = BaseParameterFit(
        parameters=dynamics.prior_parameters,
        rank=1,
        singular_values=np.ones(1),
        condition_number=1.0,
        column_scale=np.ones(91),
        identifiable_basis=np.zeros((1, 91)),
        irls_iterations=1,
        robust_weights=np.ones(1),
    )

    with pytest.raises(RuntimeError, match="overwrite"):
        write_identified_urdf(
            plan,
            physical,
            base,
            output_path=source,
            manifest_path=tmp_path / "bad.yaml",
        )

    output, manifest = write_identified_urdf(
        plan,
        physical,
        base,
        output_path=tmp_path / "nero_identified.urdf",
        manifest_path=tmp_path / "manifest.yaml",
    )
    assert output.is_file()
    assert manifest.is_file()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["friction"]["coulomb_velocity_scale_rad_s"] == pytest.approx(
        plan.preprocess.coulomb_velocity_scale_rad_s
    )
