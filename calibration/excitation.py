from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibration.dynamics_common import (
    DOF,
    DynamicsPlan,
    ExcitationProfile,
    build_reduced_model,
)


@dataclass(frozen=True)
class FourierTrajectory:
    time_s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    sin_coefficients: np.ndarray
    cos_coefficients: np.ndarray
    condition_number: float
    regressor_rank: int


def generate_optimized_trajectory(
    plan: DynamicsPlan,
    profile: ExcitationProfile,
    reference_trajectories: tuple[FourierTrajectory, ...] = (),
) -> FourierTrajectory:
    pin, model = build_reduced_model(plan.model)
    cfg = plan.excitation
    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64) + cfg.joint_limit_margin_rad
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64) - cfg.joint_limit_margin_rad
    if np.any(lower >= upper):
        raise ValueError("joint limit margin leaves an empty excitation range")
    if np.any(profile.center_rad - cfg.amplitude_rad < lower) or np.any(
        profile.center_rad + cfg.amplitude_rad > upper
    ):
        raise ValueError(
            "excitation center +/- amplitude violates URDF limits with the configured margin"
        )

    count = int(round(cfg.duration_s * cfg.sample_rate_hz))
    time_s = np.arange(count, dtype=np.float64) / cfg.sample_rate_hz
    rng = np.random.default_rng(profile.seed)
    best: FourierTrajectory | None = None
    best_objective = float("inf")
    data = model.createData()
    score_indices = np.linspace(0, count - 1, min(count, 500), dtype=int)

    for _ in range(cfg.optimization_trials):
        sin_coefficients, cos_coefficients = _random_coefficients(
            rng, cfg.harmonics, cfg.amplitude_rad
        )
        sin_coefficients, cos_coefficients = _project_coefficients(
            time_s,
            profile.center_rad,
            sin_coefficients,
            cos_coefficients,
            cfg.fundamental_hz,
            cfg.amplitude_rad,
            cfg.max_velocity_rad_s,
            cfg.max_acceleration_rad_s2,
            lower,
            upper,
        )
        q, dq, ddq = evaluate_fourier(
            time_s,
            profile.center_rad,
            sin_coefficients,
            cos_coefficients,
            cfg.fundamental_hz,
        )
        q_score = [q[score_indices]]
        dq_score = [dq[score_indices]]
        ddq_score = [ddq[score_indices]]
        for reference in reference_trajectories:
            reference_indices = np.linspace(
                0,
                reference.q.shape[0] - 1,
                min(reference.q.shape[0], 500),
                dtype=int,
            )
            q_score.append(reference.q[reference_indices])
            dq_score.append(reference.dq[reference_indices])
            ddq_score.append(reference.ddq[reference_indices])
        condition, rank = _trajectory_regressor_score(
            pin,
            model,
            data,
            np.concatenate(q_score),
            np.concatenate(dq_score),
            np.concatenate(ddq_score),
            plan.preprocess.coulomb_velocity_scale_rad_s,
        )
        # Prefer higher numerical rank first, then a lower identifiable-subspace condition number.
        objective = np.log10(max(condition, 1.0)) + 10.0 * (DOF * 10 + 2 * DOF - rank)
        if objective < best_objective:
            best_objective = objective
            best = FourierTrajectory(
                time_s=time_s.copy(),
                q=q,
                dq=dq,
                ddq=ddq,
                sin_coefficients=sin_coefficients,
                cos_coefficients=cos_coefficients,
                condition_number=condition,
                regressor_rank=rank,
            )
    if best is None:  # pragma: no cover - guarded by config validation
        raise RuntimeError("failed to generate an excitation trajectory")
    validate_trajectory(best, plan, lower=lower, upper=upper)
    # Candidate selection above may use previous training trajectories, but the
    # trajectory artifact must report its own diagnostics. The combined value is
    # reported separately after all training profiles have been generated.
    standalone_condition, standalone_rank = _trajectory_regressor_score(
        pin,
        model,
        data,
        best.q[score_indices],
        best.dq[score_indices],
        best.ddq[score_indices],
        plan.preprocess.coulomb_velocity_scale_rad_s,
    )
    return FourierTrajectory(
        time_s=best.time_s,
        q=best.q,
        dq=best.dq,
        ddq=best.ddq,
        sin_coefficients=best.sin_coefficients,
        cos_coefficients=best.cos_coefficients,
        condition_number=standalone_condition,
        regressor_rank=standalone_rank,
    )


def combined_trajectory_diagnostics(
    plan: DynamicsPlan,
    trajectories: tuple[FourierTrajectory, ...],
) -> tuple[float, int]:
    if not trajectories:
        raise ValueError("combined trajectory diagnostics require at least one trajectory")
    pin, model = build_reduced_model(plan.model)
    q_values: list[np.ndarray] = []
    dq_values: list[np.ndarray] = []
    ddq_values: list[np.ndarray] = []
    for trajectory in trajectories:
        indices = np.linspace(
            0,
            trajectory.q.shape[0] - 1,
            min(trajectory.q.shape[0], 500),
            dtype=int,
        )
        q_values.append(trajectory.q[indices])
        dq_values.append(trajectory.dq[indices])
        ddq_values.append(trajectory.ddq[indices])
    return _trajectory_regressor_score(
        pin,
        model,
        model.createData(),
        np.concatenate(q_values),
        np.concatenate(dq_values),
        np.concatenate(ddq_values),
        plan.preprocess.coulomb_velocity_scale_rad_s,
    )


def evaluate_fourier(
    time_s: np.ndarray,
    center_rad: np.ndarray,
    sin_coefficients: np.ndarray,
    cos_coefficients: np.ndarray,
    fundamental_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    center = np.asarray(center_rad, dtype=np.float64).reshape(DOF)
    sin_coefficients = np.asarray(sin_coefficients, dtype=np.float64)
    cos_coefficients = np.asarray(cos_coefficients, dtype=np.float64)
    if sin_coefficients.shape != cos_coefficients.shape or sin_coefficients.shape[0] != DOF:
        raise ValueError("Fourier coefficient arrays must both have shape (7, harmonics)")
    harmonics = np.arange(1, sin_coefficients.shape[1] + 1, dtype=np.float64)
    omega = 2.0 * np.pi * float(fundamental_hz) * harmonics
    phase = time_s[:, None] * omega[None, :]
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    q = center[None, :] + sin_phase @ sin_coefficients.T + cos_phase @ cos_coefficients.T
    dq = cos_phase @ (sin_coefficients * omega[None, :]).T - sin_phase @ (
        cos_coefficients * omega[None, :]
    ).T
    ddq = -sin_phase @ (sin_coefficients * omega[None, :] ** 2).T - cos_phase @ (
        cos_coefficients * omega[None, :] ** 2
    ).T
    return q, dq, ddq


def validate_trajectory(
    trajectory: FourierTrajectory,
    plan: DynamicsPlan,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    sample_count = np.asarray(trajectory.time_s).size
    expected_shape = (sample_count, DOF)
    if sample_count < 2 or np.any(np.diff(trajectory.time_s) <= 0):
        raise ValueError("trajectory time_s must contain increasing timestamps")
    for name in ("q", "dq", "ddq"):
        value = np.asarray(getattr(trajectory, name), dtype=np.float64)
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"trajectory {name} must be a finite {expected_shape} array")
    measured_period = float(np.median(np.diff(trajectory.time_s)))
    expected_period = 1.0 / plan.excitation.sample_rate_hz
    if not np.isclose(measured_period, expected_period, rtol=1e-4, atol=1e-9):
        raise ValueError(
            "trajectory sample period does not match excitation.sample_rate_hz: "
            f"trajectory={measured_period:.9g}s config={expected_period:.9g}s"
        )
    if lower is None or upper is None:
        _, model = build_reduced_model(plan.model)
        lower = np.asarray(model.lowerPositionLimit) + plan.excitation.joint_limit_margin_rad
        upper = np.asarray(model.upperPositionLimit) - plan.excitation.joint_limit_margin_rad
    checks = {
        "position lower": np.min(trajectory.q - lower[None, :]),
        "position upper": np.min(upper[None, :] - trajectory.q),
        "velocity": np.min(
            plan.excitation.max_velocity_rad_s[None, :] - np.abs(trajectory.dq)
        ),
        "acceleration": np.min(
            plan.excitation.max_acceleration_rad_s2[None, :] - np.abs(trajectory.ddq)
        ),
    }
    failed = {name: value for name, value in checks.items() if value < -1e-9}
    if failed:
        raise ValueError(f"generated trajectory violates constraints: {failed}")


def save_trajectory(
    path: str | Path,
    trajectory: FourierTrajectory,
    plan: DynamicsPlan,
    profile: ExcitationProfile,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time_s=trajectory.time_s,
        q_cmd=trajectory.q,
        dq_cmd=trajectory.dq,
        ddq_cmd=trajectory.ddq,
        sin_coefficients=trajectory.sin_coefficients,
        cos_coefficients=trajectory.cos_coefficients,
        fundamental_hz=np.asarray(plan.excitation.fundamental_hz),
        condition_number=np.asarray(trajectory.condition_number),
        regressor_rank=np.asarray(trajectory.regressor_rank),
        config_path=np.asarray(str(plan.source_path)),
        profile_name=np.asarray(profile.name),
        profile_role=np.asarray(profile.role),
        profile_seed=np.asarray(profile.seed),
        center_rad=np.asarray(profile.center_rad),
    )
    return output


def load_trajectory(path: str | Path) -> FourierTrajectory:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as values:
        return FourierTrajectory(
            time_s=np.asarray(values["time_s"], dtype=np.float64),
            q=np.asarray(values["q_cmd"], dtype=np.float64),
            dq=np.asarray(values["dq_cmd"], dtype=np.float64),
            ddq=np.asarray(values["ddq_cmd"], dtype=np.float64),
            sin_coefficients=np.asarray(values["sin_coefficients"], dtype=np.float64),
            cos_coefficients=np.asarray(values["cos_coefficients"], dtype=np.float64),
            condition_number=float(np.asarray(values["condition_number"])),
            regressor_rank=int(np.asarray(values["regressor_rank"])),
        )


def _random_coefficients(
    rng: np.random.Generator, harmonics: int, amplitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scale = amplitude[:, None] / np.sqrt(2.0 * harmonics)
    return rng.normal(size=(DOF, harmonics)) * scale, rng.normal(
        size=(DOF, harmonics)
    ) * scale


def _project_coefficients(
    time_s: np.ndarray,
    center: np.ndarray,
    sin_coefficients: np.ndarray,
    cos_coefficients: np.ndarray,
    fundamental_hz: float,
    amplitude: np.ndarray,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sin_coefficients = sin_coefficients.copy()
    cos_coefficients = cos_coefficients.copy()
    q, dq, ddq = evaluate_fourier(
        time_s, center, sin_coefficients, cos_coefficients, fundamental_hz
    )
    excursion = np.max(np.abs(q - center[None, :]), axis=0)
    position_room = np.minimum(center - lower, upper - center)
    factors = np.minimum.reduce(
        [
            np.ones(DOF),
            amplitude / np.maximum(excursion, 1e-12),
            position_room / np.maximum(excursion, 1e-12),
            velocity_limit / np.maximum(np.max(np.abs(dq), axis=0), 1e-12),
            acceleration_limit / np.maximum(np.max(np.abs(ddq), axis=0), 1e-12),
        ]
    )
    # A small margin prevents floating point and command-timing excursions at the limits.
    factors = np.minimum(factors * 0.98, 1.0)
    return sin_coefficients * factors[:, None], cos_coefficients * factors[:, None]


def _trajectory_regressor_score(pin, model, data, q, dq, ddq, velocity_scale):
    blocks = []
    for q_i, dq_i, ddq_i in zip(q, dq, ddq):
        inertial = np.asarray(
            pin.computeJointTorqueRegressor(model, data, q_i, dq_i, ddq_i),
            dtype=np.float64,
        )
        coulomb = np.diag(np.tanh(dq_i / velocity_scale))
        viscous = np.diag(dq_i)
        blocks.append(np.hstack((inertial, coulomb, viscous)))
    design = np.vstack(blocks)
    norms = np.linalg.norm(design, axis=0)
    active = norms > np.finfo(np.float64).eps
    normalized = design[:, active] / norms[active]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if not singular_values.size:
        return float("inf"), 0
    tolerance = singular_values[0] * 1e-8
    identifiable = singular_values[singular_values > tolerance]
    rank = int(identifiable.size)
    condition = (
        float(identifiable[0] / identifiable[-1]) if identifiable.size else float("inf")
    )
    return condition, rank
