from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from nero_collection.control.osc_qp import (
    DynamicsSnapshot,
    OSCQPConfig,
    OSCQPController,
    OSCTargetTrajectory,
)


pytestmark = pytest.mark.skipif(find_spec("osqp") is None, reason="OSQP is not installed")


class AnalyticSevenAxisModel:
    dof = 7
    position_lower = np.full(7, -2.0)
    position_upper = np.full(7, 2.0)
    velocity_limit = np.full(7, 3.0)
    effort_limit = np.full(7, 30.0)

    def snapshot(self, q: np.ndarray, dq: np.ndarray) -> DynamicsSnapshot:
        jacobian = np.zeros((6, 7), dtype=np.float64)
        jacobian[:, :6] = np.eye(6)
        return DynamicsSnapshot(
            mass_matrix=np.diag(np.linspace(1.0, 1.6, 7)),
            nonlinear_effects=np.linspace(0.1, 0.7, 7),
            jacobian=jacobian,
            frame_drift=np.zeros(6),
            pose=np.eye(4),
        )


def _controller(**overrides: object) -> OSCQPController:
    defaults: dict[str, object] = {
        "horizon_steps": 5,
        "dt_s": 0.01,
        "torque_limit": 20.0,
        "joint_position_margin_rad": 0.0,
    }
    defaults.update(overrides)
    return OSCQPController(AnalyticSevenAxisModel(), OSCQPConfig(**defaults))


def test_osc_qp_returns_future_seven_joint_torque_sequence() -> None:
    controller = _controller()
    pose = np.eye(4)
    pose[0, 3] = 0.02
    wrench = np.asarray([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    target = OSCTargetTrajectory.constant(pose, wrench, horizon_steps=5)

    result = controller.optimize_mpc(np.zeros(7), np.zeros(7), target)

    assert result.status.lower().startswith("solved")
    assert result.tau.shape == (5, 7)
    assert result.first_tau.shape == (7,)
    assert result.joint_accelerations.shape == (5, 7)
    assert result.predicted_q.shape == (5, 7)
    assert result.predicted_dq.shape == (5, 7)
    assert result.predicted_wrenches.shape == (5, 6)
    assert np.isfinite(result.tau).all()
    assert np.max(np.abs(result.tau)) <= 20.0 + 1.0e-5
    assert result.max_constraint_violation <= controller.config.maximum_constraint_violation
    assert result.predicted_wrenches[0, 2] == pytest.approx(4.0, rel=2.0e-2)


def test_osc_qp_torque_matches_constrained_rigid_body_dynamics() -> None:
    model = AnalyticSevenAxisModel()
    controller = _controller()
    target = OSCTargetTrajectory.constant(np.eye(4), np.zeros(6), horizon_steps=5)

    result = controller.optimize_mpc(np.zeros(7), np.zeros(7), target)

    snapshot = model.snapshot(np.zeros(7), np.zeros(7))
    expected = (
        result.joint_accelerations @ snapshot.mass_matrix.T
        + snapshot.nonlinear_effects
        - result.predicted_wrenches @ snapshot.jacobian
    )
    assert result.tau == pytest.approx(expected, abs=1.0e-8)


def test_osc_qp_uses_measured_wrench_feedback() -> None:
    controller = _controller(force_feedback_gain=(1.0,) * 6)
    target = OSCTargetTrajectory.constant(
        np.eye(4), np.asarray([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]), horizon_steps=5
    )

    result = controller.optimize_mpc(
        np.zeros(7), np.zeros(7), target, measured_wrench=np.zeros(6)
    )

    assert result.predicted_wrenches[0, 2] == pytest.approx(8.0, rel=2.0e-2)


def test_osc_qp_enforces_friction_pyramid() -> None:
    controller = _controller(
        friction_coefficient=0.5,
        wrench_lower=(-20.0, -20.0, 0.0, -20.0, -20.0, -20.0),
        wrench_upper=(20.0,) * 6,
    )
    target = OSCTargetTrajectory.constant(
        np.eye(4), np.asarray([10.0, 0.0, 2.0, 0.0, 0.0, 0.0]), horizon_steps=5
    )

    result = controller.optimize_mpc(np.zeros(7), np.zeros(7), target)

    force = result.predicted_wrenches
    assert np.all(np.abs(force[:, 0]) <= 0.5 * force[:, 2] + 1.0e-5)
    assert np.all(np.abs(force[:, 1]) <= 0.5 * force[:, 2] + 1.0e-5)


def test_osc_qp_rejects_wrong_horizon() -> None:
    controller = _controller()
    target = OSCTargetTrajectory.constant(np.eye(4), np.zeros(6), horizon_steps=4)

    with pytest.raises(ValueError, match="target poses"):
        controller.optimize_mpc(np.zeros(7), np.zeros(7), target)
