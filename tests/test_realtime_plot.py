from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import yaml

from nero_collection.config import (
    InverseDynamicsConfig,
    RealtimePlotConfig,
    StateParamConfig,
    _parse_inverse_dynamics,
    _parse_realtime_plot,
)
from inference.wrench_mapping import (
    PinocchioContactWrenchEstimator,
    WrenchMappingConfig,
    solve_damped_wrench,
)
from nero_collection.inverse_dynamics import PinocchioJointTorqueResidualEstimator
from nero_collection.inverse_dynamics import JointTorqueResidualEstimate
from nero_collection.realtime_dynamics import CenteredThreePointTorqueResidualStream
from nero_collection.realtime_plot import (
    CumulativeJointBuffer,
    _MatplotlibPlotWindow,
    _set_dynamic_ylim,
    RealtimeJointPlotter,
)


def test_realtime_plot_config_defaults_to_ten_second_window() -> None:
    config = _parse_realtime_plot({})

    assert config.enabled is False
    assert config.window_s == pytest.approx(10.0)
    assert config.update_rate_hz == pytest.approx(20.0)


def test_realtime_plot_config_resolves_identified_manifest(tmp_path: Path) -> None:
    config = _parse_inverse_dynamics(
        {"manifest_path": "results/dynamics_manifest.yaml"},
        tmp_path,
    )

    assert config.manifest_path == (
        tmp_path / "results" / "dynamics_manifest.yaml"
    ).resolve()


def test_realtime_plot_config_rejects_removed_dynamics_and_wrench_options() -> None:
    with pytest.raises(ValueError, match="removed options"):
        _parse_realtime_plot({"inverse_dynamics": {}})
    with pytest.raises(ValueError, match="removed options"):
        _parse_realtime_plot({"wrench_mapping": {}})


@pytest.mark.parametrize(
    "data",
    [
        {"window_s": 0.0},
        {"window_s": float("nan")},
        {"update_rate_hz": 0.0},
    ],
)
def test_realtime_plot_config_rejects_invalid_rates(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _parse_realtime_plot(data)


def test_cumulative_joint_buffer_keeps_complete_elapsed_history() -> None:
    buffer = CumulativeJointBuffer(window_s=10.0)
    for timestamp_s, value in ((0, 1.0), (5, 2.0), (11, 3.0)):
        joints = np.full(7, value, dtype=np.float64)
        buffer.append(
            timestamp_s * 1_000_000,
            joints,
            joints * 3,
        )

    time_s, tau_ext_cal, tau_ext_pred = buffer.arrays()

    assert time_s == pytest.approx([0.0, 5.0, 11.0])
    assert tau_ext_cal.shape == (3, 7)
    assert np.allclose(tau_ext_cal[:, 0], [1.0, 2.0, 3.0])
    assert np.allclose(tau_ext_pred[:, 0], [3.0, 6.0, 9.0])


def test_cumulative_joint_buffer_rejects_non_seven_dimensional_data() -> None:
    buffer = CumulativeJointBuffer(window_s=10.0)

    with pytest.raises(RuntimeError, match="7D tau_ext_cal"):
        buffer.append(1, np.zeros(6), np.zeros(7))

    with pytest.raises(RuntimeError, match="7D tau_ext_pred"):
        buffer.append(1, np.zeros(7), np.zeros(6))


def test_cumulative_joint_buffer_clear_removes_all_history() -> None:
    buffer = CumulativeJointBuffer(window_s=10.0)
    buffer.append(1, np.ones(7), np.ones(7))

    buffer.clear()

    time_s, tau_ext_cal, tau_ext_pred = buffer.arrays()
    assert time_s.shape == (0,)
    assert tau_ext_cal.shape == (0, 7)
    assert tau_ext_pred.shape == (0, 7)


def test_realtime_plot_places_cal_above_pred_without_mixing() -> None:
    assert tuple(item[0] for item in _MatplotlibPlotWindow._PLOTS) == (
        "tau_ext_cal",
        "||tau_ext||",
        "tau_ext_pred",
        "||tau_ext||",
    )


def test_realtime_plot_uses_generic_tau_ext_norm_labels() -> None:
    assert _MatplotlibPlotWindow._PLOTS[1][0] == "||tau_ext||"
    assert _MatplotlibPlotWindow._PLOTS[3][0] == "||tau_ext||"
    assert _MatplotlibPlotWindow._PLOTS[1][1] == "||tau_ext|| [N.m]"
    assert _MatplotlibPlotWindow._PLOTS[3][1] == "||tau_ext|| [N.m]"


def test_realtime_plot_y_scale_is_at_least_plus_minus_three_nm() -> None:
    class Axis:
        limits = None

        def set_ylim(self, lower, upper):
            self.limits = (lower, upper)

    axis = Axis()
    _set_dynamic_ylim(axis, np.asarray([-0.2, 0.5]))
    assert axis.limits == pytest.approx((-3.0, 3.0))

    _set_dynamic_ylim(axis, np.asarray([-4.0, 2.0]))
    assert axis.limits == pytest.approx((-4.32, 4.32))


def test_damped_wrench_maps_joint_residual_and_reports_nullspace_error() -> None:
    jacobian = np.zeros((6, 7), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    tau_residual = np.arange(1, 8, dtype=np.float64)

    wrench, error, condition = solve_damped_wrench(jacobian, tau_residual, damping=1e-6)

    assert wrench == pytest.approx(tau_residual[:6], rel=1e-9)
    assert error == pytest.approx(7.0 / np.linalg.norm(tau_residual), rel=1e-9)
    assert condition == pytest.approx(1.0)


def test_weighted_damped_wrench_reduces_influence_of_low_confidence_joint() -> None:
    jacobian = np.zeros((6, 7), dtype=np.float64)
    jacobian[0, 0] = 1.0
    jacobian[0, 1] = 1.0
    tau_residual = np.zeros(7, dtype=np.float64)
    tau_residual[:2] = [1.0, 10.0]

    unweighted, _, _ = solve_damped_wrench(
        jacobian, tau_residual, damping=1e-6
    )
    weighted, _, _ = solve_damped_wrench(
        jacobian,
        tau_residual,
        damping=1e-6,
        joint_weights=(1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0),
    )

    assert unweighted[0] == pytest.approx(5.5, rel=1e-9)
    assert weighted[0] == pytest.approx(1.0 + 0.09 / 1.01, rel=1e-9)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_contact_estimator_maps_external_joint_torque_to_tool_wrench() -> None:
    estimator = PinocchioContactWrenchEstimator(
        WrenchMappingConfig(
            urdf_path=(
                Path(__file__).resolve().parents[1]
                / "urdf"
                / "nero"
                / "nero_with_gripper.urdf"
            ),
            damping=1e-8,
        )
    )
    q = np.array([0.2, -0.7, 0.3, 1.1, -0.4, 0.5, 0.2], dtype=np.float64)
    estimator.pin.computeJointJacobians(estimator.model, estimator.data, q)
    estimator.pin.framesForwardKinematics(estimator.model, estimator.data, q)
    jacobian = np.asarray(
        estimator.pin.getFrameJacobian(
            estimator.model,
            estimator.data,
            estimator.frame_id,
            estimator.reference_frame,
        )
    )
    expected_wrench = np.array([3.0, -2.0, 5.0, 0.4, -0.3, 0.2])
    tau_ext = jacobian.T @ expected_wrench

    estimate = estimator.map_joint_torque(q, tau_ext)

    assert estimate.wrench == pytest.approx(expected_wrench, rel=1e-7, abs=1e-7)
    assert estimate.tau_external == pytest.approx(tau_ext)
    assert estimate.reconstruction_error < 1e-8


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_returns_tau_id_minus_measured() -> None:
    import pinocchio as pin

    config = InverseDynamicsConfig(
        urdf_path=(
            Path(__file__).resolve().parents[1]
            / "urdf"
            / "nero"
            / "nero_with_gripper.urdf"
        )
    )
    estimator = PinocchioJointTorqueResidualEstimator(config)
    q = pin.neutral(estimator.model)
    dq = np.zeros(7, dtype=np.float64)
    ddq = np.zeros(7, dtype=np.float64)
    tau_id = np.asarray(pin.rnea(estimator.model, estimator.data, q, dq, ddq)).copy()
    expected_residual = np.linspace(-0.3, 0.3, 7)

    estimate = estimator.estimate(q, dq, ddq, tau_id - expected_residual)

    assert estimate.tau_id == pytest.approx(tau_id)
    assert estimate.tau_model == pytest.approx(tau_id)
    assert estimate.tau_friction == pytest.approx(np.zeros(7))
    assert estimate.tau_bias == pytest.approx(np.zeros(7))
    assert estimate.tau_residual == pytest.approx(expected_residual)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_ignores_identified_friction_and_bias_for_tau_f(
    tmp_path: Path,
) -> None:
    import pinocchio as pin

    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "nero"
        / "nero_with_gripper.urdf"
    )
    coulomb = np.linspace(0.1, 0.7, 7)
    viscous = np.linspace(0.01, 0.07, 7)
    bias = np.linspace(-0.3, 0.3, 7)
    velocity_scale = 0.02
    manifest_path = tmp_path / "dynamics_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "identified_urdf": str(urdf_path),
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "friction": {
                    "coulomb_nm": coulomb.tolist(),
                    "viscous_nm_per_rad_s": viscous.tolist(),
                    "coulomb_velocity_scale_rad_s": velocity_scale,
                },
                "joint_torque_bias_nm": bias.tolist(),
            }
        ),
        encoding="utf-8",
    )
    estimator = PinocchioJointTorqueResidualEstimator(
        InverseDynamicsConfig(urdf_path=urdf_path, manifest_path=manifest_path)
    )
    q = pin.neutral(estimator.model)
    dq = np.linspace(-0.3, 0.3, 7)
    ddq = np.linspace(0.2, -0.2, 7)
    tau_id = np.asarray(pin.rnea(estimator.model, estimator.data, q, dq, ddq)).copy()
    expected_residual = np.linspace(-0.2, 0.2, 7)

    estimate = estimator.estimate(q, dq, ddq, tau_id - expected_residual)

    assert estimate.tau_id == pytest.approx(tau_id)
    assert estimate.tau_friction == pytest.approx(np.zeros(7))
    assert estimate.tau_bias == pytest.approx(np.zeros(7))
    assert estimate.tau_model == pytest.approx(tau_id)
    assert estimate.tau_residual == pytest.approx(expected_residual)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_rejects_manifest_for_another_urdf(
    tmp_path: Path,
) -> None:
    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "nero"
        / "nero_with_gripper.urdf"
    )
    manifest_path = tmp_path / "dynamics_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "identified_urdf": str(tmp_path / "another.urdf"),
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "friction": {
                    "coulomb_nm": [0.0] * 7,
                    "viscous_nm_per_rad_s": [0.0] * 7,
                    "coulomb_velocity_scale_rad_s": 0.02,
                },
                "joint_torque_bias_nm": [0.0] * 7,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="manifest/URDF mismatch"):
        PinocchioJointTorqueResidualEstimator(
            InverseDynamicsConfig(urdf_path=urdf_path, manifest_path=manifest_path)
        )


def test_realtime_tau_residual_uses_delayed_center_q_time_difference() -> None:
    class Estimator:
        def estimate(self, q, dq, ddq, tau):
            tau_model = q + 10.0 * dq + 100.0 * ddq
            return JointTorqueResidualEstimate(
                tau_id=q,
                tau_friction=10.0 * dq,
                tau_bias=100.0 * ddq,
                tau_model=tau_model,
                tau_residual=tau_model - tau,
            )

    stream = CenteredThreePointTorqueResidualStream(Estimator())
    q0 = np.zeros(7, dtype=np.float64)
    q1 = np.ones(7, dtype=np.float64)
    q2 = np.full(7, 3.0, dtype=np.float64)
    tau = np.full(7, 0.2, dtype=np.float64)

    assert stream.append(1_000_000, q0, tau) is None
    assert stream.append(1_010_000, q1, tau) is None
    residual = stream.append(1_030_000, q2, tau)

    assert residual is not None
    assert residual.timestamp_us == 1_010_000
    assert residual.q == pytest.approx(q1)
    assert residual.dq == pytest.approx(np.full(7, 100.0))
    assert residual.ddq == pytest.approx(np.zeros(7))
    assert residual.estimate.tau_residual == pytest.approx(q1 + 1000.0 - tau)


def test_realtime_tau_residual_does_not_filter_centered_dq_or_ddq() -> None:
    class OffsetFilter:
        def __init__(self, offset: float) -> None:
            self.offset = offset
            self.seen: list[np.ndarray] = []

        def apply(self, value, timestamp_us):
            self.seen.append(np.asarray(value, dtype=np.float64).copy())
            return np.asarray(value, dtype=np.float64) + self.offset

    class Estimator:
        def __init__(self) -> None:
            self.dq: np.ndarray | None = None
            self.ddq: np.ndarray | None = None

        def estimate(self, q, dq, ddq, tau):
            self.dq = dq.copy()
            self.ddq = ddq.copy()
            tau_model = dq + ddq
            return JointTorqueResidualEstimate(
                tau_id=q,
                tau_friction=dq,
                tau_bias=ddq,
                tau_model=tau_model,
                tau_residual=tau_model - tau,
            )

    tau_filter = OffsetFilter(10.0)
    estimator = Estimator()
    stream = CenteredThreePointTorqueResidualStream(
        estimator,
        tau_filter=tau_filter,
    )
    tau = np.zeros(7, dtype=np.float64)

    assert stream.append(1_000_000, np.zeros(7), tau) is None
    assert stream.append(2_000_000, np.ones(7), tau) is None
    residual = stream.append(3_000_000, np.full(7, 3.0), tau)

    assert residual is not None
    assert len(tau_filter.seen) == 1
    assert tau_filter.seen[0] == pytest.approx(tau)
    assert residual.dq == pytest.approx(np.full(7, 1.5))
    assert residual.ddq == pytest.approx(np.ones(7))
    assert estimator.dq == pytest.approx(residual.dq)
    assert estimator.ddq == pytest.approx(residual.ddq)


def test_realtime_tau_residual_filters_tau_id_before_subtracting_tau() -> None:
    class OffsetFilter:
        def apply(self, value, timestamp_us):
            return np.asarray(value, dtype=np.float64) + 5.0

    class Estimator:
        def estimate(self, q, dq, ddq, tau):
            tau_id = np.full(7, 2.0, dtype=np.float64)
            zeros = np.zeros(7, dtype=np.float64)
            return JointTorqueResidualEstimate(
                tau_id=tau_id,
                tau_friction=zeros,
                tau_bias=zeros,
                tau_model=tau_id,
                tau_residual=tau_id - tau,
            )

    stream = CenteredThreePointTorqueResidualStream(
        Estimator(),
        tau_id_filter=OffsetFilter(),
    )
    tau = np.ones(7, dtype=np.float64)

    assert stream.append(1_000_000, np.zeros(7), tau) is None
    assert stream.append(2_000_000, np.ones(7), tau) is None
    residual = stream.append(3_000_000, np.full(7, 2.0), tau)

    assert residual is not None
    assert residual.estimate.tau_id == pytest.approx(np.full(7, 7.0))
    assert residual.estimate.tau_model == pytest.approx(np.full(7, 7.0))
    assert residual.estimate.tau_residual == pytest.approx(np.full(7, 6.0))


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_realtime_plot_process_accepts_sample_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    enabled_state = StateParamConfig(enabled=True)
    plotter = RealtimeJointPlotter(
        RealtimePlotConfig(enabled=True, window_s=10.0, update_rate_hz=20.0),
        {
            "q": enabled_state,
            "velocity": enabled_state,
            "acceleration": enabled_state,
            "torque": enabled_state,
        },
    )
    values = {
        "q_follower": ("q", np.linspace(-0.5, 0.5, 7)),
        "tau_ext_cal": ("torque", np.arange(7, dtype=np.float64)),
        "tau_ext_pred": ("torque", np.arange(7, dtype=np.float64) * 2.0),
    }

    plotter.start()
    process = plotter._process
    assert process is not None
    plotter.append(1_000_000, values)
    plotter.clear_history()
    plotter.append(2_000_000, values)
    plotter.close()

    assert process.exitcode == 0
