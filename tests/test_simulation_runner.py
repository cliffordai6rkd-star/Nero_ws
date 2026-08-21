from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from calibration.dynamics_common import load_dynamics_plan
from inference.h5_observation_stream import H5ObservationEpisode, H5ObservationStream
from inference.mujoco_backend import MujocoBackendConfig, MujocoDynamicsBackend
from inference.simulation_runner import (
    SimulationRunnerConfig,
    run_h5_simulation,
)


ROOT = Path(__file__).resolve().parents[1]


def _stream() -> tuple[H5ObservationStream, np.ndarray]:
    count = 5
    timestamps = 1_000_000 + np.arange(count, dtype=np.int64) * 10_000
    q0 = np.array([0.0, -0.3, 0.3, 1.5, 0.0, 0.0, 0.0])
    q = np.repeat(q0[None], count, axis=0)
    episode = H5ObservationEpisode.from_arrays(
        state_timestamp_us=timestamps,
        q=q,
        dq=np.zeros_like(q),
        ddq=np.zeros_like(q),
        tau=np.zeros_like(q),
        wrench_ext=np.zeros((count, 6)),
        camera_timestamp_us=timestamps,
        frames=np.zeros((count, 3, 4, 3), dtype=np.uint8),
    )
    return (
        H5ObservationStream(
            episode,
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            state_alignment="previous",
        ),
        q0,
    )


class _FakePipeline:
    def __init__(self, mode: str, *, hybrid: bool = False) -> None:
        self.config = SimpleNamespace(
            predictor=SimpleNamespace(enabled=True),
            execution=SimpleNamespace(mode=mode),
        )
        self.mode = mode
        self.samples = []
        self.hybrid = hybrid

    def reset(self) -> None:
        self.samples.clear()

    def close(self) -> None:
        pass

    def step(self, sample):
        self.samples.append(sample)
        common = dict(
            control_mode=self.mode,
            action_target=np.zeros(7),
            dp_updated=True,
            pinn_updated=True,
        )
        if self.mode == "q":
            return SimpleNamespace(
                **common,
                joint_position_target=sample.q + 0.01,
                tau_command=np.zeros(7),
            )
        if self.mode == "mit":
            return SimpleNamespace(
                **common,
                joint_position_target=sample.q + 0.01,
                joint_velocity_target=np.zeros(7),
                torque_target=np.full(7, 0.1),
                tau_command=np.full(7, 0.1),
                mit_kp=np.full(7, 2.0),
                mit_kd=np.full(7, 0.2),
            )
        return SimpleNamespace(**common, tau_command=np.full(7, 0.1))


def _backend(q0: np.ndarray) -> MujocoDynamicsBackend:
    plan = load_dynamics_plan(ROOT / "calibration/config.yaml")
    return MujocoDynamicsBackend(
        plan,
        q0,
        config=MujocoBackendConfig(
            physics_dt_s=0.001,
            control_dt_s=0.01,
            mit_kp=2.0,
            mit_kd=0.2,
            q_kp=20.0,
            q_kd=2.0,
            torque_limits_nm=1.0,
        ),
    )


@pytest.mark.parametrize("mode", ["q", "mit", "tau", "osc_qp"])
def test_runner_routes_all_four_modes_through_dynamic_backend(mode: str) -> None:
    stream, q0 = _stream()
    pipeline = _FakePipeline(mode)
    backend = _backend(q0)
    result = run_h5_simulation(
        stream,
        pipeline,
        backend,
        config=SimulationRunnerConfig(
            execution_mode=mode,
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            state_alignment="previous",
        ),
    )

    assert result.sample_count == len(stream)
    assert result.execution_mode == mode
    assert result.simulated_q.shape == (len(stream), 7)
    assert np.isfinite(result.simulated_q).all()
    assert np.isfinite(result.command_torque).all()
    assert np.any(np.abs(result.simulated_q[-1] - result.simulated_q[0]) > 1.0e-12)
    if mode in {"tau", "osc_qp"}:
        np.testing.assert_allclose(result.command_torque, 0.1, atol=1.0e-12)


def test_hybrid_mode_feeds_previous_simulation_state_back_to_pipeline() -> None:
    stream, q0 = _stream()
    pipeline = _FakePipeline("tau")
    backend = _backend(q0)
    result = run_h5_simulation(
        stream,
        pipeline,
        backend,
        config=SimulationRunnerConfig(
            observation_mode="hybrid_closed_loop",
            execution_mode="tau",
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            state_alignment="previous",
        ),
    )

    assert len(pipeline.samples) == result.sample_count
    np.testing.assert_allclose(pipeline.samples[0].q, q0)
    np.testing.assert_allclose(
        pipeline.samples[1].q,
        result.simulated_q[0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(pipeline.samples[0].tau, stream[0].tau)
    np.testing.assert_allclose(pipeline.samples[1].tau, result.applied_torque[0])
    np.testing.assert_allclose(
        pipeline.samples[1].ddq,
        (result.simulated_dq[0] - np.zeros(7)) / 0.01,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        pipeline.samples[2].ddq,
        (result.simulated_dq[1] - result.simulated_dq[0]) / 0.01,
        atol=1.0e-12,
    )
    assert not np.allclose(pipeline.samples[1].q, stream[1].q)


def test_runner_uses_explicit_history_for_direct_ik_contract() -> None:
    stream, q0 = _stream()

    class DirectIKPipeline(_FakePipeline):
        def __init__(self):
            super().__init__("q")
            self.config = SimpleNamespace(
                predictor=SimpleNamespace(enabled=False),
                execution=SimpleNamespace(mode="q"),
            )
            self.history_shapes = []

        def step_direct_ik_observation_history(self, sample, images, wrenches):
            self.history_shapes.append((images.shape, wrenches.shape))
            return SimpleNamespace(
                control_mode="q",
                joint_position_command=sample.q,
                action_target=np.zeros(7),
                tau_command=np.zeros(7),
                dp_updated=True,
                pinn_updated=False,
            )

    pipeline = DirectIKPipeline()
    result = run_h5_simulation(
        stream,
        pipeline,
        _backend(q0),
        config=SimulationRunnerConfig(
            execution_mode="q",
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            state_alignment="previous",
        ),
    )
    assert result.sample_count == len(stream)
    assert pipeline.history_shapes == [((1, 3, 4, 3), (1, 1, 6))] * len(stream)


def test_predictor_disabled_defaults_to_q_even_when_legacy_execution_default_is_osc_qp() -> None:
    stream, q0 = _stream()

    class LegacyDirectPipeline(_FakePipeline):
        def __init__(self):
            super().__init__("q")
            self.config = SimpleNamespace(
                predictor=SimpleNamespace(enabled=False),
                execution=SimpleNamespace(mode="osc_qp"),
            )

        def step_direct_ik_observation_history(self, sample, images, wrenches):
            return SimpleNamespace(
                joint_position_command=sample.q + 0.01,
                action_target=np.zeros(7),
                tau_command=np.zeros(7),
                dp_updated=True,
                pinn_updated=False,
            )

    pipeline = LegacyDirectPipeline()
    result = run_h5_simulation(
        stream,
        pipeline,
        _backend(q0),
        config=SimulationRunnerConfig(
            state_rate_hz=100.0,
            history_steps=1,
            camera_history_steps=1,
            state_alignment="previous",
        ),
    )
    assert result.execution_mode == "q"
    assert np.any(np.abs(result.command_torque) > 0.0)
