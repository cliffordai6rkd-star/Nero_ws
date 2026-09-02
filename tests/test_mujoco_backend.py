from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from calibration.dynamics_common import load_dynamics_plan
from inference.mujoco_backend import (
    MujocoBackendConfig,
    MujocoCommand,
    MujocoDynamicsBackend,
)


ROOT = Path(__file__).resolve().parents[1]


def _backend(tmp_path: Path) -> MujocoDynamicsBackend:
    plan = load_dynamics_plan(ROOT / "calibration/config.yaml")
    return MujocoDynamicsBackend(
        plan,
        np.array([0.0, -0.3, 0.3, 1.5, 0.0, 0.0, 0.0]),
        scene_path=tmp_path / "sim.scene.xml",
        config=MujocoBackendConfig(
            physics_dt_s=0.001,
            control_dt_s=0.005,
            mit_kp=10.0,
            mit_kd=1.0,
            q_kp=20.0,
            q_kd=2.0,
        ),
    )


def test_backend_injects_seven_torque_actuators_and_writes_scene(tmp_path: Path) -> None:
    backend = _backend(tmp_path)

    assert backend.model.nu == 7
    assert backend.actuator_ids.tolist() == list(range(7))
    assert backend.scene_path == (tmp_path / "sim.scene.xml").resolve()
    assert backend.scene_path.is_file()
    assert [backend.mujoco.mj_id2name(backend.model, backend.mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in backend.actuator_ids] == list(backend.actuator_names)


def test_tau_mode_advances_dynamics_and_reports_applied_torque(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    initial = backend.state()

    result = backend.step("tau", tau_command=np.ones(7))

    assert result.timestamp_s == pytest.approx(0.005)
    assert np.max(np.abs(result.dq)) > 0.0
    assert np.allclose(result.command_torque, np.ones(7))
    assert np.allclose(result.applied_torque, np.ones(7), atol=1e-12)
    assert not np.allclose(result.q, initial.q)


def test_q_and_mtc_modes_use_current_simulation_state(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    q_target = backend.q + np.array([0.1, 0, 0, 0, 0, 0, 0])

    q_result = backend.step(MujocoCommand(mode="q", q_target=q_target))
    assert q_result.command_torque[0] > 0.0

    mtc_result = backend.step(
        "mtc",
        q_target=q_target,
        dq_target=np.zeros(7),
        tau_target=np.zeros(7),
    )
    assert mtc_result.command_torque[0] > 0.0


def test_legacy_position_output_without_control_mode_is_inferred_as_q(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    output = type(
        "LegacyOutput",
        (),
        {"joint_position_command": backend.q + 0.05},
    )()
    result = backend.step_output(output)
    assert result.command_torque[0] > 0.0
