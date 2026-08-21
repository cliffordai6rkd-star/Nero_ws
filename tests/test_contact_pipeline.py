from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference.config import (  # noqa: E402
    CheckpointConfig,
    ExecutionConfig,
    InferenceConfig,
    PredictorConfig,
    RobotConfig,
    RuntimeConfig,
    TorqueFilterConfig,
    load_inference_config,
)
from inference.contact_pipeline import ContactWMInferencePipeline  # noqa: E402
from inference.pipeline import InferenceInput  # noqa: E402
from nero_collection.control import (  # noqa: E402
    DynamicsSnapshot,
    OSCQPConfig,
    OSCQPResult,
)


class _DP(torch.nn.Module):
    n_obs_steps = 1
    image_key = "wrist"
    wrench_key = "wrench_ext"
    horizon = 8

    class _Encoder:
        wrench_history_steps = 1

    obs_encoder = _Encoder()

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict_action(self, _obs):
        action = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] * 8]
        )
        return {"action": action, "action_target": action[:, 0]}


class _JointDP(_DP):
    def predict_action(self, _obs):
        action = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] * 8]
        )
        return {"action": action, "action_target": action[:, 0]}


class _ContactWM(torch.nn.Module):
    history_horizon = 5
    future_horizon = 4
    action_condition_horizon = 8
    sampling_dt = 0.01

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []
        self._inference_checkpoint_config = {
            "dataloader": {
                "action_condition_features": [
                    "absolute_pose",
                    "current_ee_pose",
                    "relative_pose",
                ],
                "action_condition_mode": "composite",
            },
            "model": {
                "state_contract": "q_tau_contact",
                "state_estimator": {"sampling_dt": 0.01},
            },
        }
        self._inference_normalizer = None

    def predict(self, inputs):
        self.inputs.append(inputs)
        assert set(inputs) == {
            "q",
            "tau",
            "target_relative_pose",
        }
        assert inputs["q"].shape == (1, 5, 7)
        assert inputs["tau"].shape == (1, 5, 7)
        assert inputs["target_relative_pose"].shape == (1, 8, 21)
        q = torch.full((1, 4, 7), 0.1)
        tau = torch.full((1, 4, 7), 0.2)
        contact = torch.ones((1, 4, 1))
        return {"state_pred": {"q": q, "tau": tau, "contact_state": contact}}


class _Model:
    dof = 7
    position_lower = np.full(7, -2.0)
    position_upper = np.full(7, 2.0)
    velocity_limit = np.full(7, 10.0)
    effort_limit = np.full(7, 20.0)

    def snapshot(self, q, dq):
        del q, dq
        return DynamicsSnapshot(
            np.eye(7), np.zeros(7), np.zeros((6, 7)), np.zeros(6), np.eye(4)
        )


class _FKModel(_Model):
    def frame_pose(self, q, frame_name):
        assert frame_name == "link7"
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=np.float64)
        return pose


class _Controller:
    def __init__(self):
        self.model = _Model()
        self.config = OSCQPConfig(horizon_steps=2, dt_s=0.01)
        self.targets = []

    def optimize_mpc(self, q, dq, target, **_kwargs):
        del q, dq
        self.targets.append(target)
        return OSCQPResult(
            tau=np.ones((2, 7)),
            joint_accelerations=np.zeros((2, 7)),
            predicted_q=np.zeros((2, 7)),
            predicted_dq=np.zeros((2, 7)),
            predicted_wrenches=np.zeros((2, 6)),
            status="solved",
            iterations=1,
            solve_time_s=0.0,
            objective=0.0,
            max_constraint_violation=0.0,
        )


def _config(tmp_path: Path, mode: str) -> InferenceConfig:
    checkpoint = CheckpointConfig(tmp_path / "unused.pt", device="cpu")
    return InferenceConfig(
        dp_checkpoint=checkpoint,
        pinn_checkpoint=checkpoint,
        robot=RobotConfig(tmp_path / "unused.urdf"),
        runtime=RuntimeConfig(tmp_path / "collection.yaml"),
        predictor=PredictorConfig(
            enabled=True,
            mode="contact_world_model_opd",
        ),
        execution=ExecutionConfig(
            mode=mode,
            mit_kp=(1.0,) * 7,
            mit_kd=(1.0,) * 7,
            mit_velocity_limit=(2.0,) * 7,
            mit_feedback_torque_limit=(5.0,) * 7,
        ),
        torque_filter=TorqueFilterConfig(enabled=False),
        osc_qp=OSCQPConfig(horizon_steps=2, dt_s=0.01),
    )


def _sample(timestamp=0.0):
    return InferenceInput(
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        wrench_ext=np.zeros(6),
        timestamp_s=timestamp,
    )


def test_contact_config_exposes_four_execution_modes(tmp_path: Path):
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.pt, device: cpu}
pinn_checkpoint: {path: wm.pt, device: cpu}
predictor: {enabled: true, mode: contact_world_model_opd}
execution: {mode: osc-qp, mit_kp: 10, mit_kd: 1, mit_velocity_limit: 2}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )
    config = load_inference_config(config_file)
    assert config.execution.mode == "osc_qp"
    assert config.execution.mit_kp == (10.0,) * 7
    assert config.execution.mit_velocity_limit == (2.0,) * 7


@pytest.mark.parametrize("mode", ["mit", "q", "tau", "osc_qp"])
def test_contact_pipeline_has_four_execution_modes(tmp_path: Path, mode: str):
    controller = _Controller()
    pipeline = ContactWMInferencePipeline(
        _config(tmp_path, mode),
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=controller,
    )
    output = pipeline.step(_sample())
    assert output.control_mode == mode
    assert output.q_target.shape == (7,)
    assert output.dq_target.shape == (7,)
    assert output.tau_target.shape == (7,)
    if mode == "q":
        assert output.joint_position_command is not None
    elif mode == "osc_qp":
        assert output.qp_result is not None
        assert controller.targets[-1].joint_accelerations is not None
        assert controller.targets[-1].joint_torques is not None
    else:
        assert output.joint_position_command is None
    pipeline.close()


def test_joint_dp_actions_are_fk_converted_before_contact_wm(tmp_path: Path):
    config = replace(
        _config(tmp_path, "q"),
        action="joint",
        predictor=PredictorConfig(
            enabled=True,
            mode="contact_world_model_opd",
            inference_mode="open_loop",
        ),
        robot=RobotConfig(
            tmp_path / "unused.urdf",
            frame_name="gripper_tcp",
            action_frame_name="link7",
        ),
    )
    controller = _Controller()
    controller.model = _FKModel()
    wm = _ContactWM()
    pipeline = ContactWMInferencePipeline(
        config,
        dp_model=_JointDP(),
        pinn_model=wm,
        controller=controller,
    )

    pipeline.step(_sample())

    condition = wm.inputs[-1]["target_relative_pose"][0].detach().cpu().numpy()
    expected_translation = np.repeat([[0.1, 0.2, 0.3]], 8, axis=0)
    np.testing.assert_allclose(condition[:, :3], expected_translation)
    np.testing.assert_allclose(condition[:, 7:10], 0.0)
    np.testing.assert_allclose(condition[:, 14:17], expected_translation)
    pipeline.close()


def test_contact_pipeline_clamps_mit_velocity_and_feedback(tmp_path: Path):
    pipeline = ContactWMInferencePipeline(
        _config(tmp_path, "mit"),
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=_Controller(),
    )
    output = pipeline.step(_sample())
    # The q-only estimator's first derivative is bounded before MIT feedback.
    assert np.max(np.abs(output.dq_target)) <= 2.0 + 1.0e-8
    assert np.max(np.abs(output.tau_command - output.tau_target)) <= 5.0 + 1.0e-8
    pipeline.close()


def test_contact_pipeline_applies_feedback_limit_to_firmware_gains(tmp_path: Path):
    config = _config(tmp_path, "mit")
    config = replace(
        config,
        execution=replace(
            config.execution,
            mit_kp=(100.0,) * 7,
            mit_kd=(10.0,) * 7,
            mit_feedback_torque_limit=(0.05,) * 7,
        ),
    )
    pipeline = ContactWMInferencePipeline(
        config,
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=_Controller(),
    )
    sample = _sample()
    output = pipeline.step(sample)

    firmware_feedback = output.mit_kp * (output.q_target - sample.q) + output.mit_kd * (
        output.dq_target - sample.dq
    )
    np.testing.assert_allclose(
        firmware_feedback,
        output.tau_command - output.tau_target,
        atol=1.0e-10,
    )
    assert np.max(np.abs(firmware_feedback)) <= 0.05 + 1.0e-10
    assert np.any(output.mit_kp < np.asarray(config.execution.mit_kp))
    pipeline.close()


def test_contact_pipeline_zeroes_feedback_when_feedforward_uses_torque_cap(
    tmp_path: Path,
):
    config = _config(tmp_path, "mit")
    config = replace(
        config,
        safety=replace(config.safety, maximum_command_torque_nm=(0.2,) * 7),
        execution=replace(
            config.execution,
            mit_kp=(100.0,) * 7,
            mit_kd=(10.0,) * 7,
            mit_feedback_torque_limit=(5.0,) * 7,
        ),
    )
    pipeline = ContactWMInferencePipeline(
        config,
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=_Controller(),
    )
    output = pipeline.step(_sample())
    np.testing.assert_allclose(output.tau_target, 0.2)
    np.testing.assert_allclose(output.mit_kp, 0.0)
    np.testing.assert_allclose(output.mit_kd, 0.0)
    np.testing.assert_allclose(output.tau_command, output.tau_target)
    pipeline.close()
