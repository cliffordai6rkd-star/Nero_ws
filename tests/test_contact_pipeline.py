from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
                "action_key": "action.ee_pose",
                "action_condition_mode": "direct",
                "action_condition_horizon": 8,
            },
            "model": {
                "inputs": ["q", "dq", "delta_q", "tau"],
                "action_dim": 7,
                "state_estimator": {"sampling_dt": 0.01},
            },
        }
        self._inference_normalizer = None

    def predict(self, inputs, **_kwargs):
        self.inputs.append(inputs)
        assert set(inputs) == {
            "q",
            "dq",
            "delta_q",
            "tau",
            "action",
            "action_mask",
        }
        assert inputs["q"].shape == (1, 5, 7)
        assert inputs["dq"].shape == (1, 5, 7)
        assert inputs["delta_q"].shape == (1, 5, 7)
        assert inputs["tau"].shape == (1, 5, 7)
        assert inputs["action"].shape == (1, 8, 7)
        assert inputs["action_mask"].shape == (1, 8)
        q = torch.full((1, 4, 7), 0.1)
        dq = torch.zeros((1, 4, 7))
        delta_q = torch.zeros((1, 4, 7))
        tau = torch.full((1, 4, 7), 0.2)
        contact = torch.ones((1, 4, 1))
        return {
            "state_pred": {
                "q": q,
                "dq": dq,
                "delta_q": delta_q,
                "tau": tau,
                "contact_state": contact,
            }
        }


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

    def gravity_torque(self, q):
        del q
        return np.zeros(7)


class _FKModel(_Model):
    def frame_pose(self, q, frame_name):
        assert frame_name == "link7"
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=np.float64)
        return pose


class _Controller:
    def __init__(self):
        self.model = _Model()
        self.config = SimpleNamespace(horizon_steps=2, dt_s=0.01)


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
            inference_mode="open_loop",
        ),
        execution=ExecutionConfig(
            mode=mode,
            mit_kp=(1.0,) * 7,
            mit_kd=(1.0,) * 7,
            mit_velocity_limit=(2.0,) * 7,
            mit_feedback_torque_limit=(5.0,) * 7,
        ),
        torque_filter=TorqueFilterConfig(enabled=False),
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


def test_contact_config_exposes_three_execution_modes(tmp_path: Path):
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(
        """
dp_checkpoint: {path: dp.pt, device: cpu}
pinn_checkpoint: {path: wm.pt, device: cpu}
predictor: {enabled: true, mode: contact_world_model_opd}
execution: {mode: mtc, mit_kp: 10, mit_kd: 1, mit_velocity_limit: 2}
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )
    config = load_inference_config(config_file)
    assert config.execution.mode == "mtc"
    assert config.execution.mit_kp == (10.0,) * 7
    assert config.execution.mit_velocity_limit == (2.0,) * 7


@pytest.mark.parametrize("mode", ["mtc", "q", "tau"])
def test_contact_pipeline_has_three_execution_modes(tmp_path: Path, mode: str):
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

    condition = wm.inputs[-1]["action"][0].detach().cpu().numpy()
    expected_translation = np.repeat([[0.1, 0.2, 0.3]], 8, axis=0)
    np.testing.assert_allclose(condition[:, :3], expected_translation)
    np.testing.assert_allclose(condition[:, 3:], [[0.0, 0.0, 0.0, 1.0]] * 8)
    pipeline.close()


def test_contact_pipeline_clamps_mtc_velocity_and_feedback(tmp_path: Path):
    pipeline = ContactWMInferencePipeline(
        _config(tmp_path, "mtc"),
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=_Controller(),
    )
    output = pipeline.step(_sample())
    # The q-only estimator's first derivative is bounded before MIT feedback.
    assert np.max(np.abs(output.dq_target)) <= 2.0 + 1.0e-8
    assert np.max(np.abs(output.tau_command - output.tau_target)) <= 5.0 + 1.0e-8
    pipeline.close()


def test_contact_pipeline_reports_configured_mtc_gains(tmp_path: Path):
    config = _config(tmp_path, "mtc")
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
    assert np.isfinite(firmware_feedback).all()
    np.testing.assert_allclose(output.mit_kp, config.execution.mit_kp)
    np.testing.assert_allclose(output.mit_kd, config.execution.mit_kd)
    pipeline.close()


def test_contact_pipeline_clips_mtc_total_torque(
    tmp_path: Path,
):
    config = _config(tmp_path, "mtc")
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
    assert np.max(np.abs(output.tau_target)) <= 0.2 + 1.0e-10
    assert np.max(np.abs(output.tau_command)) <= 0.2 + 1.0e-10
    pipeline.close()


def test_contact_mtc_filters_final_blended_torque(tmp_path: Path):
    config = replace(
        _config(tmp_path, "mtc"),
        torque_filter=TorqueFilterConfig(
            enabled=True,
            median_window=1,
            lowpass_cutoff_hz=1.0,
            rate_limit_nm_s=None,
        ),
    )
    pipeline = ContactWMInferencePipeline(
        config,
        dp_model=_DP(),
        pinn_model=_ContactWM(),
        controller=_Controller(),
    )
    output = pipeline.step(_sample())

    # q/v and WM each contribute 0.1 before filtering.  The filter must see
    # their blended total (0.15), not WM tau (0.2) on its own.
    filter_alpha = 1.0 - np.exp(-2.0 * np.pi * 1.0 * 0.01)
    expected_total = 0.15 * filter_alpha
    np.testing.assert_allclose(output.tau_unfiltered, 0.15, atol=1.0e-8)
    np.testing.assert_allclose(output.tau_command, expected_total, atol=1.0e-8)
    np.testing.assert_allclose(output.tau_target, expected_total - 0.1, atol=1.0e-8)
    pipeline.close()
