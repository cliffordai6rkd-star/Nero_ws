from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference.config import (
    CheckpointConfig,
    ExecutionConfig,
    InferenceConfig,
    PredictorConfig,
    RobotConfig,
    RuntimeConfig,
    load_inference_config,
)
from inference.pipeline import InferenceInput
from inference.swm_pipeline import SWMInferencePipeline
from nero_collection.control import DynamicsSnapshot, OSCQPConfig


class _DP(torch.nn.Module):
    n_obs_steps = 1
    image_key = "wrist"
    image_keys = ("wrist",)
    horizon = 8

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict_action(self, _obs):
        action = torch.full((1, 8, 7), 0.2)
        return {"action": action, "action_target": action[:, 0]}


class _SWM(torch.nn.Module):
    history_horizon = 5
    future_horizon = 4
    action_condition_horizon = 8
    SUPPORTED_INPUTS = ("q", "dq", "delta_q", "tau")
    state_contract = "q_dq_delta_q_tau"
    flow_inference_steps = 2
    flow_solver = "euler"

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self._inference_checkpoint_config = {
            "dataloader": {
                "high_fps": 100,
                "normalize_mode": "gaussian",
                "normalize_lowdim_keys": ["q", "dq", "delta_q", "tau", "action"],
            },
            "model": {"inputs": list(self.SUPPORTED_INPUTS)},
        }
        self._inference_normalizer = {
            "normalize_mode": "gaussian",
            "normalize_lowdim_keys": ["q", "dq", "delta_q", "tau", "action"],
            "stats": {
                key: {"mean": [0.0] * 7, "std": [1.0] * 7}
                for key in ("q", "dq", "delta_q", "tau", "action")
            },
        }

    def predict(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        assert set(batch) == {"q", "dq", "delta_q", "tau", "action", "action_mask"}
        for key in ("q", "dq", "delta_q", "tau"):
            assert tuple(batch[key].shape) == (1, 5, 7)
        assert tuple(batch["action"].shape) == (1, 8, 7)
        assert tuple(batch["action_mask"].shape) == (1, 8)
        future = torch.full((1, 4, 7), 0.1)
        return {"state_pred": {key: future for key in self.SUPPORTED_INPUTS}}


class _Model:
    position_lower = np.full(7, -2.0)
    position_upper = np.full(7, 2.0)
    velocity_limit = np.full(7, 10.0)
    effort_limit = np.full(7, 20.0)

    def snapshot(self, q, dq):
        del q, dq
        return DynamicsSnapshot(
            np.eye(7), np.zeros(7), np.zeros((6, 7)), np.zeros(6), np.eye(4)
        )


class _Controller:
    def __init__(self):
        self.model = _Model()
        self.config = OSCQPConfig(horizon_steps=2, dt_s=0.01)


def _sample(timestamp: float, q_cmd: np.ndarray) -> InferenceInput:
    return InferenceInput(
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        tau=np.zeros(7),
        q_cmd=q_cmd,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        wrench_ext=np.zeros(6),
        timestamp_s=timestamp,
    )


def test_swm_uses_training_state_and_action_contract(tmp_path: Path):
    config = InferenceConfig(
        dp_checkpoint=CheckpointConfig(tmp_path / "dp.pt", device="cpu"),
        pinn_checkpoint=CheckpointConfig(tmp_path / "swm.pt", device="cpu"),
        robot=RobotConfig(tmp_path / "robot.urdf"),
        runtime=RuntimeConfig(tmp_path / "collection.yaml"),
        action="joint",
        predictor=PredictorConfig(mode="swm"),
        execution=ExecutionConfig(mode="q"),
        osc_qp=OSCQPConfig(horizon_steps=2, dt_s=0.01),
    )
    swm = _SWM()
    pipeline = SWMInferencePipeline(
        config,
        dp_model=_DP(),
        pinn_model=swm,
        controller=_Controller(),
    )
    output = pipeline.step(_sample(0.0, np.full(7, 0.3)))
    assert output.control_mode == "q"
    assert output.joint_position_command is not None
    np.testing.assert_allclose(swm.calls[0][0]["delta_q"][0, -1].numpy(), 0.3, atol=1e-6)
    assert swm.calls[0][1] == {"steps": 2, "solver": "euler"}


def test_swm_config_aliases_are_canonicalized(tmp_path: Path):
    path = tmp_path / "inference.yaml"
    path.write_text(
        """
dp_checkpoint: {path: dp.ckpt, device: cpu}
pinn_checkpoint: {path: swm.ckpt, device: cpu}
predictor: {mode: torque_world_model_opd}
action: joint
robot: {urdf_path: robot.urdf}
runtime: {collection_config: collection.yaml}
""",
        encoding="utf-8",
    )
    assert load_inference_config(path).predictor.mode == "swm_opd"
