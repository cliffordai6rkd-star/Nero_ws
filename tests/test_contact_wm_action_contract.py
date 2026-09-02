from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.config import load_inference_config
from inference.contact_wm_pipeline import ContactWMInferencePipeline


def test_contact_wm_action_contract_fk_converts_joint_and_preserves_ee_pose():
    pipeline = object.__new__(ContactWMInferencePipeline)
    pipeline._action_frame_name = "link7"
    pipeline._control_frame_name = "link7"

    class Model:
        @staticmethod
        def frame_pose(q, frame_name):
            assert frame_name == "link7"
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = np.asarray(q[:3], dtype=np.float64)
            return pose

    pipeline.model = Model()

    pipeline._dp_action_type = "joint"
    joint = pipeline._actions_for_contact_wm(
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    )
    np.testing.assert_allclose(joint, [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])

    pipeline._dp_action_type = "eepose"
    ee_pose = np.asarray([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(pipeline._actions_for_contact_wm(ee_pose), ee_pose)


def test_deployed_contact_wm_uses_recorded_tcp_urdf_frame():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "inference"
        / "configs"
        / "nero_contact_wm.yaml"
    )
    config = load_inference_config(config_path)
    assert config.robot.frame_name == "gripper_tcp"
    assert config.robot.action_frame_name == "link7"
