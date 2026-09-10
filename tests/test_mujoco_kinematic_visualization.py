from dataclasses import replace
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PINN_ROOT = Path(__file__).resolve().parents[2] / "PINN"
if PINN_ROOT.is_dir() and str(PINN_ROOT) not in sys.path:
    sys.path.insert(0, str(PINN_ROOT))

from inference.config import load_inference_config
from inference.mujoco_visualization import (
    MujocoKinematicFK,
    MujocoKinematicVisualizer,
    VisualizationPacket,
    _put_latest,
)


def test_contact_wm_batch_sampler_shape_without_repeated_condition_encoding():
    torch = pytest.importorskip("torch")
    try:
        from model.pinn_model.contact_world_model import ContactWorldModel
    except ImportError:
        pytest.skip("PINN checkout is not importable")
    from inference.contact_wm_pipeline import ContactWMInferencePipeline

    config = {
        "dataloader": {"state_history_horizon": 4, "prediction_horizon": 3, "action_condition_horizon": 2, "high_fps": 100, "expert_fps": 25},
        "model": {"inputs": ["q", "dq", "delta_q", "tau"], "outputs": ["q", "tau"], "joint_dim": 7, "action_dim": 7, "hidden_dim": 8, "state_layers": 1, "action_layers": 1, "flow_layers": 1, "flow_attention_heads": 2, "flow_ffn_multiplier": 2, "flow_inference_steps": 1, "flow_solver": "euler", "flow_source_mode": "gaussian", "state_pooling": "last", "dropout": 0.0, "state_to_action_attention_heads": 2, "runtime_checks": False, "use_action_padding_mask": False, "emit_contact_probabilities": False},
    }
    model = ContactWorldModel(config).eval()
    pipeline = ContactWMInferencePipeline.__new__(ContactWMInferencePipeline)
    pipeline.pinn = model
    pipeline._contact_input_keys = ("q", "dq", "delta_q", "tau")
    pipeline._contact_history_horizon = 4
    pipeline._contact_future_horizon = 3
    pipeline._contact_action_horizon = 2
    pipeline._contact_flow_steps = 1
    pipeline._contact_flow_solver = "euler"
    pipeline._actions_for_contact_wm = lambda value: value
    encode_calls = {"count": 0}
    original_encode = model.encode_conditions
    def counted_encode(batch):
        encode_calls["count"] += 1
        return original_encode(batch)
    model.encode_conditions = counted_encode

    history = {key: np.zeros((4, 7), dtype=np.float32) for key in pipeline._contact_input_keys}
    output = pipeline.sample_contact_futures(history, np.zeros((2, 7), dtype=np.float32), num_samples=5)
    assert output["q"].shape == (5, 3, 7)
    assert output["tau"].shape == (5, 3, 7)
    assert encode_calls["count"] == 1


def test_visualization_packet_contract_and_latest_only_queue():
    playback_q = np.arange(7, dtype=np.float64)
    packet = VisualizationPacket(
        1.0,
        np.zeros(7),
        np.zeros((3, 5, 7)),
        np.zeros((3, 5, 3)),
        prediction_id=4,
        playback_q=playback_q,
    )
    assert packet.predicted_q.shape == (3, 5, 7)
    assert packet.predicted_ee_position.shape == (3, 5, 3)
    np.testing.assert_array_equal(packet.playback_q, playback_q)
    channel = queue.Queue(maxsize=1)
    _put_latest(channel, "old")
    _put_latest(channel, "new")
    assert channel.get_nowait() == "new"


def test_visualization_packet_rejects_wrong_playback_joint_dimension():
    with pytest.raises(ValueError, match="same joint dimension"):
        VisualizationPacket(1.0, np.zeros(7), playback_q=np.zeros(6))


def test_prediction_keeps_newest_observed_pose_and_rejects_old_observation():
    config = SimpleNamespace(
        enabled=True,
        robot_joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        prediction_visualization_hz=1_000.0,
    )
    visualizer = MujocoKinematicVisualizer(config)
    try:
        newest = np.full(7, 2.0)
        visualizer.publish_observed(10.0, newest)
        visualizer.publish_observed(9.0, np.zeros(7))
        visualizer.publish_prediction(
            9.5,
            np.zeros(7),
            np.zeros((2, 3, 7)),
            prediction_id=1,
        )
        assert visualizer._latest_observed_timestamp == 10.0
        np.testing.assert_allclose(visualizer._latest_prediction.observed_q, newest)
        assert visualizer._latest_prediction.timestamp == 10.0
    finally:
        visualizer.close()


def test_mujoco_fk_uses_explicit_joint_names_and_scratch_data():
    mujoco = __import__("pytest").importorskip("mujoco")
    config = load_inference_config("inference/configs/nero_contact_wm.yaml").mujoco_visualization
    config = replace(config, enabled=True, headless=True)
    fk = MujocoKinematicFK(config)
    observed = fk.set_observed_q(np.zeros(7))
    predicted = fk.predicted_ee_positions(np.zeros((2, 4, 7)))
    assert observed.shape == (3,)
    assert predicted.shape == (2, 4, 3)
    np.testing.assert_allclose(fk.display_data.qpos[fk.addresses], np.zeros(7))
    # Scratch FK must not overwrite the displayed observed pose.
    np.testing.assert_allclose(fk.set_observed_q(np.zeros(7)), observed)
