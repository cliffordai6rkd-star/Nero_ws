from __future__ import annotations

import numpy as np
import torch

from scripts.visualize_wm_lerobotv3 import _execution_step_counts, recursive_rollout, sample


class _RecursiveModel(torch.nn.Module):
    history_horizon = 50
    future_horizon = 32
    action_condition_horizon = 8
    flow_dim = 28
    inputs = ("q", "dq", "delta_q", "tau")

    def __init__(self):
        super().__init__()
        self.anchor_actions = []
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def encode_conditions(self, inputs):
        self.anchor_actions.append(inputs["action"].detach().cpu().numpy().copy())
        # Checkpoints may include scalar condition metadata.  The sampler
        # should leave 0-D tensors untouched while repeating batched values.
        return {**inputs, "scalar_metadata": torch.tensor(1.0, device=self.dummy.device)}

    def integrate_flow(self, source_noise, encoded, **_kwargs):
        return source_noise

    def _decoded_output(self, _flow_state, encoded):
        batch = encoded["q"].shape[0]
        device = encoded["q"].device
        base = encoded["q"][:, -1:, :1]
        offsets = torch.arange(1, self.future_horizon + 1, device=device).reshape(1, -1, 1)
        stream = base + offsets
        stream = stream.expand(batch, self.future_horizon, 7)
        return {
            f"{key}_pred": stream.clone()
            for key in ("q", "dq", "delta_q", "tau")
        }


class _LegacyModel(_RecursiveModel):
    outputs = ("q", "tau")

    def _decoded_output(self, _flow_state, encoded):
        batch = encoded["q"].shape[0]
        device = encoded["q"].device
        base = encoded["q"][:, -1:, :1]
        offsets = torch.arange(1, self.future_horizon + 1, device=device).reshape(1, -1, 1)
        stream = (base + offsets).expand(batch, self.future_horizon, 7)
        return {"q_pred": stream.clone(), "tau_pred": stream.clone()}


class _StridedLegacyModel(_LegacyModel):
    temporal_stride = 2
    external_history_horizon = 50
    external_future_horizon = 16
    external_action_condition_horizon = 16
    history_horizon = 25
    future_horizon = 8
    action_condition_horizon = 8


def test_recursive_rollout_commits_16_of_32_and_reanchors_actions():
    model = _RecursiveModel()
    count = 100
    arrays = {
        key: np.zeros((count, 7), dtype=np.float64)
        for key in ("q", "dq", "delta_q", "tau")
    }
    arrays["action"] = np.arange(count * 7, dtype=np.float64).reshape(count, 7)
    arrays["timestamp"] = np.arange(count, dtype=np.float64)
    q, _, timings = recursive_rollout(
        model,
        arrays,
        start=50,
        samples=2,
        segment_steps=16,
        segments=4,
        steps=1,
        solver="euler",
    )

    assert q.shape == (2, 64, 7)
    np.testing.assert_allclose(q[0, :, 0], np.arange(1.0, 65.0))
    np.testing.assert_allclose(q[1], q[0])
    assert len(timings) == 4
    assert len(model.anchor_actions) == 4
    np.testing.assert_allclose(model.anchor_actions[0][0, 0, 0], arrays["action"][50, 0])
    np.testing.assert_allclose(model.anchor_actions[1][0, 0, 0], arrays["action"][66, 0])


def test_legacy_q_tau_sampler_does_not_require_predicted_dq_or_delta_q():
    model = _LegacyModel()
    count = 100
    arrays = {
        key: np.zeros((count, 7), dtype=np.float64)
        for key in ("q", "dq", "delta_q", "tau")
    }
    arrays["action"] = np.zeros((count, 7), dtype=np.float64)
    arrays["timestamp"] = np.arange(count, dtype=np.float64)
    q, _, timing = sample(model, arrays, start=50, samples=1, steps=1, solver="euler")

    assert q.shape == (1, 32, 7)
    assert timing["flow_integration_N_ms"] >= 0


def test_strided_sampler_restores_external_rate_and_history():
    model = _StridedLegacyModel()
    count = 100
    arrays = {
        key: np.arange(count * 7, dtype=np.float64).reshape(count, 7)
        for key in ("q", "dq", "delta_q", "tau", "action")
    }
    arrays["timestamp"] = np.arange(count, dtype=np.float64) * 1e7
    q, _, _ = sample(model, arrays, start=50, samples=1, steps=1, solver="euler")
    assert q.shape == (1, 16, 7)
    # Internal prediction tokens are repeated to the 100 Hz external timeline.
    np.testing.assert_allclose(q[0, ::2, 0], q[0, 1::2, 0])


def test_execution_steps_are_internal_tokens_for_strided_checkpoint():
    model = _StridedLegacyModel()
    assert _execution_step_counts(model, 4, 16) == (4, 8)
    assert _execution_step_counts(model, 4, 5) == (4, 5)
