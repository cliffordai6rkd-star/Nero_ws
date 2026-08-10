from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.world_model import WorldModelWrenchAdapter
from nero_collection.filters import OnePoleLowPass
from nero_collection.state_alignment import (
    _JointGroupDerivativeState,
    _append_filtered_joint_state,
)


class _TauFPredictor:
    metadata = SimpleNamespace(
        horizon=2,
        input_keys=("q", "dq", "delta_q"),
    )

    def __init__(self) -> None:
        self.inputs = None

    def predict_sequence(self, inputs):
        self.inputs = inputs
        length = inputs["q"].shape[0]
        return np.full((length, 7), 0.5)


class _InverseDynamics:
    def __init__(self) -> None:
        self.calls = []

    def estimate(self, q, dq, ddq, tau):
        self.calls.append((q, dq, ddq, tau))
        return SimpleNamespace(tau_id=np.full(7, 2.0))


class _WrenchMapper:
    def __init__(self) -> None:
        self.calls = []

    def map_joint_torque(self, q, tau_external):
        self.calls.append((q, tau_external))
        return SimpleNamespace(wrench=tau_external[:6])


def test_world_model_adapter_matches_nero_contact_chain_without_mutating_state() -> None:
    collection = SimpleNamespace(
        tau_ext_inference=SimpleNamespace(enabled=True),
        realtime_plot=SimpleNamespace(
            inverse_dynamics=object(),
            wrench_mapping=object(),
        ),
    )
    tau_f = _TauFPredictor()
    inverse_dynamics = _InverseDynamics()
    mapper = _WrenchMapper()
    adapter = WorldModelWrenchAdapter(
        collection,
        tau_f_predictor=tau_f,
        inverse_dynamics=inverse_dynamics,
        wrench_mapper=mapper,
    )
    history = {
        key: np.arange(21, dtype=float).reshape(3, 7)
        for key in ("q", "v", "a", "tau")
    }
    future = {
        key: np.full((2, 7), float(index))
        for index, key in enumerate(("q", "v", "a", "tau"), start=1)
    }

    wrench = adapter.states_to_wrenches(history, future)

    assert tau_f.inputs["q"].shape == (4, 7)
    np.testing.assert_allclose(tau_f.inputs["q"][:2], history["q"][-2:])
    np.testing.assert_allclose(tau_f.inputs["q"][2:], future["q"])
    np.testing.assert_allclose(tau_f.inputs["dq"][2:], future["v"])
    np.testing.assert_allclose(tau_f.inputs["delta_q"], 0.0)
    np.testing.assert_allclose(wrench, np.full((2, 6), 1.5))
    assert len(inverse_dynamics.calls) == 2
    assert len(mapper.calls) == 2


def test_world_model_adapter_requires_tau_ext_inference_when_no_predictor() -> None:
    collection = SimpleNamespace(
        tau_ext_inference=SimpleNamespace(enabled=False),
        realtime_plot=SimpleNamespace(
            inverse_dynamics=object(),
            wrench_mapping=object(),
        ),
    )
    inverse_dynamics = _InverseDynamics()
    mapper = _WrenchMapper()
    with pytest.raises(ValueError, match="requires tau_ext inference"):
        WorldModelWrenchAdapter(
            collection,
            inverse_dynamics=inverse_dynamics,
            wrench_mapper=mapper,
        )


def test_v4_torch_state_reconstruction_matches_nero_uniform_recurrence() -> None:
    try:
        from model.pinn_model.causal_state import (
            CausalStateEstimatorConfig,
            causal_joint_state_from_position,
        )
    except ModuleNotFoundError:
        pytest.skip("PINN package is not installed in the nero_ws test environment")

    sampling_dt = 0.0125
    mean_window = 8
    q = np.stack(
        [
            np.linspace(0.0, 0.2, 7),
            np.linspace(0.01, 0.22, 7),
            np.linspace(0.03, 0.25, 7),
            np.linspace(0.06, 0.29, 7),
            np.linspace(0.10, 0.34, 7),
            np.linspace(0.15, 0.40, 7),
        ],
        axis=0,
    )
    nero_state = _JointGroupDerivativeState(
        q_window=deque(maxlen=mean_window),
        q_filter=OnePoleLowPass(10.0),
        dq_filter=OnePoleLowPass(6.0),
        ddq_filter=OnePoleLowPass(3.0),
    )
    nero_dq = []
    nero_ddq = []
    for index, value in enumerate(q, start=1):
        state = _append_filtered_joint_state(
            nero_state,
            int(round(index * sampling_dt * 1.0e6)),
            value,
        )
        assert state is not None
        nero_dq.append(state.dq)
        nero_ddq.append(state.ddq)

    _, torch_dq, torch_ddq = causal_joint_state_from_position(
        torch.as_tensor(q, dtype=torch.float64)[None],
        CausalStateEstimatorConfig(
            sampling_dt=sampling_dt,
            q_mean_window_samples=mean_window,
            q_lowpass_cutoff_hz=10.0,
            dq_lowpass_cutoff_hz=6.0,
            ddq_lowpass_cutoff_hz=3.0,
        ),
    )

    np.testing.assert_allclose(torch_dq[0].numpy(), nero_dq, atol=1.0e-12)
    np.testing.assert_allclose(torch_ddq[0].numpy(), nero_ddq, atol=1.0e-12)
