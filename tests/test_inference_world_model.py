from types import SimpleNamespace

import numpy as np
import pytest

from inference.world_model import WorldModelWrenchAdapter


class _TauOtherPredictor:
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

    def gravity_torque(self, q):
        return np.full(7, 2.0)


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
    tau_other = _TauOtherPredictor()
    inverse_dynamics = _InverseDynamics()
    mapper = _WrenchMapper()
    adapter = WorldModelWrenchAdapter(
        collection,
        tau_other_predictor=tau_other,
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

    assert tau_other.inputs["q"].shape == (4, 7)
    np.testing.assert_allclose(tau_other.inputs["q"][:2], history["q"][-2:])
    np.testing.assert_allclose(tau_other.inputs["q"][2:], future["q"])
    np.testing.assert_allclose(tau_other.inputs["dq"][2:], future["v"])
    np.testing.assert_allclose(tau_other.inputs["delta_q"], 0.0)
    np.testing.assert_allclose(wrench, np.full((2, 6), 1.5))
    assert len(inverse_dynamics.calls) == 0
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
