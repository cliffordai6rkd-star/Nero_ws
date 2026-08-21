from __future__ import annotations

from typing import Mapping

import numpy as np

from inference.wrench_mapping import (
    PinocchioContactWrenchEstimator,
    WrenchMappingConfig,
)
from nero_collection.config import CollectionConfig, load_config
from nero_collection.inverse_dynamics import PinocchioJointTorqueResidualEstimator
from nero_collection.tau_ext_inference import SequenceTorquePredictor


class WorldModelWrenchAdapter:
    """Apply Nero's physical contact chain to a V3 state trajectory."""

    _TAU_F_TO_STATE = {
        "q": "q",
        "dq": "v",
    }

    def __init__(
        self,
        collection: CollectionConfig,
        *,
        tau_f_predictor=None,
        inverse_dynamics=None,
        wrench_mapper=None,
    ) -> None:
        if tau_f_predictor is None and not collection.tau_ext_inference.enabled:
            raise ValueError("world-model wrench mapping requires tau_ext inference")
        self.tau_f_predictor = tau_f_predictor or SequenceTorquePredictor(
            collection.tau_ext_inference.tau_f,
            name="tau_f",
        )
        self.inverse_dynamics = (
            inverse_dynamics
            or PinocchioJointTorqueResidualEstimator(
                collection.tau_ext_inference.inverse_dynamics
            )
        )
        if wrench_mapper is None:
            inverse_config = collection.tau_ext_inference.inverse_dynamics
            wrench_mapper = PinocchioContactWrenchEstimator(
                WrenchMappingConfig(
                    urdf_path=inverse_config.urdf_path,
                    delay_s=0.0,
                    locked_joint_names=inverse_config.locked_joint_names,
                    gravity_m_s2=inverse_config.gravity_m_s2,
                )
            )
        self.wrench_mapper = wrench_mapper

    @classmethod
    def from_collection_config(cls, path):
        return cls(load_config(path))

    def states_to_wrenches(
        self,
        history: Mapping[str, np.ndarray],
        future: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        history_values = {
            key: self._trajectory(f"history.{key}", history[key], 7)
            for key in ("q", "v", "a", "tau")
        }
        future_values = {
            key: self._trajectory(f"future.{key}", future[key], 7)
            for key in ("q", "v", "a", "tau")
        }
        future_horizon = future_values["q"].shape[0]
        history_horizon = history_values["q"].shape[0]
        if any(value.shape[0] != history_horizon for value in history_values.values()):
            raise ValueError("All world-model history states must share time length")
        if any(value.shape[0] != future_horizon for value in future_values.values()):
            raise ValueError("All future world-model states must share time length")

        tau_f_horizon = int(self.tau_f_predictor.metadata.horizon)
        if history_horizon < tau_f_horizon:
            raise ValueError(
                "world-model history is shorter than the tau_f checkpoint horizon: "
                f"{history_horizon} < {tau_f_horizon}"
            )
        tau_f_inputs = {}
        for tau_f_key in self.tau_f_predictor.metadata.input_keys:
            if tau_f_key == "delta_q":
                # World-model trajectories contain realized state, not a separate
                # controller command. Evaluate the tracked-command case delta_q=0.
                measured = np.zeros_like(history_values["q"][-tau_f_horizon:])
                predicted = np.zeros_like(future_values["q"])
                tau_f_inputs[tau_f_key] = np.concatenate(
                    (measured, predicted), axis=0
                )
                continue
            state_key = self._TAU_F_TO_STATE[tau_f_key]
            measured = history_values[state_key][-tau_f_horizon:]
            tau_f_inputs[tau_f_key] = np.concatenate(
                (measured, future_values[state_key]), axis=0
            )
        tau_f = self.tau_f_predictor.predict_sequence(tau_f_inputs)[
            -future_horizon:
        ]

        wrenches = []
        for step in range(future_horizon):
            residual = self.inverse_dynamics.estimate(
                future_values["q"][step],
                future_values["v"][step],
                future_values["a"][step],
                future_values["tau"][step],
            )
            tau_external = (
                future_values["tau"][step]
                - np.asarray(residual.tau_id, dtype=np.float64)
                - tau_f[step]
            )
            mapping = self.wrench_mapper.map_joint_torque(
                future_values["q"][step],
                tau_external,
            )
            wrenches.append(np.asarray(mapping.wrench, dtype=np.float64))
        return np.stack(wrenches, axis=0)

    @staticmethod
    def _trajectory(name: str, value: np.ndarray, dim: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != dim or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite [T, {dim}] trajectory")
        return array.copy()
