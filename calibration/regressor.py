from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.dynamics_common import DOF, DynamicsModelConfig, ProcessedDynamicsDataset, build_reduced_model


DYNAMIC_PARAMETER_NAMES = (
    "mass",
    "mass_com_x",
    "mass_com_y",
    "mass_com_z",
    "Ixx_origin",
    "Ixy_origin",
    "Iyy_origin",
    "Ixz_origin",
    "Iyz_origin",
    "Izz_origin",
)


@dataclass(frozen=True)
class RegressorData:
    matrix: np.ndarray
    observation: np.ndarray
    prior_parameters: np.ndarray
    parameter_names: tuple[str, ...]


class PinocchioDynamicsRegressor:
    def __init__(self, settings: DynamicsModelConfig, coulomb_velocity_scale_rad_s: float) -> None:
        self.pin, self.model = build_reduced_model(settings)
        self.data = self.model.createData()
        self.coulomb_velocity_scale_rad_s = float(coulomb_velocity_scale_rad_s)
        if self.coulomb_velocity_scale_rad_s <= 0:
            raise ValueError("coulomb velocity scale must be positive")
        self.inertial_parameter_count = 10 * DOF
        self.parameter_count = self.inertial_parameter_count + 3 * DOF
        names: list[str] = []
        for joint_name in settings.joint_names:
            names.extend(f"{joint_name}.{suffix}" for suffix in DYNAMIC_PARAMETER_NAMES)
        names.extend(f"{name}.coulomb" for name in settings.joint_names)
        names.extend(f"{name}.viscous" for name in settings.joint_names)
        names.extend(f"{name}.bias" for name in settings.joint_names)
        self.parameter_names = tuple(names)
        self.prior_parameters = np.concatenate(
            [
                *[
                    np.asarray(self.model.inertias[joint_id].toDynamicParameters(), dtype=np.float64)
                    for joint_id in range(1, DOF + 1)
                ],
                np.zeros(3 * DOF, dtype=np.float64),
            ]
        )

    def build(self, dataset: ProcessedDynamicsDataset) -> RegressorData:
        blocks: list[np.ndarray] = []
        for q, dq, ddq in zip(dataset.q, dataset.dq, dataset.ddq):
            inertial = np.asarray(
                self.pin.computeJointTorqueRegressor(self.model, self.data, q, dq, ddq),
                dtype=np.float64,
            )
            if inertial.shape != (DOF, self.inertial_parameter_count):
                raise RuntimeError(
                    "unexpected Pinocchio torque-regressor shape: "
                    f"{inertial.shape}, expected {(DOF, self.inertial_parameter_count)}"
                )
            coulomb = np.diag(np.tanh(dq / self.coulomb_velocity_scale_rad_s))
            viscous = np.diag(dq)
            bias = np.eye(DOF, dtype=np.float64)
            blocks.append(np.hstack((inertial, coulomb, viscous, bias)))
        matrix = np.vstack(blocks)
        observation = np.asarray(dataset.tau, dtype=np.float64).reshape(-1)
        if not np.isfinite(matrix).all() or not np.isfinite(observation).all():
            raise ValueError("regression inputs contain non-finite values")
        return RegressorData(
            matrix=matrix,
            observation=observation,
            prior_parameters=self.prior_parameters.copy(),
            parameter_names=self.parameter_names,
        )

    def predict(self, dataset: ProcessedDynamicsDataset, parameters: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if parameters.size != self.parameter_count or not np.isfinite(parameters).all():
            raise ValueError(f"dynamics parameters must be a finite {self.parameter_count}D vector")
        return (self.build(dataset).matrix @ parameters).reshape(-1, DOF)

    def parameters_from_model(
        self,
        model,
        coulomb: np.ndarray,
        viscous: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        active_names = tuple(str(name) for name in model.names[1:])
        expected_names = tuple(str(name) for name in self.model.names[1:])
        if active_names != expected_names:
            raise ValueError(f"identified model joint order mismatch: {active_names} != {expected_names}")
        return np.concatenate(
            [
                *[
                    np.asarray(model.inertias[joint_id].toDynamicParameters(), dtype=np.float64)
                    for joint_id in range(1, DOF + 1)
                ],
                np.asarray(coulomb, dtype=np.float64).reshape(DOF),
                np.asarray(viscous, dtype=np.float64).reshape(DOF),
                np.asarray(bias, dtype=np.float64).reshape(DOF),
            ]
        )
