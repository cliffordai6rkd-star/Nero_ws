from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)

WRENCH_EXT_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
WRENCH_EXT_UNITS = ("N", "N", "N", "N.m", "N.m", "N.m")


@dataclass(frozen=True)
class WrenchMappingConfig:
    urdf_path: Path
    frame_name: str = "gripper_tcp"
    delay_s: float = 0.0
    damping: float = 0.02
    joint_weights: tuple[float, ...] = (1.0,) * 7
    reference_frame: str = "local"
    locked_joint_names: tuple[str, ...] = (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class ContactWrenchEstimate:
    wrench: np.ndarray
    tau_id: np.ndarray
    tau_residual: np.ndarray
    reconstruction_error: float
    condition_number: float


@dataclass(frozen=True)
class JointTorqueWrenchEstimate:
    wrench: np.ndarray
    tau_external: np.ndarray
    reconstruction_error: float
    condition_number: float


class PinocchioContactWrenchEstimator:
    def __init__(self, config: WrenchMappingConfig, dof: int = 7) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Contact-wrench inference requires Pinocchio; install pin>=3,<4"
            ) from exc

        self.pin = pin
        self.config = config
        self.dof = int(dof)
        full_model = pin.buildModelFromUrdf(str(config.urdf_path))
        missing_joints = [
            name for name in config.locked_joint_names if not full_model.existJointName(name)
        ]
        if missing_joints:
            raise RuntimeError(f"Contact-wrench locked joints not found in URDF: {missing_joints}")
        locked_ids = [full_model.getJointId(name) for name in config.locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        if self.model.nq != self.dof or self.model.nv != self.dof:
            raise RuntimeError(
                "Contact-wrench model must reduce to seven arm joints; "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(config.gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(config.frame_name)
        if self.frame_id == len(self.model.frames):
            raise RuntimeError(f"Contact-wrench frame not found in URDF: {config.frame_name}")
        self.reference_frame = (
            pin.ReferenceFrame.LOCAL
            if config.reference_frame == "local"
            else pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        log.info(
            "Pinocchio contact estimator ready urdf=%s frame=%s reference=%s "
            "delay=%.3fs damping=%.4g",
            config.urdf_path,
            config.frame_name,
            config.reference_frame,
            config.delay_s,
            config.damping,
        )

    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ) -> ContactWrenchEstimate:
        q = _finite_vector("q", q, self.dof)
        dq = _finite_vector("dq", dq, self.dof)
        ddq = _finite_vector("ddq", ddq, self.dof)
        tau_measured = _finite_vector("tau", tau_measured, self.dof)
        tau_id = np.asarray(
            self.pin.rnea(self.model, self.data, q, dq, ddq),
            dtype=np.float64,
        ).copy()
        tau_residual = tau_id - tau_measured
        mapping = self.map_joint_torque(q, tau_residual)
        return ContactWrenchEstimate(
            wrench=mapping.wrench,
            tau_id=tau_id,
            tau_residual=tau_residual,
            reconstruction_error=mapping.reconstruction_error,
            condition_number=mapping.condition_number,
        )

    def map_joint_torque(
        self,
        q: np.ndarray,
        tau_external: np.ndarray,
    ) -> JointTorqueWrenchEstimate:
        q = _finite_vector("q", q, self.dof)
        tau_external = _finite_vector("tau_external", tau_external, self.dof)
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        jacobian = np.asarray(
            self.pin.getFrameJacobian(
                self.model,
                self.data,
                self.frame_id,
                self.reference_frame,
            ),
            dtype=np.float64,
        )
        wrench_vector, error, condition = solve_damped_wrench(
            jacobian,
            tau_external,
            self.config.damping,
            joint_weights=self.config.joint_weights,
        )
        spatial_force = self.pin.Force(wrench_vector)
        wrench = np.concatenate(
            (
                np.asarray(spatial_force.linear, dtype=np.float64).reshape(3),
                np.asarray(spatial_force.angular, dtype=np.float64).reshape(3),
            )
        )
        return JointTorqueWrenchEstimate(
            wrench=wrench,
            tau_external=tau_external,
            reconstruction_error=error,
            condition_number=condition,
        )


def solve_damped_wrench(
    jacobian: np.ndarray,
    tau_residual: np.ndarray,
    damping: float,
    *,
    joint_weights: tuple[float, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    tau_residual = np.asarray(tau_residual, dtype=np.float64).reshape(-1)
    if jacobian.shape[0] != 6 or jacobian.shape[1] != tau_residual.size:
        raise RuntimeError(
            "Expected a 6xN frame Jacobian and an N-dimensional torque residual; "
            f"got J={jacobian.shape}, tau={tau_residual.shape}"
        )
    if not np.isfinite(jacobian).all() or not np.isfinite(tau_residual).all():
        raise RuntimeError("Contact-wrench inputs must be finite")
    if not np.isfinite(damping) or damping <= 0:
        raise RuntimeError("Contact-wrench damping must be positive and finite")

    joint_to_wrench = jacobian.T
    weights = (
        np.ones(tau_residual.size, dtype=np.float64)
        if joint_weights is None
        else np.asarray(joint_weights, dtype=np.float64).reshape(-1)
    )
    if (
        weights.shape != tau_residual.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise RuntimeError(
            "Contact-wrench joint weights must be positive finite values matching "
            "the joint torque dimension"
        )
    sqrt_weights = np.sqrt(weights)
    weighted_joint_to_wrench = sqrt_weights[:, None] * joint_to_wrench
    weighted_tau_residual = sqrt_weights * tau_residual
    u, singular_values, vt = np.linalg.svd(
        weighted_joint_to_wrench, full_matrices=False
    )
    gains = singular_values / (singular_values * singular_values + damping * damping)
    wrench = vt.T @ (gains * (u.T @ weighted_tau_residual))
    reconstructed = joint_to_wrench @ wrench
    denominator = max(float(np.linalg.norm(tau_residual)), 1e-9)
    reconstruction_error = float(
        np.linalg.norm(reconstructed - tau_residual) / denominator
    )
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    condition_number = (
        float(np.max(singular_values) / smallest)
        if smallest > np.finfo(np.float64).eps
        else float("inf")
    )
    return wrench, reconstruction_error, condition_number


def wrench_ext_dataset_attrs(config: WrenchMappingConfig) -> dict[str, object]:
    return {
        "definition": "damped least-squares solution of tau_ext = J(q)^T wrench_ext",
        "processing_method": "pinocchio_frame_jacobian_damped_least_squares",
        "timestamp_path": "teleop/timestamp_us",
        "q_source_dataset": "teleop/q_follower",
        "tau_source_dataset": "teleop/tau_ext",
        "components_json": json.dumps(WRENCH_EXT_COMPONENTS),
        "component_units_json": json.dumps(WRENCH_EXT_UNITS),
        "frame_name": config.frame_name,
        "frame_type": "end_effector",
        "reference_frame": config.reference_frame,
        "wrench_convention": "environment_on_tool",
        "damping": config.damping,
        "joint_weights_json": json.dumps(config.joint_weights),
        "model_urdf": str(config.urdf_path),
    }


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(
            f"Contact-wrench estimator requires a finite {size}D {name}; got {vector}"
        )
    return vector.copy()
