from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from nero_collection.config import ContactWrenchConfig, InverseDynamicsConfig

log = logging.getLogger(__name__)

WRENCH_EXT_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
WRENCH_EXT_UNITS = ("N", "N", "N", "N.m", "N.m", "N.m")


@dataclass(frozen=True)
class ContactWrenchEstimate:
    wrench: np.ndarray
    tau_id: np.ndarray
    tau_residual: np.ndarray
    reconstruction_error: float
    condition_number: float


@dataclass(frozen=True)
class JointTorqueResidualEstimate:
    tau_id: np.ndarray
    tau_friction: np.ndarray
    tau_bias: np.ndarray
    tau_model: np.ndarray
    tau_residual: np.ndarray


@dataclass(frozen=True)
class JointTorqueWrenchEstimate:
    wrench: np.ndarray
    tau_external: np.ndarray
    reconstruction_error: float
    condition_number: float


@dataclass(frozen=True)
class IdentifiedJointDynamics:
    coulomb_nm: np.ndarray
    viscous_nm_per_rad_s: np.ndarray
    bias_nm: np.ndarray
    coulomb_velocity_scale_rad_s: float


class PinocchioJointTorqueResidualEstimator:
    def __init__(self, config: InverseDynamicsConfig, dof: int = 7) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Inverse-dynamics residual estimation requires Pinocchio; install pin>=3,<4"
            ) from exc

        self.pin = pin
        self.config = config
        self.dof = int(dof)
        full_model = pin.buildModelFromUrdf(str(config.urdf_path))
        missing_joints = [
            name for name in config.locked_joint_names if not full_model.existJointName(name)
        ]
        if missing_joints:
            raise RuntimeError(f"Inverse-dynamics locked joints not found in URDF: {missing_joints}")
        locked_ids = [full_model.getJointId(name) for name in config.locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        if self.model.nq != self.dof or self.model.nv != self.dof:
            raise RuntimeError(
                "Inverse-dynamics model must reduce to seven arm joints; "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(config.gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        self.identified = _load_identified_joint_dynamics(
            config.manifest_path,
            config.urdf_path,
            tuple(str(name) for name in self.model.names[1:]),
            self.dof,
        )
        log.info(
            "Pinocchio inverse-dynamics residual estimator ready urdf=%s manifest=%s "
            "(friction/bias ignored for tau_f)",
            config.urdf_path,
            config.manifest_path or "none (RNEA only)",
        )

    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ) -> JointTorqueResidualEstimate:
        q = _finite_vector("q", q, self.dof)
        dq = _finite_vector("dq", dq, self.dof)
        ddq = _finite_vector("ddq", ddq, self.dof)
        tau_measured = _finite_vector("tau", tau_measured, self.dof)
        tau_id = np.asarray(
            self.pin.rnea(self.model, self.data, q, dq, ddq),
            dtype=np.float64,
        ).copy()
        tau_model = tau_id.copy()
        zero_torque = np.zeros_like(tau_id)
        return JointTorqueResidualEstimate(
            tau_id=tau_id,
            tau_friction=zero_torque.copy(),
            tau_bias=zero_torque.copy(),
            tau_model=tau_model,
            tau_residual=tau_id - tau_measured,
        )

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        q = _finite_vector("q", q, self.dof)
        zeros = np.zeros(self.dof, dtype=np.float64)
        return np.asarray(
            self.pin.rnea(self.model, self.data, q, zeros, zeros),
            dtype=np.float64,
        ).copy()


class PinocchioContactWrenchEstimator:
    def __init__(self, config: ContactWrenchConfig, dof: int = 7) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Contact-wrench plotting requires Pinocchio; install pin>=3,<4"
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
            "Pinocchio contact estimator ready urdf=%s frame=%s reference=%s delay=%.3fs damping=%.4g",
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
        """Map external joint torque to an environment-on-tool wrench."""
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
    if weights.shape != tau_residual.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
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
    reconstruction_error = float(np.linalg.norm(reconstructed - tau_residual) / denominator)
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    condition_number = (
        float(np.max(singular_values) / smallest)
        if smallest > np.finfo(np.float64).eps
        else float("inf")
    )
    return wrench, reconstruction_error, condition_number


def wrench_ext_dataset_attrs(config: ContactWrenchConfig) -> dict[str, object]:
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
        raise RuntimeError(f"Contact-wrench estimator requires a finite {size}D {name}; got {vector}")
    return vector.copy()


def _load_identified_joint_dynamics(
    manifest_path: Path | None,
    urdf_path: Path,
    model_joint_names: tuple[str, ...],
    dof: int,
) -> IdentifiedJointDynamics:
    if manifest_path is None:
        zeros = np.zeros(dof, dtype=np.float64)
        return IdentifiedJointDynamics(
            coulomb_nm=zeros.copy(),
            viscous_nm_per_rad_s=zeros.copy(),
            bias_nm=zeros.copy(),
            coulomb_velocity_scale_rad_s=1.0,
        )

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"Identified dynamics manifest does not exist: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Identified dynamics manifest must be a mapping: {manifest_path}")

    identified_urdf_value = payload.get("identified_urdf")
    if not identified_urdf_value:
        raise RuntimeError("Identified dynamics manifest does not define identified_urdf")
    identified_urdf = Path(str(identified_urdf_value)).expanduser()
    if not identified_urdf.is_absolute():
        identified_urdf = (manifest_path.parent / identified_urdf).resolve()
    if not _same_file(identified_urdf, Path(urdf_path)):
        raise RuntimeError(
            "Identified dynamics manifest/URDF mismatch: "
            f"manifest={identified_urdf}, configured={Path(urdf_path).resolve()}"
        )

    manifest_joint_names = tuple(str(name) for name in payload.get("joint_names", ()))
    if manifest_joint_names != model_joint_names:
        raise RuntimeError(
            "Identified dynamics manifest joint order does not match the reduced model: "
            f"manifest={manifest_joint_names}, model={model_joint_names}"
        )

    friction = payload.get("friction")
    if not isinstance(friction, dict):
        raise RuntimeError("Identified dynamics manifest friction must be a mapping")
    coulomb = _manifest_vector("friction.coulomb_nm", friction.get("coulomb_nm"), dof)
    viscous = _manifest_vector(
        "friction.viscous_nm_per_rad_s",
        friction.get("viscous_nm_per_rad_s"),
        dof,
    )
    bias = _manifest_vector("joint_torque_bias_nm", payload.get("joint_torque_bias_nm"), dof)
    velocity_scale = _manifest_velocity_scale(friction)
    return IdentifiedJointDynamics(
        coulomb_nm=coulomb,
        viscous_nm_per_rad_s=viscous,
        bias_nm=bias,
        coulomb_velocity_scale_rad_s=velocity_scale,
    )


def _manifest_vector(name: str, value, dof: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Identified dynamics manifest contains invalid {name}: {value}"
        ) from exc
    if vector.size != dof or not np.isfinite(vector).all():
        raise RuntimeError(f"Identified dynamics manifest contains invalid {name}: {vector}")
    return vector.copy()


def _manifest_velocity_scale(friction: dict) -> float:
    value = friction.get("coulomb_velocity_scale_rad_s")
    if value is None:
        legacy = str(friction.get("velocity_sign_model", ""))
        match = re.fullmatch(
            r"\s*tanh\(dq\s*/\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\)\s*",
            legacy,
        )
        if match is None:
            raise RuntimeError(
                "Identified dynamics manifest must define "
                "friction.coulomb_velocity_scale_rad_s"
            )
        value = match.group(1)
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Identified dynamics manifest contains invalid "
            f"friction.coulomb_velocity_scale_rad_s: {value}"
        ) from exc
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            "Identified dynamics manifest contains invalid "
            f"friction.coulomb_velocity_scale_rad_s: {value}"
        )
    return scale


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()
