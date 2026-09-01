"""Robot dynamics primitives shared by inference and data collection.

This module deliberately contains only the Pinocchio model adapter.  The
operational-space QP controller was removed from the runtime control chain,
but MTC and IK still need a consistent ``snapshot``/gravity/frame API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class DynamicsError(RuntimeError):
    """Raised when a robot dynamics model cannot be constructed or queried."""


@dataclass(frozen=True)
class DynamicsSnapshot:
    mass_matrix: np.ndarray
    nonlinear_effects: np.ndarray
    jacobian: np.ndarray
    frame_drift: np.ndarray
    pose: np.ndarray


class RobotDynamicsModel(Protocol):
    dof: int
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity_limit: np.ndarray
    effort_limit: np.ndarray

    def snapshot(self, q: np.ndarray, dq: np.ndarray) -> DynamicsSnapshot:
        ...


def _finite_vector(name: str, value: np.ndarray, width: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({width},)")
    return result


class PinocchioDynamicsModel:
    """Seven-axis fixed-base Pinocchio model used by inference controllers."""

    def __init__(
        self,
        urdf_path: str | Path,
        frame_name: str = "gripper_tcp",
        locked_joint_names: tuple[str, ...] = (
            "gripper",
            "gripper_joint1",
            "gripper_joint2",
        ),
        gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise DynamicsError("PinocchioDynamicsModel requires pin>=3,<4") from exc

        self.pin = pin
        full_model = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
        missing = [name for name in locked_joint_names if not full_model.existJointName(name)]
        if missing:
            raise DynamicsError(f"locked joints not found in URDF: {missing}")
        locked_ids = [full_model.getJointId(name) for name in locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        self.dof = int(self.model.nv)
        if self.model.nq != 7 or self.model.nv != 7:
            raise DynamicsError(
                f"Nero dynamics requires a seven-axis model; got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        self.frame_id = int(self.model.getFrameId(frame_name))
        if self.frame_id == len(self.model.frames):
            raise DynamicsError(f"frame not found in URDF: {frame_name}")
        self.reference_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        self.position_lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.position_upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.velocity_limit = np.asarray(self.model.velocityLimit, dtype=np.float64).copy()
        self.effort_limit = np.asarray(self.model.effortLimit, dtype=np.float64).copy()

    def snapshot(self, q: np.ndarray, dq: np.ndarray) -> DynamicsSnapshot:
        q = _finite_vector("q", q, self.dof)
        dq = _finite_vector("dq", dq, self.dof)
        pin = self.pin
        mass_matrix = np.asarray(pin.crba(self.model, self.data, q), dtype=np.float64)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        nonlinear_effects = np.asarray(
            pin.nonLinearEffects(self.model, self.data, q, dq), dtype=np.float64
        ).copy()
        zeros = np.zeros(self.dof, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q, dq, zeros)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = np.asarray(
            pin.getFrameJacobian(self.model, self.data, self.frame_id, self.reference_frame),
            dtype=np.float64,
        ).copy()
        drift_motion = pin.getFrameClassicalAcceleration(
            self.model, self.data, self.frame_id, self.reference_frame
        )
        frame_drift = np.concatenate(
            (
                np.asarray(drift_motion.linear, dtype=np.float64).reshape(3),
                np.asarray(drift_motion.angular, dtype=np.float64).reshape(3),
            )
        )
        placement = self.data.oMf[self.frame_id]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(placement.rotation, dtype=np.float64)
        pose[:3, 3] = np.asarray(placement.translation, dtype=np.float64).reshape(3)
        return DynamicsSnapshot(mass_matrix, nonlinear_effects, jacobian, frame_drift, pose)

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        q_value = _finite_vector("q", q, self.dof)
        zeros = np.zeros(self.dof, dtype=np.float64)
        return np.asarray(self.pin.rnea(self.model, self.data, q_value, zeros, zeros), dtype=np.float64).copy()

    def frame_pose(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        q_value = _finite_vector("q", q, self.dof)
        frame_id = int(self.model.getFrameId(str(frame_name)))
        if frame_id == len(self.model.frames):
            raise DynamicsError(f"frame not found in URDF: {frame_name}")
        self.pin.framesForwardKinematics(self.model, self.data, q_value)
        placement = self.data.oMf[frame_id]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(placement.rotation, dtype=np.float64)
        pose[:3, 3] = np.asarray(placement.translation, dtype=np.float64).reshape(3)
        return pose

    def relative_frame_transform(
        self, q: np.ndarray, source_frame_name: str, target_frame_name: str
    ) -> np.ndarray:
        return np.linalg.inv(self.frame_pose(q, source_frame_name)) @ self.frame_pose(q, target_frame_name)


__all__ = ["DynamicsError", "DynamicsSnapshot", "RobotDynamicsModel", "PinocchioDynamicsModel"]
