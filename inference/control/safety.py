"""Small safety guards for modular control targets."""

from __future__ import annotations

import numpy as np

from inference.core.contracts import ControlTarget, Observation


class BasicSafetyGuard:
    """Apply finite-value, joint-step and torque limits to a target.

    Pose geometry and robot-specific limits belong to a dedicated resolver or
    controller.  This guard intentionally handles only coordinate-free vector
    limits that are safe to apply at the common contract boundary.
    """

    def __init__(
        self,
        *,
        maximum_joint_step: float | np.ndarray | None = None,
        maximum_torque: float | np.ndarray | None = None,
    ) -> None:
        self.maximum_joint_step = self._limit(maximum_joint_step, "maximum_joint_step")
        self.maximum_torque = self._limit(maximum_torque, "maximum_torque")

    @staticmethod
    def _limit(value: float | np.ndarray | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.repeat(result, 7)
        result = result.reshape(-1)
        if result.shape != (7,) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{name} must be a positive scalar or seven-vector")
        return result.copy()

    def reset_episode(self) -> None:
        return None

    def validate(
        self,
        observation: Observation,
        target: ControlTarget | None,
    ) -> ControlTarget | None:
        if target is None:
            return None
        q = None if target.q is None else np.asarray(target.q, dtype=np.float64).copy()
        torque = (
            None if target.torque is None else np.asarray(target.torque, dtype=np.float64).copy()
        )
        if q is not None and self.maximum_joint_step is not None:
            delta = np.clip(q - observation.q, -self.maximum_joint_step, self.maximum_joint_step)
            q = observation.q + delta
        if torque is not None and self.maximum_torque is not None:
            torque = np.clip(torque, -self.maximum_torque, self.maximum_torque)
        return ControlTarget(
            q=q,
            dq=target.dq,
            torque=torque,
            pose=target.pose,
            mode=target.mode,
            metadata={**target.metadata, "safety_guard": "basic"},
        )


__all__ = ["BasicSafetyGuard"]
