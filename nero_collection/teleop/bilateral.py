from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nero_collection.arms.base import ArmState
from nero_collection.config import BilateralMitConfig, InverseDynamicsConfig
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator
from nero_collection.filters import OnePoleLowPass


DOF = 7


class JointDynamics(Protocol):
    model: object

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        ...

    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ):
        ...


@dataclass(frozen=True)
class JointMitCommand:
    q: np.ndarray
    v_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    t_ff: np.ndarray


@dataclass(frozen=True)
class BilateralControlResult:
    leader: JointMitCommand
    follower: JointMitCommand
    gravity_leader: np.ndarray
    gravity_follower: np.ndarray
    tau_ext_follower: np.ndarray
    tau_feedback_leader: np.ndarray


class BilateralJointController:
    def __init__(
        self,
        config: BilateralMitConfig,
        dynamics_config: InverseDynamicsConfig,
        *,
        dynamics: JointDynamics | None = None,
    ) -> None:
        self.config = config
        self.dynamics = dynamics or PinocchioJointTorqueResidualEstimator(dynamics_config)
        self.feedback_filter = OnePoleLowPass(config.force_feedback_lowpass_hz)
        margin = float(config.joint_limit_margin_rad)
        self.position_lower = (
            np.asarray(self.dynamics.model.lowerPositionLimit, dtype=np.float64) + margin
        )
        self.position_upper = (
            np.asarray(self.dynamics.model.upperPositionLimit, dtype=np.float64) - margin
        )
        if self.position_lower.shape != (DOF,) or self.position_upper.shape != (DOF,):
            raise RuntimeError("Bilateral controller requires seven joint limits")
        if np.any(self.position_lower >= self.position_upper):
            raise RuntimeError("Bilateral joint-limit margin leaves an empty range")
        self._leader_q0: np.ndarray | None = None
        self._follower_q0: np.ndarray | None = None
        self._force_bias: np.ndarray | None = None
        self._previous_feedback = np.zeros(DOF, dtype=np.float64)
        self._previous_compute_t: float | None = None
        self._activation_t: float | None = None

    def activate(self, leader: ArmState, follower: ArmState) -> None:
        _validate_state("leader", leader)
        _validate_state("follower", follower)
        self._leader_q0 = np.asarray(leader.q, dtype=np.float64).copy()
        self._follower_q0 = np.asarray(follower.q, dtype=np.float64).copy()
        self._force_bias = None
        self._previous_feedback.fill(0.0)
        self._previous_compute_t = None
        self._activation_t = time.monotonic()
        self.feedback_filter = OnePoleLowPass(self.config.force_feedback_lowpass_hz)

    def compute(
        self,
        leader: ArmState,
        follower: ArmState,
        *,
        timestamp_us: int,
        tau_ext_override: np.ndarray | None = None,
    ) -> BilateralControlResult:
        _validate_state("leader", leader)
        _validate_state("follower", follower)
        if self._leader_q0 is None or self._follower_q0 is None or self._activation_t is None:
            raise RuntimeError("Bilateral controller must be activated before compute")

        cfg = self.config
        scale = _vector(cfg.position_scale)
        follower_target = self.follower_target(leader, follower)

        gravity_leader = self.dynamics.gravity_torque(leader.q) * _vector(
            cfg.leader_gravity_scale
        )
        gravity_follower = self.dynamics.gravity_torque(follower.q) * _vector(
            cfg.follower_gravity_scale
        )
        if tau_ext_override is not None:
            tau_ext = _vector(tau_ext_override)
            tau_ext_feedback = tau_ext
        elif not np.any(_vector(cfg.force_feedback_gain)):
            tau_ext = np.zeros(DOF, dtype=np.float64)
            tau_ext_feedback = tau_ext
        else:
            ddq = _state_vector("follower", "ddq", follower.ddq)
            residual = self.dynamics.estimate(
                follower.q,
                follower.dq,
                ddq,
                follower.torque,
            )
            tau_ext = np.zeros(DOF, dtype=np.float64)
            residual_torque = np.asarray(residual.tau_residual, dtype=np.float64)
            if self._force_bias is None:
                self._force_bias = residual_torque.copy()
            else:
                tau_ext = residual_torque - self._force_bias
                tau_ext = self.feedback_filter.apply(tau_ext, int(timestamp_us))
            deadband = _vector(cfg.force_feedback_deadband_nm)
            tau_ext_feedback = np.sign(tau_ext) * np.maximum(
                np.abs(tau_ext) - deadband,
                0.0,
            )
        feedback = (
            _vector(cfg.force_feedback_sign)
            * _vector(cfg.force_feedback_gain)
            * scale
            * tau_ext_feedback
        )
        feedback = np.clip(
            feedback,
            -_vector(cfg.force_feedback_limit_nm),
            _vector(cfg.force_feedback_limit_nm),
        )
        elapsed = max(0.0, time.monotonic() - self._activation_t)
        if cfg.force_feedback_ramp_s > 0:
            feedback *= min(1.0, elapsed / cfg.force_feedback_ramp_s)
        feedback = self._rate_limit_feedback(feedback)

        leader_t_ff = np.clip(
            gravity_leader + feedback,
            -_vector(cfg.leader_torque_limit_nm),
            _vector(cfg.leader_torque_limit_nm),
        )
        follower_t_ff = np.clip(
            gravity_follower,
            -_vector(cfg.follower_torque_limit_nm),
            _vector(cfg.follower_torque_limit_nm),
        )
        zeros = np.zeros(DOF, dtype=np.float64)
        return BilateralControlResult(
            leader=JointMitCommand(
                q=np.asarray(leader.q, dtype=np.float64).copy(),
                v_des=zeros.copy(),
                kp=_vector(cfg.leader_kp),
                kd=_vector(cfg.leader_kd),
                t_ff=leader_t_ff,
            ),
            follower=JointMitCommand(
                q=follower_target,
                v_des=zeros.copy(),
                kp=_vector(cfg.follower_kp),
                kd=_vector(cfg.follower_kd),
                t_ff=follower_t_ff,
            ),
            gravity_leader=gravity_leader,
            gravity_follower=gravity_follower,
            tau_ext_follower=tau_ext,
            tau_feedback_leader=feedback,
        )

    def follower_target(self, leader: ArmState, follower: ArmState) -> np.ndarray:
        """Return the side-effect-free follower target used by ``compute``."""
        _validate_state("leader", leader)
        _validate_state("follower", follower)
        if self._leader_q0 is None or self._follower_q0 is None:
            raise RuntimeError("Bilateral controller must be activated before compute")
        scale = _vector(self.config.position_scale)
        target = self._follower_q0 + scale * (leader.q - self._leader_q0)
        return np.clip(target, self.position_lower, self.position_upper)

    def hold(self, leader: ArmState, follower: ArmState) -> BilateralControlResult:
        _validate_state("leader", leader)
        _validate_state("follower", follower)
        cfg = self.config
        gravity_leader = self.dynamics.gravity_torque(leader.q) * _vector(
            cfg.leader_gravity_scale
        )
        gravity_follower = self.dynamics.gravity_torque(follower.q) * _vector(
            cfg.follower_gravity_scale
        )
        zeros = np.zeros(DOF, dtype=np.float64)
        return BilateralControlResult(
            leader=JointMitCommand(
                q=leader.q.copy(),
                v_des=zeros.copy(),
                kp=_vector(cfg.follower_kp),
                kd=_vector(cfg.follower_kd),
                t_ff=np.clip(
                    gravity_leader,
                    -_vector(cfg.leader_torque_limit_nm),
                    _vector(cfg.leader_torque_limit_nm),
                ),
            ),
            follower=JointMitCommand(
                q=follower.q.copy(),
                v_des=zeros.copy(),
                kp=_vector(cfg.follower_kp),
                kd=_vector(cfg.follower_kd),
                t_ff=np.clip(
                    gravity_follower,
                    -_vector(cfg.follower_torque_limit_nm),
                    _vector(cfg.follower_torque_limit_nm),
                ),
            ),
            gravity_leader=gravity_leader,
            gravity_follower=gravity_follower,
            tau_ext_follower=zeros.copy(),
            tau_feedback_leader=zeros.copy(),
        )

    def _rate_limit_feedback(self, feedback: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        previous_t = self._previous_compute_t
        self._previous_compute_t = now
        if previous_t is None:
            self._previous_feedback = np.zeros(DOF, dtype=np.float64)
            return self._previous_feedback.copy()
        dt = max(now - previous_t, 1e-4)
        maximum_delta = _vector(self.config.feedback_torque_rate_limit_nm_s) * dt
        delta = np.clip(feedback - self._previous_feedback, -maximum_delta, maximum_delta)
        self._previous_feedback = self._previous_feedback + delta
        return self._previous_feedback.copy()


def _validate_state(role: str, state: ArmState) -> None:
    for name in ("q", "dq", "torque"):
        _state_vector(role, name, getattr(state, name))


def _state_vector(role: str, name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (DOF,) or not np.isfinite(result).all():
        raise RuntimeError(f"Bilateral {role} state has invalid {name}: {result}")
    return result


def _vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (DOF,) or not np.isfinite(result).all():
        raise RuntimeError(f"Bilateral control vector must be finite 7D; got {result}")
    return result.copy()
