from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


class OSCQPError(RuntimeError):
    """Raised when an OSC-QP problem cannot produce a safe torque sequence."""


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


@dataclass(frozen=True)
class OSCTargetTrajectory:
    """Targets in the Jacobian frame; wrench is the environment-on-tool wrench."""

    poses: np.ndarray
    wrenches: np.ndarray
    twists: np.ndarray | None = None
    accelerations: np.ndarray | None = None

    @classmethod
    def constant(
        cls,
        pose: ArrayLike,
        wrench: ArrayLike,
        horizon_steps: int,
        twist: ArrayLike | None = None,
        acceleration: ArrayLike | None = None,
    ) -> OSCTargetTrajectory:
        pose_value = np.asarray(pose, dtype=np.float64)
        wrench_value = np.asarray(wrench, dtype=np.float64).reshape(-1)
        poses = np.repeat(pose_value[None, :, :], int(horizon_steps), axis=0)
        wrenches = np.repeat(wrench_value[None, :], int(horizon_steps), axis=0)
        twists = _repeat_optional(twist, horizon_steps)
        accelerations = _repeat_optional(acceleration, horizon_steps)
        return cls(poses, wrenches, twists, accelerations)


@dataclass(frozen=True)
class OSCQPConfig:
    horizon_steps: int = 10
    dt_s: float = 0.01
    pose_kp: tuple[float, ...] = (100.0, 100.0, 100.0, 60.0, 60.0, 60.0)
    pose_kd: tuple[float, ...] = (20.0, 20.0, 20.0, 12.0, 12.0, 12.0)
    pose_weight: tuple[float, ...] = (40.0, 40.0, 40.0, 12.0, 12.0, 12.0)
    wrench_weight: tuple[float, ...] = (8.0, 8.0, 8.0, 2.0, 2.0, 2.0)
    acceleration_weight: float = 1.0e-3
    torque_weight: float = 2.0e-4
    torque_rate_weight: float = 2.0e-3
    wrench_rate_weight: float = 1.0e-3
    force_feedback_gain: tuple[float, ...] = (0.5, 0.5, 0.5, 0.2, 0.2, 0.2)
    acceleration_limit: float | tuple[float, ...] = 20.0
    torque_limit: float | tuple[float, ...] | None = None
    wrench_lower: tuple[float, ...] = (-np.inf,) * 6
    wrench_upper: tuple[float, ...] = (np.inf,) * 6
    joint_position_margin_rad: float = 0.02
    friction_coefficient: float | None = None
    contact_normal_axis: int = 2
    osqp_max_iter: int = 4000
    osqp_eps_abs: float = 1.0e-4
    osqp_eps_rel: float = 1.0e-4
    osqp_polish: bool = False
    maximum_constraint_violation: float = 1.0e-3

    @property
    def control_frequency_hz(self) -> float:
        return 1.0 / self.dt_s


@dataclass(frozen=True)
class OSCQPResult:
    tau: np.ndarray
    joint_accelerations: np.ndarray
    predicted_q: np.ndarray
    predicted_dq: np.ndarray
    predicted_wrenches: np.ndarray
    status: str
    iterations: int
    solve_time_s: float
    objective: float
    max_constraint_violation: float

    @property
    def first_tau(self) -> np.ndarray:
        return self.tau[0].copy()


class PinocchioDynamicsModel:
    """Seven-axis fixed-base Pinocchio model used by :class:`OSCQPController`."""

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
            raise OSCQPError("PinocchioDynamicsModel requires pin>=3,<4") from exc

        self.pin = pin
        full_model = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
        missing = [name for name in locked_joint_names if not full_model.existJointName(name)]
        if missing:
            raise OSCQPError(f"OSC-QP locked joints not found in URDF: {missing}")
        locked_ids = [full_model.getJointId(name) for name in locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        self.dof = int(self.model.nv)
        if self.model.nq != 7 or self.model.nv != 7:
            raise OSCQPError(
                f"OSC-QP requires a seven-axis model; got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        self.frame_id = int(self.model.getFrameId(frame_name))
        if self.frame_id == len(self.model.frames):
            raise OSCQPError(f"OSC-QP frame not found in URDF: {frame_name}")
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
        zero_acceleration = np.zeros(self.dof, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q, dq, zero_acceleration)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = np.asarray(
            pin.getFrameJacobian(
                self.model, self.data, self.frame_id, self.reference_frame
            ),
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

    def frame_pose(self, q: ArrayLike, frame_name: str) -> np.ndarray:
        """Return an arbitrary URDF frame pose in the world frame."""
        q_value = _finite_vector("q", q, self.dof)
        frame_id = int(self.model.getFrameId(str(frame_name)))
        if frame_id == len(self.model.frames):
            raise OSCQPError(f"OSC-QP frame not found in URDF: {frame_name}")
        self.pin.framesForwardKinematics(self.model, self.data, q_value)
        placement = self.data.oMf[frame_id]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(placement.rotation, dtype=np.float64)
        pose[:3, 3] = np.asarray(placement.translation, dtype=np.float64).reshape(3)
        return pose

    def relative_frame_transform(
        self,
        q: ArrayLike,
        source_frame_name: str,
        target_frame_name: str,
    ) -> np.ndarray:
        """Return the fixed transform T_source_target at the supplied configuration."""
        source = self.frame_pose(q, source_frame_name)
        target = self.frame_pose(q, target_frame_name)
        return np.linalg.inv(source) @ target


class OSCQPController:
    """Receding-horizon operational-space QP for pose and wrench tracking."""

    def __init__(self, model: RobotDynamicsModel, config: OSCQPConfig | None = None) -> None:
        self.model = model
        self.config = config or OSCQPConfig()
        self.dof = int(model.dof)
        if self.dof != 7:
            raise ValueError(f"Nero OSC-QP requires 7 DoF; got {self.dof}")
        self._validate_config()
        self._validate_model_limits()
        self._warm_start: np.ndarray | None = None

    def optimize_mpc(
        self,
        q: ArrayLike,
        dq: ArrayLike,
        target: OSCTargetTrajectory,
        measured_wrench: ArrayLike | None = None,
        previous_tau: ArrayLike | None = None,
    ) -> OSCQPResult:
        """Solve one MPC step and return the complete future seven-joint torque sequence."""
        try:
            import osqp
        except ImportError as exc:
            raise OSCQPError("OSC-QP optimization requires osqp>=1,<2") from exc

        started_ns = perf_counter_ns()
        q0 = _finite_vector("q", q, self.dof)
        dq0 = _finite_vector("dq", dq, self.dof)
        poses, wrenches, twists, accelerations = self._validate_target(target)
        snapshot = self.model.snapshot(q0, dq0)
        self._validate_snapshot(snapshot)
        measured = (
            _finite_vector("measured_wrench", measured_wrench, 6)
            if measured_wrench is not None
            else None
        )
        previous = (
            _finite_vector("previous_tau", previous_tau, self.dof)
            if previous_tau is not None
            else snapshot.nonlinear_effects
        )

        problem = self._build_problem(
            q0,
            dq0,
            snapshot,
            poses,
            wrenches,
            twists,
            accelerations,
            measured,
            previous,
        )
        (
            hessian,
            gradient,
            constraint_matrix,
            lower,
            upper,
            q_map,
            q_offset,
            dq_map,
            dq_offset,
            tau_map,
            tau_offset,
        ) = problem
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.triu(sparse.csc_matrix(hessian), format="csc"),
            q=gradient,
            A=sparse.csc_matrix(constraint_matrix),
            l=lower,
            u=upper,
            verbose=False,
            max_iter=self.config.osqp_max_iter,
            eps_abs=self.config.osqp_eps_abs,
            eps_rel=self.config.osqp_eps_rel,
            polishing=self.config.osqp_polish,
            warm_starting=True,
        )
        if self._warm_start is not None and self._warm_start.size == gradient.size:
            solver.warm_start(x=self._warm_start)
        solution = solver.solve(raise_error=False)
        status = str(solution.info.status)
        if solution.x is None or not status.lower().startswith("solved"):
            raise OSCQPError(
                f"OSC-QP failed with status={status!r}, iterations={solution.info.iter}"
            )
        decision = np.asarray(solution.x, dtype=np.float64)
        if not np.isfinite(decision).all():
            raise OSCQPError("OSC-QP returned a non-finite decision vector")
        constraint_value = constraint_matrix @ decision
        max_constraint_violation = float(
            max(
                np.max(np.maximum(lower - constraint_value, 0.0)),
                np.max(np.maximum(constraint_value - upper, 0.0)),
            )
        )
        if max_constraint_violation > self.config.maximum_constraint_violation:
            raise OSCQPError(
                "OSC-QP solution exceeds the configured constraint tolerance: "
                f"{max_constraint_violation:.6g} > "
                f"{self.config.maximum_constraint_violation:.6g}"
            )
        self._warm_start = self._shift_warm_start(decision)

        horizon = self.config.horizon_steps
        acceleration_size = horizon * self.dof
        joint_accelerations = decision[:acceleration_size].reshape(horizon, self.dof)
        predicted_wrenches = decision[acceleration_size:].reshape(horizon, 6)
        predicted_q = (q_map @ decision + q_offset).reshape(horizon, self.dof)
        predicted_dq = (dq_map @ decision + dq_offset).reshape(horizon, self.dof)
        tau = (tau_map @ decision + tau_offset).reshape(horizon, self.dof)
        solve_time_s = (perf_counter_ns() - started_ns) * 1.0e-9
        return OSCQPResult(
            tau=tau,
            joint_accelerations=joint_accelerations,
            predicted_q=predicted_q,
            predicted_dq=predicted_dq,
            predicted_wrenches=predicted_wrenches,
            status=status,
            iterations=int(solution.info.iter),
            solve_time_s=solve_time_s,
            objective=float(solution.info.obj_val),
            max_constraint_violation=max_constraint_violation,
        )

    def _build_problem(
        self,
        q0: np.ndarray,
        dq0: np.ndarray,
        snapshot: DynamicsSnapshot,
        poses: np.ndarray,
        wrench_targets: np.ndarray,
        twists: np.ndarray,
        accelerations: np.ndarray,
        measured_wrench: np.ndarray | None,
        previous_tau: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        cfg = self.config
        horizon = cfg.horizon_steps
        dof = self.dof
        acceleration_size = horizon * dof
        variable_count = acceleration_size + horizon * 6
        dt = cfg.dt_s
        jacobian = snapshot.jacobian
        mass = snapshot.mass_matrix
        nonlinear = snapshot.nonlinear_effects

        q_map = np.zeros((horizon * dof, variable_count), dtype=np.float64)
        dq_map = np.zeros_like(q_map)
        q_offset = np.empty(horizon * dof, dtype=np.float64)
        dq_offset = np.tile(dq0, horizon)
        for stage in range(horizon):
            row = slice(stage * dof, (stage + 1) * dof)
            q_offset[row] = q0 + (stage + 1) * dt * dq0
            for control_stage in range(stage + 1):
                column = slice(control_stage * dof, (control_stage + 1) * dof)
                dq_map[row, column] = np.eye(dof) * dt
                coefficient = (stage - control_stage + 0.5) * dt * dt
                q_map[row, column] = np.eye(dof) * coefficient

        tau_map = np.zeros((horizon * dof, variable_count), dtype=np.float64)
        tau_offset = np.tile(nonlinear, horizon)
        for stage in range(horizon):
            row = slice(stage * dof, (stage + 1) * dof)
            acceleration_column = slice(stage * dof, (stage + 1) * dof)
            wrench_column = slice(
                acceleration_size + stage * 6,
                acceleration_size + (stage + 1) * 6,
            )
            tau_map[row, acceleration_column] = mass
            tau_map[row, wrench_column] = -jacobian.T

        hessian = np.eye(variable_count, dtype=np.float64) * 1.0e-9
        gradient = np.zeros(variable_count, dtype=np.float64)
        pose_kp = np.asarray(cfg.pose_kp, dtype=np.float64)
        pose_kd = np.asarray(cfg.pose_kd, dtype=np.float64)
        pose_weight = np.asarray(cfg.pose_weight, dtype=np.float64)
        wrench_weight = np.asarray(cfg.wrench_weight, dtype=np.float64)
        feedback_gain = np.asarray(cfg.force_feedback_gain, dtype=np.float64)

        for stage in range(horizon):
            acceleration_column = slice(stage * dof, (stage + 1) * dof)
            wrench_column = slice(
                acceleration_size + stage * 6,
                acceleration_size + (stage + 1) * 6,
            )
            delta_q_map = np.zeros((dof, variable_count), dtype=np.float64)
            velocity_map = np.zeros((dof, variable_count), dtype=np.float64)
            if stage:
                for control_stage in range(stage):
                    column = slice(control_stage * dof, (control_stage + 1) * dof)
                    delta_q_map[:, column] = (
                        (stage - control_stage - 0.5) * dt * dt * np.eye(dof)
                    )
                    velocity_map[:, column] = dt * np.eye(dof)
            task_map = pose_kp[:, None] * (jacobian @ delta_q_map)
            task_map += pose_kd[:, None] * (jacobian @ velocity_map)
            task_map[:, acceleration_column] += jacobian
            pose_error = _pose_error(snapshot.pose, poses[stage])
            delta_q_offset = stage * dt * dq0
            task_offset = snapshot.frame_drift - accelerations[stage]
            task_offset -= pose_kp * pose_error
            task_offset += pose_kp * (jacobian @ delta_q_offset)
            task_offset -= pose_kd * twists[stage]
            task_offset += pose_kd * (jacobian @ dq0)
            _add_squared_cost(hessian, gradient, task_map, task_offset, pose_weight)

            wrench_command = wrench_targets[stage].copy()
            if measured_wrench is not None:
                wrench_command += feedback_gain * (wrench_targets[stage] - measured_wrench)
            wrench_map = np.zeros((6, variable_count), dtype=np.float64)
            wrench_map[:, wrench_column] = np.eye(6)
            _add_squared_cost(
                hessian, gradient, wrench_map, -wrench_command, wrench_weight
            )

        acceleration_map = np.zeros((acceleration_size, variable_count), dtype=np.float64)
        acceleration_map[:, :acceleration_size] = np.eye(acceleration_size)
        _add_squared_cost(
            hessian,
            gradient,
            acceleration_map,
            np.zeros(acceleration_size),
            np.full(acceleration_size, cfg.acceleration_weight),
        )
        _add_squared_cost(
            hessian,
            gradient,
            tau_map,
            tau_offset,
            np.full(horizon * dof, cfg.torque_weight),
        )
        self._add_rate_costs(
            hessian,
            gradient,
            tau_map,
            tau_offset,
            previous_tau,
            acceleration_size,
        )

        constraint_blocks: list[np.ndarray] = [np.eye(variable_count)]
        lower_blocks: list[np.ndarray] = []
        upper_blocks: list[np.ndarray] = []
        acceleration_limit = _positive_limit(
            "acceleration_limit", cfg.acceleration_limit, dof
        )
        wrench_lower = np.asarray(cfg.wrench_lower, dtype=np.float64)
        wrench_upper = np.asarray(cfg.wrench_upper, dtype=np.float64)
        lower_blocks.append(
            np.concatenate((-np.tile(acceleration_limit, horizon), np.tile(wrench_lower, horizon)))
        )
        upper_blocks.append(
            np.concatenate((np.tile(acceleration_limit, horizon), np.tile(wrench_upper, horizon)))
        )

        position_lower = np.asarray(self.model.position_lower, dtype=np.float64)
        position_upper = np.asarray(self.model.position_upper, dtype=np.float64)
        margin = cfg.joint_position_margin_rad
        constraint_blocks.append(q_map)
        lower_blocks.append(np.tile(position_lower + margin, horizon) - q_offset)
        upper_blocks.append(np.tile(position_upper - margin, horizon) - q_offset)

        velocity_limit = np.asarray(self.model.velocity_limit, dtype=np.float64)
        constraint_blocks.append(dq_map)
        lower_blocks.append(-np.tile(velocity_limit, horizon) - dq_offset)
        upper_blocks.append(np.tile(velocity_limit, horizon) - dq_offset)

        effort_limit = (
            np.asarray(self.model.effort_limit, dtype=np.float64)
            if cfg.torque_limit is None
            else _positive_limit("torque_limit", cfg.torque_limit, dof)
        )
        constraint_blocks.append(tau_map)
        lower_blocks.append(-np.tile(effort_limit, horizon) - tau_offset)
        upper_blocks.append(np.tile(effort_limit, horizon) - tau_offset)
        self._append_friction_constraints(
            constraint_blocks, lower_blocks, upper_blocks, variable_count, acceleration_size
        )

        constraint_matrix = np.vstack(constraint_blocks)
        lower = np.concatenate(lower_blocks)
        upper = np.concatenate(upper_blocks)
        return (
            hessian,
            gradient,
            constraint_matrix,
            lower,
            upper,
            q_map,
            q_offset,
            dq_map,
            dq_offset,
            tau_map,
            tau_offset,
        )

    def _add_rate_costs(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        tau_map: np.ndarray,
        tau_offset: np.ndarray,
        previous_tau: np.ndarray,
        acceleration_size: int,
    ) -> None:
        cfg = self.config
        horizon = cfg.horizon_steps
        dof = self.dof
        variable_count = hessian.shape[0]
        for stage in range(horizon):
            row = slice(stage * dof, (stage + 1) * dof)
            rate_map = tau_map[row].copy()
            rate_offset = tau_offset[row].copy()
            if stage == 0:
                rate_offset -= previous_tau
            else:
                previous_row = slice((stage - 1) * dof, stage * dof)
                rate_map -= tau_map[previous_row]
                rate_offset -= tau_offset[previous_row]
            _add_squared_cost(
                hessian,
                gradient,
                rate_map,
                rate_offset,
                np.full(dof, cfg.torque_rate_weight),
            )

            wrench_rate_map = np.zeros((6, variable_count), dtype=np.float64)
            current_column = slice(
                acceleration_size + stage * 6,
                acceleration_size + (stage + 1) * 6,
            )
            wrench_rate_map[:, current_column] = np.eye(6)
            if stage:
                previous_column = slice(
                    acceleration_size + (stage - 1) * 6,
                    acceleration_size + stage * 6,
                )
                wrench_rate_map[:, previous_column] = -np.eye(6)
            _add_squared_cost(
                hessian,
                gradient,
                wrench_rate_map,
                np.zeros(6),
                np.full(6, cfg.wrench_rate_weight),
            )

    def _append_friction_constraints(
        self,
        matrices: list[np.ndarray],
        lowers: list[np.ndarray],
        uppers: list[np.ndarray],
        variable_count: int,
        acceleration_size: int,
    ) -> None:
        coefficient = self.config.friction_coefficient
        if coefficient is None:
            return
        normal = self.config.contact_normal_axis
        tangential = [axis for axis in range(3) if axis != normal]
        rows: list[np.ndarray] = []
        for stage in range(self.config.horizon_steps):
            wrench_start = acceleration_size + stage * 6
            for tangent in tangential:
                for sign in (-1.0, 1.0):
                    row = np.zeros(variable_count, dtype=np.float64)
                    row[wrench_start + tangent] = sign
                    row[wrench_start + normal] = -coefficient
                    rows.append(row)
        matrix = np.stack(rows, axis=0)
        matrices.append(matrix)
        lowers.append(np.full(matrix.shape[0], -np.inf))
        uppers.append(np.zeros(matrix.shape[0]))

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if not np.isfinite(cfg.dt_s) or cfg.dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        for name in ("pose_kp", "pose_kd", "pose_weight", "wrench_weight", "force_feedback_gain"):
            values = np.asarray(getattr(cfg, name), dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f"{name} must contain six finite non-negative values")
        for name in (
            "acceleration_weight",
            "torque_weight",
            "torque_rate_weight",
            "wrench_rate_weight",
        ):
            value = float(getattr(cfg, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        _positive_limit("acceleration_limit", cfg.acceleration_limit, self.dof)
        if cfg.torque_limit is not None:
            _positive_limit("torque_limit", cfg.torque_limit, self.dof)
        lower = np.asarray(cfg.wrench_lower, dtype=np.float64)
        upper = np.asarray(cfg.wrench_upper, dtype=np.float64)
        if (
            lower.shape != (6,)
            or upper.shape != (6,)
            or np.isnan(lower).any()
            or np.isnan(upper).any()
            or np.any(lower > upper)
        ):
            raise ValueError("wrench bounds must be ordered six-dimensional vectors")
        if cfg.joint_position_margin_rad < 0.0:
            raise ValueError("joint_position_margin_rad must be non-negative")
        if (
            not np.isfinite(cfg.maximum_constraint_violation)
            or cfg.maximum_constraint_violation <= 0.0
        ):
            raise ValueError("maximum_constraint_violation must be positive and finite")
        if cfg.friction_coefficient is not None:
            if not np.isfinite(cfg.friction_coefficient) or cfg.friction_coefficient <= 0.0:
                raise ValueError("friction_coefficient must be positive and finite")
            if cfg.contact_normal_axis not in (0, 1, 2):
                raise ValueError("contact_normal_axis must be 0, 1, or 2")

    def _validate_model_limits(self) -> None:
        for name in ("position_lower", "position_upper", "velocity_limit", "effort_limit"):
            values = np.asarray(getattr(self.model, name), dtype=np.float64)
            if values.shape != (self.dof,) or not np.isfinite(values).all():
                raise ValueError(f"model {name} must be a finite {self.dof}D vector")
        if np.any(np.asarray(self.model.position_lower) >= np.asarray(self.model.position_upper)):
            raise ValueError("model position limits must be strictly ordered")
        if np.any(np.asarray(self.model.velocity_limit) <= 0.0):
            raise ValueError("model velocity limits must be positive")
        if np.any(np.asarray(self.model.effort_limit) <= 0.0):
            raise ValueError("model effort limits must be positive")

    def _validate_target(
        self, target: OSCTargetTrajectory
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        horizon = self.config.horizon_steps
        poses = np.asarray(target.poses, dtype=np.float64)
        wrenches = np.asarray(target.wrenches, dtype=np.float64)
        if poses.shape != (horizon, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(f"target poses must have shape ({horizon}, 4, 4) and be finite")
        if wrenches.shape != (horizon, 6) or not np.isfinite(wrenches).all():
            raise ValueError(f"target wrenches must have shape ({horizon}, 6) and be finite")
        twists = _optional_trajectory("twists", target.twists, horizon)
        accelerations = _optional_trajectory("accelerations", target.accelerations, horizon)
        return poses, wrenches, twists, accelerations

    def _validate_snapshot(self, snapshot: DynamicsSnapshot) -> None:
        expected = {
            "mass_matrix": (self.dof, self.dof),
            "nonlinear_effects": (self.dof,),
            "jacobian": (6, self.dof),
            "frame_drift": (6,),
            "pose": (4, 4),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(snapshot, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise OSCQPError(f"Dynamics snapshot {name} must be finite with shape {shape}")

    def _shift_warm_start(self, decision: np.ndarray) -> np.ndarray:
        horizon = self.config.horizon_steps
        acceleration_size = horizon * self.dof
        accelerations = decision[:acceleration_size].reshape(horizon, self.dof)
        wrenches = decision[acceleration_size:].reshape(horizon, 6)
        return np.concatenate(
            (
                np.vstack((accelerations[1:], accelerations[-1])).reshape(-1),
                np.vstack((wrenches[1:], wrenches[-1])).reshape(-1),
            )
        )


def _add_squared_cost(
    hessian: np.ndarray,
    gradient: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray,
) -> None:
    weighted = weights[:, None] * matrix
    hessian += 2.0 * (matrix.T @ weighted)
    gradient += 2.0 * (matrix.T @ (weights * offset))


def _pose_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    translation = desired[:3, 3] - current[:3, 3]
    rotation_error = desired[:3, :3] @ current[:3, :3].T
    rotation = Rotation.from_matrix(rotation_error).as_rotvec()
    return np.concatenate((translation, rotation))


def _finite_vector(name: str, value: ArrayLike, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}D vector; got {vector}")
    return vector.copy()


def _positive_limit(name: str, value: float | tuple[float, ...], size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim == 0:
        vector = np.full(size, float(vector), dtype=np.float64)
    else:
        vector = vector.reshape(-1)
    if vector.size != size or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError(f"{name} must be positive, finite, and scalar or {size}D")
    return vector


def _repeat_optional(value: ArrayLike | None, horizon_steps: int) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return np.repeat(vector[None, :], int(horizon_steps), axis=0)


def _optional_trajectory(name: str, value: np.ndarray | None, horizon: int) -> np.ndarray:
    if value is None:
        return np.zeros((horizon, 6), dtype=np.float64)
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.shape != (horizon, 6) or not np.isfinite(trajectory).all():
        raise ValueError(f"target {name} must have shape ({horizon}, 6) and be finite")
    return trajectory
