"""Headless MuJoCo dynamics backend for offline inference.

This module is intentionally separate from the hardware runtime.  It builds a
MuJoCo model from the calibration :class:`~calibration.dynamics_common.DynamicsPlan`,
adds one torque motor for each of the seven Nero arm joints, and advances the
model using commands produced by either the legacy or contact-WM inference
pipeline.

The backend never writes ``data.qpos`` during a control step.  Position targets
are converted to torque (software PD), so all three supported execution modes
are exercised through the same MuJoCo dynamics path:

``q`` -> position servo -> torque motor
``mtc`` -> blended MTC residual + q/dq feedback -> torque motor
``tau`` -> direct torque motor command

MuJoCo is imported lazily.  Importing this module therefore remains possible in
environments that only run data processing or unit tests without MuJoCo.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from calibration.dynamics_common import DOF, DynamicsPlan, load_dynamics_plan
from calibration.simulation import (
    _convert_urdf_to_mjcf,
    _joint_qpos_addresses,
    _merge_robot_into_scene,
)


def _default_zero() -> np.ndarray:
    return np.zeros(DOF, dtype=np.float64)


def _as_joint_vector(
    value: Sequence[float] | np.ndarray | float | None,
    *,
    name: str,
    default: float | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Convert a scalar/sequence to a finite seven-element joint vector."""

    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, DOF)
    else:
        array = array.reshape(-1)
    if array.shape != (DOF,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or seven-vector")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array.copy()


def _first_joint_vector(value: Any, *, name: str) -> np.ndarray:
    """Accept a single target or a horizon and return its first joint vector."""

    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (DOF,):
        result = array
    elif array.ndim == 2 and array.shape[1] == DOF and array.shape[0] >= 1:
        result = array[0]
    else:
        raise ValueError(f"{name} must have shape (7,) or (H, 7)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(result, dtype=np.float64).copy()


@dataclass(frozen=True)
class MujocoBackendConfig:
    """Numerical settings for :class:`MujocoDynamicsBackend`.

    ``mit_kp``/``mit_kd`` are used when a command does not carry effective
    gains.  ``q_kp``/``q_kd`` are the software position-servo gains used by the
    ``q`` mode.  They are deliberately explicit because a MuJoCo motor does
    not have a built-in position command interface.
    """

    physics_dt_s: float = 0.001
    control_dt_s: float = 0.01
    mit_kp: float | tuple[float, ...] = 0.0
    mit_kd: float | tuple[float, ...] = 0.0
    q_kp: float | tuple[float, ...] = 50.0
    q_kd: float | tuple[float, ...] = 5.0
    torque_limits_nm: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MujocoCommand:
    """A normalized command accepted by :meth:`MujocoDynamicsBackend.step`."""

    mode: str
    q_target: np.ndarray | None = None
    dq_target: np.ndarray | None = None
    tau_target: np.ndarray | None = None
    tau_command: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None


@dataclass(frozen=True)
class MujocoState:
    """Arm state sampled immediately after a control step."""

    timestamp_s: float
    q: np.ndarray
    dq: np.ndarray
    applied_torque: np.ndarray
    command_torque: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray

    @property
    def tau(self) -> np.ndarray:
        """Alias used by the H5 and inference contracts."""

        return self.applied_torque


def _normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    if value not in {"q", "mtc", "tau"}:
        raise ValueError("MuJoCo execution mode must be q, mtc, or tau")
    return value


def _extract_output_command(output: Any, mode: str | None = None) -> MujocoCommand:
    """Map an ``InferenceOutput``-like object to a simulation command.

    The Contact WM pipeline's MTC ``tau_command`` already includes feedback
    evaluated at the recorded hardware state.  For a closed-loop simulation we
    intentionally use its ``torque_target`` feed-forward/residual component
    and recompute feedback from MuJoCo's current q/dq.
    """

    selected = mode
    if selected is None:
        selected = getattr(output, "control_mode", None)
    if selected is None:
        # Legacy predictor-disabled outputs predate ``control_mode`` and carry
        # their command under ``joint_position_command``.  Infer q execution
        # when no explicit contact-WM mode is present; otherwise preserve the
        # historical direct-torque default.
        if (
            getattr(output, "joint_position_target", None) is not None
            or getattr(output, "joint_position_command", None) is not None
        ):
            selected = "q"
        else:
            selected = "tau"
    selected = _normalize_mode(selected)
    if selected == "q":
        target = getattr(output, "joint_position_target", None)
        if target is None:
            target = getattr(output, "joint_position_command", None)
        return MujocoCommand(mode=selected, q_target=target)
    if selected == "mtc":
        return MujocoCommand(
            # MTC uses the MIT transport with its effective q/v gains and the
            # residual feed-forward produced by the inference pipeline.
            mode="mtc",
            q_target=getattr(output, "joint_position_target", None),
            dq_target=getattr(output, "joint_velocity_target", None),
            tau_target=getattr(output, "torque_target", None),
            kp=getattr(output, "mit_kp", None),
            kd=getattr(output, "mit_kd", None),
        )
    # Direct-torque mode exposes the actual command under tau_command.
    return MujocoCommand(
        mode=selected,
        tau_command=getattr(output, "tau_command", None),
    )


def _inject_torque_actuators(
    root: ET.Element,
    joint_names: Sequence[str],
    torque_limits: np.ndarray,
) -> tuple[str, ...]:
    """Ensure one named, unit-gear motor exists for each arm joint.

    Calibration scenes currently contain no actuators.  If a caller supplies a
    custom scene that already has one of our names, it is reused and its limits
    are refreshed.  Existing unrelated actuators are left untouched.
    """

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    names: list[str] = []
    for index, joint_name in enumerate(joint_names):
        actuator_name = f"nero_{joint_name}_torque"
        existing = next(
            (
                item
                for item in actuator.findall("motor")
                if item.get("name") == actuator_name
            ),
            None,
        )
        if existing is None:
            existing = ET.SubElement(actuator, "motor")
        existing.set("name", actuator_name)
        existing.set("joint", str(joint_name))
        existing.set("gear", "1")
        existing.set("ctrllimited", "true")
        existing.set(
            "ctrlrange",
            f"{-float(torque_limits[index]):.12g} {float(torque_limits[index]):.12g}",
        )
        existing.set("forcelimited", "true")
        existing.set(
            "forcerange",
            f"{-float(torque_limits[index]):.12g} {float(torque_limits[index]):.12g}",
        )
        names.append(actuator_name)
    return tuple(names)


class MujocoDynamicsBackend:
    """Headless MuJoCo dynamics executor for the seven-joint Nero arm.

    Parameters
    ----------
    plan:
        A loaded calibration :class:`DynamicsPlan`.  The URDF and scene
        template are converted exactly as in ``calibration.simulation``.
    initial_q:
        Seven arm joint positions in the hardware joint order.
    scene_path:
        Optional path at which the generated actuator-enabled MJCF is saved for
        inspection.  The model is loaded from the generated XML in memory.
    config:
        Physics, servo, and torque-limit settings.  Explicit keyword settings
        below override the corresponding config values.
    """

    def __init__(
        self,
        plan: DynamicsPlan,
        initial_q: Sequence[float] | np.ndarray,
        *,
        scene_path: str | Path | None = None,
        config: MujocoBackendConfig | None = None,
        physics_dt_s: float | None = None,
        control_dt_s: float | None = None,
        torque_limits_nm: Sequence[float] | np.ndarray | float | None = None,
        mit_kp: Sequence[float] | np.ndarray | float | None = None,
        mit_kd: Sequence[float] | np.ndarray | float | None = None,
        q_kp: Sequence[float] | np.ndarray | float | None = None,
        q_kd: Sequence[float] | np.ndarray | float | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by packaging envs
            raise RuntimeError("MuJoCo simulation requires mujoco>=3.3,<4") from exc

        if not isinstance(plan, DynamicsPlan):
            raise TypeError("plan must be a loaded calibration DynamicsPlan")
        self.mujoco = mujoco
        self.plan = plan
        base_config = config or MujocoBackendConfig()
        self.config = base_config
        self.physics_dt_s = float(
            base_config.physics_dt_s if physics_dt_s is None else physics_dt_s
        )
        self.control_dt_s = float(
            base_config.control_dt_s if control_dt_s is None else control_dt_s
        )
        if not np.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be positive and finite")
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive and finite")
        ratio = self.control_dt_s / self.physics_dt_s
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps, rtol=0, atol=1e-9):
            raise ValueError(
                "control_dt_s must be an integer multiple of physics_dt_s "
                f"(got {self.control_dt_s} / {self.physics_dt_s})"
            )

        plan_limits = getattr(getattr(plan, "safety", None), "max_abs_torque_nm", None)
        configured_limits = (
            base_config.torque_limits_nm
            if torque_limits_nm is None
            else torque_limits_nm
        )
        if configured_limits is None:
            configured_limits = plan_limits
        self.torque_limits_nm = _as_joint_vector(
            configured_limits,
            name="torque_limits_nm",
            default=100.0,
            positive=True,
        )
        self.mit_kp = _as_joint_vector(
            base_config.mit_kp if mit_kp is None else mit_kp,
            name="mit_kp",
            nonnegative=True,
        )
        self.mit_kd = _as_joint_vector(
            base_config.mit_kd if mit_kd is None else mit_kd,
            name="mit_kd",
            nonnegative=True,
        )
        self.q_kp = _as_joint_vector(
            base_config.q_kp if q_kp is None else q_kp,
            name="q_kp",
            nonnegative=True,
        )
        self.q_kd = _as_joint_vector(
            base_config.q_kd if q_kd is None else q_kd,
            name="q_kd",
            nonnegative=True,
        )

        initial = _first_joint_vector(initial_q, name="initial_q")
        robot_root = _convert_urdf_to_mjcf(
            mujoco,
            plan.model.urdf_path,
            plan.model.locked_joint_names,
        )
        scene_root = _merge_robot_into_scene(plan, robot_root)
        self.actuator_names = _inject_torque_actuators(
            scene_root,
            plan.model.joint_names,
            self.torque_limits_nm,
        )
        self.scene_xml = ET.tostring(scene_root, encoding="unicode")
        self.scene_path: Path | None = None
        if scene_path is not None:
            self.scene_path = Path(scene_path).expanduser().resolve()
            self.scene_path.parent.mkdir(parents=True, exist_ok=True)
            output_root = copy.deepcopy(scene_root)
            ET.indent(output_root, space="  ")
            ET.ElementTree(output_root).write(
                self.scene_path,
                encoding="utf-8",
                xml_declaration=True,
            )

        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.model.opt.timestep = self.physics_dt_s
        self.data = mujoco.MjData(self.model)
        # Keep any non-arm coordinates (for example a free base or scene
        # qpos) at the model default when resetting.  The Nero URDF currently
        # has only the seven scalar arm joints, but preserving the template
        # defaults makes the backend safe for extended scenes as well.
        self._base_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self.joint_qpos_addresses = _joint_qpos_addresses(mujoco, self.model, plan)
        self.joint_qvel_addresses = self._joint_qvel_addresses()
        self.actuator_ids = np.asarray(
            [
                int(
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        name,
                    )
                )
                for name in self.actuator_names
            ],
            dtype=np.int32,
        )
        if np.any(self.actuator_ids < 0):
            raise ValueError("generated MuJoCo torque actuator is missing")
        self._timestamp_s = 0.0
        self._last_command_torque = np.zeros(DOF, dtype=np.float64)
        self._last_applied_torque = np.zeros(DOF, dtype=np.float64)
        self.reset(initial)

    @classmethod
    def from_plan_path(
        cls,
        plan_path: str | Path,
        initial_q: Sequence[float] | np.ndarray,
        **kwargs: Any,
    ) -> "MujocoDynamicsBackend":
        """Load a calibration YAML and construct a backend."""

        return cls(load_dynamics_plan(plan_path), initial_q, **kwargs)

    def _joint_qvel_addresses(self) -> np.ndarray:
        addresses: list[int] = []
        for joint_name in self.plan.model.joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing arm joint: {joint_name}")
            addresses.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(addresses, dtype=np.int32)

    def reset(
        self,
        q: Sequence[float] | np.ndarray | None = None,
        dq: Sequence[float] | np.ndarray | None = None,
    ) -> MujocoState:
        """Reset state without changing model parameters or actuator limits."""

        if q is None:
            q_values = np.asarray(self.data.qpos[self.joint_qpos_addresses], dtype=np.float64)
        else:
            q_values = _first_joint_vector(q, name="q")
        if dq is None:
            dq_values = np.zeros(DOF, dtype=np.float64)
        else:
            dq_values = _first_joint_vector(dq, name="dq")
        self.data.qpos[:] = self._base_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[self.joint_qpos_addresses] = q_values
        self.data.qvel[self.joint_qvel_addresses] = dq_values
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._timestamp_s = 0.0
        self._last_command_torque.fill(0.0)
        self._last_applied_torque = self._read_applied_torque()
        return self.state()

    def state(self) -> MujocoState:
        """Return a defensive copy of the current arm state."""

        qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()
        return MujocoState(
            timestamp_s=float(self._timestamp_s),
            q=qpos[self.joint_qpos_addresses].copy(),
            dq=qvel[self.joint_qvel_addresses].copy(),
            applied_torque=self._read_applied_torque(),
            command_torque=self._last_command_torque.copy(),
            qpos=qpos,
            qvel=qvel,
        )

    @property
    def q(self) -> np.ndarray:
        return self.state().q

    @property
    def dq(self) -> np.ndarray:
        return self.state().dq

    @property
    def applied_torque(self) -> np.ndarray:
        return self._read_applied_torque()

    def _read_applied_torque(self) -> np.ndarray:
        # qfrc_actuator is indexed by generalized velocity (dof), not qpos.
        return np.asarray(
            self.data.qfrc_actuator[self.joint_qvel_addresses],
            dtype=np.float64,
        ).copy()

    def _clip_torque(self, values: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(values, dtype=np.float64),
            -self.torque_limits_nm,
            self.torque_limits_nm,
        )

    def _resolve_command(self, command: MujocoCommand) -> np.ndarray:
        mode = _normalize_mode(command.mode)
        q = np.asarray(self.data.qpos[self.joint_qpos_addresses], dtype=np.float64)
        dq = np.asarray(self.data.qvel[self.joint_qvel_addresses], dtype=np.float64)
        if mode == "q":
            q_target = _first_joint_vector(command.q_target, name="q_target")
            dq_target = (
                np.zeros(DOF, dtype=np.float64)
                if command.dq_target is None
                else _first_joint_vector(command.dq_target, name="dq_target")
            )
            return self._clip_torque(self.q_kp * (q_target - q) + self.q_kd * (dq_target - dq))
        if mode == "mtc":
            q_target = _first_joint_vector(command.q_target, name="q_target")
            dq_target = (
                np.zeros(DOF, dtype=np.float64)
                if command.dq_target is None
                else _first_joint_vector(command.dq_target, name="dq_target")
            )
            tau_target = (
                np.zeros(DOF, dtype=np.float64)
                if command.tau_target is None
                else _first_joint_vector(command.tau_target, name="tau_target")
            )
            kp = self.mit_kp if command.kp is None else _as_joint_vector(command.kp, name="kp", nonnegative=True)
            kd = self.mit_kd if command.kd is None else _as_joint_vector(command.kd, name="kd", nonnegative=True)
            return self._clip_torque(tau_target + kp * (q_target - q) + kd * (dq_target - dq))
        value = command.tau_command
        if value is None:
            value = command.tau_target
        return self._clip_torque(_first_joint_vector(value, name="tau_command"))

    def step(
        self,
        mode_or_command: str | MujocoCommand | Mapping[str, Any] | Any,
        *,
        q_target: Sequence[float] | np.ndarray | None = None,
        dq_target: Sequence[float] | np.ndarray | None = None,
        tau_target: Sequence[float] | np.ndarray | None = None,
        tau_command: Sequence[float] | np.ndarray | None = None,
        kp: Sequence[float] | np.ndarray | float | None = None,
        kd: Sequence[float] | np.ndarray | float | None = None,
    ) -> MujocoState:
        """Apply one command and integrate exactly ``control_dt_s``.

        ``mode_or_command`` may be a mode string, :class:`MujocoCommand`, a
        mapping with matching keys, or an existing ``InferenceOutput``.  For an
        inference output, use :meth:`step_output` when readability matters.
        """

        if isinstance(mode_or_command, MujocoCommand):
            command = mode_or_command
        elif isinstance(mode_or_command, str):
            command = MujocoCommand(
                mode=_normalize_mode(mode_or_command),
                q_target=q_target,
                dq_target=dq_target,
                tau_target=tau_target,
                tau_command=tau_command,
                kp=None if kp is None else _as_joint_vector(kp, name="kp", nonnegative=True),
                kd=None if kd is None else _as_joint_vector(kd, name="kd", nonnegative=True),
            )
        elif isinstance(mode_or_command, Mapping):
            values = dict(mode_or_command)
            mode_value = values.pop("mode", None)
            if mode_value is None:
                mode_value = values.pop("control_mode", "tau")
            command = MujocoCommand(
                mode=_normalize_mode(mode_value),
                q_target=values.pop("q_target", values.pop("joint_position_target", q_target)),
                dq_target=values.pop("dq_target", values.pop("joint_velocity_target", dq_target)),
                tau_target=values.pop("tau_target", values.pop("torque_target", tau_target)),
                tau_command=values.pop("tau_command", tau_command),
                kp=values.pop("kp", values.pop("mit_kp", kp)),
                kd=values.pop("kd", values.pop("mit_kd", kd)),
            )
        else:
            command = _extract_output_command(mode_or_command)
        self._last_command_torque.fill(0.0)
        for _ in range(self.substeps):
            torque = self._resolve_command(command)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.actuator_ids] = torque
            self.mujoco.mj_step(self.model, self.data)
            self._last_command_torque = torque.copy()
        self._timestamp_s += self.substeps * self.physics_dt_s
        self._last_applied_torque = self._read_applied_torque()
        return self.state()

    def step_output(self, output: Any, *, mode: str | None = None) -> MujocoState:
        """Execute an ``InferenceOutput``-like object in the simulation."""

        return self.step(_extract_output_command(output, mode=mode))


# Short names make the backend convenient to discover without changing the
# canonical class name used in documentation.
MujocoSimulationBackend = MujocoDynamicsBackend
MujocoBackend = MujocoDynamicsBackend


__all__ = [
    "MujocoBackendConfig",
    "MujocoCommand",
    "MujocoBackend",
    "MujocoDynamicsBackend",
    "MujocoSimulationBackend",
    "MujocoState",
]
