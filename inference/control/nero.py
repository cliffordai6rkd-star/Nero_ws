"""Compatibility controller for existing Nero pipeline outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


class NeroPipelineOutputController:
    """Keep legacy pipeline output semantics out of ``NeroInferenceRuntime``."""

    def __init__(self, *, arm: Any, config: Any, command_enabled: bool) -> None:
        self.arm = arm
        self.config = config
        self.command_enabled = bool(command_enabled)
        self._set_q_cmd = None

    def bind_q_command_sink(self, sink) -> None:
        self._set_q_cmd = sink

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset_episode(self) -> None:
        return None

    def send(self, observation, output) -> np.ndarray | None:
        if not self.command_enabled:
            return None
        if self.config.predictor.enabled:
            control_mode = getattr(output, "control_mode", None)
            if control_mode is None:
                # Legacy outputs predate ``control_mode``.  Use the selected
                # execution mode so a configured WM ``tau`` path still gets
                # the pure-torque transport contract.
                control_mode = getattr(self.config.execution, "mode", None)
            if control_mode == "q":
                if output.joint_position_command is None:
                    raise RuntimeError(
                        "contact WM q mode did not produce a joint-position command"
                    )
                q_cmd = np.asarray(output.joint_position_command, dtype=np.float64)
                self.arm.command_joint_positions(q_cmd)
            elif control_mode in {"mtc", "mit"}:
                q_target = getattr(output, "joint_position_target", None)
                dq_target = getattr(output, "joint_velocity_target", None)
                tau_target = getattr(output, "torque_target", None)
                if q_target is None or dq_target is None or tau_target is None:
                    raise RuntimeError(
                        "MTC mode requires q/dq/tau targets"
                    )
                q_cmd = np.asarray(q_target, dtype=np.float64)
                kp = np.asarray(
                    getattr(output, "mit_kp", None)
                    if getattr(output, "mit_kp", None) is not None
                    else self.config.execution.mit_kp,
                    dtype=np.float64,
                )
                kd = np.asarray(
                    getattr(output, "mit_kd", None)
                    if getattr(output, "mit_kd", None) is not None
                    else self.config.execution.mit_kd,
                    dtype=np.float64,
                )
                if kp.ndim == 0:
                    kp = np.repeat(kp, 7)
                if kd.ndim == 0:
                    kd = np.repeat(kd, 7)
                self.arm.command_joint_impedance(
                    q=q_cmd,
                    v_des=np.asarray(dq_target, dtype=np.float64),
                    kp=kp,
                    kd=kd,
                    # Firmware applies the PD terms; the pipeline supplies the
                    # remaining WM/MTC feed-forward or residual here.
                    t_ff=np.asarray(tau_target, dtype=np.float64),
                )
            elif control_mode == "tau":
                # ``tau`` is a direct WM torque contract.  Use the MIT packet
                # only as a transport envelope with both firmware feedback
                # gains disabled, so the physical command is exactly the
                # filtered ``tau_pred`` carried by ``output.tau_command``.
                q_cmd = np.asarray(observation.q, dtype=np.float64)
                self.arm.command_joint_impedance(
                    q=q_cmd,
                    v_des=np.zeros(7, dtype=np.float64),
                    kp=np.zeros(7, dtype=np.float64),
                    kd=np.zeros(7, dtype=np.float64),
                    t_ff=np.asarray(output.tau_command, dtype=np.float64),
                )
            else:
                raise ValueError(f"unsupported inference control mode {control_mode!r}")
        else:
            if output.joint_position_command is None:
                raise RuntimeError(
                    "direct IK pipeline did not produce a joint-position command"
                )
            q_cmd = np.asarray(output.joint_position_command, dtype=np.float64)
            self.arm.command_joint_positions(q_cmd)
        if self._set_q_cmd is not None:
            self._set_q_cmd(q_cmd)
        return q_cmd
