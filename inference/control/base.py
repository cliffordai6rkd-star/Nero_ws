"""Robot command adapters for the modular inference core."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from inference.core.contracts import ControlTarget, Observation


class CallableRobotController:
    """Controller adapter useful for simulation and unit tests."""

    def __init__(self, send_fn: Callable[[Observation, ControlTarget | None], Any]) -> None:
        self.send_fn = send_fn

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset_episode(self) -> None:
        return None

    def send(self, observation: Observation, target: ControlTarget | None) -> Any:
        return self.send_fn(observation, target)


class ArmRobotController:
    """Translate a generic ``ControlTarget`` to the existing ArmInterface API."""

    def __init__(
        self,
        arm: Any,
        *,
        command_enabled: bool = True,
        default_kd: np.ndarray | None = None,
    ) -> None:
        self.arm = arm
        self.command_enabled = bool(command_enabled)
        self.default_kd = np.zeros(7, dtype=np.float64) if default_kd is None else np.asarray(default_kd, dtype=np.float64)
        if self.default_kd.shape != (7,):
            raise ValueError("default_kd must be a seven-joint vector")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset_episode(self) -> None:
        return None

    def send(self, observation: Observation, target: ControlTarget | None) -> Any | None:
        if target is None or not self.command_enabled:
            return None
        mode = target.mode
        if mode in {"position", "q"}:
            if target.q is None:
                raise ValueError("position control target requires q")
            return self.arm.command_joint_positions(target.q)
        if mode in {"torque", "tau"}:
            torque = target.torque
            if torque is None:
                raise ValueError("torque control target requires torque")
            q = observation.q if target.q is None else target.q
            return self.arm.command_joint_impedance(
                q=q,
                v_des=np.zeros(7, dtype=np.float64),
                kp=np.zeros(7, dtype=np.float64),
                kd=self.default_kd,
                t_ff=torque,
            )
        if mode == "mit":
            if target.q is None or target.dq is None or target.torque is None:
                raise ValueError("MIT target requires q, dq and torque")
            return self.arm.command_joint_impedance(
                q=target.q,
                v_des=target.dq,
                kp=np.asarray(target.metadata.get("kp", np.zeros(7)), dtype=np.float64),
                kd=np.asarray(target.metadata.get("kd", self.default_kd), dtype=np.float64),
                t_ff=target.torque,
            )
        raise ValueError(f"unsupported robot control mode {mode!r}")


__all__ = ["ArmRobotController", "CallableRobotController"]

