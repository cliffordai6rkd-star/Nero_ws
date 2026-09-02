"""Multi-target torque controller for the native Contact WM contract.

MTC (multi-target controller) combines a data-style zero-velocity impedance
candidate with the torque predicted by the world model.  The controller is a
pure numerical component; the legacy runtime adapter is responsible for
transporting its result through the arm API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


DOF = 7


def _vector(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (DOF,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite seven-vector")
    return result.copy()


def _gain_vector(name: str, value: Any) -> np.ndarray:
    """Normalize a scalar or seven-joint gain to the controller width."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.repeat(result, DOF)
    result = result.reshape(-1)
    if result.shape != (DOF,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or seven-vector")
    return result.copy()


@dataclass(frozen=True)
class MTCResult:
    """One MTC computation and all of its independently auditable terms."""

    q_cmd: np.ndarray
    gravity: np.ndarray
    tau_pd: np.ndarray
    tau_qv: np.ndarray
    tau_pred: np.ndarray
    tau_command: np.ndarray
    alpha: float

    @property
    def tau_feedforward_fixed_gains(self) -> np.ndarray:
        """Feed-forward needed when the firmware keeps the full q/v gains.

        The firmware will add ``tau_pd`` through its PD path.  This residual
        is useful to the MIT transport adapter when the configured data-side
        gains must remain unchanged.
        """

        return self.tau_command - self.tau_pd


class MTCController:
    """Blend q/v/gravity and WM-predicted total torque candidates.

    ``tau_qv`` follows the data collection semantics exactly: the desired
    velocity is zero and the damping term uses the measured current velocity.
    ``alpha`` is the WM total-torque weight; the q/v candidate receives
    ``1 - alpha``.
    ``wm_delta`` reconstructs the command represented by the Contact WM contract as
    ``q_hat + delta_q_hat``; ``wm_state`` uses ``q_hat`` directly.
    """

    def __init__(
        self,
        *,
        model: Any,
        kp: Any,
        kd: Any,
        alpha: float,
        q_cmd_source: str = "wm_delta",
        gravity_torque_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.model = model
        self.kp = _gain_vector("MTC kp", kp)
        self.kd = _gain_vector("MTC kd", kd)
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("MTC kp and kd must be non-negative")
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("MTC alpha must be a finite value in [0, 1]")
        source = str(q_cmd_source).strip().lower().replace("-", "_")
        source = {
            "q_hat": "wm_state",
            "qhat": "wm_state",
            "wm_q": "wm_state",
            "q_cmd_hat": "wm_delta",
            "qcmd_hat": "wm_delta",
            "wm_q_cmd": "wm_delta",
        }.get(source, source)
        if source not in {"wm_state", "wm_delta"}:
            raise ValueError(
                "MTC q_cmd_source must be 'wm_state'/'q_hat' or "
                "'wm_delta'/'q_cmd_hat'"
            )
        self.q_cmd_source = source
        self._gravity_torque_fn = gravity_torque_fn

    def resolve_q_cmd(
        self,
        q_hat: Any,
        delta_q_hat: Any | None = None,
    ) -> np.ndarray:
        q_value = _vector("MTC q_hat", q_hat)
        if self.q_cmd_source == "wm_state":
            return q_value
        if delta_q_hat is None:
            raise ValueError("MTC wm_delta source requires delta_q_hat")
        return q_value + _vector("MTC delta_q_hat", delta_q_hat)

    def compute(
        self,
        *,
        q: Any,
        dq: Any,
        tau_pred: Any,
        q_hat: Any,
        delta_q_hat: Any | None = None,
    ) -> MTCResult:
        measured_q = _vector("MTC measured q", q)
        measured_dq = _vector("MTC measured dq", dq)
        predicted_tau = _vector("MTC tau_pred", tau_pred)
        q_cmd = self.resolve_q_cmd(q_hat, delta_q_hat)
        gravity = self._gravity_torque(measured_q)
        tau_pd = self.kp * (q_cmd - measured_q) - self.kd * measured_dq
        tau_qv = tau_pd + gravity
        # ``alpha`` is the WM total-torque weight for both synchronous and
        # timestamped execution paths.  The q/v candidate therefore receives
        # the complementary weight.
        tau_command = (1.0 - self.alpha) * tau_qv + self.alpha * predicted_tau
        return MTCResult(
            q_cmd=q_cmd,
            gravity=gravity,
            tau_pd=tau_pd,
            tau_qv=tau_qv,
            tau_pred=predicted_tau,
            tau_command=tau_command,
            alpha=self.alpha,
        )

    def compute_timestamped(
        self,
        *,
        q: Any,
        dq: Any,
        tau_ref: Any,
        q_ref: Any,
    ) -> MTCResult:
        """Compute the asynchronous WM-weighted blend at one control tick.

        ``alpha`` is the WM total-torque weight, matching :meth:`compute`.
        The q/v candidate includes the measured-state gravity torque so the
        asynchronous and synchronous paths have the same torque semantics.
        """
        measured_q = _vector("MTC measured q", q)
        measured_dq = _vector("MTC measured dq", dq)
        predicted_tau = _vector("MTC tau_ref", tau_ref)
        q_cmd = self.resolve_q_cmd(q_ref)
        gravity = self._gravity_torque(measured_q)
        tau_pd = self.kp * (q_cmd - measured_q) - self.kd * measured_dq
        tau_qv = tau_pd + gravity
        tau_command = (1.0 - self.alpha) * tau_qv + self.alpha * predicted_tau
        return MTCResult(
            q_cmd=q_cmd,
            gravity=gravity,
            tau_pd=tau_pd,
            tau_qv=tau_qv,
            tau_pred=predicted_tau,
            tau_command=tau_command,
            alpha=self.alpha,
        )

    def _gravity_torque(self, q: np.ndarray) -> np.ndarray:
        function = self._gravity_torque_fn
        if function is None:
            function = getattr(self.model, "gravity_torque", None)
        if callable(function):
            return _vector("MTC gravity torque", function(q))

        # PinocchioDynamicsModel historically exposed only ``snapshot``.  Keep
        # MTC usable with that model without forcing a second dynamics object.
        pin = getattr(self.model, "pin", None)
        pin_model = getattr(self.model, "model", None)
        pin_data = getattr(self.model, "data", None)
        if pin is not None and pin_model is not None and pin_data is not None:
            zeros = np.zeros(DOF, dtype=np.float64)
            return _vector(
                "MTC gravity torque",
                pin.rnea(pin_model, pin_data, q, zeros, zeros),
            )
        raise RuntimeError(
            "MTC requires a dynamics model exposing gravity_torque(q) or "
            "Pinocchio pin/model/data handles"
        )


__all__ = ["MTCController", "MTCResult"]
