from __future__ import annotations

import numpy as np
import pytest

from inference.control.mtc import MTCController


class _Dynamics:
    def gravity_torque(self, q):
        # Keep the callback observable while returning a non-zero term.
        return np.asarray(q, dtype=np.float64) * 0.25


def test_mtc_blends_qv_and_predicted_torque():
    controller = MTCController(
        model=_Dynamics(),
        kp=np.full(7, 2.0),
        kd=np.full(7, 0.5),
        alpha=0.25,
        q_cmd_source="wm_state",
    )
    q = np.full(7, 0.4)
    dq = np.full(7, 0.2)
    q_hat = np.full(7, 0.7)
    tau_pred = np.full(7, 1.2)

    result = controller.compute(
        q=q,
        dq=dq,
        q_hat=q_hat,
        tau_pred=tau_pred,
    )

    gravity = q * 0.25
    tau_qv = 2.0 * (q_hat - q) + 0.5 * (0.0 - dq) + gravity
    # alpha is the WM total-torque weight; q/v receives the complementary weight.
    expected = 0.75 * tau_qv + 0.25 * tau_pred
    np.testing.assert_allclose(result.q_cmd, q_hat)
    np.testing.assert_allclose(result.gravity, gravity)
    np.testing.assert_allclose(result.tau_pd, 2.0 * (q_hat - q) - 0.5 * dq)
    np.testing.assert_allclose(result.tau_qv, tau_qv)
    np.testing.assert_allclose(result.tau_pred, tau_pred)
    np.testing.assert_allclose(result.tau_command, expected)
    np.testing.assert_allclose(result.tau_feedforward_fixed_gains, expected - result.tau_pd)


def test_timestamped_mtc_includes_gravity_with_same_alpha_semantics():
    controller = MTCController(
        model=_Dynamics(),
        kp=np.full(7, 2.0),
        kd=np.full(7, 0.5),
        alpha=0.25,
        q_cmd_source="wm_state",
    )
    q = np.full(7, 0.4)
    dq = np.full(7, 0.2)
    q_ref = np.full(7, 0.7)
    tau_ref = np.full(7, 1.2)

    result = controller.compute_timestamped(
        q=q,
        dq=dq,
        q_ref=q_ref,
        tau_ref=tau_ref,
    )

    gravity = q * 0.25
    tau_pd = 2.0 * (q_ref - q) - 0.5 * dq
    tau_qv = tau_pd + gravity
    expected = 0.75 * tau_qv + 0.25 * tau_ref
    np.testing.assert_allclose(result.gravity, gravity)
    np.testing.assert_allclose(result.tau_pd, tau_pd)
    np.testing.assert_allclose(result.tau_qv, tau_qv)
    np.testing.assert_allclose(result.tau_command, expected)
    np.testing.assert_allclose(result.tau_feedforward_fixed_gains, expected - tau_pd)


def test_mtc_wm_delta_reconstructs_q_command_and_validates_inputs():
    controller = MTCController(
        model=_Dynamics(),
        kp=1.0,
        kd=0.1,
        alpha=0.5,
        q_cmd_source="q_cmd_hat",
    )
    q_hat = np.linspace(0.0, 0.6, 7)
    delta = np.full(7, 0.2)
    np.testing.assert_allclose(controller.resolve_q_cmd(q_hat, delta), q_hat + delta)

    with pytest.raises(ValueError, match="requires delta_q_hat"):
        controller.resolve_q_cmd(q_hat)
    with pytest.raises(ValueError, match="alpha"):
        MTCController(model=_Dynamics(), kp=1.0, kd=1.0, alpha=1.1)
