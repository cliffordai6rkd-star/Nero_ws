from __future__ import annotations

import numpy as np

from inference.config import TorqueFilterConfig
from inference.torque_filter import CausalTorqueCommandFilter


def test_isolated_torque_spike_is_rejected_by_median_filter() -> None:
    command_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=3,
            lowpass_cutoff_hz=None,
            rate_limit_nm_s=None,
        )
    )
    zeros = np.zeros(7)

    normal = command_filter.apply(zeros, dt_s=0.01, initial_tau=zeros)
    spike = command_filter.apply(
        np.full(7, 20.0),
        dt_s=0.01,
        initial_tau=zeros,
    )
    recovered = command_filter.apply(zeros, dt_s=0.01, initial_tau=zeros)

    np.testing.assert_allclose(normal, zeros)
    np.testing.assert_allclose(spike, zeros)
    np.testing.assert_allclose(recovered, zeros)


def test_lowpass_and_hard_rate_limit_bound_each_command_step() -> None:
    command_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=1,
            lowpass_cutoff_hz=20.0,
            rate_limit_nm_s=10.0,
        )
    )
    zeros = np.zeros(7)

    first = command_filter.apply(
        np.full(7, 20.0),
        dt_s=0.01,
        initial_tau=zeros,
    )
    second = command_filter.apply(
        np.full(7, 20.0),
        dt_s=0.01,
        initial_tau=zeros,
    )

    np.testing.assert_allclose(first, np.full(7, 0.1))
    np.testing.assert_allclose(second, np.full(7, 0.2))


def test_filter_reset_uses_new_measured_initial_torque() -> None:
    command_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=1,
            lowpass_cutoff_hz=None,
            rate_limit_nm_s=5.0,
        )
    )
    command_filter.apply(np.ones(7), dt_s=0.1, initial_tau=np.zeros(7))
    command_filter.reset()

    result = command_filter.apply(
        np.full(7, 10.0),
        dt_s=0.1,
        initial_tau=np.full(7, 4.0),
    )

    np.testing.assert_allclose(result, np.full(7, 4.5))
