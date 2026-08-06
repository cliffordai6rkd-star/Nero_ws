from __future__ import annotations

import numpy as np
import pytest

from inference.config import TorqueFilterConfig
from inference.torque_filter import CausalTorqueCommandFilter


def _vector(value: float) -> np.ndarray:
    return np.full(7, value, dtype=np.float64)


def test_causal_median_rejects_isolated_torque_spike() -> None:
    torque_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=3,
            lowpass_cutoff_hz=None,
            rate_limit_nm_s=None,
        )
    )

    first = torque_filter.apply(_vector(1.0), dt_s=0.01, initial_tau=_vector(0.0))
    second = torque_filter.apply(_vector(1.0), dt_s=0.01, initial_tau=_vector(0.0))
    spike = torque_filter.apply(_vector(100.0), dt_s=0.01, initial_tau=_vector(0.0))

    np.testing.assert_allclose(first, _vector(0.0))
    np.testing.assert_allclose(second, _vector(1.0))
    np.testing.assert_allclose(spike, _vector(1.0))


def test_torque_slew_rate_is_a_hard_per_axis_limit() -> None:
    torque_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=1,
            lowpass_cutoff_hz=None,
            rate_limit_nm_s=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
        )
    )

    output = torque_filter.apply(
        _vector(100.0),
        dt_s=0.1,
        initial_tau=_vector(0.0),
    )

    np.testing.assert_allclose(
        output,
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    )


def test_lowpass_is_causal_and_reset_clears_filter_state() -> None:
    torque_filter = CausalTorqueCommandFilter(
        TorqueFilterConfig(
            median_window=1,
            lowpass_cutoff_hz=1.0,
            rate_limit_nm_s=None,
        )
    )
    output = torque_filter.apply(
        _vector(10.0),
        dt_s=0.1,
        initial_tau=_vector(0.0),
    )
    expected = 10.0 * (1.0 - np.exp(-2.0 * np.pi * 0.1))
    np.testing.assert_allclose(output, _vector(expected))

    torque_filter.reset()
    restarted = torque_filter.apply(
        _vector(10.0),
        dt_s=0.1,
        initial_tau=_vector(5.0),
    )
    np.testing.assert_allclose(
        restarted,
        _vector(5.0 + (10.0 - 5.0) * (1.0 - np.exp(-2.0 * np.pi * 0.1))),
    )


def test_filter_rejects_invalid_dt() -> None:
    torque_filter = CausalTorqueCommandFilter(TorqueFilterConfig())
    with pytest.raises(ValueError, match="dt_s"):
        torque_filter.apply(_vector(0.0), dt_s=0.0, initial_tau=_vector(0.0))
