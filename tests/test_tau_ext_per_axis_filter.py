from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import _parse_tau_ext_inference
from nero_collection.filters import CausalWindowLowPass


def test_tau_ext_filter_accepts_per_axis_window_and_cutoff(tmp_path: Path) -> None:
    config = _parse_tau_ext_inference(
        {
            "tau_ext_filter": {
                "mode": "moving_average",
                "window": [5, 10, 5, 10, 5, 5, 5],
                "cutoff_hz": [50, 15, 50, 15, 50, 50, 50],
            }
        },
        tmp_path,
    )

    assert config.tau_ext_filter.window == (5, 10, 5, 10, 5, 5, 5)
    assert config.tau_ext_filter.cutoff_hz == (50.0, 15.0, 50.0, 15.0, 50.0, 50.0, 50.0)


def test_tau_ext_filter_applies_window_per_axis() -> None:
    # A very high cutoff makes the one-pole stage effectively transparent, so
    # the expected values isolate the per-axis moving-average windows.
    online_filter = CausalWindowLowPass(
        "moving_average",
        window=(1, 3, 2),
        cutoff_hz=(1.0e6, 1.0e6, 1.0e6),
    )
    values = np.asarray(
        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]],
        dtype=np.float64,
    )
    actual = np.stack(
        [online_filter.apply(value, 1_000_000 * (index + 1)) for index, value in enumerate(values)]
    )
    expected = np.asarray(
        [[0.0, 0.0, 0.0], [3.0, 1.0, 1.5], [6.0, 3.0, 4.5]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-9)


@pytest.mark.parametrize(
    ("field", "value"),
    (("window", [5, 5]), ("cutoff_hz", [15.0, 15.0])),
)
def test_tau_ext_filter_rejects_non_seven_axis_values(
    field: str, value: list[float], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        _parse_tau_ext_inference({"tau_ext_filter": {field: value}}, tmp_path)


def test_tau_ext_filter_rejects_even_per_axis_median_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be odd"):
        _parse_tau_ext_inference(
            {
                "tau_ext_filter": {
                    "mode": "median",
                    "window": [5, 5, 4, 5, 5, 5, 5],
                }
            },
            tmp_path,
        )
