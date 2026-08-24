from __future__ import annotations

import numpy as np
import pytest

from nero_collection.config import RealtimePlotConfig, _parse_realtime_plot
from nero_collection.realtime_plot import _MatplotlibPlotWindow, _set_dynamic_ylim


def test_realtime_plot_parses_thresholds_and_request_spelling_aliases() -> None:
    config = _parse_realtime_plot(
        {
            "precontact_threshold": 1.5,
            "contact_threshold": 3.0,
        }
    )
    assert config.precontact_threshold == pytest.approx(1.5)
    assert config.contact_threshold == pytest.approx(3.0)
    assert config.precontact_threshhold == pytest.approx(1.5)
    assert config.comntact_treshhold == pytest.approx(3.0)

    aliases = _parse_realtime_plot(
        {"precontact_threshhold": 2.0, "comntact_treshhold": 4.0}
    )
    assert aliases.precontact_threshold == pytest.approx(2.0)
    assert aliases.contact_threshold == pytest.approx(4.0)


@pytest.mark.parametrize("field", ("precontact_threshold", "contact_threshold"))
def test_realtime_plot_rejects_negative_threshold(field: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _parse_realtime_plot({field: -0.1})


def test_dynamic_ylim_includes_configured_thresholds() -> None:
    class Axis:
        limits = None

        def set_ylim(self, lower, upper):
            self.limits = (lower, upper)

    axis = Axis()
    _set_dynamic_ylim(axis, np.asarray([-0.2, 0.5]), (4.0, 2.0))
    assert axis.limits == pytest.approx((-4.32, 4.32))


def test_matplotlib_plot_draws_red_threshold_lines_on_two_tau_ext_norm_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    config = RealtimePlotConfig(
        enabled=True,
        precontact_threshold=1.5,
        contact_threshold=3.0,
    )
    window = _MatplotlibPlotWindow(config)
    try:
        assert len(window.threshold_lines) == 4
        assert {line.axes for line in window.threshold_lines} == {
            window.axes[1],
            window.axes[3],
        }
        assert all(line.get_color() == "red" for line in window.threshold_lines)
        assert all(line.get_linestyle() == "--" for line in window.threshold_lines)
    finally:
        window.close()
