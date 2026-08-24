"""Reuse the data-collection tau_ext plot during online inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from nero_collection.realtime_plot import RealtimeJointPlotter


class TauExtInferencePlotter:
    """Adapter from ``OnlineTauExtResult`` to the collection plot contract."""

    def __init__(self, collection_config: Any) -> None:
        self._plotter = RealtimeJointPlotter(
            collection_config.realtime_plot,
            collection_config.robot_states,
            collection_config.dynamics_processing,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._plotter.config.enabled)

    def start(self) -> None:
        self._plotter.start()

    def append(self, timestamp_us: int, tau_ext_cal: np.ndarray, tau_ext_pred: np.ndarray) -> None:
        if not self.enabled:
            return
        self._plotter.append(
            int(timestamp_us),
            {
                "tau_ext_cal": ("torque", np.asarray(tau_ext_cal, dtype=np.float64)),
                "tau_ext_pred": ("torque", np.asarray(tau_ext_pred, dtype=np.float64)),
            },
        )

    def clear_history(self) -> None:
        if self.enabled:
            self._plotter.clear_history()

    def close(self) -> None:
        self._plotter.close()

