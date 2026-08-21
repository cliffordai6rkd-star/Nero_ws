from __future__ import annotations

import numpy as np
import pytest

from inference.config import WrenchVisualizationConfig
from inference.runtime import NeroInferenceRuntime, _dp_contact_threshold_n
from inference.wrench_visualization import (
    InferenceWrenchPlotter,
    WrenchVisualizationBuffer,
    WrenchVisualizationSample,
    _resultant_force,
)


def test_resultant_force_uses_only_linear_force_components() -> None:
    wrench = np.asarray(
        [
            [3.0, 4.0, 0.0, 100.0, 200.0, 300.0],
            [0.0, 0.0, 12.0, -500.0, 0.0, 0.0],
        ]
    )

    assert _resultant_force(wrench) == pytest.approx([5.0, 12.0])


def test_plotter_accepts_dp_contact_threshold() -> None:
    plotter = InferenceWrenchPlotter(
        WrenchVisualizationConfig(),
        contact_threshold_n=0.6,
    )

    assert plotter.contact_threshold_n == pytest.approx(0.6)


def test_runtime_reads_contact_threshold_restored_in_dp() -> None:
    detector = type("Detector", (), {"threshold": 0.6})()
    dp = type("DP", (), {"contact_detector": detector})()
    pipeline = type("Pipeline", (), {"dp": dp})()

    assert _dp_contact_threshold_n(pipeline) == pytest.approx(0.6)
    assert _dp_contact_threshold_n(object()) is None


def test_visualization_buffer_keeps_raw_and_processed_wrench_histories() -> None:
    buffer = WrenchVisualizationBuffer(window_s=10.0)
    raw = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    processed = np.zeros(6)
    buffer.append(
        WrenchVisualizationSample(
            timestamp_us=1_000_000,
            raw_wrench=raw,
            processed_wrench=processed,
            predicted_wrench=None,
        )
    )

    timestamps, raw_history, processed_history, _ = buffer.arrays()
    assert timestamps.shape == (1,)
    np.testing.assert_allclose(raw_history[0], raw)
    np.testing.assert_allclose(processed_history[0], processed)


def test_visualization_gate_zeros_all_wrench_components_below_threshold() -> None:
    runtime = object.__new__(NeroInferenceRuntime)
    runtime._dp_contact_threshold_n = 0.6
    runtime._dp_contact_force_dims = (0, 1, 2)

    above = runtime._apply_dp_contact_gate_for_visualization(
        np.asarray([1.0, 0.0, 0.0, 4.0, 5.0, 6.0])
    )
    equal = runtime._apply_dp_contact_gate_for_visualization(
        np.asarray([0.6, 0.0, 0.0, 4.0, 5.0, 6.0])
    )
    below = runtime._apply_dp_contact_gate_for_visualization(
        np.asarray([0.2, 0.0, 0.0, 4.0, 5.0, 6.0])
    )

    np.testing.assert_allclose(above, [1.0, 0.0, 0.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(equal, np.zeros(6))
    np.testing.assert_allclose(below, np.zeros(6))
