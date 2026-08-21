from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

from inference.config import WrenchVisualizationConfig


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WrenchVisualizationSample:
    timestamp_us: int
    raw_wrench: np.ndarray
    processed_wrench: np.ndarray
    predicted_wrench: np.ndarray | None = None


@dataclass(frozen=True)
class _ClearHistory:
    pass


class WrenchVisualizationBuffer:
    """Bounded time history shared by tests and the matplotlib process."""

    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._timestamps_s: deque[float] = deque()
        self._raw: deque[np.ndarray] = deque()
        self._processed: deque[np.ndarray] = deque()
        self._predicted: deque[np.ndarray] = deque()

    def append(self, sample: WrenchVisualizationSample) -> None:
        raw = _wrench_vector("raw_wrench", sample.raw_wrench)
        processed = _wrench_vector("processed_wrench", sample.processed_wrench)
        if sample.predicted_wrench is None:
            predicted = np.full(6, np.nan, dtype=np.float64)
        else:
            predicted = _wrench_vector("predicted_wrench", sample.predicted_wrench)
        timestamp_s = int(sample.timestamp_us) * 1.0e-6
        if self._timestamps_s and timestamp_s <= self._timestamps_s[-1]:
            return
        self._timestamps_s.append(timestamp_s)
        self._raw.append(raw)
        self._processed.append(processed)
        self._predicted.append(predicted)
        cutoff_s = timestamp_s - self.window_s
        while self._timestamps_s and self._timestamps_s[0] < cutoff_s:
            self._timestamps_s.popleft()
            self._raw.popleft()
            self._processed.popleft()
            self._predicted.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._timestamps_s:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, 6), dtype=np.float64),
                np.empty((0, 6), dtype=np.float64),
                np.empty((0, 6), dtype=np.float64),
            )
        timestamps = np.asarray(self._timestamps_s, dtype=np.float64)
        return (
            timestamps - timestamps[-1],
            np.stack(self._raw, axis=0),
            np.stack(self._processed, axis=0),
            np.stack(self._predicted, axis=0),
        )

    def clear(self) -> None:
        self._timestamps_s.clear()
        self._raw.clear()
        self._processed.clear()
        self._predicted.clear()


class InferenceWrenchPlotter:
    """Plot raw and processed wrench histories without blocking inference."""

    def __init__(
        self,
        config: WrenchVisualizationConfig,
        *,
        contact_threshold_n: float | None = None,
    ) -> None:
        self.config = config
        self.contact_threshold_n = _optional_nonnegative_scalar(
            "contact_threshold_n",
            contact_threshold_n,
        )
        self._queue = None
        self._process = None
        self._closed = False
        self._process_failure_logged = False

    def start(self) -> None:
        if not self.config.enabled:
            return
        if find_spec("matplotlib") is None:
            raise RuntimeError(
                "wrench_visualization.enabled=true requires matplotlib; "
                "install matplotlib>=3.7"
            )
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=512)
        self._process = context.Process(
            target=_plot_worker,
            args=(self.config, self._queue, self.contact_threshold_n),
            name="nero-inference-wrench-plot",
            daemon=True,
        )
        self._process.start()
        log.info(
            "inference resultant-force visualization started window=%.1fs "
            "update=%.1fHz DP_contact_threshold=%s",
            self.config.window_s,
            self.config.update_rate_hz,
            (
                "none"
                if self.contact_threshold_n is None
                else f"{self.contact_threshold_n:.4g}N"
            ),
        )

    def append(
        self,
        timestamp_us: int,
        raw_wrench: np.ndarray,
        processed_wrench: np.ndarray,
        predicted_wrench: np.ndarray | None = None,
    ) -> None:
        if not self.config.enabled or self._closed:
            return
        if self._process is None or self._queue is None:
            raise RuntimeError("Inference wrench visualization has not been started")
        if not self._process.is_alive():
            self._closed = True
            if not self._process_failure_logged:
                log.warning(
                    "inference wrench visualization window closed or unavailable; "
                    "inference continues"
                )
                self._process_failure_logged = True
            return
        sample = WrenchVisualizationSample(
            timestamp_us=int(timestamp_us),
            raw_wrench=_wrench_vector("raw_wrench", raw_wrench),
            processed_wrench=_wrench_vector("processed_wrench", processed_wrench),
            predicted_wrench=(
                None
                if predicted_wrench is None
                else _wrench_vector("predicted_wrench", predicted_wrench)
            ),
        )
        _put_latest(self._queue, sample)

    def clear_history(self) -> None:
        if not self.config.enabled or self._closed:
            return
        if self._process is None or self._queue is None:
            raise RuntimeError("Inference wrench visualization has not been started")
        if not self._process.is_alive():
            self._closed = True
            return
        _put_latest(self._queue, _ClearHistory())

    def close(self) -> None:
        if self._queue is not None:
            _put_latest(self._queue, None)
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._queue is not None:
            self._queue.close()
        self._queue = None
        self._process = None
        self._closed = True


class _MatplotlibWrenchWindow:
    _LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
    _COLORS = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    )

    def __init__(
        self,
        config: WrenchVisualizationConfig,
        contact_threshold_n: float | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        self.config = config
        self.contact_threshold_n = _optional_nonnegative_scalar(
            "contact_threshold_n",
            contact_threshold_n,
        )
        self.buffer = WrenchVisualizationBuffer(config.window_s)
        self.plt = plt
        plt.ion()
        self.figure, axes_grid = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        try:
            self.figure.canvas.manager.set_window_title(
                "Nero wrench: before and after DP observation filtering"
            )
        except AttributeError:
            pass
        self.raw_force_axis = axes_grid[0, 0]
        self.raw_components_axis = axes_grid[0, 1]
        self.processed_force_axis = axes_grid[1, 0]
        self.processed_components_axis = axes_grid[1, 1]
        self.axes = (
            self.raw_force_axis,
            self.raw_components_axis,
            self.processed_force_axis,
            self.processed_components_axis,
        )
        self.raw_force_line = self.raw_force_axis.plot(
            [], [], color="tab:blue", linewidth=1.3, label="before filter ||F||"
        )[0]
        self.processed_force_line = self.processed_force_axis.plot(
            [], [], color="tab:blue", linewidth=1.3, label="DP observation ||F||"
        )[0]
        if self.contact_threshold_n is not None:
            self.processed_force_axis.axhline(
                self.contact_threshold_n,
                color="tab:red",
                linewidth=1.0,
                linestyle=":",
                label=f"DP contact threshold {self.contact_threshold_n:g} N",
            )
        self.raw_component_lines = self._make_component_lines(
            self.raw_components_axis,
            "before filter",
        )
        self.processed_component_lines = self._make_component_lines(
            self.processed_components_axis,
            "DP observation",
        )
        self._configure_force_axis(
            self.raw_force_axis,
            "Before tau_ext_filter (diagnostic only)",
        )
        self._configure_force_axis(
            self.processed_force_axis,
            "After tau_ext_filter and DP contact gate",
        )
        self.raw_force_axis.legend(loc="upper left", fontsize=9)
        self.processed_force_axis.legend(loc="upper left", fontsize=9)
        self.figure.suptitle(
            "External wrench in DP frame: raw diagnostic vs model observation"
        )
        self.figure.tight_layout()
        self.figure.show()
        self.process_events()

    def _configure_force_axis(self, axis, title: str) -> None:
        axis.set_title(title)
        axis.set_xlabel("time [s], live")
        axis.set_ylabel("force magnitude [N]")
        axis.set_xlim(-self.config.window_s, 0.0)
        axis.set_ylim(0.0, _initial_force_ylim(self.contact_threshold_n))
        axis.grid(True, alpha=0.25)

    def _make_component_lines(self, axis, prefix: str):
        lines = []
        for label, color in zip(self._LABELS, self._COLORS):
            lines.append(
                axis.plot(
                    [], [], color=color, linewidth=1.1, label=f"{prefix} {label}"
                )[0]
            )
        axis.set_title(f"{prefix.capitalize()} wrench components")
        axis.set_xlabel("time [s], live")
        axis.set_ylabel("N / N.m")
        axis.set_xlim(-self.config.window_s, 0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", fontsize=8, ncol=2)
        return tuple(lines)

    def append(self, sample: WrenchVisualizationSample) -> None:
        self.buffer.append(sample)

    def render(self) -> None:
        relative_time, raw, processed, predicted = self.buffer.arrays()
        if relative_time.size:
            raw_force = _resultant_force(raw)
            processed_force = _resultant_force(processed)
            self.raw_force_line.set_data(relative_time, raw_force)
            self.processed_force_line.set_data(relative_time, processed_force)
            for index, line in enumerate(self.raw_component_lines):
                line.set_data(relative_time, raw[:, index])
            for index, line in enumerate(self.processed_component_lines):
                line.set_data(relative_time, processed[:, index])
            for axis in self.axes:
                axis.set_xlim(-self.config.window_s, 0.0)
            _set_force_ylim(
                self.raw_force_axis,
                raw_force,
                np.empty((0,), dtype=np.float64),
                None,
            )
            _set_force_ylim(
                self.processed_force_axis,
                processed_force,
                np.empty((0,), dtype=np.float64),
                self.contact_threshold_n,
            )
            _set_component_ylim(self.raw_components_axis, raw)
            _set_component_ylim(self.processed_components_axis, processed)
            self.figure.canvas.draw()
        self.process_events()

    def clear(self) -> None:
        self.buffer.clear()
        self.raw_force_line.set_data([], [])
        self.processed_force_line.set_data([], [])
        for line in (*self.raw_component_lines, *self.processed_component_lines):
            line.set_data([], [])
        self._configure_force_axis(
            self.raw_force_axis,
            "Before tau_ext_filter (diagnostic only)",
        )
        self._configure_force_axis(
            self.processed_force_axis,
            "After tau_ext_filter and DP contact gate",
        )
        for axis in (self.raw_components_axis, self.processed_components_axis):
            axis.set_xlim(-self.config.window_s, 0.0)
            axis.set_ylim(-0.05, 0.05)
        self.figure.canvas.draw()
        self.process_events()

    def process_events(self) -> None:
        self.figure.canvas.flush_events()

    def is_open(self) -> bool:
        return bool(self.plt.fignum_exists(self.figure.number))

    def close(self) -> None:
        self.plt.close(self.figure)


def _plot_worker(
    config: WrenchVisualizationConfig,
    sample_queue,
    contact_threshold_n: float | None = None,
) -> None:
    try:
        window = _MatplotlibWrenchWindow(config, contact_threshold_n)
    except Exception:
        log.exception("failed to start inference wrench visualization")
        return
    next_render_s = time.monotonic()
    stop = False
    try:
        while not stop and window.is_open():
            timeout_s = max(0.0, min(0.05, next_render_s - time.monotonic()))
            try:
                item = sample_queue.get(timeout=timeout_s)
                received = True
            except queue.Empty:
                item = None
                received = False
            if received:
                items = [item]
                while True:
                    try:
                        items.append(sample_queue.get_nowait())
                    except queue.Empty:
                        break
                for queued_item in items:
                    if queued_item is None:
                        stop = True
                        break
                    if isinstance(queued_item, _ClearHistory):
                        window.clear()
                    else:
                        window.append(queued_item)
            now_s = time.monotonic()
            if now_s >= next_render_s:
                window.render()
                next_render_s = now_s + 1.0 / config.update_rate_hz
            else:
                window.process_events()
    except Exception:
        log.exception("inference wrench visualization failed")
    finally:
        window.close()


def _put_latest(sample_queue, value) -> None:
    try:
        sample_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        sample_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        sample_queue.put_nowait(value)
    except queue.Full:
        pass


def _wrench_vector(name: str, value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 6 or not np.isfinite(vector).all():
        raise RuntimeError(
            f"Wrench visualization requires a finite 6D {name}; got {vector}"
        )
    return vector.copy()


def _resultant_force(wrench: np.ndarray) -> np.ndarray:
    """Return ||[Fx,Fy,Fz]||; force and moment units cannot share one norm."""
    value = np.asarray(wrench, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 6:
        raise ValueError(f"wrench must end in six components, got {value.shape}")
    return np.linalg.norm(value[..., :3], axis=-1)


def _set_force_ylim(
    axis,
    primary: np.ndarray,
    secondary: np.ndarray,
    contact_threshold_n: float | None,
) -> None:
    finite_secondary = secondary[np.isfinite(secondary)]
    values = (
        primary
        if finite_secondary.size == 0
        else np.concatenate((primary, finite_secondary))
    )
    data_max = float(np.max(values))
    if contact_threshold_n is not None:
        data_max = max(data_max, contact_threshold_n)
    padding = max(data_max * 0.08, 0.05)
    axis.set_ylim(0.0, max(data_max + padding, 0.1))


def _set_component_ylim(axis, wrench: np.ndarray) -> None:
    value = np.asarray(wrench, dtype=np.float64)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        axis.set_ylim(-0.05, 0.05)
        return
    limit = max(float(np.max(np.abs(finite))) * 1.08, 0.05)
    axis.set_ylim(-limit, limit)


def _initial_force_ylim(contact_threshold_n: float | None) -> float:
    if contact_threshold_n is None:
        return 0.1
    return max(contact_threshold_n * 1.08, contact_threshold_n + 0.05, 0.1)


def _optional_nonnegative_scalar(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return scalar
