from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

from nero_collection.config import DynamicsProcessingConfig, RealtimePlotConfig, StateParamConfig

log = logging.getLogger(__name__)

_MINIMUM_Y_ABS_NM = 3.0


@dataclass(frozen=True)
class _RealtimeSample:
    timestamp_us: int
    tau_ext_cal: np.ndarray
    tau_ext_pred: np.ndarray


@dataclass(frozen=True)
class _ClearHistory:
    pass


class SlidingJointBuffer:
    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._timestamps_s: deque[float] = deque()
        self._tau_ext_cal: deque[np.ndarray] = deque()
        self._tau_ext_pred: deque[np.ndarray] = deque()

    def append(
        self,
        timestamp_us: int,
        tau_ext_cal: np.ndarray,
        tau_ext_pred: np.ndarray,
    ) -> None:
        tau_ext_cal = _plot_vector("tau_ext_cal", tau_ext_cal, 7)
        tau_ext_pred = _plot_vector("tau_ext_pred", tau_ext_pred, 7)
        timestamp_s = int(timestamp_us) / 1_000_000.0
        if self._timestamps_s and timestamp_s <= self._timestamps_s[-1]:
            return
        self._timestamps_s.append(timestamp_s)
        self._tau_ext_cal.append(tau_ext_cal)
        self._tau_ext_pred.append(tau_ext_pred)

        cutoff_s = timestamp_s - self.window_s
        while self._timestamps_s and self._timestamps_s[0] < cutoff_s:
            self._timestamps_s.popleft()
            self._tau_ext_cal.popleft()
            self._tau_ext_pred.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._timestamps_s:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
            )
        timestamps = np.asarray(self._timestamps_s, dtype=np.float64)
        return (
            timestamps - timestamps[-1],
            np.stack(self._tau_ext_cal, axis=0),
            np.stack(self._tau_ext_pred, axis=0),
        )

    def clear(self) -> None:
        self._timestamps_s.clear()
        self._tau_ext_cal.clear()
        self._tau_ext_pred.clear()


class RealtimeJointPlotter:
    _REQUIRED_DATASETS = (
        "tau_ext_cal",
        "tau_ext_pred",
    )

    def __init__(
        self,
        config: RealtimePlotConfig,
        robot_states: dict[str, StateParamConfig],
        dynamics_processing: DynamicsProcessingConfig | None = None,
    ) -> None:
        self.config = config
        self.robot_states = robot_states
        self.dynamics_processing = dynamics_processing or DynamicsProcessingConfig()
        self._queue = None
        self._process = None
        self._closed = False
        self._process_failure_logged = False

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._validate_required_states()
        if find_spec("matplotlib") is None:
            raise RuntimeError(
                "realtime_plot.enabled=true requires matplotlib; install matplotlib>=3.7"
            )
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=512)
        self._process = context.Process(
            target=_plot_worker,
            args=(self.config, self._queue),
            name="nero-realtime-plot",
            daemon=True,
        )
        self._process.start()
        log.info(
            "realtime plot process started datasets=tau_ext_cal,tau_ext_pred "
            "window=%.1fs update=%.1fHz",
            self.config.window_s,
            self.config.update_rate_hz,
        )

    def append(self, timestamp_us: int, values: dict[str, tuple[str, np.ndarray]]) -> None:
        if not self.config.enabled or self._closed:
            return
        if self._process is None or self._queue is None:
            raise RuntimeError("Realtime plot has not been started")
        if not self._process.is_alive():
            self._closed = True
            if not self._process_failure_logged:
                log.info("realtime plot window closed or unavailable; collection continues")
                self._process_failure_logged = True
            return

        missing = [name for name in self._REQUIRED_DATASETS if name not in values]
        if missing:
            raise RuntimeError(f"Realtime plot is missing teleop datasets: {missing}")
        sample = _RealtimeSample(
            timestamp_us=int(timestamp_us),
            tau_ext_cal=_plot_vector("tau_ext_cal", values["tau_ext_cal"][1], 7),
            tau_ext_pred=_plot_vector("tau_ext_pred", values["tau_ext_pred"][1], 7),
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass

    def clear_history(self) -> None:
        if not self.config.enabled or self._closed:
            return
        if self._process is None or self._queue is None:
            raise RuntimeError("Realtime plot has not been started")
        if not self._process.is_alive():
            self._closed = True
            log.info("realtime plot window closed or unavailable; collection continues")
            return
        command = _ClearHistory()
        try:
            self._queue.put(command, timeout=0.5)
        except queue.Full:
            log.warning("realtime plot queue stayed full; history was not cleared")
            return
        log.info("realtime plot history clear requested for new recording")

    def close(self) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
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

    def _validate_required_states(self) -> None:
        required = ("torque",)
        disabled = [
            name
            for name in required
            if name not in self.robot_states or not self.robot_states[name].enabled
        ]
        if disabled:
            raise RuntimeError(
                "realtime tau_ext plot requires enabled robot_states torque; "
                f"missing={disabled}"
            )


class _MatplotlibPlotWindow:
    _PLOTS = (
        ("tau_ext_cal", "external torque [N.m]", tuple(f"J{index}" for index in range(1, 8))),
        ("tau_ext_cal_l1", "||tau_ext_cal||_1 [N.m]", ("cal",)),
        ("tau_ext_pred", "external torque [N.m]", tuple(f"J{index}" for index in range(1, 8))),
        ("tau_ext_pred_l1", "||tau_ext_pred||_1 [N.m]", ("pred",)),
    )

    def __init__(self, config: RealtimePlotConfig) -> None:
        import matplotlib.pyplot as plt

        self.config = config
        self.buffer = SlidingJointBuffer(config.window_s)
        self.plt = plt
        plt.ion()
        self.figure, axes_grid = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        axes = tuple(np.asarray(axes_grid).reshape(-1))
        try:
            self.figure.canvas.manager.set_window_title("Nero realtime tau_ext")
        except AttributeError:
            pass
        colors = plt.get_cmap("tab10").colors
        line_groups: list[tuple[object, ...]] = []
        for axis, (title, ylabel, labels) in zip(axes, self._PLOTS):
            lines = tuple(
                axis.plot(
                    [], [], color=colors[index % len(colors)],
                    linewidth=1.2, label=label,
                )[0]
                for index, label in enumerate(labels)
            )
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("time [s], live")
            axis.set_xlim(-config.window_s, 0.0)
            axis.set_ylim(-_MINIMUM_Y_ABS_NM, _MINIMUM_Y_ABS_NM)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper left", ncol=2, fontsize=8)
            line_groups.append(lines)
        self.axes = tuple(axes)
        self.lines = tuple(line_groups)
        self.figure.tight_layout()
        self.figure.show()
        self.process_events()

    def append(
        self,
        sample: tuple[int, np.ndarray, np.ndarray],
    ) -> None:
        self.buffer.append(*sample)

    def render(self) -> None:
        relative_time, tau_ext_cal, tau_ext_pred = self.buffer.arrays()
        if relative_time.size:
            for index in range(7):
                self.lines[0][index].set_data(relative_time, tau_ext_cal[:, index])
                self.lines[2][index].set_data(relative_time, tau_ext_pred[:, index])
            tau_ext_cal_l1 = np.sum(np.abs(tau_ext_cal), axis=1)
            tau_ext_pred_l1 = np.sum(np.abs(tau_ext_pred), axis=1)
            self.lines[1][0].set_data(relative_time, tau_ext_cal_l1)
            self.lines[3][0].set_data(relative_time, tau_ext_pred_l1)
            _set_dynamic_ylim(self.axes[0], tau_ext_cal)
            _set_dynamic_ylim(self.axes[1], tau_ext_cal_l1[:, None])
            _set_dynamic_ylim(self.axes[2], tau_ext_pred)
            _set_dynamic_ylim(self.axes[3], tau_ext_pred_l1[:, None])
            for axis in self.axes:
                axis.set_xlim(-self.config.window_s, 0.0)
            self.figure.canvas.draw()
        self.process_events()

    def clear(self) -> None:
        self.buffer.clear()
        for axis, lines in zip(self.axes, self.lines):
            for line in lines:
                line.set_data([], [])
            axis.set_xlim(-self.config.window_s, 0.0)
            axis.set_ylim(-_MINIMUM_Y_ABS_NM, _MINIMUM_Y_ABS_NM)
        self.figure.canvas.draw()
        self.process_events()

    def process_events(self) -> None:
        self.figure.canvas.flush_events()

    def is_open(self) -> bool:
        return bool(self.plt.fignum_exists(self.figure.number))

    def close(self) -> None:
        self.plt.close(self.figure)


def _plot_worker(
    config: RealtimePlotConfig,
    sample_queue,
) -> None:
    try:
        window = _MatplotlibPlotWindow(config)
    except Exception:
        log.exception("failed to start realtime tau_ext plot")
        return
    next_render_t = time.monotonic()
    next_diagnostic_t = time.monotonic()
    stop = False
    try:
        while not stop and window.is_open():
            timeout_s = max(0.0, min(0.05, next_render_t - time.monotonic()))
            try:
                item = sample_queue.get(timeout=timeout_s)
                received_item = True
            except queue.Empty:
                item = None
                received_item = False
            if received_item:
                items = [item]
                while True:
                    try:
                        queued_item = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    items.append(queued_item)

                for queued_item in items:
                    if queued_item is None:
                        stop = True
                        break
                    if isinstance(queued_item, _ClearHistory):
                        window.clear()
                        log.debug("realtime plot history cleared")
                        continue
                    sample = queued_item
                    window.append(
                        (
                            sample.timestamp_us,
                            sample.tau_ext_cal,
                            sample.tau_ext_pred,
                        )
                    )
                    now = time.monotonic()
                    if now >= next_diagnostic_t:
                        log.debug(
                            "tau_ext cal/pred max_abs=%.4f/%.4fNm",
                            float(np.max(np.abs(sample.tau_ext_cal))),
                            float(np.max(np.abs(sample.tau_ext_pred))),
                        )
                        next_diagnostic_t = now + 2.0

            now = time.monotonic()
            if now >= next_render_t:
                window.render()
                next_render_t = now + 1.0 / config.update_rate_hz
            else:
                window.process_events()
    except Exception:
        log.exception("realtime tau_ext plot failed")
    finally:
        window.close()


def _plot_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"Realtime plot requires a finite {size}D {name} vector; got {vector}")
    return vector.copy()


def _set_dynamic_ylim(axis, data: np.ndarray) -> None:
    peak = float(np.max(np.abs(data)))
    limit = max(_MINIMUM_Y_ABS_NM, peak * 1.08)
    axis.set_ylim(-limit, limit)
