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
from nero_collection.contact_wrench import PinocchioContactWrenchEstimator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RealtimeSample:
    timestamp_us: int
    q: np.ndarray
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
        self._wrench_cal: deque[np.ndarray] = deque()
        self._wrench_pred: deque[np.ndarray] = deque()

    def append(
        self,
        timestamp_us: int,
        tau_ext_cal: np.ndarray,
        tau_ext_pred: np.ndarray,
        wrench_cal: np.ndarray,
        wrench_pred: np.ndarray,
    ) -> None:
        tau_ext_cal = _plot_vector("tau_ext_cal", tau_ext_cal, 7)
        tau_ext_pred = _plot_vector("tau_ext_pred", tau_ext_pred, 7)
        wrench_cal = _plot_vector("wrench_cal", wrench_cal, 6)
        wrench_pred = _plot_vector("wrench_pred", wrench_pred, 6)
        timestamp_s = int(timestamp_us) / 1_000_000.0
        if self._timestamps_s and timestamp_s <= self._timestamps_s[-1]:
            return
        self._timestamps_s.append(timestamp_s)
        self._tau_ext_cal.append(tau_ext_cal)
        self._tau_ext_pred.append(tau_ext_pred)
        self._wrench_cal.append(wrench_cal)
        self._wrench_pred.append(wrench_pred)

        cutoff_s = timestamp_s - self.window_s
        while self._timestamps_s and self._timestamps_s[0] < cutoff_s:
            self._timestamps_s.popleft()
            self._tau_ext_cal.popleft()
            self._tau_ext_pred.popleft()
            self._wrench_cal.popleft()
            self._wrench_pred.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._timestamps_s:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 6), dtype=np.float64),
                np.empty((0, 6), dtype=np.float64),
            )
        timestamps = np.asarray(self._timestamps_s, dtype=np.float64)
        return (
            timestamps - timestamps[-1],
            np.stack(self._tau_ext_cal, axis=0),
            np.stack(self._tau_ext_pred, axis=0),
            np.stack(self._wrench_cal, axis=0),
            np.stack(self._wrench_pred, axis=0),
        )

    def clear(self) -> None:
        self._timestamps_s.clear()
        self._tau_ext_cal.clear()
        self._tau_ext_pred.clear()
        self._wrench_cal.clear()
        self._wrench_pred.clear()


class RealtimeJointPlotter:
    _REQUIRED_DATASETS = (
        "q_follower",
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
            "realtime plot process started datasets=tau_ext_cal,tau_ext_pred,"
            "wrench_cal,wrench_pred "
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
            q=_plot_vector("q_follower", values["q_follower"][1], 7),
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
        required = ("q", "torque")
        disabled = [
            name
            for name in required
            if name not in self.robot_states or not self.robot_states[name].enabled
        ]
        if disabled:
            raise RuntimeError(
                "realtime external-wrench plot requires enabled robot_states "
                f"q and torque; missing={disabled}"
            )


class _MatplotlibPlotWindow:
    _PLOTS = (
        ("tau_ext_cal", "external torque [N.m]", tuple(f"J{index}" for index in range(1, 8))),
        (
            "wrench_cal",
            "force [N] / moment [N.m]",
            ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        ),
        ("tau_ext_pred", "external torque [N.m]", tuple(f"J{index}" for index in range(1, 8))),
        (
            "wrench_pred",
            "force [N] / moment [N.m]",
            ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        ),
    )

    def __init__(self, config: RealtimePlotConfig) -> None:
        import matplotlib.pyplot as plt

        self.config = config
        self.buffer = SlidingJointBuffer(config.window_s)
        self.plt = plt
        plt.ion()
        self.figure, axes_grid = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        axes = tuple(axes_grid.reshape(-1))
        try:
            self.figure.canvas.manager.set_window_title("Nero realtime external wrench")
        except AttributeError:
            pass
        colors = plt.get_cmap("tab10").colors
        line_groups: list[tuple[object, ...]] = []
        for axis, (title, ylabel, labels) in zip(axes, self._PLOTS):
            lines = tuple(
                axis.plot([], [], color=colors[index], linewidth=1.1, label=label)[0]
                for index, label in enumerate(labels)
            )
            if title.startswith("wrench_"):
                mapping = config.wrench_mapping
                title = f"{title} ({mapping.frame_name}/{mapping.reference_frame})"
            axis.set_title(title)
            axis.set_xlabel("time [s], live")
            axis.set_ylabel(ylabel)
            axis.set_xlim(-config.window_s, 0.0)
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
        sample: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.buffer.append(*sample)

    def render(self) -> None:
        relative_time, tau_ext_cal, tau_ext_pred, wrench_cal, wrench_pred = (
            self.buffer.arrays()
        )
        if relative_time.size:
            for axis, lines, data in zip(
                self.axes,
                self.lines,
                (tau_ext_cal, wrench_cal, tau_ext_pred, wrench_pred),
            ):
                for index, line in enumerate(lines):
                    line.set_data(relative_time, data[:, index])
                axis.set_xlim(-self.config.window_s, 0.0)
                _set_dynamic_ylim(axis, data)
            self.figure.canvas.draw()
        self.process_events()

    def clear(self) -> None:
        self.buffer.clear()
        for axis, lines in zip(self.axes, self.lines):
            for line in lines:
                line.set_data([], [])
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
    config: RealtimePlotConfig,
    sample_queue,
) -> None:
    try:
        wrench_estimator = PinocchioContactWrenchEstimator(config.wrench_mapping)
        window = _MatplotlibPlotWindow(config)
    except Exception:
        log.exception("failed to start realtime external-wrench plot")
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
                    wrench_cal = wrench_estimator.map_joint_torque(
                        sample.q,
                        sample.tau_ext_cal,
                    )
                    wrench_pred = wrench_estimator.map_joint_torque(
                        sample.q,
                        sample.tau_ext_pred,
                    )
                    window.append(
                        (
                            sample.timestamp_us,
                            sample.tau_ext_cal,
                            sample.tau_ext_pred,
                            wrench_cal.wrench,
                            wrench_pred.wrench,
                        )
                    )
                    now = time.monotonic()
                    if now >= next_diagnostic_t:
                        log.debug(
                            "tau_ext cal/pred max_abs=%.4f/%.4fNm "
                            "wrench_error=%.4f/%.4f",
                            float(np.max(np.abs(sample.tau_ext_cal))),
                            float(np.max(np.abs(sample.tau_ext_pred))),
                            wrench_cal.reconstruction_error,
                            wrench_pred.reconstruction_error,
                        )
                        next_diagnostic_t = now + 2.0

            now = time.monotonic()
            if now >= next_render_t:
                window.render()
                next_render_t = now + 1.0 / config.update_rate_hz
            else:
                window.process_events()
    except Exception:
        log.exception("realtime external-wrench estimation failed")
    finally:
        window.close()


def _plot_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"Realtime plot requires a finite {size}D {name} vector; got {vector}")
    return vector.copy()


def _set_dynamic_ylim(axis, data: np.ndarray) -> None:
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = data_max - data_min
    padding = max(span * 0.08, 1e-3)
    if span < 1e-9:
        padding = max(abs(data_min) * 0.08, 0.05)
    axis.set_ylim(data_min - padding, data_max + padding)
