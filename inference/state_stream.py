from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from nero_collection.tau_ext_inference import (
    OnlineTauExtInference,
    OnlineTauExtResult,
)
from nero_collection.time_utils import now_us


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContinuousInferenceSample:
    """One fixed-rate state/tau_ext sample read from the latest SDK cache."""

    timestamp_us: int
    acquired_timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    tau_result: OnlineTauExtResult
    raw_wrench: np.ndarray
    wrench: np.ndarray
    processed_wrench: np.ndarray


class ContinuousInferenceStateStream:
    """Read one latest SDK state per fixed-rate inference tick."""

    def __init__(
        self,
        arm: Any,
        online_tau_ext: OnlineTauExtInference,
        wrench_estimator: Any,
        *,
        on_sample: Callable[[ContinuousInferenceSample], None] | None = None,
        wrench_processor: Callable[
            [np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]
        ] | None = None,
        q_cmd_provider: Callable[[int], np.ndarray | None] | None = None,
        history_size: int = 4096,
        poll_interval_s: float = 0.001,
    ) -> None:
        if history_size < 1:
            raise ValueError("state stream history_size must be positive")
        if not np.isfinite(poll_interval_s) or poll_interval_s <= 0.0:
            raise ValueError("state stream poll_interval_s must be positive and finite")
        self.arm = arm
        self.online_tau_ext = online_tau_ext
        self.wrench_estimator = wrench_estimator
        self.on_sample = on_sample
        self.wrench_processor = wrench_processor
        self.q_cmd_provider = q_cmd_provider
        self.history_size = int(history_size)
        self.poll_interval_s = float(poll_interval_s)
        self._samples: deque[ContinuousInferenceSample] = deque(maxlen=self.history_size)
        self._lock = threading.Lock()
        self._updated = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_timestamp_us = 0
        self._latest: ContinuousInferenceSample | None = None
        self._fault: BaseException | None = None
        self._history_rollover_count = 0
        self._last_batch_size = 0
        self._last_batch_processing_s = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        with self._lock:
            self._fault = None
        self._thread = threading.Thread(
            target=self._run,
            name="nero-inference-state-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._latest = None
            self._last_timestamp_us = 0
            self._fault = None
            self._history_rollover_count = 0
            self._last_batch_size = 0
            self._last_batch_processing_s = 0.0

    @property
    def history_rollover_count(self) -> int:
        """Number of samples evicted because the bounded history was full."""
        with self._lock:
            return self._history_rollover_count

    def latest(self) -> ContinuousInferenceSample | None:
        with self._lock:
            return self._latest

    def wait_for_acquired_after(
        self,
        minimum_acquired_timestamp_us: int,
        timeout_s: float,
    ) -> ContinuousInferenceSample | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        minimum_us = int(minimum_acquired_timestamp_us)
        with self._updated:
            while True:
                sample = self._latest
                if sample is not None and sample.acquired_timestamp_us >= minimum_us:
                    return sample
                if self._fault is not None:
                    return sample
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    return sample
                self._updated.wait(remaining_s)

    def processing_status(self) -> tuple[int, float]:
        with self._lock:
            return self._last_batch_size, self._last_batch_processing_s

    def drain_after(self, timestamp_us: int) -> tuple[ContinuousInferenceSample, ...]:
        with self._lock:
            return tuple(
                sample
                for sample in self._samples
                if sample.timestamp_us > int(timestamp_us)
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self.arm.read_state()
                started_s = time.perf_counter()
                sample = self.process_state(state)
                elapsed_s = time.perf_counter() - started_s
                with self._updated:
                    self._last_batch_size = 1
                    self._last_batch_processing_s = elapsed_s
                    if sample is not None:
                        if len(self._samples) == self._samples.maxlen:
                            self._history_rollover_count += 1
                        self._samples.append(sample)
                        self._latest = sample
                    self._updated.notify_all()
                if sample is not None and self.on_sample is not None:
                    self.on_sample(sample)
            except BaseException as exc:  # keep the main loop fail-closed
                with self._updated:
                    self._fault = exc
                    self._updated.notify_all()
                log.error("continuous inference state stream stopped: %s", exc)
                self._stop_event.set()
                return
            self._stop_event.wait(self.poll_interval_s)

    def process_state(self, state: Any) -> ContinuousInferenceSample | None:
        """Convert the current tick's SDK snapshot into one inference record."""
        vectors = (state.q, state.dq, state.torque)
        if not all(
            np.asarray(value, dtype=np.float64).shape == (7,)
            and np.all(np.isfinite(value))
            for value in vectors
        ):
            return None
        timestamp_us = max(
            now_us(),
            self._last_timestamp_us + int(round(self.poll_interval_s * 1.0e6)),
        )
        acquired_timestamp_us = now_us()
        q_cmd = (
            None
            if self.q_cmd_provider is None
            else self.q_cmd_provider(timestamp_us)
        )
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)
        if q_cmd.shape != (7,) or not np.all(np.isfinite(q_cmd)):
            raise RuntimeError(
                "tau_ext inference requires a finite seven-joint q_cmd"
            )
        tau_result = self.online_tau_ext.estimate_aligned(
            timestamp_us,
            state.q,
            state.dq,
            state.torque,
            q_cmd,
        )
        self._last_timestamp_us = timestamp_us
        return self._build_sample(
            timestamp_us,
            acquired_timestamp_us,
            tau_result,
            tau_result.ddq_kf_causal,
        )

    def _build_sample(
        self,
        timestamp_us: int,
        acquired_timestamp_us: int,
        tau_result: OnlineTauExtResult,
        ddq: np.ndarray,
    ) -> ContinuousInferenceSample:
        wrench_estimate = self.wrench_estimator.map_joint_torque(
            tau_result.q,
            tau_result.tau_ext_cal,
        )
        wrench_from_filtered_tau = np.asarray(
            wrench_estimate.wrench,
            dtype=np.float64,
        ).reshape(-1)
        raw_tau_ext = tau_result.tau_ext_cal_raw
        if raw_tau_ext is None:
            raw_wrench = wrench_from_filtered_tau.copy()
        else:
            raw_estimate = self.wrench_estimator.map_joint_torque(
                tau_result.q,
                raw_tau_ext,
            )
            raw_wrench = np.asarray(
                raw_estimate.wrench,
                dtype=np.float64,
            ).reshape(-1)
        if (
            raw_wrench.shape != (6,)
            or wrench_from_filtered_tau.shape != (6,)
            or not np.all(np.isfinite(raw_wrench))
            or not np.all(np.isfinite(wrench_from_filtered_tau))
        ):
            raise RuntimeError(
                "mapped raw/filtered inference wrenches must be finite 6-vectors"
            )
        if self.wrench_processor is None:
            wrench = wrench_from_filtered_tau.copy()
            processed_wrench = wrench_from_filtered_tau.copy()
        else:
            wrench, processed_wrench = self.wrench_processor(
                raw_wrench.copy(),
                wrench_from_filtered_tau.copy(),
                timestamp_us,
            )
            wrench = np.asarray(wrench, dtype=np.float64).reshape(-1)
            processed_wrench = np.asarray(processed_wrench, dtype=np.float64).reshape(-1)
            if (
                wrench.shape != (6,)
                or processed_wrench.shape != (6,)
                or not np.all(np.isfinite(wrench))
                or not np.all(np.isfinite(processed_wrench))
            ):
                raise RuntimeError("wrench processor must return finite 6-vectors")
        return ContinuousInferenceSample(
            timestamp_us=timestamp_us,
            acquired_timestamp_us=acquired_timestamp_us,
            q=np.asarray(tau_result.q, dtype=np.float64).copy(),
            dq=np.asarray(tau_result.dq, dtype=np.float64).copy(),
            ddq=np.asarray(ddq, dtype=np.float64).copy(),
            tau=np.asarray(tau_result.tau, dtype=np.float64).copy(),
            tau_result=tau_result,
            raw_wrench=raw_wrench.copy(),
            wrench=wrench.copy(),
            processed_wrench=processed_wrench.copy(),
        )
