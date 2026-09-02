"""Timestamped asynchronous fast/slow inference primitives.

The high-level DP, fast Contact WM and low-level controller run on separate
threads.  Every object exchanged between them is keyed by monotonic seconds;
there is no chunk-index state in this module.  The implementation is generic
enough for hardware and deterministic unit-test adapters, while the runtime
integration can supply model-specific callables.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


DOF = 7


def monotonic_time() -> float:
    """Return the only clock used by asynchronous inference."""

    return time.monotonic()


def _vector(name: str, value: Any, width: int = DOF) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {width}-vector")
    return result.copy()


def _trajectory(name: str, value: Any, width: int = DOF) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape [T,{width}], got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


@dataclass(frozen=True)
class StateSample:
    timestamp_s: float
    q: np.ndarray
    dq: np.ndarray
    delta_q: np.ndarray
    tau: np.ndarray
    q_cmd: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("state timestamp must be finite")
        object.__setattr__(self, "timestamp_s", timestamp)
        for name in ("q", "dq", "delta_q", "tau"):
            object.__setattr__(self, name, _vector(name, getattr(self, name)))
        if self.q_cmd is not None:
            object.__setattr__(self, "q_cmd", _vector("q_cmd", self.q_cmd))


@dataclass(frozen=True)
class StateHistorySnapshot:
    timestamps_s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    delta_q: np.ndarray
    tau: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64).reshape(-1)
        if timestamps.size < 1 or not np.all(np.isfinite(timestamps)):
            raise ValueError("state history timestamps must be finite and non-empty")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("state history timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps_s", timestamps.copy())
        for name in ("q", "dq", "delta_q", "tau"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (timestamps.size, DOF) or not np.all(np.isfinite(value)):
                raise ValueError(f"state history {name} must have shape [{timestamps.size},7]")
            object.__setattr__(self, name, value.copy())

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "timestamp":
            return self.timestamps_s.copy()
        if key == "timestamps":
            return self.timestamps_s.copy()
        if key in {"q", "dq", "delta_q", "tau"}:
            return getattr(self, key).copy()
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return ("q", "dq", "delta_q", "tau")

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "delta_q": self.delta_q.copy(),
            "tau": self.tau.copy(),
            "timestamps": self.timestamps_s.copy(),
        }


class StateHistoryBuffer:
    """Thread-safe fixed-rate state history with timestamp interpolation."""

    def __init__(
        self,
        *,
        rate_hz: float = 100.0,
        horizon_s: float = 0.5,
        maxlen: int | None = None,
    ) -> None:
        self.rate_hz = float(rate_hz)
        self.horizon_s = float(horizon_s)
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("state history rate_hz must be positive and finite")
        if not np.isfinite(self.horizon_s) or self.horizon_s <= 0.0:
            raise ValueError("state history horizon_s must be positive and finite")
        self.dt_s = 1.0 / self.rate_hz
        self.sample_count = max(1, int(round(self.horizon_s * self.rate_hz)))
        capacity = self.sample_count + 256 if maxlen is None else int(maxlen)
        if capacity < self.sample_count:
            raise ValueError("state history maxlen is shorter than the requested horizon")
        self._samples: deque[StateSample] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    @property
    def latest_timestamp_s(self) -> float | None:
        with self._lock:
            return None if not self._samples else self._samples[-1].timestamp_s

    def latest(self) -> StateSample | None:
        with self._lock:
            return None if not self._samples else self._samples[-1]

    def append(
        self,
        timestamp_s: float,
        q: Any,
        dq: Any,
        delta_q: Any | None = None,
        tau: Any | None = None,
        *,
        q_cmd: Any | None = None,
    ) -> None:
        q_value = _vector("q", q)
        dq_value = _vector("dq", dq)
        tau_value = np.zeros(DOF, dtype=np.float64) if tau is None else _vector("tau", tau)
        if q_cmd is not None:
            command = _vector("q_cmd", q_cmd)
            delta_value = command - q_value
        elif delta_q is not None:
            delta_value = _vector("delta_q", delta_q)
            command = q_value + delta_value
        else:
            command = None
            delta_value = np.zeros(DOF, dtype=np.float64)
        sample = StateSample(timestamp_s, q_value, dq_value, delta_value, tau_value, command)
        with self._lock:
            if self._samples and sample.timestamp_s < self._samples[-1].timestamp_s:
                raise ValueError("state history timestamps must be non-decreasing")
            if self._samples and sample.timestamp_s == self._samples[-1].timestamp_s:
                self._samples[-1] = sample
            else:
                self._samples.append(sample)

    def query(
        self,
        t0: float,
        t1: float | None = None,
        *,
        horizon_s: float | None = None,
    ) -> StateHistorySnapshot | None:
        """Return exactly the model history grid ending at ``t0``.

        For the 100 Hz/0.5 s Contact WM contract this returns 50 rows at
        ``t0-0.49`` through ``t0``.  State channels are linearly interpolated;
        a recorded command is held causally and ``delta_q`` is recomputed as
        ``q_cmd-q`` to match training-time command semantics.

        ``query(start, end)`` is accepted as a timestamp-range spelling for
        integrations that keep the interval explicit; the returned grid still
        uses the configured fixed sample count and ends at ``end``.
        """

        end = float(t0 if t1 is None else t1)
        duration = (
            self.horizon_s
            if horizon_s is None and t1 is None
            else (float(t1) - float(t0) if horizon_s is None else float(horizon_s))
        )
        if not np.isfinite(end) or not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("state query time and horizon must be finite and positive")
        count = max(1, int(round(duration * self.rate_hz)))
        query_times = end - np.arange(count - 1, -1, -1, dtype=np.float64) * self.dt_s
        with self._lock:
            samples = tuple(self._samples)
        if len(samples) < 1:
            return None
        source_times = np.asarray([sample.timestamp_s for sample in samples], dtype=np.float64)
        if source_times[0] > query_times[0] + 1.0e-9 or source_times[-1] < end - 1.0e-9:
            return None
        values = {}
        for key in ("q", "dq", "tau"):
            source = np.stack([getattr(sample, key) for sample in samples], axis=0)
            values[key] = np.column_stack(
                [np.interp(query_times, source_times, source[:, axis]) for axis in range(DOF)]
            )
        commands = [sample.q_cmd for sample in samples]
        if all(command is not None for command in commands):
            command_array = np.stack([command for command in commands], axis=0)
            indices = np.searchsorted(source_times, query_times + 1.0e-9, side="right") - 1
            indices = np.clip(indices, 0, len(samples) - 1)
            command_values = command_array[indices]
            delta_values = command_values - values["q"]
        else:
            source_delta = np.stack([sample.delta_q for sample in samples], axis=0)
            delta_values = np.column_stack(
                [np.interp(query_times, source_times, source_delta[:, axis]) for axis in range(DOF)]
            )
        return StateHistorySnapshot(query_times, values["q"], values["dq"], delta_values, values["tau"])


@dataclass(frozen=True)
class ActionPlan:
    start_time_s: float
    step_s: float
    values: np.ndarray
    generation: int

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.step_s * self.values.shape[0]


@dataclass(frozen=True)
class ActionTrajectory:
    timestamps_s: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64).reshape(-1)
        values = _trajectory("action trajectory", self.values)
        if values.shape[0] != timestamps.size:
            raise ValueError("action trajectory timestamps and values must have equal length")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("action trajectory timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps_s", timestamps.copy())
        object.__setattr__(self, "values", values)


class ActionPlanBuffer:
    """Timestamped DP plan storage with causal ZOH resampling."""

    def __init__(self, *, max_plans: int = 16, default_step_s: float = 0.04) -> None:
        self.default_step_s = float(default_step_s)
        if not np.isfinite(self.default_step_s) or self.default_step_s <= 0.0:
            raise ValueError("action plan default_step_s must be positive and finite")
        self._plans: deque[ActionPlan] = deque(maxlen=int(max_plans))
        self._lock = threading.Lock()
        self._generation = 0

    def clear(self) -> None:
        with self._lock:
            self._plans.clear()
            self._generation = 0

    def append(self, values: Any, *, start_time_s: float, step_s: float | None = None) -> ActionPlan:
        actions = _trajectory("action plan", values)
        start = float(start_time_s)
        step = self.default_step_s if step_s is None else float(step_s)
        if not np.isfinite(start) or not np.isfinite(step) or step <= 0.0:
            raise ValueError("action plan start_time_s/step_s must be finite and positive")
        with self._lock:
            self._generation += 1
            plan = ActionPlan(start, step, actions, self._generation)
            self._plans.append(plan)
        return plan

    def snapshot(self) -> tuple[ActionPlan, ...]:
        with self._lock:
            return tuple(self._plans)

    def query_with_timestamps(
        self,
        start_time_s: float,
        end_time_s: float,
        *,
        rate_hz: float = 100.0,
    ) -> ActionTrajectory | None:
        start = float(start_time_s)
        end = float(end_time_s)
        rate = float(rate_hz)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("action query requires finite start < end")
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("action query rate_hz must be positive and finite")
        count = max(1, int(round((end - start) * rate)))
        times = start + np.arange(count, dtype=np.float64) / rate
        with self._lock:
            plans = tuple(self._plans)
        if not plans:
            return None
        result = np.empty((count, DOF), dtype=np.float64)
        for index, timestamp in enumerate(times):
            candidates = [
                plan
                for plan in plans
                if plan.start_time_s - 1.0e-9 <= timestamp < plan.end_time_s + 1.0e-9
            ]
            if not candidates:
                return None
            plan = max(candidates, key=lambda value: value.generation)
            token = int(math.floor((timestamp - plan.start_time_s + 1.0e-9) / plan.step_s))
            token = int(np.clip(token, 0, plan.values.shape[0] - 1))
            result[index] = plan.values[token]
        return ActionTrajectory(times, result)

    def query(self, start_time_s: float, end_time_s: float, *, rate_hz: float = 100.0) -> np.ndarray | None:
        trajectory = self.query_with_timestamps(start_time_s, end_time_s, rate_hz=rate_hz)
        return None if trajectory is None else trajectory.values


@dataclass(frozen=True)
class WMTarget:
    timestamp_s: float
    q_ref: np.ndarray
    tau_ref: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "q_ref", _vector("q_ref", self.q_ref))
        object.__setattr__(self, "tau_ref", _vector("tau_ref", self.tau_ref))


@dataclass(frozen=True)
class _TargetSegment:
    generation: int
    timestamps_s: np.ndarray
    q_ref: np.ndarray
    tau_ref: np.ndarray

    @property
    def start(self) -> float:
        return float(self.timestamps_s[0])

    @property
    def end(self) -> float:
        return float(self.timestamps_s[-1])


class WMTargetBuffer:
    """Timestamped WM targets with short overlap blending on replacement."""

    def __init__(self, *, blend_duration_s: float = 0.04, max_segments: int = 16) -> None:
        self.blend_duration_s = float(blend_duration_s)
        if not np.isfinite(self.blend_duration_s) or self.blend_duration_s < 0.0:
            raise ValueError("WM blend_duration_s must be finite and non-negative")
        self._segments: deque[_TargetSegment] = deque(maxlen=int(max_segments))
        self._lock = threading.Lock()
        self._generation = 0

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._generation = 0

    def append(self, timestamps_s: Any, q_ref: Any, tau_ref: Any) -> int:
        timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
        q_values = _trajectory("WM q_ref", q_ref)
        tau_values = _trajectory("WM tau_ref", tau_ref)
        if timestamps.size != q_values.shape[0] or timestamps.size != tau_values.shape[0]:
            raise ValueError("WM target timestamps and trajectories must have equal length")
        if timestamps.size < 1 or not np.all(np.isfinite(timestamps)):
            raise ValueError("WM target timestamps must be finite and non-empty")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("WM target timestamps must be strictly increasing")
        with self._lock:
            self._generation += 1
            self._segments.append(
                _TargetSegment(self._generation, timestamps.copy(), q_values, tau_values)
            )
            return self._generation

    @staticmethod
    def _value(segment: _TargetSegment, timestamp_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        if timestamp_s < segment.start - 1.0e-9 or timestamp_s > segment.end + 1.0e-9:
            return None
        q = np.array(
            [np.interp(timestamp_s, segment.timestamps_s, segment.q_ref[:, axis]) for axis in range(DOF)],
            dtype=np.float64,
        )
        tau = np.array(
            [np.interp(timestamp_s, segment.timestamps_s, segment.tau_ref[:, axis]) for axis in range(DOF)],
            dtype=np.float64,
        )
        return q, tau

    def query(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("WM target query timestamp must be finite")
        with self._lock:
            segments = tuple(self._segments)
        candidates = [segment for segment in segments if self._value(segment, timestamp) is not None]
        if not candidates:
            return None
        newest = max(candidates, key=lambda segment: segment.generation)
        current = self._value(newest, timestamp)
        assert current is not None
        if self.blend_duration_s <= 0.0 or timestamp > newest.start + self.blend_duration_s:
            return current[0].copy(), current[1].copy()
        older = [segment for segment in candidates if segment.generation < newest.generation]
        if not older:
            return current[0].copy(), current[1].copy()
        previous = self._value(max(older, key=lambda segment: segment.generation), timestamp)
        if previous is None:
            return current[0].copy(), current[1].copy()
        weight = float(np.clip((timestamp - newest.start) / self.blend_duration_s, 0.0, 1.0))
        return (
            (1.0 - weight) * previous[0] + weight * current[0],
            (1.0 - weight) * previous[1] + weight * current[1],
        )

    def query_target(self, timestamp_s: float) -> WMTarget | None:
        value = self.query(timestamp_s)
        if value is None:
            return None
        return WMTarget(float(timestamp_s), value[0], value[1])


class _LatestWorker:
    def __init__(self, *, name: str) -> None:
        self._name = name
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fault(self) -> BaseException | None:
        with self._condition:
            return self._fault

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        with self._condition:
            self._fault = None
        self._thread = threading.Thread(target=self._run_guarded, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            with self._condition:
                self._fault = exc
                self._condition.notify_all()

    def _run(self) -> None:  # pragma: no cover - abstract worker hook
        raise NotImplementedError


class DPWorker(_LatestWorker):
    """Run slow DP on the newest camera snapshot and publish timestamped plans."""

    def __init__(
        self,
        infer_fn: Callable[[Any], Any],
        action_buffer: ActionPlanBuffer,
        *,
        step_s: float = 0.04,
        source: Callable[[], Any | None] | None = None,
        result_start_time_fn: Callable[[float, float, Any], float] | None = None,
    ) -> None:
        super().__init__(name="nero-dp-worker")
        self.infer_fn = infer_fn
        self.action_buffer = action_buffer
        self.step_s = float(step_s)
        self.source = source
        self.result_start_time_fn = result_start_time_fn
        self._pending: Any | None = None
        self._pending_time_s: float | None = None
        self._pending_lock = threading.Lock()
        self._updates = 0

    @property
    def updates(self) -> int:
        return self._updates

    def submit(self, observation: Any, *, timestamp_s: float | None = None) -> None:
        timestamp = monotonic_time() if timestamp_s is None else float(timestamp_s)
        with self._pending_lock:
            self._pending = observation
            self._pending_time_s = timestamp
        with self._condition:
            self._condition.notify_all()

    @staticmethod
    def _extract_action(result: Any) -> np.ndarray:
        if hasattr(result, "values"):
            result = result.values
        if isinstance(result, Mapping):
            for key in ("action", "action_chunk", "values", "action_pred", "trajectory"):
                if key in result:
                    result = result[key]
                    break
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        array = np.asarray(result, dtype=np.float64)
        if array.ndim == 3:
            array = array[0]
        return _trajectory("DP action output", array)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self.source is not None:
                observation = self.source()
                if observation is not None:
                    # Observation.timestamp_us is a legacy wall-clock field.
                    # Timestamped plans use the monotonic capture boundary so
                    # they share one clock with StateHistory/WMTargetBuffer.
                    self.submit(observation, timestamp_s=monotonic_time())
            with self._pending_lock:
                observation = self._pending
                _input_time = self._pending_time_s
                self._pending = None
                self._pending_time_s = None
            if observation is None:
                with self._condition:
                    self._condition.wait(timeout=0.002)
                continue
            started = monotonic_time()
            result = self.infer_fn(observation)
            returned = monotonic_time()
            values = self._extract_action(result)
            start_time = returned if self.result_start_time_fn is None else float(
                self.result_start_time_fn(started, returned, observation)
            )
            if not np.isfinite(start_time) or start_time < 0.0:
                raise ValueError("DP action_start_time must be finite and non-negative")
            self.action_buffer.append(values, start_time_s=start_time, step_s=self.step_s)
            self._updates += 1


class WMWorker(_LatestWorker):
    """Periodically snapshot state/action buffers and publish future targets."""

    def __init__(
        self,
        state_buffer: StateHistoryBuffer,
        action_buffer: ActionPlanBuffer,
        target_buffer: WMTargetBuffer,
        infer_fn: Callable[[StateHistorySnapshot, ActionTrajectory], Any],
        *,
        prediction_horizon: int = 32,
        prediction_dt_s: float = 0.01,
        request_period_s: float = 1.0 / 16.0,
        state_horizon_s: float = 0.5,
        action_horizon_s: float = 0.32,
        auto_request: bool = True,
    ) -> None:
        super().__init__(name="nero-wm-worker")
        self.state_buffer = state_buffer
        self.action_buffer = action_buffer
        self.target_buffer = target_buffer
        self.infer_fn = infer_fn
        self.prediction_horizon = int(prediction_horizon)
        self.prediction_dt_s = float(prediction_dt_s)
        self.request_period_s = float(request_period_s)
        self.state_horizon_s = float(state_horizon_s)
        self.action_horizon_s = float(action_horizon_s)
        self.auto_request = bool(auto_request)
        if self.prediction_horizon < 1 or not np.isfinite(self.prediction_dt_s) or self.prediction_dt_s <= 0.0:
            raise ValueError("WM prediction horizon/dt must be positive")
        if not np.isfinite(self.request_period_s) or self.request_period_s <= 0.0:
            raise ValueError("WM request_period_s must be positive and finite")
        self._requested: deque[float] = deque(maxlen=1)
        self._request_lock = threading.Lock()
        self._inference_count = 0
        self._last_latency_s: float | None = None

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def last_latency_s(self) -> float | None:
        return self._last_latency_s

    def request(self, t_start: float | None = None) -> bool:
        timestamp = monotonic_time() if t_start is None else float(t_start)
        if not np.isfinite(timestamp):
            raise ValueError("WM request timestamp must be finite")
        with self._request_lock:
            if self._requested:
                return False
            self._requested.append(timestamp)
        with self._condition:
            self._condition.notify_all()
        return True

    @staticmethod
    def _extract_prediction(result: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(result, Mapping):
            q = next((result[key] for key in ("q_ref", "q_pred", "q") if key in result), None)
            tau = next((result[key] for key in ("tau_ref", "tau_pred", "tau") if key in result), None)
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            q, tau = result
        else:
            raise ValueError("WM inference must return (q_ref, tau_ref) or a mapping")
        if q is None or tau is None:
            raise ValueError("WM inference output requires q_ref and tau_ref")
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        if hasattr(tau, "detach"):
            tau = tau.detach().cpu().numpy()
        q_values = np.asarray(q, dtype=np.float64)
        tau_values = np.asarray(tau, dtype=np.float64)
        if q_values.ndim == 3:
            q_values = q_values[0]
        if tau_values.ndim == 3:
            tau_values = tau_values[0]
        return _trajectory("WM q_ref", q_values), _trajectory("WM tau_ref", tau_values)

    def _run(self) -> None:
        next_request = monotonic_time()
        while not self._stop_event.is_set():
            now = monotonic_time()
            if self.auto_request and now >= next_request:
                self.request(now)
                next_request += self.request_period_s
                if next_request < now:
                    next_request = now + self.request_period_s
            with self._request_lock:
                t_start = self._requested.popleft() if self._requested else None
            if t_start is None:
                with self._condition:
                    timeout = max(0.0005, min(0.005, next_request - monotonic_time())) if self.auto_request else 0.005
                    self._condition.wait(timeout=timeout)
                continue
            latest_state_time = self.state_buffer.latest_timestamp_s
            if latest_state_time is None:
                continue
            # The state producer is independent of this scheduler and normally
            # trails ``now`` by a fraction of one control period.  Snapshot at
            # the newest available absolute state time instead of repeatedly
            # rejecting an otherwise valid 0.5 s history for that tiny gap.
            t_start = min(float(t_start), float(latest_state_time))
            history = self.state_buffer.query(t_start, horizon_s=self.state_horizon_s)
            action = self.action_buffer.query_with_timestamps(
                t_start,
                t_start + self.action_horizon_s,
                rate_hz=1.0 / self.prediction_dt_s,
            )
            if history is None or action is None:
                continue
            started = monotonic_time()
            result = self.infer_fn(history, action)
            returned = monotonic_time()
            q_values, tau_values = self._extract_prediction(result)
            if q_values.shape[0] != self.prediction_horizon or tau_values.shape[0] != self.prediction_horizon:
                raise ValueError(
                    "WM prediction horizon must be "
                    f"{self.prediction_horizon}, got q={q_values.shape[0]} tau={tau_values.shape[0]}"
                )
            first_valid_idx = int(math.ceil(max(0.0, returned - t_start) / self.prediction_dt_s))
            first_valid_idx = int(np.clip(first_valid_idx, 0, self.prediction_horizon))
            if first_valid_idx < self.prediction_horizon:
                timestamps = t_start + (
                    np.arange(first_valid_idx, self.prediction_horizon, dtype=np.float64) + 1.0
                ) * self.prediction_dt_s
                self.target_buffer.append(
                    timestamps,
                    q_values[first_valid_idx:],
                    tau_values[first_valid_idx:],
                )
            self._last_latency_s = returned - started
            self._inference_count += 1


class ControlWorker(_LatestWorker):
    """Consume timestamped WM targets at a strict fixed-rate cadence."""

    def __init__(
        self,
        state_buffer: StateHistoryBuffer,
        target_buffer: WMTargetBuffer,
        control_fn: Callable[[StateSample, WMTarget | None, float], Any],
        *,
        rate_hz: float = 100.0,
        action_buffer: ActionPlanBuffer | None = None,
    ) -> None:
        super().__init__(name="nero-control-worker")
        self.state_buffer = state_buffer
        self.target_buffer = target_buffer
        self.control_fn = control_fn
        # When the WM is disabled, the same 100 Hz loop can consume the
        # timestamped DP action plan directly.  Keeping this optional preserves
        # the normal WMTargetBuffer path without introducing a second control
        # thread or a chunk-index scheduler.
        self.action_buffer = action_buffer
        self.rate_hz = float(rate_hz)
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("control rate_hz must be positive and finite")
        self.period_s = 1.0 / self.rate_hz
        self._cycles = 0
        self._last_output: Any = None

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def last_output(self) -> Any:
        return self._last_output

    def _run(self) -> None:
        deadline = monotonic_time()
        while not self._stop_event.is_set():
            now = monotonic_time()
            state = self.state_buffer.latest()
            if state is not None:
                target = self.target_buffer.query_target(now)
                if target is None and self.action_buffer is not None:
                    action = self.action_buffer.query_with_timestamps(
                        now,
                        now + self.period_s,
                        rate_hz=self.rate_hz,
                    )
                    if action is not None:
                        target = WMTarget(
                            now,
                            action.values[0],
                            np.zeros(DOF, dtype=np.float64),
                        )
                self._last_output = self.control_fn(state, target, now)
                self._cycles += 1
            deadline += self.period_s
            sleep_s = deadline - monotonic_time()
            if sleep_s > 0.0:
                self._stop_event.wait(sleep_s)
            else:
                deadline = monotonic_time()


class TimestampFastSlowRuntime:
    """Small coordinator joining DP/control workers and an optional WM worker."""

    def __init__(
        self,
        *,
        state_buffer: StateHistoryBuffer,
        action_buffer: ActionPlanBuffer,
        target_buffer: WMTargetBuffer,
        dp_worker: DPWorker,
        wm_worker: WMWorker | None,
        control_worker: ControlWorker,
    ) -> None:
        self.state_buffer = state_buffer
        self.action_buffer = action_buffer
        self.target_buffer = target_buffer
        self.dp_worker = dp_worker
        self.wm_worker = wm_worker
        self.control_worker = control_worker

    def start(self) -> None:
        self.dp_worker.start()
        if self.wm_worker is not None:
            self.wm_worker.start()
        self.control_worker.start()

    def stop(self) -> None:
        self.control_worker.stop()
        if self.wm_worker is not None:
            self.wm_worker.stop()
        self.dp_worker.stop()

    def reset(self) -> None:
        self.stop()
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.target_buffer.clear()
        self.start()

    @property
    def cycles(self) -> int:
        return self.control_worker.cycles

    @property
    def last_output(self) -> Any:
        return self.control_worker.last_output


__all__ = [
    "ActionPlan",
    "ActionPlanBuffer",
    "ActionTrajectory",
    "ControlWorker",
    "DPWorker",
    "StateHistoryBuffer",
    "StateHistorySnapshot",
    "StateSample",
    "TimestampFastSlowRuntime",
    "WMTarget",
    "WMTargetBuffer",
    "WMWorker",
    "monotonic_time",
]
