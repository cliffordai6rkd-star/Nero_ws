from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np

from nero_collection.filters import OnePoleLowPass


class StateTimingError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlignedArmSample:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    torque: np.ndarray
    current: np.ndarray


@dataclass(frozen=True)
class _TimedVector:
    timestamp_us: int
    value: np.ndarray


@dataclass(frozen=True)
class _TimedMotorState:
    timestamp_us: int
    velocity: float
    ddq: float
    torque: float
    current: float


@dataclass(frozen=True)
class _TimedJointState:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


@dataclass
class _JointGroupDerivativeState:
    q_window: deque[np.ndarray]
    q_filter: OnePoleLowPass | None
    dq_filter: OnePoleLowPass | None
    ddq_filter: OnePoleLowPass | None
    # Legacy helper compatibility; production state alignment sets this false.
    use_q_mean_window: bool = True
    previous_timestamp_us: int | None = None
    previous_q_filtered: np.ndarray | None = None
    previous_q_velocity: np.ndarray | None = None
    previous_dq_filtered: np.ndarray | None = None


class DelayedStateTimeline:
    """Build filtered joint-group derivatives, then sample a delayed timeline."""

    def __init__(
        self,
        *,
        dof: int,
        q_groups: tuple[tuple[str, tuple[int, ...]], ...],
        delay_s: float,
        output_rate_hz: float,
        q_mean_window_samples: int = 5,
        q_lowpass_cutoff_hz: float | None = 10.0,
        dq_lowpass_cutoff_hz: float | None = 6.0,
        ddq_lowpass_cutoff_hz: float | None = 3.0,
        maximum_input_gap_s: float = 0.03,
        history_size: int = 512,
    ) -> None:
        self.dof = int(dof)
        self.q_groups = q_groups
        self.delay_us = int(round(float(delay_s) * 1_000_000.0))
        self.output_rate_hz = float(output_rate_hz)
        if self.dof <= 0:
            raise ValueError("state timeline dof must be positive")
        if self.delay_us <= 0:
            raise ValueError("state alignment delay must be positive")
        if not np.isfinite(self.output_rate_hz) or self.output_rate_hz <= 0:
            raise ValueError("state timeline output rate must be positive and finite")
        if not np.isfinite(maximum_input_gap_s) or maximum_input_gap_s <= 0:
            raise ValueError("maximum input gap must be positive and finite")
        self.maximum_input_gap_us = int(round(maximum_input_gap_s * 1_000_000.0))
        configured_q_mean_window = int(q_mean_window_samples)
        if configured_q_mean_window <= 0:
            raise ValueError("q-mean window must be a positive integer")
        # Keep accepting the old configuration field, but do not use a sliding
        # q mean in the production derivative path.
        self.q_mean_window_samples = 1
        covered = sorted(index for _, indices in q_groups for index in indices)
        if covered != list(range(self.dof)):
            raise ValueError(f"q groups must cover joints 0..{self.dof - 1}; got {covered}")
        self._q_derivative_state = {
            name: _JointGroupDerivativeState(
                q_window=deque(maxlen=1),
                q_filter=_make_lowpass(q_lowpass_cutoff_hz),
                dq_filter=None,
                ddq_filter=None,
                use_q_mean_window=False,
            )
            for name, _ in self.q_groups
        }
        self._q_state_history = {
            name: deque(maxlen=history_size) for name, _ in self.q_groups
        }
        self._motor_history: list[deque[_TimedMotorState]] = [
            deque(maxlen=history_size) for _ in range(self.dof)
        ]
        self._motor_dq_filters = [
            _make_lowpass(dq_lowpass_cutoff_hz) for _ in range(self.dof)
        ]
        self._motor_ddq_filters = [
            _make_lowpass(ddq_lowpass_cutoff_hz) for _ in range(self.dof)
        ]
        self._motor_previous_timestamp_us: list[int | None] = [None] * self.dof
        self._motor_previous_dq: list[float | None] = [None] * self.dof
        self._joint_group_for_index = {
            joint_index: (name, offset)
            for name, indices in self.q_groups
            for offset, joint_index in enumerate(indices)
        }
        output_period_us = int(np.ceil(1_000_000.0 / self.output_rate_hz))
        self._max_nearest_age_us = max(self.delay_us, 2 * output_period_us)
        self._lock = Lock()

    def append_q_group(self, name: str, timestamp_us: int, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        expected = dict(self.q_groups).get(name)
        if expected is None:
            raise KeyError(f"unknown q group {name!r}")
        if value.shape != (len(expected),) or not np.isfinite(value).all():
            return
        with self._lock:
            derivative_state = self._q_derivative_state[name]
            state = _append_filtered_joint_state(
                derivative_state,
                int(timestamp_us),
                value,
                maximum_gap_us=self.maximum_input_gap_us,
                stream_name=name,
            )
            if state is None:
                return
            _append_joint_state(self._q_state_history[name], state)

    def append_motor(
        self,
        joint_index: int,
        timestamp_us: int,
        *,
        velocity: float,
        torque: float,
        current: float,
    ) -> None:
        if not 0 <= joint_index < self.dof:
            raise IndexError(f"motor joint index out of range: {joint_index}")
        value = np.asarray((velocity, torque, current), dtype=np.float64)
        if not np.isfinite(value).all():
            return
        with self._lock:
            history = self._motor_history[joint_index]
            timestamp_us = int(timestamp_us)
            _validate_timestamp_gap(
                history[-1].timestamp_us if history else None,
                timestamp_us,
                self.maximum_input_gap_us,
                f"motor_state_{joint_index + 1}",
            )
            if history and timestamp_us == history[-1].timestamp_us:
                return

            velocity_filtered = float(
                _apply_lowpass(
                    self._motor_dq_filters[joint_index],
                    np.asarray([velocity], dtype=np.float64),
                    timestamp_us,
                )[0]
            )
            previous_timestamp_us = self._motor_previous_timestamp_us[joint_index]
            previous_dq = self._motor_previous_dq[joint_index]
            if previous_timestamp_us is None or previous_dq is None:
                ddq_from_dq = 0.0
            else:
                dt = (timestamp_us - previous_timestamp_us) * 1.0e-6
                ddq_from_dq = (
                    0.0 if dt <= 0.0 else (velocity_filtered - previous_dq) / dt
                )
            self._motor_previous_timestamp_us[joint_index] = timestamp_us
            self._motor_previous_dq[joint_index] = velocity_filtered

            ddq_from_q = self._q_second_derivative_at(joint_index, timestamp_us)
            ddq_fused = _fuse_acceleration(ddq_from_dq, ddq_from_q)
            ddq_filtered = float(
                _apply_lowpass(
                    self._motor_ddq_filters[joint_index],
                    np.asarray([ddq_fused], dtype=np.float64),
                    timestamp_us,
                )[0]
            )
            history.append(
                _TimedMotorState(
                    timestamp_us=timestamp_us,
                    velocity=velocity_filtered,
                    ddq=ddq_filtered,
                    torque=float(torque),
                    current=float(current),
                )
            )

    def _q_second_derivative_at(
        self,
        joint_index: int,
        timestamp_us: int,
    ) -> float:
        group_name, offset = self._joint_group_for_index[joint_index]
        state = _nearest(self._q_state_history[group_name], timestamp_us)
        if (
            state is None
            or abs(state.timestamp_us - timestamp_us) > self._max_nearest_age_us
        ):
            return float("nan")
        return float(state.ddq[offset])

    def aligned_sample(self, now_timestamp_us: int) -> AlignedArmSample | None:
        target_us = self._target_timestamp_us(int(now_timestamp_us))
        q = np.empty(self.dof, dtype=np.float64)
        dq = np.empty(self.dof, dtype=np.float64)
        ddq = np.empty(self.dof, dtype=np.float64)
        torque = np.empty(self.dof, dtype=np.float64)
        current = np.empty(self.dof, dtype=np.float64)
        with self._lock:
            for name, indices in self.q_groups:
                state = _nearest(self._q_state_history[name], target_us)
                if state is None or abs(state.timestamp_us - target_us) > self._max_nearest_age_us:
                    return None
                joint_indices = list(indices)
                q[joint_indices] = state.q
            for joint_index, history in enumerate(self._motor_history):
                motor = _nearest(history, target_us)
                if motor is None or abs(motor.timestamp_us - target_us) > self._max_nearest_age_us:
                    return None
                dq[joint_index] = motor.velocity
                ddq[joint_index] = motor.ddq
                torque[joint_index] = motor.torque
                current[joint_index] = motor.current
        return AlignedArmSample(
            timestamp_us=target_us,
            q=q,
            dq=dq,
            ddq=ddq,
            torque=torque,
            current=current,
        )

    def _target_timestamp_us(self, now_timestamp_us: int) -> int:
        delayed_us = now_timestamp_us - self.delay_us
        index = int(np.floor(delayed_us * self.output_rate_hz / 1_000_000.0))
        return int(round(index * 1_000_000.0 / self.output_rate_hz))


def _append_sample(
    history: deque[_TimedVector],
    timestamp_us: int,
    value: np.ndarray,
) -> bool:
    if timestamp_us <= 0:
        return False
    sample = _TimedVector(timestamp_us, value.copy())
    if history and timestamp_us < history[-1].timestamp_us:
        return False
    if history and timestamp_us == history[-1].timestamp_us:
        history[-1] = sample
        return False
    history.append(sample)
    return True


def _append_joint_state(
    history: deque[_TimedJointState],
    state: _TimedJointState,
) -> None:
    if history and state.timestamp_us <= history[-1].timestamp_us:
        if state.timestamp_us == history[-1].timestamp_us:
            history[-1] = state
        return
    history.append(state)


def _append_filtered_joint_state(
    state: _JointGroupDerivativeState,
    timestamp_us: int,
    q: np.ndarray,
    *,
    maximum_gap_us: int = 30_000,
    stream_name: str = "joint_state",
) -> _TimedJointState | None:
    if timestamp_us <= 0:
        return None
    if state.previous_timestamp_us is not None:
        _validate_timestamp_gap(
            state.previous_timestamp_us,
            timestamp_us,
            maximum_gap_us,
            stream_name,
        )
        if timestamp_us == state.previous_timestamp_us:
            return None

    q = np.asarray(q, dtype=np.float64).copy()
    if state.previous_timestamp_us is None:
        if state.use_q_mean_window:
            state.q_window.extend(q.copy() for _ in range(state.q_window.maxlen or 0))
        q_filtered = _apply_lowpass(state.q_filter, q, timestamp_us)
        state.previous_q_velocity = np.zeros_like(q)
        dq_filtered = _apply_lowpass(state.dq_filter, np.zeros_like(q), timestamp_us)
        ddq_filtered = _apply_lowpass(state.ddq_filter, np.zeros_like(q), timestamp_us)
        state.previous_timestamp_us = timestamp_us
        state.previous_q_filtered = q_filtered.copy()
        state.previous_dq_filtered = dq_filtered.copy()
        return _TimedJointState(
            timestamp_us=timestamp_us,
            q=q,
            dq=dq_filtered,
            ddq=ddq_filtered,
        )

    dt = (timestamp_us - state.previous_timestamp_us) * 1e-6
    if dt <= 0.0:
        return None
    if state.use_q_mean_window:
        state.q_window.append(q.copy())
        q_input = np.mean(np.stack(state.q_window, axis=0), axis=0)
    else:
        q_input = q
    q_filtered = _apply_lowpass(state.q_filter, q_input, timestamp_us)
    if not state.use_q_mean_window:
        q_velocity = (q_filtered - state.previous_q_filtered) / dt
        previous_q_velocity = (
            state.previous_q_velocity
            if state.previous_q_velocity is not None
            else np.zeros_like(q_velocity)
        )
        ddq = (q_velocity - previous_q_velocity) / dt
        state.previous_q_velocity = q_velocity.copy()
        state.previous_timestamp_us = timestamp_us
        state.previous_q_filtered = q_filtered.copy()
        return _TimedJointState(
            timestamp_us=timestamp_us,
            q=q,
            dq=q_velocity,
            ddq=ddq,
        )
    dq_raw = (q_filtered - state.previous_q_filtered) / dt
    dq_filtered = _apply_lowpass(state.dq_filter, dq_raw, timestamp_us)
    ddq_raw = (dq_filtered - state.previous_dq_filtered) / dt
    ddq_filtered = _apply_lowpass(state.ddq_filter, ddq_raw, timestamp_us)

    state.previous_timestamp_us = timestamp_us
    state.previous_q_filtered = q_filtered.copy()
    state.previous_dq_filtered = dq_filtered.copy()
    return _TimedJointState(
        timestamp_us=timestamp_us,
        q=q,
        dq=dq_filtered,
        ddq=ddq_filtered,
    )


def _validate_timestamp_gap(
    previous_timestamp_us: int | None,
    timestamp_us: int,
    maximum_gap_us: int,
    stream_name: str,
) -> None:
    if previous_timestamp_us is None:
        return
    gap_us = int(timestamp_us) - int(previous_timestamp_us)
    if gap_us < 0:
        raise StateTimingError(
            f"CAN timestamp moved backwards stream={stream_name} gap={gap_us * 1e-3:.3f}ms"
        )
    if gap_us > maximum_gap_us:
        raise StateTimingError(
            f"CAN frame gap exceeded stream={stream_name} gap={gap_us * 1e-3:.3f}ms "
            f"limit={maximum_gap_us * 1e-3:.3f}ms"
        )


def _make_lowpass(cutoff_hz: float | None) -> OnePoleLowPass | None:
    return OnePoleLowPass(cutoff_hz) if cutoff_hz is not None else None


def _apply_lowpass(
    filt: OnePoleLowPass | None,
    value: np.ndarray,
    timestamp_us: int,
) -> np.ndarray:
    return filt.apply(value, timestamp_us) if filt is not None else value.copy()


def _fuse_acceleration(ddq_from_dq: float, ddq_from_q: float) -> float:
    """Fuse two independent acceleration estimates with a least-squares mean."""
    estimates = np.asarray((ddq_from_dq, ddq_from_q), dtype=np.float64)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _nearest(history: deque, target_us: int):
    if not history:
        return None
    nearest = history[0]
    nearest_distance = abs(nearest.timestamp_us - target_us)
    for sample in history:
        distance = abs(sample.timestamp_us - target_us)
        if distance > nearest_distance and sample.timestamp_us > target_us:
            break
        if distance < nearest_distance:
            nearest = sample
            nearest_distance = distance
    return nearest
