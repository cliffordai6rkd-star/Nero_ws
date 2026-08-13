from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from threading import Lock

import numpy as np

from nero_collection.filters import OnePoleLowPass


log = logging.getLogger(__name__)


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
    q_source_timestamp_us: np.ndarray
    motor_source_timestamp_us: np.ndarray


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


class CanStateAssembler:
    """Assemble complete arm states on the real CAN source timeline.

    A state is finalized when every required source has reached a new watermark.
    The watermark is itself one of the source timestamps; no periodic timestamp is
    synthesized. Each source contributes its nearest real observation and the
    complete frame is rejected when its source skew exceeds the configured limit.
    """

    def __init__(
        self,
        *,
        dof: int,
        q_groups: tuple[tuple[str, tuple[int, ...]], ...],
        maximum_source_skew_s: float,
        q_mean_window_samples: int = 5,
        q_lowpass_cutoff_hz: float | None = 10.0,
        dq_lowpass_cutoff_hz: float | None = 6.0,
        ddq_lowpass_cutoff_hz: float | None = 3.0,
        maximum_input_gap_s: float = 0.03,
        history_size: int = 512,
    ) -> None:
        self.dof = int(dof)
        self.q_groups = q_groups
        self.maximum_source_skew_us = int(
            round(float(maximum_source_skew_s) * 1_000_000.0)
        )
        if self.dof <= 0:
            raise ValueError("state timeline dof must be positive")
        if self.maximum_source_skew_us <= 0:
            raise ValueError("maximum state source skew must be positive")
        if not np.isfinite(maximum_input_gap_s) or maximum_input_gap_s <= 0:
            raise ValueError("maximum input gap must be positive and finite")
        self.maximum_input_gap_us = int(round(maximum_input_gap_s * 1_000_000.0))
        configured_q_mean_window = int(q_mean_window_samples)
        if configured_q_mean_window <= 0:
            raise ValueError("q-mean window must be a positive integer")
        self.q_mean_window_samples = configured_q_mean_window
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
        self._completed: deque[AlignedArmSample] = deque()
        self._last_watermark_us = 0
        self._consumed_q_timestamp_us = {name: 0 for name, _ in self.q_groups}
        self._consumed_motor_timestamp_us = [0] * self.dof
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
            self._finalize_available_locked()

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

            ddq_filtered = float(
                _apply_lowpass(
                    self._motor_ddq_filters[joint_index],
                    np.asarray([ddq_from_dq], dtype=np.float64),
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
            self._finalize_available_locked()

    def drain_completed(self) -> tuple[AlignedArmSample, ...]:
        with self._lock:
            samples = tuple(self._completed)
            self._completed.clear()
            return samples

    def _finalize_available_locked(self) -> None:
        histories = list(self._q_state_history.values()) + self._motor_history
        if any(not history for history in histories):
            return
        if any(
            history[-1].timestamp_us <= self._consumed_q_timestamp_us[name]
            for name, history in self._q_state_history.items()
        ) or any(
            history[-1].timestamp_us <= self._consumed_motor_timestamp_us[index]
            for index, history in enumerate(self._motor_history)
        ):
            return
        watermark_us = min(history[-1].timestamp_us for history in histories)
        if watermark_us <= self._last_watermark_us:
            return
        sample = self._assemble_locked(watermark_us)
        if sample is None:
            return
        self._completed.append(sample)
        self._last_watermark_us = watermark_us
        self._discard_consumed_locked(sample)

    def _assemble_locked(self, watermark_us: int) -> AlignedArmSample | None:
        q = np.empty(self.dof, dtype=np.float64)
        dq = np.empty(self.dof, dtype=np.float64)
        ddq = np.empty(self.dof, dtype=np.float64)
        torque = np.empty(self.dof, dtype=np.float64)
        current = np.empty(self.dof, dtype=np.float64)
        q_source_timestamp_us = np.empty(self.dof, dtype=np.int64)
        motor_source_timestamp_us = np.empty(self.dof, dtype=np.int64)
        for name, indices in self.q_groups:
            state = _nearest(
                self._q_state_history[name],
                watermark_us,
            )
            if state is None:
                return None
            joint_indices = list(indices)
            q[joint_indices] = state.q
            q_source_timestamp_us[joint_indices] = state.timestamp_us
        for joint_index, history in enumerate(self._motor_history):
            motor = _nearest(
                history,
                watermark_us,
            )
            if motor is None:
                return None
            dq[joint_index] = motor.velocity
            ddq[joint_index] = motor.ddq
            torque[joint_index] = motor.torque
            current[joint_index] = motor.current
            motor_source_timestamp_us[joint_index] = motor.timestamp_us
        source_timestamps = np.concatenate(
            (q_source_timestamp_us, motor_source_timestamp_us)
        )
        source_skew_us = int(np.max(source_timestamps) - np.min(source_timestamps))
        if source_skew_us > self.maximum_source_skew_us:
            # This candidate is incomplete in time, but the CAN stream itself is
            # still healthy. Retire only its oldest source observation so a
            # later real observation can form the next complete state.
            oldest_source_timestamp_us = int(np.min(source_timestamps))
            self._discard_oldest_sources_locked(
                q_source_timestamp_us,
                motor_source_timestamp_us,
                oldest_source_timestamp_us,
            )
            log.warning(
                "rejected complete CAN state candidate due to source skew: "
                "watermark=%d skew=%.3fms limit=%.3fms oldest_source=%d",
                watermark_us,
                source_skew_us * 1e-3,
                self.maximum_source_skew_us * 1e-3,
                oldest_source_timestamp_us,
            )
            return None
        return AlignedArmSample(
            timestamp_us=watermark_us,
            q=q,
            dq=dq,
            ddq=ddq,
            torque=torque,
            current=current,
            q_source_timestamp_us=q_source_timestamp_us,
            motor_source_timestamp_us=motor_source_timestamp_us,
        )

    def _discard_consumed_locked(self, sample: AlignedArmSample) -> None:
        for name, indices in self.q_groups:
            selected_us = int(sample.q_source_timestamp_us[indices[0]])
            self._consumed_q_timestamp_us[name] = selected_us
            history = self._q_state_history[name]
            # A physical source observation may contribute to at most one
            # complete state.  Remove the selected sample itself as well as
            # anything older so nearest matching cannot silently reuse it.
            while history and history[0].timestamp_us <= selected_us:
                history.popleft()
        for index, history in enumerate(self._motor_history):
            selected_us = int(sample.motor_source_timestamp_us[index])
            self._consumed_motor_timestamp_us[index] = selected_us
            while history and history[0].timestamp_us <= selected_us:
                history.popleft()

    def _discard_oldest_sources_locked(
        self,
        q_source_timestamp_us: np.ndarray,
        motor_source_timestamp_us: np.ndarray,
        oldest_source_timestamp_us: int,
    ) -> None:
        for name, indices in self.q_groups:
            selected_us = int(q_source_timestamp_us[indices[0]])
            if selected_us != oldest_source_timestamp_us:
                continue
            self._consumed_q_timestamp_us[name] = max(
                self._consumed_q_timestamp_us[name], selected_us
            )
            history = self._q_state_history[name]
            while history and history[0].timestamp_us <= selected_us:
                history.popleft()
        for index, history in enumerate(self._motor_history):
            selected_us = int(motor_source_timestamp_us[index])
            if selected_us != oldest_source_timestamp_us:
                continue
            self._consumed_motor_timestamp_us[index] = max(
                self._consumed_motor_timestamp_us[index], selected_us
            )
            while history and history[0].timestamp_us <= selected_us:
                history.popleft()


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


def _nearest(
    history: deque,
    target_us: int,
    *,
    maximum_timestamp_us: int | None = None,
):
    if not history:
        return None
    nearest = None
    nearest_distance = None
    for sample in history:
        if (
            maximum_timestamp_us is not None
            and sample.timestamp_us > maximum_timestamp_us
        ):
            break
        distance = abs(sample.timestamp_us - target_us)
        if (
            nearest_distance is not None
            and distance > nearest_distance
            and sample.timestamp_us > target_us
        ):
            break
        if nearest_distance is None or distance < nearest_distance:
            nearest = sample
            nearest_distance = distance
    return nearest
