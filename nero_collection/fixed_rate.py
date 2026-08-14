from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from nero_collection.time_utils import now_us


@dataclass(frozen=True)
class FixedRateSampleTiming:
    index: int
    command_s: float
    read_s: float
    safety_s: float
    append_s: float
    camera_s: float
    total_s: float


@dataclass(frozen=True)
class CapturedArmSample:
    timestamp_us: int
    state: object
    q_cmd: np.ndarray
    timing: FixedRateSampleTiming


class FixedRateTicker:
    def __init__(
        self,
        sample_rate_hz: float,
        maximum_lateness_s: float,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        rate = float(sample_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if maximum_lateness_s <= 0.0 or not np.isfinite(maximum_lateness_s):
            raise ValueError("maximum_lateness_s must be positive and finite")
        self.period_s = 1.0 / rate
        self.maximum_lateness_s = float(maximum_lateness_s)
        self._monotonic = time.monotonic if monotonic is None else monotonic
        self._sleep = time.sleep if sleep is None else sleep
        self._started_s = self._monotonic()
        self._tick_index = 0

    def wait(self, context: str) -> tuple[int, float]:
        tick_index = self._tick_index
        deadline_s = self._started_s + tick_index * self.period_s
        remaining_s = deadline_s - self._monotonic()
        lateness_s = 0.0
        if remaining_s > 0.0:
            self._sleep(remaining_s)
        else:
            lateness_s = -remaining_s
            if tick_index > 0 and lateness_s > self.maximum_lateness_s:
                raise RuntimeError(
                    f"fixed-rate collection missed deadline by {lateness_s:.4f}s "
                    f"at tick {tick_index}; limit={self.maximum_lateness_s:.4f}s; "
                    f"context={context}"
                )
        self._tick_index += 1
        return tick_index, lateness_s


class FixedRateJointCollector:
    """Own command, latest-state capture, timestamping and recording at one rate."""

    def __init__(
        self,
        arm,
        cameras,
        buffer,
        *,
        sample_rate_hz: float,
        maximum_lateness_s: float,
        state_timeout_s: float,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        timestamp_us: Callable[[], int] | None = None,
    ) -> None:
        if state_timeout_s <= 0.0 or not np.isfinite(state_timeout_s):
            raise ValueError("state_timeout_s must be positive and finite")
        self.arm = arm
        self.cameras = cameras
        self.buffer = buffer
        self.state_timeout_s = float(state_timeout_s)
        self._monotonic = time.monotonic if monotonic is None else monotonic
        self._sleep = time.sleep if sleep is None else sleep
        self._timestamp_us = now_us if timestamp_us is None else timestamp_us
        self._ticker = FixedRateTicker(
            sample_rate_hz,
            maximum_lateness_s,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )
        self.period_s = self._ticker.period_s
        self.last_lateness_s = 0.0

    def replace_buffer(self, buffer) -> None:
        self.buffer = buffer

    def capture(
        self,
        q_cmd: np.ndarray,
        *,
        store: bool,
        context: str,
        validate: Callable[[object, np.ndarray], None],
    ) -> CapturedArmSample:
        command = np.asarray(q_cmd, dtype=np.float64).reshape(-1)
        if command.size != 7 or not np.isfinite(command).all():
            raise ValueError(f"fixed-rate collection requires a finite 7D q_cmd: {command}")

        tick_index, self.last_lateness_s = self._ticker.wait(context)

        sample_started_s = self._monotonic()
        self.arm.command_joint_positions(command)
        command_finished_s = self._monotonic()
        state = self._read_finite_state(context)
        read_finished_s = self._monotonic()
        validate(state, command)
        safety_finished_s = self._monotonic()
        captured_timestamp_us = int(self._timestamp_us())
        self.buffer.append_teleop(
            captured_timestamp_us,
            follower_values(state, command),
            store=store,
        )
        append_finished_s = self._monotonic()
        for frame in self.cameras.poll():
            if store:
                self.buffer.append_camera(
                    frame.camera_name,
                    frame.timestamp_us,
                    frame.frame,
                )
        sample_finished_s = self._monotonic()
        timing = FixedRateSampleTiming(
            index=tick_index,
            command_s=command_finished_s - sample_started_s,
            read_s=read_finished_s - command_finished_s,
            safety_s=safety_finished_s - read_finished_s,
            append_s=append_finished_s - safety_finished_s,
            camera_s=sample_finished_s - append_finished_s,
            total_s=sample_finished_s - sample_started_s,
        )
        return CapturedArmSample(
            timestamp_us=captured_timestamp_us,
            state=state,
            q_cmd=command.copy(),
            timing=timing,
        )

    def _read_finite_state(self, context: str):
        deadline_s = self._monotonic() + self.state_timeout_s
        invalid_fields = ()
        while True:
            state = self.arm.read_state()
            invalid_fields = tuple(
                name
                for name in ("q", "dq", "torque")
                if not _is_finite_joint_vector(getattr(state, name, None))
            )
            if not invalid_fields:
                return state
            if self._monotonic() >= deadline_s:
                raise RuntimeError(
                    "follower SDK cache did not produce a finite state within "
                    f"{self.state_timeout_s:.3f}s at {context}; "
                    f"invalid fields={invalid_fields}"
                )
            self._sleep(0.001)


def follower_values(state, q_cmd: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    return {
        "q_follower": ("q", np.asarray(state.q, dtype=np.float64)),
        "q_cmd": ("q", np.asarray(q_cmd, dtype=np.float64)),
        "dq_follower": ("velocity", np.asarray(state.dq, dtype=np.float64)),
        "ee_pose_follower": ("ee_pose", np.asarray(state.ee_pose, dtype=np.float64)),
        "tau_follower": ("torque", np.asarray(state.torque, dtype=np.float64)),
        "current_follower": ("current", np.asarray(state.current, dtype=np.float64)),
    }


def _is_finite_joint_vector(value) -> bool:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return vector.size == 7 and np.isfinite(vector).all()
