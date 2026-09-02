from __future__ import annotations

import time

import numpy as np

from inference.async_fast_slow import (
    ActionPlanBuffer,
    StateHistoryBuffer,
    WMTargetBuffer,
    WMWorker,
)


def _fill_history(buffer: StateHistoryBuffer, end: float) -> None:
    for index in range(60):
        timestamp = end - 0.59 + index * 0.01
        q = np.full(7, float(index))
        buffer.append(timestamp, q, np.zeros(7), tau=q, q_cmd=q + 1.0)


def test_state_history_is_fixed_rate_and_recomputes_delta_q() -> None:
    buffer = StateHistoryBuffer()
    _fill_history(buffer, 10.0)

    history = buffer.query(10.0)

    assert history is not None
    assert history.timestamps_s.shape == (50,)
    np.testing.assert_allclose(history.timestamps_s, 9.51 + np.arange(50) * 0.01)
    np.testing.assert_allclose(history.delta_q[-1], np.ones(7))


def test_action_plan_uses_zoh_on_absolute_timestamps() -> None:
    buffer = ActionPlanBuffer(default_step_s=0.04)
    values = np.arange(56, dtype=np.float64).reshape(8, 7)
    buffer.append(values, start_time_s=5.0)

    result = buffer.query_with_timestamps(5.0, 5.32)

    assert result is not None
    assert result.values.shape == (32, 7)
    np.testing.assert_allclose(result.values[0], values[0])
    np.testing.assert_allclose(result.values[4], values[1])


def test_wm_worker_drops_expired_prediction_prefix() -> None:
    state = StateHistoryBuffer()
    action = ActionPlanBuffer(default_step_s=0.04)
    target = WMTargetBuffer()
    t_start = time.monotonic()
    _fill_history(state, t_start)
    action.append(np.zeros((8, 7)), start_time_s=t_start - 0.01)

    def infer(_history, _action):
        time.sleep(0.025)
        return {"q_ref": np.arange(32 * 7, dtype=np.float64).reshape(32, 7), "tau_ref": np.zeros((32, 7))}

    worker = WMWorker(
        state,
        action,
        target,
        infer,
        request_period_s=1.0,
        auto_request=False,
    )
    assert worker.request(t_start)
    worker.start()
    deadline = time.monotonic() + 1.0
    while worker.inference_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    worker.stop()

    assert worker.inference_count == 1
    segments = tuple(target._segments)
    assert len(segments) == 1
    assert segments[0].timestamps_s[0] > t_start + 0.02
    assert segments[0].q_ref[0, 0] >= 21.0


def test_wm_target_blends_overlapping_segments() -> None:
    target = WMTargetBuffer(blend_duration_s=0.04)
    timestamps = np.array([1.0, 1.01, 1.02])
    target.append(timestamps, np.zeros((3, 7)), np.zeros((3, 7)))
    target.append(timestamps, np.full((3, 7), 4.0), np.full((3, 7), 8.0))

    q_ref, tau_ref = target.query(1.02)

    np.testing.assert_allclose(q_ref, np.full(7, 2.0))
    np.testing.assert_allclose(tau_ref, np.full(7, 4.0))
