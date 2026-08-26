from __future__ import annotations

import numpy as np

from inference.stages import ActionPlanExecutor, DPObservationBuffer


def _frame(marker: float) -> np.ndarray:
    return np.full((4, 5, 3), marker, dtype=np.float32)


def test_buffer_builds_causal_multicamera_snapshot() -> None:
    buffer = DPObservationBuffer(
        image_keys=("side", "wrist"),
        anchor_image_key="wrist",
        n_obs_steps=2,
        wrench_history_steps=1,
        observation_step_s=0.1,
        uses_wrench_observation=False,
    )
    for side_s, wrist_s, side_marker, wrist_marker, can_s in (
        (0.026, 0.040, 0.1, 0.3, 0.040),
        (0.066, 0.080, 0.2, 0.4, 0.080),
        # A side frame newer than the wrist anchor must not be selected.
        (0.106, 0.080, 0.9, 0.4, 0.081),
    ):
        buffer.append(
            {"side": _frame(side_marker), "wrist": _frame(wrist_marker)},
            np.full(6, can_s),
            image_timestamp_s={"side": side_s, "wrist": wrist_s},
            state_timestamp_s=can_s,
        )

    result = buffer.timed_snapshot(0.080)
    assert result is not None
    images, wrenches = result
    np.testing.assert_allclose(images["side"][:, 0, 0, 0], [0.1, 0.2])
    np.testing.assert_allclose(images["wrist"][:, 0, 0, 0], [0.3, 0.4])
    assert wrenches.shape == (2, 1, 6)


def test_open_loop_alignment_rejects_future_or_stale_can() -> None:
    buffer = DPObservationBuffer(
        image_keys=("wrist",),
        anchor_image_key="wrist",
        n_obs_steps=2,
        wrench_history_steps=1,
        uses_wrench_observation=True,
        observation_step_s=0.1,
        inference_mode="open_loop",
        maximum_alignment_gap_s=0.03,
    )
    buffer.append(
        _frame(0.1),
        np.zeros(6),
        image_timestamp_s=0.0,
        state_timestamp_s=0.0,
        allow_backfill=False,
    )
    buffer.append(
        _frame(0.2),
        np.ones(6),
        image_timestamp_s=0.1,
        state_timestamp_s=0.05,
        allow_backfill=False,
    )
    buffer.append_open_loop_can_observation(np.ones(6), 0.08)
    assert buffer.timed_snapshot_if_aligned(0.1) is not None

    buffer.clear()
    buffer.append(
        _frame(0.1),
        np.zeros(6),
        image_timestamp_s=0.0,
        state_timestamp_s=0.0,
        allow_backfill=False,
    )
    buffer.append(
        _frame(0.2),
        np.ones(6),
        image_timestamp_s=0.1,
        state_timestamp_s=0.0,
        allow_backfill=False,
    )
    buffer.append_open_loop_can_observation(np.ones(6), 0.06)
    assert buffer.timed_snapshot_if_aligned(0.1) is None


def test_continuous_can_forwards_world_model_values() -> None:
    calls: list[tuple[float, dict[str, np.ndarray]]] = []
    buffer = DPObservationBuffer(
        image_keys=("wrist",),
        anchor_image_key="wrist",
        n_obs_steps=1,
        on_continuous_can=lambda timestamp, values: calls.append((timestamp, dict(values))),
    )
    buffer.append_continuous_can_observation(
        q=np.zeros(7),
        dq=np.ones(7),
        ddq=np.full(7, 2.0),
        tau=np.full(7, 3.0),
        wrench=np.full(6, 4.0),
        timestamp_s=1.25,
    )
    assert len(calls) == 1
    assert calls[0][0] == 1.25
    np.testing.assert_allclose(calls[0][1]["wrench"], 4.0)


def test_action_plan_executor_advances_by_timestamp_and_reports_completion() -> None:
    executor = ActionPlanExecutor()
    plan = np.stack([np.full(7, value) for value in (0.0, 1.0, 2.0)])
    executor.install(
        plan,
        held_action=plan[-1],
        timestamp_s=1.0,
        scheduled=True,
        step_s=0.1,
    )
    assert executor.active
    assert executor.index == 0
    assert not executor.advance(1.05)
    assert executor.advance(1.2) is False
    assert executor.index == 2
    assert executor.advance(1.3) is True
    assert not executor.active
    np.testing.assert_allclose(executor.action_chunk, plan[-1:])
