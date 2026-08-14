from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nero_collection.fixed_rate import FixedRateJointCollector, FixedRateTicker


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration_s: float) -> None:
        self.now += duration_s


def test_fixed_rate_ticker_produces_exact_100hz_deadlines() -> None:
    clock = _Clock()
    ticker = FixedRateTicker(
        100.0,
        0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    observed = []
    for _ in range(5):
        tick, lateness = ticker.wait("test")
        observed.append((tick, clock.now, lateness))

    assert [value[0] for value in observed] == list(range(5))
    assert [value[1] for value in observed] == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04]
    )
    assert [value[2] for value in observed] == pytest.approx(np.zeros(5))


def test_fixed_rate_ticker_fails_instead_of_silently_recording_a_gap() -> None:
    clock = _Clock()
    ticker = FixedRateTicker(
        100.0,
        0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    ticker.wait("first")
    clock.now = 0.07

    with pytest.raises(RuntimeError, match="missed deadline"):
        ticker.wait("delayed save")


def test_joint_collector_records_raw_state_and_command_on_every_tick() -> None:
    clock = _Clock()

    class Arm:
        def __init__(self) -> None:
            self.commands = []

        def command_joint_positions(self, q_cmd) -> None:
            self.commands.append(np.asarray(q_cmd).copy())

        def read_state(self):
            value = float(len(self.commands))
            return SimpleNamespace(
                q=np.full(7, value),
                dq=np.zeros(7),
                torque=np.zeros(7),
                current=np.zeros(7),
                ee_pose=np.eye(4),
            )

    class Buffer:
        def __init__(self) -> None:
            self.rows = []

        def append_teleop(self, timestamp_us, values, *, store) -> None:
            if store:
                self.rows.append((timestamp_us, values))

    arm = Arm()
    buffer = Buffer()
    collector = FixedRateJointCollector(
        arm,
        SimpleNamespace(poll=lambda: ()),
        buffer,
        sample_rate_hz=100.0,
        maximum_lateness_s=0.05,
        state_timeout_s=0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timestamp_us=lambda: int(round(clock.now * 1.0e6)) + 1,
    )

    for index in range(4):
        collector.capture(
            np.full(7, index),
            store=True,
            context="test",
            validate=lambda _state, _command: None,
        )

    assert len(arm.commands) == len(buffer.rows) == 4
    assert np.diff([row[0] for row in buffer.rows]) == pytest.approx(
        np.full(3, 10_000)
    )
    for index, (_, values) in enumerate(buffer.rows):
        assert values["q_cmd"][1] == pytest.approx(np.full(7, index))
        assert values["q_follower"][1] == pytest.approx(np.full(7, index + 1))


def test_replacing_episode_buffer_does_not_reset_the_100hz_clock() -> None:
    clock = _Clock()

    class Arm:
        def command_joint_positions(self, _q_cmd) -> None:
            pass

        def read_state(self):
            return SimpleNamespace(
                q=np.zeros(7),
                dq=np.zeros(7),
                torque=np.zeros(7),
                current=np.zeros(7),
                ee_pose=np.eye(4),
            )

    class Buffer:
        def __init__(self) -> None:
            self.timestamps = []

        def append_teleop(self, timestamp_us, _values, *, store) -> None:
            if store:
                self.timestamps.append(timestamp_us)

    first = Buffer()
    second = Buffer()
    collector = FixedRateJointCollector(
        Arm(),
        SimpleNamespace(poll=lambda: ()),
        first,
        sample_rate_hz=100.0,
        maximum_lateness_s=0.05,
        state_timeout_s=0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timestamp_us=lambda: int(round(clock.now * 1.0e6)) + 1,
    )
    for _ in range(2):
        collector.capture(
            np.zeros(7),
            store=True,
            context="before split",
            validate=lambda _state, _command: None,
        )
    collector.replace_buffer(second)
    for _ in range(2):
        collector.capture(
            np.zeros(7),
            store=True,
            context="after split",
            validate=lambda _state, _command: None,
        )

    combined = first.timestamps + second.timestamps
    assert combined == [1, 10_001, 20_001, 30_001]
