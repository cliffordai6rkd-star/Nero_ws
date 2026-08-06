from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from inference.state_stream import ContinuousInferenceStateStream
from nero_collection.arms.base import ArmState
from nero_collection.tau_f_inference import OnlineTauFResult


class _Arm:
    def __init__(self) -> None:
        self.timestamp_us = 1_000

    def read_state(self) -> ArmState:
        self.timestamp_us += 1_000
        zeros = np.zeros(7, dtype=np.float64)
        return ArmState(
            q=zeros.copy(),
            dq=zeros.copy(),
            ddq=zeros.copy(),
            ee_pose=np.eye(4),
            torque=zeros.copy(),
            current=zeros.copy(),
            timestamp_us=self.timestamp_us,
            acquired_timestamp_us=self.timestamp_us,
            q_timestamp_us=self.timestamp_us,
            q_acquired_timestamp_us=self.timestamp_us,
        )


class _TauF:
    def __init__(self) -> None:
        self.count = 0

    def estimate_aligned_raw(self, timestamp_us, q, dq, ddq, tau):
        self.count += 1
        zeros = np.zeros(7, dtype=np.float64)
        return OnlineTauFResult(
            timestamp_us=timestamp_us,
            q=np.asarray(q).copy(),
            dq=np.asarray(dq).copy(),
            ddq=np.asarray(ddq).copy(),
            tau=np.asarray(tau).copy(),
            tau_id=zeros.copy(),
            tau_f_cal=zeros.copy(),
            tau_f_pred=zeros.copy(),
            tau_ext=zeros.copy(),
        )


class _Wrench:
    def map_joint_torque(self, q, tau):
        return SimpleNamespace(wrench=np.zeros(6, dtype=np.float64))


def test_continuous_state_stream_consumes_every_sample_independently() -> None:
    ready = threading.Event()
    seen = []
    tau_f = _TauF()

    def on_sample(sample) -> None:
        seen.append(sample.timestamp_us)
        if len(seen) >= 5:
            ready.set()

    stream = ContinuousInferenceStateStream(
        _Arm(),
        tau_f,
        _Wrench(),
        on_sample=on_sample,
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    stream.stop()

    assert stream.fault is None
    assert tau_f.count >= 5
    assert seen == sorted(set(seen))
    assert stream.latest() is not None
    drained = stream.drain_after(0)
    assert len(drained) >= 5
    assert [sample.timestamp_us for sample in drained] == sorted(
        sample.timestamp_us for sample in drained
    )


def test_continuous_state_stream_reports_bounded_history_rollover() -> None:
    ready = threading.Event()
    stream = ContinuousInferenceStateStream(
        _Arm(),
        _TauF(),
        _Wrench(),
        on_sample=lambda _sample: ready.set(),
        history_size=2,
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    deadline = threading.Event()
    # The producer is independent of the consumer; allow it to evict at least
    # one old record from the bounded ring before stopping it.
    for _ in range(100):
        if stream.history_rollover_count > 0:
            break
        deadline.wait(0.001)
    stream.stop()

    assert stream.history_rollover_count > 0
    assert len(stream.drain_after(0)) <= 2


def test_continuous_state_stream_drains_all_isolated_process_states() -> None:
    class Arm:
        def __init__(self) -> None:
            self.sent = False

        def drain_states(self):
            if self.sent:
                return SimpleNamespace(states=(), dropped=0)
            self.sent = True
            source = _Arm()
            return SimpleNamespace(
                states=(source.read_state(), source.read_state(), source.read_state()),
                dropped=0,
            )

    ready = threading.Event()
    seen = []
    stream = ContinuousInferenceStateStream(
        Arm(),
        _TauF(),
        _Wrench(),
        on_sample=lambda sample: (seen.append(sample.timestamp_us), ready.set()),
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    for _ in range(100):
        if len(seen) == 3:
            break
        threading.Event().wait(0.001)
    stream.stop()

    assert stream.fault is None
    assert len(seen) == 3
    assert seen == sorted(seen)


def test_continuous_state_stream_fails_closed_on_hardware_ring_overrun() -> None:
    class Arm:
        @staticmethod
        def drain_states():
            return SimpleNamespace(states=(), dropped=2)

    stream = ContinuousInferenceStateStream(
        Arm(),
        _TauF(),
        _Wrench(),
        poll_interval_s=0.0001,
    )

    stream.start()
    for _ in range(100):
        if stream.fault is not None:
            break
        threading.Event().wait(0.001)
    stream.stop()

    assert isinstance(stream.fault, RuntimeError)
    assert "2 aligned states were overwritten" in str(stream.fault)


def test_continuous_state_stream_uses_one_tau_f_batch_for_ring_backlog() -> None:
    class Arm:
        def __init__(self) -> None:
            source = _Arm()
            self.states = (
                source.read_state(),
                source.read_state(),
                source.read_state(),
            )

        def drain_states(self):
            states, self.states = self.states, ()
            return SimpleNamespace(states=states, dropped=0)

    class BatchTauF(_TauF):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def estimate_aligned_raw_batch(self, samples):
            self.batch_sizes.append(len(samples))
            return tuple(
                self.estimate_aligned_raw(timestamp_us, q, dq, ddq, tau)
                for timestamp_us, q, dq, ddq, tau in samples
            )

    ready = threading.Event()
    tau_f = BatchTauF()
    stream = ContinuousInferenceStateStream(
        Arm(),
        tau_f,
        _Wrench(),
        on_sample=lambda sample: ready.set(),
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    stream.stop()

    assert stream.fault is None
    assert tau_f.batch_sizes == [3]
    assert len(stream.drain_after(0)) == 3
