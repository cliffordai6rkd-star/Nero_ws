from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from inference.state_stream import ContinuousInferenceStateStream
from nero_collection.arms.base import ArmState
from nero_collection.tau_ext_inference import OnlineTauExtResult


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


class _TauExt:
    def __init__(self) -> None:
        self.count = 0

    def estimate_aligned(self, timestamp_us, q, dq, tau, q_cmd):
        self.count += 1
        zeros = np.zeros(7, dtype=np.float64)
        return OnlineTauExtResult(
            timestamp_us=timestamp_us,
            q=np.asarray(q).copy(),
            dq=np.asarray(dq).copy(),
            ddq_kf_causal=zeros.copy(),
            tau=np.asarray(tau).copy(),
            tau_id=zeros.copy(),
            tau_id_filtered=zeros.copy(),
            tau_other_pred=zeros.copy(),
            tau_next_pred=zeros.copy(),
            tau_ext_cal=zeros.copy(),
            tau_ext_pred=zeros.copy(),
        )


class _Wrench:
    def map_joint_torque(self, q, tau):
        return SimpleNamespace(wrench=np.zeros(6, dtype=np.float64))


def test_continuous_state_stream_passes_zoh_q_cmd_to_dual_tau_ext() -> None:
    class TauExt:
        def __init__(self) -> None:
            self.q_cmds = []

        def estimate_aligned(
            self,
            timestamp_us,
            q,
            dq,
            tau,
            q_cmd,
        ):
            self.q_cmds.append(np.asarray(q_cmd).copy())
            zeros = np.zeros(7)
            return OnlineTauExtResult(
                timestamp_us=timestamp_us,
                q=np.asarray(q),
                dq=np.asarray(dq),
                ddq_kf_causal=zeros,
                tau=np.asarray(tau),
                tau_id=zeros,
                tau_id_filtered=zeros,
                tau_other_pred=zeros,
                tau_next_pred=zeros,
                tau_ext_cal=zeros,
                tau_ext_pred=zeros,
            )

    arm = _Arm()
    tau_ext = TauExt()
    q_cmd = np.linspace(0.1, 0.7, 7)
    stream = ContinuousInferenceStateStream(
        arm,
        tau_ext,
        _Wrench(),
        q_cmd_provider=lambda _timestamp_us: q_cmd.copy(),
    )

    sample = stream.process_state(arm.read_state())

    assert sample is not None
    assert len(tau_ext.q_cmds) == 1
    np.testing.assert_allclose(tau_ext.q_cmds[0], q_cmd)


def test_state_stream_maps_raw_and_filtered_tau_ext_without_refiltering() -> None:
    class Wrench:
        @staticmethod
        def map_joint_torque(_q, tau):
            return SimpleNamespace(wrench=np.asarray(tau, dtype=np.float64)[:6])

    processor_inputs = []

    def process(raw_wrench, filtered_wrench, _timestamp_us):
        processor_inputs.append((raw_wrench.copy(), filtered_wrench.copy()))
        return filtered_wrench.copy(), filtered_wrench.copy()

    stream = ContinuousInferenceStateStream(
        _Arm(),
        _TauExt(),
        Wrench(),
        wrench_processor=process,
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
    )
    zeros = np.zeros(7)
    result = OnlineTauExtResult(
        timestamp_us=1_000,
        q=zeros,
        dq=zeros,
        ddq_kf_causal=zeros,
        tau=zeros,
        tau_id=zeros,
        tau_id_filtered=zeros,
        tau_other_pred=zeros,
        tau_next_pred=zeros,
        tau_ext_cal=np.ones(7),
        tau_ext_pred=np.ones(7),
        tau_ext_cal_raw=np.full(7, 2.0),
        tau_ext_pred_raw=np.full(7, 2.0),
    )

    sample = stream._build_sample(1_000, 1_000, result, zeros)

    np.testing.assert_allclose(sample.raw_wrench, 2.0)
    np.testing.assert_allclose(sample.wrench, 1.0)
    np.testing.assert_allclose(processor_inputs[0][0], 2.0)
    np.testing.assert_allclose(processor_inputs[0][1], 1.0)


def test_continuous_state_stream_consumes_every_sample_independently() -> None:
    ready = threading.Event()
    seen = []
    tau_ext = _TauExt()

    def on_sample(sample) -> None:
        seen.append(sample.timestamp_us)
        if len(seen) >= 5:
            ready.set()

    stream = ContinuousInferenceStateStream(
        _Arm(),
        tau_ext,
        _Wrench(),
        on_sample=on_sample,
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    stream.stop()

    assert stream.fault is None
    assert tau_ext.count >= 5
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
        _TauExt(),
        _Wrench(),
        on_sample=lambda _sample: ready.set(),
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
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


def test_continuous_state_stream_reads_one_latest_state_per_tick() -> None:
    class Arm:
        source = _Arm()

        def read_state(self):
            return self.source.read_state()

    ready = threading.Event()
    seen = []
    stream = ContinuousInferenceStateStream(
        Arm(),
        _TauExt(),
        _Wrench(),
        on_sample=lambda sample: (seen.append(sample.timestamp_us), ready.set()),
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    for _ in range(100):
        if len(seen) >= 3:
            break
        threading.Event().wait(0.001)
    stream.stop()

    assert stream.fault is None
    assert len(seen) >= 3
    assert seen == sorted(seen)


def test_continuous_state_stream_fails_closed_on_sdk_read_error() -> None:
    class Arm:
        @staticmethod
        def read_state():
            raise RuntimeError("SDK read failed")

    stream = ContinuousInferenceStateStream(
        Arm(),
        _TauExt(),
        _Wrench(),
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
        poll_interval_s=0.0001,
    )

    stream.start()
    for _ in range(100):
        if stream.fault is not None:
            break
        threading.Event().wait(0.001)
    stream.stop()

    assert isinstance(stream.fault, RuntimeError)
    assert "SDK read failed" in str(stream.fault)


def test_continuous_state_stream_does_not_expand_sdk_backlog() -> None:
    class Arm:
        source = _Arm()

        def read_state(self):
            return self.source.read_state()

    ready = threading.Event()
    tau_ext = _TauExt()
    stream = ContinuousInferenceStateStream(
        Arm(),
        tau_ext,
        _Wrench(),
        on_sample=lambda sample: ready.set(),
        q_cmd_provider=lambda _timestamp_us: np.zeros(7),
        poll_interval_s=0.0001,
    )

    stream.start()
    assert ready.wait(timeout=1.0)
    stream.stop()

    assert stream.fault is None
    assert tau_ext.count >= 1
    assert len(stream.drain_after(0)) >= 1
