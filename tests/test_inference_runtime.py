from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.config import (
    CheckpointConfig,
    InferenceConfig,
    ObservationProtectionConfig,
    PredictorConfig,
    RobotConfig,
    RuntimeConfig,
    load_inference_config,
)
from inference.runtime import NeroInferenceRuntime
from nero_collection.arms.base import ArmState
from nero_collection.cameras import CameraFrame
from nero_collection.tau_ext_inference import OnlineTauExtResult


ROOT = Path(__file__).resolve().parents[1]


class _Arm:
    def __init__(self):
        self.timestamp = 1000
        self.commands = []
        self.reset_commands = []
        self.connected = False
        self.enabled = False
        self.disable_count = 0
        self.follower_mode_count = 0
        self.impedance_mode_count = 0
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.ddq = np.zeros(7)
        self.nan_q_reads = 0
        self.nan_torque_reads = 0
        self.state_alignment_args = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.disable_count += 1

    def set_follower_mode(self):
        self.follower_mode_count += 1

    def configure_state_alignment(self, *args):
        self.state_alignment_args = args

    def validate_joint_impedance_support(self):
        pass

    def configure_joint_impedance_mode(self):
        self.impedance_mode_count += 1

    def read_state(self):
        self.timestamp += 1000
        zeros = np.zeros(7)
        q = self.q.copy()
        if self.nan_q_reads > 0:
            self.nan_q_reads -= 1
            q[:] = np.nan
        torque = zeros.copy()
        if self.nan_torque_reads > 0:
            self.nan_torque_reads -= 1
            torque[:] = np.nan
        return ArmState(
            q=q,
            dq=self.dq.copy(),
            ddq=self.ddq.copy(),
            ee_pose=np.eye(4),
            torque=torque,
            current=zeros.copy(),
            timestamp_us=self.timestamp,
            acquired_timestamp_us=self.timestamp,
            q_timestamp_us=self.timestamp,
            q_acquired_timestamp_us=self.timestamp,
        )

    def command_joint_impedance(self, **kwargs):
        self.commands.append(kwargs)

    def command_joint_positions(self, q):
        self.commands.append({"q_position": np.asarray(q).copy()})

    def move_joints(self, q):
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.reset_commands.append(self.q.copy())

    def wait_motion_done(self, timeout_s):
        return True


class _Cameras:
    def start(self):
        pass

    def stop(self):
        pass

    def poll(self):
        return [
            CameraFrame(
                camera_name="wrist",
                timestamp_us=1000,
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        ]


class _TauExt:
    def __init__(self):
        self.reset_count = 0
        self.estimate_count = 0

    def warm_up(self):
        pass

    def reset_recurrent_state(self):
        self.reset_count += 1

    def reset_episode(self):
        self.reset_count += 1

    def estimate_aligned(self, timestamp_us, q, dq, tau, q_cmd):
        self.estimate_count += 1
        return OnlineTauExtResult(
            timestamp_us=timestamp_us,
            q=np.asarray(q).copy(),
            dq=np.asarray(dq).copy(),
            ddq_kf_causal=np.zeros(7),
            tau=np.asarray(tau).copy(),
            tau_id=np.zeros(7),
            tau_id_filtered=np.zeros(7),
            tau_f_pred=np.zeros(7),
            tau_next_pred=np.zeros(7),
            tau_ext_cal=np.zeros(7),
            tau_ext_pred=np.zeros(7),
        )


class _Wrench:
    def __init__(self, values=None):
        self.values = iter(values) if values is not None else None
        self.map_count = 0

    def map_joint_torque(self, q, tau):
        self.map_count += 1
        value = 0.0 if self.values is None else next(self.values)
        return SimpleNamespace(wrench=np.full(6, value, dtype=np.float64))


class _Dynamics:
    def snapshot(self, q, dq):
        return SimpleNamespace(pose=np.eye(4))


class _Pipeline:
    def __init__(self):
        self.controller = SimpleNamespace(model=_Dynamics())
        self.inputs = []
        self.closed = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def step(self, value):
        self.inputs.append(value)
        return SimpleNamespace(tau_command=np.ones(7))

    def close(self):
        self.closed = True


def _config() -> InferenceConfig:
    checkpoint = CheckpointConfig(ROOT / "unused.ckpt", device="cpu")
    return InferenceConfig(
        dp_checkpoint=checkpoint,
        pinn_checkpoint=checkpoint,
        robot=RobotConfig(ROOT / "urdf/nero/nero_with_gripper.urdf"),
        runtime=RuntimeConfig(
            ROOT / "configs/master_slave_can.yaml",
            maximum_state_age_s=1.0e12,
        ),
    )


def _use_fast_reset(runtime: NeroInferenceRuntime) -> None:
    command = replace(
        runtime.collection.teleop.command,
        reset_wait_s=0.0,
        reset_test_sample_time=1,
        reset_interpolation_enabled=False,
    )
    runtime.collection = replace(
        runtime.collection,
        teleop=replace(runtime.collection.teleop, command=command),
    )


def _protected_config(
    *,
    warmup_duration_s: float,
    predictor_enabled: bool = True,
    median_window: int = 5,
    cutoff_hz: float = 5.0,
) -> InferenceConfig:
    config = _config()
    return replace(
        config,
        pinn_checkpoint=config.pinn_checkpoint if predictor_enabled else None,
        predictor=replace(config.predictor, enabled=predictor_enabled),
        observation_protection=ObservationProtectionConfig(
            enabled=True,
            warmup_duration_s=warmup_duration_s,
            wrench_median_window=median_window,
            wrench_lowpass_cutoff_hz=cutoff_hz,
        ),
    )


@pytest.mark.parametrize(
    ("protection_yaml", "message"),
    [
        ("warmup_duration_s: -0.1", "warmup_duration_s"),
        ("wrench_median_window: 4", "positive odd"),
        ("wrench_lowpass_cutoff_hz: 0", "lowpass_cutoff_hz"),
    ],
)
def test_observation_protection_config_validation(
    tmp_path: Path,
    protection_yaml: str,
    message: str,
) -> None:
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        f"""
dp_checkpoint: {{path: dp.ckpt, device: cpu}}
predictor: {{enabled: false}}
robot: {{urdf_path: robot.urdf}}
runtime: {{collection_config: collection.yaml}}
observation_protection:
  enabled: true
  {protection_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_inference_config(config_path)


def test_runtime_waits_for_consumer_only_when_hardware_ring_is_fresh() -> None:
    config = replace(
        _config(),
        runtime=replace(_config().runtime, maximum_state_age_s=0.1),
    )
    arm = _Arm()
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=False,
        pipeline=_Pipeline(),
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
        continuous_state_stream=True,
    )
    current_us = time.time_ns() // 1_000
    stale = SimpleNamespace(
        timestamp_us=2_000,
        acquired_timestamp_us=current_us - 1_000_000,
    )
    fresh = SimpleNamespace(
        timestamp_us=3_000,
        acquired_timestamp_us=current_us,
    )

    class Stream:
        fault = None

        def __init__(self) -> None:
            self.wait_calls = []

        @staticmethod
        def latest():
            return stale

        def wait_for_acquired_after(self, minimum_us, timeout_s):
            self.wait_calls.append((minimum_us, timeout_s))
            return fresh

        @staticmethod
        def processing_status():
            return 100, 0.02

    stream = Stream()
    runtime._state_stream = stream
    arm.peek_latest_state = lambda: SimpleNamespace(
        acquired_timestamp_us=time.time_ns() // 1_000
    )

    sample = runtime._read_inference_sample()

    assert sample is fresh
    assert len(stream.wait_calls) == 1
    assert stream.wait_calls[0][1] == pytest.approx(0.1)


def test_observation_warmup_samples_tau_f_and_wrench_without_inference() -> None:
    arm, pipeline, tau_ext, wrench = _Arm(), _Pipeline(), _TauExt(), _Wrench()
    runtime = NeroInferenceRuntime(
        _protected_config(warmup_duration_s=0.002),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=tau_ext,
        wrench_estimator=wrench,
    )
    _use_fast_reset(runtime)
    runtime.start()

    first = runtime.step()
    second = runtime.step()
    ready = runtime.step()

    assert first is None
    assert second is None
    assert ready is not None
    assert tau_ext.estimate_count == 3
    assert wrench.map_count == 3
    assert len(pipeline.inputs) == 1
    assert arm.commands == []
    runtime.stop()


def test_observation_wrench_filter_rejects_an_isolated_spike() -> None:
    pipeline = _Pipeline()
    runtime = NeroInferenceRuntime(
        _protected_config(
            warmup_duration_s=0.0,
            median_window=3,
            cutoff_hz=5.0,
        ),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench([1.0, 1000.0, 1.0]),
    )
    _use_fast_reset(runtime)
    runtime.start()

    assert runtime.step() is not None
    assert runtime.step() is not None
    assert runtime.step() is not None

    np.testing.assert_allclose(
        np.stack([sample.wrench_ext for sample in pipeline.inputs]),
        np.ones((3, 6)),
    )
    runtime.stop()


def test_observation_wrench_lowpass_smooths_step_input() -> None:
    pipeline = _Pipeline()
    cutoff_hz = 5.0
    runtime = NeroInferenceRuntime(
        _protected_config(
            warmup_duration_s=0.0,
            median_window=1,
            cutoff_hz=cutoff_hz,
        ),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench([0.0, 10.0]),
    )
    _use_fast_reset(runtime)
    runtime.start()

    runtime.step()
    runtime.step()

    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * 0.001)
    np.testing.assert_allclose(
        pipeline.inputs[-1].wrench_ext,
        np.full(6, alpha * 10.0),
    )
    runtime.stop()


def test_tau_ext_post_filter_bypasses_second_wrench_lowpass() -> None:
    pipeline = _Pipeline()
    tau_ext = _TauExt()
    tau_ext.config = SimpleNamespace(
        tau_ext_filter=SimpleNamespace(enabled=True),
    )
    runtime = NeroInferenceRuntime(
        _protected_config(
            warmup_duration_s=0.0,
            median_window=1,
            cutoff_hz=5.0,
        ),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=tau_ext,
        wrench_estimator=_Wrench([0.0, 10.0]),
    )
    _use_fast_reset(runtime)
    runtime.start()

    runtime.step()
    runtime.step()

    np.testing.assert_allclose(pipeline.inputs[-1].wrench_ext, 10.0)
    runtime.stop()


def test_episode_reset_restarts_warmup_and_clears_wrench_filter() -> None:
    pipeline = _Pipeline()
    runtime = NeroInferenceRuntime(
        _protected_config(
            warmup_duration_s=0.002,
            predictor_enabled=False,
            median_window=3,
            cutoff_hz=0.1,
        ),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
    )
    _use_fast_reset(runtime)
    runtime.start()

    assert [runtime.step(), runtime.step(), runtime.step()][-1] is not None
    runtime._restart_episode()
    restarted = [runtime.step(), runtime.step(), runtime.step()]

    assert restarted[0] is None
    assert restarted[1] is None
    assert restarted[2] is not None
    assert len(pipeline.inputs) == 2
    np.testing.assert_allclose(pipeline.inputs[0].wrench_ext, np.full(6, 10.0))
    np.testing.assert_allclose(pipeline.inputs[1].wrench_ext, np.zeros(6))
    runtime.stop()


def test_predictor_control_mode_and_commands_wait_for_observation_warmup() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    runtime = NeroInferenceRuntime(
        _protected_config(warmup_duration_s=0.002),
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()

    assert arm.impedance_mode_count == 0
    assert runtime.step() is None
    assert runtime.step() is None
    assert runtime.step() is None
    assert arm.impedance_mode_count == 1
    assert pipeline.inputs == []
    assert arm.commands == []

    assert runtime.step() is not None
    assert len(pipeline.inputs) == 1
    assert len(arm.commands) == 1
    runtime.stop()


def test_runtime_computes_wrench_and_requires_explicit_command_enable() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()
    output = runtime.step()
    runtime.stop()

    assert output is not None
    assert len(pipeline.inputs) == 1
    assert arm.commands == []
    assert pipeline.closed


def test_runtime_uses_aligned_dq_and_tau_ext_kalman_acceleration() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    arm.dq = np.linspace(0.1, 0.7, 7)
    arm.ddq = np.linspace(-0.7, -0.1, 7)
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    runtime.start()
    runtime.step()
    runtime.stop()

    command = runtime.collection.teleop.command
    q_state = runtime.collection.robot_states["q"]
    dq_state = runtime.collection.robot_states["velocity"]
    ddq_state = runtime.collection.robot_states["acceleration"]
    assert arm.state_alignment_args == (
        command.state_alignment_delay_s,
        command.sample_rate_hz,
        q_state.mean_window,
        q_state.lowpass_cutoff_hz if q_state.lowpass else None,
        dq_state.lowpass_cutoff_hz if dq_state.lowpass else None,
        ddq_state.lowpass_cutoff_hz if ddq_state.lowpass else None,
        command.maximum_can_frame_gap_s,
    )
    np.testing.assert_allclose(pipeline.inputs[-1].dq, arm.dq)
    np.testing.assert_allclose(pipeline.inputs[-1].ddq, 0.0)
    assert not np.allclose(pipeline.inputs[-1].ddq, arm.ddq)


def test_runtime_skips_transient_incomplete_aligned_state() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()
    arm.nan_torque_reads = 1

    skipped = runtime.step()
    recovered = runtime.step()
    runtime.stop()

    assert skipped is None
    assert recovered is not None
    assert len(pipeline.inputs) == 1


@pytest.mark.parametrize(
    ("executing", "expects_output"),
    ((False, False), (True, True)),
)
def test_open_loop_camera_staleness_waits_while_sampling_but_not_while_executing(
    monkeypatch: pytest.MonkeyPatch,
    executing: bool,
    expects_output: bool,
) -> None:
    arm, pipeline = _Arm(), _Pipeline()
    arm.timestamp = 200_000
    pipeline.open_loop_execution_active = executing
    config = replace(
        _config(),
        predictor=replace(_config().predictor, inference_mode="open_loop"),
        runtime=replace(_config().runtime, maximum_state_age_s=0.1),
    )
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    monkeypatch.setattr("inference.runtime.now_us", lambda: 202_500)
    runtime.start()
    if executing:
        runtime._latest_frame = CameraFrame(
            camera_name="wrist",
            timestamp_us=1000,
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )

    output = runtime.step()
    runtime.stop()

    assert (output is not None) is expects_output
    assert len(pipeline.inputs) == int(expects_output)


def test_runtime_sends_qp_torque_only_when_explicitly_enabled() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()
    runtime.step()
    runtime.stop()

    np.testing.assert_allclose(arm.commands[0]["t_ff"], np.ones(7))
    np.testing.assert_allclose(arm.commands[-1]["t_ff"], np.zeros(7))


def test_startup_reset_waits_through_transient_nan_verification_state() -> None:
    arm = _Arm()
    original_wait_motion_done = arm.wait_motion_done

    def wait_then_interrupt_one_state(timeout_s):
        result = original_wait_motion_done(timeout_s)
        arm.nan_q_reads = 1
        return result

    arm.wait_motion_done = wait_then_interrupt_one_state
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=_Pipeline(),
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    runtime.start()
    assert arm.reset_commands
    assert arm.enabled
    assert arm.disable_count == 0
    runtime.stop()


def test_runtime_sends_joint_positions_when_predictor_is_disabled() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    desired_q = np.linspace(0.0, 0.06, 7)

    def direct_step(value):
        pipeline.inputs.append(value)
        return SimpleNamespace(
            tau_command=np.zeros(7),
            joint_position_command=desired_q,
        )

    pipeline.step = direct_step
    config = replace(
        _config(),
        pinn_checkpoint=None,
        predictor=PredictorConfig(enabled=False, action_chunk_mode="first"),
    )
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()
    runtime.step()
    runtime.stop()

    assert len(arm.commands) == 1
    np.testing.assert_allclose(arm.commands[0]["q_position"], desired_q)


def test_runtime_maximum_steps_resets_and_final_quit_preserves_enabled_arm() -> None:
    arm, pipeline, tau_ext = _Arm(), _Pipeline(), _TauExt()
    config = replace(
        _config(),
        runtime=replace(_config().runtime, maximum_inference_steps=2),
    )
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=tau_ext,
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    keys = iter((None, None, "q"))

    cycles = runtime.run(read_key=lambda _timeout: next(keys))

    assert cycles == 2
    assert len(arm.reset_commands) == 3  # startup, step limit, final shutdown
    assert pipeline.reset_count == 3
    assert tau_ext.reset_count == 3
    assert arm.enabled
    assert arm.disable_count == 0
    assert not arm.connected


def test_runtime_i_resets_current_episode_before_q_exits() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    keys = iter(("i", "q"))

    cycles = runtime.run(read_key=lambda _timeout: next(keys))

    assert cycles == 0
    assert len(arm.reset_commands) == 3  # startup, i, q
    assert arm.enabled
    assert arm.disable_count == 0


def test_runtime_ctrl_c_resets_and_preserves_follower_enabled_state() -> None:
    arm = _Arm()
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=_Pipeline(),
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    def interrupt(_timeout):
        raise KeyboardInterrupt

    cycles = runtime.run(read_key=interrupt)

    assert cycles == 0
    assert len(arm.reset_commands) == 2  # startup and Ctrl-C shutdown
    assert arm.enabled
    assert arm.disable_count == 0
    assert not arm.connected


def test_runtime_exception_immediately_resets_and_preserves_enabled_arm() -> None:
    arm, pipeline = _Arm(), _Pipeline()

    def fail_step(_value):
        raise RuntimeError("synthetic OSC-QP failure")

    pipeline.step = fail_step
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    with pytest.raises(RuntimeError, match="synthetic OSC-QP failure"):
        runtime.run()

    assert len(arm.reset_commands) == 2  # startup and exception recovery
    assert arm.enabled
    assert arm.disable_count == 0
    assert not arm.connected
    assert pipeline.closed


def test_runtime_never_disables_when_exception_reset_also_fails() -> None:
    arm, pipeline = _Arm(), _Pipeline()

    def fail_step(_value):
        raise RuntimeError("synthetic inference failure")

    pipeline.step = fail_step
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=True,
        pipeline=pipeline,
        arm=arm,
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    original_reset = runtime._reset_arm_to_rest
    reset_calls = 0

    def fail_second_reset(reason):
        nonlocal reset_calls
        reset_calls += 1
        if reset_calls == 2:
            raise RuntimeError("synthetic reset failure")
        return original_reset(reason)

    runtime._reset_arm_to_rest = fail_second_reset

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        runtime.run()

    assert reset_calls == 2
    assert arm.disable_count == 0
    assert arm.enabled
    assert not arm.connected
    # Startup reset plus the emergency current-q position hold.
    assert len(arm.reset_commands) == 2
