from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.config import (
    CheckpointConfig,
    ArchitectureConfig,
    InferenceConfig,
    ObservationProtectionConfig,
    PredictorConfig,
    RobotConfig,
    RuntimeConfig,
    load_inference_config,
)
from inference.runtime import NeroInferenceRuntime
from inference.control.nero import NeroPipelineOutputController
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


class _TwoCameras(_Cameras):
    def poll(self):
        return [
            CameraFrame(
                camera_name="side",
                timestamp_us=26_000,
                frame=np.full((8, 8, 3), 1, dtype=np.uint8),
            ),
            CameraFrame(
                camera_name="wrist",
                timestamp_us=40_000,
                frame=np.full((8, 8, 3), 2, dtype=np.uint8),
            ),
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
            tau_other_pred=np.zeros(7),
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


def test_wm_tau_output_uses_zero_firmware_gains() -> None:
    """Direct WM torque mode must not add runtime damping or PD feedback."""

    arm = _Arm()
    config = SimpleNamespace(
        predictor=SimpleNamespace(enabled=True),
        execution=SimpleNamespace(mit_kp=np.full(7, 3.0), mit_kd=np.full(7, 0.5)),
        runtime=SimpleNamespace(command_kd=np.full(7, 9.0)),
    )
    controller = NeroPipelineOutputController(
        arm=arm,
        config=config,
        command_enabled=True,
    )
    observation = SimpleNamespace(q=np.full(7, 0.2))
    output = SimpleNamespace(
        control_mode="tau",
        tau_command=np.linspace(-0.3, 0.3, 7),
    )

    controller.send(observation, output)

    command = arm.commands[-1]
    np.testing.assert_allclose(command["q"], observation.q)
    np.testing.assert_allclose(command["v_des"], np.zeros(7))
    np.testing.assert_allclose(command["kp"], np.zeros(7))
    np.testing.assert_allclose(command["kd"], np.zeros(7))
    np.testing.assert_allclose(command["t_ff"], output.tau_command)


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


def test_observation_warmup_samples_tau_other_and_wrench_without_inference() -> None:
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

    assert first is None
    assert second is not None
    assert tau_ext.estimate_count == 2
    assert wrench.map_count == 2
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

    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * 0.01)
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


def test_tau_ext_post_filter_is_dp_input_while_plot_keeps_raw_signal() -> None:
    config = _protected_config(
        warmup_duration_s=0.0,
        median_window=1,
        cutoff_hz=5.0,
    )
    config = replace(
        config,
        wrench_visualization=replace(config.wrench_visualization, enabled=True),
    )
    plotted = []
    plotter = SimpleNamespace(
        append=lambda timestamp_us, raw, processed: plotted.append(
            (timestamp_us, np.asarray(raw).copy(), np.asarray(processed).copy())
        )
    )
    tau_ext = _TauExt()
    tau_ext.config = SimpleNamespace(
        tau_ext_filter=SimpleNamespace(
            enabled=True,
            mode="moving_average",
            window=20,
            cutoff_hz=15.0,
        ),
    )
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=False,
        pipeline=_Pipeline(),
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=tau_ext,
        wrench_estimator=_Wrench(),
        wrench_plotter=plotter,
    )

    dp_wrench, plotted_wrench = runtime._process_stream_wrench(
        np.full(6, 20.0),
        np.full(6, 10.0),
        123_000,
    )

    np.testing.assert_allclose(dp_wrench, 10.0)
    np.testing.assert_allclose(plotted_wrench, 10.0)
    assert plotted[0][0] == 123_000
    np.testing.assert_allclose(plotted[0][1], 20.0)
    np.testing.assert_allclose(plotted[0][2], 10.0)


def test_runtime_passes_independent_multicamera_timestamps() -> None:
    pipeline = _Pipeline()
    pipeline._image_keys = ("side", "wrist")
    runtime = NeroInferenceRuntime(
        _config(),
        backend="mock",
        command_enabled=False,
        pipeline=pipeline,
        arm=_Arm(),
        cameras=_TwoCameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)
    runtime.start()

    runtime.step()

    runtime.stop()
    assert pipeline.inputs[-1].image_timestamp_s == {
        "side": pytest.approx(0.026),
        "wrist": pytest.approx(0.040),
    }
    assert isinstance(pipeline.inputs[-1].image, dict)


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
        wrench_estimator=_Wrench([10.0, 10.0, 0.0, 0.0]),
    )
    _use_fast_reset(runtime)
    runtime.start()

    assert [runtime.step(), runtime.step()][-1] is not None
    runtime._restart_episode()
    restarted = [runtime.step(), runtime.step()]

    assert restarted[0] is None
    assert restarted[1] is not None
    assert len(pipeline.inputs) == 2
    np.testing.assert_allclose(pipeline.inputs[0].wrench_ext, np.full(6, 10.0))
    np.testing.assert_allclose(pipeline.inputs[-1].wrench_ext, np.zeros(6))
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


def test_runtime_uses_injected_modular_inference_without_legacy_input_adapter() -> None:
    class Modular:
        def __init__(self):
            self.sampler = object()
            self.controller = None
            self.calls = []

        def start(self):
            self.calls.append("start")

        def reset_episode(self):
            self.calls.append("reset")

        def step(self):
            self.calls.append("step")
            # The runtime sampler is attached during construction.  Consume it
            # here to ensure the modular path receives canonical observations.
            observation = self.sampler.sample()
            return observation

        def close(self):
            self.calls.append("close")

    base = _config()
    config = replace(
        base,
        architecture=ArchitectureConfig(
            enabled=True,
            policy_type="tavla",
            world_model_type="none",
        ),
    )
    modular = Modular()
    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=False,
        modular_inference=modular,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    runtime.start()
    result = runtime.step()
    runtime.stop()

    assert result is not None
    assert modular.sampler is runtime.observation_sampler
    assert modular.calls == ["start", "reset", "step", "close"]


def test_tavla_architecture_requires_explicit_model_builder() -> None:
    base = _config()
    config = replace(
        base,
        architecture=ArchitectureConfig(
            enabled=True,
            policy_type="tavla",
            world_model_type="none",
        ),
    )
    with pytest.raises(ValueError, match="modular_inference or modular_builder"):
        NeroInferenceRuntime(
            config,
            backend="mock",
            command_enabled=False,
            arm=_Arm(),
            cameras=_Cameras(),
            online_tau_ext=_TauExt(),
            wrench_estimator=_Wrench(),
        )


def test_runtime_modular_builder_receives_runtime_owned_sampler() -> None:
    class Modular:
        def __init__(self, sampler):
            self.sampler = sampler
            self.controller = None
            self.started = False

        def start(self):
            self.started = True

        def reset_episode(self):
            return None

        def step(self):
            return self.sampler.sample()

        def close(self):
            self.started = False

    base = _config()
    config = replace(
        base,
        architecture=ArchitectureConfig(
            enabled=True,
            policy_type="tavla",
            world_model_type="none",
        ),
    )
    seen = []

    def builder(runtime):
        seen.append(runtime)
        return Modular(runtime.observation_sampler)

    runtime = NeroInferenceRuntime(
        config,
        backend="mock",
        command_enabled=False,
        modular_builder=builder,
        arm=_Arm(),
        cameras=_Cameras(),
        online_tau_ext=_TauExt(),
        wrench_estimator=_Wrench(),
    )
    _use_fast_reset(runtime)

    assert seen == [runtime]
    runtime.start()
    assert runtime.step() is not None
    runtime.stop()


def test_runtime_uses_sdk_dq_and_internal_kalman_acceleration() -> None:
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

    np.testing.assert_allclose(pipeline.inputs[-1].dq, arm.dq)
    np.testing.assert_allclose(pipeline.inputs[-1].ddq, 0.0)
    assert not np.array_equal(pipeline.inputs[-1].ddq, arm.ddq)


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


def test_runtime_sends_contact_mit_effective_gains() -> None:
    arm, pipeline = _Arm(), _Pipeline()
    effective_kp = np.full(7, 0.25)
    effective_kd = np.full(7, 0.125)

    def contact_step(value):
        pipeline.inputs.append(value)
        return SimpleNamespace(
            tau_command=np.full(7, 0.5),
            control_mode="mit",
            joint_position_target=np.full(7, 0.1),
            joint_velocity_target=np.full(7, 0.2),
            torque_target=np.full(7, 0.3),
            mit_kp=effective_kp,
            mit_kd=effective_kd,
        )

    pipeline.step = contact_step
    base = _config()
    config = replace(
        base,
        predictor=replace(base.predictor, mode="contact_world_model_opd"),
        execution=replace(
            base.execution,
            mode="mit",
            mit_kp=(20.0,) * 7,
            mit_kd=(2.0,) * 7,
        ),
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

    assert len(arm.commands) >= 1
    np.testing.assert_allclose(arm.commands[0]["kp"], effective_kp)
    np.testing.assert_allclose(arm.commands[0]["kd"], effective_kd)
    np.testing.assert_allclose(arm.commands[0]["t_ff"], 0.3)


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


def test_runtime_s_executes_one_cycle_then_waits_for_next_request() -> None:
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
    # The None entry represents time spent paused after the first single step.
    keys = iter(("s", None, "s", "q"))

    cycles = runtime.run(read_key=lambda _timeout: next(keys))

    assert cycles == 2
    assert len(pipeline.inputs) == 2
    assert arm.enabled
    assert arm.disable_count == 0


def test_runtime_single_step_starts_paused() -> None:
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
    keys = iter((None, "s", "q"))

    cycles = runtime.run(read_key=lambda _timeout: next(keys), single_step=True)

    assert cycles == 1
    assert len(pipeline.inputs) == 1
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
