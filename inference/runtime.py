from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from inference.config import InferenceConfig
from inference.pipeline import InferenceInput, NeroInferencePipeline
from inference.state_stream import (
    ContinuousInferenceSample,
    ContinuousInferenceStateStream,
)
from inference.wrench_visualization import InferenceWrenchPlotter
from nero_collection.arms.factory import build_arm
from nero_collection.cameras import CameraFrame, CameraManager, CameraVisualizer
from nero_collection.config import CollectionConfig, load_config
from nero_collection.contact_wrench import PinocchioContactWrenchEstimator
from nero_collection.filters import OnePoleLowPass
from nero_collection.tau_ext_inference import OnlineTauExtInference
from nero_collection.time_utils import now_us


log = logging.getLogger(__name__)


class NeroInferenceRuntime:
    """Own the follower arm/camera and execute the complete online control chain."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        backend: str | None = None,
        command_enabled: bool = False,
        pipeline: NeroInferencePipeline | None = None,
        arm: Any | None = None,
        cameras: CameraManager | None = None,
        online_tau_ext: OnlineTauExtInference | None = None,
        wrench_estimator: PinocchioContactWrenchEstimator | None = None,
        wrench_plotter: InferenceWrenchPlotter | None = None,
        continuous_state_stream: bool | None = None,
    ) -> None:
        self.config = config
        collection = load_config(config.runtime.collection_config)
        if backend is not None:
            collection = _with_backend(collection, backend)
        self.collection = collection
        pair = next(
            (
                value
                for value in collection.teleop.master_slave
                if value.name == config.runtime.arm_pair
            ),
            None,
        )
        if pair is None:
            names = [value.name for value in collection.teleop.master_slave]
            raise ValueError(
                f"runtime.arm_pair={config.runtime.arm_pair!r} not found; available={names}"
            )
        self.pair = pair
        self.arm = arm or build_arm(pair.follower, collection.teleop.backend)
        self.cameras = cameras or CameraManager.from_config(
            collection.cameras,
            visualizer=CameraVisualizer.from_config(collection.cameras),
        )
        self.pipeline = pipeline or NeroInferencePipeline(config)
        self.online_tau_ext = online_tau_ext or OnlineTauExtInference(
            collection.tau_ext_inference,
            collection.realtime_plot.inverse_dynamics,
            collection.dynamics_processing,
            collection.robot_states,
        )
        self.wrench_estimator = wrench_estimator or PinocchioContactWrenchEstimator(
            collection.realtime_plot.wrench_mapping
        )
        self._dp_contact_threshold_n = _dp_contact_threshold_n(self.pipeline)
        self._dp_contact_force_dims, self._dp_contact_history_reducer = (
            _dp_contact_gate_settings(self.pipeline)
        )
        self._dp_contact_history_steps = max(
            int(getattr(self.pipeline, "wrench_history_steps", 1)),
            1,
        )
        self._dp_contact_history: deque[float] = deque(
            maxlen=self._dp_contact_history_steps
        )
        self.wrench_plotter = wrench_plotter or InferenceWrenchPlotter(
            config.wrench_visualization,
            contact_threshold_n=self._dp_contact_threshold_n,
        )
        self.command_enabled = bool(command_enabled)
        self._latest_frame: CameraFrame | None = None
        self._last_arm_timestamp_us = 0
        self._last_valid_arm_state_s: float | None = None
        self._last_invalid_state_warning_s = 0.0
        self._last_state_consumer_lag_warning_s = 0.0
        self._last_stale_camera_warning_s = 0.0
        self._started = False
        self._episode_index = 0
        protection = config.observation_protection
        online_tau_ext_config = getattr(self.online_tau_ext, "config", None)
        tau_ext_filter = getattr(online_tau_ext_config, "tau_ext_filter", None)
        self._tau_ext_post_filter_enabled = bool(
            getattr(tau_ext_filter, "enabled", False)
        )
        self._wrench_filter = OnePoleLowPass(
            protection.wrench_lowpass_cutoff_hz,
            protection.wrench_median_window,
        )
        if continuous_state_stream is None:
            continuous_state_stream = collection.teleop.backend != "mock"
        self._continuous_state_stream_enabled = bool(continuous_state_stream)
        self._last_consumed_state_timestamp_us = 0
        self._last_state_stream_rollover_count = 0
        self._q_cmd_lock = threading.Lock()
        self._q_cmd_history: deque[tuple[int, np.ndarray]] = deque(maxlen=4096)
        self._state_stream = ContinuousInferenceStateStream(
            self.arm,
            self.online_tau_ext,
            self.wrench_estimator,
            wrench_processor=self._process_stream_wrench,
            on_sample=self._on_stream_sample,
            q_cmd_provider=self._get_q_cmd,
            poll_interval_s=1.0 / collection.teleop.command.sample_rate_hz,
        )
        self._observation_warmup_started_us: int | None = None
        self._inference_control_mode_ready = False

    def _process_stream_wrench(
        self,
        raw_wrench: np.ndarray,
        wrench_from_filtered_tau: np.ndarray,
        timestamp_us: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter/gate each fixed-rate wrench sample before it enters DP history."""
        wrench = np.asarray(wrench_from_filtered_tau, dtype=np.float64).reshape(-1)
        if (
            self.config.observation_protection.enabled
            and not self._tau_ext_post_filter_enabled
        ):
            wrench = self._wrench_filter.apply(wrench, int(timestamp_us))
        processed_wrench = self._apply_dp_contact_gate_for_visualization(wrench)
        if self.config.wrench_visualization.enabled:
            self.wrench_plotter.append(
                int(timestamp_us),
                raw_wrench,
                processed_wrench,
            )
        return wrench, processed_wrench

    def _on_stream_sample(self, _sample: ContinuousInferenceSample) -> None:
        self._last_valid_arm_state_s = time.monotonic()

    def _start_state_stream(self) -> None:
        if not self._continuous_state_stream_enabled:
            return
        self._state_stream.clear()
        self._last_consumed_state_timestamp_us = 0
        self._last_state_stream_rollover_count = self._state_stream.history_rollover_count
        self._last_arm_timestamp_us = 0
        log.info(
            "continuous inference state stream started history=%d poll=%.1fms",
            self._state_stream.history_size,
            self._state_stream.poll_interval_s * 1.0e3,
        )
        self._state_stream.start()

    def _stop_state_stream(self, *, clear: bool = False) -> None:
        if not self._continuous_state_stream_enabled:
            return
        self._state_stream.stop()
        if clear:
            self._state_stream.clear()
        self._last_consumed_state_timestamp_us = 0
        self._last_state_stream_rollover_count = self._state_stream.history_rollover_count

    def start(self) -> None:
        if self._started:
            return
        self.arm.connect()
        try:
            self.arm.set_follower_mode()
            self.arm.enable()
            if self.command_enabled:
                self._reset_arm_to_rest("startup")
            else:
                self._wait_for_finite_arm_state()
            if not self.config.observation_protection.enabled:
                self._prepare_inference_control_mode()
                self._inference_control_mode_ready = True
            initial_state = self._wait_for_finite_arm_state()
            self._set_q_cmd(
                initial_state.q,
                timestamp_us=int(
                    getattr(initial_state, "q_timestamp_us", 0)
                    or getattr(initial_state, "timestamp_us", 0)
                    or now_us()
                ),
            )
            self.cameras.start()
            self.wrench_plotter.start()
            self.online_tau_ext.warm_up()
            self._reset_online_tau_ext_episode()
            self.pipeline.reset()
            self._reset_observation_protection()
            self._start_state_stream()
            self._started = True
        except BaseException:
            self._stop_state_stream(clear=True)
            self.cameras.stop()
            self.wrench_plotter.close()
            recovered = False
            if self.command_enabled:
                try:
                    self._reset_arm_to_rest("startup exception")
                    recovered = True
                    log.info(
                        "startup exception recovery complete; arm remains enabled "
                        "in follower mode"
                    )
                except Exception:
                    log.exception("arm reset failed after inference startup exception")
            if not recovered:
                self._best_effort_follower_enabled_hold(
                    allow_position_hold=self.command_enabled,
                )
            self.arm.disconnect()
            try:
                self.pipeline.close()
            except Exception as exc:
                log.warning("pipeline close failed after startup exception: %s", exc)
            raise

    def run(
        self,
        duration_s: float | None = None,
        *,
        read_key: Callable[[float], str | None] | None = None,
    ) -> int:
        if not self._started:
            self.start()
        started_s = time.perf_counter()
        total_cycles = 0
        episode_steps = 0
        self._episode_index = 1
        final_reset_succeeded = False
        failed = False
        log.info(
            "inference episode %d started; maximum_steps=%s",
            self._episode_index,
            self.config.runtime.maximum_inference_steps,
        )
        try:
            while duration_s is None or time.perf_counter() - started_s < duration_s:
                key = read_key(0.0) if read_key is not None else None
                if key in {"q", "Q", "\x03"}:
                    log.info("inference quit requested by keyboard")
                    break
                if key in {"i", "I"}:
                    log.info(
                        "ending inference episode %d at step %d: keyboard i",
                        self._episode_index,
                        episode_steps,
                    )
                    self._restart_episode()
                    self._episode_index += 1
                    episode_steps = 0
                    log.info("inference episode %d started", self._episode_index)
                    continue
                output = self.step()
                if output is None:
                    time.sleep(0.0005)
                    continue
                total_cycles += 1
                episode_steps += 1
                maximum_steps = self.config.runtime.maximum_inference_steps
                if maximum_steps is not None and episode_steps >= maximum_steps:
                    log.info(
                        "ending inference episode %d: reached maximum_steps=%d",
                        self._episode_index,
                        maximum_steps,
                    )
                    self._restart_episode()
                    self._episode_index += 1
                    episode_steps = 0
                    log.info("inference episode %d started", self._episode_index)
        except KeyboardInterrupt:
            log.info("inference interrupted by Ctrl-C")
        except BaseException as exc:
            failed = True
            log.error(
                "inference exception detected; starting immediate arm reset: %s",
                exc,
            )
            raise
        finally:
            # Do not feed reset/episode-shutdown motion into the recurrent
            # estimator or the wrench visualization.
            self._stop_state_stream(clear=True)
            if self._started and self.command_enabled:
                try:
                    reason = "exception recovery" if failed else "shutdown"
                    self._reset_arm_to_rest(reason)
                    final_reset_succeeded = True
                    if failed:
                        log.info(
                            "exception recovery reset complete; arm remains enabled "
                            "in follower mode"
                        )
                    else:
                        log.info(
                            "final reset complete; arm remains enabled in follower mode"
                        )
                except Exception:
                    log.exception(
                        "final reset failed; preserving follower enabled state "
                        "without disabling"
                    )
                    self._best_effort_follower_enabled_hold(
                        allow_position_hold=True,
                    )
            if self._started:
                try:
                    self._end_episode_state()
                except Exception:
                    log.exception("failed to clear inference episode state during shutdown")
            self.stop(
                preserve_arm_enabled=(
                    final_reset_succeeded or self.command_enabled or failed
                ),
            )
        return total_cycles

    def _restart_episode(self) -> None:
        self._end_episode_state()
        if self.command_enabled:
            self._reset_arm_to_rest("episode boundary")
        if not self.config.observation_protection.enabled:
            self._prepare_inference_control_mode()
            self._inference_control_mode_ready = True
        self._start_state_stream()

    def _end_episode_state(self) -> None:
        self._stop_state_stream(clear=True)
        self.pipeline.reset()
        self._reset_online_tau_ext_episode()
        self.wrench_plotter.clear_history()
        self._latest_frame = None
        self._reset_observation_protection()

    def _reset_observation_protection(self) -> None:
        self._wrench_filter.reset()
        self._dp_contact_history.clear()
        self._observation_warmup_started_us = None
        self._inference_control_mode_ready = not (
            self.command_enabled
            and self.config.predictor.enabled
            and self.config.observation_protection.enabled
        )

    def _observation_is_ready(self, timestamp_us: int) -> bool:
        protection = self.config.observation_protection
        if not protection.enabled:
            return True
        if self._observation_warmup_started_us is None:
            self._observation_warmup_started_us = int(timestamp_us)
            log.info(
                "observation protection settling for %.3fs; policy inference is held",
                protection.warmup_duration_s,
            )
        elapsed_s = (
            int(timestamp_us) - self._observation_warmup_started_us
        ) * 1.0e-6
        if elapsed_s < protection.warmup_duration_s:
            return False
        if not self._inference_control_mode_ready:
            self._prepare_inference_control_mode()
            self._inference_control_mode_ready = True
            log.info(
                "observation protection settled after %.3fs; inference control mode ready",
                elapsed_s,
            )
            # The mode transition consumes a state read. Start inference from
            # the next complete control sample.
            return False
        return True

    def _reset_online_tau_ext_episode(self) -> None:
        self.online_tau_ext.reset_episode()

    def _prepare_inference_control_mode(self) -> None:
        if not self.command_enabled or not self.config.predictor.enabled:
            return
        self.arm.validate_joint_impedance_support()
        self.arm.configure_joint_impedance_mode()
        # A firmware motion-mode transition can briefly expose incomplete SDK
        # cache values. Do not begin control until one finite snapshot arrives.
        self._wait_for_finite_arm_state()

    def _best_effort_follower_enabled_hold(
        self,
        *,
        allow_position_hold: bool,
    ) -> None:
        """Never disable after a recovery failure; hold current q when possible."""
        try:
            self.arm.set_follower_mode()
        except Exception as exc:
            log.error("failed to restore follower mode during emergency hold: %s", exc)
        try:
            self.arm.enable()
        except Exception as exc:
            log.error("failed to confirm arm enable during emergency hold: %s", exc)
        if allow_position_hold:
            try:
                q = np.asarray(self.arm.read_state().q, dtype=np.float64).reshape(-1)
                if q.shape != (7,) or not np.all(np.isfinite(q)):
                    raise RuntimeError(f"invalid emergency hold joints: {q}")
                self.arm.move_joints(q)
                log.warning(
                    "full reset unavailable; commanding current-q position hold=%s",
                    np.array2string(q, precision=5, suppress_small=True),
                )
            except Exception as exc:
                log.error("failed to command emergency current-q hold: %s", exc)
        try:
            self.arm.set_follower_mode()
            self.arm.enable()
        except Exception as exc:
            log.error(
                "failed final follower/enable confirmation during emergency hold: %s",
                exc,
            )

    def _reset_arm_to_rest(self, reason: str) -> None:
        rest_q = np.asarray(self.pair.follower.rest_q, dtype=np.float64).reshape(-1)
        if rest_q.shape != (7,) or not np.all(np.isfinite(rest_q)):
            raise RuntimeError(
                "runtime reset requires a finite seven-joint follower.rest_q in "
                f"{self.config.runtime.collection_config}; got {rest_q}"
            )
        command = self.collection.teleop.command
        log.info(
            "resetting inference arm to follower.rest_q reason=%s target=%s",
            reason,
            np.array2string(rest_q, precision=5, suppress_small=True),
        )
        self.arm.set_follower_mode()
        self.arm.enable()
        self._wait_for_finite_arm_state()
        reset_target = rest_q.copy()
        self._move_arm_to_reset_target(reset_target)
        deadline = time.monotonic() + command.reset_timeout_s
        while True:
            if command.reset_wait_s > 0:
                time.sleep(command.reset_wait_s)
            sample_count = max(int(command.reset_test_sample_time), 1)
            samples = []
            sample_period_s = 1.0 / max(float(command.idle_rate_hz), 1.0)
            for sample_index in range(sample_count):
                # Position/motion-mode transitions can briefly expose incomplete
                # SDK cache values. Retry until one finite control snapshot arrives.
                state = self._wait_for_finite_arm_state()
                q = np.asarray(state.q, dtype=np.float64).reshape(-1)
                samples.append(q)
                if sample_index + 1 < sample_count:
                    time.sleep(sample_period_s)
            error = rest_q - np.mean(samples, axis=0)
            maximum_error = float(np.max(np.abs(error)))
            log.info(
                "inference reset check from %d averaged samples: max joint error %.6f rad",
                sample_count,
                maximum_error,
            )
            if maximum_error <= command.reset_error_limit_rad:
                self.arm.set_follower_mode()
                self.arm.enable()
                final_state = self._wait_for_finite_arm_state()
                self._set_q_cmd(
                    final_state.q,
                    timestamp_us=int(
                        getattr(final_state, "q_timestamp_us", 0)
                        or getattr(final_state, "timestamp_us", 0)
                        or now_us()
                    ),
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Inference arm reset self-check failed: max joint error "
                    f"{maximum_error:.6f} rad > limit "
                    f"{command.reset_error_limit_rad:.6f} rad"
                )
            reset_target = reset_target + _limit_reset_correction(
                error,
                command.joint_step_limit_rad,
            )
            log.warning(
                "reset error exceeds limit; fine-tuning target=%s",
                np.array2string(reset_target, precision=6, suppress_small=True),
            )
            self._move_arm_to_reset_target(reset_target)

    def _wait_for_finite_arm_state(self):
        timeout_s = self.collection.teleop.command.input_ready_timeout_s
        deadline_s = time.monotonic() + timeout_s
        last_state = None
        while time.monotonic() < deadline_s:
            last_state = self.arm.read_state()
            vectors = (last_state.q, last_state.dq, last_state.torque)
            if all(
                np.asarray(value, dtype=np.float64).shape == (7,)
                and np.all(np.isfinite(value))
                for value in vectors
            ):
                self._last_valid_arm_state_s = time.monotonic()
                return last_state
            time.sleep(0.002)
        raise RuntimeError(
            "timed out waiting for finite q/dq/tau after mode reset; "
            f"last_state={last_state}"
        )

    def _move_arm_to_reset_target(self, target: np.ndarray) -> None:
        command = self.collection.teleop.command
        start = np.asarray(self.arm.read_state().q, dtype=np.float64).reshape(-1)
        if start.shape != (7,) or not np.all(np.isfinite(start)):
            raise RuntimeError(f"invalid reset start joints: {start}")
        maximum_delta = float(np.max(np.abs(target - start)))
        if not command.reset_interpolation_enabled or maximum_delta < 1.0e-9:
            steps = 1
        else:
            duration_s = max(
                command.reset_min_duration_s,
                maximum_delta / command.reset_joint_speed_rad_s,
            )
            steps = max(
                1,
                math.ceil(duration_s * command.reset_interpolation_rate_hz),
                math.ceil(maximum_delta / command.reset_max_step_rad),
            )
        log.info(
            "inference reset interpolation steps=%d max_delta=%.4f rad",
            steps,
            maximum_delta,
        )
        interpolation_started_s = time.monotonic()
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            self.arm.move_joints(start + alpha * (target - start))
            if steps > 1:
                deadline_s = (
                    interpolation_started_s
                    + step_index / command.reset_interpolation_rate_hz
                )
                remaining_s = deadline_s - time.monotonic()
                if remaining_s > 0:
                    time.sleep(remaining_s)
        if not self.arm.wait_motion_done(command.reset_timeout_s):
            raise RuntimeError("timed out waiting for inference arm reset motion")

    def _read_inference_sample(self) -> ContinuousInferenceSample | None:
        """Return the next canonical state sample without blocking on DP work."""
        if self._continuous_state_stream_enabled:
            fault = self._state_stream.fault
            if fault is not None:
                raise RuntimeError("continuous inference state stream failed") from fault
            sample = self._state_stream.latest()
            if sample is None or sample.timestamp_us <= self._last_arm_timestamp_us:
                return None
        else:
            state = self.arm.read_state()
            sample = self._state_stream.process_state(state)
            if sample is None:
                monotonic_s = time.monotonic()
                last_valid_s = self._last_valid_arm_state_s
                invalid_duration_s = (
                    0.0 if last_valid_s is None else monotonic_s - last_valid_s
                )
                if invalid_duration_s <= self.config.runtime.maximum_state_age_s:
                    return None
                raise RuntimeError(
                    "arm SDK cache remained incomplete for "
                    f"{invalid_duration_s:.3f}s"
                )

        current_us = now_us()
        age_s = max(
            0.0,
            (current_us - int(sample.acquired_timestamp_us)) * 1.0e-6,
        )
        if age_s > self.config.runtime.maximum_state_age_s:
            hardware_age_s = None
            peek_latest = getattr(self.arm, "peek_latest_state", None)
            if callable(peek_latest):
                hardware_state = peek_latest()
                hardware_acquired_us = int(
                    getattr(hardware_state, "acquired_timestamp_us", 0)
                )
                if hardware_acquired_us > 0:
                    hardware_age_s = max(
                        0.0,
                        (current_us - hardware_acquired_us) * 1.0e-6,
                    )
            maximum_age_s = self.config.runtime.maximum_state_age_s
            if hardware_age_s is not None and hardware_age_s <= maximum_age_s:
                monotonic_s = time.monotonic()
                if monotonic_s - self._last_state_consumer_lag_warning_s >= 0.5:
                    batch_size, batch_processing_s = (
                        self._state_stream.processing_status()
                    )
                    log.warning(
                        "inference state consumer is behind fresh hardware data: "
                        "canonical_age=%.3fs hardware_age=%.3fs "
                        "last_batch=%d batch_processing=%.3fs",
                        age_s,
                        hardware_age_s,
                        batch_size,
                        batch_processing_s,
                    )
                    self._last_state_consumer_lag_warning_s = monotonic_s
                minimum_acquired_us = current_us - int(
                    round(maximum_age_s * 1.0e6)
                )
                sample = self._state_stream.wait_for_acquired_after(
                    minimum_acquired_us,
                    maximum_age_s,
                )
                fault = self._state_stream.fault
                if fault is not None:
                    raise RuntimeError(
                        "continuous inference state stream failed"
                    ) from fault
                if sample is None:
                    return None
                current_us = now_us()
                age_s = max(
                    0.0,
                    (current_us - int(sample.acquired_timestamp_us)) * 1.0e-6,
                )
            if age_s <= maximum_age_s:
                self._last_valid_arm_state_s = time.monotonic()
                self._last_arm_timestamp_us = sample.timestamp_us
                return sample
            batch_size, batch_processing_s = self._state_stream.processing_status()
            hardware_detail = (
                "unavailable"
                if hardware_age_s is None
                else f"{hardware_age_s:.3f}s"
            )
            raise RuntimeError(
                f"follower state is stale: age={age_s:.3f}s, "
                f"limit={maximum_age_s:.3f}s, hardware_age={hardware_detail}, "
                f"last_batch={batch_size}, batch_processing={batch_processing_s:.3f}s"
            )
        self._last_valid_arm_state_s = time.monotonic()
        self._last_arm_timestamp_us = sample.timestamp_us
        return sample

    def _drain_state_observations(self) -> None:
        """Backfill fixed-rate state records produced while the main loop was busy."""
        if not self._continuous_state_stream_enabled:
            return
        records = self._state_stream.drain_after(
            self._last_consumed_state_timestamp_us
        )
        if not records:
            return
        rollover_count = self._state_stream.history_rollover_count
        if rollover_count > self._last_state_stream_rollover_count:
            log.debug(
                "continuous state ring rolled over evictions=%d history=%d",
                rollover_count - self._last_state_stream_rollover_count,
                self._state_stream.history_size,
            )
            self._last_state_stream_rollover_count = rollover_count
        for record in records:
            self.pipeline.append_continuous_can_observation(
                q=record.q,
                dq=record.dq,
                ddq=record.ddq,
                tau=record.tau,
                wrench=record.wrench,
                timestamp_s=record.timestamp_us * 1.0e-6,
            )
        self._last_consumed_state_timestamp_us = records[-1].timestamp_us

    def step(self):
        if not self._started:
            raise RuntimeError("runtime must be started before step()")
        open_loop = self.config.predictor.inference_mode == "open_loop"
        executing = bool(
            getattr(self.pipeline, "open_loop_execution_active", False)
        )
        for frame in self.cameras.poll():
            if frame.camera_name == self.config.runtime.camera:
                self._latest_frame = frame
        if self._latest_frame is None:
            return None

        current_us = now_us()
        camera_age_s = max(
            0.0,
            (current_us - int(self._latest_frame.timestamp_us)) * 1.0e-6,
        )
        camera_is_stale = camera_age_s > self.config.runtime.maximum_state_age_s
        if camera_is_stale:
            if open_loop and not executing:
                monotonic_s = time.monotonic()
                if monotonic_s - self._last_stale_camera_warning_s >= 0.5:
                    log.warning(
                        "waiting for a fresh camera frame before the next open-loop "
                        "observation batch camera=%s age=%.3fs limit=%.3fs",
                        self.config.runtime.camera,
                        camera_age_s,
                        self.config.runtime.maximum_state_age_s,
                    )
                    self._last_stale_camera_warning_s = monotonic_s
            if not open_loop:
                raise RuntimeError(
                    f"camera {self.config.runtime.camera!r} is stale: "
                    f"age={camera_age_s:.3f}s, "
                    f"limit={self.config.runtime.maximum_state_age_s:.3f}s"
                )
        sample = self._read_inference_sample()
        if sample is None:
            return None
        self._drain_state_observations()
        if not self._observation_is_ready(sample.timestamp_us):
            return None
        if open_loop and not executing and camera_is_stale:
            return None

        rotation = None
        if self.collection.realtime_plot.wrench_mapping.reference_frame == "local":
            model = getattr(self.pipeline, "model", self.pipeline.controller.model)
            pose = model.snapshot(
                sample.q, sample.dq
            ).pose
            rotation = pose[:3, :3].copy()

        output = self.pipeline.step(
            InferenceInput(
                q=sample.q,
                dq=sample.dq,
                ddq=sample.ddq,
                tau=sample.tau,
                image=self._latest_frame.frame,
                wrench_ext=sample.wrench,
                timestamp_s=sample.timestamp_us * 1.0e-6,
                wrench_to_control_rotation=rotation,
                image_timestamp_s=self._latest_frame.timestamp_us * 1.0e-6,
            )
        )
        if self.command_enabled:
            if self.config.predictor.enabled:
                q_cmd = np.asarray(sample.q, dtype=np.float64)
                self.arm.command_joint_impedance(
                    q=q_cmd,
                    v_des=np.zeros(7, dtype=np.float64),
                    kp=np.zeros(7, dtype=np.float64),
                    kd=np.asarray(self.config.runtime.command_kd, dtype=np.float64),
                    t_ff=output.tau_command,
                )
            else:
                if output.joint_position_command is None:
                    raise RuntimeError(
                        "direct IK pipeline did not produce a joint-position command"
                    )
                q_cmd = np.asarray(output.joint_position_command, dtype=np.float64)
                self.arm.command_joint_positions(q_cmd)
            self._set_q_cmd(q_cmd)
        return output

    def _get_q_cmd(self, timestamp_us: int) -> np.ndarray | None:
        with self._q_cmd_lock:
            if not self._q_cmd_history:
                return None
            for command_timestamp_us, q_cmd in reversed(self._q_cmd_history):
                if command_timestamp_us <= int(timestamp_us):
                    return q_cmd.copy()
            return self._q_cmd_history[0][1].copy()

    def _set_q_cmd(
        self,
        q_cmd: np.ndarray,
        *,
        timestamp_us: int | None = None,
    ) -> None:
        value = np.asarray(q_cmd, dtype=np.float64).reshape(-1)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError(f"q_cmd must be a finite seven-joint vector; got {value}")
        with self._q_cmd_lock:
            self._q_cmd_history.append(
                (int(now_us() if timestamp_us is None else timestamp_us), value.copy())
            )

    def _apply_dp_contact_gate_for_visualization(
        self,
        wrench: np.ndarray,
    ) -> np.ndarray:
        """Mirror the DP contact mask for the processed-observation plot.

        The actual DP model still applies its gate after normalizing the raw
        physical wrench. This display copy shows the corresponding physical
        wrench with the same history reducer and threshold.
        """
        value = np.asarray(wrench, dtype=np.float64).reshape(-1)
        if self._dp_contact_threshold_n is None:
            return value.copy()
        force = value[list(self._dp_contact_force_dims)]
        magnitude = float(np.linalg.norm(force))
        self._dp_contact_history.append(magnitude)
        history = np.asarray(self._dp_contact_history, dtype=np.float64)
        if self._dp_contact_history_reducer == "last":
            reduced = float(history[-1])
        elif self._dp_contact_history_reducer == "max":
            reduced = float(np.max(history))
        else:
            reduced = float(np.mean(history))
        if reduced <= self._dp_contact_threshold_n:
            return np.zeros(6, dtype=np.float64)
        return value.copy()

    def stop(self, *, preserve_arm_enabled: bool = False) -> None:
        if not self._started:
            self.pipeline.close()
            return
        try:
            if (
                not preserve_arm_enabled
                and self.command_enabled
                and self.config.predictor.enabled
            ):
                try:
                    state = self.arm.read_state()
                    self.arm.command_joint_impedance(
                        q=state.q,
                        v_des=np.zeros(7),
                        kp=np.zeros(7),
                        kd=np.asarray(self.config.runtime.command_kd),
                        t_ff=np.zeros(7),
                    )
                except Exception as exc:
                    log.warning("failed to send zero-torque shutdown command: %s", exc)
        finally:
            self._stop_state_stream(clear=True)
            self.cameras.stop()
            self.wrench_plotter.close()
            if not preserve_arm_enabled:
                try:
                    self.arm.disable()
                except Exception as exc:
                    log.warning("arm disable failed during shutdown: %s", exc)
            self.arm.disconnect()
            self.pipeline.close()
            self._started = False


def _limit_reset_correction(error: np.ndarray, limit_rad: float | None) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    if limit_rad is None:
        return value
    return np.clip(value, -float(limit_rad), float(limit_rad))


def _rotate_wrench_to_control(
    wrench: np.ndarray,
    rotation: np.ndarray | None,
) -> np.ndarray:
    """Match the frame conversion used by the OSC-QP pipeline."""
    value = np.asarray(wrench, dtype=np.float64).reshape(-1)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"inference wrench must be a finite 6-vector; got {value}")
    if rotation is None:
        return value.copy()
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"wrench control-frame rotation must be a finite 3x3 matrix; got {matrix}"
        )
    result = value.copy()
    result[:3] = matrix @ value[:3]
    result[3:] = matrix @ value[3:]
    return result


def _dp_contact_threshold_n(pipeline: Any) -> float | None:
    """Read the physical-unit force gate restored from the DP checkpoint."""
    detector = getattr(getattr(pipeline, "dp", None), "contact_detector", None)
    threshold = getattr(detector, "threshold", None)
    if threshold is None:
        return None
    value = float(threshold)
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError(
            f"DP contact detector threshold must be non-negative and finite; got {threshold}"
        )
    return value


def _dp_contact_gate_settings(pipeline: Any) -> tuple[tuple[int, ...], str]:
    detector = getattr(getattr(pipeline, "dp", None), "contact_detector", None)
    raw_dims = getattr(detector, "force_dims", (0, 1, 2))
    dims = tuple(int(dim) for dim in raw_dims)
    if not dims or any(dim < 0 or dim >= 6 for dim in dims):
        raise RuntimeError(
            "DP contact detector force_dims must select wrench components; "
            f"got {dims}"
        )
    reducer = str(getattr(detector, "history_reducer", "mean")).lower()
    if reducer not in {"last", "max", "mean"}:
        raise RuntimeError(
            "DP contact detector history_reducer must be last, max, or mean; "
            f"got {reducer!r}"
        )
    return dims, reducer


def _with_backend(config: CollectionConfig, backend: str) -> CollectionConfig:
    normalized = backend.lower().replace("-", "_")
    cameras = config.cameras
    if normalized in {"mock", "sim", "simulation"}:
        cameras = tuple(replace(camera, backend="mock") for camera in cameras)
        normalized = "mock"
    return replace(
        config,
        teleop=replace(config.teleop, backend=normalized),
        cameras=cameras,
    )
