from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from inference.config import InferenceConfig, resolve_camera_key
from inference.pipeline import NeroInferencePipeline
from inference.state_stream import (
    ContinuousInferenceSample,
    ContinuousInferenceStateStream,
)
from inference.wrench_mapping import (
    PinocchioContactWrenchEstimator,
    WrenchMappingConfig,
)
from inference.wrench_visualization import InferenceWrenchPlotter
from inference.diagnostics.tau_ext import TauExtInferencePlotter
from inference.core.nero_sampler import NeroObservationSampler
from inference.core.base import InferenceBase
from inference.core.legacy_runner import ModularInferenceRunner, NeroPipelineRunner
from inference.control.nero import NeroPipelineOutputController
from nero_collection.arms.factory import build_arm
from nero_collection.cameras import CameraFrame, CameraManager, CameraVisualizer
from nero_collection.config import CollectionConfig, load_config
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
        modular_inference: InferenceBase | None = None,
        modular_builder: Callable[["NeroInferenceRuntime"], InferenceBase] | None = None,
        modular_image_keys: tuple[str, ...] | None = None,
        arm: Any | None = None,
        cameras: CameraManager | None = None,
        online_tau_ext: OnlineTauExtInference | None = None,
        wrench_estimator: PinocchioContactWrenchEstimator | None = None,
        wrench_plotter: InferenceWrenchPlotter | None = None,
        tau_ext_plotter: TauExtInferencePlotter | None = None,
        continuous_state_stream: bool | None = None,
    ) -> None:
        self.config = config
        # ``pipeline`` historically means the legacy Nero model-stage object.
        # Accepting an ``InferenceBase`` there as well keeps migration callers
        # source-compatible, while the explicit argument is clearer for new
        # code.  A builder is evaluated after the runtime sampler is created so
        # official-model code can inject that exact sampler into TAVLA.
        if isinstance(pipeline, InferenceBase):
            if modular_inference is not None:
                raise ValueError(
                    "pass a modular inference object either as pipeline or "
                    "modular_inference, not both"
                )
            modular_inference = pipeline
            pipeline = None
        if modular_inference is not None and modular_builder is not None:
            raise ValueError("pass only one of modular_inference/modular_builder")
        self.modular_inference = modular_inference
        self._modular_builder = modular_builder
        self._modular_image_keys = (
            None
            if modular_image_keys is None
            else tuple(str(key) for key in modular_image_keys)
        )
        if self._modular_image_keys is not None and not self._modular_image_keys:
            raise ValueError("modular_image_keys must contain at least one camera key")
        self._modular_mode = modular_inference is not None or modular_builder is not None
        requested_policy = str(config.architecture.policy_type).strip().lower()
        if (
            config.architecture.enabled
            and requested_policy in {"tavla", "tavla_inference", "tavla_pipeline"}
            and not self._modular_mode
        ):
            raise ValueError(
                "architecture.policy_type='tavla' requires an injected "
                "modular_inference or modular_builder; the official TAVLA "
                "checkpoint loader is repository-specific"
            )
        if config.architecture.enabled:
            log.info(
                "modular inference architecture enabled policy=%s world_model=%s; "
                "legacy pipeline remains the model-stage compatibility adapter",
                config.architecture.policy_type,
                config.architecture.world_model_type,
            )
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
        if self._modular_mode:
            # The actual modular object may be produced below, once the shared
            # observation sampler exists.  Keep ``self.pipeline`` as a public
            # compatibility alias after construction.
            self.pipeline = modular_inference
        elif pipeline is not None:
            self.pipeline = pipeline
        elif config.predictor.enabled and config.predictor.mode in {
            "contact_world_model",
            "contact_world_model_opd",
            "contact_wm",
            "contact_wm_opd",
        }:
            from inference.contact_pipeline import ContactWMInferencePipeline

            self.pipeline = ContactWMInferencePipeline(config)
        elif config.predictor.enabled and config.predictor.mode in {
            "swm",
            "swm_opd",
            "torque_world_model",
            "torque_world_model_opd",
            "torque_wm",
            "torque_wm_opd",
        }:
            from inference.swm_pipeline import SWMInferencePipeline

            self.pipeline = SWMInferencePipeline(config)
        else:
            self.pipeline = NeroInferencePipeline(config)
        self.online_tau_ext = online_tau_ext or OnlineTauExtInference(
            collection.tau_ext_inference,
            collection.tau_ext_inference.inverse_dynamics,
            collection.dynamics_processing,
            collection.robot_states,
            source_sample_rate_hz=collection.teleop.command.sample_rate_hz,
        )
        self._wrench_mapping = WrenchMappingConfig(
            urdf_path=config.robot.urdf_path,
            frame_name=config.robot.frame_name,
            delay_s=0.0,
            locked_joint_names=config.robot.locked_joint_names,
            gravity_m_s2=collection.tau_ext_inference.inverse_dynamics.gravity_m_s2,
        )
        self.wrench_estimator = wrench_estimator or PinocchioContactWrenchEstimator(
            self._wrench_mapping
        )
        self._dp_contact_threshold_n = _dp_contact_threshold_n(self.pipeline)
        self._dp_contact_force_dims, _ = _dp_contact_gate_settings(self.pipeline)
        self.wrench_plotter = wrench_plotter or InferenceWrenchPlotter(
            config.wrench_visualization,
            contact_threshold_n=self._dp_contact_threshold_n,
        )
        self.tau_ext_plotter = tau_ext_plotter or TauExtInferencePlotter(collection)
        self.command_enabled = bool(command_enabled)
        self.robot_controller = NeroPipelineOutputController(
            arm=self.arm,
            config=self.config,
            command_enabled=self.command_enabled,
        )
        self.robot_controller.bind_q_command_sink(self._set_q_cmd)
        self._latest_frame: CameraFrame | None = None
        configured_image_keys = tuple(getattr(self.pipeline, "_image_keys", ()))
        if self._modular_image_keys is not None:
            configured_image_keys = self._modular_image_keys
        elif self._modular_mode:
            # TAVLA/other modular policies usually consume the complete image
            # mapping.  Keep the historical single-camera default unless the
            # caller explicitly opts into additional cameras.
            configured_image_keys = tuple(
                getattr(
                    getattr(self.modular_inference, "policy", None),
                    "image_keys",
                    (),
                )
            ) or configured_image_keys
        collection_camera_keys = tuple(
            str(camera.name) for camera in collection.cameras if camera.enabled
        )
        # The policy/checkpoint defines the image keys.  If an injected legacy
        # pipeline does not expose them, use the enabled acquisition keys as a
        # compatibility fallback and still resolve one deterministic anchor.
        resolution_keys = configured_image_keys or collection_camera_keys
        if not resolution_keys and config.runtime.camera is not None:
            resolution_keys = (str(config.runtime.camera),)
        if not resolution_keys:
            raise ValueError(
                "cannot resolve a camera anchor: neither the policy checkpoint "
                "nor the collection config declares an enabled camera"
            )
        resolved_camera = resolve_camera_key(
            config.runtime.camera,
            resolution_keys,
            collection_camera_keys,
        )
        self.resolved_camera = resolved_camera
        self._pipeline_image_keys = tuple(
            str(key) for key in (configured_image_keys or (resolved_camera,))
        )
        log.info(
            "camera contract resolved anchor=%s policy_keys=%s collection_keys=%s",
            resolved_camera,
            list(self._pipeline_image_keys),
            list(collection_camera_keys),
        )
        self._latest_frames: dict[str, CameraFrame] = {}
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
        if self._tau_ext_post_filter_enabled:
            log.info(
                "DP wrench observation uses collection tau_ext_filter "
                "mode=%s window=%s cutoff_hz=%s; no second inference low-pass",
                getattr(tau_ext_filter, "mode", "unknown"),
                getattr(tau_ext_filter, "window", "unknown"),
                getattr(tau_ext_filter, "cutoff_hz", "unknown"),
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
        self._observation_sampler = NeroObservationSampler(
            cameras=self.cameras,
            camera_keys=self._pipeline_image_keys,
            primary_camera=resolved_camera,
            maximum_state_age_s=config.runtime.maximum_state_age_s,
            read_state=self._read_inference_sample,
            drain_state=self._drain_state_observations,
            observation_ready=self._observation_is_ready,
            open_loop=lambda: self.config.predictor.inference_mode == "open_loop",
            open_loop_active=lambda: bool(
                getattr(self.pipeline, "open_loop_execution_active", False)
            ),
            wrench_rotation=self._wrench_rotation_for_sample,
        )
        # Public component handles make the runtime usable as a compatibility
        # container while callers migrate to ``InferenceBase`` directly.
        self.observation_sampler = self._observation_sampler
        self.controller = self.robot_controller
        self.diagnostics = (self.tau_ext_plotter, self.wrench_plotter)

        if self._modular_builder is not None:
            built = self._modular_builder(self)
            if built is None or not callable(getattr(built, "step", None)):
                raise TypeError(
                    "modular_builder must return an inference object exposing step()"
                )
            self.modular_inference = built
            self.pipeline = built

        if self._modular_mode:
            if self.modular_inference is None:
                raise RuntimeError("modular inference was not constructed")
            # Reuse the runtime-owned sampler.  A builder may have created the
            # policy before runtime construction with a placeholder sampler;
            # replacing it here prevents duplicate camera/state consumers.
            if hasattr(self.modular_inference, "sampler"):
                self.modular_inference.sampler = self._observation_sampler
            modular_controller = getattr(self.modular_inference, "controller", None)
            if modular_controller is not None:
                self.controller = modular_controller
            self.pipeline_runner = ModularInferenceRunner(self.modular_inference)
        else:
            self.pipeline_runner = NeroPipelineRunner(
                pipeline=self.pipeline,
                sampler=self._observation_sampler,
                controller=self.robot_controller,
                image_keys=self._pipeline_image_keys,
                on_observation=self._on_modular_observation,
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

    def _wrench_rotation_for_sample(
        self,
        sample: ContinuousInferenceSample,
    ) -> np.ndarray | None:
        """Return the wrench-to-control rotation for the modular sampler."""
        if self._wrench_mapping.reference_frame != "local":
            return None
        # Do not use ``getattr(obj, name, obj.other)`` here: Python evaluates
        # the default expression before calling ``getattr``.  A modular
        # pipeline may expose ``model`` without exposing the legacy
        # ``controller`` attribute, so resolving the fallback eagerly would
        # raise an unrelated AttributeError.
        model = getattr(self.pipeline, "model", None)
        if model is None and self._modular_mode:
            policy = getattr(self.modular_inference, "policy", None)
            model = getattr(policy, "model", None)
        if model is None:
            legacy_controller = getattr(self.pipeline, "controller", None)
            model = getattr(legacy_controller, "model", None)
        snapshot = getattr(model, "snapshot", None)
        if not callable(snapshot):
            # A modular policy is allowed to operate without a dynamics model;
            # in that case its wrench is already expressed in control frame.
            if self._modular_mode:
                return None
            raise RuntimeError(
                "pipeline must expose a dynamics model for local wrench rotation"
            )
        pose = snapshot(sample.q, sample.dq).pose
        return np.asarray(pose[:3, :3], dtype=np.float64).copy()

    def _on_modular_observation(self, _observation) -> None:
        self._latest_frame = self._observation_sampler.latest_frame
        self._latest_frames = self._observation_sampler.latest_frames

    def _on_stream_sample(self, sample: ContinuousInferenceSample) -> None:
        self._last_valid_arm_state_s = time.monotonic()
        if bool(getattr(self.tau_ext_plotter, "enabled", False)):
            try:
                self.tau_ext_plotter.append(
                    sample.timestamp_us,
                    sample.tau_result.tau_ext_cal,
                    sample.tau_result.tau_ext_pred,
                )
            except Exception:
                # Diagnostics must never stop the fixed-rate state stream.
                log.warning("tau_ext inference plot update failed", exc_info=True)

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
            self.tau_ext_plotter.start()
            self.online_tau_ext.warm_up()
            self._reset_online_tau_ext_episode()
            if self._modular_mode:
                # InferenceBase owns policy/controller lifecycle and requires
                # ``start`` before its episode reset.  The shared sampler is
                # already attached during construction.
                self.pipeline_runner.start()
                self.pipeline_runner.reset_episode()
            else:
                self.pipeline.reset()
                self._observation_sampler.reset_episode()
            self._reset_observation_protection()
            self._start_state_stream()
            self._started = True
        except BaseException:
            self._stop_state_stream(clear=True)
            self.cameras.stop()
            self.wrench_plotter.close()
            self.tau_ext_plotter.close()
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
                self._close_model_pipeline()
            except Exception as exc:
                log.warning("pipeline close failed after startup exception: %s", exc)
            raise

    def run(
        self,
        duration_s: float | None = None,
        *,
        read_key: Callable[[float], str | None] | None = None,
        single_step: bool = False,
    ) -> int:
        if not self._started:
            self.start()
        started_s = time.perf_counter()
        total_cycles = 0
        episode_steps = 0
        self._episode_index = 1
        # In single-step mode the key request remains armed until a complete
        # state/image sample is available. This makes one ``s`` correspond to
        # exactly one pipeline control cycle even when cameras or CAN lag.
        manual_step = bool(single_step)
        step_requested = False
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
                if key in {"s", "S"}:
                    manual_step = True
                    step_requested = True
                    log.info(
                        "single-step requested; press s for the next control cycle "
                        "or c to resume continuous inference"
                    )
                elif key in {"c", "C"}:
                    manual_step = False
                    step_requested = False
                    log.info("continuous inference resumed")
                if manual_step and not step_requested:
                    # TerminalKeys is non-blocking; avoid spinning while the
                    # operator is inspecting the last command/visualization.
                    time.sleep(0.005)
                    continue
                output = self.step()
                if output is None:
                    time.sleep(0.0005 if not manual_step else 0.005)
                    continue
                total_cycles += 1
                episode_steps += 1
                if manual_step:
                    step_requested = False
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
        # SWM's delta_q is relative to the command held in this episode.  Do
        # not carry a prior episode's q_cmd across the reset; seed the new
        # history from the post-reset measured state before restarting the
        # fixed-rate stream.
        with self._q_cmd_lock:
            self._q_cmd_history.clear()
        if not self.config.observation_protection.enabled:
            self._prepare_inference_control_mode()
            self._inference_control_mode_ready = True
        state = self._wait_for_finite_arm_state()
        self._set_q_cmd(state.q, timestamp_us=now_us())
        self._start_state_stream()

    def _end_episode_state(self) -> None:
        self._stop_state_stream(clear=True)
        with self._q_cmd_lock:
            self._q_cmd_history.clear()
        if self._modular_mode:
            self.pipeline_runner.reset_episode()
        else:
            self.pipeline.reset()
            self._observation_sampler.reset_episode()
        self._reset_online_tau_ext_episode()
        self.wrench_plotter.clear_history()
        self.tau_ext_plotter.clear_history()
        self._latest_frame = None
        self._latest_frames.clear()
        self._reset_observation_protection()

    def _reset_observation_protection(self) -> None:
        self._wrench_filter.reset()
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

    def _close_model_pipeline(self) -> None:
        """Close whichever model contract this runtime is hosting."""
        if self._modular_mode:
            self.pipeline_runner.close()
            return
        closer = getattr(self.pipeline, "close", None)
        if callable(closer):
            closer()

    def _prepare_inference_control_mode(self) -> None:
        # A modular controller is responsible for selecting/configuring its
        # transport.  The legacy predictor mode checks below must not force a
        # joint-impedance transition for TAVLA or another injected policy.
        if self._modular_mode:
            return
        if not self.command_enabled or not self.config.predictor.enabled:
            return
        if (
            self.config.predictor.mode
            in {
                "contact_world_model",
                "contact_world_model_opd",
                "contact_wm",
                "contact_wm_opd",
            }
            and self.config.execution.mode == "q"
        ):
            return
        if (
            self.config.predictor.mode
            in {
                "swm",
                "swm_opd",
                "torque_world_model",
                "torque_world_model_opd",
                "torque_wm",
                "torque_wm_opd",
            }
            and self.config.execution.mode == "q"
        ):
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
        # Legacy DP/WM pipelines maintain their own high-rate CAN history.
        # Modular policies receive the canonical observation directly; only
        # call an optional history hook when the injected object explicitly
        # provides one.
        append_history = getattr(self.pipeline, "append_continuous_can_observation", None)
        if self._modular_mode and not callable(append_history):
            self._last_consumed_state_timestamp_us = records[-1].timestamp_us
            return
        for record in records:
            values = dict(
                q=record.q,
                dq=record.dq,
                ddq=record.ddq,
                tau=record.tau,
                wrench=record.wrench,
                timestamp_s=record.timestamp_us * 1.0e-6,
            )
            if self.config.predictor.mode in {
                "swm",
                "swm_opd",
                "torque_world_model",
                "torque_world_model_opd",
                "torque_wm",
                "torque_wm_opd",
            }:
                values["q_cmd"] = getattr(record, "q_cmd", None)
            if callable(append_history):
                append_history(**values)
        self._last_consumed_state_timestamp_us = records[-1].timestamp_us

    def step(self):
        if not self._started:
            raise RuntimeError("runtime must be started before step()")
        return self.pipeline_runner.step()

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
        """Mirror the DP's samplewise physical-wrench gate for visualization."""
        value = np.asarray(wrench, dtype=np.float64).reshape(-1)
        if self._dp_contact_threshold_n is None:
            return value.copy()
        force = value[list(self._dp_contact_force_dims)]
        magnitude = float(np.linalg.norm(force))
        if magnitude <= self._dp_contact_threshold_n:
            return np.zeros(6, dtype=np.float64)
        return value.copy()

    def stop(self, *, preserve_arm_enabled: bool = False) -> None:
        if not self._started:
            self._close_model_pipeline()
            return
        try:
            if (
                not self._modular_mode
                and not preserve_arm_enabled
                and self.command_enabled
                and self.config.predictor.enabled
                and not (
                    self.config.predictor.mode
                    in {
                        "contact_world_model",
                        "contact_world_model_opd",
                        "contact_wm",
                        "contact_wm_opd",
                    }
                    and self.config.execution.mode == "q"
                )
                and not (
                    self.config.predictor.mode
                    in {
                        "swm",
                        "swm_opd",
                        "torque_world_model",
                        "torque_world_model_opd",
                        "torque_wm",
                        "torque_wm_opd",
                    }
                    and self.config.execution.mode == "q"
                )
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
            self.tau_ext_plotter.close()
            if not preserve_arm_enabled:
                try:
                    self.arm.disable()
                except Exception as exc:
                    log.warning("arm disable failed during shutdown: %s", exc)
            self.arm.disconnect()
            self._close_model_pipeline()
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
