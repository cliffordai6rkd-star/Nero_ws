from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from inference.checkpoints import call_pinn, restore_checkpoint_model
from inference.config import InferenceConfig
from inference.torque_filter import CausalTorqueCommandFilter
from nero_collection.control import (
    OSCQPController,
    OSCQPResult,
    OSCTargetTrajectory,
    PinocchioDynamicsModel,
)


log = logging.getLogger(__name__)


_Q_TAU_WORLD_MODEL_MODES = frozenset(
    ("world_model_v4", "world_model_v5")
)
_WORLD_MODEL_MODES = frozenset(
    ("world_model_v3", *_Q_TAU_WORLD_MODEL_MODES)
)


@dataclass(frozen=True)
class InferenceInput:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    image: np.ndarray
    wrench_ext: np.ndarray
    timestamp_s: float
    wrench_to_control_rotation: np.ndarray | None = None
    image_timestamp_s: float | None = None


@dataclass(frozen=True)
class InferenceOutput:
    tau_command: np.ndarray
    tau_unfiltered: np.ndarray
    action_target: np.ndarray
    # The complete, unmodified chunk returned by the DP checkpoint. This is
    # kept separate from action_target, which may be safety-clipped/selected.
    dp_action_chunk: np.ndarray | None
    target_wrench: np.ndarray
    qp_result: OSCQPResult | None
    joint_position_command: np.ndarray | None
    ik_result: IKResult | None
    dp_updated: bool
    pinn_updated: bool


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    rotation_error_rad: float


@dataclass(frozen=True)
class _DPPrediction:
    action_target: np.ndarray
    raw_action_chunk: np.ndarray
    action_chunk: np.ndarray
    elapsed_s: float


class NeroInferencePipeline:
    """Configurable open-loop/asynchronous DP with predictor or direct IK."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        dp_model: Any | None = None,
        pinn_model: Any | None = None,
        controller: OSCQPController | None = None,
        world_model_wrench_adapter: Any | None = None,
    ) -> None:
        self.config = config
        self._predictor_enabled = bool(config.predictor.enabled)
        self.dp = dp_model or restore_checkpoint_model(
            config.dp_checkpoint.path,
            config.dp_checkpoint.device,
            use_ema=config.dp_checkpoint.use_ema,
            kind="DP",
            model_overrides=_dp_model_overrides(config),
        )
        if self._predictor_enabled:
            if pinn_model is not None:
                self.pinn = pinn_model
            else:
                if config.pinn_checkpoint is None:
                    raise ValueError(
                        "pinn_checkpoint is required when predictor.enabled=true"
                    )
                self.pinn = restore_checkpoint_model(
                    config.pinn_checkpoint.path,
                    config.pinn_checkpoint.device,
                    use_ema=config.pinn_checkpoint.use_ema,
                    kind="PINN",
                    pinn_mode=config.predictor.mode,
                )
        else:
            self.pinn = None
        self._predictor_mode = config.predictor.mode
        self._world_model_wrench_adapter = world_model_wrench_adapter
        if (
            self._predictor_enabled
            and self._predictor_mode in _WORLD_MODEL_MODES
        ):
            if self._world_model_wrench_adapter is None:
                from inference.world_model import WorldModelWrenchAdapter

                self._world_model_wrench_adapter = (
                    WorldModelWrenchAdapter.from_collection_config(
                        config.runtime.collection_config
                    )
                )
        elif self._predictor_enabled and self._predictor_mode != "wrench_gru":
            raise ValueError(f"unsupported predictor mode: {self._predictor_mode!r}")
        if controller is None:
            dynamics = PinocchioDynamicsModel(
                config.robot.urdf_path,
                frame_name=config.robot.frame_name,
                locked_joint_names=config.robot.locked_joint_names,
            )
            controller = OSCQPController(dynamics, config.osc_qp)
        self.controller = controller
        self.model = controller.model

        self._n_obs_steps = int(getattr(self.dp, "n_obs_steps", 1))
        self._wrench_history_steps = int(
            getattr(getattr(self.dp, "obs_encoder", None), "wrench_history_steps", 1)
        )
        if self._n_obs_steps < 1 or self._wrench_history_steps < 1:
            raise ValueError(
                "DP checkpoint observation contract requires positive n_obs_steps "
                "and wrench_history_steps"
            )
        log.info(
            "DP observation contract images=%d CAN-derived samples/image=%d "
            "total_CAN_samples/batch=%d",
            self._n_obs_steps,
            self._wrench_history_steps,
            self._n_obs_steps * self._wrench_history_steps,
        )
        self._image_key = str(getattr(self.dp, "image_key", "wrist"))
        self._wrench_key = str(getattr(self.dp, "wrench_key", "wrench_ext"))
        self._images: deque[np.ndarray] = deque(maxlen=self._n_obs_steps)
        self._wrenches: deque[np.ndarray] = deque(
            maxlen=self._n_obs_steps * self._wrench_history_steps
        )
        self._timed_images: deque[tuple[float, np.ndarray]] = deque(maxlen=256)
        self._timed_wrenches: deque[tuple[float, np.ndarray]] = deque(maxlen=2048)
        self._latest_policy_anchor_s: float | None = None
        self._last_submitted_policy_anchor_s: float | None = None
        self._open_loop_observation_start_s: float | None = None
        self._wm_history_horizon = int(
            getattr(self.pinn, "history_horizon", 1)
        )
        self._wm_history = {
            "q": deque(maxlen=self._wm_history_horizon),
            "v": deque(maxlen=self._wm_history_horizon),
            "a": deque(maxlen=self._wm_history_horizon),
            "tau": deque(maxlen=self._wm_history_horizon),
            "wrench": deque(maxlen=self._wm_history_horizon),
        }
        self._action: np.ndarray | None = None
        self._action_chunk: np.ndarray | None = None
        self._dp_action_chunk: np.ndarray | None = None
        self._dp_chunk_sequence = 0
        self._pinn_held_action: np.ndarray | None = None
        self._execution_plan: np.ndarray | None = None
        self._execution_plan_index = 0
        self._execution_next_step_s: float | None = None
        self._execution_plan_step_s: float | None = None
        self._execution_plan_active = False
        self._pending_dp_prediction: _DPPrediction | None = None
        checkpoint_config = getattr(self.pinn, "_inference_checkpoint_config", {})
        contact_gate_config = (
            checkpoint_config.get("contact_gate", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        self._pinn_contact_gate_enabled = bool(
            contact_gate_config.get("enabled", False)
        )
        self._pinn_contact_probability_threshold = float(
            contact_gate_config.get("probability_threshold", 0.5)
        )
        pinn_model_config = (
            checkpoint_config.get("model", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        action_key = (
            pinn_model_config.get("action_key")
            if isinstance(pinn_model_config, Mapping)
            else None
        )
        self._pinn_action_key = None if action_key is None else str(action_key)
        action_normalizer_key = (
            pinn_model_config.get("action_normalizer_key")
            if isinstance(pinn_model_config, Mapping)
            else None
        )
        self._pinn_action_normalizer_key = (
            None
            if action_normalizer_key is None
            else str(action_normalizer_key)
        )
        self._pinn_action_condition_mode = str(
            pinn_model_config.get("action_condition_mode", "relative_pose")
        ).lower()
        if self._pinn_action_condition_mode not in {
            "absolute_pose",
            "relative_pose",
        }:
            raise ValueError(
                "PINN checkpoint model.action_condition_mode must be "
                "'absolute_pose' or 'relative_pose', got "
                f"{self._pinn_action_condition_mode!r}"
            )
        self._pinn_future_horizon = int(getattr(self.pinn, "future_horizon", 1))
        configured_action_fill = config.predictor.action_condition_fill
        self._pinn_action_condition_fill = (
            "hold"
            if configured_action_fill == "auto"
            and self._predictor_mode == "world_model_v5"
            else (
                "chunk"
                if configured_action_fill == "auto"
                else configured_action_fill
            )
        )
        loss_config = (
            checkpoint_config.get("loss", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        state_estimator_config = (
            pinn_model_config.get("state_estimator", {})
            if isinstance(pinn_model_config, Mapping)
            else {}
        )
        sampling_dt = getattr(self.pinn, "sampling_dt", None)
        if sampling_dt is None and isinstance(state_estimator_config, Mapping):
            sampling_dt = state_estimator_config.get("sampling_dt")
        if sampling_dt is None and isinstance(loss_config, Mapping):
            sampling_dt = loss_config.get("sampling_dt")
        self._wm_sampling_dt_s = (
            float(sampling_dt) if sampling_dt is not None else None
        )
        if (
            self._wm_sampling_dt_s is not None
            and (
                not np.isfinite(self._wm_sampling_dt_s)
                or self._wm_sampling_dt_s <= 0.0
            )
        ):
            raise ValueError("PINN checkpoint loss.sampling_dt must be positive")
        self._wm_previous_sample: tuple[float, dict[str, np.ndarray]] | None = None
        self._wm_next_sample_s: float | None = None
        action_frame_name = pinn_model_config.get("action_current_frame_name")
        if action_frame_name is None:
            action_frame_name = config.robot.action_frame_name or config.robot.frame_name
        self._action_frame_name = str(action_frame_name)
        self._control_frame_name = config.robot.frame_name
        self._action_to_control_transform: np.ndarray | None = None
        self._target_wrenches = np.zeros(
            (self.controller.config.horizon_steps, 6), dtype=np.float64
        )
        self._last_tau: np.ndarray | None = None
        self._torque_filter = CausalTorqueCommandFilter(config.torque_filter)
        self._last_control_timestamp_s: float | None = None
        self._pinn_state: Any = None
        self._dp_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nero-dp")
        self._dp_future: Future[_DPPrediction] | None = None
        self._timing_started_s = perf_counter()
        self._timing_cycles = 0
        self._timing_pinn_s = 0.0
        self._timing_qp_s = 0.0
        self._timing_ik_s = 0.0
        self._timing_dp_s: list[float] = []

    def reset(self) -> None:
        # Do not reset checkpoint-owned recurrent/model state while an old DP
        # episode is still using it in the worker thread.
        future = self._dp_future
        self._dp_future = None
        if future is not None and not future.cancel():
            try:
                future.result()
            except Exception as exc:
                log.warning("discarding failed DP result during episode reset: %s", exc)
        self._images.clear()
        self._wrenches.clear()
        self._timed_images.clear()
        self._timed_wrenches.clear()
        self._latest_policy_anchor_s = None
        self._last_submitted_policy_anchor_s = None
        self._open_loop_observation_start_s = None
        for values in self._wm_history.values():
            values.clear()
        self._action = None
        self._action_chunk = None
        self._dp_action_chunk = None
        self._dp_chunk_sequence = 0
        self._pinn_held_action = None
        self._execution_plan = None
        self._execution_plan_index = 0
        self._execution_next_step_s = None
        self._execution_plan_step_s = None
        self._execution_plan_active = False
        self._pending_dp_prediction = None
        self._target_wrenches[:] = 0.0
        self._last_tau = None
        self._torque_filter.reset()
        self._last_control_timestamp_s = None
        self._pinn_state = None
        self._wm_previous_sample = None
        self._wm_next_sample_s = None
        self._reset_timing()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        for model in (self.dp, self.pinn):
            if model is None:
                continue
            if hasattr(model, "reset"):
                model.reset()

    def close(self) -> None:
        """Stop the asynchronous DP worker after the control session."""
        self._dp_executor.shutdown(wait=True, cancel_futures=True)

    @property
    def observation_steps(self) -> int:
        return self._n_obs_steps

    @property
    def wrench_history_steps(self) -> int:
        return self._wrench_history_steps

    @property
    def observation_step_s(self) -> float | None:
        config = getattr(self.dp, "_inference_checkpoint_config", {})
        if not isinstance(config, Mapping):
            return None
        task = config.get("task", {})
        dataset = task.get("dataset", {}) if isinstance(task, Mapping) else {}
        value = dataset.get("timestamp_step_sec") if isinstance(dataset, Mapping) else None
        if value is None:
            return None
        try:
            step_s = float(value)
        except (TypeError, ValueError):
            return None
        return step_s if np.isfinite(step_s) and step_s > 0 else None

    @property
    def open_loop_execution_active(self) -> bool:
        """Whether a frozen open-loop action plan is currently executing."""
        return (
            self.config.predictor.inference_mode == "open_loop"
            and self._execution_plan_active
        )

    def step(self, sample: InferenceInput) -> InferenceOutput:
        """Run one state-driven control cycle using the configured DP execution mode."""
        cycle_started_s = perf_counter()
        self._validate_input(sample)
        open_loop = self.config.predictor.inference_mode == "open_loop"
        plan_completed = self._advance_execution_plan(sample.timestamp_s)
        if open_loop and plan_completed:
            # Do not let images captured while the previous action was moving
            # leak into the next policy observation.  The completion sample is
            # the first possible member of the next frozen observation batch.
            self._clear_dp_observations(start_s=sample.timestamp_s)
        if not open_loop or not self._execution_plan_active:
            self._append_observation(
                sample.image,
                sample.wrench_ext,
                image_timestamp_s=sample.image_timestamp_s,
                state_timestamp_s=sample.timestamp_s,
                allow_backfill=not open_loop,
            )
        self._append_world_model_observation(sample)
        current_control_pose = self.controller.model.snapshot(sample.q, sample.dq).pose
        current_action_pose = self._current_action_pose(
            sample.q,
            current_control_pose,
        )

        dp_updated = self._update_dp_execution(
            sample.timestamp_s,
            current_action_pose,
        )
        if self._action is None:
            self._set_idle_action(current_action_pose)

        if not self._predictor_enabled:
            return self._step_direct_ik(
                sample,
                current_action_pose,
                current_control_pose,
                dp_updated=dp_updated,
                cycle_started_s=cycle_started_s,
            )

        # Every completed PINN inference immediately becomes the wrench reference.
        pinn_started_s = perf_counter()
        self._target_wrenches = self._predict_wrenches(
            sample,
            current_action_pose,
        )
        self._timing_pinn_s += perf_counter() - pinn_started_s
        self._update_qp_timestep(sample.timestamp_s)

        assert self._action_chunk is not None
        action_poses = np.stack(
            [_action_to_pose(action) for action in self._action_chunk],
            axis=0,
        )
        control_poses = self._action_poses_to_control(
            action_poses,
            sample.q,
            current_action_pose,
            current_control_pose,
        )
        control_poses = _fit_pose_horizon(
            control_poses,
            self.controller.config.horizon_steps,
        )
        target = OSCTargetTrajectory(
            poses=control_poses,
            wrenches=_rotate_wrenches(
                self._target_wrenches,
                sample.wrench_to_control_rotation,
            ),
        )
        measured_wrench = _rotate_wrenches(
            sample.wrench_ext[None],
            sample.wrench_to_control_rotation,
        )[0]
        qp_started_s = perf_counter()
        result = self.controller.optimize_mpc(
            sample.q,
            sample.dq,
            target,
            measured_wrench=measured_wrench,
            previous_tau=self._last_tau,
        )
        self._timing_qp_s += perf_counter() - qp_started_s
        tau_unfiltered = self._clip_tau(result.first_tau)
        initial_tau = self._clip_tau(sample.tau)
        tau = self._torque_filter.apply(
            tau_unfiltered,
            dt_s=self.controller.config.dt_s,
            initial_tau=initial_tau,
        )
        tau = self._clip_tau(tau)
        self._last_tau = tau
        output = InferenceOutput(
            tau_command=tau,
            tau_unfiltered=tau_unfiltered,
            action_target=self._action.copy(),
            dp_action_chunk=(
                None
                if self._dp_action_chunk is None
                else self._dp_action_chunk.copy()
            ),
            target_wrench=target.wrenches[0].copy(),
            qp_result=result,
            joint_position_command=None,
            ik_result=None,
            dp_updated=dp_updated,
            pinn_updated=True,
        )
        self._timing_cycles += 1
        self._report_timing(perf_counter() - cycle_started_s)
        return output

    def _update_dp_execution(
        self,
        timestamp_s: float,
        current_action_pose: np.ndarray,
    ) -> bool:
        """Update the high-level action plan for both IK and predictor branches."""
        inference_mode = self.config.predictor.inference_mode
        plan_locked = self._action_plan_is_locked()

        if inference_mode == "open_loop":
            if self._execution_plan_active:
                return False
            snapshot = self._observation_snapshot_for_dp()
            if snapshot is None:
                return False
            images, wrenches = snapshot
            prediction = self._predict_action(images, wrenches)
            self._timing_dp_s.append(prediction.elapsed_s)
            self._install_dp_prediction(
                prediction,
                current_action_pose,
                timestamp_s,
                scheduled=True,
            )
            return True

        dp_updated = False
        if (
            plan_locked
            and not self._execution_plan_active
            and self._pending_dp_prediction is not None
        ):
            pending = self._pending_dp_prediction
            self._pending_dp_prediction = None
            self._install_dp_prediction(
                pending,
                current_action_pose,
                timestamp_s,
                scheduled=True,
            )
            dp_updated = True

        if self._dp_future is not None and self._dp_future.done():
            prediction = self._dp_future.result()
            self._timing_dp_s.append(prediction.elapsed_s)
            self._dp_future = None
            if plan_locked and self._execution_plan_active:
                # Keep the newest result, but never interrupt a complete chunk or
                # an in-progress interpolation trajectory.
                self._pending_dp_prediction = prediction
            else:
                self._install_dp_prediction(
                    prediction,
                    current_action_pose,
                    timestamp_s,
                    scheduled=plan_locked,
                )
                dp_updated = True

        if self._dp_future is None:
            snapshot = self._observation_snapshot_for_dp()
            if snapshot is not None:
                images, wrenches = snapshot
                self._dp_future = self._dp_executor.submit(
                    self._predict_action,
                    images,
                    wrenches,
                )
        return dp_updated

    def _install_dp_prediction(
        self,
        prediction: _DPPrediction,
        current_action_pose: np.ndarray,
        timestamp_s: float,
        *,
        scheduled: bool,
    ) -> None:
        safe_chunk = self._safe_action_chunk(
            prediction.action_chunk,
            current_action_pose,
        )
        self._dp_action_chunk = np.asarray(
            prediction.raw_action_chunk, dtype=np.float64
        ).copy()
        self._dp_chunk_sequence += 1
        current_action = _pose_to_action(current_action_pose)
        for action_index, action_pred in enumerate(self._dp_action_chunk):
            log.info(
                "DP action chunk #%d[%02d] delta_xyz_rotvec=%s",
                self._dp_chunk_sequence,
                action_index,
                _format_action_delta(current_action, action_pred),
            )
        chunk_mode = self.config.predictor.action_chunk_mode
        execution_mode = self.config.predictor.action_execution_mode
        if execution_mode in {"linear_target", "minimum_jerk_target"}:
            target_action = _select_action_chunk(safe_chunk, chunk_mode)
            interpolation_steps = self.config.predictor.action_interpolation_steps
            duration_s = self.config.predictor.action_interpolation_duration_s
            if duration_s is None:
                duration_s = self._action_execution_step_s()
            duration_s = float(duration_s)
            if not np.isfinite(duration_s) or duration_s <= 0.0:
                raise ValueError(
                    "minimum-jerk action duration must be positive and finite"
                )
            plan = _minimum_jerk_action_plan(
                current_action,
                target_action,
                interpolation_steps,
            )
            plan_step_s = duration_s / interpolation_steps
            held_action = target_action
            log.info(
                "DP minimum-jerk target chunk_mode=%s steps=%d "
                "duration=%.4fs target_delta=%s",
                chunk_mode,
                interpolation_steps,
                duration_s,
                _format_action_delta(current_action, target_action),
            )
        elif chunk_mode == "all":
            plan = safe_chunk
            plan_step_s = self._action_execution_step_s()
            held_action = plan[0]
        else:
            plan = _select_action_chunk(safe_chunk, chunk_mode)[None]
            plan_step_s = self._action_execution_step_s()
            held_action = plan[0]
        self._execution_plan = np.asarray(plan, dtype=np.float64).copy()
        self._execution_plan_index = 0
        self._action = self._execution_plan[0].copy()
        self._action_chunk = self._execution_plan.copy()
        self._pinn_held_action = np.asarray(held_action, dtype=np.float64).copy()
        if scheduled:
            self._execution_plan_active = True
            self._execution_plan_step_s = plan_step_s
            self._execution_next_step_s = float(timestamp_s) + plan_step_s
        else:
            self._execution_plan_active = False
            self._execution_next_step_s = None
            self._execution_plan_step_s = None

    def _advance_execution_plan(self, timestamp_s: float) -> bool:
        if (
            not self._execution_plan_active
            or self._execution_plan is None
            or self._execution_next_step_s is None
            or self._execution_plan_step_s is None
        ):
            return False
        timestamp_s = float(timestamp_s)
        step_s = self._execution_plan_step_s
        while timestamp_s + 1.0e-9 >= self._execution_next_step_s:
            next_index = self._execution_plan_index + 1
            if next_index >= len(self._execution_plan):
                self._execution_plan_active = False
                self._execution_next_step_s = None
                self._execution_plan_step_s = None
                self._execution_plan_index = len(self._execution_plan) - 1
                self._action = self._execution_plan[-1].copy()
                self._action_chunk = self._execution_plan[-1:].copy()
                return True
            self._execution_plan_index = next_index
            self._action = self._execution_plan[next_index].copy()
            self._action_chunk = self._execution_plan[next_index:].copy()
            self._execution_next_step_s += step_s
        return False

    def _clear_dp_observations(self, *, start_s: float | None = None) -> None:
        self._images.clear()
        self._wrenches.clear()
        self._timed_images.clear()
        self._timed_wrenches.clear()
        self._latest_policy_anchor_s = None
        self._last_submitted_policy_anchor_s = None
        self._open_loop_observation_start_s = (
            None if start_s is None else float(start_s)
        )

    def _action_plan_is_locked(self) -> bool:
        return (
            self.config.predictor.action_chunk_mode == "all"
            or self.config.predictor.action_execution_mode
            in {"linear_target", "minimum_jerk_target"}
        )

    def _action_execution_step_s(self) -> float:
        configured = self.config.predictor.action_step_s
        if configured is not None:
            return float(configured)
        return self.observation_step_s or 0.1

    def _set_idle_action(self, current_action_pose: np.ndarray) -> None:
        self._action = _pose_to_action(current_action_pose)
        self._pinn_held_action = self._action.copy()
        self._action_chunk = np.repeat(
            self._action[None],
            max(1, self._pinn_future_horizon),
            axis=0,
        )

    def step_direct_ik_synchronous(self, sample: InferenceInput) -> InferenceOutput:
        """Run one deterministic DP -> IK step for offline dataset inference.

        Unlike :meth:`step`, this method does not submit work to the asynchronous
        DP worker. Each call appends exactly one observation and consumes its DP
        result before returning, so output does not depend on host playback speed.
        """
        if self._predictor_enabled:
            raise RuntimeError(
                "synchronous direct-IK inference requires predictor.enabled=false"
            )
        cycle_started_s = perf_counter()
        self._validate_input(sample)
        self._append_observation(sample.image, sample.wrench_ext)
        images, wrenches = self._observation_snapshot()
        return self._step_direct_ik_from_snapshot(
            sample,
            images,
            wrenches,
            cycle_started_s=cycle_started_s,
        )

    def step_direct_ik_observation_history(
        self,
        sample: InferenceInput,
        image_history: np.ndarray,
        wrench_history: np.ndarray,
    ) -> InferenceOutput:
        """Run direct IK from an already aligned model-contract observation."""
        if self._predictor_enabled:
            raise RuntimeError(
                "direct-IK observation inference requires predictor.enabled=false"
            )
        cycle_started_s = perf_counter()
        self._validate_input(sample)
        images = np.asarray(image_history)
        expected_image_prefix = (self._n_obs_steps,)
        if (
            images.ndim != 4
            or images.shape[:1] != expected_image_prefix
            or images.shape[-1] != 3
        ):
            raise ValueError(
                "image_history must have shape "
                f"[{self._n_obs_steps},H,W,3], got {images.shape}"
            )
        processed_images = []
        for image in images:
            value = np.asarray(image, dtype=np.float32)
            if value.max(initial=0.0) > 1.0:
                value /= 255.0
            processed_images.append(np.moveaxis(value, -1, 0))
        wrenches = np.asarray(wrench_history, dtype=np.float32)
        expected_wrench_shape = (
            self._n_obs_steps,
            self._wrench_history_steps,
            6,
        )
        if wrenches.shape != expected_wrench_shape or not np.all(np.isfinite(wrenches)):
            raise ValueError(
                f"wrench_history must have finite shape {expected_wrench_shape}, "
                f"got {wrenches.shape}"
            )
        return self._step_direct_ik_from_snapshot(
            sample,
            np.stack(processed_images),
            wrenches,
            cycle_started_s=cycle_started_s,
        )

    def _step_direct_ik_from_snapshot(
        self,
        sample: InferenceInput,
        images: np.ndarray,
        wrenches: np.ndarray,
        *,
        cycle_started_s: float,
    ) -> InferenceOutput:
        current_control_pose = self.model.snapshot(sample.q, sample.dq).pose
        current_action_pose = self._current_action_pose(
            sample.q,
            current_control_pose,
        )
        self._advance_execution_plan(sample.timestamp_s)
        scheduled = (
            self.config.predictor.inference_mode == "open_loop"
            or self._action_plan_is_locked()
        )
        if scheduled and self._execution_plan_active:
            assert self._action is not None
            return self._step_direct_ik(
                sample,
                current_action_pose,
                current_control_pose,
                dp_updated=False,
                cycle_started_s=cycle_started_s,
            )
        prediction = self._predict_action(images, wrenches)
        self._timing_dp_s.append(prediction.elapsed_s)
        self._install_dp_prediction(
            prediction,
            current_action_pose,
            sample.timestamp_s,
            scheduled=scheduled,
        )
        return self._step_direct_ik(
            sample,
            current_action_pose,
            current_control_pose,
            dp_updated=True,
            cycle_started_s=cycle_started_s,
        )

    def _step_direct_ik(
        self,
        sample: InferenceInput,
        current_action_pose: np.ndarray,
        current_control_pose: np.ndarray,
        *,
        dp_updated: bool,
        cycle_started_s: float,
    ) -> InferenceOutput:
        """Execute the selected DP pose through IK without the force expert/QP."""
        assert self._action is not None
        action_pose = _action_to_pose(self._action)
        control_pose = self._action_poses_to_control(
            action_pose[None],
            sample.q,
            current_action_pose,
            current_control_pose,
        )[0]
        ik_started_s = perf_counter()
        ik_result = self._solve_ik(sample.q, control_pose)
        self._timing_ik_s += perf_counter() - ik_started_s
        if not ik_result.converged:
            raise RuntimeError(
                "IK did not converge for the selected DP action: "
                f"position_error={ik_result.position_error_m:.6g} m, "
                f"rotation_error={ik_result.rotation_error_rad:.6g} rad, "
                f"iterations={ik_result.iterations}"
            )
        q_command = self._safe_joint_position_command(sample.q, ik_result.q)
        zeros7 = np.zeros(7, dtype=np.float64)
        output = InferenceOutput(
            tau_command=zeros7.copy(),
            tau_unfiltered=zeros7.copy(),
            action_target=self._action.copy(),
            dp_action_chunk=(
                None
                if self._dp_action_chunk is None
                else self._dp_action_chunk.copy()
            ),
            target_wrench=np.zeros(6, dtype=np.float64),
            qp_result=None,
            joint_position_command=q_command,
            ik_result=ik_result,
            dp_updated=dp_updated,
            pinn_updated=False,
        )
        self._timing_cycles += 1
        self._report_timing(perf_counter() - cycle_started_s)
        return output

    def _solve_ik(self, q_seed: np.ndarray, target_pose: np.ndarray) -> IKResult:
        cfg = self.config.ik
        q = _numpy_vector(q_seed, 7, "IK q_seed").copy()
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK target_pose must be a finite 4x4 matrix")
        lower, upper = self._joint_position_bounds(cfg.joint_position_margin_rad)
        q = np.clip(q, lower, upper)
        position_error = np.inf
        rotation_error = np.inf
        for iteration in range(cfg.max_iterations + 1):
            snapshot = self.model.snapshot(q, np.zeros(7, dtype=np.float64))
            position_delta = target[:3, 3] - snapshot.pose[:3, 3]
            rotation_delta = Rotation.from_matrix(
                target[:3, :3] @ snapshot.pose[:3, :3].T
            ).as_rotvec()
            position_error = float(np.linalg.norm(position_delta))
            rotation_error = float(np.linalg.norm(rotation_delta))
            if (
                position_error <= cfg.position_tolerance_m
                and rotation_error <= cfg.rotation_tolerance_rad
            ):
                return IKResult(
                    q=q.copy(),
                    converged=True,
                    iterations=iteration,
                    position_error_m=position_error,
                    rotation_error_rad=rotation_error,
                )
            if iteration == cfg.max_iterations:
                break
            jacobian = np.asarray(snapshot.jacobian, dtype=np.float64)
            if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
                raise RuntimeError(
                    f"IK requires a finite 6x7 frame Jacobian, got {jacobian.shape}"
                )
            error = np.concatenate((position_delta, rotation_delta))
            regularized = jacobian @ jacobian.T + np.eye(6) * cfg.damping**2
            delta_q = jacobian.T @ np.linalg.solve(regularized, error)
            delta_q *= cfg.step_gain
            step_norm = float(np.linalg.norm(delta_q))
            if step_norm > cfg.maximum_iteration_step_rad:
                delta_q *= cfg.maximum_iteration_step_rad / step_norm
            q = np.clip(q + delta_q, lower, upper)
        return IKResult(
            q=q.copy(),
            converged=False,
            iterations=cfg.max_iterations,
            position_error_m=position_error,
            rotation_error_rad=rotation_error,
        )

    def _joint_position_bounds(self, margin_rad: float) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.model.position_lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.model.position_upper, dtype=np.float64).reshape(-1)
        if lower.shape != (7,) or upper.shape != (7,):
            raise RuntimeError("IK model joint limits must be 7-vectors")
        lower = lower + margin_rad
        upper = upper - margin_rad
        if np.any(lower >= upper):
            raise RuntimeError("IK joint-position margin leaves an empty joint range")
        return lower, upper

    def _safe_joint_position_command(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
    ) -> np.ndarray:
        current = _numpy_vector(current_q, 7, "current q")
        target = _numpy_vector(target_q, 7, "IK target q")
        limit = np.asarray(
            self.config.safety.maximum_joint_position_step_rad,
            dtype=np.float64,
        )
        if limit.ndim == 0:
            limit = np.repeat(limit, 7)
        limit = limit.reshape(-1)
        if (
            limit.shape != (7,)
            or not np.all(np.isfinite(limit))
            or np.any(limit <= 0)
        ):
            raise ValueError(
                "maximum_joint_position_step_rad must be a positive scalar or 7-vector"
            )
        lower, upper = self._joint_position_bounds(
            self.config.ik.joint_position_margin_rad
        )
        bounded_target = np.clip(target, lower, upper)
        command = current + np.clip(bounded_target - current, -limit, limit)
        physical_lower, physical_upper = self._joint_position_bounds(0.0)
        return np.clip(command, physical_lower, physical_upper)

    def _update_qp_timestep(self, timestamp_s: float) -> None:
        """Use measured loop time for QP rollout instead of a configured nominal Hz."""
        previous = self._last_control_timestamp_s
        self._last_control_timestamp_s = float(timestamp_s)
        if previous is None:
            return
        measured_dt = float(timestamp_s) - previous
        if measured_dt <= 0.0:
            raise ValueError("InferenceInput.timestamp_s must increase every control step")
        self.controller.config = replace(self.controller.config, dt_s=measured_dt)

    def _append_observation(
        self,
        image: np.ndarray,
        wrench: np.ndarray,
        *,
        image_timestamp_s: float | None = None,
        state_timestamp_s: float | None = None,
        allow_backfill: bool = True,
    ) -> None:
        image_value = np.asarray(image)
        if image_value.ndim != 3 or image_value.shape[-1] != 3:
            raise ValueError("image must have shape [H,W,3]")
        image_value = image_value.astype(np.float32)
        if image_value.max(initial=0.0) > 1.0:
            image_value /= 255.0
        chw_image = np.moveaxis(image_value, -1, 0)
        wrench_value = np.asarray(wrench, dtype=np.float32).copy()
        if image_timestamp_s is not None and state_timestamp_s is not None:
            image_time = float(image_timestamp_s)
            state_time = float(state_timestamp_s)
            if not np.isfinite(image_time) or not np.isfinite(state_time):
                raise ValueError("observation timestamps must be finite")
            if (
                not self._timed_wrenches
                or state_time > self._timed_wrenches[-1][0]
            ):
                self._timed_wrenches.append((state_time, wrench_value))
            image_is_in_current_batch = (
                self._open_loop_observation_start_s is None
                or image_time + 1.0e-9 >= self._open_loop_observation_start_s
            )
            if (
                image_is_in_current_batch
                and (not self._timed_images or image_time > self._timed_images[-1][0])
            ):
                self._timed_images.append((image_time, chw_image))
                step_s = self.observation_step_s
                if (
                    self._latest_policy_anchor_s is None
                    or step_s is None
                    or image_time - self._latest_policy_anchor_s >= step_s * 0.999
                ):
                    self._latest_policy_anchor_s = image_time
            return

        self._images.append(chw_image)
        self._wrenches.append(wrench_value)
        if not allow_backfill:
            # A pure open-loop batch must contain the checkpoint's actual
            # number of observations; never synthesize its prefix by repeating
            # the first frame.
            return
        while len(self._images) < self._n_obs_steps:
            self._images.appendleft(self._images[0].copy())
        while len(self._wrenches) < self._wrenches.maxlen:
            self._wrenches.appendleft(self._wrenches[0].copy())

    def _observation_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack(tuple(self._images), axis=0),
            np.stack(tuple(self._wrenches), axis=0).reshape(
                self._n_obs_steps, self._wrench_history_steps, 6
            ),
        )

    def _observation_snapshot_for_dp(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self._timed_images:
            anchor_s = self._latest_policy_anchor_s
            if anchor_s is None or (
                self._last_submitted_policy_anchor_s is not None
                and anchor_s <= self._last_submitted_policy_anchor_s
            ):
                return None
            if self.config.predictor.inference_mode == "open_loop":
                if len(self._timed_images) < self._n_obs_steps:
                    return None
                step_s = self.observation_step_s or 0.1
                if (
                    self._n_obs_steps > 1
                    and anchor_s - self._timed_images[0][0]
                    < (self._n_obs_steps - 1) * step_s * 0.999
                ):
                    return None
                wrench_step_s = step_s / max(self._wrench_history_steps, 1)
                required_wrench_span_s = (
                    (self._n_obs_steps - 1) * step_s
                    + (self._wrench_history_steps - 1) * wrench_step_s
                )
                if (
                    not self._timed_wrenches
                    or anchor_s - self._timed_wrenches[0][0]
                    < required_wrench_span_s * 0.999
                ):
                    return None
            if self.config.predictor.inference_mode == "open_loop":
                snapshot = self._timed_observation_snapshot_if_aligned(anchor_s)
                if snapshot is None:
                    return None
            else:
                snapshot = self._timed_observation_snapshot(anchor_s)
            self._last_submitted_policy_anchor_s = anchor_s
            return snapshot
        if self.config.predictor.inference_mode == "open_loop" and (
            len(self._images) < self._n_obs_steps
            or len(self._wrenches) < self._wrenches.maxlen
        ):
            return None
        return self._observation_snapshot()

    def _timed_observation_snapshot(
        self,
        anchor_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_times = np.asarray(
            [timestamp for timestamp, _ in self._timed_images],
            dtype=np.float64,
        )
        wrench_times = np.asarray(
            [timestamp for timestamp, _ in self._timed_wrenches],
            dtype=np.float64,
        )
        if image_times.size == 0 or wrench_times.size == 0:
            raise RuntimeError("timed DP observation requires image and wrench samples")
        image_step_s = self.observation_step_s
        if image_step_s is None:
            image_step_s = 0.1
        image_targets = anchor_s + np.arange(
            -(self._n_obs_steps - 1),
            1,
            dtype=np.float64,
        ) * image_step_s
        image_indices = _nearest_timestamp_indices(image_times, image_targets)
        images = np.stack(
            [self._timed_images[int(index)][1] for index in image_indices],
            axis=0,
        )

        wrench_step_s = image_step_s / max(self._wrench_history_steps, 1)
        history_offsets = np.arange(
            -(self._wrench_history_steps - 1),
            1,
            dtype=np.float64,
        ) * wrench_step_s
        wrench_targets = image_targets[:, None] + history_offsets[None, :]
        wrench_indices = _nearest_timestamp_indices(
            wrench_times,
            wrench_targets.reshape(-1),
        ).reshape(self._n_obs_steps, self._wrench_history_steps)
        wrenches = np.stack(
            [self._timed_wrenches[int(index)][1] for index in wrench_indices.reshape(-1)],
            axis=0,
        ).reshape(self._n_obs_steps, self._wrench_history_steps, 6)
        return images, wrenches

    def append_open_loop_can_observation(
        self,
        wrench: np.ndarray,
        timestamp_s: float,
    ) -> None:
        """Append one CAN-derived sample without reusing a stale image."""
        if self.config.predictor.inference_mode != "open_loop":
            raise RuntimeError(
                "CAN-only observation append is only valid in open-loop mode"
            )
        value = np.asarray(wrench, dtype=np.float32).reshape(-1)
        timestamp_s = float(timestamp_s)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("CAN-derived wrench must be a finite 6-vector")
        if not np.isfinite(timestamp_s):
            raise ValueError("CAN observation timestamp must be finite")
        if not self._timed_wrenches or timestamp_s > self._timed_wrenches[-1][0]:
            self._timed_wrenches.append((timestamp_s, value.copy()))

    def append_continuous_can_observation(
        self,
        *,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
        wrench: np.ndarray,
        timestamp_s: float,
    ) -> None:
        """Backfill one timestamp-consistent record from the runtime CAN ring."""
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("continuous CAN observation timestamp must be finite")
        wrench_value = np.asarray(wrench, dtype=np.float32).reshape(-1)
        if wrench_value.shape != (6,) or not np.all(np.isfinite(wrench_value)):
            raise ValueError("continuous CAN wrench must be a finite 6-vector")
        if not self._timed_wrenches or timestamp_s > self._timed_wrenches[-1][0]:
            self._timed_wrenches.append((timestamp_s, wrench_value.copy()))
        self._append_world_model_observation_values(
            timestamp_s,
            {
                "q": q,
                "v": dq,
                "a": ddq,
                "tau": tau,
                "wrench": wrench_value,
            },
        )

    def _timed_observation_snapshot_if_aligned(
        self,
        anchor_s: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a complete checkpoint batch only after image/CAN alignment."""
        image_times = np.asarray(
            [timestamp for timestamp, _ in self._timed_images], dtype=np.float64
        )
        can_times = np.asarray(
            [timestamp for timestamp, _ in self._timed_wrenches], dtype=np.float64
        )
        if image_times.size < self._n_obs_steps or can_times.size == 0:
            return None
        image_step_s = self.observation_step_s or 0.1
        nominal_image_times = anchor_s + np.arange(
            -(self._n_obs_steps - 1), 1, dtype=np.float64
        ) * image_step_s
        image_indices = _nearest_timestamp_indices(image_times, nominal_image_times)
        if np.unique(image_indices).size != self._n_obs_steps:
            return None
        selected_image_times = image_times[image_indices]
        if np.any(np.diff(selected_image_times) <= 0.0):
            return None

        can_step_s = image_step_s / self._wrench_history_steps
        can_offsets = np.arange(
            -(self._wrench_history_steps - 1), 1, dtype=np.float64
        ) * can_step_s
        can_targets = selected_image_times[:, None] + can_offsets[None, :]
        if (
            can_times[0] > float(np.min(can_targets))
            or can_times[-1] < float(np.max(can_targets))
        ):
            return None
        can_indices = _nearest_timestamp_indices(
            can_times, can_targets.reshape(-1)
        ).reshape(self._n_obs_steps, self._wrench_history_steps)
        if any(
            np.unique(indices).size != self._wrench_history_steps
            for indices in can_indices
        ):
            return None
        if np.unique(can_indices).size != can_indices.size:
            return None
        matched_can_times = can_times[can_indices]
        if np.any(np.diff(matched_can_times, axis=1) <= 0.0):
            return None
        gaps = np.abs(matched_can_times - can_targets)
        if np.any(
            gaps > self.config.runtime.maximum_observation_alignment_gap_s
        ):
            return None

        images = np.stack(
            [self._timed_images[int(index)][1] for index in image_indices], axis=0
        )
        can_values = np.stack(
            [self._timed_wrenches[int(index)][1] for index in can_indices.reshape(-1)],
            axis=0,
        ).reshape(self._n_obs_steps, self._wrench_history_steps, 6)
        return images, can_values

    def _append_world_model_observation(self, sample: InferenceInput) -> None:
        self._append_world_model_observation_values(
            float(sample.timestamp_s),
            {
                "q": sample.q,
                "v": sample.dq,
                "a": sample.ddq,
                "tau": sample.tau,
                "wrench": sample.wrench_ext,
            },
        )

    def _append_world_model_observation_values(
        self,
        timestamp_s: float,
        values: Mapping[str, np.ndarray],
    ) -> None:
        if (
            not self._predictor_enabled
            or self._predictor_mode not in _WORLD_MODEL_MODES
        ):
            return
        values = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in values.items()
        }
        expected_shapes = {
            "q": (7,),
            "v": (7,),
            "a": (7,),
            "tau": (7,),
            "wrench": (6,),
        }
        if set(values) != set(expected_shapes) or any(
            values[key].shape != shape or not np.all(np.isfinite(values[key]))
            for key, shape in expected_shapes.items()
        ):
            raise ValueError(
                "continuous world-model observation must contain finite "
                "q/v/a/tau [7] and wrench [6] vectors"
            )
        sampling_dt = self._wm_sampling_dt_s
        if sampling_dt is None:
            previous = self._wm_previous_sample
            if previous is not None and timestamp_s <= previous[0]:
                if timestamp_s == previous[0]:
                    return
                raise ValueError("world-model observation timestamps must increase")
            self._append_world_model_values(values)
            self._wm_previous_sample = (timestamp_s, values)
            return
        previous = self._wm_previous_sample
        if previous is None:
            self._append_world_model_values(values)
            self._wm_previous_sample = (timestamp_s, values)
            self._wm_next_sample_s = timestamp_s + sampling_dt
            return
        previous_s, previous_values = previous
        if timestamp_s <= previous_s:
            if timestamp_s == previous_s:
                return
            raise ValueError("world-model observation timestamps must increase")
        assert self._wm_next_sample_s is not None
        while self._wm_next_sample_s <= timestamp_s + 1.0e-9:
            alpha = (self._wm_next_sample_s - previous_s) / (
                timestamp_s - previous_s
            )
            interpolated = {
                key: previous_values[key]
                + np.float32(alpha) * (values[key] - previous_values[key])
                for key in values
            }
            self._append_world_model_values(interpolated)
            self._wm_next_sample_s += sampling_dt
        self._wm_previous_sample = (timestamp_s, values)

    def _append_world_model_values(
        self,
        values: Mapping[str, np.ndarray],
    ) -> None:
        for key, value in values.items():
            history = self._wm_history[key]
            history.append(np.asarray(value, dtype=np.float32).copy())
            while len(history) < history.maxlen:
                history.appendleft(history[0].copy())

    def _predict_action(
        self, images: np.ndarray, wrench: np.ndarray
    ) -> _DPPrediction:
        import torch

        started_s = perf_counter()
        device = _model_device(self.dp)
        obs = {
            self._image_key: torch.from_numpy(images[None]).to(device),
            self._wrench_key: torch.from_numpy(wrench[None]).to(device),
        }
        with torch.inference_mode():
            output = self.dp.predict_action(obs)
        if not isinstance(output, Mapping) or "action_target" not in output:
            raise RuntimeError("DP predict_action must return action_target")
        action_target = _numpy_vector(
            output["action_target"],
            7,
            "DP action_target",
        )
        raw_action_chunk = _numpy_action_chunk(
            output.get("action", output["action_target"]),
        )
        action_chunk = _dp_execution_action_chunk(
            raw_action_chunk,
            model_horizon=getattr(self.dp, "horizon", None),
        )
        return _DPPrediction(
            action_target=action_target,
            raw_action_chunk=raw_action_chunk,
            action_chunk=action_chunk,
            elapsed_s=perf_counter() - started_s,
        )

    def _report_timing(self, latest_cycle_s: float) -> None:
        timing = self.config.timing
        if not timing.enabled:
            return
        now_s = perf_counter()
        interval_s = now_s - self._timing_started_s
        if interval_s < timing.report_interval_s:
            return
        cycles = max(self._timing_cycles, 1)
        control_hz = self._timing_cycles / interval_s
        pinn_ms = 1.0e3 * self._timing_pinn_s / cycles
        qp_ms = 1.0e3 * self._timing_qp_s / cycles
        ik_ms = 1.0e3 * self._timing_ik_s / cycles
        if self._timing_dp_s:
            dp_text = f"{1.0e3 * np.mean(self._timing_dp_s):.2f} ms"
        else:
            dp_text = "pending"
        backend_text = (
            f"IK_avg={ik_ms:.2f} ms"
            if not self._predictor_enabled
            else f"PINN_avg={pinn_ms:.2f} ms OSC-QP_avg={qp_ms:.2f} ms"
        )
        print(
            "[inference timing] "
            f"control={control_hz:.2f} Hz "
            f"cycle_latest={latest_cycle_s * 1.0e3:.2f} ms "
            f"{backend_text} "
            f"DP_avg={dp_text}",
            flush=True,
        )
        self._reset_timing(now_s)

    def _reset_timing(self, now_s: float | None = None) -> None:
        self._timing_started_s = perf_counter() if now_s is None else now_s
        self._timing_cycles = 0
        self._timing_pinn_s = 0.0
        self._timing_qp_s = 0.0
        self._timing_ik_s = 0.0
        self._timing_dp_s = []

    def _predict_wrenches(
        self,
        sample: InferenceInput,
        current_action_pose: np.ndarray,
    ) -> np.ndarray:
        if self._predictor_mode in _WORLD_MODEL_MODES:
            wrench = self._predict_world_model_wrenches(
                sample,
                current_action_pose,
            )
        else:
            wrench = self._predict_gru_wrenches(sample, current_action_pose)
        limits = np.asarray(
            [self.config.safety.maximum_target_force_n] * 3
            + [self.config.safety.maximum_target_moment_nm] * 3
        )
        wrench = np.clip(wrench, -limits, limits)
        return _fit_horizon(wrench, self.controller.config.horizon_steps)

    def _predict_gru_wrenches(
        self,
        sample: InferenceInput,
        current_action_pose: np.ndarray,
    ) -> np.ndarray:
        import torch

        device = _model_device(self.pinn)
        values = {
            "q": sample.q,
            "v": sample.dq,
            "dq": sample.dq,
            "a": sample.ddq,
            "ddq": sample.ddq,
            "tau": sample.tau,
            "action": self._action,
            "action_target": self._action,
            "u": self._action,
            "ee_pose": self._action,
            "wrench_ext": sample.wrench_ext,
        }
        active_inputs = tuple(getattr(self.pinn, "active_inputs", values.keys()))
        missing = sorted(set(active_inputs) - set(values))
        if missing:
            raise RuntimeError(f"PINN checkpoint requests unsupported inputs: {missing}")
        inputs = {}
        for key in active_inputs:
            value = torch.as_tensor(
                values[key], dtype=torch.float32, device=device
            )[None]
            inputs[key] = self._normalize_pinn_input(key, value)
        self._add_action_condition(inputs, device, current_action_pose)
        with torch.inference_mode():
            output, state = call_pinn(self.pinn, inputs, self._pinn_state)
        self._pinn_state = (
            self.pinn.detach_recurrent_state(state)
            if hasattr(self.pinn, "detach_recurrent_state")
            else state
        )
        return _numpy_wrench_trajectory(output)

    def _predict_world_model_wrenches(
        self,
        sample: InferenceInput,
        current_action_pose: np.ndarray,
    ) -> np.ndarray:
        del sample
        import torch

        device = _model_device(self.pinn)
        history = {
            key: np.stack(tuple(values), axis=0)
            for key, values in self._wm_history.items()
        }
        inputs = {
            key: self._normalize_pinn_input(
                key,
                torch.as_tensor(value, dtype=torch.float32, device=device)[None],
            )
            for key, value in history.items()
        }
        self._add_action_condition(inputs, device, current_action_pose)
        with torch.inference_mode():
            output = self.pinn.predict(inputs)
        if not isinstance(output, Mapping) or not isinstance(
            output.get("state_pred"), Mapping
        ):
            raise RuntimeError(
                f"{self._predictor_mode} predict() must return a state_pred mapping"
            )
        normalized_states = output["state_pred"]
        future = {}
        output_keys = (
            ("q", "tau")
            if self._predictor_mode in _Q_TAU_WORLD_MODEL_MODES
            else ("q", "v", "a", "tau")
        )
        physical_tensors = {}
        for key in output_keys:
            if key not in normalized_states:
                raise RuntimeError(
                    f"{self._predictor_mode} state_pred is missing {key!r}"
                )
            value = normalized_states[key]
            if value.ndim != 3 or value.shape != (
                1,
                self._pinn_future_horizon,
                7,
            ):
                raise RuntimeError(
                    f"{self._predictor_mode} state_pred[{key!r}] must have shape "
                    f"[1, {self._pinn_future_horizon}, 7], got {tuple(value.shape)}"
                )
            physical = self._denormalize_pinn_output(key, value)
            physical_tensors[key] = physical
            future[key] = physical[0].detach().cpu().numpy().astype(np.float64)
        if self._predictor_mode in _Q_TAU_WORLD_MODEL_MODES:
            reconstruct = getattr(self.pinn, "reconstruct_future_state", None)
            if not callable(reconstruct):
                raise RuntimeError(
                    f"{self._predictor_mode} checkpoint does not expose "
                    "reconstruct_future_state()"
                )
            q_history = torch.as_tensor(
                history["q"],
                dtype=physical_tensors["q"].dtype,
                device=device,
            )[None]
            reconstructed = reconstruct(
                q_history,
                physical_tensors["q"],
            )
            if not isinstance(reconstructed, Mapping):
                raise RuntimeError(
                    f"{self._predictor_mode} state reconstruction must return "
                    "a mapping"
                )
            for key in ("v", "a"):
                value = reconstructed.get(key)
                if value is None or value.shape != (
                    1,
                    self._pinn_future_horizon,
                    7,
                ):
                    shape = None if value is None else tuple(value.shape)
                    raise RuntimeError(
                        f"{self._predictor_mode} reconstructed "
                        f"{key!r} must have shape "
                        f"[1, {self._pinn_future_horizon}, 7], got {shape}"
                    )
                future[key] = (
                    value[0].detach().cpu().numpy().astype(np.float64)
                )
        if self._world_model_wrench_adapter is None:
            raise RuntimeError(
                f"{self._predictor_mode} wrench adapter is not initialized"
            )
        wrench = self._world_model_wrench_adapter.states_to_wrenches(
            {key: history[key] for key in ("q", "v", "a", "tau")},
            future,
        )
        if (
            self._predictor_mode in _Q_TAU_WORLD_MODEL_MODES
            and self._pinn_contact_gate_enabled
        ):
            probability = output.get("contact_probability")
            if probability is None or probability.shape != (
                1,
                self._pinn_future_horizon,
                1,
            ):
                shape = None if probability is None else tuple(probability.shape)
                raise RuntimeError(
                    f"contact-enabled {self._predictor_mode} must return "
                    "contact_probability with shape "
                    f"[1, {self._pinn_future_horizon}, 1], got {shape}"
                )
            probability_numpy = (
                probability[0, :, 0].detach().cpu().numpy().astype(np.float64)
            )
            contact = (
                probability_numpy
                >= self._pinn_contact_probability_threshold
            ).astype(np.float64)
            wrench = np.asarray(wrench, dtype=np.float64) * contact[:, None]
        return wrench

    def _add_action_condition(
        self,
        inputs: dict[str, Any],
        device: Any,
        current_action_pose: np.ndarray,
    ) -> None:
        if self._pinn_action_key is None:
            return
        import torch

        if self._action_chunk is None:
            raise RuntimeError("PINN action condition requested before DP action initialization")
        if self._pinn_action_condition_fill == "hold":
            if self._pinn_held_action is None:
                raise RuntimeError(
                    "held PINN action is unavailable before action initialization"
                )
            future_action = np.repeat(
                self._pinn_held_action[None],
                self._pinn_future_horizon,
                axis=0,
            )
        else:
            future_action = _fit_action_horizon(
                self._action_chunk,
                self._pinn_future_horizon,
            )
        future_tensor = torch.as_tensor(
            future_action,
            dtype=torch.float32,
            device=device,
        )[None]
        if self._pinn_action_condition_mode == "relative_pose":
            current_tensor = torch.as_tensor(
                _pose_to_action(current_action_pose),
                dtype=torch.float32,
                device=device,
            )[None]
            future_tensor = _relative_action_pose_torch(
                current_tensor,
                future_tensor,
            )
        if self._pinn_action_normalizer_key is not None:
            future_tensor = self._normalize_pinn_input(
                self._pinn_action_normalizer_key,
                future_tensor,
            )
        inputs[self._pinn_action_key] = future_tensor

    def _normalize_pinn_input(self, key: str, value: Any) -> Any:
        metadata = getattr(self.pinn, "_inference_normalizer", None)
        config = getattr(self.pinn, "_inference_checkpoint_config", {})
        if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
            return value
        normalize_keys = set(
            metadata.get("normalize_lowdim_keys")
            or config.get("dataloader", {}).get("normalize_lowdim_keys")
            or ()
        )
        if key not in normalize_keys:
            return value
        stats = metadata.get("stats", {}).get(key)
        if not isinstance(stats, Mapping):
            raise RuntimeError(f"PINN checkpoint normalizer is missing stats for {key!r}")
        mode = metadata.get("normalize_mode") or config.get("dataloader", {}).get(
            "normalize_mode"
        )
        eps = float(metadata.get("eps", 1.0e-6))
        if mode == "gaussian":
            mean = _stat_tensor(stats["mean"], value)
            std = _stat_tensor(stats["std"], value)
            return (value - mean) / (std + eps)
        if mode == "limit":
            minimum = _stat_tensor(stats["min"], value)
            maximum = _stat_tensor(stats["max"], value)
            return 2.0 * (value - minimum) / (maximum - minimum + eps) - 1.0
        if mode == "quantile":
            q01 = _stat_tensor(stats["q01"], value)
            q99 = _stat_tensor(stats["q99"], value)
            return (2.0 * (value - q01) / (q99 - q01 + eps) - 1.0).clamp(-1, 1)
        raise RuntimeError(f"unsupported PINN normalization mode: {mode!r}")

    def _denormalize_pinn_output(self, key: str, value: Any) -> Any:
        metadata = getattr(self.pinn, "_inference_normalizer", None)
        config = getattr(self.pinn, "_inference_checkpoint_config", {})
        if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
            return value
        normalize_keys = set(
            metadata.get("normalize_lowdim_keys")
            or config.get("dataloader", {}).get("normalize_lowdim_keys")
            or ()
        )
        if key not in normalize_keys:
            return value
        stats = metadata.get("stats", {}).get(key)
        if not isinstance(stats, Mapping):
            raise RuntimeError(f"PINN checkpoint normalizer is missing stats for {key!r}")
        mode = metadata.get("normalize_mode") or config.get("dataloader", {}).get(
            "normalize_mode"
        )
        eps = float(metadata.get("eps", 1.0e-6))
        if mode == "gaussian":
            mean = _stat_tensor(stats["mean"], value)
            std = _stat_tensor(stats["std"], value)
            return value * (std + eps) + mean
        if mode == "limit":
            minimum = _stat_tensor(stats["min"], value)
            maximum = _stat_tensor(stats["max"], value)
            return (value + 1.0) * (maximum - minimum + eps) / 2.0 + minimum
        if mode == "quantile":
            q01 = _stat_tensor(stats["q01"], value)
            q99 = _stat_tensor(stats["q99"], value)
            return (value + 1.0) * (q99 - q01 + eps) / 2.0 + q01
        raise RuntimeError(f"unsupported PINN normalization mode: {mode!r}")

    def _current_action_pose(
        self,
        q: np.ndarray,
        current_control_pose: np.ndarray,
    ) -> np.ndarray:
        if self._action_frame_name == self._control_frame_name:
            return current_control_pose.copy()
        frame_pose = getattr(self.controller.model, "frame_pose", None)
        if not callable(frame_pose):
            raise RuntimeError(
                "PINN action frame differs from OSC control frame, but the dynamics "
                "model does not expose frame_pose()"
            )
        return np.asarray(
            frame_pose(q, self._action_frame_name),
            dtype=np.float64,
        )

    def _action_poses_to_control(
        self,
        action_poses: np.ndarray,
        q: np.ndarray,
        current_action_pose: np.ndarray,
        current_control_pose: np.ndarray,
    ) -> np.ndarray:
        del q
        if self._action_frame_name == self._control_frame_name:
            return action_poses.copy()
        if self._action_to_control_transform is None:
            self._action_to_control_transform = (
                np.linalg.inv(current_action_pose) @ current_control_pose
            )
        return np.einsum(
            "nij,jk->nik",
            action_poses,
            self._action_to_control_transform,
        )

    def _safe_action_chunk(
        self,
        actions: np.ndarray,
        current_pose: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [self._safe_action(action, current_pose) for action in actions],
            axis=0,
        )

    def _safe_action(self, action: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        action_value = _numpy_vector(action, 7, "action")
        target = _action_to_pose(action_value)
        delta = target[:3, 3] - current_pose[:3, 3]
        norm = np.linalg.norm(delta)
        limit = self.config.safety.maximum_action_translation_step_m
        if norm > limit:
            target[:3, 3] = current_pose[:3, 3] + delta * (limit / norm)
        relative = Rotation.from_matrix(
            target[:3, :3] @ current_pose[:3, :3].T
        ).as_rotvec()
        angle = np.linalg.norm(relative)
        rotation_limit = self.config.safety.maximum_action_rotation_step_rad
        if angle > rotation_limit:
            target[:3, :3] = (
                Rotation.from_rotvec(relative * (rotation_limit / angle)).as_matrix()
                @ current_pose[:3, :3]
            )
        safe_action = _pose_to_action(target)
        # A rotation matrix loses the quaternion double-cover sign. Preserve the
        # DP sign so absolute-pose PINN conditioning stays in its training domain.
        if np.dot(safe_action[3:], action_value[3:]) < 0.0:
            safe_action[3:] *= -1.0
        return safe_action

    def _clip_tau(self, tau: np.ndarray) -> np.ndarray:
        limit = np.asarray(
            self.config.safety.maximum_command_torque_nm, dtype=np.float64
        )
        if limit.ndim == 0:
            limit = np.repeat(limit, 7)
        if limit.shape != (7,) or np.any(limit <= 0):
            raise ValueError("maximum_command_torque_nm must be positive scalar or 7-vector")
        return np.clip(tau, -limit, limit)

    @staticmethod
    def _validate_input(sample: InferenceInput) -> None:
        for name, size in (("q", 7), ("dq", 7), ("ddq", 7), ("tau", 7), ("wrench_ext", 6)):
            _numpy_vector(getattr(sample, name), size, name)
        if not np.isfinite(sample.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if (
            sample.image_timestamp_s is not None
            and not np.isfinite(sample.image_timestamp_s)
        ):
            raise ValueError("image_timestamp_s must be finite when provided")
        if sample.wrench_to_control_rotation is not None:
            rotation = np.asarray(sample.wrench_to_control_rotation, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("wrench_to_control_rotation must be a finite 3x3 matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
                raise ValueError("wrench_to_control_rotation must be orthonormal")


def _dp_model_overrides(config: InferenceConfig) -> dict[str, Any]:
    scheduler_targets = {
        "ddim": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
        "ddpm": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
    }
    overrides: dict[str, Any] = {
        "noise_scheduler._target_": scheduler_targets[config.dp_sampling.method],
        "num_inference_steps": config.dp_sampling.num_inference_steps,
    }
    path = config.dp_checkpoint.dino_model_path
    if path is None:
        return overrides
    if not path.is_dir():
        raise ValueError(f"DP DINO model directory does not exist: {path}")
    overrides["dino_model_name_or_path"] = str(path)
    return overrides


def _model_device(model: Any) -> Any:
    import torch

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _numpy_vector(value: Any, size: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim > 1 and result.shape[0] == 1:
        result = result[0]
    result = result.reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size}-vector, got {result.shape}")
    return result


def _numpy_wrench_trajectory(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    elif result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or result.shape[1] != 6 or not np.all(np.isfinite(result)):
        raise ValueError(
            f"PINN target wrench must have shape [F,6] or [1,F,6], got {result.shape}"
        )
    return result


def _numpy_action_chunk(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    elif result.ndim == 2 and result.shape == (1, 7):
        pass
    elif result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or result.shape[1] != 7 or not np.all(np.isfinite(result)):
        raise ValueError(
            f"DP future action must have shape [T,7] or [1,T,7], got {result.shape}"
        )
    for action in result:
        if np.linalg.norm(action[3:]) < 1.0e-8:
            raise ValueError("DP future action contains a zero quaternion")
    return result


def _nearest_timestamp_indices(
    sorted_timestamps: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    source = np.asarray(sorted_timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if source.size == 0 or np.any(np.diff(source) <= 0.0):
        raise ValueError("observation timestamps must be non-empty and increasing")
    right = np.searchsorted(source, values, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - values) < np.abs(source[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def _fit_horizon(wrenches: np.ndarray, horizon: int) -> np.ndarray:
    if len(wrenches) >= horizon:
        return wrenches[:horizon].copy()
    padding = np.repeat(wrenches[-1:], horizon - len(wrenches), axis=0)
    return np.concatenate((wrenches, padding), axis=0)


def _fit_action_horizon(actions: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("action horizon must be positive")
    values = _numpy_action_chunk(actions)
    if len(values) >= horizon:
        return values[:horizon].copy()
    padding = np.repeat(values[-1:], horizon - len(values), axis=0)
    return np.concatenate((values, padding), axis=0)


def _fit_pose_horizon(poses: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"pose trajectory must have shape [T,4,4], got {values.shape}")
    if len(values) >= horizon:
        return values[:horizon].copy()
    padding = np.repeat(values[-1:], horizon - len(values), axis=0)
    return np.concatenate((values, padding), axis=0)


def _minimum_jerk_action_plan(
    start_action: np.ndarray,
    target_action: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Generate a C2 minimum-jerk Cartesian pose trajectory to one target."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("minimum-jerk trajectory steps must be a positive integer")
    start = _numpy_vector(start_action, 7, "minimum-jerk action start")
    target = _numpy_vector(target_action, 7, "minimum-jerk action target")
    start_quaternion = start[3:].copy()
    target_quaternion = target[3:].copy()
    start_norm = np.linalg.norm(start_quaternion)
    target_norm = np.linalg.norm(target_quaternion)
    if start_norm < 1.0e-8 or target_norm < 1.0e-8:
        raise ValueError("minimum-jerk trajectory requires non-zero quaternions")
    start_quaternion /= start_norm
    target_quaternion /= target_norm
    # Flip only the equivalent start representation so the final waypoint keeps
    # the exact DP quaternion hemisphere while following the shorter rotation.
    if np.dot(start_quaternion, target_quaternion) < 0.0:
        start_quaternion *= -1.0

    progress = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    time_scale = (
        10.0 * progress**3
        - 15.0 * progress**4
        + 6.0 * progress**5
    )
    position = start[:3] + time_scale[:, None] * (target[:3] - start[:3])
    key_rotations = Rotation.from_quat(
        np.stack((start_quaternion, target_quaternion), axis=0)
    )
    quaternion = Slerp(
        np.asarray([0.0, 1.0], dtype=np.float64),
        key_rotations,
    )(time_scale).as_quat()
    quaternion[0] = start_quaternion
    quaternion[-1] = target_quaternion
    return np.concatenate((position, quaternion), axis=1)


def _mean_pose_chunk(actions: np.ndarray) -> np.ndarray:
    values = _numpy_action_chunk(actions)
    position = values[:, :3].mean(axis=0)
    quaternions = values[:, 3:].copy()
    reference = quaternions[0]
    signs = np.where((quaternions @ reference)[:, None] < 0.0, -1.0, 1.0)
    quaternion = (quaternions * signs).mean(axis=0)
    norm = np.linalg.norm(quaternion)
    quaternion = reference if norm < 1.0e-8 else quaternion / norm
    return np.concatenate((position, quaternion))




def _dp_execution_action_chunk(
    actions: np.ndarray,
    *,
    model_horizon: Any = None,
) -> np.ndarray:
    """Reduce the current DP's 8x8 high-rate output to seven row targets.

    The force-aware DP predicts eight future 10 Hz rows, and each row contains
    eight high-rate poses. Average each row with quaternion-safe pose averaging,
    then discard row zero to preserve the previous DP execution convention.
    Legacy checkpoints with a non-64 horizon keep their existing action chunk.
    """
    values = _numpy_action_chunk(actions)
    try:
        horizon = int(model_horizon)
    except (TypeError, ValueError):
        horizon = len(values)
    if horizon != 64:
        return values.copy()
    if values.shape != (64, 7):
        raise ValueError(
            "64-step DP must return action with shape [64,7] or [1,64,7], "
            f"got {values.shape}"
        )
    rows = values.reshape(8, 8, 7)
    row_targets = np.stack([_mean_pose_chunk(row) for row in rows], axis=0)
    return row_targets[1:].copy()


def _select_action_chunk(actions: np.ndarray, mode: str) -> np.ndarray:
    values = _numpy_action_chunk(actions)
    if mode == "first":
        return values[0].copy()
    if mode == "mean":
        return _mean_pose_chunk(values)
    if mode == "last":
        return values[-1].copy()
    if mode == "middle":
        return values[3].copy()
    raise ValueError(f"unsupported action chunk mode: {mode!r}")


def _relative_action_pose_torch(current_pose: Any, future_pose: Any) -> Any:
    """Match PINN training: world-frame delta xyz and q_current^-1 * q_future."""
    import torch
    import torch.nn.functional as functional

    if current_pose.ndim != 2 or current_pose.shape[-1] != 7:
        raise ValueError("current pose must have shape [B,7]")
    if future_pose.ndim != 3 or future_pose.shape[-1] != 7:
        raise ValueError("future pose must have shape [B,F,7]")
    if current_pose.shape[0] != future_pose.shape[0]:
        raise ValueError("current and future poses must share batch size")
    relative_position = future_pose[..., :3] - current_pose[:, None, :3]
    current_quaternion = functional.normalize(current_pose[:, 3:], dim=-1, eps=1.0e-8)
    future_quaternion = functional.normalize(future_pose[..., 3:], dim=-1, eps=1.0e-8)
    inverse_current = current_quaternion.clone()
    inverse_current[..., :3] = -inverse_current[..., :3]
    inverse_current = inverse_current[:, None].expand_as(future_quaternion)
    ax, ay, az, aw = inverse_current.unbind(dim=-1)
    bx, by, bz, bw = future_quaternion.unbind(dim=-1)
    relative_quaternion = torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )
    relative_quaternion = functional.normalize(
        relative_quaternion,
        dim=-1,
        eps=1.0e-8,
    )
    sign = torch.where(
        relative_quaternion[..., 3:4] < 0.0,
        -torch.ones_like(relative_quaternion[..., 3:4]),
        torch.ones_like(relative_quaternion[..., 3:4]),
    )
    return torch.cat((relative_position, relative_quaternion * sign), dim=-1)


def _stat_tensor(value: Any, like: Any) -> Any:
    import torch

    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _rotate_wrenches(
    wrenches: np.ndarray,
    rotation: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(wrenches, dtype=np.float64)
    if rotation is None:
        return values.copy()
    matrix = np.asarray(rotation, dtype=np.float64)
    result = values.copy()
    result[:, :3] = values[:, :3] @ matrix.T
    result[:, 3:] = values[:, 3:] @ matrix.T
    return result


def _action_to_pose(action: np.ndarray) -> np.ndarray:
    value = _numpy_vector(action, 7, "action")
    quaternion = value[3:]
    if np.linalg.norm(quaternion) < 1.0e-8:
        raise ValueError("action quaternion cannot be zero")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = value[:3]
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return pose


def _pose_to_action(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("pose must have shape [4,4]")
    return np.concatenate((value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat()))


def _format_action_vector(value: np.ndarray) -> str:
    return np.array2string(
        np.asarray(value, dtype=np.float64).reshape(-1),
        formatter={"float_kind": lambda item: f"{item:.6f}"},
        separator=", ",
    )


def _format_action_delta(current_action: np.ndarray, predicted_action: np.ndarray) -> str:
    """Format a geometric action delta as xyz metres plus rotation-vector radians."""
    current_pose = _action_to_pose(current_action)
    predicted_pose = _action_to_pose(predicted_action)
    translation = predicted_pose[:3, 3] - current_pose[:3, 3]
    rotation = Rotation.from_matrix(
        predicted_pose[:3, :3] @ current_pose[:3, :3].T
    ).as_rotvec()
    return _format_action_vector(np.concatenate((translation, rotation)))
