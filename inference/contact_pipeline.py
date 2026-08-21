"""Dedicated contact world-model inference pipeline.

The contact WM contract is intentionally separate from the legacy wrench
pipeline.  Its model input is only ``q``/``tau`` history plus the 8-token,
21-dimensional action condition.  Predicted q is differentiated causally at
the output boundary for MIT control; dq is never fed back into the model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from inference.pipeline import (
    InferenceInput,
    InferenceOutput,
    NeroInferencePipeline,
    _fit_action_horizon,
    _fit_horizon,
    _fit_pose_horizon,
    _numpy_action_chunk,
    _pose_to_action,
    _rotate_wrenches,
    _synchronize_model,
    _CONTACT_WORLD_MODEL_MODES,
)
from nero_collection.control import OSCTargetTrajectory


class ContactWMInferencePipeline(NeroInferencePipeline):
    """Run a q/tau contact WM with MIT, OSC-QP, q, or tau execution."""

    def __init__(self, config, **kwargs) -> None:
        # if not config.predictor.enabled:
        #     raise ValueError("contact WM inference requires predictor.enabled=true")
        predictor_mode = str(config.predictor.mode).strip().lower().replace("-", "_")
        if predictor_mode not in _CONTACT_WORLD_MODEL_MODES:
            raise ValueError(
                "ContactWMInferencePipeline requires a contact_world_model mode"
            )
        super().__init__(config, _allow_contact_world_model=True, **kwargs)
        checkpoint_config = getattr(self.pinn, "_inference_checkpoint_config", {})
        checkpoint_data = (
            checkpoint_config.get("dataloader") or {}
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        configured_history = getattr(self.pinn, "history_horizon", None)
        if configured_history is None:
            configured_history = checkpoint_data.get("state_history_horizon", 50)
        self._contact_history_horizon = int(configured_history)
        if self._contact_history_horizon < 1:
            raise ValueError("contact WM history_horizon must be positive")
        self._contact_history = {
            "q": deque(maxlen=self._contact_history_horizon),
            "tau": deque(maxlen=self._contact_history_horizon),
        }
        self._contact_previous: tuple[float, dict[str, np.ndarray]] | None = None
        self._contact_next_sample_s: float | None = None
        self._contact_sampling_dt_s = self._wm_sampling_dt_s or 0.01
        checkpoint_config = getattr(self.pinn, "_inference_checkpoint_config", {})
        model_config = (
            checkpoint_config.get("model") or {}
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        data_config = (
            checkpoint_config.get("dataloader") or {}
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        self._pinn_action_key = str(
            model_config.get("action_key", "target_relative_pose")
        )
        self._pinn_action_normalizer_key = str(
            model_config.get("action_normalizer_key", "target_relative_pose")
        )
        action_horizon = getattr(self.pinn, "action_condition_horizon", None)
        if action_horizon is None and isinstance(data_config, Mapping):
            action_horizon = data_config.get("action_condition_horizon", 8)
        self._pinn_action_horizon = int(action_horizon or 8)
        configured_features = (
            data_config.get("action_condition_features")
            if isinstance(data_config, Mapping)
            else None
        )
        action_dim = model_config.get("action_dim", 7)
        if configured_features is None and action_dim is not None and int(action_dim) == 21:
            configured_features = (
                "absolute_pose",
                "current_ee_pose",
                "relative_pose",
            )
        self._pinn_action_condition_features = tuple(
            str(value).lower()
            for value in (configured_features or ("relative_pose",))
        )
        mode = str(config.execution.mode).strip().lower().replace("-", "_")
        if mode not in {"mit", "osc_qp", "q", "tau"}:
            raise ValueError("contact execution mode must be mit, osc_qp, q, or tau")
        self._contact_execution_mode = mode

    def reset(self) -> None:
        super().reset()
        for values in self._contact_history.values():
            values.clear()
        self._contact_previous = None
        self._contact_next_sample_s = None

    def _append_world_model_observation(self, sample: InferenceInput) -> None:
        self._append_contact_observation_values(
            float(sample.timestamp_s), sample.q, sample.tau
        )

    def _append_world_model_observation_values(
        self,
        timestamp_s: float,
        values: Mapping[str, np.ndarray],
    ) -> None:
        # Keep the inherited public helper useful for the continuous CAN ring,
        # but intentionally discard dq/ddq/wrench at this contract boundary.
        if "q" not in values or "tau" not in values:
            raise ValueError("contact WM observation requires q and tau")
        self._append_contact_observation_values(
            timestamp_s, values["q"], values["tau"]
        )

    def _append_contact_observation_values(
        self,
        timestamp_s: float,
        q: Any,
        tau: Any,
    ) -> None:
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("contact WM timestamp must be finite")
        q_value = np.asarray(q, dtype=np.float32).reshape(-1)
        tau_value = np.asarray(tau, dtype=np.float32).reshape(-1)
        if (
            q_value.shape != (7,)
            or tau_value.shape != (7,)
            or not np.isfinite(q_value).all()
            or not np.isfinite(tau_value).all()
        ):
            raise ValueError("contact WM q and tau must be finite seven-vectors")
        current = {"q": q_value.copy(), "tau": tau_value.copy()}
        previous = self._contact_previous
        if previous is None:
            self._append_contact_values(current)
            self._contact_previous = (timestamp_s, current)
            self._contact_next_sample_s = timestamp_s + self._contact_sampling_dt_s
            return
        previous_s, previous_values = previous
        if timestamp_s <= previous_s:
            if timestamp_s == previous_s:
                return
            raise ValueError("contact WM observation timestamps must increase")
        assert self._contact_next_sample_s is not None
        while self._contact_next_sample_s <= timestamp_s + 1.0e-9:
            alpha = (self._contact_next_sample_s - previous_s) / (
                timestamp_s - previous_s
            )
            interpolated = {
                key: previous_values[key]
                + np.float32(alpha) * (current[key] - previous_values[key])
                for key in current
            }
            self._append_contact_values(interpolated)
            self._contact_next_sample_s += self._contact_sampling_dt_s
        self._contact_previous = (timestamp_s, current)

    def _append_contact_values(self, values: Mapping[str, np.ndarray]) -> None:
        for key in ("q", "tau"):
            history = self._contact_history[key]
            value = np.asarray(values[key], dtype=np.float32).copy()
            history.append(value)
            while len(history) < history.maxlen:
                history.appendleft(history[0].copy())

    def step(self, sample: InferenceInput) -> InferenceOutput:
        cycle_started = perf_counter()
        self._last_dp_inference_time_s = None
        self._last_wm_inference_time_s = None
        self._validate_input(sample)
        open_loop = self.config.predictor.inference_mode == "open_loop"
        plan_completed = self._advance_execution_plan(sample.timestamp_s)
        if open_loop and plan_completed:
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
            sample.q, current_control_pose
        )
        dp_updated = self._update_dp_execution(
            sample.timestamp_s, current_action_pose, sample.q
        )
        if self._action is None:
            self._set_idle_action(current_action_pose, sample.q)

        pinn_started = perf_counter()
        current_wm_pose = self._current_wm_pose(
            sample.q,
            current_action_pose,
            current_control_pose,
        )
        reference = self._predict_contact_reference(current_wm_pose)
        output = self._execute_reference(sample, current_control_pose, reference, dp_updated)
        self._timing_pinn_s += perf_counter() - pinn_started
        self._timing_cycles += 1
        self._report_timing(perf_counter() - cycle_started)
        return output

    def _predict_contact_reference(
        self,
        current_action_pose: np.ndarray,
    ) -> Any:
        import torch

        if any(len(values) < 1 for values in self._contact_history.values()):
            raise RuntimeError("contact WM history is empty")
        device = self._device()
        history = {
            key: np.stack(tuple(values), axis=0)
            for key, values in self._contact_history.items()
        }
        inputs = {
            key: self._normalize_pinn_input(
                key,
                torch.as_tensor(value, dtype=torch.float32, device=device)[None],
            )
            for key, value in history.items()
        }
        self._add_contact_action_condition(inputs, device, current_action_pose)
        _synchronize_model(self.pinn)
        wm_started_s = perf_counter()
        with torch.inference_mode():
            output = self.pinn.predict(inputs)
        _synchronize_model(self.pinn)
        self._record_wm_inference_timing(perf_counter() - wm_started_s)
        if not isinstance(output, Mapping) or not isinstance(
            output.get("state_pred"), Mapping
        ):
            raise RuntimeError("contact WM predict() must return state_pred mapping")
        state_pred = output["state_pred"]
        future_tensors: dict[str, Any] = {}
        for key in ("q", "tau"):
            value = state_pred.get(key)
            if value is not None and not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            if value is None or value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 7:
                shape = None if value is None else tuple(value.shape)
                raise RuntimeError(
                    f"contact WM state_pred[{key!r}] must have shape [1,T,7], got {shape}"
                )
            future_tensors[key] = self._denormalize_pinn_output(key, value)
            if not torch.isfinite(future_tensors[key]).all():
                raise RuntimeError(f"contact WM state_pred[{key!r}] is non-finite")
        q_tensor = future_tensors["q"]
        if future_tensors["tau"].shape[1] != q_tensor.shape[1]:
            raise RuntimeError("contact WM q and tau future horizons differ")
        reconstructed = self._reconstruct_future_q_state(
            history["q"], q_tensor, device
        )
        if not isinstance(reconstructed, Mapping):
            raise RuntimeError("contact WM q state reconstruction returned no mapping")
        future = {
            "q": q_tensor[0].detach().cpu().numpy().astype(np.float64),
            "tau": future_tensors["tau"][0].detach().cpu().numpy().astype(np.float64),
            "v": self._require_reconstructed(
                reconstructed, "v", q_tensor.shape[1]
            ),
            "a": self._require_reconstructed(
                reconstructed, "a", q_tensor.shape[1]
            ),
        }
        contact = state_pred.get("contact_state")
        if contact is None:
            contact = output.get("contact_state_pred")
        contact_numpy = None
        if contact is not None:
            if hasattr(contact, "detach"):
                contact_numpy = contact[0].detach().cpu().numpy().astype(np.float64)
            else:
                contact_numpy = np.asarray(contact, dtype=np.float64)
                if contact_numpy.ndim == 3 and contact_numpy.shape[0] == 1:
                    contact_numpy = contact_numpy[0]
            if contact_numpy.ndim == 1:
                contact_numpy = contact_numpy[:, None]
            if contact_numpy.shape != (q_tensor.shape[1], 1):
                raise RuntimeError(
                    "contact WM contact_state must have shape [T,1]"
                )
        return _ContactReference(
            q=future["q"],
            dq=future["v"],
            ddq=future["a"],
            tau=future["tau"],
            contact_state=contact_numpy,
        )

    def _device(self):
        import torch

        try:
            return next(self.pinn.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _reconstruct_future_q_state(self, q_history_numpy, q_future, device):
        """Use the same causal estimator used by contact-WM training."""
        import torch

        try:
            from model.pinn_model.causal_state import (
                CausalStateEstimatorConfig,
                future_joint_state_from_position,
            )
        except ImportError:
            # A model loaded from the PINN package uses the authoritative
            # implementation above.  Keep a numerically identical small
            # fallback for injected/test models so the inference package does
            # not acquire a hard import dependency just to differentiate q.
            return self._fallback_future_q_state(q_history_numpy, q_future, device)
        checkpoint_config = getattr(self.pinn, "_inference_checkpoint_config", {})
        estimator_config = CausalStateEstimatorConfig.from_model_config(
            checkpoint_config
        )
        q_history = torch.as_tensor(
            q_history_numpy, dtype=q_future.dtype, device=device
        )[None]
        return future_joint_state_from_position(
            q_history, q_future, estimator_config
        )

    def _fallback_future_q_state(self, q_history_numpy, q_future, device):
        import torch

        checkpoint_config = getattr(self.pinn, "_inference_checkpoint_config", {})
        model_config = (
            checkpoint_config.get("model") or {}
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        estimator = (
            model_config.get("state_estimator") or {}
            if isinstance(model_config, Mapping)
            else {}
        )
        loss_config = (
            checkpoint_config.get("loss") or {}
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        dt = float(
            estimator.get(
                "sampling_dt",
                loss_config.get("sampling_dt", self._contact_sampling_dt_s),
            )
        )
        window_size = max(int(estimator.get("q_mean_window_samples", 10)), 1)
        q_cutoff = estimator.get("q_lowpass_cutoff_hz", 10.0)
        dq_cutoff = estimator.get("dq_lowpass_cutoff_hz", 6.0)
        ddq_cutoff = estimator.get("ddq_lowpass_cutoff_hz", 3.0)

        q_history = torch.as_tensor(
            q_history_numpy, dtype=q_future.dtype, device=device
        )[None]
        q = torch.cat((q_history, q_future), dim=1)

        def alpha(cutoff):
            if cutoff is None:
                return None
            return 1.0 - torch.exp(
                q.new_tensor(-2.0 * math.pi * float(cutoff) * dt)
            )

        q_alpha, dq_alpha, ddq_alpha = (
            alpha(q_cutoff),
            alpha(dq_cutoff),
            alpha(ddq_cutoff),
        )
        q_window = [q[:, 0]] * window_size
        q_window_sum = q[:, 0] * window_size
        previous_q = q[:, 0]
        previous_dq = torch.zeros_like(previous_q)
        previous_ddq = torch.zeros_like(previous_q)
        dq_values = [previous_dq]
        ddq_values = [previous_ddq]

        def one_pole(value, previous, filter_alpha):
            if filter_alpha is None:
                return value
            return filter_alpha * value + (1.0 - filter_alpha) * previous

        for index in range(1, q.shape[1]):
            q_window_sum = q_window_sum - q_window[0] + q[:, index]
            q_window = q_window[1:] + [q[:, index]]
            q_mean = q_window_sum / window_size
            q_filtered = one_pole(q_mean, previous_q, q_alpha)
            dq_filtered = one_pole(
                (q_filtered - previous_q) / dt, previous_dq, dq_alpha
            )
            ddq_filtered = one_pole(
                (dq_filtered - previous_dq) / dt, previous_ddq, ddq_alpha
            )
            dq_values.append(dq_filtered)
            ddq_values.append(ddq_filtered)
            previous_q, previous_dq, previous_ddq = (
                q_filtered,
                dq_filtered,
                ddq_filtered,
            )
        future_horizon = q_future.shape[1]
        return {
            "q": q_future,
            "v": torch.stack(dq_values, dim=1)[:, -future_horizon:],
            "a": torch.stack(ddq_values, dim=1)[:, -future_horizon:],
        }

    @staticmethod
    def _require_reconstructed(
        mapping: Mapping[str, Any], key: str, horizon: int
    ) -> np.ndarray:
        value = mapping.get(key)
        if value is None:
            raise RuntimeError(f"contact WM reconstruction is missing {key!r}")
        if hasattr(value, "detach"):
            value = value[0].detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
        if result.shape != (horizon, 7) or not np.isfinite(result).all():
            raise RuntimeError(
                f"contact WM reconstructed {key!r} must have shape [{horizon},7]"
            )
        return result

    def _contact_action_chunk(
        self, current_action_pose: np.ndarray | None = None
    ) -> np.ndarray:
        if self._dp_action_chunk is None:
            if self._action is None:
                raise RuntimeError("contact WM action condition requested before DP action")
            values = _fit_action_horizon(
                self._action[None],
                self._pinn_action_horizon or 8,
                require_quaternion=self._dp_action_type == "eepose",
            )
            values = self._actions_for_wm(values)
            return (
                self._safe_action_chunk(values, current_action_pose)
                if current_action_pose is not None
                and self._dp_action_type == "eepose"
                else values
            )
        values = _numpy_action_chunk(
            self._dp_action_chunk,
            require_quaternion=self._dp_action_type == "eepose",
        )
        values = self._actions_for_wm(values)
        horizon = self._pinn_action_horizon or 8
        if values.shape[0] > horizon and values.shape[0] % horizon == 0:
            values = values.reshape(horizon, values.shape[0] // horizon, 7)
            values = np.stack(
                [_mean_pose(row) for row in values], axis=0
            )
        values = _fit_action_horizon(
            values,
            horizon,
            require_quaternion=True,
        )
        return (
            self._safe_action_chunk(values, current_action_pose)
            if current_action_pose is not None
            and self._dp_action_type == "eepose"
            else values
        )

    def _add_contact_action_condition(self, inputs, device, current_action_pose) -> None:
        import torch
        import torch.nn.functional as functional

        if self._action is None:
            raise RuntimeError("contact WM action condition requested before DP action")
        targets = torch.as_tensor(
            self._contact_action_chunk(current_action_pose),
            dtype=torch.float32,
            device=device,
        )[None]
        current = torch.as_tensor(
            _pose_to_action(current_action_pose), dtype=torch.float32, device=device
        )[None]
        current = current[:, None, :].expand(-1, targets.shape[1], -1)
        current_q = functional.normalize(current[..., 3:], dim=-1)
        target_q = functional.normalize(targets[..., 3:], dim=-1)
        current_pose = torch.cat((current[..., :3], current_q), dim=-1)
        absolute_pose = torch.cat((targets[..., :3], target_q), dim=-1)
        relative_pose = _contact_relative_pose_torch(current_pose, absolute_pose)
        values = {
            "absolute_pose": absolute_pose,
            "current_ee_pose": current_pose,
            "relative_pose": relative_pose,
        }
        try:
            condition = torch.cat(
                [values[name] for name in self._pinn_action_condition_features],
                dim=-1,
            )
        except KeyError as exc:
            raise RuntimeError(f"unsupported contact action feature {exc.args[0]!r}") from exc
        expected_action_dim = getattr(self.pinn, "action_dim", None)
        if expected_action_dim is not None and condition.shape[-1] != int(expected_action_dim):
            raise RuntimeError(
                "contact WM action condition width does not match checkpoint: "
                f"{condition.shape[-1]} != {int(expected_action_dim)}"
            )
        key = self._pinn_action_key or "target_relative_pose"
        if self._pinn_action_normalizer_key is not None:
            condition = self._normalize_pinn_input(
                self._pinn_action_normalizer_key, condition
            )
        inputs[key] = condition

    def _execute_reference(
        self,
        sample: InferenceInput,
        current_control_pose: np.ndarray,
        reference: Any,
        dp_updated: bool,
    ) -> InferenceOutput:
        mode = self._contact_execution_mode
        horizon = self.controller.config.horizon_steps
        q_future = np.asarray(reference.q, dtype=np.float64)
        dq_future = np.asarray(reference.dq, dtype=np.float64)
        ddq_future = np.asarray(reference.ddq, dtype=np.float64)
        tau_future = np.asarray(reference.tau, dtype=np.float64)
        dq_future = self._limit_mit_velocity(dq_future)
        q_ref = self._fit_joint(q_future, horizon)
        dq_ref = self._fit_joint(dq_future, horizon)
        ddq_ref = self._fit_joint(ddq_future, horizon)
        tau_ref = self._fit_joint(tau_future, horizon)
        q_command = self._safe_joint_position_command(sample.q, q_ref[0])
        q_command_trajectory = q_ref.copy()
        q_command_trajectory[0] = q_command
        tau_ff = (
            self._filtered_tau(tau_ref[0], sample.tau)
            if mode in {"mit", "tau"}
            else self._clip_tau(tau_ref[0])
        )
        contact_state = (
            None
            if reference.contact_state is None
            else reference.contact_state[0].copy()
        )
        common = dict(
            action_target=(self._action.copy() if self._action is not None else np.zeros(7)),
            dp_action_chunk=(
                None
                if self._dp_action_chunk is None
                else self._dp_action_chunk.copy()
            ),
            target_wrench=np.zeros(6, dtype=np.float64),
            qp_result=None,
            ik_result=None,
            dp_updated=dp_updated,
            pinn_updated=True,
            dp_inference_time_s=self._last_dp_inference_time_s,
            wm_inference_time_s=self._last_wm_inference_time_s,
            joint_position_target=q_command.copy(),
            joint_velocity_target=dq_ref[0].copy(),
            torque_target=tau_ff.copy(),
            mit_kp=None,
            mit_kd=None,
            contact_state=contact_state,
            control_mode=mode,
            joint_position_trajectory=q_future.copy(),
            joint_velocity_trajectory=dq_future.copy(),
            joint_acceleration_trajectory=ddq_future.copy(),
            torque_trajectory=tau_future.copy(),
        )
        if mode == "q":
            zeros = np.zeros(7, dtype=np.float64)
            return InferenceOutput(
                tau_command=zeros.copy(),
                tau_unfiltered=zeros.copy(),
                joint_position_command=q_command,
                **common,
            )
        if mode == "mit":
            kp = np.asarray(self.config.execution.mit_kp, dtype=np.float64)
            kd = np.asarray(self.config.execution.mit_kd, dtype=np.float64)
            raw_feedback = kp * (q_command - sample.q) + kd * (
                dq_ref[0] - sample.dq
            )
            feedback = raw_feedback.copy()
            feedback_limit = self.config.execution.mit_feedback_torque_limit
            if feedback_limit is not None:
                limit = np.asarray(feedback_limit, dtype=np.float64)
                if limit.ndim == 0:
                    limit = np.repeat(limit, 7)
                feedback = np.clip(feedback, -limit, limit)
            total = self._clip_tau(tau_ff + feedback)
            # The arm firmware evaluates kp*(q_des-q)+kd*(dq_des-dq).  Scale
            # both gains together so the physical feedback it computes at the
            # current sample equals the bounded software result.  This keeps
            # q/dq targets intact while making the configured limits effective
            # on the actual move_mit command.
            bounded_feedback = total - tau_ff
            gain_scale = np.ones(7, dtype=np.float64)
            nonzero = np.abs(raw_feedback) > 1.0e-12
            gain_scale[nonzero] = bounded_feedback[nonzero] / raw_feedback[nonzero]
            gain_scale = np.clip(gain_scale, 0.0, 1.0)
            effective_kp = kp * gain_scale
            effective_kd = kd * gain_scale
            return InferenceOutput(
                tau_command=total,
                tau_unfiltered=total.copy(),
                joint_position_command=None,
                **{
                    **common,
                    "mit_kp": effective_kp,
                    "mit_kd": effective_kd,
                },
            )
        if mode == "tau":
            return InferenceOutput(
                tau_command=tau_ff.copy(),
                tau_unfiltered=tau_ff.copy(),
                joint_position_command=None,
                **common,
            )

        self._update_qp_timestep(sample.timestamp_s)
        poses = _fit_pose_horizon(
            np.repeat(current_control_pose[None], horizon, axis=0), horizon
        )
        target = OSCTargetTrajectory(
            poses=poses,
            wrenches=np.zeros((horizon, 6), dtype=np.float64),
            joint_positions=q_command_trajectory,
            joint_velocities=dq_ref,
            joint_accelerations=ddq_ref,
            joint_torques=tau_ref,
        )
        measured_wrench = _rotate_wrenches(
            np.asarray(sample.wrench_ext, dtype=np.float64)[None],
            sample.wrench_to_control_rotation,
        )[0]
        result = self.controller.optimize_mpc(
            sample.q,
            sample.dq,
            target,
            measured_wrench=measured_wrench,
            previous_tau=self._last_tau,
        )
        tau_unfiltered = self._clip_tau(result.first_tau)
        tau_command = self._filtered_tau(tau_unfiltered, sample.tau)
        self._last_tau = tau_command.copy()
        return InferenceOutput(
            tau_command=tau_command,
            tau_unfiltered=tau_unfiltered,
            **{
                **common,
                "joint_position_command": None,
                "qp_result": result,
                "target_wrench": target.wrenches[0].copy(),
            },
        )

    def _fit_joint(self, values: np.ndarray, horizon: int) -> np.ndarray:
        return _fit_horizon(np.asarray(values, dtype=np.float64), horizon)

    def _limit_mit_velocity(self, values: np.ndarray) -> np.ndarray:
        limit = np.asarray(self.config.execution.mit_velocity_limit, dtype=np.float64)
        if limit.ndim == 0:
            limit = np.repeat(limit, 7)
        limit = limit.reshape(-1)
        if limit.shape != (7,) or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError("execution.mit_velocity_limit must be positive and finite")
        return np.clip(values, -limit[None], limit[None])

    def _filtered_tau(self, value: np.ndarray, initial: np.ndarray) -> np.ndarray:
        clipped = self._clip_tau(value)
        filtered = self._torque_filter.apply(
            clipped,
            dt_s=self.controller.config.dt_s,
            initial_tau=self._clip_tau(initial),
        )
        return self._clip_tau(filtered)


# Short alias for callers that refer to the new path as the contact pipeline.
ContactInferencePipeline = ContactWMInferencePipeline


@dataclass(frozen=True)
class _ContactReference:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    contact_state: np.ndarray | None = None


def _mean_pose(values: np.ndarray) -> np.ndarray:
    values = _numpy_action_chunk(values)
    position = values[:, :3].mean(axis=0)
    quaternions = values[:, 3:].copy()
    reference = quaternions[0]
    signs = np.where((quaternions @ reference)[:, None] < 0.0, -1.0, 1.0)
    quaternion = (quaternions * signs).mean(axis=0)
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-8:
        quaternion = reference
    else:
        quaternion = quaternion / norm
    return np.concatenate((position, quaternion))


def _contact_relative_pose_torch(current, target):
    """Dataset-compatible body-frame relative pose (xyzw quaternions)."""
    import torch
    import torch.nn.functional as functional

    current_q = functional.normalize(current[..., 3:], dim=-1)
    target_q = functional.normalize(target[..., 3:], dim=-1)
    inverse = current_q.clone()
    inverse[..., :3] = -inverse[..., :3]
    ax, ay, az, aw = inverse.unbind(dim=-1)
    bx, by, bz, bw = target_q.unbind(dim=-1)
    relative_q = torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )
    # Rotate world-frame translation by the inverse current orientation.
    vector = target[..., :3] - current[..., :3]
    q_xyz = inverse[..., :3]
    q_w = inverse[..., 3:4]
    cross = torch.linalg.cross(q_xyz, vector, dim=-1)
    rotated = vector + 2.0 * (
        q_w * cross + torch.linalg.cross(q_xyz, cross, dim=-1)
    )
    relative_q = functional.normalize(relative_q, dim=-1)
    sign = torch.where(relative_q[..., 3:4] < 0.0, -1.0, 1.0)
    return torch.cat((rotated, relative_q * sign), dim=-1)
