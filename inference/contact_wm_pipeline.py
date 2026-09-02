"""Inference pipeline for the native PINN Contact World Model v2.

The model condition is ``q``, ``dq``, ``delta_q``, ``tau`` plus an 8-token
direct joint action and mask.  The implementation reuses the mature DP
observation/action scheduling in ``NeroInferencePipeline`` while keeping the
Contact WM state contract explicit at the boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from inference.pipeline import (
    InferenceInput,
    InferenceOutput,
    NeroInferencePipeline,
    _CONTACT_WORLD_MODEL_MODES,
    _fit_horizon,
    _model_device,
    _numpy_vector,
    _synchronize_model,
)
from inference.control.mtc import MTCController


class ContactWMInferencePipeline(NeroInferencePipeline):
    """Run ContactWorldModel v2 at its training rate."""

    def __init__(self, config, **kwargs) -> None:
        if not bool(config.predictor.enabled):
            raise ValueError("ContactWMInferencePipeline requires predictor.enabled=true")
        if str(config.action).strip().lower() != "joint":
            raise ValueError(
                "Contact WM inference requires action: joint (the checkpoint action is "
                "direct absolute joint position)"
            )
        mode = str(config.predictor.mode).strip().lower().replace("-", "_")
        if mode in {
            "contact_world_model",
            "contact_world_model_opd",
            "torque_world_model",
            "torque_world_model_opd",
            "torque_wm",
            "torque_wm_opd",
            "world_model",
            "world_model_v3",
            "world_model_v4",
            "world_model_v5",
            "contact_wm",
            "contact_wm_opd",
        }:
            canonical_mode = (
                "contact_world_model_opd"
                if mode.endswith("_opd")
                else "contact_world_model"
            )
            config = replace(
                config,
                predictor=replace(config.predictor, mode=canonical_mode),
            )
            mode = canonical_mode
        if mode not in _CONTACT_WORLD_MODEL_MODES and mode not in {
            "contact_world_model",
            "contact_world_model_opd",
            "torque_world_model",
            "torque_world_model_opd",
            "torque_wm",
            "torque_wm_opd",
        }:
            raise ValueError(
                "ContactWMInferencePipeline requires predictor.mode=contact_world_model "
                "or contact_world_model_opd"
            )
        # The base class treats Contact WM as an explicitly opted-in model
        # contract; no legacy Contact WM/TorqueWorldModel path is used.
        super().__init__(config, _allow_contact_world_model=True, **kwargs)

        model = self.pinn
        model_version = getattr(model, "MODEL_VERSION", None)
        if model_version is not None and model_version != "contact_world_model_v2":
            raise ValueError(
                "Contact WM checkpoint must expose MODEL_VERSION="
                "'contact_world_model_v2'"
            )

        checkpoint = getattr(model, "_inference_checkpoint_config", {})
        data_cfg = checkpoint.get("dataloader", {}) if isinstance(checkpoint, Mapping) else {}
        model_cfg = checkpoint.get("model", {}) if isinstance(checkpoint, Mapping) else {}
        configured_inputs = tuple(getattr(model, "inputs", ()) or ())
        if not configured_inputs and isinstance(model_cfg, Mapping):
            configured_inputs = tuple(model_cfg.get("inputs", ()) or ())
        required_inputs = ("q", "dq", "delta_q", "tau")
        if configured_inputs and tuple(str(key).lower() for key in configured_inputs) != required_inputs:
            raise ValueError(
                "Contact WM checkpoint model.inputs must be exactly "
                f"{list(required_inputs)}"
            )
        action_dim = int(
            getattr(
                model,
                "action_dim",
                model_cfg.get("action_dim", 7)
                if isinstance(model_cfg, Mapping)
                else 7,
            )
            or 7
        )
        if action_dim != 7:
            raise ValueError(
                "Contact WM direct joint action_dim must be 7; "
                f"checkpoint declares {action_dim}"
            )
        self._contact_history_horizon = int(
            getattr(model, "history_horizon", data_cfg.get("state_history_horizon", 50))
        )
        self._contact_future_horizon = int(
            getattr(model, "future_horizon", data_cfg.get("prediction_horizon", 32))
        )
        self._contact_action_horizon = int(
            getattr(
                model,
                "action_condition_horizon",
                data_cfg.get("action_condition_horizon", 8),
            )
        )
        self._contact_action_start_offset = int(
            data_cfg.get("action_start_offset", 1)
        ) if isinstance(data_cfg, Mapping) else 1
        if self._contact_action_start_offset < 0:
            raise ValueError("Contact WM dataloader.action_start_offset must be non-negative")
        # The official DP policy already slices its diffusion horizon at
        # ``action_start_index = n_obs_steps - 1``.  Do not apply this dataset
        # offset a second time to the emitted action chunk; the executor's
        # remainder is the exact sequence consumed by the Contact WM condition.
        expert_fps = data_cfg.get("expert_fps", 25.0) if isinstance(data_cfg, Mapping) else 25.0
        self._contact_expert_fps = float(expert_fps)
        if not np.isfinite(self._contact_expert_fps) or self._contact_expert_fps <= 0.0:
            raise ValueError("Contact WM dataloader.expert_fps must be positive and finite")
        self._contact_action_period_s = 1.0 / self._contact_expert_fps
        if min(self._contact_history_horizon, self._contact_future_horizon, self._contact_action_horizon) < 1:
            raise ValueError("Contact WM horizons must be positive")
        self._contact_history = {
            key: deque(maxlen=self._contact_history_horizon)
            for key in ("q", "dq", "delta_q", "tau")
        }
        self._contact_previous: tuple[float, dict[str, np.ndarray]] | None = None
        # Used only when an injected/offline caller does not provide q_cmd.
        # The hardware runtime supplies the authoritative command history via
        # ContinuousInferenceStateStream.q_cmd_provider.
        self._contact_last_q_cmd: np.ndarray | None = None
        high_fps = (
            data_cfg.get("high_fps", 100.0)
            if isinstance(data_cfg, Mapping)
            else 100.0
        )
        loss_cfg = checkpoint.get("loss", {}) if isinstance(checkpoint, Mapping) else {}
        configured_dt = getattr(model, "sampling_dt", None)
        if configured_dt is None and isinstance(model_cfg, Mapping):
            configured_dt = model_cfg.get("sampling_dt")
        if configured_dt is None and isinstance(loss_cfg, Mapping):
            # Contact WM's smoothness/kinematic losses use this same physical state
            # period; prefer it when the checkpoint records it explicitly.
            configured_dt = loss_cfg.get("dt")
        self._contact_sampling_dt_s = (
            float(configured_dt)
            if configured_dt is not None
            else (float(1.0 / float(high_fps)) if high_fps else 0.01)
        )
        if not np.isfinite(self._contact_sampling_dt_s) or self._contact_sampling_dt_s <= 0:
            self._contact_sampling_dt_s = 0.01
        self._contact_next_sample_s: float | None = None
        flow_steps = getattr(
            model,
            "flow_inference_steps",
            model_cfg.get("flow_inference_steps", 8),
        )
        self._contact_flow_steps = int(flow_steps or 8)
        self._contact_flow_solver = str(
            getattr(model, "flow_solver", model_cfg.get("flow_solver", "heun"))
        ).lower()
        if self._contact_flow_steps < 1:
            raise ValueError("Contact WM flow inference steps must be positive")
        if self._contact_flow_solver not in {"euler", "heun"}:
            raise ValueError("Contact WM flow solver must be 'euler' or 'heun'")
        self._contact_execution_mode = str(config.execution.mode).strip().lower().replace("-", "_")
        if self._contact_execution_mode not in {"q", "mtc", "tau"}:
            raise ValueError("Contact WM execution.mode must be q, mtc, or tau")
        self.mtc_controller = (
            MTCController(
                model=self.model,
                kp=config.execution.mit_kp,
                kd=config.execution.mit_kd,
                alpha=config.execution.mtc_alpha,
                q_cmd_source=config.execution.mtc_q_cmd_source,
            )
            if self._contact_execution_mode == "mtc"
            else None
        )
        self._validate_action_cadence()
        dp_action_start = getattr(self.dp, "action_start_index", None)
        if dp_action_start is not None and int(dp_action_start) != self._contact_action_start_offset:
            raise ValueError(
                "Contact WM action_start_offset must match the DP checkpoint action_start_index: "
                f"contact_world_model={self._contact_action_start_offset} dp={int(dp_action_start)}"
            )

    def _validate_action_cadence(self) -> None:
        """Require the high-level action clock to match Contact WM training."""

        expected = float(self._contact_action_period_s)
        tolerance = max(1.0e-6, expected * 1.0e-3)
        configured = self.config.predictor.action_step_s
        if configured is not None and abs(float(configured) - expected) > tolerance:
            raise ValueError(
                "Contact WM predictor.action_step_s must match checkpoint expert_fps: "
                f"configured={float(configured):.9g}s expected={expected:.9g}s"
            )
        # A DP checkpoint with an explicit timestamp contract must emit action
        # tokens at the same cadence.  A missing value is allowed for injected
        # models and older checkpoints, which then use the Contact WM cadence below.
        dp_step = self.observation_step_s
        if dp_step is not None and abs(float(dp_step) - expected) > tolerance:
            raise ValueError(
                "Contact WM requires DP action cadence to match checkpoint expert_fps: "
                f"dp={float(dp_step):.9g}s expected={expected:.9g}s"
            )

    def _action_execution_step_s(self) -> float:
        """Advance DP/Contact WM action chunks at the checkpoint's expert rate."""

        return float(self._contact_action_period_s)

    def reset(self) -> None:
        super().reset()
        for values in self._contact_history.values():
            values.clear()
        self._contact_previous = None
        self._contact_last_q_cmd = None
        self._contact_next_sample_s = None

    # Keep the inherited continuous-observation bridge, but replace its old
    # q/v/a/wrench state with the four streams used by Contact WM training.
    def _append_world_model_observation(self, sample: InferenceInput) -> None:
        q_cmd = sample.q_cmd
        if q_cmd is None:
            # Online runtime normally supplies q_cmd from the command history.
            # For offline/injected callers, reuse the previous Contact WM command;
            # never substitute the DP action because it is only a condition and
            # is not necessarily what the low-level transport applied.
            q_cmd = self._contact_last_q_cmd
        if q_cmd is None:
            q_cmd = sample.q
        self._append_contact_observation_values(
            float(sample.timestamp_s), sample.q, sample.dq, sample.tau, q_cmd
        )

    def _append_world_model_observation_values(
        self, timestamp_s: float, values: Mapping[str, np.ndarray]
    ) -> None:
        required = ("q", "tau")
        if any(key not in values for key in required) or ("dq" not in values and "v" not in values):
            raise ValueError("Contact WM observation requires q, dq, and tau")
        dq_value = values.get("dq", values.get("v"))
        q_cmd = values.get("q_cmd")
        if q_cmd is None and values.get("delta_q") is not None:
            q_cmd = np.asarray(values["q"], dtype=np.float32) + np.asarray(
                values["delta_q"], dtype=np.float32
            )
        if q_cmd is None:
            q_cmd = self._contact_last_q_cmd
        if q_cmd is None:
            q_cmd = values["q"]
        self._append_contact_observation_values(
            timestamp_s, values["q"], dq_value, values["tau"], q_cmd
        )

    def _append_contact_observation_values(
        self,
        timestamp_s: float,
        q: Any,
        dq: Any,
        tau: Any,
        q_cmd: Any,
    ) -> None:
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("Contact WM observation timestamp must be finite")
        vectors = {
            "q": np.asarray(q, dtype=np.float32).reshape(-1),
            "dq": np.asarray(dq, dtype=np.float32).reshape(-1),
            "tau": np.asarray(tau, dtype=np.float32).reshape(-1),
            "delta_q": np.asarray(q_cmd, dtype=np.float32).reshape(-1)
            - np.asarray(q, dtype=np.float32).reshape(-1),
        }
        if any(
            value.shape != (7,) or not np.isfinite(value).all()
            for value in vectors.values()
        ):
            raise ValueError("Contact WM q/dq/delta_q/tau must be finite seven-vectors")
        previous = self._contact_previous
        if previous is None:
            self._append_contact_values(vectors)
            self._contact_previous = (timestamp_s, vectors)
            self._contact_next_sample_s = timestamp_s + self._contact_sampling_dt_s
            return
        previous_s, previous_values = previous
        timestamp_tolerance_s = 1.0e-9
        if timestamp_s <= previous_s:
            if abs(timestamp_s - previous_s) <= timestamp_tolerance_s:
                return
            raise ValueError("Contact WM observation timestamps must increase")
        assert self._contact_next_sample_s is not None
        while self._contact_next_sample_s <= timestamp_s + 1.0e-9:
            alpha = (self._contact_next_sample_s - previous_s) / (timestamp_s - previous_s)
            interpolated = {
                key: previous_values[key] + np.float32(alpha) * (vectors[key] - previous_values[key])
                for key in vectors
            }
            # The command itself is held (ZOH) in the training timeline, but
            # ``delta_q`` is a derived stream: ``q_cmd - q``.  Holding the
            # previous delta directly would keep the old measured q embedded
            # in it and silently change the model input during timestamp
            # interpolation.  Recover the previous held command and derive
            # delta_q from the interpolated q instead.
            if alpha >= 1.0 - 1.0e-9:
                interpolated["delta_q"] = vectors["delta_q"].copy()
            else:
                previous_q_cmd = previous_values["q"] + previous_values["delta_q"]
                interpolated["delta_q"] = previous_q_cmd - interpolated["q"]
            self._append_contact_values(interpolated)
            self._contact_next_sample_s += self._contact_sampling_dt_s
        self._contact_previous = (timestamp_s, vectors)

    def _append_contact_values(self, values: Mapping[str, np.ndarray]) -> None:
        for key, value in values.items():
            history = self._contact_history[key]
            history.append(np.asarray(value, dtype=np.float32).copy())
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
        current_control_pose = self.model.snapshot(sample.q, sample.dq).pose
        self._update_control_timestep(sample.timestamp_s)
        # Contact WM actions are joint-space values; no FK frame conversion is needed
        # (and a test/controller model need not expose frame_pose()).
        current_action_pose = current_control_pose
        dp_updated = self._update_dp_execution(
            sample.timestamp_s, current_action_pose, sample.q
        )
        if self._action is None:
            self._set_idle_action(current_action_pose, sample.q)

        pinn_started = perf_counter()
        reference = self._predict_contact_reference()
        self._timing_pinn_s += perf_counter() - pinn_started
        if self._dp_action_chunk is None:
            # The flow model is stochastic. Until the first real DP chunk is
            # available, never turn its random source trajectory into motion.
            output = self._idle_contact_output(sample, dp_updated)
        else:
            output = self._execute_contact_reference(
                sample, current_control_pose, reference, dp_updated
            )
        self._remember_contact_command(sample, output)
        self._timing_cycles += 1
        self._report_timing(perf_counter() - cycle_started)
        return output

    def _remember_contact_command(
        self, sample: InferenceInput, output: InferenceOutput
    ) -> None:
        """Remember the command that the runtime transport will apply next."""

        if self._contact_execution_mode == "q":
            command = output.joint_position_command
        elif self._contact_execution_mode == "mtc":
            command = output.joint_position_target
        else:
            # Pure torque transport holds q at the measured state;
            # their torque target is not a commanded joint position.
            command = sample.q
        if command is None:
            command = sample.q
        value = np.asarray(command, dtype=np.float64).reshape(-1)
        if value.shape == (7,) and np.all(np.isfinite(value)):
            self._contact_last_q_cmd = value.copy()

    def _predict_contact_reference(self) -> dict[str, Any]:
        import torch

        if any(len(values) < 1 for values in self._contact_history.values()):
            raise RuntimeError("Contact WM state history is empty")
        device = _model_device(self.pinn)
        history = {
            key: np.stack(tuple(values), axis=0) for key, values in self._contact_history.items()
        }
        inputs = {
            key: self._normalize_pinn_input(
                key,
                torch.as_tensor(value, dtype=torch.float32, device=device)[None],
            )
            for key, value in history.items()
        }
        actions, mask = self._contact_action_condition(device)
        inputs["action"] = self._normalize_pinn_input("action", actions)
        inputs["action_mask"] = mask
        _synchronize_model(self.pinn)
        started = perf_counter()
        with torch.inference_mode():
            try:
                output = self.pinn.predict(
                    inputs,
                    steps=self._contact_flow_steps,
                    solver=self._contact_flow_solver,
                )
            except TypeError:
                # Small injected/test models may only accept the batch.
                output = self.pinn.predict(inputs)
        _synchronize_model(self.pinn)
        self._record_wm_inference_timing(perf_counter() - started)
        if not isinstance(output, Mapping):
            raise RuntimeError("Contact WM predict() must return a mapping")
        # ContactWorldModel v2 returns flat ``<stream>_pred`` tensors. Keep
        # accepting the historical nested mapping for injected callers, but
        # normalize both forms to one internal state mapping here.
        nested = output.get("state_pred")
        state_pred = nested if isinstance(nested, Mapping) else {}
        result: dict[str, Any] = {}
        for key in ("q", "dq", "delta_q", "tau"):
            value = state_pred.get(key, output.get(f"{key}_pred"))
            if value is None:
                raise RuntimeError(f"Contact WM prediction is missing {key!r}")
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 7:
                raise RuntimeError(
                    f"Contact WM prediction[{key!r}] must have shape [1,T,7], got {tuple(value.shape)}"
                )
            if value.shape[1] != self._contact_future_horizon:
                raise RuntimeError(
                    f"Contact WM prediction[{key!r}] horizon {value.shape[1]} != {self._contact_future_horizon}"
                )
            physical = self._denormalize_pinn_output(key, value)
            if not torch.is_tensor(physical):
                physical = torch.as_tensor(physical, dtype=torch.float32, device=device)
            if not torch.isfinite(physical).all():
                raise RuntimeError(f"Contact WM state_pred[{key!r}] is non-finite")
            result[key] = physical[0].detach().cpu().numpy().astype(np.float64)
        contact = state_pred.get("contact_state")
        if contact is None:
            contact = output.get("contact_state")
        if contact is None:
            contact = output.get("contact_state_pred")
        if contact is None:
            logits = output.get("contact_logits")
            if logits is None:
                logits = output.get("contact_phase_logits")
            if logits is None:
                logits = output.get("contact_probability")
            if logits is None:
                logits = output.get("contact_phase_probability")
            if logits is not None:
                if hasattr(logits, "detach"):
                    contact = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits_array = np.asarray(logits)
                    contact = np.argmax(logits_array, axis=-1)[..., None]
        if contact is not None:
            if hasattr(contact, "detach"):
                contact = contact.detach().cpu().numpy().astype(np.float64)
            else:
                contact = np.asarray(contact, dtype=np.float64)
            if contact.ndim == 3 and contact.shape[0] == 1:
                contact = contact[0]
            if contact.ndim == 1:
                contact = contact[:, None]
            if contact.shape != (self._contact_future_horizon, 1):
                raise RuntimeError(
                    "Contact WM contact_state must have shape "
                    f"[{self._contact_future_horizon}, 1], got {contact.shape}"
                )
            if not np.isfinite(contact).all():
                raise RuntimeError("Contact WM contact_state is non-finite")
            result["contact_state"] = contact
        return result

    def predict_contact_reference(self, history: Any, action: Any) -> dict[str, np.ndarray]:
        """Predict a Contact WM trajectory from timestamped buffer snapshots.

        This is deliberately independent from the legacy ``step`` state.  The
        asynchronous WM worker can therefore run while the DP worker and the
        100 Hz controller continue operating.  ``history`` must expose the
        four Contact WM streams as ``[50, 7]`` arrays; ``action`` may be a
        100 Hz ZOH trajectory and is reduced to the checkpoint's eight action
        tokens at the same 25 Hz cadence used during training.
        """
        import torch

        arrays: dict[str, np.ndarray] = {}
        for key in ("q", "dq", "delta_q", "tau"):
            value = getattr(history, key, None)
            if value is None and isinstance(history, Mapping):
                value = history.get(key)
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (self._contact_history_horizon, 7) or not np.isfinite(value).all():
                raise ValueError(
                    f"Contact WM history[{key!r}] must have shape "
                    f"[{self._contact_history_horizon},7], got {value.shape}"
                )
            arrays[key] = value

        action_values = getattr(action, "values", action)
        action_values = np.asarray(action_values, dtype=np.float32)
        if action_values.ndim == 3 and action_values.shape[0] == 1:
            action_values = action_values[0]
        if action_values.ndim != 2 or action_values.shape[1] != 7 or not np.isfinite(action_values).all():
            raise ValueError(f"Contact WM action condition must be [T,7], got {action_values.shape}")
        # The WM checkpoint was trained with eight 25 Hz action tokens.  The
        # ActionPlanBuffer supplies a 100 Hz ZOH trajectory, so selecting every
        # fourth sample preserves the token-at-the-start-of-bin semantics.
        token_count = self._contact_action_horizon
        if action_values.shape[0] >= token_count:
            stride = max(1, int(round(action_values.shape[0] / token_count)))
            indices = np.minimum(np.arange(token_count) * stride, action_values.shape[0] - 1)
            action_condition = action_values[indices]
        else:
            action_condition = np.repeat(action_values[-1:, :], token_count, axis=0)
            action_condition[: action_values.shape[0]] = action_values

        device = _model_device(self.pinn)
        inputs = {
            key: self._normalize_pinn_input(
                key,
                torch.as_tensor(value, dtype=torch.float32, device=device)[None],
            )
            for key, value in arrays.items()
        }
        inputs["action"] = self._normalize_pinn_input(
            "action",
            torch.as_tensor(action_condition, dtype=torch.float32, device=device)[None],
        )
        inputs["action_mask"] = torch.ones(
            (1, token_count), dtype=torch.float32, device=device
        )
        _synchronize_model(self.pinn)
        with torch.inference_mode():
            try:
                output = self.pinn.predict(
                    inputs,
                    steps=self._contact_flow_steps,
                    solver=self._contact_flow_solver,
                )
            except TypeError:
                output = self.pinn.predict(inputs)
        _synchronize_model(self.pinn)
        if not isinstance(output, Mapping):
            raise RuntimeError("Contact WM predict() must return a mapping")
        nested = output.get("state_pred")
        state_pred = nested if isinstance(nested, Mapping) else {}
        result: dict[str, np.ndarray] = {}
        for key in ("q", "tau"):
            value = state_pred.get(key, output.get(f"{key}_pred"))
            if value is None:
                raise RuntimeError(f"Contact WM prediction is missing {key!r}")
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 7:
                raise RuntimeError(
                    f"Contact WM prediction[{key!r}] must have shape [1,T,7], "
                    f"got {tuple(value.shape)}"
                )
            if value.shape[1] != self._contact_future_horizon:
                raise RuntimeError(
                    f"Contact WM prediction[{key!r}] horizon {value.shape[1]} "
                    f"!= {self._contact_future_horizon}"
                )
            physical = self._denormalize_pinn_output(key, value)
            if not torch.isfinite(physical).all():
                raise RuntimeError(f"Contact WM prediction[{key!r}] is non-finite")
            result[f"{key}_ref"] = physical[0].detach().cpu().numpy().astype(np.float64)
        return result

    def _contact_action_condition(self, device):
        import torch

        # ActionPlanExecutor advances this remainder at the 25 Hz action clock.
        # The raw DP chunk is only a fallback for callers that do not use the
        # executor; feeding it first would repeat token zero for the whole plan.
        source = self._action_chunk
        if source is None:
            source = self._dp_action_chunk
        if source is None:
            source = self._action
        if source is None:
            source = np.zeros(7, dtype=np.float64)
        source = np.asarray(source, dtype=np.float32)
        if source.ndim == 1:
            source = source[None]
        if source.ndim != 2 or source.shape[1] != 7:
            raise RuntimeError(f"Contact WM direct action must be [H,7], got {source.shape}")
        valid = min(source.shape[0], self._contact_action_horizon)
        values = np.repeat(source[-1:, :], self._contact_action_horizon, axis=0)
        values[:valid] = source[:valid]
        # Training pads/holds the direct action chunk and marks every token
        # valid.  Keep that contract even during startup when only one DP
        # action is available.
        mask = np.ones(self._contact_action_horizon, dtype=np.float32)
        return (
            torch.as_tensor(values, dtype=torch.float32, device=device)[None],
            torch.as_tensor(mask, dtype=torch.float32, device=device)[None],
        )

    def _idle_contact_output(self, sample: InferenceInput, dp_updated: bool) -> InferenceOutput:
        """Hold the measured state while waiting for the first DP action."""
        horizon = self._contact_future_horizon
        q = _numpy_vector(sample.q, 7, "sample q")
        dq = np.zeros(7, dtype=np.float64)
        tau = np.zeros(7, dtype=np.float64)
        q_trajectory = np.repeat(q[None], horizon, axis=0)
        dq_trajectory = np.repeat(dq[None], horizon, axis=0)
        tau_trajectory = np.repeat(tau[None], horizon, axis=0)
        common = dict(
            action_target=q.copy(),
            dp_action_chunk=None,
            target_wrench=np.zeros(6, dtype=np.float64),
            qp_result=None,
            ik_result=None,
            dp_updated=dp_updated,
            pinn_updated=False,
            dp_inference_time_s=self._last_dp_inference_time_s,
            wm_inference_time_s=self._last_wm_inference_time_s,
            joint_position_target=q.copy(),
            joint_velocity_target=dq.copy(),
            torque_target=tau.copy(),
            contact_state=None,
            control_mode=self._contact_execution_mode,
            joint_position_trajectory=q_trajectory,
            joint_velocity_trajectory=dq_trajectory,
            joint_acceleration_trajectory=np.zeros_like(dq_trajectory),
            torque_trajectory=tau_trajectory,
        )
        if self._contact_execution_mode == "q":
            return InferenceOutput(
                tau_command=tau.copy(),
                tau_unfiltered=tau.copy(),
                joint_position_command=q.copy(),
                **common,
            )
        if self._contact_execution_mode == "mtc":
            kp = np.asarray(self.config.execution.mit_kp, dtype=np.float64)
            kd = np.asarray(self.config.execution.mit_kd, dtype=np.float64)
            return InferenceOutput(
                tau_command=tau.copy(),
                tau_unfiltered=tau.copy(),
                joint_position_command=None,
                mit_kp=kp,
                mit_kd=kd,
                mtc_tau_qv=tau.copy(),
                mtc_tau_pred=tau.copy(),
                mtc_alpha=float(self.config.execution.mtc_alpha),
                **common,
            )
        return InferenceOutput(
            tau_command=tau.copy(),
            tau_unfiltered=tau.copy(),
            joint_position_command=None,
            **common,
        )

    def _execute_contact_reference(
        self,
        sample: InferenceInput,
        current_control_pose: np.ndarray,
        reference: Mapping[str, Any],
        dp_updated: bool,
    ) -> InferenceOutput:
        horizon = self._contact_future_horizon
        q_future = _fit_horizon(reference["q"], horizon)
        # Bound the complete trajectory before it reaches direct-q transport;
        # later tokens must not cross hard limits or jump implausibly.
        q_future = self._safe_joint_position_trajectory(
            sample.q,
            q_future,
            dt_s=self._contact_sampling_dt_s,
        )
        dq_future = self._limit_predicted_velocity(_fit_horizon(reference["dq"], horizon))
        dq_future = self._limit_model_velocity_trajectory(dq_future)
        tau_future = _fit_horizon(reference["tau"], horizon)
        # Contact WM predicts dq but not ddq; keep a finite-difference trajectory for
        # diagnostics and downstream consumers.
        dt = self._contact_sampling_dt_s
        ddq_future = np.empty_like(dq_future)
        ddq_future[0] = (dq_future[0] - np.asarray(sample.dq, dtype=np.float64)) / dt
        if horizon > 1:
            ddq_future[1:] = np.diff(dq_future, axis=0) / dt
        q_trajectory = q_future.copy()
        q_command = q_trajectory[0].copy()
        # Keep the raw WM prediction for diagnostics; only ``tau_command`` is
        # passed through the causal safety filter before transport.
        tau_pred = self._clip_tau(tau_future[0])
        tau_ff = self._filtered_tau(tau_pred, sample.tau)
        contact = reference.get("contact_state")
        contact0 = None if contact is None else np.asarray(contact[0]).copy()
        common = dict(
            action_target=(self._action.copy() if self._action is not None else sample.q.copy()),
            dp_action_chunk=(None if self._dp_action_chunk is None else self._dp_action_chunk.copy()),
            target_wrench=np.zeros(6, dtype=np.float64),
            qp_result=None,
            ik_result=None,
            dp_updated=dp_updated,
            pinn_updated=True,
            dp_inference_time_s=self._last_dp_inference_time_s,
            wm_inference_time_s=self._last_wm_inference_time_s,
            joint_position_target=q_command.copy(),
            joint_velocity_target=dq_future[0].copy(),
            torque_target=tau_ff.copy(),
            contact_state=contact0,
            control_mode=self._contact_execution_mode,
            joint_position_trajectory=q_trajectory.copy(),
            joint_velocity_trajectory=dq_future.copy(),
            joint_acceleration_trajectory=ddq_future.copy(),
            torque_trajectory=tau_future.copy(),
        )
        zeros = np.zeros(7, dtype=np.float64)
        if self._contact_execution_mode == "q":
            return InferenceOutput(
                tau_command=zeros.copy(), tau_unfiltered=zeros.copy(),
                joint_position_command=q_command, **common
            )
        if self._contact_execution_mode == "mtc":
            assert self.mtc_controller is not None
            delta_future = _fit_horizon(reference["delta_q"], horizon)
            raw_q_cmd = self.mtc_controller.resolve_q_cmd(
                q_command, delta_future[0]
            )
            safe_q_cmd = self._safe_joint_position_trajectory(
                sample.q,
                raw_q_cmd[None],
                dt_s=self._contact_sampling_dt_s,
            )[0]
            delta_for_control = (
                safe_q_cmd - q_command
                if self.mtc_controller.q_cmd_source == "wm_delta"
                else None
            )
            mtc = self.mtc_controller.compute(
                q=sample.q,
                dq=sample.dq,
                tau_pred=tau_ff,
                q_hat=q_command,
                delta_q_hat=delta_for_control,
            )
            # Preserve the data-side firmware gains.  The feed-forward is the
            # residual required so firmware's own PD term completes tau_cmd.
            kp = np.asarray(self.config.execution.mit_kp, dtype=np.float64)
            kd = np.asarray(self.config.execution.mit_kd, dtype=np.float64)
            # ``maximum_command_torque_nm`` is also the transport limit for
            # ``t_ff``.  Recompute the effective total after clipping the
            # residual so diagnostics describe the torque the fixed-gain MIT
            # command can actually request at this sampled q/dq.
            requested_total = self._clip_tau(mtc.tau_command)
            t_ff = self._clip_tau(requested_total - mtc.tau_pd)
            effective_total = self._clip_tau(mtc.tau_pd + t_ff)
            return InferenceOutput(
                tau_command=effective_total,
                tau_unfiltered=effective_total.copy(),
                joint_position_command=None,
                mit_kp=kp,
                mit_kd=kd,
                **{
                    **common,
                    "joint_position_target": mtc.q_cmd.copy(),
                    "joint_velocity_target": np.zeros(7, dtype=np.float64),
                    "torque_target": t_ff,
                    "mtc_tau_qv": mtc.tau_qv.copy(),
                    "mtc_tau_pred": mtc.tau_pred.copy(),
                    "mtc_alpha": mtc.alpha,
                },
            )
        if self._contact_execution_mode == "tau":
            return InferenceOutput(
                tau_command=tau_ff.copy(), tau_unfiltered=tau_pred.copy(),
                joint_position_command=None, **common
            )

        raise RuntimeError(f"unsupported Contact WM execution mode {self._contact_execution_mode!r}")

    def _limit_predicted_velocity(self, values: np.ndarray) -> np.ndarray:
        limit = np.asarray(self.config.execution.mit_velocity_limit, dtype=np.float64)
        if limit.ndim == 0:
            limit = np.repeat(limit, 7)
        limit = limit.reshape(-1)
        if limit.shape != (7,) or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError("execution.mit_velocity_limit must be positive and finite")
        return np.clip(values, -limit[None], limit[None])

    def _limit_model_velocity_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Apply the robot model's physical velocity limits when available."""

        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != 7 or not np.all(np.isfinite(result)):
            raise ValueError("Contact WM joint velocity trajectory must be finite [H,7]")
        model_limit = getattr(self.model, "velocity_limit", None)
        if model_limit is None:
            return result.copy()
        limit = np.asarray(model_limit, dtype=np.float64).reshape(-1)
        if limit.shape != (7,) or not np.all(np.isfinite(limit)) or np.any(limit <= 0.0):
            # Some simulation models use +inf for an unconstrained axis.  In
            # that case retain the configured MIT limit instead of rejecting a
            # valid checkpoint solely because the optional model metadata is
            # unbounded.
            if limit.shape != (7,) or np.any(limit <= 0.0) or np.any(np.isnan(limit)):
                raise ValueError("dynamics model.velocity_limit must be positive [7]")
            finite = np.isfinite(limit)
            clipped = result.copy()
            clipped[:, finite] = np.clip(
                clipped[:, finite], -limit[finite], limit[finite]
            )
            return clipped
        return np.clip(result, -limit[None], limit[None])

    def _safe_joint_position_trajectory(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        *,
        dt_s: float,
    ) -> np.ndarray:
        """Clamp every future joint token to limits and per-step motion bounds."""

        current = _numpy_vector(current_q, 7, "current q")
        targets = np.asarray(target_q, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 7 or targets.shape[0] < 1:
            raise ValueError(f"Contact WM joint position trajectory must be [H,7], got {targets.shape}")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Contact WM joint position trajectory must be finite")
        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("Contact WM sampling dt must be positive and finite")

        configured_step = np.asarray(
            self.config.safety.maximum_joint_position_step_rad,
            dtype=np.float64,
        )
        if configured_step.ndim == 0:
            configured_step = np.repeat(configured_step, 7)
        configured_step = configured_step.reshape(-1)
        if (
            configured_step.shape != (7,)
            or not np.all(np.isfinite(configured_step))
            or np.any(configured_step <= 0.0)
        ):
            raise ValueError(
                "maximum_joint_position_step_rad must be a positive scalar or 7-vector"
            )

        physical_lower, physical_upper = self._joint_position_bounds(0.0)
        soft_lower, soft_upper = self._joint_position_bounds(
            self.config.ik.joint_position_margin_rad
        )
        step_limit = configured_step.copy()
        model_velocity = getattr(self.model, "velocity_limit", None)
        if model_velocity is not None:
            velocity = np.asarray(model_velocity, dtype=np.float64).reshape(-1)
            if velocity.shape != (7,) or np.any(np.isnan(velocity)) or np.any(velocity <= 0.0):
                raise ValueError("dynamics model.velocity_limit must be positive [7]")
            finite = np.isfinite(velocity)
            step_limit[finite] = np.minimum(step_limit[finite], velocity[finite] * dt_s)

        result = np.empty_like(targets)
        previous = np.clip(current, physical_lower, physical_upper)
        for index, target in enumerate(targets):
            bounded = np.clip(target, soft_lower, soft_upper)
            command = previous + np.clip(bounded - previous, -step_limit, step_limit)
            command = np.clip(command, physical_lower, physical_upper)
            result[index] = command
            previous = command
        return result

    def _filtered_tau(self, value: np.ndarray, initial: np.ndarray) -> np.ndarray:
        clipped = self._clip_tau(value)
        filtered = self._torque_filter.apply(
            clipped, dt_s=self._control_dt_s, initial_tau=self._clip_tau(initial)
        )
        return self._clip_tau(filtered)


ContactWorldModelInferencePipeline = ContactWMInferencePipeline
ContactWMPipeline = ContactWMInferencePipeline
ContactInferencePipeline = ContactWMInferencePipeline

__all__ = [
    "ContactWMInferencePipeline",
    "ContactWorldModelInferencePipeline",
    "ContactWMPipeline",
    "ContactInferencePipeline",
]
