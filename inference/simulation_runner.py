"""Offline H5 -> MuJoCo inference runner.

This module is an intentionally separate execution path from the CAN runtime.
It reuses the deployed inference pipelines, but replaces the hardware transport
with :class:`inference.mujoco_backend.MujocoDynamicsBackend`.  H5 observations
are consumed on a regular state clock and every command is applied through
MuJoCo torque actuators.

Two observation contracts are available:

``recorded``
    Feed q/dq/ddq/tau/wrench from the recording to the policy.  This is the
    deterministic contract check: the simulated state is logged separately and
    is not fed back into the policy.

``hybrid_closed_loop``
    Keep the recorded image and wrench streams, while feeding the simulated
    q/dq and the torque measured from the previous simulation step back to the
    policy.  This exposes drift and controller interactions without pretending
    that H5 images are rendered from MuJoCo.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from inference.config import InferenceConfig
from inference.h5_observation_stream import (
    H5ObservationEpisode,
    H5ObservationStream,
)
from inference.mujoco_backend import MujocoBackendConfig, MujocoDynamicsBackend
from inference.pipeline import InferenceInput, InferenceOutput, NeroInferencePipeline


log = logging.getLogger(__name__)

ObservationMode = Literal["recorded", "hybrid_closed_loop"]
EXECUTION_MODES = ("mit", "osc_qp", "q", "tau")
OBSERVATION_MODES = ("recorded", "hybrid_closed_loop")


@dataclass(frozen=True)
class SimulationRunnerConfig:
    """Numerical and replay settings for :func:`run_h5_simulation`."""

    observation_mode: ObservationMode = "recorded"
    execution_mode: str | None = None
    state_rate_hz: float = 100.0
    camera_rate_hz: float = 25.0
    history_steps: int = 50
    camera_history_steps: int = 1
    camera_history_step_s: float | None = None
    left_pad: bool = True
    state_alignment: str = "previous"
    max_state_alignment_gap_s: float | None = None
    max_camera_age_s: float | None = None
    physics_dt_s: float = 0.001
    max_steps: int | None = None
    realtime: bool = False
    viewer: bool = False
    allow_asynchronous: bool = False

    def __post_init__(self) -> None:
        mode = str(self.observation_mode).strip().lower().replace("-", "_")
        if mode not in OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be 'recorded' or 'hybrid_closed_loop'"
            )
        object.__setattr__(self, "observation_mode", mode)
        if self.execution_mode is not None:
            mode = str(self.execution_mode).strip().lower().replace("-", "_")
            if mode not in EXECUTION_MODES:
                raise ValueError(
                    "execution_mode must be one of 'mit', 'osc_qp', 'q', or 'tau'"
                )
            object.__setattr__(self, "execution_mode", mode)
        for name in ("state_rate_hz", "camera_rate_hz"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if int(self.history_steps) < 1 or int(self.camera_history_steps) < 1:
            raise ValueError("history_steps and camera_history_steps must be positive")
        object.__setattr__(self, "history_steps", int(self.history_steps))
        object.__setattr__(self, "camera_history_steps", int(self.camera_history_steps))
        if self.camera_history_step_s is not None:
            value = float(self.camera_history_step_s)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("camera_history_step_s must be positive and finite")
            object.__setattr__(self, "camera_history_step_s", value)
        alignment = str(self.state_alignment).strip().lower().replace("-", "_")
        if alignment not in {"nearest", "previous"}:
            raise ValueError("state_alignment must be 'nearest' or 'previous'")
        object.__setattr__(self, "state_alignment", alignment)
        for name in ("max_state_alignment_gap_s", "max_camera_age_s"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be non-negative and finite")
                object.__setattr__(self, name, value)
        physics_dt = float(self.physics_dt_s)
        if not np.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError("physics_dt_s must be positive and finite")
        object.__setattr__(self, "physics_dt_s", physics_dt)
        if self.max_steps is not None:
            if int(self.max_steps) < 1:
                raise ValueError("max_steps must be positive")
            object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "allow_asynchronous", bool(self.allow_asynchronous))


@dataclass(frozen=True)
class SimulationRunResult:
    """Numerical trace produced by one offline simulation run."""

    timestamps_s: np.ndarray
    camera_indices: np.ndarray
    camera_age_s: np.ndarray
    recorded_q: np.ndarray
    recorded_dq: np.ndarray
    recorded_ddq: np.ndarray
    recorded_tau: np.ndarray
    simulated_q: np.ndarray
    simulated_dq: np.ndarray
    simulated_ddq: np.ndarray
    applied_torque: np.ndarray
    command_torque: np.ndarray
    q_target: np.ndarray
    dq_target: np.ndarray
    tau_target: np.ndarray
    action_target: np.ndarray
    dp_inference_time_s: np.ndarray
    wm_inference_time_s: np.ndarray
    dp_updated: np.ndarray
    pinn_updated: np.ndarray
    contact_state: np.ndarray
    contact_count: np.ndarray
    execution_mode: str
    observation_mode: str

    @property
    def sample_count(self) -> int:
        return int(self.timestamps_s.size)

    def save_npz(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Save a portable trace with JSON metadata."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "format": "nero_h5_mujoco_inference/v1",
            "samples": self.sample_count,
            "execution_mode": self.execution_mode,
            "observation_mode": self.observation_mode,
        }
        if metadata:
            info.update(metadata)
        np.savez_compressed(
            output,
            metadata_json=np.asarray(json.dumps(info, sort_keys=True, default=str)),
            timestamps_s=self.timestamps_s,
            camera_indices=self.camera_indices,
            camera_age_s=self.camera_age_s,
            recorded_q=self.recorded_q,
            recorded_dq=self.recorded_dq,
            recorded_ddq=self.recorded_ddq,
            recorded_tau=self.recorded_tau,
            simulated_q=self.simulated_q,
            simulated_dq=self.simulated_dq,
            simulated_ddq=self.simulated_ddq,
            applied_torque=self.applied_torque,
            command_torque=self.command_torque,
            q_target=self.q_target,
            dq_target=self.dq_target,
            tau_target=self.tau_target,
            action_target=self.action_target,
            dp_inference_time_s=self.dp_inference_time_s,
            wm_inference_time_s=self.wm_inference_time_s,
            dp_updated=self.dp_updated,
            pinn_updated=self.pinn_updated,
            contact_state=self.contact_state,
            contact_count=self.contact_count,
        )
        return output


def build_pipeline(
    config: InferenceConfig,
    *,
    dp_model: Any | None = None,
    pinn_model: Any | None = None,
    controller: Any | None = None,
) -> NeroInferencePipeline:
    """Construct the appropriate legacy or contact-WM pipeline.

    The factory is kept here rather than in the hardware runtime so the
    simulation branch cannot accidentally initialize CAN resources.
    """

    predictor_mode = str(config.predictor.mode).strip().lower().replace("-", "_")
    if config.predictor.enabled and predictor_mode in {
        "contact_world_model",
        "contact_world_model_opd",
        "contact_wm",
        "contact_wm_opd",
    }:
        from inference.contact_pipeline import ContactWMInferencePipeline

        return ContactWMInferencePipeline(
            config,
            dp_model=dp_model,
            pinn_model=pinn_model,
            controller=controller,
        )
    return NeroInferencePipeline(
        config,
        dp_model=dp_model,
        pinn_model=pinn_model,
        controller=controller,
    )


def make_observation_stream(
    episode: H5ObservationEpisode,
    config: SimulationRunnerConfig,
    *,
    start_timestamp_us: int | None = None,
    stop_timestamp_us: int | None = None,
) -> H5ObservationStream:
    """Create a fixed-rate stream from an already loaded episode."""

    max_state_gap_us = (
        None
        if config.max_state_alignment_gap_s is None
        else round(config.max_state_alignment_gap_s * 1.0e6)
    )
    max_camera_age_us = (
        None
        if config.max_camera_age_s is None
        else round(config.max_camera_age_s * 1.0e6)
    )
    return H5ObservationStream(
        episode,
        state_rate_hz=config.state_rate_hz,
        camera_rate_hz=config.camera_rate_hz,
        history_steps=config.history_steps,
        camera_history_steps=config.camera_history_steps,
        camera_history_step_s=config.camera_history_step_s,
        left_pad=config.left_pad,
        state_alignment=config.state_alignment,
        max_state_alignment_gap_us=max_state_gap_us,
        max_camera_age_us=max_camera_age_us,
        start_timestamp_us=start_timestamp_us,
        stop_timestamp_us=stop_timestamp_us,
    )


def backend_config_from_inference(
    inference_config: InferenceConfig,
    plan: Any,
    runner_config: SimulationRunnerConfig,
) -> MujocoBackendConfig:
    """Translate deployed safety/gain settings to the MuJoCo backend.

    The effective torque limit is the stricter of the calibrated dynamics
    limit and the inference safety limit.  This prevents an offline run from
    silently testing torques that the real transport would reject.
    """

    plan_limits = np.asarray(plan.safety.max_abs_torque_nm, dtype=np.float64)
    configured = np.asarray(
        inference_config.safety.maximum_command_torque_nm,
        dtype=np.float64,
    )
    if configured.ndim == 0:
        configured = np.repeat(configured, 7)
    configured = configured.reshape(-1)
    if plan_limits.shape != (7,) or configured.shape != (7,):
        raise ValueError("simulation torque limits must be seven-vectors")
    if (
        not np.isfinite(plan_limits).all()
        or not np.isfinite(configured).all()
        or np.any(plan_limits <= 0.0)
        or np.any(configured <= 0.0)
    ):
        raise ValueError("simulation torque limits must be finite and positive")
    limits = np.minimum(plan_limits, configured)
    kp = np.asarray(inference_config.execution.mit_kp, dtype=np.float64)
    kd = np.asarray(inference_config.execution.mit_kd, dtype=np.float64)
    if kp.ndim == 0:
        kp = np.repeat(kp, 7)
    if kd.ndim == 0:
        kd = np.repeat(kd, 7)
    kp = kp.reshape(-1)
    kd = kd.reshape(-1)
    if (
        kp.shape != (7,)
        or kd.shape != (7,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
        or np.any(kp < 0.0)
        or np.any(kd < 0.0)
    ):
        raise ValueError("simulation MIT gains must be finite non-negative seven-vectors")
    # q mode has no separate hardware gain contract.  Reuse MIT gains when
    # configured; a conservative software-servo fallback keeps direct-IK
    # configs with zero MIT gains dynamically controllable.
    q_kp = np.where(kp > 0.0, kp, 50.0)
    q_kd = np.where(kd > 0.0, kd, 5.0)
    return MujocoBackendConfig(
        physics_dt_s=runner_config.physics_dt_s,
        control_dt_s=1.0 / runner_config.state_rate_hz,
        mit_kp=tuple(kp.tolist()),
        mit_kd=tuple(kd.tolist()),
        q_kp=tuple(q_kp.tolist()),
        q_kd=tuple(q_kd.tolist()),
        torque_limits_nm=tuple(limits.tolist()),
    )


def _pipeline_step(
    pipeline: Any,
    sample: InferenceInput,
    tick: Any,
    stream: H5ObservationStream,
) -> InferenceOutput:
    """Run one deterministic policy step when the direct-IK branch is used."""

    predictor = getattr(getattr(pipeline, "config", None), "predictor", None)
    predictor_enabled = bool(getattr(predictor, "enabled", True))
    if not predictor_enabled and hasattr(
        pipeline, "step_direct_ik_observation_history"
    ):
        wrench_history = _direct_ik_wrench_history(pipeline, tick, stream)
        image_history = tick.image_history
        configured_image_keys = tuple(getattr(pipeline, "_image_keys", ()))
        if len(configured_image_keys) > 1:
            image_history = getattr(tick, "image_histories", None)
            if not image_history:
                raise ValueError(
                    "multi-camera direct-IK pipeline requires image histories by camera"
                )
        return pipeline.step_direct_ik_observation_history(
            sample,
            image_history,
            wrench_history,
        )
    return pipeline.step(sample)


def _direct_ik_wrench_history(
    pipeline: Any,
    tick: Any,
    stream: H5ObservationStream,
) -> np.ndarray:
    """Build the DP contract ``[image_step, CAN_step, 6]`` wrench window.

    ``H5ObservationTick.wrench_history`` is intentionally a state-clock
    history for the contact WM.  The legacy DP encoder instead expects a
    short CAN history for every image observation, so align each historical
    camera frame causally to the recorded teleop stream and backfill its
    preceding wrench samples.
    """

    image_history = np.asarray(tick.image_history)
    image_steps = int(image_history.shape[0])
    wrench_steps = int(getattr(pipeline, "wrench_history_steps", 1))
    if wrench_steps < 1:
        raise ValueError("pipeline.wrench_history_steps must be positive")
    camera_indices = np.asarray(tick.camera_history_indices, dtype=np.int64).reshape(-1)
    if camera_indices.shape != (image_steps,):
        raise ValueError(
            "camera history and image history lengths differ: "
            f"{camera_indices.shape} vs {image_steps}"
        )
    state_timestamps = stream.episode.state_timestamp_us
    camera_timestamps = stream.episode.camera_timestamp_us[camera_indices]
    # Camera observations must never expose a future CAN sample.  This is
    # deliberately causal even when the overall stream uses nearest state
    # alignment for a recorded-contract diagnostic.
    state_indices = np.searchsorted(
        state_timestamps, camera_timestamps, side="right"
    ).astype(np.int64) - 1
    state_indices = np.maximum(state_indices, 0)
    offsets = np.arange(wrench_steps - 1, -1, -1, dtype=np.int64)
    indices = np.clip(
        state_indices[:, None] - offsets[None, :],
        0,
        stream.episode.state_timestamp_us.size - 1,
    )
    return np.asarray(stream.episode.wrench_ext[indices], dtype=np.float32)


def _vector_or_nan(value: Any, *, width: int = 7) -> np.ndarray:
    if value is None:
        return np.full(width, np.nan, dtype=np.float64)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.full(width, np.nan, dtype=np.float64)
    if array.shape == (width,):
        result = array
    elif array.ndim == 2 and array.shape[1] == width and array.shape[0] >= 1:
        result = array[0]
    else:
        return np.full(width, np.nan, dtype=np.float64)
    if not np.isfinite(result).all():
        return np.full(width, np.nan, dtype=np.float64)
    return result.copy()


def _timing_scalar(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) and result >= 0.0 else np.nan


def _contact_scalar(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.nan
    return float(array[0]) if array.size and np.isfinite(array[0]) else np.nan


def run_h5_simulation(
    stream: H5ObservationStream,
    pipeline: Any,
    backend: MujocoDynamicsBackend,
    *,
    config: SimulationRunnerConfig | None = None,
    execution_mode: str | None = None,
) -> SimulationRunResult:
    """Run a pipeline over H5 observations and dynamically integrate MuJoCo.

    ``backend`` must be initialized at the first stream q/dq.  The function
    intentionally accepts injected pipeline/backend instances so tests can
    exercise timing and command routing without loading neural checkpoints.
    """

    runner = config or SimulationRunnerConfig(execution_mode=execution_mode)
    if execution_mode is not None and runner.execution_mode != str(execution_mode).strip().lower().replace("-", "_"):
        raise ValueError("execution_mode conflicts with SimulationRunnerConfig")
    mode = runner.execution_mode
    if mode is None:
        pipeline_config = getattr(pipeline, "config", None)
        predictor_config = getattr(pipeline_config, "predictor", None)
        if not bool(getattr(predictor_config, "enabled", True)):
            # The predictor-disabled contract is DP -> IK -> q.  Older direct
            # IK YAML files predate ``execution.mode`` and therefore inherit
            # the global osc_qp default; do not accidentally turn their zero
            # tau placeholder into a torque simulation.
            mode = "q"
        else:
            mode = str(
                getattr(getattr(pipeline_config, "execution", None), "mode", "q")
            ).strip().lower().replace("-", "_")
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode {mode!r}")
    predictor_config = getattr(getattr(pipeline, "config", None), "predictor", None)
    if (
        bool(getattr(predictor_config, "enabled", False))
        and str(getattr(predictor_config, "inference_mode", "open_loop"))
        .strip()
        .lower()
        .replace("-", "_")
        not in {"open_loop", "openloop"}
        and not runner.allow_asynchronous
    ):
        raise ValueError(
            "offline MuJoCo inference requires predictor.inference_mode=open_loop "
            "for deterministic results; pass allow_asynchronous=True only when "
            "thread-scheduled replay is intentional"
        )
    if len(stream) < 1:
        raise ValueError("observation stream is empty")
    expected_dt = 1.0 / runner.state_rate_hz
    if not np.isclose(backend.control_dt_s, expected_dt, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "backend.control_dt_s must equal the H5 state period: "
            f"{backend.control_dt_s} != {expected_dt}"
        )
    if runner.max_steps is None:
        ticks = stream
    else:
        ticks = stream[: runner.max_steps]
    if not ticks:
        raise ValueError("selected observation range is empty")

    origin_us = int(stream.timestamps_us[0])
    # Keep the velocity immediately before the last control step.  At the next
    # tick ``sim_before.dq`` is the post-step velocity, so their difference is
    # the acceleration over exactly one control period.
    previous_sim_dq = backend.dq.copy()
    realtime_started = time.monotonic()

    records: dict[str, list[Any]] = {
        "timestamps_s": [],
        "camera_indices": [],
        "camera_age_s": [],
        "recorded_q": [],
        "recorded_dq": [],
        "recorded_ddq": [],
        "recorded_tau": [],
        "simulated_q": [],
        "simulated_dq": [],
        "simulated_ddq": [],
        "applied_torque": [],
        "command_torque": [],
        "q_target": [],
        "dq_target": [],
        "tau_target": [],
        "action_target": [],
        "dp_inference_time_s": [],
        "wm_inference_time_s": [],
        "dp_updated": [],
        "pinn_updated": [],
        "contact_state": [],
        "contact_count": [],
    }

    viewer = None
    try:
        # Pipeline state belongs to this replay only.  A failed reset is
        # surfaced instead of silently mixing histories from a previous
        # episode, while the finally block still closes model workers.
        if hasattr(pipeline, "reset"):
            pipeline.reset()
        if runner.viewer:
            try:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(backend.model, backend.data)
            except Exception as exc:  # pragma: no cover - depends on display server
                raise RuntimeError(
                    "MuJoCo viewer could not be started; use viewer=False on headless hosts"
                ) from exc
        for index, tick in enumerate(ticks):
            timestamp_s = (int(tick.timestamp_us) - origin_us) * 1.0e-6
            sim_before = backend.state()
            if runner.observation_mode == "recorded":
                q, dq, ddq, tau = tick.q, tick.dq, tick.ddq, tick.tau
            else:
                q = sim_before.q
                dq = sim_before.dq
                ddq = (dq - previous_sim_dq) / expected_dt
                # Seed the first hybrid tick with the measured H5 torque; after
                # that, feed back the torque actually produced by MuJoCo.
                tau = tick.tau if index == 0 else sim_before.applied_torque
            image_timestamp_s = (
                int(tick.camera_timestamp_us) - origin_us
            ) * 1.0e-6
            configured_image_keys = tuple(getattr(pipeline, "_image_keys", ()))
            if len(configured_image_keys) > 1:
                image = {
                    key: np.asarray(tick.image_histories[key][-1])
                    for key in configured_image_keys
                }
            else:
                image = np.asarray(tick.image)
            sample = InferenceInput(
                q=np.asarray(q, dtype=np.float64).copy(),
                dq=np.asarray(dq, dtype=np.float64).copy(),
                ddq=np.asarray(ddq, dtype=np.float64).copy(),
                tau=np.asarray(tau, dtype=np.float64).copy(),
                image=image,
                wrench_ext=np.asarray(tick.wrench_ext, dtype=np.float64).copy(),
                timestamp_s=float(timestamp_s),
                image_timestamp_s=float(image_timestamp_s),
            )
            output = _pipeline_step(pipeline, sample, tick, stream)
            state = backend.step_output(output, mode=mode)
            previous_sim_dq = sim_before.dq.copy()

            records["timestamps_s"].append(timestamp_s)
            records["camera_indices"].append(int(tick.camera_index))
            records["camera_age_s"].append(float(tick.camera_age_us) * 1.0e-6)
            records["recorded_q"].append(np.asarray(tick.q, dtype=np.float64).copy())
            records["recorded_dq"].append(np.asarray(tick.dq, dtype=np.float64).copy())
            records["recorded_ddq"].append(np.asarray(tick.ddq, dtype=np.float64).copy())
            records["recorded_tau"].append(np.asarray(tick.tau, dtype=np.float64).copy())
            records["simulated_q"].append(state.q.copy())
            records["simulated_dq"].append(state.dq.copy())
            records["simulated_ddq"].append((state.dq - sim_before.dq) / expected_dt)
            records["applied_torque"].append(state.applied_torque.copy())
            records["command_torque"].append(state.command_torque.copy())
            records["q_target"].append(
                _vector_or_nan(
                    getattr(output, "joint_position_target", None)
                    if getattr(output, "joint_position_target", None) is not None
                    else getattr(output, "joint_position_command", None)
                )
            )
            records["dq_target"].append(
                _vector_or_nan(getattr(output, "joint_velocity_target", None))
            )
            records["tau_target"].append(
                _vector_or_nan(
                    getattr(output, "torque_target", None)
                    if getattr(output, "torque_target", None) is not None
                    else getattr(output, "tau_command", None)
                )
            )
            records["action_target"].append(_vector_or_nan(getattr(output, "action_target", None)))
            records["dp_inference_time_s"].append(
                _timing_scalar(getattr(output, "dp_inference_time_s", None))
            )
            records["wm_inference_time_s"].append(
                _timing_scalar(getattr(output, "wm_inference_time_s", None))
            )
            records["dp_updated"].append(bool(getattr(output, "dp_updated", False)))
            records["pinn_updated"].append(bool(getattr(output, "pinn_updated", False)))
            records["contact_state"].append(_contact_scalar(getattr(output, "contact_state", None)))
            records["contact_count"].append(int(getattr(backend.data, "ncon", 0)))

            if viewer is not None:
                viewer.sync()
            if runner.realtime:
                target = realtime_started + timestamp_s
                remaining = target - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            if (index + 1) % 100 == 0:
                log.info(
                    "simulated %d/%d ticks mode=%s q=%s tau=%s",
                    index + 1,
                    len(ticks),
                    mode,
                    np.array2string(state.q, precision=3),
                    np.array2string(state.command_torque, precision=3),
                )
    finally:
        if viewer is not None:
            viewer.close()
        if hasattr(pipeline, "close"):
            pipeline.close()

    arrays = {
        key: np.asarray(value)
        for key, value in records.items()
    }
    arrays["timestamps_s"] = arrays["timestamps_s"].astype(np.float64)
    arrays["camera_indices"] = arrays["camera_indices"].astype(np.int64)
    arrays["dp_updated"] = arrays["dp_updated"].astype(bool)
    arrays["pinn_updated"] = arrays["pinn_updated"].astype(bool)
    arrays["dp_inference_time_s"] = arrays["dp_inference_time_s"].astype(np.float64)
    arrays["wm_inference_time_s"] = arrays["wm_inference_time_s"].astype(np.float64)
    arrays["contact_count"] = arrays["contact_count"].astype(np.int64)
    return SimulationRunResult(
        **arrays,
        execution_mode=mode,
        observation_mode=runner.observation_mode,
    )


__all__ = [
    "EXECUTION_MODES",
    "OBSERVATION_MODES",
    "SimulationRunnerConfig",
    "SimulationRunResult",
    "backend_config_from_inference",
    "build_pipeline",
    "make_observation_stream",
    "run_h5_simulation",
]
