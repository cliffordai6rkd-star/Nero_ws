from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

import numpy as np
import yaml


@dataclass(frozen=True)
class CheckpointConfig:
    path: Path
    device: str = "cuda:0"
    use_ema: bool = True
    # Visual backbone family embedded by the checkpoint. ``imagenet`` keeps
    # the checkpoint's ResNet/ImageNet weights; ``dino`` enables the optional
    # path override for DINO-based policies.
    image_encoder: str | None = None
    dino_model_path: Path | None = None


@dataclass(frozen=True)
class PredictorConfig:
    """Select the runtime contract implemented by the PINN checkpoint."""

    enabled: bool = True
    mode: str = "wrench_gru"
    inference_mode: str = "asynchronous"
    action_chunk_mode: str = "mean"
    action_step_s: float | None = None
    action_condition_fill: str = "auto"
    # ``chunk`` preserves checkpoint waypoints. ``minimum_jerk_target`` resolves
    # one target pose and follows a C2 time-scaled trajectory before replanning.
    action_execution_mode: str = "chunk"
    action_interpolation_duration_s: float | None = None
    action_interpolation_steps: int = 10


@dataclass(frozen=True)
class ExecutionConfig:
    """How a Contact WM q/tau prediction is sent to the arm."""

    # Inference exposes only direct q/tau and the MTC blend.  MIT remains a
    # firmware transport primitive used by MTC, not a standalone mode.
    mode: str = "tau"
    mit_kp: float | tuple[float, ...] = (0.0,) * 7
    mit_kd: float | tuple[float, ...] = (0.0,) * 7
    # Optional clamps applied only to the reconstructed MIT velocity target
    # and the feedback contribution.  A scalar broadcasts to all joints.
    mit_velocity_limit: float | tuple[float, ...] = 5.0
    mit_feedback_torque_limit: float | tuple[float, ...] | None = None
    # Multi-target controller (MTC) blends a data-style q/v/gravity torque
    # candidate with the WM-predicted total torque.
    mtc_alpha: float = 0.5
    # ``wm_state`` uses q_hat; ``wm_delta`` reconstructs q_cmd_hat as
    # q_hat + delta_q_hat.  ``q_hat`` and ``q_cmd_hat`` are accepted aliases.
    mtc_q_cmd_source: str = "wm_delta"


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 100
    position_tolerance_m: float = 1.0e-4
    rotation_tolerance_rad: float = 1.0e-3
    damping: float = 1.0e-3
    step_gain: float = 1.0
    maximum_iteration_step_rad: float = 0.2
    joint_position_margin_rad: float = 0.02


@dataclass(frozen=True)
class DPSamplingConfig:
    method: str = "ddim"
    num_inference_steps: int = 8
    use_tau_ext_observation: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    maximum_action_translation_step_m: float = 0.05
    maximum_action_rotation_step_rad: float = 0.5
    maximum_target_force_n: float = 40.0
    maximum_target_moment_nm: float = 5.0
    maximum_command_torque_nm: float | tuple[float, ...] = 25.0
    maximum_joint_position_step_rad: float | tuple[float, ...] = 0.1


@dataclass(frozen=True)
class RobotConfig:
    urdf_path: Path
    frame_name: str = "gripper_tcp"
    action_frame_name: str | None = None
    locked_joint_names: tuple[str, ...] = (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )


@dataclass(frozen=True)
class TimingConfig:
    enabled: bool = True
    report_interval_s: float = 1.0


@dataclass(frozen=True)
class WrenchVisualizationConfig:
    """Non-blocking comparison of estimated/predicted resultant force."""

    enabled: bool = False
    window_s: float = 10.0
    update_rate_hz: float = 20.0


@dataclass(frozen=True)
class TorqueFilterConfig:
    enabled: bool = True
    median_window: int = 3
    lowpass_cutoff_hz: float | None = 15.0
    rate_limit_nm_s: float | tuple[float, ...] | None = 50.0


@dataclass(frozen=True)
class ObservationProtectionConfig:
    """Settle observation estimators before allowing policy inference."""

    enabled: bool = False
    warmup_duration_s: float = 2.0
    wrench_median_window: int = 5
    wrench_lowpass_cutoff_hz: float = 5.0


@dataclass(frozen=True)
class RuntimeConfig:
    collection_config: Path
    arm_pair: str = "main"
    # Optional override for the camera used as the temporal/staleness anchor.
    # The normal deployment path resolves this from the checkpoint image
    # contract and the enabled cameras in ``collection_config``.
    camera: str | None = None
    maximum_state_age_s: float = 0.1
    maximum_observation_alignment_gap_s: float = 0.03
    maximum_inference_steps: int | None = None
    command_kd: tuple[float, ...] = (0.0,) * 7


@dataclass(frozen=True)
class ArchitectureConfig:
    """Selection for the modular orchestration layer.

    ``enabled`` remains false by default while legacy runtime compatibility is
    maintained.  New deployments can opt into the same contracts without
    changing checkpoint-specific settings.
    """

    enabled: bool = False
    policy_type: str = "legacy_pipeline"
    world_model_type: str = "none"


@dataclass(frozen=True)
class InferenceConfig:
    dp_checkpoint: CheckpointConfig
    robot: RobotConfig
    runtime: RuntimeConfig
    # Semantic contract of each seven-dimensional DP action.
    # ``eepose`` is [x,y,z,qx,qy,qz,qw]; ``joint`` is absolute arm q.  Contact
    # WM action conditions are always absolute ee_pose; joint actions are FK
    # converted at the WM boundary.
    action: str = "eepose"
    # Canonical name for the Contact World Model checkpoint.  The historical
    # ``pinn_checkpoint`` field remains as a constructor/read compatibility
    # alias for older scripts and tests.
    contactworldmodel: CheckpointConfig | None = None
    pinn_checkpoint: CheckpointConfig | None = None
    dp_sampling: DPSamplingConfig = field(default_factory=DPSamplingConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    torque_filter: TorqueFilterConfig = field(default_factory=TorqueFilterConfig)
    observation_protection: ObservationProtectionConfig = field(
        default_factory=ObservationProtectionConfig
    )
    timing: TimingConfig = field(default_factory=TimingConfig)
    wrench_visualization: WrenchVisualizationConfig = field(
        default_factory=WrenchVisualizationConfig
    )
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    def __post_init__(self) -> None:
        # A config loaded from YAML normally supplies only the canonical field;
        # older Python callers may still supply only ``pinn_checkpoint``.  Do
        # not overwrite a deliberately different alias during dataclasses.replace
        # so both migration paths remain usable.
        if self.contactworldmodel is None and self.pinn_checkpoint is not None:
            object.__setattr__(self, "contactworldmodel", self.pinn_checkpoint)
        elif self.pinn_checkpoint is None and self.contactworldmodel is not None:
            object.__setattr__(self, "pinn_checkpoint", self.contactworldmodel)


T = TypeVar("T")


def resolve_camera_key(
    configured: str | None,
    checkpoint_image_keys: Iterable[str],
    collection_camera_keys: Iterable[str] | None = None,
    *,
    preferred: Iterable[str] = ("wrist",),
) -> str:
    """Resolve the temporal camera anchor from model and acquisition contracts.

    ``checkpoint_image_keys`` describes the keys the policy actually consumes;
    ``collection_camera_keys`` describes enabled acquisition streams.  An
    explicit ``configured`` value remains a supported override, but is always
    checked against both contracts; every checkpoint key must be present in the
    enabled collection set.  When no override is supplied, a unique
    model/acquisition intersection is selected.  Multi-camera datasets use the
    wrist stream as the conventional anchor because the training converters
    timestamp/aligned rows against ``cameras/wrist``; any other ambiguous case
    is rejected instead of silently choosing dictionary order.
    """

    model_keys = tuple(dict.fromkeys(str(key).strip() for key in checkpoint_image_keys))
    model_keys = tuple(key for key in model_keys if key)
    if not model_keys:
        raise ValueError("checkpoint must expose at least one image key")

    collection_keys: tuple[str, ...] | None
    if collection_camera_keys is None:
        collection_keys = None
        candidates = list(model_keys)
    else:
        collection_keys = tuple(
            dict.fromkeys(str(key).strip() for key in collection_camera_keys)
        )
        collection_keys = tuple(key for key in collection_keys if key)
        missing = [key for key in model_keys if key not in collection_keys]
        if missing:
            raise ValueError(
                "checkpoint requires cameras that are not enabled in the "
                "collection config: missing="
                f"{missing}, checkpoint image keys={list(model_keys)}, "
                f"enabled collection cameras={list(collection_keys)}"
            )
        candidates = [key for key in model_keys if key in collection_keys]
        if not candidates:
            raise ValueError(
                "no camera satisfies both contracts: checkpoint image keys="
                f"{list(model_keys)}, enabled collection cameras={list(collection_keys)}"
            )

    selected = None if configured is None else str(configured).strip()
    if selected == "":
        raise ValueError("runtime.camera must be a non-empty camera name or null")
    if selected is not None:
        if selected not in model_keys:
            raise ValueError(
                f"runtime.camera={selected!r} is not declared by the checkpoint; "
                f"image keys={list(model_keys)}"
            )
        if collection_keys is not None and selected not in collection_keys:
            raise ValueError(
                f"runtime.camera={selected!r} is not enabled in the collection "
                f"config; cameras={list(collection_keys)}"
            )
        return selected

    if len(candidates) == 1:
        return candidates[0]
    for key in preferred:
        preferred_key = str(key).strip()
        if preferred_key in candidates:
            return preferred_key
    raise ValueError(
        "camera anchor is ambiguous; checkpoint/collection intersection="
        f"{list(candidates)}. Set runtime.camera explicitly or declare a "
        "primary camera in the acquisition configuration."
    )


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load runtime settings only; model architecture always comes from checkpoints."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("inference config root must be a mapping")
    _reject_unknown(raw, InferenceConfig, "config")
    base = config_path.parent
    dp = _checkpoint(raw.get("dp_checkpoint"), base, "dp_checkpoint")
    action = str(raw.get("action", "eepose")).strip().lower().replace("-", "_")
    if action not in {"eepose", "joint"}:
        raise ValueError("action must be 'eepose' or 'joint'")
    dp_sampling = _dataclass_from_mapping(
        DPSamplingConfig,
        raw.get("dp_sampling", {}),
        "dp_sampling",
    )
    sampling_method = dp_sampling.method.strip().lower()
    if sampling_method not in {"ddim", "ddpm"}:
        raise ValueError("dp_sampling.method must be 'ddim' or 'ddpm'")
    if (
        not isinstance(dp_sampling.num_inference_steps, int)
        or isinstance(dp_sampling.num_inference_steps, bool)
        or dp_sampling.num_inference_steps < 1
    ):
        raise ValueError("dp_sampling.num_inference_steps must be positive")
    if not isinstance(dp_sampling.use_tau_ext_observation, bool):
        raise ValueError("dp_sampling.use_tau_ext_observation must be a boolean")
    dp_sampling = DPSamplingConfig(
        method=sampling_method,
        num_inference_steps=int(dp_sampling.num_inference_steps),
        use_tau_ext_observation=dp_sampling.use_tau_ext_observation,
    )
    predictor = _dataclass_from_mapping(
        PredictorConfig,
        raw.get("predictor", {}),
        "predictor",
    )
    if not isinstance(predictor.enabled, bool):
        raise ValueError("predictor.enabled must be a boolean")
    predictor_mode = predictor.mode.strip().lower()
    if predictor_mode in {
        "world_model",
        "world_model_v3",
        "world_model_v4",
        "world_model_v5",
        "swm",
        "swm_opd",
        "torque_world_model",
        "torque_world_model_opd",
        "torque_wm",
        "torque_wm_opd",
        "contact_wm",
        "contact_wm_opd",
    }:
        predictor_mode = (
            "contact_world_model_opd"
            if predictor_mode.endswith("_opd")
            else "contact_world_model"
        )
    if predictor_mode not in {
        "wrench_gru",
        "contact_world_model",
        "contact_world_model_opd",
    }:
        raise ValueError(
            "predictor.mode must be 'wrench_gru', 'contact_world_model', "
            "or 'contact_world_model_opd'"
        )
    inference_mode = predictor.inference_mode.strip().lower().replace("-", "_")
    if inference_mode == "async":
        inference_mode = "asynchronous"
    if inference_mode not in {"asynchronous", "open_loop"}:
        raise ValueError(
            "predictor.inference_mode must be 'asynchronous' or 'open_loop'"
        )
    action_chunk_mode = predictor.action_chunk_mode.strip().lower()
    if action_chunk_mode not in {"first", "mean", "all", "last", "middle"}:
        raise ValueError(
            "predictor.action_chunk_mode must be 'first', 'mean', 'all', or 'last'"
        )
    action_step_s = predictor.action_step_s
    if action_step_s is not None:
        action_step_s = float(action_step_s)
        if not np.isfinite(action_step_s) or action_step_s <= 0.0:
            raise ValueError("predictor.action_step_s must be positive or null")
    action_condition_fill = predictor.action_condition_fill.strip().lower()
    if action_condition_fill not in {"auto", "chunk", "hold"}:
        raise ValueError(
            "predictor.action_condition_fill must be 'auto', 'chunk', or 'hold'"
        )
    action_execution_mode = (
        predictor.action_execution_mode.strip().lower().replace("-", "_")
    )
    if action_execution_mode == "linear_target":
        action_execution_mode = "minimum_jerk_target"
    if action_execution_mode not in {"chunk", "minimum_jerk_target"}:
        raise ValueError(
            "predictor.action_execution_mode must be 'chunk' or "
            "'minimum_jerk_target'"
        )
    if action_execution_mode == "minimum_jerk_target" and action_chunk_mode not in {
        "first",
        "mean",
        "last",
        "middle"
    }:
        raise ValueError(
            "predictor.action_chunk_mode must be 'first', 'mean', or 'last' when "
            "action_execution_mode='minimum_jerk_target'"
        )
    action_interpolation_duration_s = predictor.action_interpolation_duration_s
    if action_interpolation_duration_s is not None:
        action_interpolation_duration_s = float(action_interpolation_duration_s)
        if (
            not np.isfinite(action_interpolation_duration_s)
            or action_interpolation_duration_s <= 0.0
        ):
            raise ValueError(
                "predictor.action_interpolation_duration_s must be positive or null"
            )
    action_interpolation_steps = predictor.action_interpolation_steps
    if (
        not isinstance(action_interpolation_steps, int)
        or isinstance(action_interpolation_steps, bool)
        or action_interpolation_steps < 1
    ):
        raise ValueError(
            "predictor.action_interpolation_steps must be a positive integer"
        )
    predictor = PredictorConfig(
        enabled=predictor.enabled,
        mode=predictor_mode,
        inference_mode=inference_mode,
        action_chunk_mode=action_chunk_mode,
        action_step_s=action_step_s,
        action_condition_fill=action_condition_fill,
        action_execution_mode=action_execution_mode,
        action_interpolation_duration_s=action_interpolation_duration_s,
        action_interpolation_steps=action_interpolation_steps,
    )
    execution = _dataclass_from_mapping(
        ExecutionConfig,
        raw.get("execution", {}),
        "execution",
    )
    execution_mode = execution.mode.strip().lower().replace("-", "_")
    if execution_mode not in {"mtc", "q", "tau"}:
        raise ValueError(
            "execution.mode must be one of 'mtc', 'q', or 'tau'"
        )
    kp_array = np.asarray(execution.mit_kp, dtype=np.float64)
    kd_array = np.asarray(execution.mit_kd, dtype=np.float64)
    if kp_array.ndim == 0:
        kp_array = np.repeat(kp_array, 7)
    else:
        kp_array = kp_array.reshape(-1)
    if kd_array.ndim == 0:
        kd_array = np.repeat(kd_array, 7)
    else:
        kd_array = kd_array.reshape(-1)
    if kp_array.shape != (7,) or kd_array.shape != (7,):
        raise ValueError("execution.mit_kp and execution.mit_kd must contain seven values")
    mit_kp = tuple(float(value) for value in kp_array)
    mit_kd = tuple(float(value) for value in kd_array)
    velocity_limit = np.asarray(execution.mit_velocity_limit, dtype=np.float64)
    if velocity_limit.ndim == 0:
        velocity_limit = np.repeat(velocity_limit, 7)
    else:
        velocity_limit = velocity_limit.reshape(-1)
    if (
        velocity_limit.shape != (7,)
        or not np.isfinite(velocity_limit).all()
        or np.any(velocity_limit <= 0.0)
    ):
        raise ValueError("execution.mit_velocity_limit must be positive and finite")
    feedback_limit = execution.mit_feedback_torque_limit
    if feedback_limit is not None:
        feedback_limit_array = np.asarray(feedback_limit, dtype=np.float64)
        if feedback_limit_array.ndim == 0:
            feedback_limit_array = np.repeat(feedback_limit_array, 7)
        else:
            feedback_limit_array = feedback_limit_array.reshape(-1)
        if (
            feedback_limit_array.shape != (7,)
            or not np.isfinite(feedback_limit_array).all()
            or np.any(feedback_limit_array <= 0.0)
        ):
            raise ValueError(
                "execution.mit_feedback_torque_limit must be positive and finite or null"
            )
        feedback_limit = tuple(float(value) for value in feedback_limit_array)
    if (
        not np.isfinite(kp_array).all()
        or not np.isfinite(kd_array).all()
        or np.any(kp_array < 0.0)
        or np.any(kd_array < 0.0)
    ):
        raise ValueError("execution.mit_kp and execution.mit_kd must be finite and non-negative")
    try:
        mtc_alpha = float(execution.mtc_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution.mtc_alpha must be a finite value in [0, 1]") from exc
    if not np.isfinite(mtc_alpha) or not 0.0 <= mtc_alpha <= 1.0:
        raise ValueError("execution.mtc_alpha must be a finite value in [0, 1]")
    mtc_q_cmd_source = (
        str(execution.mtc_q_cmd_source).strip().lower().replace("-", "_")
    )
    mtc_q_cmd_source = {
        "q_hat": "wm_state",
        "qhat": "wm_state",
        "wm_q": "wm_state",
        "q_cmd_hat": "wm_delta",
        "qcmd_hat": "wm_delta",
        "wm_q_cmd": "wm_delta",
    }.get(mtc_q_cmd_source, mtc_q_cmd_source)
    if mtc_q_cmd_source not in {"wm_state", "wm_delta"}:
        raise ValueError(
            "execution.mtc_q_cmd_source must be 'wm_state'/'q_hat' or "
            "'wm_delta'/'q_cmd_hat'"
        )
    execution = ExecutionConfig(
        mode=execution_mode,
        mit_kp=mit_kp,
        mit_kd=mit_kd,
        mit_velocity_limit=tuple(float(value) for value in velocity_limit),
        mit_feedback_torque_limit=feedback_limit,
        mtc_alpha=mtc_alpha,
        mtc_q_cmd_source=mtc_q_cmd_source,
    )
    contact_raw = raw.get("contactworldmodel")
    legacy_pinn_raw = raw.get("pinn_checkpoint")
    if contact_raw is not None and legacy_pinn_raw is not None:
        raise ValueError(
            "configure only one of contactworldmodel or pinn_checkpoint"
        )
    checkpoint_raw = contact_raw if contact_raw is not None else legacy_pinn_raw
    checkpoint_name = "contactworldmodel" if contact_raw is not None else "pinn_checkpoint"
    if checkpoint_raw is None:
        if predictor.enabled:
            raise ValueError(
                "contactworldmodel is required when predictor.enabled=true"
            )
        contactworldmodel = None
    else:
        contactworldmodel = _checkpoint(checkpoint_raw, base, checkpoint_name)
    robot_raw = _mapping(raw.get("robot"), "robot")
    _reject_unknown(robot_raw, RobotConfig, "robot")
    urdf_path = _required_path(robot_raw.get("urdf_path"), base, "robot.urdf_path")
    robot = RobotConfig(
        urdf_path=urdf_path,
        frame_name=str(robot_raw.get("frame_name", "gripper_base")),
        action_frame_name=(
            None
            if robot_raw.get("action_frame_name") is None
            else str(robot_raw["action_frame_name"])
        ),
        locked_joint_names=tuple(
            str(value)
            for value in robot_raw.get(
                "locked_joint_names",
                ("gripper", "gripper_joint1", "gripper_joint2"),
            )
        ),
    )
    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    _reject_unknown(runtime_raw, RuntimeConfig, "runtime")
    runtime_camera_raw = runtime_raw.get("camera")
    runtime_camera = (
        None
        if runtime_camera_raw is None
        else str(runtime_camera_raw).strip()
    )
    if runtime_camera == "":
        raise ValueError("runtime.camera must be a non-empty camera name or null")
    runtime = RuntimeConfig(
        collection_config=_required_path(
            runtime_raw.get("collection_config"),
            base,
            "runtime.collection_config",
        ),
        arm_pair=str(runtime_raw.get("arm_pair", "main")),
        camera=runtime_camera,
        maximum_state_age_s=float(runtime_raw.get("maximum_state_age_s", 0.1)),
        maximum_observation_alignment_gap_s=float(
            runtime_raw.get("maximum_observation_alignment_gap_s", 0.03)
        ),
        maximum_inference_steps=runtime_raw.get("maximum_inference_steps"),
        command_kd=tuple(float(value) for value in runtime_raw.get("command_kd", (0.0,) * 7)),
    )
    if runtime.maximum_state_age_s <= 0:
        raise ValueError("runtime.maximum_state_age_s must be positive")
    if (
        not np.isfinite(runtime.maximum_observation_alignment_gap_s)
        or runtime.maximum_observation_alignment_gap_s <= 0
    ):
        raise ValueError(
            "runtime.maximum_observation_alignment_gap_s must be positive and finite"
        )
    if runtime.maximum_inference_steps is not None and (
        not isinstance(runtime.maximum_inference_steps, int)
        or isinstance(runtime.maximum_inference_steps, bool)
        or runtime.maximum_inference_steps < 1
    ):
        raise ValueError(
            "runtime.maximum_inference_steps must be a positive integer or null"
        )
    if len(runtime.command_kd) != 7 or any(value < 0 for value in runtime.command_kd):
        raise ValueError("runtime.command_kd must contain seven non-negative values")
    safety = _dataclass_from_mapping(SafetyConfig, raw.get("safety", {}), "safety")
    joint_step = np.asarray(
        safety.maximum_joint_position_step_rad,
        dtype=np.float64,
    )
    if joint_step.ndim > 0:
        joint_step = joint_step.reshape(-1)
        if joint_step.size != 7:
            raise ValueError(
                "safety.maximum_joint_position_step_rad must be scalar or a 7-vector"
            )
    if not np.isfinite(joint_step).all() or np.any(joint_step <= 0):
        raise ValueError(
            "safety.maximum_joint_position_step_rad must be positive and finite"
        )
    ik = _dataclass_from_mapping(IKConfig, raw.get("ik", {}), "ik")
    if ik.max_iterations < 1:
        raise ValueError("ik.max_iterations must be positive")
    for name in (
        "position_tolerance_m",
        "rotation_tolerance_rad",
        "damping",
        "step_gain",
        "maximum_iteration_step_rad",
    ):
        if not np.isfinite(getattr(ik, name)) or getattr(ik, name) <= 0:
            raise ValueError(f"ik.{name} must be positive and finite")
    if not np.isfinite(ik.joint_position_margin_rad) or ik.joint_position_margin_rad < 0:
        raise ValueError("ik.joint_position_margin_rad must be non-negative and finite")
    torque_filter = _dataclass_from_mapping(
        TorqueFilterConfig,
        raw.get("torque_filter", {}),
        "torque_filter",
    )
    if torque_filter.median_window < 1 or torque_filter.median_window % 2 == 0:
        raise ValueError("torque_filter.median_window must be a positive odd integer")
    if (
        torque_filter.lowpass_cutoff_hz is not None
        and torque_filter.lowpass_cutoff_hz <= 0
    ):
        raise ValueError("torque_filter.lowpass_cutoff_hz must be positive or null")
    if torque_filter.rate_limit_nm_s is not None:
        rate_limit = np.asarray(torque_filter.rate_limit_nm_s, dtype=np.float64)
        if rate_limit.ndim > 0:
            rate_limit = rate_limit.reshape(-1)
            if rate_limit.size != 7:
                raise ValueError(
                    "torque_filter.rate_limit_nm_s must be scalar, null, or a 7-vector"
                )
        if not np.isfinite(rate_limit).all() or np.any(rate_limit <= 0):
            raise ValueError("torque_filter.rate_limit_nm_s must be positive and finite")
    observation_protection = _dataclass_from_mapping(
        ObservationProtectionConfig,
        raw.get("observation_protection", {}),
        "observation_protection",
    )
    if not isinstance(observation_protection.enabled, bool):
        raise ValueError("observation_protection.enabled must be a boolean")
    if (
        not np.isfinite(observation_protection.warmup_duration_s)
        or observation_protection.warmup_duration_s < 0
    ):
        raise ValueError(
            "observation_protection.warmup_duration_s must be non-negative and finite"
        )
    median_window = observation_protection.wrench_median_window
    if (
        not isinstance(median_window, int)
        or isinstance(median_window, bool)
        or median_window < 1
        or median_window % 2 == 0
    ):
        raise ValueError(
            "observation_protection.wrench_median_window must be a positive odd integer"
        )
    cutoff_hz = observation_protection.wrench_lowpass_cutoff_hz
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(
            "observation_protection.wrench_lowpass_cutoff_hz must be positive and finite"
        )
    timing = _dataclass_from_mapping(TimingConfig, raw.get("timing", {}), "timing")
    if timing.report_interval_s <= 0:
        raise ValueError("timing.report_interval_s must be positive")
    wrench_visualization = _dataclass_from_mapping(
        WrenchVisualizationConfig,
        raw.get("wrench_visualization", {}),
        "wrench_visualization",
    )
    if not isinstance(wrench_visualization.enabled, bool):
        raise ValueError("wrench_visualization.enabled must be a boolean")
    for name in ("window_s", "update_rate_hz"):
        value = getattr(wrench_visualization, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"wrench_visualization.{name} must be positive and finite")
    architecture = _dataclass_from_mapping(
        ArchitectureConfig,
        raw.get("architecture", {}),
        "architecture",
    )
    if not isinstance(architecture.enabled, bool):
        raise ValueError("architecture.enabled must be a boolean")
    policy_type = str(architecture.policy_type).strip().lower()
    world_model_type = str(architecture.world_model_type).strip().lower()
    world_model_type = {
        "swm": "contact_wm",
        "swm_opd": "contact_wm",
        "torque_world_model": "contact_wm",
        "torque_world_model_opd": "contact_wm",
        "torque_wm": "contact_wm",
        "torque_wm_opd": "contact_wm",
        "world_model": "contact_wm",
        "world_model_v3": "contact_wm",
        "world_model_v4": "contact_wm",
        "world_model_v5": "contact_wm",
        "contact_world_model": "contact_wm",
        "contact_world_model_opd": "contact_wm",
    }.get(world_model_type, world_model_type)
    if not policy_type:
        raise ValueError("architecture.policy_type must be non-empty")
    if not world_model_type:
        raise ValueError("architecture.world_model_type must be non-empty")
    architecture = ArchitectureConfig(
        enabled=architecture.enabled,
        policy_type=policy_type,
        world_model_type=world_model_type,
    )
    return InferenceConfig(
        dp_checkpoint=dp,
        contactworldmodel=contactworldmodel,
        robot=robot,
        runtime=runtime,
        action=action,
        dp_sampling=dp_sampling,
        ik=ik,
        predictor=predictor,
        execution=execution,
        safety=safety,
        torque_filter=torque_filter,
        observation_protection=observation_protection,
        timing=timing,
        wrench_visualization=wrench_visualization,
        architecture=architecture,
    )


def _checkpoint(value: Any, base: Path, name: str) -> CheckpointConfig:
    raw = _mapping(value, name)
    _reject_unknown(raw, CheckpointConfig, name)
    image_encoder = raw.get("image_encoder")
    if image_encoder is not None:
        image_encoder = str(image_encoder).strip().lower().replace("-", "_")
        aliases = {
            "imagenet": "imagenet",
            "imagenet1k": "imagenet",
            "imagenet1k_v1": "imagenet",
            "resnet": "imagenet",
            "dino": "dino",
            "dinov2": "dino",
            "dinov3": "dino",
        }
        if image_encoder not in aliases:
            raise ValueError(
                f"{name}.image_encoder must be 'imagenet' or 'dino', "
                f"got {image_encoder!r}"
            )
        image_encoder = aliases[image_encoder]
    return CheckpointConfig(
        path=_required_path(raw.get("path"), base, f"{name}.path"),
        device=str(raw.get("device", "cuda:0")),
        use_ema=bool(raw.get("use_ema", True)),
        image_encoder=image_encoder,
        dino_model_path=(
            None
            if raw.get("dino_model_path") is None
            else _required_path(
                raw.get("dino_model_path"),
                base,
                f"{name}.dino_model_path",
            )
        ),
    )


def _dataclass_from_mapping(cls: type[T], value: Any, name: str) -> T:
    raw = _mapping(value, name)
    _reject_unknown(raw, cls, name)
    allowed = {item.name for item in fields(cls)}
    values = {key: val for key, val in raw.items() if key in allowed}
    for key, val in tuple(values.items()):
        annotation = next(item.type for item in fields(cls) if item.name == key)
        if "tuple" in str(annotation) and isinstance(val, list):
            values[key] = tuple(val)
    return cls(**values)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _reject_unknown(raw: Mapping[str, Any], cls: type[Any], name: str) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {unknown}")
