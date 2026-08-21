from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TELEOP_MODES = {"master_slave", "meta_quest3_vr", "keyboard_3d_mouse"}


@dataclass(frozen=True)
class StateParamConfig:
    enabled: bool = True
    lowpass: bool = False
    lowpass_cutoff_hz: float = 12.0
    mean_window: int = 1
    median_window: int = 1


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    prefix: str = "episode"
    discard_initial_s: float = 2.0


@dataclass(frozen=True)
class DynamicsProcessingConfig:
    enabled: bool = False
    state_method: str = "finite_difference"
    spline_smoothing_rad2: float = 1.0e-5
    fourier_fundamental_hz: float = 0.1
    fourier_harmonics: int = 8
    torque_lowpass_hz: float = 12.0
    torque_median_window: int = 3
    min_samples: int = 20


@dataclass(frozen=True)
class InverseDynamicsConfig:
    urdf_path: Path = Path("urdf/nero/nero_with_gripper.urdf")
    manifest_path: Path | None = None
    delay_s: float = 0.0
    locked_joint_names: tuple[str, ...] = (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class RealtimePlotConfig:
    enabled: bool = False
    window_s: float = 10.0
    update_rate_hz: float = 20.0


@dataclass(frozen=True)
class SequenceCheckpointConfig:
    checkpoint_path: Path | None = None
    device: str = "cpu"
    observation_sample_rate_hz: float | None = None
    horizon: int | None = None
    input_keys: tuple[str, ...] | None = None
    output_key: str | None = None


@dataclass(frozen=True)
class CausalKalmanConfig:
    position_std: tuple[float, ...] = (5.0e-4,) * 7
    velocity_std: tuple[float, ...] = (3.0e-2,) * 7
    jerk_std: tuple[float, ...] = (2.0,) * 7
    initial_position_std: tuple[float, ...] = (1.0e-2,) * 7
    initial_velocity_std: tuple[float, ...] = (2.0e-1,) * 7
    initial_acceleration_std: tuple[float, ...] = (5.0,) * 7
    max_gap_s: float = 0.1


@dataclass(frozen=True)
class TauExtFilterConfig:
    enabled: bool = True
    mode: str = "hampel_butterworth"
    window: int = 5
    cutoff_hz: float = 8.0
    hampel_n_sigma: float = 3.0
    order: int = 4
    sample_rate_hz: float = 100.0


@dataclass(frozen=True)
class SourceButterworthFilterConfig:
    """Variable-dt anti-alias filter applied before model stride selection."""

    enabled: bool = False
    cutoff_hz: float = 15.0
    order: int = 2


@dataclass(frozen=True)
class TauExtInferenceConfig:
    enabled: bool = False
    feedback_source: str = "tau_free"
    observation_gap_warning_s: float = 0.06
    maximum_prediction_age_s: float = 0.06
    tau_f: SequenceCheckpointConfig = field(default_factory=SequenceCheckpointConfig)
    tau_next: SequenceCheckpointConfig = field(default_factory=SequenceCheckpointConfig)
    source_butterworth_filter: SourceButterworthFilterConfig = field(
        default_factory=SourceButterworthFilterConfig
    )
    state_estimator: CausalKalmanConfig = field(default_factory=CausalKalmanConfig)
    tau_ext_filter: TauExtFilterConfig = field(default_factory=TauExtFilterConfig)
    inverse_dynamics: InverseDynamicsConfig = field(default_factory=InverseDynamicsConfig)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool = True
    backend: str = "orbbec_dabai"
    device: str | int | None = None
    pixel_format: str = "MJPG"
    buffer_size: int = 1
    startup_timeout_s: float = 3.0
    warmup_s: float = 0.0
    frame_timeout_s: float = 1.0
    visualize: bool = False
    serial_number: str | None = None
    width: int = 640
    height: int = 480
    fps: float = 30.0
    exposure: int | None = None
    exposure_dynamic_framerate: bool | None = None
    depth: bool = False
    crop: tuple[int | None, int | None, int | None, int | None] = (0, None, 0, None)
    output_size: tuple[int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmEndpointConfig:
    name: str
    can_id: int | None = None
    channel: str = "can0"
    usb_serial: str | None = None
    interface: str = "socketcan"
    bitrate: int = 1_000_000
    firmware: str = "V120"
    rest_q: tuple[float, ...] = ()
    config_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmPairConfig:
    name: str
    leader: ArmEndpointConfig
    follower: ArmEndpointConfig


@dataclass(frozen=True)
class BilateralMitConfig:
    leader_kp: tuple[float, ...] = (0.0,) * 7
    leader_kd: tuple[float, ...] = (0.5,) * 7
    follower_kp: tuple[float, ...] = (8.0,) * 7
    follower_kd: tuple[float, ...] = (0.8,) * 7
    position_scale: tuple[float, ...] = (1.0,) * 7
    leader_gravity_scale: tuple[float, ...] = (1.0,) * 7
    follower_gravity_scale: tuple[float, ...] = (1.0,) * 7
    force_feedback_gain: tuple[float, ...] = (0.05,) * 7
    force_feedback_sign: tuple[float, ...] = (1.0,) * 7
    force_feedback_deadband_nm: tuple[float, ...] = (0.20,) * 7
    force_feedback_limit_nm: tuple[float, ...] = (1.0, 1.0, 0.8, 0.8, 0.4, 0.4, 0.4)
    leader_torque_limit_nm: tuple[float, ...] = (20.0, 20.0, 14.0, 14.0, 7.0, 7.0, 7.0)
    follower_torque_limit_nm: tuple[float, ...] = (20.0, 20.0, 14.0, 14.0, 7.0, 7.0, 7.0)
    feedback_torque_rate_limit_nm_s: tuple[float, ...] = (10.0,) * 7
    force_feedback_lowpass_hz: float | None = 5.0
    force_feedback_ramp_s: float = 2.0
    joint_limit_margin_rad: float = 0.10


@dataclass(frozen=True)
class CommandConfig:
    sample_rate_hz: float = 100.0
    control_watchdog_timeout_s: float = 0.05
    idle_rate_hz: float = 30.0
    input_ready_timeout_s: float = 3.0
    reset_on_start: bool = False
    reset_after_episode: bool = True
    reset_timeout_s: float = 10.0
    reset_wait_s: float = 0.8
    reset_test_sample_time: int = 5
    reset_error_limit_rad: float = 0.02
    joint_step_limit_rad: float | None = 0.08
    control_mode: str = "mit"
    bilateral_mit: BilateralMitConfig = field(default_factory=BilateralMitConfig)
    role_switch_settle_s: float = 0.3
    role_switch_timeout_s: float = 3.0
    reset_interpolation_enabled: bool = True
    reset_interpolation_rate_hz: float = 30.0
    reset_joint_speed_rad_s: float = 1.0
    reset_min_duration_s: float = 0.2
    reset_max_step_rad: float = 0.05


@dataclass(frozen=True)
class TeleopConfig:
    mode: str = "master_slave"
    protocol: str = "can"
    backend: str = "pyagxarm"
    master_slave: tuple[ArmPairConfig, ...] = ()
    command: CommandConfig = field(default_factory=CommandConfig)


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool = True
    effector: str = "AGX_GRIPPER"
    attach_to: str = "follower"
    teleop_enabled: bool = False
    scale: float = 1.0
    offset_m: float = 0.0
    min_width_m: float = 0.0
    max_width_m: float = 0.07
    force_n: float = 1.0
    command_rate_hz: float = 30.0
    deadband_m: float = 0.0005
    keepalive_s: float = 0.5


@dataclass(frozen=True)
class CollectionConfig:
    teleop: TeleopConfig
    output: OutputConfig
    cameras: tuple[CameraConfig, ...] = ()
    gripper: GripperConfig = field(default_factory=GripperConfig)
    realtime_plot: RealtimePlotConfig = field(default_factory=RealtimePlotConfig)
    tau_ext_inference: TauExtInferenceConfig = field(default_factory=TauExtInferenceConfig)
    dynamics_processing: DynamicsProcessingConfig = field(default_factory=DynamicsProcessingConfig)
    robot_states: dict[str, StateParamConfig] = field(default_factory=dict)
    raw_yaml: str = ""


DEFAULT_STATE_PARAMS = {
    "q": StateParamConfig(
        enabled=True,
        lowpass=True,
        lowpass_cutoff_hz=10.0,
        mean_window=5,
    ),
    "velocity": StateParamConfig(enabled=True, lowpass=True, lowpass_cutoff_hz=6.0),
    "ee_pose": StateParamConfig(enabled=True, lowpass=False),
    "torque": StateParamConfig(enabled=True, lowpass=False),
    "tau_id": StateParamConfig(enabled=True, lowpass=False),
    "tau_id_filtered": StateParamConfig(enabled=True, lowpass=False),
    "current": StateParamConfig(enabled=False, lowpass=False),
}


def load_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path).expanduser().resolve()
    raw_yaml = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_yaml) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    teleop = _parse_teleop(data.get("teleop", {}))
    if teleop.mode not in SUPPORTED_TELEOP_MODES:
        raise ValueError(f"Unsupported teleop.mode={teleop.mode!r}; choose one of {sorted(SUPPORTED_TELEOP_MODES)}")
    if teleop.mode != "master_slave":
        raise NotImplementedError(f"teleop.mode={teleop.mode!r} is reserved but not implemented yet")
    if teleop.protocol != "can":
        raise NotImplementedError("Only CAN protocol is implemented in this first pass")
    if not teleop.master_slave:
        raise ValueError("teleop.master_slave.arm_pairs must contain at least one leader/follower pair")

    output = _parse_output(data.get("output", {}), config_path.parent)
    cameras = tuple(_parse_camera(item) for item in data.get("cameras", []) if item.get("enabled", True))
    gripper = _parse_gripper(data.get("gripper", {}))
    realtime_plot = _parse_realtime_plot(data.get("realtime_plot", {}))
    tau_ext_inference = _parse_tau_ext_inference(
        data.get("tau_ext_inference", {}),
        config_path.parent,
    )
    if "tau_f_inference" in data:
        raise ValueError(
            "tau_f_inference was removed; configure the two-model "
            "tau_ext_inference block instead."
        )
    if realtime_plot.enabled and not tau_ext_inference.enabled:
        raise ValueError(
            "realtime_plot.enabled=true requires tau_ext_inference.enabled=true"
        )
    dynamics_processing = _parse_dynamics_processing(data.get("dynamics_processing", {}))
    robot_states = _parse_state_params(data.get("robot_states", {}))
    return CollectionConfig(
        teleop=teleop,
        output=output,
        cameras=cameras,
        gripper=gripper,
        realtime_plot=realtime_plot,
        tau_ext_inference=tau_ext_inference,
        dynamics_processing=dynamics_processing,
        robot_states=robot_states,
        raw_yaml=raw_yaml,
    )


def _parse_teleop(data: dict[str, Any]) -> TeleopConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop must be a mapping")
    master_slave_data = data.get("master_slave", {})
    arm_pairs_data = master_slave_data.get("arm_pairs", []) if isinstance(master_slave_data, dict) else []
    pairs = tuple(_parse_arm_pair(item) for item in arm_pairs_data)
    return TeleopConfig(
        mode=str(data.get("mode", "master_slave")),
        protocol=str(data.get("protocol", "can")),
        backend=str(data.get("backend", "pyagxarm")),
        master_slave=pairs,
        command=_parse_command(data.get("command", {})),
    )


def _parse_command(data: dict[str, Any]) -> CommandConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop.command must be a mapping")
    removed = {
        "mit",
        "teleop_mapping",
        "pre_teleop_align_enabled",
        "pre_teleop_align_error_limit_rad",
        "idle_follow_enabled",
    }.intersection(data)
    if removed:
        raise ValueError(
            "legacy teleop.command fields were removed; configure bilateral_mit only: "
            + ", ".join(sorted(removed))
        )
    role_switch_settle_s = float(data.get("role_switch_settle_s", 0.3))
    role_switch_timeout_s = float(data.get("role_switch_timeout_s", 3.0))
    reset_interpolation_rate_hz = float(data.get("reset_interpolation_rate_hz", 30.0))
    reset_joint_speed_rad_s = float(data.get("reset_joint_speed_rad_s", 1.0))
    reset_min_duration_s = float(data.get("reset_min_duration_s", 0.2))
    reset_max_step_rad = float(data.get("reset_max_step_rad", 0.05))
    control_mode = str(data.get("control_mode", "mit")).lower()
    removed_timing = {
        "state_alignment_delay_s",
        "maximum_state_source_skew_s",
        "maximum_arm_pair_skew_s",
        "maximum_can_frame_gap_s",
    }.intersection(data)
    if removed_timing:
        raise ValueError(
            "event-alignment teleop.command fields were removed: "
            + ", ".join(sorted(removed_timing))
        )
    sample_rate_hz = float(data.get("sample_rate_hz", 100.0))
    control_watchdog_timeout_s = float(data.get("control_watchdog_timeout_s", 0.05))
    if control_mode not in {"mit", "position"}:
        raise ValueError("teleop.command.control_mode must be mit or position")
    if not isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("teleop.command.sample_rate_hz must be positive and finite")
    if not isfinite(control_watchdog_timeout_s) or control_watchdog_timeout_s <= 0:
        raise ValueError(
            "teleop.command.control_watchdog_timeout_s must be positive and finite"
        )
    if control_watchdog_timeout_s <= 1.0 / sample_rate_hz:
        raise ValueError(
            "teleop.command.control_watchdog_timeout_s must exceed one sample period"
        )
    if role_switch_settle_s < 0:
        raise ValueError("teleop.command.role_switch_settle_s must be non-negative")
    if role_switch_timeout_s <= 0:
        raise ValueError("teleop.command.role_switch_timeout_s must be positive")
    if reset_interpolation_rate_hz <= 0:
        raise ValueError("teleop.command.reset_interpolation_rate_hz must be positive")
    if reset_joint_speed_rad_s <= 0:
        raise ValueError("teleop.command.reset_joint_speed_rad_s must be positive")
    if reset_min_duration_s < 0:
        raise ValueError("teleop.command.reset_min_duration_s must be non-negative")
    if reset_max_step_rad <= 0:
        raise ValueError("teleop.command.reset_max_step_rad must be positive")
    return CommandConfig(
        sample_rate_hz=sample_rate_hz,
        control_watchdog_timeout_s=control_watchdog_timeout_s,
        idle_rate_hz=float(data.get("idle_rate_hz", 30.0)),
        input_ready_timeout_s=float(data.get("input_ready_timeout_s", 3.0)),
        reset_on_start=bool(data.get("reset_on_start", False)),
        reset_after_episode=bool(data.get("reset_after_episode", True)),
        reset_timeout_s=float(data.get("reset_timeout_s", 10.0)),
        reset_wait_s=float(data.get("reset_wait_s", 0.8)),
        reset_test_sample_time=int(data.get("reset_test_sample_time", 5)),
        reset_error_limit_rad=float(data.get("reset_error_limit_rad", 0.02)),
        joint_step_limit_rad=_optional_float(data.get("joint_step_limit_rad", 0.08)),
        control_mode=control_mode,
        bilateral_mit=_parse_bilateral_mit(data.get("bilateral_mit", {})),
        role_switch_settle_s=role_switch_settle_s,
        role_switch_timeout_s=role_switch_timeout_s,
        reset_interpolation_enabled=bool(data.get("reset_interpolation_enabled", True)),
        reset_interpolation_rate_hz=reset_interpolation_rate_hz,
        reset_joint_speed_rad_s=reset_joint_speed_rad_s,
        reset_min_duration_s=reset_min_duration_s,
        reset_max_step_rad=reset_max_step_rad,
    )


def _parse_bilateral_mit(data: dict[str, Any]) -> BilateralMitConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop.command.bilateral_mit must be a mapping")
    defaults = BilateralMitConfig()
    values = {
        name: _joint_vector(data.get(name, getattr(defaults, name)), f"bilateral_mit.{name}")
        for name in (
            "leader_kp",
            "leader_kd",
            "follower_kp",
            "follower_kd",
            "position_scale",
            "leader_gravity_scale",
            "follower_gravity_scale",
            "force_feedback_gain",
            "force_feedback_sign",
            "force_feedback_deadband_nm",
            "force_feedback_limit_nm",
            "leader_torque_limit_nm",
            "follower_torque_limit_nm",
            "feedback_torque_rate_limit_nm_s",
        )
    }
    for name in ("leader_kp", "follower_kp"):
        if any(value < 0.0 or value > 500.0 for value in values[name]):
            raise ValueError(f"teleop.command.bilateral_mit.{name} must be within [0, 500]")
    for name in ("leader_kd", "follower_kd"):
        if any(value < 0.0 or value > 5.0 for value in values[name]):
            raise ValueError(f"teleop.command.bilateral_mit.{name} must be within [0, 5]")
    for name in (
        "leader_gravity_scale",
        "follower_gravity_scale",
        "force_feedback_gain",
        "force_feedback_deadband_nm",
        "force_feedback_limit_nm",
        "leader_torque_limit_nm",
        "follower_torque_limit_nm",
        "feedback_torque_rate_limit_nm_s",
    ):
        if any(value < 0.0 for value in values[name]):
            raise ValueError(f"teleop.command.bilateral_mit.{name} must be non-negative")
    if any(abs(value) > 2.0 for value in values["position_scale"]):
        raise ValueError("teleop.command.bilateral_mit.position_scale magnitude must not exceed 2")
    if any(value not in {-1.0, 1.0} for value in values["force_feedback_sign"]):
        raise ValueError("teleop.command.bilateral_mit.force_feedback_sign values must be +/-1")
    nero_limits = (24.0, 24.0, 16.0, 16.0, 8.0, 8.0, 8.0)
    for name in ("leader_torque_limit_nm", "follower_torque_limit_nm"):
        if any(value > limit for value, limit in zip(values[name], nero_limits)):
            raise ValueError(f"teleop.command.bilateral_mit.{name} exceeds Nero MIT limits")
    raw_cutoff = data.get(
        "force_feedback_lowpass_hz", defaults.force_feedback_lowpass_hz
    )
    cutoff = None if raw_cutoff is None else float(raw_cutoff)
    ramp = float(data.get("force_feedback_ramp_s", defaults.force_feedback_ramp_s))
    margin = float(data.get("joint_limit_margin_rad", defaults.joint_limit_margin_rad))
    if cutoff is not None and (not isfinite(cutoff) or cutoff <= 0):
        raise ValueError(
            "teleop.command.bilateral_mit.force_feedback_lowpass_hz must be "
            "positive or null"
        )
    if not isfinite(ramp) or ramp < 0:
        raise ValueError("teleop.command.bilateral_mit.force_feedback_ramp_s must be non-negative")
    if not isfinite(margin) or margin < 0:
        raise ValueError("teleop.command.bilateral_mit.joint_limit_margin_rad must be non-negative")
    return BilateralMitConfig(
        **values,
        force_feedback_lowpass_hz=cutoff,
        force_feedback_ramp_s=ramp,
        joint_limit_margin_rad=margin,
    )


def _parse_output(data: dict[str, Any], base_dir: Path) -> OutputConfig:
    if not isinstance(data, dict):
        raise ValueError("output must be a mapping")
    directory = Path(data.get("directory", "runs/nero_master_slave")).expanduser()
    if not directory.is_absolute():
        directory = (base_dir / directory).resolve()
    discard_initial_s = float(data.get("discard_initial_s", 2.0))
    if not isfinite(discard_initial_s) or discard_initial_s < 0:
        raise ValueError("output.discard_initial_s must be non-negative and finite")
    return OutputConfig(
        directory=directory,
        prefix=str(data.get("prefix", "episode")),
        discard_initial_s=discard_initial_s,
    )


def _parse_dynamics_processing(data: dict[str, Any]) -> DynamicsProcessingConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("dynamics_processing must be a mapping")
    state_method = str(data.get("state_method", "finite_difference")).lower()
    if state_method not in {"finite_difference", "spline", "fourier"}:
        raise ValueError(
            "dynamics_processing.state_method must be finite_difference, spline, or fourier"
        )
    smoothing = float(data.get("spline_smoothing_rad2", 1.0e-5))
    fundamental_hz = float(data.get("fourier_fundamental_hz", 0.1))
    harmonics = int(data.get("fourier_harmonics", 8))
    torque_lowpass_hz = float(data.get("torque_lowpass_hz", 12.0))
    torque_median_window = int(data.get("torque_median_window", 3))
    min_samples = int(data.get("min_samples", 20))
    if not isfinite(smoothing) or smoothing < 0:
        raise ValueError("dynamics_processing.spline_smoothing_rad2 must be non-negative and finite")
    if not isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("dynamics_processing.fourier_fundamental_hz must be positive and finite")
    if harmonics < 1:
        raise ValueError("dynamics_processing.fourier_harmonics must be positive")
    if not isfinite(torque_lowpass_hz) or torque_lowpass_hz <= 0:
        raise ValueError("dynamics_processing.torque_lowpass_hz must be positive and finite")
    if torque_median_window < 1 or torque_median_window % 2 == 0:
        raise ValueError("dynamics_processing.torque_median_window must be a positive odd integer")
    minimum_samples = 3 if state_method == "finite_difference" else 4
    if min_samples < minimum_samples:
        raise ValueError(f"dynamics_processing.min_samples must be at least {minimum_samples}")
    return DynamicsProcessingConfig(
        enabled=bool(data.get("enabled", False)),
        state_method=state_method,
        spline_smoothing_rad2=smoothing,
        fourier_fundamental_hz=fundamental_hz,
        fourier_harmonics=harmonics,
        torque_lowpass_hz=torque_lowpass_hz,
        torque_median_window=torque_median_window,
        min_samples=min_samples,
    )


def _parse_camera(data: dict[str, Any]) -> CameraConfig:
    if not isinstance(data, dict):
        raise ValueError("Each camera entry must be a mapping")
    name = data.get("name")
    if not name:
        raise ValueError("Each enabled camera must define a name")
    known = {
        "name",
        "enabled",
        "backend",
        "device",
        "pixel_format",
        "buffer_size",
        "startup_timeout_s",
        "warmup_s",
        "frame_timeout_s",
        "visualize",
        "serial_number",
        "width",
        "height",
        "fps",
        "exposure",
        "exposure_dynamic_framerate",
        "depth",
        "crop",
        "output_size",
    }
    crop = tuple(data.get("crop", (0, None, 0, None)))
    output_size = data.get("output_size")
    device = data.get("device")
    if device is not None and not isinstance(device, (str, int)):
        raise ValueError("camera.device must be a device path or integer index")
    backend = str(data.get("backend", "orbbec_dabai"))
    normalized_backend = backend.lower().replace("-", "_")
    serial_number_value = data.get("serial_number")
    serial_number = (
        str(serial_number_value).strip() if serial_number_value is not None else None
    )
    if serial_number == "":
        raise ValueError("camera.serial_number must be non-empty")
    if normalized_backend in {"v4l2", "opencv_v4l2", "opencv"}:
        if device is None and serial_number is None:
            raise ValueError("V4L2 camera configuration must define device or serial_number")
        if device is not None and serial_number is not None:
            raise ValueError("V4L2 camera configuration must not define both device and serial_number")
    pixel_format = str(data.get("pixel_format", "MJPG")).upper()
    if len(pixel_format) != 4 or not pixel_format.isascii():
        raise ValueError("camera.pixel_format must be a four-character V4L2 code")
    width = int(data.get("width", 640))
    height = int(data.get("height", 480))
    fps = float(data.get("fps", 30.0))
    buffer_size = int(data.get("buffer_size", 1))
    startup_timeout_s = float(data.get("startup_timeout_s", 3.0))
    warmup_s = float(data.get("warmup_s", 0.0))
    frame_timeout_s = float(data.get("frame_timeout_s", 1.0))
    visualize = data.get("visualize", False)
    if not isinstance(visualize, bool):
        raise ValueError("camera.visualize must be boolean")
    exposure_dynamic_framerate = data.get("exposure_dynamic_framerate")
    if exposure_dynamic_framerate is not None and not isinstance(
        exposure_dynamic_framerate, bool
    ):
        raise ValueError("camera.exposure_dynamic_framerate must be boolean or null")
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("camera fps must be positive and finite")
    if buffer_size <= 0:
        raise ValueError("camera.buffer_size must be positive")
    if not isfinite(startup_timeout_s) or startup_timeout_s <= 0:
        raise ValueError("camera.startup_timeout_s must be positive and finite")
    if not isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("camera.warmup_s must be non-negative and finite")
    if not isfinite(frame_timeout_s) or frame_timeout_s <= 0:
        raise ValueError("camera.frame_timeout_s must be positive and finite")
    normalized_crop = _normalize_crop(crop)
    normalized_output_size = tuple(int(value) for value in output_size) if output_size else None
    if normalized_output_size is not None and (
        len(normalized_output_size) != 2 or any(value <= 0 for value in normalized_output_size)
    ):
        raise ValueError("camera.output_size must contain positive [width, height]")
    return CameraConfig(
        name=str(name),
        enabled=bool(data.get("enabled", True)),
        backend=backend,
        device=device,
        pixel_format=pixel_format,
        buffer_size=buffer_size,
        startup_timeout_s=startup_timeout_s,
        warmup_s=warmup_s,
        frame_timeout_s=frame_timeout_s,
        visualize=visualize,
        serial_number=serial_number,
        width=width,
        height=height,
        fps=fps,
        exposure=data.get("exposure"),
        exposure_dynamic_framerate=exposure_dynamic_framerate,
        depth=bool(data.get("depth", False)),
        crop=normalized_crop,
        output_size=normalized_output_size,
        extra={key: value for key, value in data.items() if key not in known},
    )


def _parse_gripper(data: dict[str, Any]) -> GripperConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("gripper must be a mapping")
    attach_to = str(data.get("attach_to", "follower"))
    if attach_to not in {"leader", "follower", "both"}:
        raise ValueError("gripper.attach_to must be one of: leader, follower, both")
    scale = float(data.get("scale", 1.0))
    offset_m = float(data.get("offset_m", 0.0))
    min_width_m = float(data.get("min_width_m", 0.0))
    max_width_m = float(data.get("max_width_m", 0.07))
    force_n = float(data.get("force_n", 1.0))
    command_rate_hz = float(data.get("command_rate_hz", 30.0))
    deadband_m = float(data.get("deadband_m", 0.0005))
    keepalive_s = float(data.get("keepalive_s", 0.5))
    numeric_values = (
        scale,
        offset_m,
        min_width_m,
        max_width_m,
        force_n,
        command_rate_hz,
        deadband_m,
        keepalive_s,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise ValueError("gripper numeric parameters must be finite")
    if scale == 0:
        raise ValueError("gripper.scale must be non-zero")
    if min_width_m < 0 or max_width_m <= min_width_m:
        raise ValueError("gripper width range must satisfy 0 <= min_width_m < max_width_m")
    if force_n < 0:
        raise ValueError("gripper.force_n must be non-negative")
    if command_rate_hz <= 0:
        raise ValueError("gripper.command_rate_hz must be positive")
    if deadband_m < 0:
        raise ValueError("gripper.deadband_m must be non-negative")
    if keepalive_s <= 0:
        raise ValueError("gripper.keepalive_s must be positive")
    return GripperConfig(
        enabled=bool(data.get("enabled", True)),
        effector=str(data.get("effector", "AGX_GRIPPER")),
        attach_to=attach_to,
        teleop_enabled=bool(data.get("teleop_enabled", False)),
        scale=scale,
        offset_m=offset_m,
        min_width_m=min_width_m,
        max_width_m=max_width_m,
        force_n=force_n,
        command_rate_hz=command_rate_hz,
        deadband_m=deadband_m,
        keepalive_s=keepalive_s,
    )


def _parse_realtime_plot(
    data: dict[str, Any],
) -> RealtimePlotConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("realtime_plot must be a mapping")
    window_s = float(data.get("window_s", 10.0))
    update_rate_hz = float(data.get("update_rate_hz", 20.0))
    if not isfinite(window_s) or window_s <= 0:
        raise ValueError("realtime_plot.window_s must be positive and finite")
    if not isfinite(update_rate_hz) or update_rate_hz <= 0:
        raise ValueError("realtime_plot.update_rate_hz must be positive and finite")
    unknown = set(data) - {"enabled", "window_s", "update_rate_hz"}
    if unknown:
        raise ValueError(
            f"realtime_plot contains removed options: {sorted(unknown)}; "
            "configure inverse_dynamics under tau_ext_inference"
        )
    return RealtimePlotConfig(
        enabled=bool(data.get("enabled", False)),
        window_s=window_s,
        update_rate_hz=update_rate_hz,
    )


def _parse_inverse_dynamics(
    data: dict[str, Any],
    config_dir: Path | None = None,
    prefix: str = "inverse_dynamics",
) -> InverseDynamicsConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{prefix} must be a mapping")
    allowed = {"urdf_path", "manifest_path", "delay_s", "locked_joint_names", "gravity_m_s2"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{prefix} contains unknown options: {sorted(unknown)}")
    base_dir = Path.cwd() if config_dir is None else Path(config_dir)
    urdf_path = Path(data.get("urdf_path", "../urdf/nero/nero_with_gripper.urdf")).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (base_dir / urdf_path).resolve()
    manifest_value = data.get("manifest_path")
    manifest_path = None
    if manifest_value is not None:
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (base_dir / manifest_path).resolve()
    delay_s = float(data.get("delay_s", 0.0))
    if not isfinite(delay_s) or delay_s < 0:
        raise ValueError(f"{prefix}.delay_s must be non-negative and finite")
    locked_joint_names = tuple(
        str(name)
        for name in data.get(
            "locked_joint_names", ("gripper", "gripper_joint1", "gripper_joint2")
        )
    )
    if not all(locked_joint_names):
        raise ValueError(f"{prefix}.locked_joint_names must contain valid names")
    gravity = tuple(float(value) for value in data.get("gravity_m_s2", (0.0, 0.0, -9.81)))
    if len(gravity) != 3 or not all(isfinite(value) for value in gravity):
        raise ValueError(f"{prefix}.gravity_m_s2 must contain three finite values")
    return InverseDynamicsConfig(
        urdf_path=urdf_path,
        manifest_path=manifest_path,
        delay_s=delay_s,
        locked_joint_names=locked_joint_names,
        gravity_m_s2=gravity,
    )


def _parse_tau_ext_inference(
    data: dict[str, Any],
    config_dir: Path | None = None,
) -> TauExtInferenceConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("tau_ext_inference must be a mapping")
    removed = {
        "observation_interval_s",
        "observation_stride_frames",
        "observation_sample_rate_hz",
    }.intersection(data)
    if removed:
        raise ValueError(
            "global model observation selection was removed; configure each "
            "checkpoint branch's observation_sample_rate_hz instead: "
            + ", ".join(sorted(removed))
        )
    unknown = set(data) - {
        "enabled",
        "feedback_source",
        "observation_gap_warning_s",
        "maximum_prediction_age_s",
        "tau_f",
        "tau_next",
        "source_butterworth_filter",
        "state_estimator",
        "tau_ext_filter",
        "inverse_dynamics",
    }
    if unknown:
        raise ValueError(
            f"tau_ext_inference contains unknown options: {sorted(unknown)}"
        )

    enabled = bool(data.get("enabled", False))
    feedback_source = str(data.get("feedback_source", "tau_free")).strip().lower()
    if feedback_source not in {"tau_f", "tau_free"}:
        raise ValueError(
            "tau_ext_inference.feedback_source must be tau_f or tau_free"
        )
    observation_gap_warning_s = float(data.get("observation_gap_warning_s", 0.06))
    maximum_prediction_age_s = float(data.get("maximum_prediction_age_s", 0.06))
    for name, value in (
        ("observation_gap_warning_s", observation_gap_warning_s),
        ("maximum_prediction_age_s", maximum_prediction_age_s),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"tau_ext_inference.{name} must be positive and finite")
    tau_f = _parse_sequence_checkpoint(
        data.get("tau_f", {}),
        "tau_ext_inference.tau_f",
        config_dir,
    )
    tau_next = _parse_sequence_checkpoint(
        data.get("tau_next", {}),
        "tau_ext_inference.tau_next",
        config_dir,
    )
    source_butterworth_filter = _parse_source_butterworth_filter(
        data.get("source_butterworth_filter", {})
    )
    state_estimator = _parse_causal_kalman(
        data.get("state_estimator", {}),
        "tau_ext_inference.state_estimator",
    )
    tau_ext_filter = _parse_tau_ext_filter(data.get("tau_ext_filter", {}))
    inverse_dynamics = _parse_inverse_dynamics(
        data.get("inverse_dynamics", {}),
        config_dir,
        "tau_ext_inference.inverse_dynamics",
    )
    return TauExtInferenceConfig(
        enabled=enabled,
        feedback_source=feedback_source,
        observation_gap_warning_s=observation_gap_warning_s,
        maximum_prediction_age_s=maximum_prediction_age_s,
        tau_f=tau_f,
        tau_next=tau_next,
        source_butterworth_filter=source_butterworth_filter,
        state_estimator=state_estimator,
        tau_ext_filter=tau_ext_filter,
        inverse_dynamics=inverse_dynamics,
    )


def _parse_source_butterworth_filter(
    data: dict[str, Any],
) -> SourceButterworthFilterConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            "tau_ext_inference.source_butterworth_filter must be a mapping"
        )
    allowed = {"enabled", "cutoff_hz", "order"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            "tau_ext_inference.source_butterworth_filter contains unknown "
            f"options: {sorted(unknown)}"
        )
    enabled = bool(data.get("enabled", False))
    cutoff_hz = float(data.get("cutoff_hz", 15.0))
    order = int(data.get("order", 2))
    if not isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError(
            "tau_ext_inference.source_butterworth_filter.cutoff_hz must be "
            "positive and finite"
        )
    if order != 2:
        raise ValueError(
            "tau_ext_inference.source_butterworth_filter.order must be 2"
        )
    return SourceButterworthFilterConfig(
        enabled=enabled,
        cutoff_hz=cutoff_hz,
        order=order,
    )


def _parse_tau_ext_filter(data: dict[str, Any]) -> TauExtFilterConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("tau_ext_inference.tau_ext_filter must be a mapping")
    allowed = {
        "enabled",
        "mode",
        "window",
        "cutoff_hz",
        "hampel_n_sigma",
        "order",
        "sample_rate_hz",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter contains unknown options: "
            f"{sorted(unknown)}"
        )
    enabled = bool(data.get("enabled", True))
    mode = str(data.get("mode", "hampel_butterworth")).strip().lower()
    if mode not in {"hampel_butterworth", "moving_average", "median"}:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.mode must be "
            "hampel_butterworth, moving_average, or median"
        )
    window = int(data.get("window", 5))
    if window < 1:
        raise ValueError("tau_ext_inference.tau_ext_filter.window must be positive")
    if mode in {"hampel_butterworth", "median"} and window % 2 == 0:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.window must be odd in "
            "hampel_butterworth or median mode"
        )
    cutoff_hz = float(data.get("cutoff_hz", 8.0))
    if not isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.cutoff_hz must be positive and finite"
        )
    hampel_n_sigma = float(data.get("hampel_n_sigma", 3.0))
    if not isfinite(hampel_n_sigma) or hampel_n_sigma <= 0.0:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.hampel_n_sigma must be positive "
            "and finite"
        )
    order = int(data.get("order", 4))
    if order < 1:
        raise ValueError("tau_ext_inference.tau_ext_filter.order must be positive")
    sample_rate_hz = float(data.get("sample_rate_hz", 100.0))
    if not isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.sample_rate_hz must be positive "
            "and finite"
        )
    if mode == "hampel_butterworth" and cutoff_hz >= 0.5 * sample_rate_hz:
        raise ValueError(
            "tau_ext_inference.tau_ext_filter.cutoff_hz must be below the "
            "Nyquist frequency"
        )
    return TauExtFilterConfig(
        enabled=enabled,
        mode=mode,
        window=window,
        cutoff_hz=cutoff_hz,
        hampel_n_sigma=hampel_n_sigma,
        order=order,
        sample_rate_hz=sample_rate_hz,
    )


def _parse_sequence_checkpoint(
    data: dict[str, Any],
    name: str,
    config_dir: Path | None,
) -> SequenceCheckpointConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    allowed = {
        "checkpoint_path",
        "device",
        "observation_sample_rate_hz",
        "horizon",
        "input_keys",
        "output_key",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown options: {sorted(unknown)}")

    checkpoint_path = None
    checkpoint_value = data.get("checkpoint_path")
    if checkpoint_value is not None:
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_absolute():
            base_dir = Path.cwd() if config_dir is None else Path(config_dir)
            config_relative = (base_dir / checkpoint_path).resolve()
            project_relative = (base_dir.parent / checkpoint_path).resolve()
            checkpoint_path = (
                project_relative
                if not config_relative.exists() and project_relative.exists()
                else config_relative
            )

    device = str(data.get("device", "cpu")).strip()
    if not device:
        raise ValueError(f"{name}.device must not be empty")
    sample_rate_value = data.get("observation_sample_rate_hz")
    if isinstance(sample_rate_value, bool):
        raise ValueError(
            f"{name}.observation_sample_rate_hz must be positive and finite"
        )
    observation_sample_rate_hz = (
        None if sample_rate_value is None else float(sample_rate_value)
    )
    if observation_sample_rate_hz is not None and (
        not isfinite(observation_sample_rate_hz)
        or observation_sample_rate_hz <= 0.0
    ):
        raise ValueError(
            f"{name}.observation_sample_rate_hz must be positive and finite"
        )
    horizon_value = data.get("horizon")
    horizon = None if horizon_value is None else int(horizon_value)
    if horizon is not None and horizon <= 0:
        raise ValueError(f"{name}.horizon must be positive when provided")

    input_value = data.get("input_keys")
    input_keys = None if input_value is None else tuple(str(key) for key in input_value)
    allowed_inputs = {"q", "dq", "ddq", "delta_q", "tau", "tau_id"}
    if input_keys is not None:
        if not input_keys or len(set(input_keys)) != len(input_keys):
            raise ValueError(f"{name}.input_keys must be non-empty and unique")
        unknown_inputs = sorted(set(input_keys) - allowed_inputs)
        if unknown_inputs:
            raise ValueError(
                f"{name}.input_keys contains unsupported values: {unknown_inputs}"
            )

    output_value = data.get("output_key")
    output_key = None if output_value is None else str(output_value).strip()
    if output_key == "":
        raise ValueError(f"{name}.output_key must not be empty")
    return SequenceCheckpointConfig(
        checkpoint_path=checkpoint_path,
        device=device,
        observation_sample_rate_hz=observation_sample_rate_hz,
        horizon=horizon,
        input_keys=input_keys,
        output_key=output_key,
    )


def _parse_causal_kalman(data: dict[str, Any], name: str) -> CausalKalmanConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    parameter_names = (
        "position_std",
        "velocity_std",
        "jerk_std",
        "initial_position_std",
        "initial_velocity_std",
        "initial_acceleration_std",
    )
    unknown = set(data) - {*parameter_names, "max_gap_s"}
    if unknown:
        raise ValueError(f"{name} contains unknown options: {sorted(unknown)}")
    defaults = CausalKalmanConfig()
    parameters = {
        key: _positive_joint_parameter(data.get(key, getattr(defaults, key)), f"{name}.{key}")
        for key in parameter_names
    }
    max_gap_s = float(data.get("max_gap_s", defaults.max_gap_s))
    if not isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError(f"{name}.max_gap_s must be positive and finite")
    return CausalKalmanConfig(**parameters, max_gap_s=max_gap_s)


def _positive_joint_parameter(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        result = (float(value),) * 7
    else:
        result = tuple(float(item) for item in value)
    if len(result) != 7 or any(not isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{name} must be a positive scalar or seven positive values")
    return result


def _parse_arm_pair(data: dict[str, Any]) -> ArmPairConfig:
    if not isinstance(data, dict):
        raise ValueError("Each arm pair must be a mapping")
    name = str(data.get("name", f"pair_{id(data):x}"))
    return ArmPairConfig(
        name=name,
        leader=_parse_arm_endpoint(data.get("leader", {}), f"{name}_leader"),
        follower=_parse_arm_endpoint(data.get("follower", {}), f"{name}_follower"),
    )


def _parse_arm_endpoint(data: dict[str, Any], default_name: str) -> ArmEndpointConfig:
    if not isinstance(data, dict):
        raise ValueError("leader/follower arm config must be a mapping")
    known = {
        "name",
        "can_id",
        "id",
        "channel",
        "usb_serial",
        "interface",
        "bitrate",
        "firmware",
        "rest_q",
        "config_kwargs",
    }
    config_kwargs = dict(data.get("config_kwargs", {}))
    for key, value in data.items():
        if key not in known:
            config_kwargs.setdefault(key, value)
    can_id = data.get("can_id", data.get("id"))
    return ArmEndpointConfig(
        name=str(data.get("name", default_name)),
        can_id=int(can_id) if can_id is not None else None,
        channel=str(data.get("channel", "can0")),
        usb_serial=(
            str(data["usb_serial"]).strip()
            if data.get("usb_serial") is not None
            else None
        ),
        interface=str(data.get("interface", "socketcan")),
        bitrate=int(data.get("bitrate", 1_000_000)),
        firmware=str(data.get("firmware", "V120")),
        rest_q=tuple(float(x) for x in data.get("rest_q", ())),
        config_kwargs=config_kwargs,
    )


def _parse_state_params(data: dict[str, Any]) -> dict[str, StateParamConfig]:
    if not isinstance(data, dict):
        raise ValueError("robot_states must be a mapping")
    params = dict(DEFAULT_STATE_PARAMS)
    for name, value in data.items():
        state_name = str(name)
        parsed = _parse_state_param(
            value,
            defaults=params.get(state_name, StateParamConfig()),
        )
        params[state_name] = parsed
    return params


def _parse_state_param(
    value: Any,
    *,
    defaults: StateParamConfig | None = None,
) -> StateParamConfig:
    defaults = defaults or StateParamConfig()
    if isinstance(value, bool):
        return StateParamConfig(
            enabled=value,
            lowpass=defaults.lowpass,
            lowpass_cutoff_hz=defaults.lowpass_cutoff_hz,
            mean_window=defaults.mean_window,
            median_window=defaults.median_window,
        )
    if value is None:
        return StateParamConfig(
            enabled=True,
            lowpass=defaults.lowpass,
            lowpass_cutoff_hz=defaults.lowpass_cutoff_hz,
            mean_window=defaults.mean_window,
            median_window=defaults.median_window,
        )
    if not isinstance(value, dict):
        raise ValueError("Each robot_states item must be bool or mapping")
    if "velocity_lowpass_cutoff_hz" in value:
        raise ValueError(
            "velocity_lowpass_cutoff_hz is no longer supported; q/dq/ddq filtering is "
            "defined by the per-joint-group derivative estimator"
        )
    lowpass_cutoff_hz = float(
        value.get("lowpass_cutoff_hz", defaults.lowpass_cutoff_hz)
    )
    if not isfinite(lowpass_cutoff_hz) or lowpass_cutoff_hz <= 0:
        raise ValueError("lowpass_cutoff_hz must be positive and finite")
    mean_window = int(value.get("mean_window", defaults.mean_window))
    if mean_window < 1:
        raise ValueError("mean_window must be a positive integer")
    median_window = int(value.get("median_window", defaults.median_window))
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    return StateParamConfig(
        enabled=bool(value.get("enabled", defaults.enabled)),
        lowpass=bool(value.get("lowpass", defaults.lowpass)),
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        mean_window=mean_window,
        median_window=median_window,
    )


def _normalize_crop(crop: tuple[Any, ...]) -> tuple[int | None, int | None, int | None, int | None]:
    if len(crop) != 4:
        raise ValueError("camera.crop must contain four values: [y0, y1, x0, x1]")
    normalized = tuple(None if value is None else int(value) for value in crop)
    if any(value is not None and value < 0 for value in normalized):
        raise ValueError("camera.crop values must be non-negative or null")
    y0, y1, x0, x1 = normalized
    if y0 is not None and y1 is not None and y1 <= y0:
        raise ValueError("camera.crop y1 must be greater than y0")
    if x0 is not None and x1 is not None and x1 <= x0:
        raise ValueError("camera.crop x1 must be greater than x0")
    return normalized  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _joint_vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(f"teleop.command.{name} must contain exactly 7 values")
    vector = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in vector):
        raise ValueError(f"teleop.command.{name} values must be finite")
    return vector
