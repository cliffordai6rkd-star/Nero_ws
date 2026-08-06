from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DOF = 7


@dataclass(frozen=True)
class DynamicsModelConfig:
    urdf_path: Path
    locked_joint_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    gravity_m_s2: np.ndarray


@dataclass(frozen=True)
class ExcitationProfile:
    name: str
    role: str
    seed: int
    center_rad: np.ndarray
    repetitions: int
    trajectory_path: Path
    dataset_path: Path


@dataclass(frozen=True)
class ExcitationConfig:
    sample_rate_hz: float
    duration_s: float
    fundamental_hz: float
    harmonics: int
    optimization_trials: int
    amplitude_rad: np.ndarray
    max_velocity_rad_s: np.ndarray
    max_acceleration_rad_s2: np.ndarray
    joint_limit_margin_rad: float
    max_tracking_error_rad: np.ndarray
    start_move_speed_rad_s: float
    profiles: tuple[ExcitationProfile, ...]


@dataclass(frozen=True)
class CollectionSafetyConfig:
    approved: bool
    max_abs_torque_nm: np.ndarray
    max_timestamp_gap_s: float


@dataclass(frozen=True)
class PreprocessConfig:
    state_method: str
    spline_smoothing_rad2: float
    fourier_harmonics: int
    torque_lowpass_hz: float
    torque_median_window: int
    outlier_z: float
    endpoint_trim_s: float
    validation_fraction: float
    split_seed: int
    min_samples: int
    coulomb_velocity_scale_rad_s: float


@dataclass(frozen=True)
class IdentificationConfig:
    svd_relative_tolerance: float
    huber_delta: float
    max_irls_iterations: int
    ridge: float
    physical_prior_weight: float
    mass_bounds_kg: tuple[float, float]
    max_abs_com_m: float
    max_coulomb_nm: np.ndarray
    max_viscous_nm_per_rad_s: np.ndarray
    max_abs_bias_nm: np.ndarray
    max_physical_evaluations: int
    physical_tolerance: float
    physical_optimizer_backend: str


@dataclass(frozen=True)
class MujocoSimulationConfig:
    scene_template_path: Path
    end_effector_body: str
    floor_z_m: float
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    display_rate_hz: float
    playback_speed: float
    collision_sample_stride: int
    ignored_contact_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DynamicsPlan:
    source_path: Path
    collection_config_path: Path
    pair_name: str
    model: DynamicsModelConfig
    excitation: ExcitationConfig
    safety: CollectionSafetyConfig
    preprocess: PreprocessConfig
    identification: IdentificationConfig
    simulation: MujocoSimulationConfig


@dataclass(frozen=True)
class DynamicsDataset:
    timestamp_us: np.ndarray
    q: np.ndarray
    q_cmd: np.ndarray
    tau: np.ndarray
    current: np.ndarray
    trajectory_id: np.ndarray
    metadata: dict[str, Any]
    motor_timestamp_us: np.ndarray | None = None
    motor_acquired_timestamp_us: np.ndarray | None = None
    q_can_timestamp_us: np.ndarray | None = None
    q_acquired_timestamp_us: np.ndarray | None = None


@dataclass(frozen=True)
class ProcessedDynamicsDataset:
    timestamp_us: np.ndarray
    time_s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    current: np.ndarray
    q_cmd: np.ndarray
    trajectory_id: np.ndarray
    source_indices: np.ndarray


def load_dynamics_plan(path: str | Path) -> DynamicsPlan:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("dynamics calibration config must be a YAML mapping")
    base = source.parent

    model_raw = _mapping(raw.get("model"), "model")
    excitation_raw = _mapping(raw.get("excitation"), "excitation")
    safety_raw = _mapping(raw.get("safety"), "safety")
    preprocess_raw = _mapping(raw.get("preprocess"), "preprocess")
    identification_raw = _mapping(raw.get("identification"), "identification")
    simulation_raw = _mapping(raw.get("simulation"), "simulation")

    urdf_path = _path(model_raw.get("urdf_path", "../urdf/nero/nero_with_gripper.urdf"), base)
    if not urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {urdf_path}")
    joint_names = tuple(str(v) for v in model_raw.get("joint_names", [f"joint{i}" for i in range(1, 8)]))
    if len(joint_names) != DOF or len(set(joint_names)) != DOF:
        raise ValueError("model.joint_names must contain seven unique names in hardware order")
    model = DynamicsModelConfig(
        urdf_path=urdf_path,
        locked_joint_names=tuple(str(v) for v in model_raw.get("locked_joint_names", [])),
        joint_names=joint_names,
        gravity_m_s2=_vector(model_raw.get("gravity_m_s2", [0, 0, -9.81]), 3, "model.gravity_m_s2"),
    )

    profile_items = excitation_raw.get("profiles")
    if not isinstance(profile_items, list) or len(profile_items) < 2:
        raise ValueError("excitation.profiles must contain at least two trajectory profiles")
    profiles: list[ExcitationProfile] = []
    profile_names: set[str] = set()
    for index, item in enumerate(profile_items):
        profile_raw = _mapping(item, f"excitation.profiles[{index}]")
        name = str(profile_raw.get("name", "")).strip()
        if not name or name in profile_names:
            raise ValueError(f"excitation profile name must be non-empty and unique: {name!r}")
        profile_names.add(name)
        role = str(profile_raw.get("role", "train")).lower()
        if role not in {"train", "validation"}:
            raise ValueError(f"excitation profile {name!r} role must be train or validation")
        repetitions = int(profile_raw.get("repetitions", 2))
        if repetitions < 1:
            raise ValueError(f"excitation profile {name!r} repetitions must be positive")
        profiles.append(
            ExcitationProfile(
                name=name,
                role=role,
                seed=int(profile_raw.get("seed")),
                center_rad=_vector(
                    profile_raw.get("center_rad"),
                    DOF,
                    f"excitation.profiles[{index}].center_rad",
                ),
                repetitions=repetitions,
                trajectory_path=_path(
                    profile_raw.get("trajectory_path", f"data/excitation_{name}.npz"),
                    base,
                ),
                dataset_path=_path(
                    profile_raw.get("dataset_path", f"data/dynamics_{name}.npz"),
                    base,
                ),
            )
        )
    if not any(profile.role == "train" for profile in profiles):
        raise ValueError("excitation.profiles must include at least one training profile")
    if not any(profile.role == "validation" for profile in profiles):
        raise ValueError("excitation.profiles must include at least one validation profile")

    excitation = ExcitationConfig(
        sample_rate_hz=_positive(excitation_raw.get("sample_rate_hz", 100.0), "excitation.sample_rate_hz"),
        duration_s=_positive(excitation_raw.get("duration_s", 30.0), "excitation.duration_s"),
        fundamental_hz=_positive(excitation_raw.get("fundamental_hz", 0.1), "excitation.fundamental_hz"),
        harmonics=int(excitation_raw.get("harmonics", 5)),
        optimization_trials=int(excitation_raw.get("optimization_trials", 250)),
        amplitude_rad=_positive_vector(excitation_raw.get("amplitude_rad"), DOF, "excitation.amplitude_rad"),
        max_velocity_rad_s=_positive_vector(excitation_raw.get("max_velocity_rad_s"), DOF, "excitation.max_velocity_rad_s"),
        max_acceleration_rad_s2=_positive_vector(excitation_raw.get("max_acceleration_rad_s2"), DOF, "excitation.max_acceleration_rad_s2"),
        joint_limit_margin_rad=_nonnegative(excitation_raw.get("joint_limit_margin_rad", 0.08), "excitation.joint_limit_margin_rad"),
        max_tracking_error_rad=_positive_vector(excitation_raw.get("max_tracking_error_rad", [0.15] * DOF), DOF, "excitation.max_tracking_error_rad"),
        start_move_speed_rad_s=_positive(excitation_raw.get("start_move_speed_rad_s", 0.15), "excitation.start_move_speed_rad_s"),
        profiles=tuple(profiles),
    )
    if excitation.harmonics < 1 or excitation.optimization_trials < 1:
        raise ValueError("excitation.harmonics and optimization_trials must be positive")
    if excitation.duration_s * excitation.sample_rate_hz < 100:
        raise ValueError("excitation trajectory must contain at least 100 samples")

    safety = CollectionSafetyConfig(
        approved=bool(safety_raw.get("approved", False)),
        max_abs_torque_nm=_positive_vector(safety_raw.get("max_abs_torque_nm"), DOF, "safety.max_abs_torque_nm"),
        max_timestamp_gap_s=_positive(safety_raw.get("max_timestamp_gap_s", 0.1), "safety.max_timestamp_gap_s"),
    )

    state_method = str(preprocess_raw.get("state_method", "spline")).lower()
    if state_method not in {"spline", "fourier"}:
        raise ValueError("preprocess.state_method must be spline or fourier")
    median_window = int(preprocess_raw.get("torque_median_window", 3))
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("preprocess.torque_median_window must be a positive odd integer")
    preprocess = PreprocessConfig(
        state_method=state_method,
        spline_smoothing_rad2=_nonnegative(preprocess_raw.get("spline_smoothing_rad2", 1e-5), "preprocess.spline_smoothing_rad2"),
        fourier_harmonics=int(preprocess_raw.get("fourier_harmonics", excitation.harmonics)),
        torque_lowpass_hz=_positive(preprocess_raw.get("torque_lowpass_hz", 12.0), "preprocess.torque_lowpass_hz"),
        torque_median_window=median_window,
        outlier_z=_positive(preprocess_raw.get("outlier_z", 6.0), "preprocess.outlier_z"),
        endpoint_trim_s=_nonnegative(preprocess_raw.get("endpoint_trim_s", 0.25), "preprocess.endpoint_trim_s"),
        validation_fraction=float(preprocess_raw.get("validation_fraction", 0.25)),
        split_seed=int(preprocess_raw.get("split_seed", 20260719)),
        min_samples=int(preprocess_raw.get("min_samples", 500)),
        coulomb_velocity_scale_rad_s=_positive(preprocess_raw.get("coulomb_velocity_scale_rad_s", 0.02), "preprocess.coulomb_velocity_scale_rad_s"),
    )
    if preprocess.fourier_harmonics < 1 or preprocess.min_samples < 50:
        raise ValueError("preprocess Fourier harmonics and min_samples are too small")
    if not 0.1 <= preprocess.validation_fraction <= 0.5:
        raise ValueError("preprocess.validation_fraction must be in [0.1, 0.5]")

    mass_bounds = _vector(identification_raw.get("mass_bounds_kg", [1e-4, 10.0]), 2, "identification.mass_bounds_kg")
    if mass_bounds[0] <= 0 or mass_bounds[1] <= mass_bounds[0]:
        raise ValueError("identification.mass_bounds_kg must be positive increasing bounds")
    identification = IdentificationConfig(
        svd_relative_tolerance=_positive(identification_raw.get("svd_relative_tolerance", 1e-8), "identification.svd_relative_tolerance"),
        huber_delta=_positive(identification_raw.get("huber_delta", 1.5), "identification.huber_delta"),
        max_irls_iterations=int(identification_raw.get("max_irls_iterations", 30)),
        ridge=_nonnegative(identification_raw.get("ridge", 1e-8), "identification.ridge"),
        physical_prior_weight=_positive(identification_raw.get("physical_prior_weight", 1e-3), "identification.physical_prior_weight"),
        mass_bounds_kg=(float(mass_bounds[0]), float(mass_bounds[1])),
        max_abs_com_m=_positive(identification_raw.get("max_abs_com_m", 0.5), "identification.max_abs_com_m"),
        max_coulomb_nm=_positive_vector(identification_raw.get("max_coulomb_nm"), DOF, "identification.max_coulomb_nm"),
        max_viscous_nm_per_rad_s=_positive_vector(identification_raw.get("max_viscous_nm_per_rad_s"), DOF, "identification.max_viscous_nm_per_rad_s"),
        max_abs_bias_nm=_positive_vector(identification_raw.get("max_abs_bias_nm"), DOF, "identification.max_abs_bias_nm"),
        max_physical_evaluations=int(
            identification_raw.get("max_physical_evaluations", 10000)
        ),
        physical_tolerance=_positive(
            identification_raw.get("physical_tolerance", 1e-7),
            "identification.physical_tolerance",
        ),
        physical_optimizer_backend=str(
            identification_raw.get("physical_optimizer_backend", "scipy")
        ).lower(),
    )
    if identification.max_irls_iterations < 1 or identification.max_physical_evaluations < 1:
        raise ValueError("identification iteration limits must be positive")
    if identification.physical_optimizer_backend not in {"scipy", "jax"}:
        raise ValueError(
            "identification.physical_optimizer_backend must be scipy or jax"
        )

    workspace_min = _vector(
        simulation_raw.get("workspace_min_m", [-0.8, -0.8, 0.0]),
        3,
        "simulation.workspace_min_m",
    )
    workspace_max = _vector(
        simulation_raw.get("workspace_max_m", [0.8, 0.8, 1.2]),
        3,
        "simulation.workspace_max_m",
    )
    if np.any(workspace_min >= workspace_max):
        raise ValueError("simulation workspace min must be below max on every axis")
    simulation = MujocoSimulationConfig(
        scene_template_path=_path(
            simulation_raw.get("scene_template_path", "mujoco/scene_template.xml"),
            base,
        ),
        end_effector_body=str(
            simulation_raw.get("end_effector_body", "gripper_base")
        ),
        floor_z_m=_finite(
            simulation_raw.get("floor_z_m", -0.02),
            "simulation.floor_z_m",
        ),
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        display_rate_hz=_positive(
            simulation_raw.get("display_rate_hz", 30.0),
            "simulation.display_rate_hz",
        ),
        playback_speed=_positive(
            simulation_raw.get("playback_speed", 1.0),
            "simulation.playback_speed",
        ),
        collision_sample_stride=int(
            simulation_raw.get("collision_sample_stride", 10)
        ),
        ignored_contact_pairs=_contact_pairs(
            simulation_raw.get("ignored_contact_pairs", []),
            "simulation.ignored_contact_pairs",
        ),
    )
    if not simulation.scene_template_path.is_file():
        raise ValueError(f"MuJoCo scene template does not exist: {simulation.scene_template_path}")
    if not simulation.end_effector_body:
        raise ValueError("simulation.end_effector_body must not be empty")
    if simulation.collision_sample_stride < 1:
        raise ValueError("simulation.collision_sample_stride must be positive")

    return DynamicsPlan(
        source_path=source,
        collection_config_path=_path(raw.get("collection_config", "../configs/master_slave_can.yaml"), base),
        pair_name=str(raw.get("pair", "main")),
        model=model,
        excitation=excitation,
        safety=safety,
        preprocess=preprocess,
        identification=identification,
        simulation=simulation,
    )


def save_dynamics_dataset(path: str | Path, dataset: DynamicsDataset) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamp_us=np.asarray(dataset.timestamp_us, dtype=np.int64),
        q=np.asarray(dataset.q, dtype=np.float64),
        q_cmd=np.asarray(dataset.q_cmd, dtype=np.float64),
        tau=np.asarray(dataset.tau, dtype=np.float64),
        current=np.asarray(dataset.current, dtype=np.float64),
        motor_timestamp_us=(
            np.asarray(dataset.motor_timestamp_us, dtype=np.int64)
            if dataset.motor_timestamp_us is not None
            else np.repeat(np.asarray(dataset.timestamp_us, dtype=np.int64)[:, None], DOF, axis=1)
        ),
        motor_acquired_timestamp_us=(
            np.asarray(dataset.motor_acquired_timestamp_us, dtype=np.int64)
            if dataset.motor_acquired_timestamp_us is not None
            else np.repeat(np.asarray(dataset.timestamp_us, dtype=np.int64)[:, None], DOF, axis=1)
        ),
        q_can_timestamp_us=(
            np.asarray(dataset.q_can_timestamp_us, dtype=np.int64)
            if dataset.q_can_timestamp_us is not None
            else np.asarray(dataset.timestamp_us, dtype=np.int64)
        ),
        q_acquired_timestamp_us=(
            np.asarray(dataset.q_acquired_timestamp_us, dtype=np.int64)
            if dataset.q_acquired_timestamp_us is not None
            else np.asarray(dataset.timestamp_us, dtype=np.int64)
        ),
        trajectory_id=np.asarray(dataset.trajectory_id, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True)),
    )
    return output


def load_dynamics_dataset(path: str | Path) -> DynamicsDataset:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as values:
        required = {"timestamp_us", "q", "q_cmd", "tau", "current", "trajectory_id", "metadata_json"}
        missing = sorted(required.difference(values.files))
        if missing:
            raise ValueError(f"dynamics dataset is missing arrays: {missing}")
        dataset = DynamicsDataset(
            timestamp_us=np.asarray(values["timestamp_us"], dtype=np.int64),
            q=np.asarray(values["q"], dtype=np.float64),
            q_cmd=np.asarray(values["q_cmd"], dtype=np.float64),
            tau=np.asarray(values["tau"], dtype=np.float64),
            current=np.asarray(values["current"], dtype=np.float64),
            motor_timestamp_us=(
                np.asarray(values["motor_timestamp_us"], dtype=np.int64)
                if "motor_timestamp_us" in values.files
                else np.repeat(
                    np.asarray(values["timestamp_us"], dtype=np.int64)[:, None],
                    DOF,
                    axis=1,
                )
            ),
            motor_acquired_timestamp_us=(
                np.asarray(values["motor_acquired_timestamp_us"], dtype=np.int64)
                if "motor_acquired_timestamp_us" in values.files
                else np.repeat(
                    np.asarray(values["timestamp_us"], dtype=np.int64)[:, None],
                    DOF,
                    axis=1,
                )
            ),
            q_can_timestamp_us=(
                np.asarray(values["q_can_timestamp_us"], dtype=np.int64)
                if "q_can_timestamp_us" in values.files
                else np.asarray(values["timestamp_us"], dtype=np.int64)
            ),
            q_acquired_timestamp_us=(
                np.asarray(values["q_acquired_timestamp_us"], dtype=np.int64)
                if "q_acquired_timestamp_us" in values.files
                else np.asarray(values["timestamp_us"], dtype=np.int64)
            ),
            trajectory_id=np.asarray(values["trajectory_id"], dtype=np.int32),
            metadata=json.loads(str(np.asarray(values["metadata_json"]).item())),
        )
    validate_dynamics_dataset(dataset)
    return dataset


def validate_dynamics_dataset(dataset: DynamicsDataset) -> None:
    n = dataset.timestamp_us.size
    if n < 2:
        raise ValueError("dynamics dataset must contain at least two samples")
    for name in ("q", "q_cmd", "tau", "current"):
        value = np.asarray(getattr(dataset, name))
        if value.shape != (n, DOF):
            raise ValueError(f"{name} must have shape ({n}, {DOF}); got {value.shape}")
    if dataset.trajectory_id.shape != (n,):
        raise ValueError("trajectory_id must have shape (N,)")
    if dataset.motor_timestamp_us is None or np.asarray(dataset.motor_timestamp_us).shape != (n, DOF):
        raise ValueError(f"motor_timestamp_us must have shape ({n}, {DOF})")
    if (
        dataset.motor_acquired_timestamp_us is None
        or np.asarray(dataset.motor_acquired_timestamp_us).shape != (n, DOF)
    ):
        raise ValueError(f"motor_acquired_timestamp_us must have shape ({n}, {DOF})")
    if np.any(np.asarray(dataset.motor_acquired_timestamp_us) <= 0):
        raise ValueError("motor_acquired_timestamp_us must contain positive timestamps")
    if dataset.q_can_timestamp_us is None or np.asarray(dataset.q_can_timestamp_us).shape != (n,):
        raise ValueError("q_can_timestamp_us must have shape (N,)")
    if (
        dataset.q_acquired_timestamp_us is None
        or np.asarray(dataset.q_acquired_timestamp_us).shape != (n,)
    ):
        raise ValueError("q_acquired_timestamp_us must have shape (N,)")
    if np.any(np.asarray(dataset.q_acquired_timestamp_us) <= 0):
        raise ValueError("q_acquired_timestamp_us must contain positive timestamps")
    if np.any(np.diff(dataset.timestamp_us) <= 0):
        raise ValueError("dataset timestamps must be strictly increasing")
    for name in ("q", "q_cmd", "tau"):
        if not np.isfinite(getattr(dataset, name)).all():
            raise ValueError(f"dataset {name} contains non-finite values")
    if np.isinf(dataset.current).any():
        raise ValueError("dataset current may contain NaN for unavailable feedback, but not infinity")


def build_reduced_model(settings: DynamicsModelConfig):
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("Dynamics identification requires Pinocchio 3.x (pip package pin)") from exc
    full = pin.buildModelFromUrdf(str(settings.urdf_path))
    missing = [name for name in settings.locked_joint_names if not full.existJointName(name)]
    if missing:
        raise ValueError(f"locked joints not found in URDF: {missing}")
    locked_ids = [full.getJointId(name) for name in settings.locked_joint_names]
    model = pin.buildReducedModel(full, locked_ids, pin.neutral(full)) if locked_ids else full
    actual = tuple(str(name) for name in model.names[1:])
    if actual != settings.joint_names:
        raise ValueError(
            "Pinocchio active joint order does not match configured hardware order: "
            f"pinocchio={actual}, hardware={settings.joint_names}"
        )
    if model.nq != DOF or model.nv != DOF:
        raise ValueError(f"reduced model must have {DOF} DoF; got nq={model.nq}, nv={model.nv}")
    model.gravity.linear[:] = settings.gravity_m_s2
    return pin, model


def setup_socketcan(channel: str, bitrate: int) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "setup_can.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--bitrate",
            str(int(bitrate)),
            str(channel),
        ],
        cwd=repository_root,
        check=True,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return array


def _positive_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = _vector(value, size, name)
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain positive values")
    return array


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _contact_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of two-name pairs")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two geom names")
        first, second = (str(part).strip() for part in item)
        if not first or not second or first == second:
            raise ValueError(f"{name}[{index}] contains invalid geom names")
        pairs.append(tuple(sorted((first, second))))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{name} contains duplicate pairs")
    return tuple(pairs)
