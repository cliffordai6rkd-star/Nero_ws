from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration.dynamics_common import (
    DOF,
    DynamicsModelConfig,
    MujocoSimulationConfig,
    build_reduced_model,
)


log = logging.getLogger(__name__)


SEGMENT_NAMES = (
    "empirical_workspace_low_speed",
    "empirical_workspace_static",
)
TAU_REFINEMENT_SEGMENT_NAMES = (
    "empirical_workspace_low_speed",
    "empirical_workspace_static",
    "bounded_target_switches",
)


@dataclass(frozen=True)
class CoverageRun:
    name: str
    seed: int
    duration_s: float
    trajectory_path: Path


@dataclass(frozen=True)
class CoverageExcitationConfig:
    planner: str
    sample_rate_hz: float
    joint_limit_margin_rad: float
    max_velocity_rad_s: np.ndarray
    max_acceleration_rad_s2: np.ndarray
    section_fractions: np.ndarray
    reference_h5_path: Path
    reference_dataset: str
    waypoint_min_delta_rad: float
    workspace_voxel_counts: np.ndarray
    replay_speed_scale: float
    static_transition_speed_scale: float
    static_hold_s: float
    static_pose_count: int
    replay_pose_count: int
    jump_pose_count: int
    jump_min_delta_rad: float
    jump_max_delta_rad: float
    jump_transition_speed_scale: float
    jump_hold_s: float
    run: CoverageRun


@dataclass(frozen=True)
class CoverageHardwareConfig:
    approved: bool
    preflight_report_path: Path
    max_tracking_error_rad: np.ndarray
    max_abs_torque_nm: np.ndarray
    max_timestamp_gap_s: float
    start_move_speed_rad_s: float
    record_cameras: bool


@dataclass(frozen=True)
class CoveragePlan:
    source_path: Path
    collection_config_path: Path
    pair_name: str
    model: DynamicsModelConfig
    excitation: CoverageExcitationConfig
    simulation: MujocoSimulationConfig
    hardware: CoverageHardwareConfig


@dataclass(frozen=True)
class CoverageTrajectory:
    name: str
    time_s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    segment_id: np.ndarray
    segment_names: tuple[str, ...] = SEGMENT_NAMES
    reference_h5_path: str = ""
    reference_h5_sha256: str = ""
    workspace_min_m: np.ndarray | None = None
    workspace_max_m: np.ndarray | None = None
    workspace_convex_hull_volume_m3: float = 0.0

    @property
    def duration_s(self) -> float:
        if self.time_s.size < 2:
            return 0.0
        return float(self.time_s[-1] + self.time_s[1] - self.time_s[0])


def load_coverage_plan(path: str | Path) -> CoveragePlan:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("free-space coverage config must be a YAML mapping")
    base = source.parent
    model_raw = _mapping(raw.get("model"), "model")
    trajectory_raw = _mapping(raw.get("trajectory"), "trajectory")
    simulation_raw = _mapping(raw.get("simulation"), "simulation")
    hardware_raw = _mapping(raw.get("hardware"), "hardware")

    urdf_path = _path(model_raw.get("urdf_path"), base)
    if not urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {urdf_path}")
    joint_names = tuple(
        str(value)
        for value in model_raw.get("joint_names", [f"joint{i}" for i in range(1, 8)])
    )
    if len(joint_names) != DOF or len(set(joint_names)) != DOF:
        raise ValueError("model.joint_names must contain seven unique names")
    model = DynamicsModelConfig(
        urdf_path=urdf_path,
        locked_joint_names=tuple(str(value) for value in model_raw.get("locked_joint_names", ())),
        joint_names=joint_names,
        gravity_m_s2=_vector(model_raw.get("gravity_m_s2", [0, 0, -9.81]), 3, "model.gravity_m_s2"),
    )

    run_raw = _mapping(trajectory_raw.get("run"), "trajectory.run")
    run_name = str(run_raw.get("name", "coverage")).strip()
    if not run_name:
        raise ValueError("trajectory.run.name must not be empty")
    run = CoverageRun(
        name=run_name,
        seed=int(run_raw.get("seed", 20260724)),
        duration_s=_positive(run_raw.get("duration_s", 900.0), "trajectory.run.duration_s"),
        trajectory_path=_path(
            run_raw.get("trajectory_path", "../calibration/data/free_space_coverage.npz"),
            base,
        ),
    )

    planner = str(trajectory_raw.get("planner", "empirical")).strip().lower()
    expected_sections = {
        "empirical": len(SEGMENT_NAMES),
        "tau_refinement": len(TAU_REFINEMENT_SEGMENT_NAMES),
    }
    if planner not in expected_sections:
        raise ValueError("trajectory.planner must be empirical or tau_refinement")
    section_fractions = _vector(
        trajectory_raw.get(
            "section_fractions",
            [0.82, 0.18] if planner == "empirical" else [0.45, 0.35, 0.20],
        ),
        expected_sections[planner],
        "trajectory.section_fractions",
    )
    if np.any(section_fractions <= 0) or not np.isclose(np.sum(section_fractions), 1.0):
        raise ValueError("trajectory.section_fractions must be positive and sum to one")
    reference_h5_path = _path(
        trajectory_raw.get("reference_h5_path"),
        base,
    )
    if not reference_h5_path.is_file():
        raise ValueError(f"trajectory reference H5 does not exist: {reference_h5_path}")
    workspace_voxel_counts = np.asarray(
        trajectory_raw.get("workspace_voxel_counts", [6, 8, 5]), dtype=np.int64
    ).reshape(-1)
    if workspace_voxel_counts.shape != (3,) or np.any(workspace_voxel_counts < 2):
        raise ValueError("trajectory.workspace_voxel_counts must contain three integers >= 2")

    excitation = CoverageExcitationConfig(
        planner=planner,
        sample_rate_hz=_positive(trajectory_raw.get("sample_rate_hz", 100.0), "trajectory.sample_rate_hz"),
        joint_limit_margin_rad=_nonnegative(trajectory_raw.get("joint_limit_margin_rad", 0.10), "trajectory.joint_limit_margin_rad"),
        max_velocity_rad_s=_positive_vector(trajectory_raw.get("max_velocity_rad_s"), DOF, "trajectory.max_velocity_rad_s"),
        max_acceleration_rad_s2=_positive_vector(trajectory_raw.get("max_acceleration_rad_s2"), DOF, "trajectory.max_acceleration_rad_s2"),
        section_fractions=section_fractions,
        reference_h5_path=reference_h5_path,
        reference_dataset=str(
            trajectory_raw.get("reference_dataset", "teleop/q_follower")
        ),
        waypoint_min_delta_rad=_positive(
            trajectory_raw.get("waypoint_min_delta_rad", 0.05),
            "trajectory.waypoint_min_delta_rad",
        ),
        workspace_voxel_counts=workspace_voxel_counts,
        replay_speed_scale=_fraction(
            trajectory_raw.get("replay_speed_scale", 0.10),
            "trajectory.replay_speed_scale",
            include_one=True,
        ),
        static_transition_speed_scale=_fraction(
            trajectory_raw.get("static_transition_speed_scale", 0.10),
            "trajectory.static_transition_speed_scale",
            include_one=True,
        ),
        static_hold_s=_nonnegative(
            trajectory_raw.get("static_hold_s", 0.50),
            "trajectory.static_hold_s",
        ),
        static_pose_count=int(trajectory_raw.get("static_pose_count", 48)),
        replay_pose_count=int(trajectory_raw.get("replay_pose_count", 128)),
        jump_pose_count=int(trajectory_raw.get("jump_pose_count", 64)),
        jump_min_delta_rad=_positive(
            trajectory_raw.get("jump_min_delta_rad", 0.08),
            "trajectory.jump_min_delta_rad",
        ),
        jump_max_delta_rad=_positive(
            trajectory_raw.get("jump_max_delta_rad", 0.35),
            "trajectory.jump_max_delta_rad",
        ),
        jump_transition_speed_scale=_fraction(
            trajectory_raw.get("jump_transition_speed_scale", 0.60),
            "trajectory.jump_transition_speed_scale",
            include_one=True,
        ),
        jump_hold_s=_nonnegative(
            trajectory_raw.get("jump_hold_s", 0.25),
            "trajectory.jump_hold_s",
        ),
        run=run,
    )
    if excitation.static_pose_count < 4:
        raise ValueError("trajectory.static_pose_count must be at least four")
    if excitation.replay_pose_count < 4:
        raise ValueError("trajectory.replay_pose_count must be at least four")
    if excitation.jump_pose_count < 2:
        raise ValueError("trajectory.jump_pose_count must be at least two")
    if excitation.jump_max_delta_rad <= excitation.jump_min_delta_rad:
        raise ValueError("trajectory.jump_max_delta_rad must exceed jump_min_delta_rad")

    workspace_min = _vector(simulation_raw.get("workspace_min_m"), 3, "simulation.workspace_min_m")
    workspace_max = _vector(simulation_raw.get("workspace_max_m"), 3, "simulation.workspace_max_m")
    if np.any(workspace_min >= workspace_max):
        raise ValueError("simulation workspace minimum must be below maximum")
    simulation = MujocoSimulationConfig(
        scene_template_path=_path(simulation_raw.get("scene_template_path"), base),
        end_effector_body=str(simulation_raw.get("end_effector_body", "gripper_base")),
        floor_z_m=_finite(simulation_raw.get("floor_z_m", -0.02), "simulation.floor_z_m"),
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        display_rate_hz=_positive(simulation_raw.get("display_rate_hz", 60.0), "simulation.display_rate_hz"),
        playback_speed=_positive(simulation_raw.get("playback_speed", 100.0), "simulation.playback_speed"),
        collision_sample_stride=int(simulation_raw.get("collision_sample_stride", 1)),
        ignored_contact_pairs=_contact_pairs(simulation_raw.get("ignored_contact_pairs", []), "simulation.ignored_contact_pairs"),
    )
    if not simulation.scene_template_path.is_file():
        raise ValueError(f"MuJoCo scene template does not exist: {simulation.scene_template_path}")
    if simulation.collision_sample_stride < 1:
        raise ValueError("simulation.collision_sample_stride must be positive")

    hardware = CoverageHardwareConfig(
        approved=bool(hardware_raw.get("approved", False)),
        preflight_report_path=_path(hardware_raw.get("preflight_report_path", "data/free_space_preflight.json"), base),
        max_tracking_error_rad=_positive_vector(hardware_raw.get("max_tracking_error_rad"), DOF, "hardware.max_tracking_error_rad"),
        max_abs_torque_nm=_positive_vector(hardware_raw.get("max_abs_torque_nm"), DOF, "hardware.max_abs_torque_nm"),
        max_timestamp_gap_s=_positive(hardware_raw.get("max_timestamp_gap_s", 0.1), "hardware.max_timestamp_gap_s"),
        start_move_speed_rad_s=_positive(hardware_raw.get("start_move_speed_rad_s", 0.25), "hardware.start_move_speed_rad_s"),
        record_cameras=bool(hardware_raw.get("record_cameras", False)),
    )
    return CoveragePlan(
        source_path=source,
        collection_config_path=_path(raw.get("collection_config", "master_slave_can.yaml"), base),
        pair_name=str(raw.get("pair", "main")),
        model=model,
        excitation=excitation,
        simulation=simulation,
        hardware=hardware,
    )


def generate_coverage_trajectory(plan: CoveragePlan) -> CoverageTrajectory:
    from calibration.simulation import MujocoPoseSafetyChecker

    _, model = build_reduced_model(plan.model)
    cfg = plan.excitation
    run = cfg.run
    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64) + cfg.joint_limit_margin_rad
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64) - cfg.joint_limit_margin_rad
    reference_q = _load_reference_joint_path(cfg)
    outside = np.any((reference_q < lower[None, :]) | (reference_q > upper[None, :]), axis=1)
    if np.any(outside):
        raise ValueError(
            f"reference H5 contains {int(np.count_nonzero(outside))} poses outside "
            "the configured URDF soft limits"
        )
    checker = MujocoPoseSafetyChecker(plan)
    safe_mask = checker.safe_mask(reference_q)
    if not np.all(safe_mask):
        raise ValueError(
            f"reference H5 contains {int(np.count_nonzero(~safe_mask))} poses rejected "
            "by the configured MuJoCo scene"
        )
    reference_xyz = _forward_kinematics_positions(
        model,
        reference_q,
        plan.simulation.end_effector_body,
    )
    workspace_min = np.min(reference_xyz, axis=0)
    workspace_max = np.max(reference_xyz, axis=0)
    hull_volume = _convex_hull_volume(reference_xyz)
    static_targets = _workspace_static_targets(
        reference_q,
        reference_xyz,
        cfg.workspace_voxel_counts,
        cfg.static_pose_count,
    )

    sample_count = int(round(run.duration_s * cfg.sample_rate_hz))
    section_counts = _allocate_counts(sample_count, cfg.section_fractions)
    if cfg.planner == "tau_refinement":
        segment_names = TAU_REFINEMENT_SEGMENT_NAMES
        representative_targets = _workspace_static_targets(
            reference_q,
            reference_xyz,
            cfg.workspace_voxel_counts,
            max(cfg.replay_pose_count, cfg.static_pose_count, cfg.jump_pose_count),
        )
        replay_waypoints = np.asarray(
            _order_static_targets(reference_q[0], representative_targets, checker),
            dtype=np.float64,
        )
    else:
        segment_names = SEGMENT_NAMES
        replay_waypoints = _simplify_reference_path(
            reference_q,
            cfg.waypoint_min_delta_rad,
        )
    log.info(
        "empirical workspace planner=%s source=%s samples=%d replay_waypoints=%d "
        "static_targets=%d "
        "xyz_min=%s xyz_max=%s hull_volume=%.6fm3",
        cfg.planner,
        cfg.reference_h5_path,
        reference_q.shape[0],
        replay_waypoints.shape[0],
        static_targets.shape[0],
        np.array2string(workspace_min, precision=4),
        np.array2string(workspace_max, precision=4),
        hull_volume,
    )
    start = replay_waypoints[0].copy()
    replay = _fit_waypoint_route(
        cfg,
        int(section_counts[0]),
        start,
        [(target, 0) for target in replay_waypoints[1:]],
        cfg.replay_speed_scale,
        "empirical workspace low-speed replay",
    )
    static_route = _order_static_targets(replay[-1], static_targets, checker)
    hold_samples = max(1, int(round(cfg.static_hold_s * cfg.sample_rate_hz)))
    static = _fit_waypoint_route(
        cfg,
        int(section_counts[1]),
        replay[-1],
        [(target, hold_samples) for target in static_route],
        cfg.static_transition_speed_scale,
        "empirical distributed static poses",
    )
    sections = [replay, static]
    if cfg.planner == "tau_refinement":
        rng = np.random.default_rng(run.seed)
        jump_targets = _bounded_jump_targets(
            static[-1],
            representative_targets,
            cfg.jump_pose_count,
            cfg.jump_min_delta_rad,
            cfg.jump_max_delta_rad,
            checker,
            rng,
        )
        jump = _fit_bounded_transition_route(
            cfg,
            int(section_counts[2]),
            static[-1],
            jump_targets,
            cfg.jump_transition_speed_scale,
            cfg.jump_hold_s,
            "bounded inference-like target switches",
        )
        sections.append(jump)
        log.info(
            "tau-refinement sections low_speed=%d static=%d target_switches=%d "
            "representative_targets=%d jump_events=%d",
            int(section_counts[0]),
            int(section_counts[1]),
            int(section_counts[2]),
            representative_targets.shape[0],
            len(jump_targets),
        )
    q = np.concatenate(sections, axis=0)
    dt = 1.0 / cfg.sample_rate_hz
    dq = np.gradient(q, dt, axis=0, edge_order=2)
    ddq = np.gradient(dq, dt, axis=0, edge_order=2)
    segment_id = np.concatenate(
        [
            np.full(count, index, dtype=np.int8)
            for index, count in enumerate(section_counts)
            if count > 0
        ]
    )
    trajectory = CoverageTrajectory(
        name=run.name,
        time_s=np.arange(sample_count, dtype=np.float64) * dt,
        q=q,
        dq=dq,
        ddq=ddq,
        segment_id=segment_id,
        segment_names=segment_names,
        reference_h5_path=str(cfg.reference_h5_path),
        reference_h5_sha256=_file_sha256(cfg.reference_h5_path),
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        workspace_convex_hull_volume_m3=hull_volume,
    )
    validate_coverage_trajectory(trajectory, plan)
    return trajectory


def validate_coverage_trajectory(
    trajectory: CoverageTrajectory,
    plan: CoveragePlan,
) -> None:
    cfg = plan.excitation
    count = trajectory.time_s.size
    if count < 3 or trajectory.q.shape != (count, DOF):
        raise ValueError("coverage trajectory must contain aligned (N, 7) joint samples")
    if trajectory.dq.shape != (count, DOF) or trajectory.ddq.shape != (count, DOF):
        raise ValueError("coverage trajectory derivatives must have shape (N, 7)")
    if trajectory.segment_id.shape != (count,):
        raise ValueError("coverage trajectory segment_id must have shape (N,)")
    expected_segments = (
        TAU_REFINEMENT_SEGMENT_NAMES
        if cfg.planner == "tau_refinement"
        else SEGMENT_NAMES
    )
    if tuple(trajectory.segment_names) != expected_segments:
        raise ValueError(f"coverage trajectory segments must be {expected_segments}")
    if np.any(trajectory.segment_id < 0) or np.any(
        trajectory.segment_id >= len(trajectory.segment_names)
    ):
        raise ValueError("coverage trajectory contains invalid segment identifiers")
    if not all(np.isfinite(value).all() for value in (trajectory.time_s, trajectory.q, trajectory.dq, trajectory.ddq)):
        raise ValueError("coverage trajectory contains non-finite values")
    dt = np.diff(trajectory.time_s)
    if np.any(dt <= 0) or not np.allclose(dt, 1.0 / cfg.sample_rate_hz, rtol=1e-8, atol=1e-10):
        raise ValueError("coverage trajectory timeline does not match configured sample rate")
    _, model = build_reduced_model(plan.model)
    lower = np.asarray(model.lowerPositionLimit) + cfg.joint_limit_margin_rad
    upper = np.asarray(model.upperPositionLimit) - cfg.joint_limit_margin_rad
    failures = {
        "position_lower": float(np.min(trajectory.q - lower[None, :])),
        "position_upper": float(np.min(upper[None, :] - trajectory.q)),
        "velocity": float(np.min(cfg.max_velocity_rad_s[None, :] - np.abs(trajectory.dq))),
        "acceleration": float(np.min(cfg.max_acceleration_rad_s2[None, :] - np.abs(trajectory.ddq))),
    }
    failed = {name: value for name, value in failures.items() if value < -1e-7}
    if failed:
        raise ValueError(f"coverage trajectory violates configured constraints: {failed}")
    if Path(trajectory.reference_h5_path).resolve() != cfg.reference_h5_path:
        raise ValueError("coverage trajectory was generated from a different reference H5")
    if trajectory.reference_h5_sha256 != _file_sha256(cfg.reference_h5_path):
        raise ValueError("reference H5 changed after trajectory generation; regenerate the trajectory")
    for name, value in (
        ("workspace_min_m", trajectory.workspace_min_m),
        ("workspace_max_m", trajectory.workspace_max_m),
    ):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"coverage trajectory {name} must be a finite XYZ vector")


def save_coverage_trajectory(path: str | Path, trajectory: CoverageTrajectory) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        name=np.asarray(trajectory.name),
        time_s=trajectory.time_s,
        q_cmd=trajectory.q,
        dq_cmd=trajectory.dq,
        ddq_cmd=trajectory.ddq,
        segment_id=trajectory.segment_id,
        segment_names=np.asarray(trajectory.segment_names),
        reference_h5_path=np.asarray(trajectory.reference_h5_path),
        reference_h5_sha256=np.asarray(trajectory.reference_h5_sha256),
        workspace_min_m=np.asarray(trajectory.workspace_min_m, dtype=np.float64),
        workspace_max_m=np.asarray(trajectory.workspace_max_m, dtype=np.float64),
        workspace_convex_hull_volume_m3=np.asarray(
            trajectory.workspace_convex_hull_volume_m3, dtype=np.float64
        ),
    )
    return output


def load_coverage_trajectory(path: str | Path) -> CoverageTrajectory:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        return CoverageTrajectory(
            name=str(np.asarray(data["name"]).item()),
            time_s=np.asarray(data["time_s"], dtype=np.float64),
            q=np.asarray(data["q_cmd"], dtype=np.float64),
            dq=np.asarray(data["dq_cmd"], dtype=np.float64),
            ddq=np.asarray(data["ddq_cmd"], dtype=np.float64),
            segment_id=np.asarray(data["segment_id"], dtype=np.int8),
            segment_names=tuple(str(value) for value in data["segment_names"]),
            reference_h5_path=(
                str(np.asarray(data["reference_h5_path"]).item())
                if "reference_h5_path" in data else ""
            ),
            reference_h5_sha256=(
                str(np.asarray(data["reference_h5_sha256"]).item())
                if "reference_h5_sha256" in data else ""
            ),
            workspace_min_m=(
                np.asarray(data["workspace_min_m"], dtype=np.float64)
                if "workspace_min_m" in data else None
            ),
            workspace_max_m=(
                np.asarray(data["workspace_max_m"], dtype=np.float64)
                if "workspace_max_m" in data else None
            ),
            workspace_convex_hull_volume_m3=(
                float(np.asarray(data["workspace_convex_hull_volume_m3"]).item())
                if "workspace_convex_hull_volume_m3" in data else 0.0
            ),
        )


def trajectory_sha256(trajectory: CoverageTrajectory) -> str:
    digest = hashlib.sha256()
    for value in (trajectory.time_s, trajectory.q, trajectory.dq, trajectory.ddq, trajectory.segment_id):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(trajectory.name.encode("utf-8"))
    for segment_name in trajectory.segment_names:
        digest.update(segment_name.encode("utf-8"))
        digest.update(b"\0")
    digest.update(trajectory.reference_h5_path.encode("utf-8"))
    digest.update(trajectory.reference_h5_sha256.encode("ascii"))
    return digest.hexdigest()


def config_sha256(path: str | Path) -> str:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(value, dict):
        hardware = value.get("hardware")
        if isinstance(hardware, dict):
            hardware = dict(hardware)
            hardware.pop("approved", None)
            value = dict(value)
            value["hardware"] = hardware
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_preflight_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_preflight_report(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preflight report must contain a JSON object")
    return value


def _load_reference_joint_path(cfg: CoverageExcitationConfig) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("empirical workspace planning requires h5py") from exc
    with h5py.File(cfg.reference_h5_path, "r") as h5:
        if cfg.reference_dataset not in h5:
            raise ValueError(
                f"reference H5 is missing dataset {cfg.reference_dataset!r}"
            )
        q = np.asarray(h5[cfg.reference_dataset], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != DOF or q.shape[0] < 3:
        raise ValueError(
            f"reference joint dataset must have shape (N, {DOF}); got {q.shape}"
        )
    if not np.isfinite(q).all():
        raise ValueError("reference joint dataset contains non-finite values")
    return q


def _forward_kinematics_positions(model, q: np.ndarray, frame_name: str) -> np.ndarray:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("empirical workspace planning requires pinocchio") from exc
    frame_id = int(model.getFrameId(frame_name))
    if frame_id >= len(model.frames):
        raise ValueError(f"Pinocchio model is missing end-effector frame {frame_name!r}")
    data = model.createData()
    positions = np.empty((q.shape[0], 3), dtype=np.float64)
    for index, pose in enumerate(q):
        pin.framesForwardKinematics(model, data, pose)
        positions[index] = np.asarray(data.oMf[frame_id].translation, dtype=np.float64)
    return positions


def _convex_hull_volume(xyz: np.ndarray) -> float:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("empirical workspace planning requires scipy") from exc
    if xyz.shape[0] < 4:
        return 0.0
    return float(ConvexHull(xyz).volume)


def _simplify_reference_path(q: np.ndarray, minimum_delta_rad: float) -> np.ndarray:
    selected = [0]
    for index in range(1, q.shape[0] - 1):
        if np.max(np.abs(q[index] - q[selected[-1]])) >= minimum_delta_rad:
            selected.append(index)
    if selected[-1] != q.shape[0] - 1:
        selected.append(q.shape[0] - 1)
    return q[np.asarray(selected, dtype=np.int64)]


def _workspace_static_targets(q, xyz, voxel_counts, target_count):
    workspace_min = np.min(xyz, axis=0)
    extent = np.maximum(np.max(xyz, axis=0) - workspace_min, 1e-9)
    normalized = np.clip((xyz - workspace_min) / extent, 0.0, 1.0)
    counts = np.asarray(voxel_counts, dtype=np.int64)
    voxel = np.minimum((normalized * counts[None, :]).astype(np.int64), counts - 1)
    representatives: dict[tuple[int, int, int], tuple[float, int]] = {}
    for index, key_array in enumerate(voxel):
        key = tuple(int(value) for value in key_array)
        center = (key_array.astype(np.float64) + 0.5) / counts
        distance = float(np.sum((normalized[index] - center) ** 2))
        previous = representatives.get(key)
        if previous is None or distance < previous[0]:
            representatives[key] = (distance, index)
    candidate_indices = {value[1] for value in representatives.values()}
    for axis in range(3):
        candidate_indices.add(int(np.argmin(xyz[:, axis])))
        candidate_indices.add(int(np.argmax(xyz[:, axis])))
    candidates = np.asarray(sorted(candidate_indices), dtype=np.int64)
    if candidates.size <= int(target_count):
        return q[candidates]

    selected_local: list[int] = []
    for axis in range(3):
        selected_local.append(int(np.argmin(normalized[candidates, axis])))
        selected_local.append(int(np.argmax(normalized[candidates, axis])))
    selected_local = list(dict.fromkeys(selected_local))
    minimum_distance = np.full(candidates.size, np.inf, dtype=np.float64)
    while len(selected_local) < int(target_count):
        latest = normalized[candidates[selected_local[-1]]]
        distance = np.sum((normalized[candidates] - latest[None, :]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected_local, dtype=np.int64)] = -1.0
        selected_local.append(int(np.argmax(minimum_distance)))
    return q[candidates[np.asarray(selected_local, dtype=np.int64)]]


def _order_static_targets(start, targets, checker):
    remaining = [np.asarray(target, dtype=np.float64) for target in targets]
    route: list[np.ndarray] = []
    current = np.asarray(start, dtype=np.float64)
    scale = np.maximum(np.ptp(np.stack(remaining), axis=0), 0.10)
    while remaining:
        order = np.argsort(
            [np.linalg.norm((target - current) / scale) for target in remaining]
        )
        selected = None
        for candidate_index in order:
            target = remaining[int(candidate_index)]
            if checker.leg_is_safe(current, target):
                selected = int(candidate_index)
                break
        if selected is None:
            raise ValueError("cannot safely connect the remaining empirical static targets")
        current = remaining.pop(selected)
        route.append(current)
    return route


def _bounded_jump_targets(
    start: np.ndarray,
    candidates: np.ndarray,
    target_count: int,
    minimum_delta_rad: float,
    maximum_delta_rad: float,
    checker,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != DOF:
        raise ValueError("bounded target-switch candidates must have shape (N, 7)")
    current = np.asarray(start, dtype=np.float64)
    selected: list[np.ndarray] = []
    amplitude_cycle = np.linspace(minimum_delta_rad, maximum_delta_rad, 5)
    for event_index in range(int(target_count)):
        delta = np.max(np.abs(candidates - current[None, :]), axis=1)
        eligible = np.flatnonzero(
            (delta >= minimum_delta_rad) & (delta <= maximum_delta_rad)
        )
        if eligible.size == 0:
            raise ValueError(
                "cannot find an empirical target within the configured jump amplitude "
                f"range [{minimum_delta_rad:.3f}, {maximum_delta_rad:.3f}] rad"
            )
        desired = amplitude_cycle[event_index % amplitude_cycle.size]
        jitter = rng.uniform(0.0, 1.0e-6, size=eligible.size)
        order = eligible[np.argsort(np.abs(delta[eligible] - desired) + jitter)]
        target = None
        for candidate_index in order:
            candidate = candidates[int(candidate_index)]
            if checker.leg_is_safe(current, candidate):
                target = candidate.copy()
                break
        if target is None:
            raise ValueError(
                "cannot safely connect an empirical target-switch leg within the "
                "configured jump amplitude range"
            )
        selected.append(target)
        current = target
    return selected


def _fit_bounded_transition_route(
    cfg: CoverageExcitationConfig,
    count: int,
    start: np.ndarray,
    targets: list[np.ndarray],
    speed_scale: float,
    hold_s: float,
    description: str,
) -> np.ndarray:
    if not targets:
        raise ValueError(f"{description} requires at least one target")
    speed_pattern = np.asarray((0.55, 0.70, 0.85, 1.00), dtype=np.float64)
    cursor = np.asarray(start, dtype=np.float64)
    legs: list[tuple[np.ndarray, int]] = []
    for index, target in enumerate(targets):
        target = np.asarray(target, dtype=np.float64)
        leg_speed_scale = float(speed_scale * speed_pattern[index % speed_pattern.size])
        required = _required_leg_samples(cfg, cursor, target, leg_speed_scale)
        legs.append((target, required))
        cursor = target

    base_hold = max(1, int(round(float(hold_s) * cfg.sample_rate_hz)))
    available = int(count) - 1
    minimum = sum(required + base_hold for _, required in legs)
    if minimum > available:
        raise ValueError(
            f"{description} is too short: needs {minimum / cfg.sample_rate_hz:.2f}s, "
            f"allocated {int(count) / cfg.sample_rate_hz:.2f}s"
        )
    extra_holds = _allocate_counts(
        available - sum(required for _, required in legs),
        np.ones(len(legs), dtype=np.float64),
    )
    values = [np.asarray(start, dtype=np.float64).copy()]
    current = values[0]
    for (target, required), hold_count in zip(legs, extra_holds):
        current = _append_leg_with_count(values, current, target, int(required))
        values.extend(current.copy() for _ in range(int(hold_count)))
    if len(values) != int(count):
        raise RuntimeError(f"{description} produced the wrong sample count")
    return np.stack(values, axis=0)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_waypoint_route(cfg, count, start, route, speed_scale, description):
    variable_route = [
        (target, hold, float(speed_scale)) for target, hold in route
    ]
    return _fit_variable_speed_route(cfg, count, start, variable_route, description)


def _fit_variable_speed_route(cfg, count, start, route, description):
    legs: list[tuple[np.ndarray, np.ndarray, int]] = []
    speed_scales: list[float] = []
    cursor = np.asarray(start, dtype=np.float64)
    for target, hold, speed_scale in route:
        target = np.asarray(target, dtype=np.float64)
        legs.append((cursor, target, int(hold)))
        speed_scales.append(float(speed_scale))
        cursor = target
    required = np.asarray(
        [
            _required_leg_samples(cfg, first, target, speed_scale)
            for (first, target, _), speed_scale in zip(legs, speed_scales)
        ],
        dtype=int,
    )
    total_hold = sum(hold for _, _, hold in legs)
    available = int(count) - 1
    minimum = int(np.sum(required)) + total_hold
    if minimum > available:
        raise ValueError(
            f"{description} is too short: needs {minimum / cfg.sample_rate_hz:.2f}s, "
            f"allocated {int(count) / cfg.sample_rate_hz:.2f}s"
        )
    leg_counts = required + _proportional_extra_counts(
        available - minimum,
        required,
    )
    values = [np.asarray(start, dtype=np.float64).copy()]
    current = values[0]
    for (_, target, hold), leg_count in zip(legs, leg_counts):
        current = _append_leg_with_count(values, current, target, int(leg_count))
        values.extend([current.copy() for _ in range(hold)])
    if len(values) != int(count):
        raise RuntimeError(f"{description} produced the wrong sample count")
    return np.stack(values, axis=0)


def _required_leg_samples(cfg, start, target, speed_scale):
    delta = np.abs(np.asarray(target) - np.asarray(start))
    if np.max(delta) < 1e-12:
        return 0
    velocity_duration = np.max(1.875 * delta / (cfg.max_velocity_rad_s * speed_scale))
    acceleration_duration = np.max(np.sqrt(5.8 * delta / cfg.max_acceleration_rad_s2))
    duration = max(0.10, float(velocity_duration), float(acceleration_duration))
    return max(2, int(np.ceil(duration * cfg.sample_rate_hz)) + 2)


def _append_leg_with_count(values, start, target, sample_count):
    if sample_count == 0:
        return np.asarray(target, dtype=np.float64).copy()
    phase = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)[1:]
    blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    leg = start[None, :] + blend[:, None] * (target - start)[None, :]
    values.extend(row.copy() for row in leg)
    return target.copy()


def _proportional_extra_counts(total, weights):
    if total < 0:
        raise ValueError("trajectory leg allocation cannot have negative slack")
    result = np.zeros(len(weights), dtype=int)
    if total == 0:
        return result
    weights = np.asarray(weights, dtype=np.float64)
    if np.sum(weights) <= 0:
        result[0] = total
        return result
    exact = total * weights / np.sum(weights)
    result = np.floor(exact).astype(int)
    remainder = total - int(np.sum(result))
    if remainder:
        order = np.argsort(-(exact - result))
        result[order[:remainder]] += 1
    return result


def _allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights > 0
    if not np.any(positive):
        raise ValueError("trajectory allocation requires at least one positive weight")
    if total < int(np.count_nonzero(positive)):
        raise ValueError("trajectory section is too short for its configured subdivisions")
    exact = total * weights / np.sum(weights)
    counts = np.floor(exact).astype(int)
    counts[positive & (counts == 0)] = 1
    remainder = total - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for index in order[:remainder]:
            counts[index] += 1
    elif remainder < 0:
        order = np.argsort(exact - np.floor(exact))
        for index in order:
            if remainder == 0:
                break
            if positive[index] and counts[index] > 1:
                counts[index] -= 1
                remainder += 1
    if int(np.sum(counts)) != total:
        raise RuntimeError("failed to allocate exact trajectory sample count")
    return counts


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


def _fraction(value: Any, name: str, *, include_one: bool = False) -> float:
    result = _finite(value, name)
    upper_valid = result <= 1.0 if include_one else result < 1.0
    if result <= 0 or not upper_valid:
        operator = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{name} must be in {operator}")
    return result


def _contact_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    pairs = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{name}[{index}] must contain two geometry names")
        first, second = sorted(str(part).strip() for part in item)
        if not first or not second or first == second:
            raise ValueError(f"{name}[{index}] contains invalid geometry names")
        pairs.append((first, second))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{name} contains duplicate pairs")
    return tuple(pairs)
