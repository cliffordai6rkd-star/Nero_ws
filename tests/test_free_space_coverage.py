from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from calibration.free_space_cli import (
    FREE_SPACE_EPISODE_MAX_SAMPLES,
    _SampleTiming,
    _check_measurement,
    _format_sample_timing,
    _format_timing_diagnostics,
    _verify_hardware_preflight,
)
from calibration.free_space_coverage import (
    JOINT_POSE_COVERAGE_SEGMENT_NAMES,
    REPRESENTATIVE_REPLAY_SEGMENT_NAMES,
    TAU_REFINEMENT_SEGMENT_NAMES,
    _build_joint_pose_coverage_tree,
    _fit_joint_pose_coverage_route,
    config_sha256,
    generate_coverage_trajectory,
    load_coverage_plan,
    trajectory_sha256,
    validate_coverage_trajectory,
    write_preflight_report,
)
from nero_collection.config import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tau_refinement_coverage.yaml"
JOINT_COVERAGE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "joint_pose_coverage.yaml"
)
REPLAY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "representative_replay.yaml"
)


class _AlwaysSafeChecker:
    def safe_mask(self, q):
        return np.ones(np.asarray(q).shape[0], dtype=bool)

    def leg_is_safe(self, _start, _target):
        return True


def test_config_defines_thirty_minute_tau_refinement_run() -> None:
    plan = load_coverage_plan(CONFIG)
    collection = load_config(plan.collection_config_path)

    assert plan.excitation.sample_rate_hz == pytest.approx(100.0)
    assert collection.output.directory.name == "next_data_recollected"
    assert collection.output.prefix == "episode"
    assert plan.excitation.planner == "tau_refinement"
    assert plan.excitation.run.name == "tau_refinement_low_static_switches"
    assert plan.excitation.run.duration_s == pytest.approx(1800.0)
    assert not hasattr(plan.excitation, "profiles")
    assert 1800.0 * plan.excitation.section_fractions == pytest.approx(
        [810.0, 630.0, 360.0]
    )
    stored = 180_000 - int(round(collection.output.discard_initial_s * 100.0))
    assert (stored + FREE_SPACE_EPISODE_MAX_SAMPLES - 1) // FREE_SPACE_EPISODE_MAX_SAMPLES == 6


def test_joint_pose_coverage_config_is_50hz_with_half_second_holds() -> None:
    plan = load_coverage_plan(JOINT_COVERAGE_CONFIG)

    assert plan.excitation.planner == "joint_pose_coverage"
    assert plan.excitation.sample_rate_hz == pytest.approx(50.0)
    assert plan.excitation.static_hold_s == pytest.approx(0.5)
    assert plan.excitation.joint_pose_count == 160
    assert plan.excitation.joint_candidate_count == 8192
    assert plan.excitation.joint_position_min_rad == pytest.approx(
        [-1.10, -0.60, -0.95, 0.55, -1.35, -0.62, -0.10]
    )
    assert plan.excitation.joint_position_max_rad == pytest.approx(
        [1.15, 1.00, 1.10, 2.08, 1.30, 0.88, 1.51]
    )


def test_representative_replay_selects_traceable_teleop_episodes() -> None:
    plan = load_coverage_plan(REPLAY_CONFIG)
    trajectory = generate_coverage_trajectory(plan)

    assert plan.excitation.planner == "representative_replay"
    assert plan.excitation.sample_rate_hz == pytest.approx(50.0)
    assert trajectory.segment_names == REPRESENTATIVE_REPLAY_SEGMENT_NAMES
    assert [Path(value).name for value in trajectory.source_h5_paths] == [
        "episode_0000_20260811_163210.h5",
        "episode_0014_20260811_202154.h5",
        "episode_0020_20260811_221223.h5",
        "episode_0027_20260812_154638.h5",
    ]
    assert np.all(
        np.max(np.abs(trajectory.dq), axis=0)
        <= plan.excitation.max_velocity_rad_s + 1.0e-7
    )
    assert np.all(
        np.max(np.abs(trajectory.ddq), axis=0)
        <= plan.excitation.max_acceleration_rad_s2 + 1.0e-7
    )


def test_joint_pose_tree_spans_multiple_joints_and_route_contains_holds() -> None:
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    nodes, parents = _build_joint_pose_coverage_tree(
        np.zeros(7),
        lower,
        upper,
        candidate_count=128,
        pose_count=12,
        connection_step_rad=0.4,
        checker=_AlwaysSafeChecker(),
        seed=17,
    )
    cfg = SimpleNamespace(
        sample_rate_hz=50.0,
        static_hold_s=0.5,
        joint_transition_speed_scale=0.65,
        max_velocity_rad_s=np.ones(7),
        max_acceleration_rad_s2=np.full(7, 3.0),
    )
    q, segment_id = _fit_joint_pose_coverage_route(cfg, 2000, nodes, parents)

    assert nodes.shape == (12, 7)
    assert parents.shape == (12,)
    assert np.all(parents[1:] < np.arange(1, 12))
    assert np.count_nonzero(np.abs(np.diff(nodes, axis=0)) > 1e-4, axis=1).min() >= 2
    assert q.shape == (2000, 7)
    assert set(np.unique(segment_id)) == {0, 1}
    assert np.count_nonzero(segment_id == 1) >= 12 * 25
    assert JOINT_POSE_COVERAGE_SEGMENT_NAMES == (
        "joint_pose_transition",
        "joint_pose_hold",
    )


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_generated_coverage_is_100hz_bounded_and_has_expected_segments() -> None:
    plan = load_coverage_plan(CONFIG)
    trajectory = generate_coverage_trajectory(plan)
    validate_coverage_trajectory(trajectory, plan)
    assert trajectory.time_s.size == 180_000
    assert np.diff(trajectory.time_s) == pytest.approx(0.01)
    assert np.all(
        np.max(np.abs(trajectory.dq), axis=0)
        <= plan.excitation.max_velocity_rad_s + 1e-7
    )
    assert np.all(
        np.max(np.abs(trajectory.ddq), axis=0)
        <= plan.excitation.max_acceleration_rad_s2 + 1e-7
    )
    assert trajectory.segment_names == TAU_REFINEMENT_SEGMENT_NAMES
    assert np.bincount(trajectory.segment_id, minlength=3).tolist() == [81000, 63000, 36000]
    low_speed = trajectory.segment_id == 0
    assert np.all(
        np.max(np.abs(trajectory.dq[low_speed]), axis=0)
        <= plan.excitation.max_velocity_rad_s * plan.excitation.replay_speed_scale
        + 1e-7
    )
    static = trajectory.segment_id == 1
    switches = trajectory.segment_id == 2
    assert np.count_nonzero(np.max(np.abs(trajectory.dq[static]), axis=1) < 0.01) > 20_000
    assert np.max(np.abs(trajectory.dq[switches])) > 4.0 * np.max(
        np.abs(trajectory.dq[low_speed])
    )

    import h5py

    with h5py.File(plan.excitation.reference_h5_path, "r") as h5:
        human_q = np.asarray(h5[plan.excitation.reference_dataset], dtype=np.float64)
    assert np.all(
        np.min(trajectory.q, axis=0) >= np.min(human_q, axis=0) - 1e-7
    )
    assert np.all(
        np.max(trajectory.q, axis=0) <= np.max(human_q, axis=0) + 1e-7
    )


def test_config_fingerprint_ignores_only_hardware_approval(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(raw), encoding="utf-8")
    raw["hardware"]["approved"] = not raw["hardware"]["approved"]
    second.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert config_sha256(first) == config_sha256(second)

    raw["trajectory"]["sample_rate_hz"] = 99.0
    second.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert config_sha256(first) != config_sha256(second)


def test_free_space_timing_diagnostics_report_first_sample_and_stable_p95() -> None:
    first = _SampleTiming(0, 0.001, 0.002, 0.003, 0.070, 0.004, 0.080)
    stable = _SampleTiming(1, 0.001, 0.002, 0.001, 0.0008, 0.0002, 0.005)

    assert _format_sample_timing(first) == (
        "sample=0 command=1.000ms read=2.000ms safety=3.000ms "
        "append=70.000ms camera=4.000ms total=80.000ms"
    )
    assert _format_timing_diagnostics([first]) == (
        "previous [sample=0 command=1.000ms read=2.000ms safety=3.000ms "
        "append=70.000ms camera=4.000ms total=80.000ms]; stable samples=0"
    )
    diagnostics = _format_timing_diagnostics([first, stable])
    assert "stable samples=1" in diagnostics
    assert "total mean=5.000ms p95=5.000ms max=5.000ms" in diagnostics
    assert "append mean=0.800ms p95=0.800ms max=0.800ms" in diagnostics


def test_tracking_error_reports_trajectory_context() -> None:
    state = SimpleNamespace(q=np.zeros(7), torque=np.zeros(7))
    plan = SimpleNamespace(
        hardware=SimpleNamespace(
            max_tracking_error_rad=np.full(7, 0.1),
            max_abs_torque_nm=np.full(7, 10.0),
        )
    )

    with pytest.raises(RuntimeError) as caught:
        _check_measurement(
            state,
            np.asarray([0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            plan,
            np.full(7, -1.0),
            np.full(7, 1.0),
            sample_index=2545,
            trajectory_time_s=25.45,
            segment_name="single_joint",
            dq_cmd=np.asarray([1.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    message = str(caught.value)
    assert "sample=2545 time=25.450s segment=single_joint" in message
    assert "dq_cmd=[1.19 0." in message


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_hardware_preflight_rejects_changed_trajectory(tmp_path: Path) -> None:
    plan = load_coverage_plan(CONFIG)
    trajectory = generate_coverage_trajectory(plan)
    report_path = tmp_path / "preflight.json"
    write_preflight_report(
        report_path,
        {
            "passed": True,
            "config_sha256": config_sha256(plan.source_path),
            "trajectory": {
                "passed": True,
                "trajectory_sha256": trajectory_sha256(trajectory),
            },
        },
    )
    test_plan = replace(
        plan,
        hardware=replace(plan.hardware, preflight_report_path=report_path),
    )
    _verify_hardware_preflight(test_plan, trajectory)

    changed = replace(trajectory, q=trajectory.q.copy())
    changed.q[100, 0] += 1e-4
    with pytest.raises(RuntimeError, match="changed after simulation"):
        _verify_hardware_preflight(test_plan, changed)
