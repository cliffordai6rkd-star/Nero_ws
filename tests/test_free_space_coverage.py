from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import h5py

from calibration.free_space_cli import (
    FREE_SPACE_EPISODE_MAX_SAMPLES,
    _SampleTiming,
    _check_measurement,
    _execute_trajectory,
    _format_sample_timing,
    _format_timing_diagnostics,
    _verify_hardware_preflight,
    _wait_for_static_stability,
)
from calibration.free_space_coverage import (
    JOINT_POSE_COVERAGE_SEGMENT_NAMES,
    REPRESENTATIVE_REPLAY_SEGMENT_NAMES,
    TAU_REFINEMENT_SEGMENT_NAMES,
    _build_joint_pose_coverage_tree,
    _fit_joint_pose_coverage_route,
    _select_safe_joint_pose_root,
    config_sha256,
    empirical_joint_range_statistics,
    effective_joint_position_limits,
    generate_coverage_trajectory,
    hardware_joint_position_limits,
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


def test_joint_pose_coverage_config_is_short_conservative_static_run() -> None:
    plan = load_coverage_plan(JOINT_COVERAGE_CONFIG)

    assert plan.excitation.planner == "joint_pose_coverage"
    assert plan.excitation.sample_rate_hz == pytest.approx(100.0)
    assert plan.excitation.run.duration_s == pytest.approx(1200.0)
    assert plan.excitation.static_hold_s == pytest.approx(0.8)
    assert plan.excitation.joint_pose_count == 20
    assert plan.excitation.joint_route_passes == 8
    assert plan.excitation.joint_candidate_count == 4096
    assert plan.excitation.joint_range_fraction == pytest.approx(0.75)
    assert plan.excitation.joint_position_min_rad is None
    assert plan.excitation.joint_position_max_rad is None
    assert plan.excitation.joint_range_source_directory.name == (
        "next_background_data"
    )
    assert plan.excitation.joint_range_quantiles == pytest.approx([0.01, 0.99])
    assert plan.excitation.joint_range_exclude_sources == ("free_space_coverage",)
    assert plan.excitation.reference_h5_path is None
    assert plan.excitation.static_position_threshold_rad == pytest.approx(0.04)
    assert plan.excitation.static_velocity_threshold_rad_s == pytest.approx(0.03)
    assert plan.excitation.static_stability_duration_s == pytest.approx(0.25)
    assert plan.excitation.static_stability_timeout_s == pytest.approx(3.0)


def test_effective_limits_apply_margin_then_centered_range_fraction() -> None:
    plan = load_coverage_plan(JOINT_COVERAGE_CONFIG)
    model = SimpleNamespace(
        lowerPositionLimit=np.full(7, -2.0),
        upperPositionLimit=np.full(7, 2.0),
    )

    lower, upper = effective_joint_position_limits(model, plan.excitation)

    stats = empirical_joint_range_statistics(plan.excitation)
    centered_lower = np.full(7, -1.44)
    centered_upper = np.full(7, 1.44)
    assert lower == pytest.approx(
        np.maximum(centered_lower, stats["quantile_minimum"])
    )
    assert upper == pytest.approx(
        np.minimum(centered_upper, stats["quantile_maximum"])
    )


def test_empirical_joint_range_excludes_automatic_collection(tmp_path: Path) -> None:
    manual = np.asarray(
        [
            np.linspace(-0.7, 0.1, 7),
            np.linspace(-0.3, 0.5, 7),
            np.linspace(0.1, 0.9, 7),
        ],
        dtype=np.float64,
    )
    automatic = np.full((3, 7), 10.0, dtype=np.float64)

    def write_episode(path: Path, q: np.ndarray, metadata: dict) -> None:
        with h5py.File(path, "w") as h5:
            teleop = h5.create_group("teleop")
            teleop.create_dataset("q_follower", data=q)
            meta = h5.create_group("metadata")
            meta.create_dataset("episode_json", data=json.dumps(metadata))

    write_episode(tmp_path / "manual.h5", manual, {"source": "teleop"})
    write_episode(
        tmp_path / "automatic.h5",
        automatic,
        {"source": "free_space_coverage"},
    )
    cfg = SimpleNamespace(
        joint_range_source_directory=tmp_path,
        joint_range_exclude_sources=("free_space_coverage",),
        joint_range_quantiles=np.asarray([0.0, 1.0]),
        reference_dataset="teleop/q_follower",
    )

    stats = empirical_joint_range_statistics(cfg)

    assert stats["sample_count"] == 3
    assert [path.name for path in stats["included_paths"]] == ["manual.h5"]
    assert [path.name for path in stats["excluded_paths"]] == ["automatic.h5"]
    assert stats["minimum"] == pytest.approx(np.min(manual, axis=0))
    assert stats["maximum"] == pytest.approx(np.max(manual, axis=0))


def test_hardware_limits_do_not_apply_target_coverage_bounds() -> None:
    plan = load_coverage_plan(JOINT_COVERAGE_CONFIG)
    model = SimpleNamespace(
        lowerPositionLimit=np.full(7, -2.0),
        upperPositionLimit=np.full(7, 2.0),
    )

    lower, upper = hardware_joint_position_limits(model, plan.excitation)

    assert lower == pytest.approx(np.full(7, -1.92))
    assert upper == pytest.approx(np.full(7, 1.92))


def test_safe_joint_pose_root_prefers_center_and_is_deterministic() -> None:
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    first = _select_safe_joint_pose_root(lower, upper, _AlwaysSafeChecker(), seed=31)
    second = _select_safe_joint_pose_root(lower, upper, _AlwaysSafeChecker(), seed=31)

    assert first.shape == (1, 7)
    assert first == pytest.approx(np.zeros((1, 7)))
    assert second == pytest.approx(first)


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

    repeated_nodes, repeated_parents = _build_joint_pose_coverage_tree(
        np.zeros(7),
        lower,
        upper,
        candidate_count=128,
        pose_count=12,
        connection_step_rad=0.4,
        checker=_AlwaysSafeChecker(),
        seed=17,
    )

    assert nodes.shape == (12, 7)
    assert parents.shape == (12,)
    assert np.array_equal(parents[1:], np.arange(11))
    assert repeated_nodes == pytest.approx(nodes)
    assert np.array_equal(repeated_parents, parents)
    assert np.count_nonzero(np.abs(np.diff(nodes, axis=0)) > 1e-4, axis=1).min() >= 2
    normalized = (nodes - lower) / (upper - lower)
    pair_distance = np.linalg.norm(
        normalized[:, None, :] - normalized[None, :, :], axis=2
    )
    pair_distance += np.eye(nodes.shape[0]) * 100.0
    assert np.min(pair_distance) > 0.05
    assert q.shape == (2000, 7)
    assert set(np.unique(segment_id)) == {0, 1}
    assert np.count_nonzero(segment_id == 1) == 12 * 25
    assert JOINT_POSE_COVERAGE_SEGMENT_NAMES == (
        "joint_pose_transition",
        "joint_pose_hold",
    )


def test_static_stability_requires_one_continuous_stable_window(monkeypatch) -> None:
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    class Arm:
        def __init__(self):
            self.reads = 0
            self.commands = 0

        def command_joint_positions(self, _target):
            self.commands += 1

        def read_state(self):
            velocity = 0.04 if self.reads == 2 else 0.0
            self.reads += 1
            return SimpleNamespace(
                q=np.full(7, 0.005),
                dq=np.full(7, velocity),
                torque=np.zeros(7),
            )

    clock = Clock()
    monkeypatch.setattr("calibration.free_space_cli.time.monotonic", clock.monotonic)
    monkeypatch.setattr("calibration.free_space_cli.time.sleep", clock.sleep)
    plan = SimpleNamespace(
        excitation=SimpleNamespace(
            sample_rate_hz=10.0,
            static_stability_timeout_s=2.0,
            static_stability_duration_s=0.25,
            static_position_threshold_rad=0.02,
            static_velocity_threshold_rad_s=0.03,
        ),
        hardware=SimpleNamespace(
            max_timestamp_gap_s=0.2,
            max_tracking_error_rad=np.ones(7),
            max_abs_torque_nm=np.ones(7),
        ),
    )
    arm = Arm()

    def capture_state():
        if arm.reads:
            clock.sleep(0.1)
        arm.command_joint_positions(np.zeros(7))
        state = arm.read_state()
        return SimpleNamespace(state=state)

    _wait_for_static_stability(
        capture_state,
        np.zeros(7),
        plan,
        sample_index=8,
        trajectory_time_s=1.6,
    )

    assert arm.reads == 7
    assert arm.commands == arm.reads
    assert clock.now == pytest.approx(0.6)


def test_static_settling_commands_and_reads_are_recorded_continuously(monkeypatch) -> None:
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    class Arm:
        def __init__(self):
            self.commands = 0
            self.reads = 0

        def command_joint_positions(self, _target):
            self.commands += 1

        def read_state(self):
            self.reads += 1
            return SimpleNamespace(
                q=np.zeros(7),
                dq=np.zeros(7),
                torque=np.zeros(7),
                current=np.zeros(7),
                ee_pose=np.eye(4),
            )

    class Buffer:
        def __init__(self):
            self.config = SimpleNamespace(
                output=SimpleNamespace(discard_initial_s=0.0)
            )
            self.sample_count = 0
            self.timestamps = []

        def append_teleop(self, timestamp, _values, *, store):
            self.sample_count += int(store)
            if store:
                self.timestamps.append(timestamp)

    clock = Clock()
    monkeypatch.setattr("calibration.free_space_cli.time.monotonic", clock.monotonic)
    monkeypatch.setattr("calibration.free_space_cli.time.sleep", clock.sleep)
    monkeypatch.setattr("nero_collection.fixed_rate.time.monotonic", clock.monotonic)
    monkeypatch.setattr("nero_collection.fixed_rate.time.sleep", clock.sleep)
    monkeypatch.setattr(
        "nero_collection.fixed_rate.now_us",
        lambda: int(round(clock.now * 1.0e6)) + 1,
    )
    trajectory = SimpleNamespace(
        q=np.zeros((3, 7)),
        dq=np.zeros((3, 7)),
        time_s=np.arange(3, dtype=np.float64) * 0.1,
        segment_id=np.ones(3, dtype=np.int8),
        segment_names=JOINT_POSE_COVERAGE_SEGMENT_NAMES,
    )
    plan = SimpleNamespace(
        excitation=SimpleNamespace(
            planner="joint_pose_coverage",
            sample_rate_hz=10.0,
            static_stability_timeout_s=1.0,
            static_stability_duration_s=0.2,
            static_position_threshold_rad=0.02,
            static_velocity_threshold_rad_s=0.03,
        ),
        hardware=SimpleNamespace(
            max_timestamp_gap_s=0.2,
            max_tracking_error_rad=np.ones(7),
            max_abs_torque_nm=np.ones(7),
        ),
    )
    arm = Arm()
    buffer = Buffer()

    _execute_trajectory(
        arm,
        SimpleNamespace(poll=lambda: ()),
        buffer,
        trajectory,
        plan,
        np.full(7, -1.0),
        np.full(7, 1.0),
    )

    assert buffer.sample_count == 6
    assert arm.reads == 6
    assert arm.commands == 6
    assert np.diff(buffer.timestamps) == pytest.approx(np.full(5, 100_000))


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
