from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_multicamera_can_stability import (
    IntervalTracker,
    Thresholds,
    _configure_read_only_alignment,
    _select_endpoints,
    evaluate_findings,
)


def test_interval_tracker_ignores_duplicates_and_reports_source_rate() -> None:
    tracker = IntervalTracker()
    tracker.observe(1_000_000, 1.0)
    tracker.observe(1_000_000, 1.005)
    tracker.observe(1_010_000, 1.010)
    tracker.observe(1_020_000, 1.020)

    summary = tracker.summary()

    assert summary["samples"] == 3
    assert summary["frequency_hz"] == pytest.approx(100.0)
    assert summary["maximum_gap_ms"] == pytest.approx(10.0)
    assert summary["maximum_silence_ms"] == pytest.approx(5.0)


def test_interval_tracker_reports_timestamp_regression() -> None:
    tracker = IntervalTracker()
    tracker.observe(20_000, 1.0)

    assert not tracker.observe(19_000, 1.1)
    assert tracker.summary()["timestamp_regressions"] == 1


def test_select_endpoints_defaults_to_both_arms_in_all_pairs() -> None:
    leader = SimpleNamespace(name="leader")
    follower = SimpleNamespace(name="follower")
    config = SimpleNamespace(
        teleop=SimpleNamespace(
            master_slave=(SimpleNamespace(name="main", leader=leader, follower=follower),)
        )
    )

    assert _select_endpoints(config, None, "both") == [leader, follower]
    assert _select_endpoints(config, ["main"], "follower") == [follower]


def test_read_only_arm_setup_only_configures_state_alignment() -> None:
    calls = []

    class Arm:
        def configure_state_alignment(self, *args):
            calls.append(args)

        def __getattr__(self, name):
            if name.startswith(("enable", "disable", "set_", "move_", "command_")):
                raise AssertionError(f"unexpected control lookup: {name}")
            raise AttributeError(name)

    state = SimpleNamespace(mean_window=10, lowpass=True, lowpass_cutoff_hz=5.0)
    config = SimpleNamespace(
        robot_states={"q": state, "velocity": state, "acceleration": state},
        teleop=SimpleNamespace(
            command=SimpleNamespace(
                state_alignment_delay_s=0.015,
                sample_rate_hz=100.0,
                maximum_can_frame_gap_s=0.03,
            )
        ),
    )

    _configure_read_only_alignment(Arm(), config)

    assert calls == [(0.015, 100.0, 10, 5.0, 5.0, 5.0, 0.03)]


def test_evaluate_findings_detects_camera_induced_can_gap_growth() -> None:
    thresholds = Thresholds(
        poll_rate_hz=100.0,
        maximum_can_gap_ms=30.0,
        minimum_output_rate_ratio=0.9,
        minimum_camera_rate_ratio=0.8,
        maximum_loop_gap_ms=50.0,
        maximum_abs_dq_rad_s=1.0,
        maximum_abs_ddq_rad_s2=20.0,
        maximum_p99_gap_growth=1.5,
        p99_growth_margin_ms=2.0,
    )

    def arm(p99: float) -> dict:
        return {
            "invalid_state_reads": 0,
            "output": {"frequency_hz": 100.0},
            "raw_streams": {
                "joint_12": {
                    "p99_gap_ms": p99,
                    "maximum_gap_ms": p99,
                    "maximum_silence_ms": p99,
                    "timestamp_regressions": 0,
                }
            },
            "maximum_abs_reported_dq_rad_s": np.zeros(7).tolist(),
            "maximum_abs_reported_ddq_rad_s2": np.zeros(7).tolist(),
        }

    phases = {
        "can_only": {
            "duration_s": 10.0,
            "loop": {"maximum": 10.0},
            "arms": {"arm": arm(10.0)},
            "cameras": {},
        },
        "camera_startup": {
            "duration_s": 1.0,
            "loop": {"maximum": 10.0},
            "arms": {"arm": arm(10.0)},
            "cameras": {},
        },
        "can_with_cameras": {
            "duration_s": 10.0,
            "loop": {"maximum": 10.0},
            "arms": {"arm": arm(18.0)},
            "cameras": {"wrist": {"frequency_hz": 20.0}},
        },
    }

    findings = evaluate_findings(
        phases,
        camera_fps={"wrist": 20.0},
        thresholds=thresholds,
    )

    assert any("p99 CAN gap grew" in finding for finding in findings)


def test_evaluate_findings_accepts_stable_read_only_streams() -> None:
    thresholds = Thresholds(
        poll_rate_hz=100.0,
        maximum_can_gap_ms=30.0,
        minimum_output_rate_ratio=0.9,
        minimum_camera_rate_ratio=0.8,
        maximum_loop_gap_ms=50.0,
        maximum_abs_dq_rad_s=1.0,
        maximum_abs_ddq_rad_s2=20.0,
        maximum_p99_gap_growth=1.5,
        p99_growth_margin_ms=2.0,
    )
    stream = {
        "p99_gap_ms": 10.0,
        "maximum_gap_ms": 11.0,
        "maximum_silence_ms": 11.0,
        "timestamp_regressions": 0,
    }
    arm = {
        "invalid_state_reads": 0,
        "output": {"frequency_hz": 100.0},
        "raw_streams": {"joint_12": stream},
        "maximum_abs_reported_dq_rad_s": np.zeros(7).tolist(),
        "maximum_abs_reported_ddq_rad_s2": np.zeros(7).tolist(),
    }
    phases = {
        "can_only": {
            "duration_s": 10.0,
            "loop": {"maximum": 11.0},
            "arms": {"arm": arm},
            "cameras": {},
        },
        "camera_startup": {
            "duration_s": 1.0,
            "loop": {"maximum": 11.0},
            "arms": {"arm": arm},
            "cameras": {},
        },
        "can_with_cameras": {
            "duration_s": 10.0,
            "loop": {"maximum": 11.0},
            "arms": {"arm": arm},
            "cameras": {"wrist": {"frequency_hz": 20.0}},
        },
    }

    assert evaluate_findings(
        phases,
        camera_fps={"wrist": 20.0},
        thresholds=thresholds,
    ) == []
