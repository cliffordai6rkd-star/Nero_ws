#!/usr/bin/env python3
"""Read-only stress test for simultaneous Nero CAN and camera acquisition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.arms.base import ArmInterface, ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.cameras import CameraFrame, CameraManager
from nero_collection.config import ArmEndpointConfig, CollectionConfig, load_config


log = logging.getLogger(__name__)


@dataclass
class IntervalTracker:
    first_timestamp_us: int | None = None
    last_timestamp_us: int | None = None
    intervals_ms: list[float] = field(default_factory=list)
    regressions: int = 0
    last_change_monotonic_s: float | None = None
    maximum_silence_ms: float = 0.0

    def observe(self, timestamp_us: int, monotonic_s: float) -> bool:
        timestamp_us = int(timestamp_us)
        if timestamp_us <= 0:
            return False
        previous = self.last_timestamp_us
        if previous is None:
            self.first_timestamp_us = timestamp_us
            self.last_timestamp_us = timestamp_us
            self.last_change_monotonic_s = monotonic_s
            return True
        if timestamp_us < previous:
            self.regressions += 1
            return False
        if timestamp_us == previous:
            if self.last_change_monotonic_s is not None:
                self.maximum_silence_ms = max(
                    self.maximum_silence_ms,
                    (monotonic_s - self.last_change_monotonic_s) * 1_000.0,
                )
            return False
        self.intervals_ms.append((timestamp_us - previous) * 1.0e-3)
        self.last_timestamp_us = timestamp_us
        self.last_change_monotonic_s = monotonic_s
        return True

    def sample_silence(self, monotonic_s: float) -> None:
        if self.last_change_monotonic_s is not None:
            self.maximum_silence_ms = max(
                self.maximum_silence_ms,
                (monotonic_s - self.last_change_monotonic_s) * 1_000.0,
            )

    def summary(self) -> dict[str, float | int | None]:
        values = np.asarray(self.intervals_ms, dtype=np.float64)
        elapsed_s = (
            (self.last_timestamp_us - self.first_timestamp_us) * 1.0e-6
            if self.last_timestamp_us is not None and self.first_timestamp_us is not None
            else 0.0
        )
        return {
            "samples": len(self.intervals_ms) + (1 if self.last_timestamp_us is not None else 0),
            "frequency_hz": len(self.intervals_ms) / elapsed_s if elapsed_s > 0.0 else None,
            "mean_gap_ms": float(np.mean(values)) if values.size else None,
            "p95_gap_ms": _percentile(values, 95.0),
            "p99_gap_ms": _percentile(values, 99.0),
            "maximum_gap_ms": float(np.max(values)) if values.size else None,
            "maximum_silence_ms": self.maximum_silence_ms,
            "timestamp_regressions": self.regressions,
        }


@dataclass
class ArmPhaseMetrics:
    output_timestamps: IntervalTracker = field(default_factory=IntervalTracker)
    raw_streams: dict[str, IntervalTracker] = field(default_factory=dict)
    read_latency_ms: list[float] = field(default_factory=list)
    invalid_states: int = 0
    total_states: int = 0
    maximum_abs_q_step_rad: np.ndarray = field(default_factory=lambda: np.zeros(7))
    maximum_abs_reported_dq_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(7))
    maximum_abs_reported_ddq_rad_s2: np.ndarray = field(default_factory=lambda: np.zeros(7))
    maximum_abs_fd_dq_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(7))
    maximum_abs_fd_ddq_rad_s2: np.ndarray = field(default_factory=lambda: np.zeros(7))
    _previous_q: np.ndarray | None = None
    _previous_q_timestamp_us: int | None = None
    _previous_fd_dq: np.ndarray | None = None

    def observe_state(
        self,
        state: ArmState,
        *,
        monotonic_s: float,
        read_latency_ms: float,
        derivatives_enabled: bool,
    ) -> None:
        self.total_states += 1
        self.read_latency_ms.append(float(read_latency_ms))
        vectors = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in (
            state.q,
            state.dq,
            state.ddq,
            state.torque,
            state.current,
        ))
        if any(value.shape != (7,) or not np.isfinite(value).all() for value in vectors):
            self.invalid_states += 1
            return
        q, dq, ddq, _, _ = vectors
        timestamp_us = int(state.q_timestamp_us or state.timestamp_us)
        is_new = self.output_timestamps.observe(timestamp_us, monotonic_s)
        if derivatives_enabled:
            self.maximum_abs_reported_dq_rad_s = np.maximum(
                self.maximum_abs_reported_dq_rad_s, np.abs(dq)
            )
            self.maximum_abs_reported_ddq_rad_s2 = np.maximum(
                self.maximum_abs_reported_ddq_rad_s2, np.abs(ddq)
            )
        if not is_new:
            return
        previous_q = self._previous_q
        previous_timestamp_us = self._previous_q_timestamp_us
        if previous_q is not None and previous_timestamp_us is not None:
            dt_s = (timestamp_us - previous_timestamp_us) * 1.0e-6
            if dt_s > 0.0:
                q_step = q - previous_q
                fd_dq = q_step / dt_s
                if derivatives_enabled:
                    self.maximum_abs_q_step_rad = np.maximum(
                        self.maximum_abs_q_step_rad, np.abs(q_step)
                    )
                    self.maximum_abs_fd_dq_rad_s = np.maximum(
                        self.maximum_abs_fd_dq_rad_s, np.abs(fd_dq)
                    )
                    if self._previous_fd_dq is not None:
                        fd_ddq = (fd_dq - self._previous_fd_dq) / dt_s
                        self.maximum_abs_fd_ddq_rad_s2 = np.maximum(
                            self.maximum_abs_fd_ddq_rad_s2, np.abs(fd_ddq)
                        )
                self._previous_fd_dq = fd_dq
        self._previous_q = q.copy()
        self._previous_q_timestamp_us = timestamp_us

    def observe_raw_streams(self, timestamps: dict[str, int], monotonic_s: float) -> None:
        for name, timestamp_us in timestamps.items():
            self.raw_streams.setdefault(name, IntervalTracker()).observe(
                int(timestamp_us), monotonic_s
            )
        for tracker in self.raw_streams.values():
            tracker.sample_silence(monotonic_s)

    def summary(self) -> dict[str, Any]:
        latency = np.asarray(self.read_latency_ms, dtype=np.float64)
        return {
            "total_state_reads": self.total_states,
            "invalid_state_reads": self.invalid_states,
            "valid_state_ratio": (
                (self.total_states - self.invalid_states) / self.total_states
                if self.total_states
                else 0.0
            ),
            "output": self.output_timestamps.summary(),
            "raw_streams": {
                name: tracker.summary() for name, tracker in sorted(self.raw_streams.items())
            },
            "read_latency_ms": _array_summary(latency),
            "maximum_abs_q_step_rad": self.maximum_abs_q_step_rad.tolist(),
            "maximum_abs_reported_dq_rad_s": self.maximum_abs_reported_dq_rad_s.tolist(),
            "maximum_abs_reported_ddq_rad_s2": self.maximum_abs_reported_ddq_rad_s2.tolist(),
            "maximum_abs_fd_dq_rad_s": self.maximum_abs_fd_dq_rad_s.tolist(),
            "maximum_abs_fd_ddq_rad_s2": self.maximum_abs_fd_ddq_rad_s2.tolist(),
        }


@dataclass
class PhaseMetrics:
    name: str
    duration_s: float = 0.0
    loop_intervals_ms: list[float] = field(default_factory=list)
    deadline_misses: int = 0
    arms: dict[str, ArmPhaseMetrics] = field(default_factory=dict)
    cameras: dict[str, IntervalTracker] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_s": self.duration_s,
            "loop": {
                **_array_summary(np.asarray(self.loop_intervals_ms, dtype=np.float64)),
                "deadline_misses": self.deadline_misses,
            },
            "arms": {name: metrics.summary() for name, metrics in sorted(self.arms.items())},
            "cameras": {
                name: tracker.summary() for name, tracker in sorted(self.cameras.items())
            },
        }


@dataclass(frozen=True)
class Thresholds:
    poll_rate_hz: float
    maximum_can_gap_ms: float
    minimum_output_rate_ratio: float
    minimum_camera_rate_ratio: float
    maximum_loop_gap_ms: float
    maximum_abs_dq_rad_s: float
    maximum_abs_ddq_rad_s2: float
    maximum_p99_gap_growth: float
    p99_growth_margin_ms: float


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.camera_fps is not None:
        config = replace(
            config,
            cameras=tuple(
                replace(camera, fps=args.camera_fps, visualize=False)
                for camera in config.cameras
            ),
        )
    endpoints = _select_endpoints(config, args.pair, args.role)
    camera_manager = CameraManager.from_config(config.cameras)
    if len(camera_manager.cameras) < args.minimum_cameras:
        configured = [camera.name for camera in camera_manager.cameras]
        raise RuntimeError(
            f"multi-camera test requires at least {args.minimum_cameras} usable enabled cameras; "
            f"created={configured}. Commented or enabled=false entries are not tested."
        )

    arms = {
        f"{endpoint.name}@{endpoint.channel}": build_arm(
            endpoint, args.arm_backend or config.teleop.backend
        )
        for endpoint in endpoints
    }
    print("READ-ONLY TEST: connect/configure/read/disconnect only.")
    print("No enable, disable, mode switch, gripper, move_js, move_mit, or other control call is made.")
    print("Stop every other arm controller first; this script does not cancel a previously active command.")
    print("Keep every monitored arm stationary and keep the E-stop accessible.")
    print(f"arms={list(arms)} cameras={[camera.name for camera in camera_manager.cameras]}")

    thresholds = Thresholds(
        poll_rate_hz=args.poll_rate_hz,
        maximum_can_gap_ms=args.maximum_can_gap_ms,
        minimum_output_rate_ratio=args.minimum_output_rate_ratio,
        minimum_camera_rate_ratio=args.minimum_camera_rate_ratio,
        maximum_loop_gap_ms=args.maximum_loop_gap_ms,
        maximum_abs_dq_rad_s=args.maximum_abs_dq_rad_s,
        maximum_abs_ddq_rad_s2=args.maximum_abs_ddq_rad_s2,
        maximum_p99_gap_growth=args.maximum_p99_gap_growth,
        p99_growth_margin_ms=args.p99_growth_margin_ms,
    )

    connected: list[ArmInterface] = []
    cameras_started = False
    phases: dict[str, PhaseMetrics] = {}
    try:
        for arm in arms.values():
            arm.connect()
            connected.append(arm)
            _configure_read_only_alignment(arm, config)
        _wait_for_valid_states(arms, args.arm_warmup_timeout_s)

        phases["can_only"] = _run_phase(
            "can_only",
            args.baseline_duration_s,
            arms,
            camera_manager=None,
            poll_rate_hz=args.poll_rate_hz,
            derivative_warmup_s=args.derivative_warmup_s,
            report_interval_s=args.report_interval_s,
        )

        startup_result: dict[str, BaseException] = {}
        startup_done = threading.Event()

        def start_cameras() -> None:
            try:
                camera_manager.start()
            except BaseException as exc:  # propagated in the main thread after cleanup
                startup_result["error"] = exc
            finally:
                startup_done.set()

        startup_thread = threading.Thread(
            target=start_cameras,
            name="camera-startup-monitor",
            daemon=True,
        )
        startup_thread.start()
        phases["camera_startup"] = _run_until(
            "camera_startup",
            startup_done.is_set,
            arms,
            poll_rate_hz=args.poll_rate_hz,
            timeout_s=args.camera_startup_monitor_timeout_s,
            report_interval_s=args.report_interval_s,
        )
        startup_thread.join(timeout=1.0)
        if startup_thread.is_alive():
            raise RuntimeError("camera startup thread did not stop after the monitoring timeout")
        if "error" in startup_result:
            raise RuntimeError(f"camera startup failed: {startup_result['error']}") from startup_result["error"]
        cameras_started = True

        phases["can_with_cameras"] = _run_phase(
            "can_with_cameras",
            args.duration_s,
            arms,
            camera_manager=camera_manager,
            poll_rate_hz=args.poll_rate_hz,
            derivative_warmup_s=args.derivative_warmup_s,
            report_interval_s=args.report_interval_s,
        )
    finally:
        if cameras_started or camera_manager.cameras:
            camera_manager.stop()
        for arm in reversed(connected):
            try:
                arm.disconnect()
            except Exception as exc:
                log.warning("arm disconnect failed: %s", exc)

    summaries = {name: phase.summary() for name, phase in phases.items()}
    findings = evaluate_findings(
        summaries,
        camera_fps={camera.name: float(camera.config.fps) for camera in camera_manager.cameras},
        thresholds=thresholds,
    )
    _print_report(summaries, findings)
    report = {
        "read_only": True,
        "config": str(Path(args.config).expanduser().resolve()),
        "arms": list(arms),
        "cameras": [camera.name for camera in camera_manager.cameras],
        "thresholds": asdict(thresholds),
        "phases": summaries,
        "findings": findings,
        "passed": not findings,
    }
    if args.json_output is not None:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON report: {output}")
    print("VERDICT:", "PASS" if not findings else "FAIL")
    return 0 if not findings else 2


def _configure_read_only_alignment(arm: ArmInterface, config: CollectionConfig) -> None:
    q_state = config.robot_states["q"]
    dq_state = config.robot_states["velocity"]
    ddq_state = config.robot_states["acceleration"]
    command = config.teleop.command
    arm.configure_state_alignment(
        command.state_alignment_delay_s,
        command.sample_rate_hz,
        q_state.mean_window,
        q_state.lowpass_cutoff_hz if q_state.lowpass else None,
        dq_state.lowpass_cutoff_hz if dq_state.lowpass else None,
        ddq_state.lowpass_cutoff_hz if ddq_state.lowpass else None,
        command.maximum_can_frame_gap_s,
    )


def _wait_for_valid_states(arms: dict[str, ArmInterface], timeout_s: float) -> None:
    pending = set(arms)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for name in tuple(pending):
            state = arms[name].read_state()
            values = (state.q, state.dq, state.ddq, state.torque, state.current)
            if all(
                np.asarray(value).shape == (7,) and np.isfinite(value).all()
                for value in values
            ):
                pending.remove(name)
        if pending:
            time.sleep(0.005)
    if pending:
        raise RuntimeError(
            f"timed out waiting for valid read-only aligned state from {sorted(pending)}; "
            "verify the arms are already in the expected follower/normal feedback mode"
        )


def _run_phase(
    name: str,
    duration_s: float,
    arms: dict[str, ArmInterface],
    *,
    camera_manager: CameraManager | None,
    poll_rate_hz: float,
    derivative_warmup_s: float,
    report_interval_s: float,
) -> PhaseMetrics:
    started_s = time.monotonic()
    return _sample_loop(
        name,
        lambda: time.monotonic() - started_s >= duration_s,
        arms,
        camera_manager=camera_manager,
        poll_rate_hz=poll_rate_hz,
        derivative_warmup_s=derivative_warmup_s,
        report_interval_s=report_interval_s,
        timeout_s=duration_s + max(5.0, duration_s * 0.1),
    )


def _run_until(
    name: str,
    done: Callable[[], bool],
    arms: dict[str, ArmInterface],
    *,
    poll_rate_hz: float,
    timeout_s: float,
    report_interval_s: float,
) -> PhaseMetrics:
    return _sample_loop(
        name,
        done,
        arms,
        camera_manager=None,
        poll_rate_hz=poll_rate_hz,
        derivative_warmup_s=0.0,
        report_interval_s=report_interval_s,
        timeout_s=timeout_s,
    )


def _sample_loop(
    name: str,
    done: Callable[[], bool],
    arms: dict[str, ArmInterface],
    *,
    camera_manager: CameraManager | None,
    poll_rate_hz: float,
    derivative_warmup_s: float,
    report_interval_s: float,
    timeout_s: float,
) -> PhaseMetrics:
    metrics = PhaseMetrics(
        name=name,
        arms={arm_name: ArmPhaseMetrics() for arm_name in arms},
    )
    period_s = 1.0 / poll_rate_hz
    started_s = time.monotonic()
    previous_loop_s: float | None = None
    next_deadline_s = started_s
    next_report_s = started_s + report_interval_s
    first_iteration = True
    while first_iteration or not done():
        first_iteration = False
        loop_s = time.monotonic()
        if loop_s - started_s > timeout_s:
            raise RuntimeError(f"phase {name!r} exceeded timeout {timeout_s:.1f}s")
        if previous_loop_s is not None:
            metrics.loop_intervals_ms.append((loop_s - previous_loop_s) * 1_000.0)
        previous_loop_s = loop_s
        derivatives_enabled = loop_s - started_s >= derivative_warmup_s

        for arm_name, arm in arms.items():
            read_started_s = time.monotonic()
            state = arm.read_state()
            read_finished_s = time.monotonic()
            arm_metrics = metrics.arms[arm_name]
            arm_metrics.observe_state(
                state,
                monotonic_s=read_finished_s,
                read_latency_ms=(read_finished_s - read_started_s) * 1_000.0,
                derivatives_enabled=derivatives_enabled,
            )
            arm_metrics.observe_raw_streams(
                _raw_stream_timestamp_snapshot(arm), read_finished_s
            )

        if camera_manager is not None:
            for frame in camera_manager.poll():
                _validate_camera_frame(frame)
                metrics.cameras.setdefault(frame.camera_name, IntervalTracker()).observe(
                    frame.timestamp_us, time.monotonic()
                )

        next_deadline_s += period_s
        remaining_s = next_deadline_s - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        else:
            metrics.deadline_misses += 1
            if remaining_s < -period_s:
                next_deadline_s = time.monotonic()
        if time.monotonic() >= next_report_s:
            _print_live_status(metrics, time.monotonic() - started_s)
            next_report_s += report_interval_s
    metrics.duration_s = time.monotonic() - started_s
    for arm_metrics in metrics.arms.values():
        for tracker in arm_metrics.raw_streams.values():
            tracker.sample_silence(time.monotonic())
    return metrics


def _raw_stream_timestamp_snapshot(arm: ArmInterface) -> dict[str, int]:
    # The adapter sampler and this diagnostic intentionally observe the same parser-derived
    # timestamps. Reading them avoids adding another set of SDK getters to the CAN workload.
    timestamps = getattr(arm, "_state_capture_timestamps_us", None)
    if timestamps is None:
        state = arm.read_state()
        result = {
            f"q_joint_{joint + 1}": int(value)
            for joint, value in enumerate(np.asarray(state.q_component_timestamp_us).reshape(-1))
        }
        result.update(
            {
                f"motor_state_{joint + 1}": int(value)
                for joint, value in enumerate(np.asarray(state.motor_timestamp_us).reshape(-1))
            }
        )
        return result
    for _ in range(3):
        try:
            return {str(name): int(value) for name, value in dict(timestamps).items()}
        except RuntimeError:
            time.sleep(0)
    raise RuntimeError(f"could not snapshot raw CAN timestamps for arm {arm.name}")


def _validate_camera_frame(frame: CameraFrame) -> None:
    values = np.asarray(frame.frame)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError(
            f"camera {frame.camera_name} returned invalid frame "
            f"shape={values.shape} dtype={values.dtype}"
        )


def evaluate_findings(
    phases: dict[str, dict[str, Any]],
    *,
    camera_fps: dict[str, float],
    thresholds: Thresholds,
) -> list[str]:
    findings: list[str] = []
    baseline = phases.get("can_only", {})
    loaded = phases.get("can_with_cameras", {})
    startup = phases.get("camera_startup", {})
    for phase_name, phase in phases.items():
        loop_max = _number(phase.get("loop", {}).get("maximum"))
        if loop_max is not None and loop_max > thresholds.maximum_loop_gap_ms:
            findings.append(
                f"{phase_name}: host loop maximum gap {loop_max:.3f} ms exceeds "
                f"{thresholds.maximum_loop_gap_ms:.3f} ms"
            )
        for arm_name, arm in phase.get("arms", {}).items():
            if arm.get("invalid_state_reads", 0):
                findings.append(
                    f"{phase_name}/{arm_name}: {arm['invalid_state_reads']} invalid aligned states"
                )
            output_hz = _number(arm.get("output", {}).get("frequency_hz"))
            minimum_hz = thresholds.poll_rate_hz * thresholds.minimum_output_rate_ratio
            phase_is_long_enough = float(phase.get("duration_s", 0.0)) >= 0.5
            if phase_is_long_enough and (output_hz is None or output_hz < minimum_hz):
                findings.append(
                    f"{phase_name}/{arm_name}: aligned q rate {output_hz} Hz below {minimum_hz:.2f} Hz"
                )
            raw_streams = arm.get("raw_streams", {})
            if not raw_streams:
                findings.append(f"{phase_name}/{arm_name}: no raw CAN stream timestamps observed")
            for stream_name, stream in raw_streams.items():
                maximum_gap = _number(stream.get("maximum_gap_ms"))
                maximum_silence = _number(stream.get("maximum_silence_ms"))
                regressions = int(stream.get("timestamp_regressions", 0))
                if maximum_gap is not None and maximum_gap > thresholds.maximum_can_gap_ms:
                    findings.append(
                        f"{phase_name}/{arm_name}/{stream_name}: CAN gap {maximum_gap:.3f} ms "
                        f"exceeds {thresholds.maximum_can_gap_ms:.3f} ms"
                    )
                if maximum_silence is not None and maximum_silence > thresholds.maximum_can_gap_ms:
                    findings.append(
                        f"{phase_name}/{arm_name}/{stream_name}: host observed no update for "
                        f"{maximum_silence:.3f} ms"
                    )
                if regressions:
                    findings.append(
                        f"{phase_name}/{arm_name}/{stream_name}: {regressions} timestamp regressions"
                    )
            if phase_name == "can_with_cameras":
                dq = np.asarray(arm.get("maximum_abs_reported_dq_rad_s", []), dtype=np.float64)
                ddq = np.asarray(arm.get("maximum_abs_reported_ddq_rad_s2", []), dtype=np.float64)
                if dq.size and np.max(dq) > thresholds.maximum_abs_dq_rad_s:
                    findings.append(
                        f"{phase_name}/{arm_name}: reported |dq| max {np.max(dq):.3f} rad/s exceeds "
                        f"stationary limit {thresholds.maximum_abs_dq_rad_s:.3f}"
                    )
                if ddq.size and np.max(ddq) > thresholds.maximum_abs_ddq_rad_s2:
                    findings.append(
                        f"{phase_name}/{arm_name}: reported |ddq| max {np.max(ddq):.3f} rad/s^2 exceeds "
                        f"stationary limit {thresholds.maximum_abs_ddq_rad_s2:.3f}"
                    )

    for camera_name, requested_hz in camera_fps.items():
        camera = loaded.get("cameras", {}).get(camera_name)
        measured_hz = _number(camera.get("frequency_hz")) if camera else None
        minimum_hz = requested_hz * thresholds.minimum_camera_rate_ratio
        if measured_hz is None or measured_hz < minimum_hz:
            findings.append(
                f"can_with_cameras/{camera_name}: camera rate {measured_hz} Hz below {minimum_hz:.2f} Hz "
                f"({thresholds.minimum_camera_rate_ratio:.0%} of requested {requested_hz:.2f} Hz)"
            )

    for arm_name, loaded_arm in loaded.get("arms", {}).items():
        baseline_arm = baseline.get("arms", {}).get(arm_name, {})
        for stream_name, loaded_stream in loaded_arm.get("raw_streams", {}).items():
            baseline_stream = baseline_arm.get("raw_streams", {}).get(stream_name)
            if baseline_stream is None:
                continue
            base_p99 = _number(baseline_stream.get("p99_gap_ms"))
            load_p99 = _number(loaded_stream.get("p99_gap_ms"))
            if base_p99 is None or load_p99 is None:
                continue
            allowed = max(
                base_p99 * thresholds.maximum_p99_gap_growth,
                base_p99 + thresholds.p99_growth_margin_ms,
            )
            if load_p99 > allowed:
                findings.append(
                    f"can_with_cameras/{arm_name}/{stream_name}: p99 CAN gap grew from "
                    f"{base_p99:.3f} to {load_p99:.3f} ms (allowed {allowed:.3f} ms)"
                )
    if startup and startup.get("duration_s", 0.0) <= 0.0:
        findings.append("camera_startup: no CAN samples were collected during camera startup")
    return findings


def _select_endpoints(
    config: CollectionConfig,
    pair_names: list[str] | None,
    role: str,
) -> list[ArmEndpointConfig]:
    selected_pairs = [
        pair
        for pair in config.teleop.master_slave
        if not pair_names or pair.name in pair_names
    ]
    missing = set(pair_names or ()) - {pair.name for pair in selected_pairs}
    if missing:
        raise RuntimeError(f"unknown arm pair(s): {sorted(missing)}")
    endpoints: list[ArmEndpointConfig] = []
    for pair in selected_pairs:
        if role in {"leader", "both"}:
            endpoints.append(pair.leader)
        if role in {"follower", "both"}:
            endpoints.append(pair.follower)
    if not endpoints:
        raise RuntimeError("no arm endpoints selected")
    return endpoints


def _print_live_status(metrics: PhaseMetrics, elapsed_s: float) -> None:
    arm_parts = []
    for name, arm in metrics.arms.items():
        output = arm.output_timestamps.summary()
        raw_max = max(
            (
                _number(tracker.summary()["maximum_gap_ms"]) or 0.0
                for tracker in arm.raw_streams.values()
            ),
            default=0.0,
        )
        arm_parts.append(
            f"{name}:q={_fmt_number(output['frequency_hz'])}Hz raw_max={raw_max:.2f}ms "
            f"invalid={arm.invalid_states}"
        )
    print(f"[{metrics.name} {elapsed_s:6.1f}s] " + " | ".join(arm_parts), flush=True)


def _print_report(phases: dict[str, dict[str, Any]], findings: list[str]) -> None:
    for phase_name, phase in phases.items():
        print(f"\n=== {phase_name} ({phase['duration_s']:.2f}s) ===")
        loop = phase["loop"]
        print(
            "loop: "
            f"mean={_fmt_number(loop.get('mean'))} ms "
            f"p99={_fmt_number(loop.get('p99'))} ms "
            f"max={_fmt_number(loop.get('maximum'))} ms "
            f"misses={loop.get('deadline_misses', 0)}"
        )
        for arm_name, arm in phase["arms"].items():
            output = arm["output"]
            print(
                f"arm {arm_name}: q={_fmt_number(output.get('frequency_hz'))} Hz "
                f"q_gap_p99={_fmt_number(output.get('p99_gap_ms'))} ms "
                f"q_gap_max={_fmt_number(output.get('maximum_gap_ms'))} ms "
                f"valid={arm['valid_state_ratio']:.3%}"
            )
            print(
                "  maxima by joint: "
                f"|dq|={_fmt_vector(arm['maximum_abs_reported_dq_rad_s'])} rad/s "
                f"|ddq|={_fmt_vector(arm['maximum_abs_reported_ddq_rad_s2'])} rad/s^2"
            )
            for stream_name, stream in arm["raw_streams"].items():
                print(
                    f"  {stream_name:20s} "
                    f"rate={_fmt_number(stream.get('frequency_hz')):>7s} Hz "
                    f"p99={_fmt_number(stream.get('p99_gap_ms')):>7s} ms "
                    f"max={_fmt_number(stream.get('maximum_gap_ms')):>7s} ms "
                    f"silence={_fmt_number(stream.get('maximum_silence_ms')):>7s} ms"
                )
        for camera_name, camera in phase["cameras"].items():
            print(
                f"camera {camera_name}: rate={_fmt_number(camera.get('frequency_hz'))} Hz "
                f"p99={_fmt_number(camera.get('p99_gap_ms'))} ms "
                f"max={_fmt_number(camera.get('maximum_gap_ms'))} ms"
            )
    if findings:
        print("\nFindings:")
        for finding in findings:
            print(f"- {finding}")


def _array_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean": float(np.mean(values)) if values.size else None,
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "maximum": float(np.max(values)) if values.size else None,
    }


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _fmt_number(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.3f}"


def _fmt_vector(values: Any) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ",".join(f"{value:.3f}" for value in array) + "]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only baseline-versus-multi-camera stress test for Nero CAN q/dq/ddq timing. "
            "The script never enables, changes mode, or sends motion/gripper commands."
        )
    )
    parser.add_argument("--config", default="configs/master_slave_can.yaml")
    parser.add_argument("--pair", action="append", help="Arm pair to monitor; repeatable, default: all")
    parser.add_argument("--role", choices=("leader", "follower", "both"), default="both")
    parser.add_argument("--arm-backend", choices=("pyagxarm", "mock"))
    parser.add_argument("--baseline-duration-s", type=float, default=15.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--poll-rate-hz", type=float, default=100.0)
    parser.add_argument("--minimum-cameras", type=int, default=2)
    parser.add_argument(
        "--camera-fps",
        type=float,
        help="Override every enabled camera FPS in memory; does not modify the YAML file",
    )
    parser.add_argument("--arm-warmup-timeout-s", type=float, default=5.0)
    parser.add_argument("--camera-startup-monitor-timeout-s", type=float, default=60.0)
    parser.add_argument("--derivative-warmup-s", type=float, default=2.0)
    parser.add_argument("--report-interval-s", type=float, default=5.0)
    parser.add_argument("--maximum-can-gap-ms", type=float)
    parser.add_argument("--maximum-loop-gap-ms", type=float, default=50.0)
    parser.add_argument("--minimum-output-rate-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-camera-rate-ratio", type=float, default=0.80)
    parser.add_argument("--maximum-abs-dq-rad-s", type=float, default=1.0)
    parser.add_argument("--maximum-abs-ddq-rad-s2", type=float, default=20.0)
    parser.add_argument("--maximum-p99-gap-growth", type=float, default=1.5)
    parser.add_argument("--p99-growth-margin-ms", type=float, default=2.0)
    parser.add_argument("--json-output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    positive = (
        "baseline_duration_s",
        "duration_s",
        "poll_rate_hz",
        "arm_warmup_timeout_s",
        "camera_startup_monitor_timeout_s",
        "report_interval_s",
        "maximum_loop_gap_ms",
        "maximum_abs_dq_rad_s",
        "maximum_abs_ddq_rad_s2",
        "maximum_p99_gap_growth",
        "p99_growth_margin_ms",
    )
    for name in positive:
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.derivative_warmup_s < 0.0:
        parser.error("--derivative-warmup-s must be non-negative")
    if args.minimum_cameras < 1:
        parser.error("--minimum-cameras must be positive")
    if args.camera_fps is not None and args.camera_fps <= 0.0:
        parser.error("--camera-fps must be positive")
    for name in ("minimum_output_rate_ratio", "minimum_camera_rate_ratio"):
        if not 0.0 < getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within (0, 1]")
    config = load_config(args.config)
    configured_gap_ms = config.teleop.command.maximum_can_frame_gap_s * 1_000.0
    args.maximum_can_gap_ms = (
        configured_gap_ms if args.maximum_can_gap_ms is None else args.maximum_can_gap_ms
    )
    if args.maximum_can_gap_ms <= 0.0:
        parser.error("--maximum-can-gap-ms must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
