#!/usr/bin/env python3
"""Run one safety-limited TA-VLA inference on a 7-DoF AgileX Nero.

The default path captures a real observation, requests one action chunk, and
saves it without commanding the robot.  A physical command is sent only when
``--motion`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from .nero_runtime import (
    CameraStream,
    clip_joint_target,
    create_nero_robot,
    read_joint_effort,
    read_joint_state,
    require_normal,
    send_move_j,
)
from .tavla_client import TavlaRemotePolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.100.101")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--task",
        choices=("usb", "button", "cucumber"),
        default="usb",
    )
    parser.add_argument("--can", default="can1")
    parser.add_argument("--firmware", default="V112")
    parser.add_argument("--side-camera", type=int, default=4)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.02,
        help="maximum displacement from the latest state for every joint (rad)",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=5,
        help="Nero motion speed percentage; used only with --motion",
    )
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument(
        "--motion",
        action="store_true",
        help="send exactly the first live-state-clipped target to the robot",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="enable all joints before readiness checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/tavla/nero_tavla_actions.npy"),
    )
    return parser.parse_args(argv)


def capture_effort_history(
    robot: Any,
    policy: TavlaRemotePolicy,
    *,
    duration_s: float = 2.2,
    sample_rate_hz: float = 25.0,
) -> int:
    """Capture training-rate torque history without issuing a robot command."""
    if duration_s <= 2.0:
        raise ValueError("duration_s must exceed the two-second TA-VLA history")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    started = time.monotonic()
    next_tick = started
    samples = 0
    while time.monotonic() - started < duration_s:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        timestamp = time.monotonic()
        policy.observe_effort(read_joint_effort(robot), timestamp=timestamp)
        samples += 1
        next_tick += 1.0 / sample_rate_hz
    if not policy.ready:
        raise RuntimeError(
            f"25 Hz torque history is not ready after {samples} samples"
        )
    return samples


def wait_for_motion(robot: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        time.sleep(0.1)
        status = robot.get_arm_status()
        if status is None:
            raise RuntimeError("lost arm-status feedback during motion")
        arm_status = getattr(status.msg, "arm_status", None)
        if arm_status != 0:
            raise RuntimeError(f"robot stopped with arm status {arm_status}")
        if getattr(status.msg, "motion_status", None) == 0:
            return
    raise TimeoutError(f"motion did not finish within {timeout_s:.1f} seconds")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    if not 0.0 < cfg.max_joint_step <= 0.52:
        raise ValueError("max-joint-step must be in (0, 0.52] rad")
    if not 1 <= cfg.speed_percent <= 100:
        raise ValueError("speed-percent must be in [1, 100]")
    if cfg.timeout <= 0.0:
        raise ValueError("timeout must be positive")

    side_camera: CameraStream | None = None
    wrist_camera: CameraStream | None = None
    robot: Any | None = None
    try:
        side_camera = CameraStream(cfg.side_camera, "side")
        wrist_camera = CameraStream(cfg.wrist_camera, "wrist")
        robot = create_nero_robot(cfg.can, cfg.firmware)
        robot.connect()
        if cfg.enable:
            if robot.enable() is False:
                raise RuntimeError("failed to enable all Nero joints")
            time.sleep(0.3)
        require_normal(robot)

        policy = TavlaRemotePolicy(
            cfg.host,
            port=cfg.port,
            task=cfg.task,
            camera_color="bgr",
        )
        samples = capture_effort_history(robot, policy)
        require_normal(robot)

        state = read_joint_state(robot)
        side = side_camera.latest(timeout_s=3.0)
        wrist = wrist_camera.latest(timeout_s=3.0)
        inference_started = time.monotonic()
        observation = policy.make_observation(
            side_image=side,
            wrist_image=wrist,
            state=state,
        )
        actions = policy.infer_observation(
            observation,
            current_state=state,
            max_joint_step=cfg.max_joint_step,
        )
        latency_ms = (time.monotonic() - inference_started) * 1000.0
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(cfg.output, actions)

        # Reapply the safety limit against a fresh state immediately before a
        # possible command; inference latency can make the sampled state stale.
        live_state = read_joint_state(robot)
        target = clip_joint_target(actions[0], live_state, cfg.max_joint_step)
        delta = target - live_state
        print(f"inference samples={samples}, latency_ms={latency_ms:.1f}")
        print("state_rad=", live_state.tolist())
        print("target_rad=", target.tolist())
        print("delta_rad=", delta.tolist())
        print("actions_saved=", cfg.output)

        if not cfg.motion:
            print("DRY RUN: no robot command sent (pass --motion to execute one target).")
            return

        require_normal(robot)
        robot.set_speed_percent(cfg.speed_percent)
        send_move_j(robot, target)
        print(
            "MOVE_J sent once: "
            f"speed_percent={cfg.speed_percent}, max_step={cfg.max_joint_step} rad"
        )
        wait_for_motion(robot, cfg.timeout)
        print("MOTION_DONE final_state_rad=", read_joint_state(robot).tolist())
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                logging.exception("failed to disconnect Nero")
        if side_camera is not None:
            side_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()


if __name__ == "__main__":
    main()
