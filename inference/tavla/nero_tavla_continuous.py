#!/usr/bin/env python3
"""Run persistent asynchronous TA-VLA inference for an AgileX Nero.

Torque history is sampled at 25 Hz, inference runs in one worker, and each
complete 50-step action chunk is consumed at 20 Hz by default.  The next chunk
is prefetched while the current chunk is still being consumed.  The default is
a motion-disabled dry run; pass ``--motion`` to enable and command the robot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import signal
import threading
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
    status_and_enable,
)
from .tavla_client import ACTION_HORIZON, TavlaRemotePolicy


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
    parser.add_argument("--speed-percent", type=int, default=100)
    parser.add_argument(
        "--action-rate",
        type=float,
        default=20.0,
        help="action-chunk consumption and joint-target rate in Hz",
    )
    parser.add_argument(
        "--prefetch-actions",
        type=int,
        default=20,
        help="request the next 50-step chunk at or below this queue depth",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="0 runs until Ctrl-C; a positive value stops after this many targets",
    )
    parser.add_argument(
        "--motion",
        action="store_true",
        help="enable the arm and send targets; omission keeps a read-only dry run",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("runs/tavla/nero_tavla_continuous.log"),
    )
    return parser.parse_args(argv)


def _validate_args(cfg: argparse.Namespace) -> None:
    if not 1 <= cfg.speed_percent <= 100:
        raise ValueError("speed-percent must be in [1, 100]")
    if not 1.0 <= cfg.action_rate <= 50.0:
        raise ValueError("action-rate must be in [1, 50] Hz")
    if not 0 <= cfg.prefetch_actions < ACTION_HORIZON:
        raise ValueError(
            f"prefetch-actions must be in [0, {ACTION_HORIZON - 1}]"
        )
    if not 0.0 < cfg.max_joint_step <= 0.52:
        raise ValueError("max-joint-step must be in (0, 0.52] rad")
    if cfg.max_actions < 0:
        raise ValueError("max-actions must be non-negative")


def _is_normal(status: Any, enabled: list[bool]) -> bool:
    return (
        getattr(status.msg, "arm_status", None) == 0
        and len(enabled) == 7
        and all(enabled)
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    _validate_args(cfg)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    cfg.log.parent.mkdir(parents=True, exist_ok=True)

    side: CameraStream | None = None
    wrist: CameraStream | None = None
    robot: Any | None = None
    log_file = cfg.log.open("a", encoding="utf-8")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tavla-inference")
    pending: Future[np.ndarray] | None = None
    action_queue: list[np.ndarray] = []
    fault_latched = False
    action_count = 0
    last_target_sent = False
    control_period = 1.0 / 25.0
    action_period = 1.0 / cfg.action_rate
    next_control_tick = time.monotonic()
    next_action_tick = next_control_tick
    motor_feedback_ok = True

    def write_log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        print(line, file=log_file, flush=True)
        if message.startswith(("START", "STOP", "FAULT", "FINAL_TARGET_FAULT")):
            print(line, flush=True)

    try:
        side = CameraStream(cfg.side_camera, "side")
        wrist = CameraStream(cfg.wrist_camera, "wrist")
        robot = create_nero_robot(cfg.can, cfg.firmware)
        robot.connect()
        if cfg.motion:
            if robot.enable() is False:
                raise RuntimeError("failed to enable Nero")
            require_normal(robot)
            robot.set_speed_percent(cfg.speed_percent)

        policy = TavlaRemotePolicy(
            cfg.host,
            port=cfg.port,
            task=cfg.task,
            camera_color="bgr",
        )
        write_log(
            f"START mode={'MOTION' if cfg.motion else 'DRY_RUN'} "
            f"speed_percent={cfg.speed_percent} max_step={cfg.max_joint_step} "
            f"action_rate={cfg.action_rate:g}Hz prefetch={cfg.prefetch_actions} "
            f"task={cfg.task}"
        )

        while not stop.is_set():
            now = time.monotonic()
            next_deadline = min(next_control_tick, next_action_tick)
            if now < next_deadline:
                time.sleep(next_deadline - now)
            now = time.monotonic()
            control_due = now >= next_control_tick
            action_due = now >= next_action_tick
            if control_due:
                while next_control_tick <= now:
                    next_control_tick += control_period
            if action_due:
                while next_action_tick <= now:
                    next_action_tick += action_period

            normal = True
            if cfg.motion and (control_due or action_due):
                status, enabled = status_and_enable(robot)
                normal = _is_normal(status, enabled)
                if not normal:
                    if not fault_latched:
                        write_log(
                            f"FAULT_LATCHED status={status.msg.arm_status} "
                            f"enabled={enabled}; no further targets will be sent"
                        )
                    fault_latched = True
                    action_queue.clear()
                    continue
                if fault_latched:
                    # A physical fault latch must never clear itself merely
                    # because later feedback happens to look normal.
                    continue

            if control_due:
                try:
                    policy.observe_effort(
                        read_joint_effort(robot),
                        timestamp=time.monotonic(),
                    )
                    motor_feedback_ok = True
                except Exception as exc:
                    write_log(f"MOTOR_FEEDBACK_ERROR {exc}; pausing target dispatch")
                    motor_feedback_ok = False

            if pending is not None and pending.done():
                try:
                    action_chunk = pending.result()
                    action_queue.extend(
                        np.asarray(action, dtype=np.float32).copy()
                        for action in action_chunk
                    )
                    write_log(
                        f"INFERENCE_READY chunk={len(action_chunk)} "
                        f"queued={len(action_queue)}"
                    )
                except Exception as exc:
                    write_log(f"INFERENCE_ERROR {exc}")
                    action_queue.clear()
                pending = None

            if action_due and normal and motor_feedback_ok and action_queue:
                try:
                    current = read_joint_state(robot)
                    target = clip_joint_target(
                        action_queue[0],
                        current,
                        cfg.max_joint_step,
                    )
                    if cfg.motion:
                        send_move_j(robot, target)
                        event = "STREAM_MOVE_J"
                        last_target_sent = True
                    else:
                        event = "DRY_RUN_TARGET"
                    action_queue.pop(0)
                    action_count += 1
                    write_log(
                        f"{event} count={action_count} "
                        f"queue_remaining={len(action_queue)} "
                        f"current={current.tolist()} target={target.tolist()}"
                    )
                    if cfg.max_actions and action_count >= cfg.max_actions:
                        stop.set()
                except Exception as exc:
                    write_log(f"FAULT_LATCHED target dispatch failed: {exc}")
                    fault_latched = True
                    action_queue.clear()

            # Snapshot the inputs in the control thread.  Only the network/model
            # request runs in the worker, so camera and torque buffers are never
            # mutated concurrently with observation construction.
            if (
                control_due
                and policy.ready
                and pending is None
                and len(action_queue) <= cfg.prefetch_actions
                and not fault_latched
                and not stop.is_set()
            ):
                try:
                    current_state = read_joint_state(robot)
                    observation = policy.make_observation(
                        side_image=side.latest(),
                        wrist_image=wrist.latest(),
                        state=current_state,
                    )
                    pending = worker.submit(
                        policy.infer_observation,
                        observation,
                        current_state=current_state.copy(),
                        max_joint_step=cfg.max_joint_step,
                    )
                    write_log(f"INFERENCE_SENT queue={len(action_queue)}")
                except Exception as exc:
                    write_log(f"OBSERVATION_ERROR {exc}")

        # Let the last issued target finish while retaining the CAN link.  No
        # new target, disable, or electronic emergency-stop command is sent.
        while cfg.motion and last_target_sent and not fault_latched:
            time.sleep(0.04)
            status, enabled = status_and_enable(robot)
            if not _is_normal(status, enabled):
                write_log(
                    f"FINAL_TARGET_FAULT status={status.msg.arm_status} "
                    f"enabled={enabled}"
                )
                break
            if getattr(status.msg, "motion_status", None) == 0:
                write_log(f"FINAL_TARGET_REACHED count={action_count}")
                break
        write_log(f"STOP actions={action_count} fault_latched={fault_latched}")
    finally:
        if pending is not None:
            pending.cancel()
        worker.shutdown(wait=False, cancel_futures=True)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                write_log(f"DISCONNECT_ERROR {exc}")
        if side is not None:
            side.close()
        if wrist is not None:
            wrist.close()
        log_file.close()


if __name__ == "__main__":
    main()
