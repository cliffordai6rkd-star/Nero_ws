#!/usr/bin/env python3
"""Recover SocketCAN feedback for Nero without resetting or moving the arm.

The command only reconfigures the host CAN interfaces and performs read-only
feedback checks.  It never calls reset(), enable(), disable(), move_joints(),
or any mode-switch command on the arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.arms.factory import build_arm
from nero_collection.config import ArmEndpointConfig
from nero_collection.socketcan import (
    capture_frames,
    configure_interface,
    interface_exists,
    link_details,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    endpoint = _endpoint(args)
    interfaces = tuple(dict.fromkeys((*args.interfaces, endpoint.channel)))

    print("Reconfiguring host CAN interfaces; the arm will not be reset or moved.", flush=True)
    for interface in interfaces:
        if not interface_exists(interface):
            print(f"ERROR: {interface} is not present. Reconnect the USB-CAN adapter first.", flush=True)
            return 2
        print(f"Configuring {interface} bitrate={endpoint.bitrate}", flush=True)
        configure_interface(interface, endpoint.bitrate)
        print(link_details(interface, maximum_lines=6), flush=True)

    if shutil.which("candump") is None:
        print("ERROR: candump is not installed; install can-utils before checking feedback.", flush=True)
        return 2

    for interface in interfaces:
        print(f"Checking CAN frames on {interface} for {args.duration:g}s", flush=True)
        frames = capture_frames(interface, args.duration)
        if frames:
            print(f"  received {len(frames)} frame(s)", flush=True)
        else:
            level = "ERROR" if interface == endpoint.channel else "WARN"
            print(f"  {level}: no frames received on {interface}", flush=True)
            if interface == endpoint.channel:
                print("  Check arm power, CANH/CANL, termination, and USB-CAN mapping.", flush=True)
                return 3

    return _check_arm_feedback(endpoint, args)


def _check_arm_feedback(endpoint: ArmEndpointConfig, args: argparse.Namespace) -> int:
    print(f"Opening read-only feedback connection on {endpoint.channel}", flush=True)
    arm = build_arm(endpoint, "pyagxarm")
    try:
        arm.connect()
        deadline = time.monotonic() + args.timeout
        last_error = "no valid feedback yet"
        while time.monotonic() < deadline:
            try:
                state = arm.read_state()
                q = np.asarray(state.q)
                q_stamps = np.asarray(state.q_component_timestamp_us)
                motor_stamps = np.asarray(state.motor_timestamp_us)
                age_q = (
                    time.time() - float(q_stamps.min()) / 1e6
                    if q_stamps.shape == (7,) and q_stamps.min() > 0
                    else float("inf")
                )
                age_motor = (
                    time.time() - float(motor_stamps.min()) / 1e6
                    if motor_stamps.shape == (7,) and motor_stamps.min() > 0
                    else float("inf")
                )
                if (
                    q.shape == (7,)
                    and q_stamps.shape == (7,)
                    and motor_stamps.shape == (7,)
                    and q_stamps.min() > 0
                    and motor_stamps.min() > 0
                    and bool(np.isfinite(q).all())
                    and max(age_q, age_motor) <= args.max_age
                ):
                    print(
                        "Feedback recovered: "
                        f"q_age={age_q:.3f}s motor_age={age_motor:.3f}s q={q.tolist()}",
                        flush=True,
                    )
                    return 0
                last_error = (
                    f"stale/invalid feedback q_age={age_q:.3f}s motor_age={age_motor:.3f}s "
                    f"q_stamps={q_stamps.tolist()} motor_stamps={motor_stamps.tolist()}"
                )
            except Exception as exc:  # SDK may briefly have an empty cache after CAN restart.
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(args.poll)
        print(f"ERROR: feedback did not recover within {args.timeout:g}s: {last_error}", flush=True)
        return 4
    finally:
        arm.disconnect()


def _endpoint(args: argparse.Namespace) -> ArmEndpointConfig:
    values: dict[str, Any] = {
        "name": args.name,
        "channel": args.channel,
        "interface": args.interface,
        "bitrate": args.bitrate,
        "firmware": args.firmware,
    }
    if args.config is not None:
        import yaml

        data = yaml.safe_load(Path(args.config).read_text()) or {}
        raw = (data.get("hardware") or {}).get("endpoint")
        if raw is None:
            raise ValueError(f"{args.config} has no hardware.endpoint")
        for key in ("name", "can_id", "channel", "usb_serial", "interface", "bitrate", "firmware", "rest_q", "config_kwargs"):
            if key in raw:
                values[key] = raw[key]
    return ArmEndpointConfig(**values)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="pi0_wm YAML; reads hardware.endpoint")
    parser.add_argument("--interfaces", nargs="+", default=["can_master", "can_slave"])
    parser.add_argument("--channel", default="can_slave")
    parser.add_argument("--name", default="follower")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--firmware", default="V120")
    parser.add_argument("--duration", type=float, default=1.0, help="candump duration per interface")
    parser.add_argument("--timeout", type=float, default=5.0, help="feedback recovery timeout")
    parser.add_argument("--poll", type=float, default=0.05, help="feedback polling interval")
    parser.add_argument("--max-age", type=float, default=0.1, help="maximum accepted feedback age in seconds")
    args = parser.parse_args(argv)
    if args.bitrate <= 0 or args.duration <= 0 or args.timeout <= 0 or args.poll <= 0 or args.max_age <= 0:
        parser.error("bitrate, duration, timeout and max-age must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
