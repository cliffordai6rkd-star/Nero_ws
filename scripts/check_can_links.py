#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.socketcan import (
    capture_frames,
    configure_interface,
    interface_exists,
    link_details,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if shutil.which("candump") is None:
        raise RuntimeError("candump not found; install it with: sudo apt-get install -y can-utils")

    for interface in args.interfaces:
        print(f"\n========== {interface} ==========", flush=True)
        if not interface_exists(interface):
            print(f"ERROR: {interface} does not exist.")
            _print_available_can_interfaces()
            continue

        print(f"[1/3] Configure {interface} bitrate={args.bitrate}", flush=True)
        configure_interface(interface, args.bitrate)
        print("[2/3] Link status", flush=True)
        print(link_details(interface))
        print(f"[3/3] candump {interface} for {args.duration:g}s", flush=True)
        frames = capture_frames(interface, args.duration)
        if frames:
            print(f"OK: received {len(frames)} frame(s) on {interface}. First frames:")
            print("\n".join(frames[:10]))
        else:
            print(f"WARN: received 0 frames on {interface}.")
            print(
                "      Check arm power, CANH/CANL, termination, USB-CAN mapping, "
                "and whether this interface is connected to the arm."
            )
    return 0


def _print_available_can_interfaces() -> None:
    print("Available CAN interfaces:")
    result = subprocess.run(
        ["ip", "-brief", "link", "show", "type", "can"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout.rstrip() or "  (none)")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure SocketCAN and verify that each interface receives frames."
    )
    parser.add_argument(
        "interfaces",
        nargs="*",
        default=["can_master", "can_slave"],
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=int(os.environ.get("CAN_BITRATE", "1000000")),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float(os.environ.get("CANDUMP_SECONDS", "3")),
    )
    args = parser.parse_args(argv)
    if args.bitrate <= 0:
        parser.error("--bitrate must be positive")
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
