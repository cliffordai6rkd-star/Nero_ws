#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.socketcan import configure_interface, link_details


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for interface in args.interfaces:
        print(f"Configuring {interface} bitrate={args.bitrate}", flush=True)
        configure_interface(interface, args.bitrate)
        details = link_details(interface, maximum_lines=6)
        if details:
            print(details)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure Nero SocketCAN interfaces.")
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
    args = parser.parse_args(argv)
    if args.bitrate <= 0:
        parser.error("--bitrate must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
