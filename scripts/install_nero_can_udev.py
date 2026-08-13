#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RULE = REPO_ROOT / "configs" / "udev" / "99-nero-usb-can.rules"
SYSTEM_RULE = Path("/etc/udev/rules.d/99-nero-usb-can.rules")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not SOURCE_RULE.is_file():
        raise RuntimeError(f"udev rule not found: {SOURCE_RULE}")

    subprocess.run(["udevadm", "verify", str(SOURCE_RULE)], check=True)
    if args.check:
        print(f"udev rule is valid: {SOURCE_RULE}")
        return 0

    if shutil.which("sudo") is None:
        raise RuntimeError("sudo is required to install a rule under /etc/udev/rules.d")
    subprocess.run(
        ["sudo", "install", "-m", "0644", str(SOURCE_RULE), str(SYSTEM_RULE)],
        check=True,
    )
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
    print(f"installed: {SYSTEM_RULE}")
    print("Stop teleoperation, then unplug and reconnect both USB-CAN adapters.")
    print("After reconnecting, verify that can_master and can_slave are present:")
    print("  ip -brief link show type can")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install stable USB-CAN names for the Nero master/slave arms."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the project rule without installing it.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
