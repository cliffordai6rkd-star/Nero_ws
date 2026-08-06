#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


def main() -> int:
    args = _parse_args()
    fw = _firmware(args.firmware)
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        comm="can",
        firmeware_version=fw,
        channel=args.channel,
        interface=args.interface,
        bitrate=args.bitrate,
    )
    robot = AgxArmFactory.create_arm(cfg)
    print(f"connect channel={args.channel} firmware={args.firmware}")
    robot.connect()

    print("firmware:", _safe_call(robot, "get_firmware"))
    print("arm_status:", _unwrap(_safe_call(robot, "get_arm_status")))
    print("joint_enable_list:", _safe_call(robot, "get_joints_enable_status_list"))
    print("joint_angles:", _fmt_array(_msg(_safe_call(robot, "get_joint_angles"))))
    print("leader_joint_angles:", _fmt_array(_msg(_safe_call(robot, "get_leader_joint_angles"))))

    print("\ndriver states:")
    for i in range(1, 8):
        ds = _unwrap(_safe_call(robot, "get_driver_states", i))
        if ds is None:
            print(i, "None")
            continue
        foc = getattr(ds, "foc_status", None)
        enable_status = getattr(foc, "driver_enable_status", None) if foc is not None else None
        error_status = getattr(foc, "driver_error_status", None) if foc is not None else None
        print(i, "driver_enable_status =", enable_status, "driver_error_status =", error_status, "raw =", ds)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read real Nero joint enable/driver status from one CAN channel.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="V120", choices=("DEFAULT", "V111", "V112", "V120"))
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    return parser.parse_args()


def _firmware(name: str):
    return getattr(NeroFW, name)


def _safe_call(obj, name: str, *args):
    try:
        method = getattr(obj, name)
        return method(*args)
    except Exception as exc:
        return f"<{name} failed: {exc}>"


def _unwrap(value):
    if hasattr(value, "msg"):
        return value.msg
    return value


def _msg(value):
    if hasattr(value, "msg"):
        return value.msg
    return value


def _fmt_array(value) -> str:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return repr(value)
    if array.size == 0:
        return "[]"
    return "[" + ", ".join(f"{x:+.5f}" for x in array.tolist()) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
