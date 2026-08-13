#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.arms.factory import build_arm
from nero_collection.config import ArmEndpointConfig, load_config


def main() -> int:
    args = _parse_args()
    endpoint = _endpoint_from_args(args)
    arm = build_arm(endpoint, args.backend)
    period = 1.0 / max(args.hz, 1e-6)

    print(f"Connecting {endpoint.name} on {endpoint.channel}; source={args.source}")
    arm.connect()
    try:
        if args.disable:
            arm.disable()
            print("disable: OK")
            if args.set_leader_mode or args.set_follower_mode:
                print("mode switch skipped because --disable was requested")
        else:
            if args.set_leader_mode:
                arm.set_leader_mode()
                print("set_leader_mode: OK")
            if args.set_follower_mode:
                arm.set_follower_mode()
                print("set_follower_mode: OK")

        print("Press Ctrl-C to stop.")
        while True:
            if args.source == "leader":
                q = arm.read_leader_joint_positions()
            else:
                q = arm.read_state().q
            print(_fmt_q(q), flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0
    finally:
        arm.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print realtime Nero arm joint q from one CAN channel.")
    parser.add_argument("--config", default=None, help="Optional config path. If set, --role selects arm endpoint from it.")
    parser.add_argument("--pair", default="main")
    parser.add_argument("--role", choices=("leader", "follower"), default="follower")
    parser.add_argument("--channel", default="can_slave")
    parser.add_argument("--name", default="single_arm")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--firmware", default="V120")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"), default="pyagxarm")
    parser.add_argument("--source", choices=("normal", "leader"), default="normal")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--set-leader-mode", action="store_true")
    parser.add_argument("--set-follower-mode", action="store_true")
    parser.add_argument("--disable", action="store_true", help="Disable arm motors after connect before printing q.")
    return parser.parse_args()


def _endpoint_from_args(args: argparse.Namespace) -> ArmEndpointConfig:
    if args.config:
        cfg = load_config(args.config)
        for pair in cfg.teleop.master_slave:
            if pair.name == args.pair:
                return pair.leader if args.role == "leader" else pair.follower
        raise RuntimeError(f"No pair named {args.pair!r}")
    return ArmEndpointConfig(
        name=args.name,
        channel=args.channel,
        interface=args.interface,
        bitrate=args.bitrate,
        firmware=args.firmware,
    )


def _fmt_q(q: np.ndarray) -> str:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if not q.size or not np.isfinite(q).all():
        return "q: INVALID " + np.array2string(q, precision=4, suppress_small=False)
    return "q: [" + ", ".join(f"{x:+.5f}" for x in q.tolist()) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
