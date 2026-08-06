#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.arms.factory import build_arm
from nero_collection.config import ArmEndpointConfig, load_config

log = logging.getLogger(__name__)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    pair = _select_pair(config.teleop.master_slave, args.pair)
    endpoint = pair.leader if args.role == "leader" else pair.follower
    arm = build_arm(endpoint, args.backend or config.teleop.backend)

    print(f"Testing {args.role} arm: {endpoint.name} on {endpoint.channel}")
    try:
        arm.connect()
        print("connect: OK")

        q = _try_wait_valid(lambda: arm.read_state().q, args.timeout_s, "normal joint q")
        if q is None:
            print("read get_joint_angles: WARN unavailable before leader_mode; continuing")
        else:
            print("read get_joint_angles: OK", _fmt(q))

        print("set_normal_mode...")
        arm.set_normal_mode()
        time.sleep(args.settle_s)
        print("set_normal_mode: OK")

        if args.enable:
            print("enable...")
            arm.enable()
            print("enable: OK")

        print("set_leader_mode...")
        arm.set_leader_mode()
        time.sleep(args.settle_s)
        leader_q = _wait_valid(arm.read_leader_joint_positions, args.timeout_s, "leader joint q")
        print("set_leader_mode/read leader q: OK", _fmt(leader_q))

        if args.include_follower_mode:
            print("set_follower_mode...")
            arm.set_follower_mode()
            time.sleep(args.settle_s)
            print("set_follower_mode: OK")

            print("return set_leader_mode...")
            arm.set_leader_mode()
            time.sleep(args.settle_s)
            leader_q = _wait_valid(arm.read_leader_joint_positions, args.timeout_s, "leader joint q after return")
            print("return leader_mode/read leader q: OK", _fmt(leader_q))

        print("mode check: PASS")
        return 0
    finally:
        arm.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a Nero arm can switch modes safely.")
    parser.add_argument("--config", default="configs/master_slave_can.yaml")
    parser.add_argument("--pair", default="main")
    parser.add_argument("--role", choices=("leader", "follower"), default="leader")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"))
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--enable", action="store_true", help="Also test arm.enable(). Use only when the arm is safe to enable.")
    parser.add_argument(
        "--include-follower-mode",
        action="store_true",
        help="Also test set_follower_mode() on this arm. Use carefully for the master/leader arm.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _select_pair(pairs, name: str):
    for pair in pairs:
        if pair.name == name:
            return pair
    raise RuntimeError(f"No arm pair named {name!r}; available: {[pair.name for pair in pairs]}")


def _wait_valid(read_fn, timeout_s: float, label: str) -> np.ndarray:
    result = _try_wait_valid(read_fn, timeout_s, label)
    if result is None:
        raise RuntimeError(f"Timed out waiting for {label}")
    return result


def _try_wait_valid(read_fn, timeout_s: float, label: str) -> np.ndarray | None:
    deadline = time.monotonic() + timeout_s
    last = np.empty((0,), dtype=np.float64)
    while time.monotonic() < deadline:
        last = np.asarray(read_fn(), dtype=np.float64).reshape(-1)
        if last.size and np.isfinite(last).all():
            return last
        time.sleep(0.05)
    log.warning("timed out waiting for %s; last=%s", label, last)
    return None


def _fmt(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in q.tolist()) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
