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
from nero_collection.config import load_config


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    pair = _select_pair(config.teleop.master_slave, args.pair)
    endpoint = pair.leader
    arm = build_arm(endpoint, args.backend or config.teleop.backend)
    zeros = np.zeros(7, dtype=np.float64)
    last_q: np.ndarray | None = None
    mit_sent = False

    print(f"Probing leader arm: {endpoint.name} on {endpoint.channel}")
    print("Keep the arm away from joint limits and keep the E-stop within reach.")
    try:
        arm.connect()
        print("connect: OK")
        if args.enable:
            print("enable requested; this may reset/clear errors if the first enable attempt fails...")
            arm.enable()
            print("enable: OK")
        else:
            print("enable: skipped (use --enable only if the arm is not already enabled)")
        arm.validate_joint_impedance_support()
        print("move_mit API: available")

        arm.set_leader_mode()
        time.sleep(args.settle_s)
        last_q = _read_valid_q(arm, args.timeout_s)
        role_before = arm.read_control_role(refresh=True)
        print(f"before MIT: role={role_before or 'unknown'} q={_fmt(last_q)}")
        if role_before != "leader":
            raise RuntimeError(f"Leader mode was not confirmed before MIT probe: {role_before}")

        if not args.zero_probe:
            print("READ-ONLY PASS: no MIT command sent. Add --zero-probe to test compatibility.")
            return 0

        print("sending one zero-gain MIT command at the current position...")
        arm.command_joint_impedance(last_q, zeros, zeros, zeros, zeros)
        mit_sent = True
        time.sleep(args.settle_s)
        role_after_zero = arm.read_control_role(refresh=True)
        print(f"after zero MIT: role={role_after_zero or 'unknown'}")
        if role_after_zero != "leader":
            print("RESULT: move_mit did not preserve confirmed leader mode; active test stopped.")
            return 2

        if args.damping_kd is None and args.torque_ff is None:
            print("ZERO-PROBE PASS: leader mode remained confirmed. No nonzero impedance was sent.")
            return 0

        kd = zeros.copy()
        t_ff = zeros.copy()
        if args.damping_kd is not None:
            kd[args.joint - 1] = args.damping_kd
            test_label = f"kd={args.damping_kd:.4g}"
            instruction = "move that joint slowly"
        else:
            t_ff[args.joint - 1] = args.torque_ff
            test_label = f"t_ff={args.torque_ff:+.4g} N.m"
            instruction = "hold the arm lightly and observe the torque direction"
        period_s = 1.0 / args.rate_hz
        deadline = time.monotonic() + args.duration_s
        print(
            f"applying joint{args.joint} {test_label} for "
            f"{args.duration_s:.1f}s; {instruction} and press Ctrl-C to stop"
        )
        next_role_check = time.monotonic() + args.role_check_interval_s
        command_count = 0
        while time.monotonic() < deadline:
            started = time.monotonic()
            last_q = _read_valid_q(arm, args.timeout_s)
            arm.command_joint_impedance(last_q, zeros, zeros, kd, t_ff)
            command_count += 1
            if started >= next_role_check:
                role = arm.read_control_role(refresh=True)
                print(
                    f"role={role or 'unknown'} q{args.joint}={last_q[args.joint - 1]:+.4f} "
                    f"commands={command_count}",
                    flush=True,
                )
                if role != "leader":
                    raise RuntimeError(f"Leader mode was lost during active test: {role}")
                next_role_check = started + args.role_check_interval_s
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))

        print("ACTIVE PROBE COMPLETE: compare the felt effect with the zero-gain baseline.")
        return 0
    except KeyboardInterrupt:
        print("\nstopped by operator")
        return 130
    finally:
        if mit_sent and last_q is not None:
            try:
                arm.command_joint_impedance(last_q, zeros, zeros, zeros, zeros)
                print("cleanup: zero MIT command sent")
            except Exception as exc:
                print(f"cleanup WARNING: failed to send zero MIT command: {exc}")
        try:
            final_role = arm.read_control_role(refresh=True)
            print(f"final role: {final_role or 'unknown'}")
        except Exception:
            pass
        arm.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether Nero preserves leader mode while accepting MIT impedance commands."
    )
    parser.add_argument("--config", default="configs/master_slave_can.yaml")
    parser.add_argument("--pair", default="main")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"))
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable the arm before probing. By default the script preserves the existing enable state.",
    )
    parser.add_argument(
        "--zero-probe",
        action="store_true",
        help="Explicitly authorize one zero-gain move_mit command. Without it the script is read-only.",
    )
    parser.add_argument(
        "--damping-kd",
        type=float,
        default=None,
        help="After a successful zero probe, apply this positive damping gain to one joint.",
    )
    parser.add_argument(
        "--torque-ff",
        type=float,
        default=None,
        help="After a successful zero probe, continuously apply this feedforward torque in N.m.",
    )
    parser.add_argument("--joint", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--role-check-interval-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    if args.damping_kd is not None and args.torque_ff is not None:
        parser.error("--damping-kd and --torque-ff are mutually exclusive")
    if args.damping_kd is not None:
        if not args.zero_probe:
            parser.error("--damping-kd requires --zero-probe")
        if not 0.0 < args.damping_kd <= 0.5:
            parser.error("--damping-kd must be in the conservative range (0, 0.5]")
    if args.torque_ff is not None:
        if not args.zero_probe:
            parser.error("--torque-ff requires --zero-probe")
        if not 0.0 < abs(args.torque_ff) <= 3.0:
            parser.error("--torque-ff magnitude must be in the allowed range (0, 3.0] N.m")
    if not 0.1 <= args.duration_s <= 30.0:
        parser.error("--duration-s must be in [0.1, 30]")
    if not 1.0 <= args.rate_hz <= 50.0:
        parser.error("--rate-hz must be in [1, 50]")
    if args.timeout_s <= 0 or args.settle_s < 0 or args.role_check_interval_s <= 0:
        parser.error("timeouts and role-check interval must be positive")
    return args


def _select_pair(pairs, name: str):
    for pair in pairs:
        if pair.name == name:
            return pair
    raise RuntimeError(f"No arm pair named {name!r}; available: {[pair.name for pair in pairs]}")


def _read_valid_q(arm, timeout_s: float) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    last = np.empty((0,), dtype=np.float64)
    while time.monotonic() < deadline:
        last = np.asarray(arm.read_leader_joint_positions(), dtype=np.float64).reshape(-1)
        if last.size == 7 and np.isfinite(last).all():
            return last
        time.sleep(0.02)
    raise RuntimeError(f"Timed out waiting for a valid 7D leader q; last={last}")


def _fmt(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in q.tolist()) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
