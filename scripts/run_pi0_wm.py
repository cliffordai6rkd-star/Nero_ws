#!/usr/bin/env python3
"""Independent pi0 -> CaRS-WM -> position-only deployment."""
from pathlib import Path
import argparse
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.pi0_wm.config import load_config
from inference.pi0_wm.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'inference/configs/pi0_wm.yaml'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='default; no real arm commands')
    mode.add_argument('--enable-commands', action='store_true', help='explicitly enable real q position commands')
    parser.add_argument('--mock', action='store_true', help='mock pi0, WM, arm, cameras; no hardware opened')
    parser.add_argument('--mock-wm', action='store_true', help='mock only WM, for websocket integration tests')
    parser.add_argument('--headless', action='store_true', help='MuJoCo FK without viewer, no camera windows')
    parser.add_argument('--steps', type=int, help='number of external control steps after calibration')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config(args.config)
    if args.headless:
        config['mujoco']['headless'] = True
        for camera in config['cameras']:
            camera['visualize'] = False
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    result = Runtime(config, enable_commands=args.enable_commands, mock=args.mock, mock_wm=args.mock_wm).run(args.steps)
    logging.info('pi0_wm finished: %s', result)


if __name__ == '__main__':
    main()
