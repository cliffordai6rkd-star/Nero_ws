#!/usr/bin/env python3
"""Create the Nero Python environment with uv.

Python packages are resolved from ``pyproject.toml``/``uv.lock``. The sibling
PINN checkout remains an explicit editable install because it is intentionally
outside this repository; CAN/V4L2 utilities and device permissions are handled
by Ubuntu rather than by uv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.10"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv was not found; install it with "
            "curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

    venv = (ROOT / args.venv).resolve()
    if not venv.exists():
        print(f"Creating uv environment: {venv}", flush=True)
        _run([uv, "venv", "--python", args.python_version, str(venv)])
    env_python = venv / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if not env_python.is_file():
        raise RuntimeError(f"environment Python was not found: {env_python}")

    # Keep the lockfile as the source of truth. ``inference`` selects the
    # current LeRobot/CUDA stack and ``hardware`` adds the real-arm SDK. The
    # optional ``legacy-dp`` extra is intentionally not installed here because
    # its headless OpenCV dependency conflicts with the camera preview path.
    print("Syncing Nero inference dependencies with uv", flush=True)
    sync_env = os.environ.copy()
    sync_env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    _run(
        [
            uv,
            "sync",
            "--python",
            str(env_python),
            "--extra",
            "inference",
            "--extra",
            "hardware",
            "--locked",
        ],
        env=sync_env,
    )

    pinn = ROOT.parent / "PINN"
    if (pinn / "setup.py").is_file() or (pinn / "pyproject.toml").is_file():
        print("Installing sibling PINN project", flush=True)
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(env_python),
                "--no-deps",
                "-e",
                str(pinn),
            ]
        )
    else:
        print(f"PINN project not found at {pinn}; native WM checkpoints unavailable")

    print("\nEnvironment is ready.")
    print("  uv run python -m inference.cli --config inference/configs/nero_contact_wm.yaml")
    return 0


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update the Nero uv environment."
    )
    parser.add_argument(
        "--venv",
        default=os.environ.get("NERO_UV_VENV", ".venv"),
        help="virtual environment directory relative to the repository root",
    )
    parser.add_argument(
        "--python-version",
        default=os.environ.get("NERO_PYTHON_VERSION", PYTHON_VERSION),
        help="Python version managed by uv (the DP stack currently requires 3.10)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
