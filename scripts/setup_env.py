#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYAGXARM_COMMIT = "799b8412fbe8b9156bc9892d3dbeb2df7e98be71"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "conda was not found; install Miniconda or Anaconda, then rerun this script"
        )

    diffusion_policy = ROOT / "third_party" / "diffusion_policy"
    pinn = ROOT.parent / "PINN"
    if not (diffusion_policy / "setup.py").is_file():
        print("Initializing diffusion_policy submodule", flush=True)
        _run(
            [
                "git",
                "-C",
                str(ROOT),
                "submodule",
                "update",
                "--init",
                "third_party/diffusion_policy",
            ]
        )

    prefix = _find_conda_environment(conda, args.env_name)
    print(f"Setting up conda environment: {args.env_name}", flush=True)
    if prefix is None:
        _run(
            [
                conda,
                "env",
                "create",
                "-n",
                args.env_name,
                "-f",
                str(ROOT / "environment.yml"),
            ]
        )
    else:
        _run(
            [
                conda,
                "env",
                "update",
                "-n",
                args.env_name,
                "-f",
                str(ROOT / "environment.yml"),
                "--prune",
            ]
        )
    prefix = _find_conda_environment(conda, args.env_name)
    if prefix is None:
        raise RuntimeError(f"conda environment was not created: {args.env_name}")
    env_python = prefix / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not env_python.is_file():
        raise RuntimeError(f"environment Python was not found: {env_python}")

    print("Installing nero_collection in editable mode", flush=True)
    _pip(env_python, args.pypi_index_url, "--no-deps", "-e", str(ROOT))

    print("Installing pinned CUDA 12.6 PyTorch stack", flush=True)
    _pip(
        env_python,
        args.pytorch_index_url,
        f"torch=={args.pytorch_version}",
        f"torchvision=={args.torchvision_version}",
    )
    _pip(
        env_python,
        args.pytorch_index_url,
        f"torchcodec=={args.torchcodec_version}",
    )

    print("Installing LeRobot and diffusion_policy", flush=True)
    _pip(env_python, args.pypi_index_url, "lerobot==0.4.0")
    _pip(
        env_python,
        args.pypi_index_url,
        "-e",
        f"{diffusion_policy}[training,test]",
    )

    if (pinn / "setup.py").is_file():
        print("Installing sibling PINN project", flush=True)
        _pip(env_python, args.pypi_index_url, "--no-deps", "-e", str(pinn))
    else:
        print(f"PINN project not found at {pinn}; native PINN checkpoints unavailable")

    print("Installing AgileX pyAgxArm SDK", flush=True)
    _pip(env_python, args.pypi_index_url, "python-can>=3.3.4")
    _pip(
        env_python,
        args.pypi_index_url,
        "--force-reinstall",
        f"git+https://github.com/agilexrobotics/pyAgxArm.git@{PYAGXARM_COMMIT}",
    )

    print(f"\nEnvironment is ready. Activate it with:\n  conda activate {args.env_name}")
    if shutil.which("candump") is None or shutil.which("v4l2-ctl") is None:
        print("\nOptional system tools:\n  sudo apt-get install -y can-utils v4l-utils")
    else:
        print("\ncan-utils and v4l-utils look available.")
    return 0


def _find_conda_environment(conda: str, name: str) -> Path | None:
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    for raw_prefix in payload.get("envs", []):
        prefix = Path(raw_prefix).expanduser().resolve()
        if prefix.name == name:
            return prefix
    return None


def _pip(env_python: Path, index_url: str, *requirements: str) -> None:
    _run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--index-url",
            index_url,
            *requirements,
        ]
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update the Nero conda environment using its Python directly."
    )
    parser.add_argument("--env-name", default=os.environ.get("NERO_ENV_NAME", "nero"))
    parser.add_argument(
        "--pypi-index-url",
        default=os.environ.get(
            "NERO_PYPI_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ),
    )
    parser.add_argument(
        "--pytorch-index-url",
        default=os.environ.get(
            "NERO_PYTORCH_INDEX_URL", "https://mirror.nju.edu.cn/pytorch/whl/cu126"
        ),
    )
    parser.add_argument(
        "--pytorch-version",
        default=os.environ.get("NERO_PYTORCH_VERSION", "2.7.1"),
    )
    parser.add_argument(
        "--torchvision-version",
        default=os.environ.get("NERO_TORCHVISION_VERSION", "0.22.1"),
    )
    parser.add_argument(
        "--torchcodec-version",
        default=os.environ.get("NERO_TORCHCODEC_VERSION", "0.5"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
