#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    recordings = [path.expanduser().resolve() for path in args.recordings]
    missing = [path for path in recordings if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Rerun recording does not exist: {names}")
    result = subprocess.run(
        [sys.executable, "-m", "rerun", *(str(path) for path in recordings)],
        check=False,
    )
    return int(result.returncode)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open one or more .rrd recordings with the Rerun viewer."
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
