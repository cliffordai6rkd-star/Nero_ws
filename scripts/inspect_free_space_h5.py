#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path, details = find_latest_episode(args.runs_dir, args.source)
    print(f"path: {path}")
    print(f"format: {details['format']}")
    print(f"samples: {details['samples']}")
    print(f"trajectory: {details['trajectory_name']} {details['trajectory_seed']}")
    print(f"trajectory sha256: {details['trajectory_sha256']}")
    return 0


def find_latest_episode(
    runs_dir: Path,
    source: str = "free_space_coverage",
) -> tuple[Path, dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("H5 inspection requires h5py") from exc

    directory = runs_dir.expanduser().resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Runs directory does not exist: {directory}")
    paths = sorted(
        directory.glob("episode_*.h5"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        with h5py.File(path, "r") as h5:
            episode = h5.get("metadata/episode_json")
            if episode is None:
                continue
            metadata = _decode_metadata(episode[()], path)
            if metadata.get("source") != source:
                continue
            timestamp = h5.get("teleop/timestamp_us")
            if timestamp is None:
                raise RuntimeError(f"Episode is missing teleop/timestamp_us: {path}")
            required = ("trajectory_name", "trajectory_seed", "trajectory_sha256")
            missing = [name for name in required if name not in metadata]
            if missing:
                raise RuntimeError(
                    f"Episode metadata is missing {', '.join(missing)}: {path}"
                )
            return path, {
                "format": h5.attrs.get("format", "unknown"),
                "samples": int(timestamp.shape[0]),
                **{name: metadata[name] for name in required},
            }
    raise RuntimeError(
        f"No {source!r} H5 episode was found in {directory}"
    )


def _decode_metadata(value: Any, path: Path) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError(f"Invalid metadata/episode_json value in {path}")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid metadata/episode_json in {path}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"metadata/episode_json must be an object in {path}")
    return decoded


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the newest free-space H5 episode and its trajectory metadata."
    )
    parser.add_argument("runs_dir", nargs="?", type=Path, default=Path("runs/insert_usb"))
    parser.add_argument("--source", default="free_space_coverage")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
