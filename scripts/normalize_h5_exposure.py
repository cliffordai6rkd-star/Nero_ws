#!/usr/bin/env python3
"""Normalize exposure-scaled camera frames in Nero episode H5 files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


_EXPOSURE_LINE = re.compile(r"^(\s+exposure:\s*)(\d+)(\s*)$", re.MULTILINE)


def normalize_episode(path: Path, target_exposure: int, block_frames: int) -> str:
    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        raw_config = h5["config_yaml"][()]
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    raw_config = str(raw_config)
    matches = list(_EXPOSURE_LINE.finditer(raw_config))
    exposures = [int(match.group(2)) for match in matches]
    if len(exposures) != 2 or exposures[0] != exposures[1]:
        raise RuntimeError(f"expected two matching camera exposures, found {exposures}")
    source_exposure = exposures[0]
    if source_exposure == target_exposure:
        return "skipped"
    if source_exposure <= 0 or target_exposure <= 0:
        raise RuntimeError(
            f"exposures must be positive, found source={source_exposure}, target={target_exposure}"
        )

    scale = target_exposure / source_exposure
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".exposure.tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        shutil.copy2(path, temporary_path)

        with h5py.File(temporary_path, "r+") as h5:
            frame_shapes: dict[str, tuple[int, ...]] = {}
            for camera_name, group in h5["cameras"].items():
                frames = group["frames"]
                if frames.dtype != np.dtype("uint8") or frames.ndim != 4 or frames.shape[-1] != 3:
                    raise RuntimeError(
                        f"{camera_name}/frames has unsupported dtype/shape: "
                        f"{frames.dtype} {frames.shape}"
                    )
                frame_shapes[camera_name] = tuple(frames.shape)
                for start in range(0, frames.shape[0], block_frames):
                    stop = min(start + block_frames, frames.shape[0])
                    values = np.asarray(frames[start:stop], dtype=np.float32)
                    frames[start:stop] = np.clip(
                        np.rint(values * scale), 0, 255
                    ).astype(np.uint8)

            updated_config, replacement_count = _EXPOSURE_LINE.subn(
                lambda match: f"{match.group(1)}{target_exposure}{match.group(3)}",
                raw_config,
            )
            if replacement_count != 2:
                raise RuntimeError(
                    f"expected two exposure replacements, got {replacement_count}"
                )
            h5["config_yaml"][()] = updated_config
            h5.flush()

        with h5py.File(temporary_path, "r") as h5:
            updated_config = h5["config_yaml"][()]
            if isinstance(updated_config, bytes):
                updated_config = updated_config.decode("utf-8")
            updated_exposures = [
                int(match.group(2))
                for match in _EXPOSURE_LINE.finditer(str(updated_config))
            ]
            if updated_exposures != [target_exposure, target_exposure]:
                raise RuntimeError(
                    f"post-write exposure verification failed: {updated_exposures}"
                )
            for camera_name, expected_shape in frame_shapes.items():
                frames = h5[f"cameras/{camera_name}/frames"]
                if tuple(frames.shape) != expected_shape or frames.dtype != np.dtype("uint8"):
                    raise RuntimeError(f"post-write frame verification failed for {camera_name}")

        os.replace(temporary_path, path)
        temporary_path = None
        return f"converted {source_exposure}->{target_exposure}"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _import_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError("H5 exposure normalization requires a working h5py installation") from exc
    return h5py


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale uint8 camera frames to a target exposure and update the "
            "embedded camera configuration. Existing target-exposure files are skipped."
        )
    )
    parser.add_argument("root", type=Path, help="Directory containing episode_*.h5 files")
    parser.add_argument("--target-exposure", type=int, default=200)
    parser.add_argument("--block-frames", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many files that need conversion, then exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"H5 root does not exist: {root}")
    if args.target_exposure <= 0:
        raise ValueError("--target-exposure must be positive")
    if args.block_frames <= 0:
        raise ValueError("--block-frames must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    paths = sorted(root.glob("episode_*.h5"))
    converted = skipped = 0
    for path in paths:
        if args.limit is not None and converted >= args.limit:
            break
        result = normalize_episode(path, args.target_exposure, args.block_frames)
        if result == "skipped":
            skipped += 1
        else:
            converted += 1
        print(f"{path.name}: {result}", flush=True)
    print(f"completed: converted={converted}, skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
