#!/usr/bin/env python3
"""Downsample one H5 camera stream onto another camera's timeline."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Alignment:
    source_indices: np.ndarray
    source_timestamps_us: np.ndarray
    reference_timestamps_us: np.ndarray

    @property
    def absolute_error_us(self) -> np.ndarray:
        return np.abs(self.source_timestamps_us - self.reference_timestamps_us)


def nearest_monotonic_indices(
    source_timestamps_us: np.ndarray,
    reference_timestamps_us: np.ndarray,
) -> np.ndarray:
    source = _validate_timestamps(source_timestamps_us, "source")
    reference = _validate_timestamps(reference_timestamps_us, "reference")
    if source.size < reference.size:
        raise ValueError(
            "source camera has fewer frames than reference camera: "
            f"{source.size} < {reference.size}"
        )

    # Select exactly one increasing source index per reference timestamp while
    # minimizing total absolute timestamp error. This handles isolated dropped
    # source frames without duplicating images or breaking temporal order.
    source_count = source.size
    reference_count = reference.size
    infinity = np.iinfo(np.int64).max // 4
    previous = np.full(source_count + 1, infinity, dtype=np.int64)
    previous[:] = 0
    selected = np.zeros((reference_count, source_count), dtype=np.bool_)

    for reference_index, target in enumerate(reference):
        current = np.full(source_count + 1, infinity, dtype=np.int64)
        minimum_source = reference_index + 1
        maximum_source = source_count - (reference_count - reference_index - 1)
        for source_prefix in range(minimum_source, maximum_source + 1):
            skip_cost = current[source_prefix - 1]
            prior_cost = previous[source_prefix - 1]
            match_cost = (
                infinity
                if prior_cost >= infinity
                else prior_cost + abs(int(source[source_prefix - 1]) - int(target))
            )
            if match_cost < skip_cost:
                current[source_prefix] = match_cost
                selected[reference_index, source_prefix - 1] = True
            else:
                current[source_prefix] = skip_cost
        previous = current

    indices = np.empty(reference_count, dtype=np.int64)
    source_prefix = source_count
    for reference_index in range(reference_count - 1, -1, -1):
        while source_prefix > 0 and not selected[reference_index, source_prefix - 1]:
            source_prefix -= 1
        if source_prefix == 0:
            raise RuntimeError("failed to reconstruct monotonic camera alignment")
        indices[reference_index] = source_prefix - 1
        source_prefix -= 1
    return indices


def inspect_episode(
    path: Path,
    source_camera: str,
    reference_camera: str,
) -> Alignment:
    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        source_group = _camera_group(h5, path, source_camera)
        reference_group = _camera_group(h5, path, reference_camera)
        source_timestamps = np.asarray(source_group["timestamp_us"][:], dtype=np.int64)
        reference_timestamps = np.asarray(
            reference_group["timestamp_us"][:], dtype=np.int64
        )
        _validate_frame_count(source_group, source_timestamps.size, path, source_camera)
        _validate_frame_count(
            reference_group, reference_timestamps.size, path, reference_camera
        )

    indices = nearest_monotonic_indices(source_timestamps, reference_timestamps)
    return Alignment(
        source_indices=indices,
        source_timestamps_us=source_timestamps[indices],
        reference_timestamps_us=reference_timestamps,
    )


def downsample_episode(
    path: Path,
    source_camera: str,
    reference_camera: str,
    *,
    block_frames: int = 32,
) -> Alignment:
    expected = inspect_episode(path, source_camera, reference_camera)
    temp_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".downsample.tmp", dir=path.parent
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        shutil.copy2(path, temp_path)

        h5py = _import_h5py()
        with h5py.File(temp_path, "r+") as h5:
            group = _camera_group(h5, temp_path, source_camera)
            frames = group["frames"]
            timestamps = group["timestamp_us"]
            target_count = expected.source_indices.size
            original_count = int(frames.shape[0])
            replacement_frames = _create_replacement_dataset(
                group,
                "frames.downsample_tmp",
                frames,
                (target_count, *frames.shape[1:]),
            )
            replacement_timestamps = _create_replacement_dataset(
                group,
                "timestamp_us.downsample_tmp",
                timestamps,
                (target_count,),
            )
            for start in range(0, target_count, block_frames):
                stop = min(start + block_frames, target_count)
                selected = expected.source_indices[start:stop]
                replacement_frames[start:stop] = frames[selected]
            replacement_timestamps[:] = expected.source_timestamps_us
            del group["frames"]
            del group["timestamp_us"]
            group.move("frames.downsample_tmp", "frames")
            group.move("timestamp_us.downsample_tmp", "timestamp_us")
            group.attrs["downsample_reference_timeline"] = (
                f"cameras/{reference_camera}/timestamp_us"
            )
            group.attrs["downsample_method"] = "nearest_timestamp_one_to_one"
            group.attrs["downsample_source_frame_count"] = original_count
            h5.flush()

        actual = inspect_episode(temp_path, source_camera, reference_camera)
        if not np.array_equal(actual.source_timestamps_us, expected.source_timestamps_us):
            raise RuntimeError(f"post-write timestamp verification failed: {path}")
        if actual.source_indices.size != expected.source_indices.size:
            raise RuntimeError(f"post-write frame-count verification failed: {path}")
        os.replace(temp_path, path)
        temp_path = None
        return expected
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _validate_timestamps(values: np.ndarray, name: str) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.int64).reshape(-1)
    if timestamps.size == 0:
        raise ValueError(f"{name} camera timestamp vector is empty")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} camera timestamps must be strictly increasing")
    return timestamps


def _camera_group(h5, path: Path, camera_name: str):
    group_path = f"cameras/{camera_name}"
    if group_path not in h5:
        raise ValueError(f"missing {group_path} in {path}")
    group = h5[group_path]
    missing = [name for name in ("frames", "timestamp_us") if name not in group]
    if missing:
        raise ValueError(f"missing {group_path} datasets {missing} in {path}")
    return group


def _validate_frame_count(group, timestamp_count: int, path: Path, camera: str) -> None:
    frame_count = int(group["frames"].shape[0])
    if frame_count != timestamp_count:
        raise ValueError(
            f"camera {camera} frame/timestamp mismatch in {path}: "
            f"{frame_count} != {timestamp_count}"
        )


def _create_replacement_dataset(group, name: str, source, shape: tuple[int, ...]):
    options = {}
    if source.chunks is not None:
        options["chunks"] = tuple(min(size, limit) for size, limit in zip(source.chunks, shape))
    if source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    dataset = group.create_dataset(name, shape=shape, dtype=source.dtype, **options)
    for key, value in source.attrs.items():
        dataset.attrs[key] = value
    return dataset


def _import_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError("camera downsampling requires a working h5py installation") from exc
    return h5py


def _frequency_hz(timestamps_us: np.ndarray) -> float:
    if timestamps_us.size < 2:
        return float("nan")
    return float(1.0e6 / np.median(np.diff(timestamps_us)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match each reference-camera timestamp to the nearest unique source-camera "
            "frame. Without --apply, only validate and report the planned changes."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="H5 files or directories")
    parser.add_argument("--source-camera", default="side")
    parser.add_argument("--reference-camera", default="wrist")
    parser.add_argument("--apply", action="store_true", help="atomically overwrite H5 files")
    return parser.parse_args(argv)


def _episode_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        resolved = item.expanduser().resolve()
        if resolved.is_dir():
            paths.update(resolved.glob("episode_*.h5"))
        elif resolved.is_file():
            paths.add(resolved)
        else:
            raise FileNotFoundError(resolved)
    if not paths:
        raise ValueError("no H5 episodes found")
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _episode_paths(args.paths)
    total_before = 0
    total_after = 0
    maximum_error_us = 0
    for path in paths:
        h5py = _import_h5py()
        with h5py.File(path, "r") as h5:
            source_before = np.asarray(
                h5[f"cameras/{args.source_camera}/timestamp_us"][:], dtype=np.int64
            )
        alignment = (
            downsample_episode(path, args.source_camera, args.reference_camera)
            if args.apply
            else inspect_episode(path, args.source_camera, args.reference_camera)
        )
        errors = alignment.absolute_error_us
        total_before += source_before.size
        total_after += alignment.source_indices.size
        maximum_error_us = max(maximum_error_us, int(errors.max()))
        print(
            f"{path}: {source_before.size}->{alignment.source_indices.size} frames "
            f"source={_frequency_hz(source_before):.2f}Hz "
            f"reference={_frequency_hz(alignment.reference_timestamps_us):.2f}Hz "
            f"match_p99={np.percentile(errors, 99) / 1000.0:.2f}ms "
            f"match_max={errors.max() / 1000.0:.2f}ms"
        )
    action = "overwrote" if args.apply else "would overwrite"
    print(
        f"{action} {len(paths)} episodes: {total_before}->{total_after} source frames; "
        f"maximum timestamp error={maximum_error_us / 1000.0:.2f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
