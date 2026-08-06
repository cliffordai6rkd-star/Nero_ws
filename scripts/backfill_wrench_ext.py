#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nero_collection.config import load_config
from nero_collection.contact_wrench import (
    PinocchioContactWrenchEstimator,
    wrench_ext_dataset_attrs,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATASETS = (
    "teleop/timestamp_us",
    "teleop/q_follower",
    "teleop/tau_ext",
)


class MissingSourceData(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute end-effector wrench_ext for existing Nero H5 episodes and "
            "atomically write teleop/wrench_ext."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="H5 files or directories containing episode_*.h5 files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "master_slave_can.yaml",
        help="Collection config supplying URDF and wrench mapping settings",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes; without this flag the command only validates and reports",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute episodes that already contain teleop/wrench_ext",
    )
    return parser.parse_args()


def discover_h5_files(inputs: list[Path], recursive: bool) -> list[Path]:
    files: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".h5":
                raise RuntimeError(f"Input file is not an H5 episode: {path}")
            files.add(path)
            continue
        if not path.is_dir():
            raise RuntimeError(f"Input does not exist: {path}")
        pattern = "**/episode_*.h5" if recursive else "episode_*.h5"
        files.update(candidate.resolve() for candidate in path.glob(pattern))
    return sorted(files)


def inspect_episode(path: Path) -> tuple[int, bool]:
    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        missing = [name for name in REQUIRED_DATASETS if name not in h5]
        if missing:
            raise MissingSourceData(f"missing datasets: {missing}")
        timestamp = np.asarray(h5["teleop/timestamp_us"][:], dtype=np.int64)
        q = np.asarray(h5["teleop/q_follower"])
        tau_ext = np.asarray(h5["teleop/tau_ext"])
        _validate_source_arrays(timestamp, q, tau_ext)
        has_wrench = "teleop/wrench_ext" in h5
        if has_wrench:
            wrench = np.asarray(h5["teleop/wrench_ext"])
            if wrench.shape != (timestamp.size, 6) or not np.isfinite(wrench).all():
                raise RuntimeError(
                    f"existing teleop/wrench_ext is invalid: shape={wrench.shape}"
                )
    return timestamp.size, has_wrench


def compute_episode_wrench(
    path: Path,
    estimator: PinocchioContactWrenchEstimator,
) -> np.ndarray:
    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        timestamp = np.asarray(h5["teleop/timestamp_us"][:], dtype=np.int64)
        q = np.asarray(h5["teleop/q_follower"][:], dtype=np.float64)
        tau_ext = np.asarray(h5["teleop/tau_ext"][:], dtype=np.float64)
    _validate_source_arrays(timestamp, q, tau_ext)
    wrench = np.empty((timestamp.size, 6), dtype=np.float64)
    for index, (q_value, tau_value) in enumerate(zip(q, tau_ext)):
        wrench[index] = estimator.map_joint_torque(q_value, tau_value).wrench
    if not np.isfinite(wrench).all():
        raise RuntimeError("computed wrench_ext contains non-finite values")
    return wrench


def write_wrench_atomic(
    path: Path,
    wrench: np.ndarray,
    attrs: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    h5py = _import_h5py()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copy2(path, temporary_path)
        with h5py.File(temporary_path, "r+") as h5:
            teleop = h5["teleop"]
            if "wrench_ext" in teleop:
                if not overwrite:
                    raise RuntimeError("teleop/wrench_ext already exists")
                del teleop["wrench_ext"]
            dataset = teleop.create_dataset(
                "wrench_ext",
                data=wrench,
                compression="gzip",
                compression_opts=4,
            )
            dataset.attrs["state_name"] = "wrench"
            dataset.attrs["lowpass"] = False
            dataset.attrs["median_window"] = 1
            for name, value in attrs.items():
                dataset.attrs[name] = value
            h5.flush()
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_source_arrays(
    timestamp: np.ndarray,
    q: np.ndarray,
    tau_ext: np.ndarray,
) -> None:
    sample_count = timestamp.size
    if timestamp.shape != (sample_count,) or sample_count == 0:
        raise RuntimeError(f"invalid teleop/timestamp_us shape: {timestamp.shape}")
    if q.shape != (sample_count, 7):
        raise RuntimeError(f"invalid teleop/q_follower shape: {q.shape}")
    if tau_ext.shape != (sample_count, 7):
        raise RuntimeError(f"invalid teleop/tau_ext shape: {tau_ext.shape}")
    if np.any(np.diff(timestamp) <= 0):
        raise RuntimeError("teleop/timestamp_us is not strictly increasing")
    if not np.isfinite(q).all() or not np.isfinite(tau_ext).all():
        raise RuntimeError("q_follower or tau_ext contains non-finite values")


def _import_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "A working h5py installation compatible with NumPy is required"
        ) from exc
    return h5py


def main() -> int:
    args = parse_args()
    files = discover_h5_files(args.inputs, args.recursive)
    if not files:
        raise RuntimeError("No episode H5 files were found")
    config = load_config(args.config.expanduser().resolve())
    mapping_config = config.realtime_plot.wrench_mapping
    attrs = wrench_ext_dataset_attrs(mapping_config)
    estimator = None
    written = 0
    ready = 0
    skipped = 0
    failed = 0
    for path in files:
        try:
            sample_count, has_wrench = inspect_episode(path)
            if has_wrench and not args.overwrite:
                print(f"SKIP  {path}: wrench_ext already exists")
                skipped += 1
                continue
            if not args.write:
                action = "overwrite" if has_wrench else "add"
                print(f"READY {path}: {action} wrench_ext for {sample_count} samples")
                ready += 1
                continue
            if estimator is None:
                estimator = PinocchioContactWrenchEstimator(mapping_config)
            wrench = compute_episode_wrench(path, estimator)
            write_wrench_atomic(path, wrench, attrs, overwrite=args.overwrite)
            print(f"WRITE {path}: wrench_ext shape={wrench.shape}")
            written += 1
        except MissingSourceData as exc:
            print(f"SKIP  {path}: {exc}")
            skipped += 1
        except Exception as exc:
            print(f"ERROR {path}: {exc}")
            failed += 1
    mode = "write" if args.write else "check"
    print(
        f"Summary mode={mode} files={len(files)} written={written} ready={ready} "
        f"skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
