#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIMELINE = "episode_time"
WRENCH_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
WRENCH_COLORS = (
    (230, 69, 83),
    (38, 166, 154),
    (233, 166, 49),
    (76, 114, 176),
    (170, 91, 170),
    (89, 161, 79),
)


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    episode_index: int
    camera_name: str
    teleop_time_s: np.ndarray
    camera_time_s: np.ndarray
    camera_frames: np.ndarray
    ee_pose: np.ndarray
    wrench_ext: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one Nero H5 episode in Rerun: camera, 3D end-effector "
            "trajectory, and external wrench."
        )
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing episode_NNNN_*.h5 files, for example runs/insert_usb",
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--camera",
        help="Camera group name; defaults to wrist when present, otherwise the first camera",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Write an .rrd recording instead of spawning the Rerun viewer",
    )
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.save is not None and args.save.suffix.lower() != ".rrd":
        parser.error("--save must use the .rrd extension")
    return args


def resolve_episode(runs_dir: Path, episode_index: int) -> Path:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise RuntimeError(f"Runs directory does not exist: {runs_dir}")
    matches = sorted(runs_dir.glob(f"episode_{episode_index:04d}_*.h5"))
    if not matches:
        raise RuntimeError(
            f"Episode {episode_index} was not found in {runs_dir}; expected "
            f"episode_{episode_index:04d}_*.h5"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Episode {episode_index} is ambiguous in {runs_dir}: {names}"
        )
    return matches[0]


def load_episode(path: Path, episode_index: int, camera_name: str | None) -> EpisodeData:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Reading Nero episodes requires a working h5py installation compatible with NumPy"
        ) from exc

    with h5py.File(path, "r") as h5:
        required = (
            "teleop/timestamp_us",
            "teleop/ee_pose_follower",
            "teleop/wrench_ext",
        )
        missing = [name for name in required if name not in h5]
        if missing:
            raise RuntimeError(f"Episode {path.name} is missing datasets: {missing}")
        if "cameras" not in h5:
            raise RuntimeError(f"Episode {path.name} does not contain camera data")

        available_cameras = sorted(
            name
            for name, group in h5["cameras"].items()
            if "frames" in group and "timestamp_us" in group
        )
        selected_camera = _select_camera(camera_name, available_cameras, path)
        camera_group = h5[f"cameras/{selected_camera}"]

        teleop_timestamp_us = np.asarray(h5["teleop/timestamp_us"][:], dtype=np.int64)
        ee_pose = np.asarray(h5["teleop/ee_pose_follower"][:], dtype=np.float64)
        wrench_ext = np.asarray(h5["teleop/wrench_ext"][:], dtype=np.float64)
        camera_timestamp_us = np.asarray(camera_group["timestamp_us"][:], dtype=np.int64)
        camera_frames = np.asarray(camera_group["frames"][:], dtype=np.uint8)

    _validate_episode_arrays(
        path,
        teleop_timestamp_us,
        ee_pose,
        camera_timestamp_us,
        camera_frames,
        wrench_ext,
    )
    origin_us = int(teleop_timestamp_us[0])
    return EpisodeData(
        path=path,
        episode_index=episode_index,
        camera_name=selected_camera,
        teleop_time_s=(teleop_timestamp_us - origin_us).astype(np.float64) * 1.0e-6,
        camera_time_s=(camera_timestamp_us - origin_us).astype(np.float64) * 1.0e-6,
        camera_frames=camera_frames,
        ee_pose=ee_pose,
        wrench_ext=wrench_ext,
    )


def log_episode(data: EpisodeData, wrench_ext: np.ndarray, save_path: Path | None) -> None:
    rr, rrb = _import_rerun()
    blueprint = _make_blueprint(rrb, data.camera_name)
    rr.init(
        f"nero_h5_episode_{data.episode_index:04d}",
        spawn=save_path is None,
        default_blueprint=blueprint,
    )
    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path, default_blueprint=blueprint)

    positions = data.ee_pose[:, :3, 3]
    rr.log("trajectory", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "trajectory/path",
        rr.LineStrips3D([positions], colors=[48, 157, 219], radii=[0.002]),
        static=True,
    )
    rr.send_columns(
        "trajectory/current",
        indexes=[rr.TimeColumn(TIMELINE, duration=data.teleop_time_s)],
        columns=rr.Points3D.columns(
            positions=positions,
            colors=np.full(positions.shape[0], 0xF05941FF, dtype=np.uint32),
            radii=np.full(positions.shape[0], 0.008, dtype=np.float32),
        ),
    )

    for index, (label, color) in enumerate(zip(WRENCH_LABELS, WRENCH_COLORS)):
        group = "force" if index < 3 else "moment"
        entity_path = f"wrench_ext/{group}/{label}"
        rr.log(
            entity_path,
            rr.SeriesLines(colors=[color], names=label, widths=[2.0]),
            static=True,
        )
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn(TIMELINE, duration=data.teleop_time_s)],
            columns=rr.Scalars.columns(scalars=wrench_ext[:, index]),
        )

    image_path = f"camera/{data.camera_name}/image"
    for timestamp_s, frame in zip(data.camera_time_s, data.camera_frames):
        rr.set_time(TIMELINE, duration=float(timestamp_s))
        rr.log(image_path, rr.Image(frame, color_model="RGB"))
    rr.disconnect()


def _make_blueprint(rrb, camera_name: str):
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin=f"/camera/{camera_name}",
                name=f"Camera: {camera_name}",
            ),
            rrb.Spatial3DView(
                origin="/trajectory",
                name="End-effector trajectory",
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/wrench_ext/force",
                    name="External force [N]",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/wrench_ext/moment",
                    name="External moment [N.m]",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                row_shares=[1.0, 1.0],
            ),
            column_shares=[1.0, 1.2, 1.0],
        ),
        collapse_panels=True,
    )


def _import_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "Rerun SDK is not installed; run: python -m pip install 'rerun-sdk>=0.26,<0.27'"
        ) from exc
    if not callable(getattr(rr, "init", None)):
        raise RuntimeError(
            "The unrelated PyPI package 'rerun' is shadowing rerun-sdk. Remove it with "
            "`python -m pip uninstall rerun`, then install `rerun-sdk>=0.26,<0.27`."
        )
    import rerun.blueprint as rrb

    return rr, rrb


def _select_camera(requested: str | None, available: list[str], path: Path) -> str:
    if not available:
        raise RuntimeError(f"Episode {path.name} has no complete camera streams")
    if requested is not None:
        if requested not in available:
            raise RuntimeError(
                f"Camera {requested!r} is not in {path.name}; available={available}"
            )
        return requested
    return "wrist" if "wrist" in available else available[0]


def _validate_episode_arrays(
    path: Path,
    timestamp_us: np.ndarray,
    ee_pose: np.ndarray,
    camera_timestamp_us: np.ndarray,
    camera_frames: np.ndarray,
    wrench_ext: np.ndarray,
) -> None:
    sample_count = timestamp_us.size
    expected = {
        "teleop/timestamp_us": (sample_count,),
        "teleop/ee_pose_follower": (sample_count, 4, 4),
        "teleop/wrench_ext": (sample_count, 6),
    }
    actual = {
        "teleop/timestamp_us": timestamp_us.shape,
        "teleop/ee_pose_follower": ee_pose.shape,
        "teleop/wrench_ext": wrench_ext.shape,
    }
    invalid = [
        f"{name}: expected {expected[name]}, got {shape}"
        for name, shape in actual.items()
        if shape != expected[name]
    ]
    if sample_count == 0:
        invalid.append("teleop timeline is empty")
    if camera_timestamp_us.shape != (camera_frames.shape[0],):
        invalid.append(
            "camera timestamp/frame mismatch: "
            f"{camera_timestamp_us.shape} versus {camera_frames.shape}"
        )
    if camera_frames.ndim != 4 or camera_frames.shape[-1] not in (3, 4):
        invalid.append(f"camera frames must be NxHxWx3/4, got {camera_frames.shape}")
    if invalid:
        raise RuntimeError(f"Invalid episode {path.name}: " + "; ".join(invalid))
    for name, values in (
        ("teleop/ee_pose_follower", ee_pose),
        ("teleop/wrench_ext", wrench_ext),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"Episode {path.name} contains non-finite {name} values")
    if np.any(np.diff(timestamp_us) <= 0):
        raise RuntimeError(f"Episode {path.name} teleop timestamps are not strictly increasing")
    if camera_timestamp_us.size == 0 or np.any(np.diff(camera_timestamp_us) <= 0):
        raise RuntimeError(f"Episode {path.name} camera timestamps are empty or not increasing")


def main() -> int:
    args = parse_args()
    episode_path = resolve_episode(args.runs_dir, args.episode)
    data = load_episode(episode_path, args.episode, args.camera)
    print(
        f"Loading {episode_path.name}: teleop={data.teleop_time_s.size}, "
        f"camera={data.camera_name}:{data.camera_time_s.size}"
    )
    print("External wrench source: teleop/wrench_ext")
    log_episode(data, data.wrench_ext, args.save)
    if args.save is not None:
        print(f"Saved Rerun recording: {args.save.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
