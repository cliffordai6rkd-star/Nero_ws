#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIMELINE = "episode_time"
TAU_EXT_DATASETS = (
    "tau_ext_cal",
    "tau_ext_pred",
    "tau_ext_cal_raw",
    "tau_ext_pred_raw",
)
TAU_EXT_COLORS = (
    (48, 114, 176),
    (240, 89, 65),
    (76, 166, 154),
    (233, 166, 49),
)


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    episode_index: int
    teleop_time_s: np.ndarray
    cameras: tuple["CameraData", ...]
    ee_pose: np.ndarray
    tau_ext: dict[str, np.ndarray]

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    @property
    def camera_name(self) -> str:
        """Compatibility alias for callers that expect one camera name."""

        return self.cameras[0].name

    @property
    def camera_time_s(self) -> np.ndarray:
        """Compatibility alias for the first selected camera timeline."""

        return self.cameras[0].time_s

    @property
    def camera_frames(self) -> np.ndarray:
        """Compatibility alias for the first selected camera frames."""

        return self.cameras[0].frames

    @property
    def tau_ext_names(self) -> tuple[str, ...]:
        return tuple(self.tau_ext)


@dataclass(frozen=True)
class CameraData:
    name: str
    time_s: np.ndarray
    frames: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one Nero H5 episode in Rerun: camera streams, 3D "
            "end-effector trajectory, and ||tau_ext||."
        )
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing episode_NNNN_*.h5 files, for example runs/insert_usb",
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--cam",
        "--camera",
        dest="camera",
        help=(
            "Camera groups separated by '/'; for example --cam wrist/side. "
            "Defaults to all complete camera streams"
        ),
    )
    parser.add_argument(
        "--tau-ext-source",
        choices=("auto", *TAU_EXT_DATASETS),
        default="auto",
        help=(
            "Teleop external torque dataset(s) to plot; auto plots calibrated "
            "and predicted tau_ext when available"
        ),
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


def load_episode(
    path: Path,
    episode_index: int,
    camera_name: str | None,
    tau_ext_source: str = "auto",
) -> EpisodeData:
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
        selected_cameras = _select_cameras(camera_name, available_cameras, path)
        selected_tau_ext = _select_tau_ext(tau_ext_source, h5["teleop"], path)

        teleop_timestamp_us = np.asarray(h5["teleop/timestamp_us"][:], dtype=np.int64)
        ee_pose = np.asarray(h5["teleop/ee_pose_follower"][:], dtype=np.float64)
        tau_ext = {
            name: np.asarray(h5[f"teleop/{name}"][:], dtype=np.float64)
            for name in selected_tau_ext
        }
        cameras = tuple(
            CameraData(
                name=name,
                time_s=np.asarray(h5[f"cameras/{name}/timestamp_us"][:], dtype=np.int64),
                frames=np.asarray(h5[f"cameras/{name}/frames"][:], dtype=np.uint8),
            )
            for name in selected_cameras
        )

    _validate_episode_arrays(
        path,
        teleop_timestamp_us,
        ee_pose,
        cameras,
        tau_ext,
    )
    origin_us = int(teleop_timestamp_us[0])
    normalized_cameras = tuple(
        CameraData(
            name=camera.name,
            time_s=(camera.time_s - origin_us).astype(np.float64) * 1.0e-6,
            frames=camera.frames,
        )
        for camera in cameras
    )
    return EpisodeData(
        path=path,
        episode_index=episode_index,
        teleop_time_s=(teleop_timestamp_us - origin_us).astype(np.float64) * 1.0e-6,
        cameras=normalized_cameras,
        ee_pose=ee_pose,
        tau_ext=tau_ext,
    )


def log_episode(data: EpisodeData, save_path: Path | None) -> None:
    rr, rrb = _import_rerun()
    blueprint = _make_blueprint(rrb, data.camera_names)
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

    for index, (name, values) in enumerate(data.tau_ext.items()):
        entity_path = f"tau_ext/{name}/norm"
        norm = np.linalg.norm(values, axis=1)
        color = TAU_EXT_COLORS[index % len(TAU_EXT_COLORS)]
        rr.log(
            entity_path,
            rr.SeriesLines(
                colors=[color],
                names=f"||{name}|| [N.m]",
                widths=[2.0],
            ),
            static=True,
        )
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn(TIMELINE, duration=data.teleop_time_s)],
            columns=rr.Scalars.columns(scalars=norm),
        )

    for camera in data.cameras:
        image_path = f"camera/{camera.name}/image"
        for timestamp_s, frame in zip(camera.time_s, camera.frames):
            rr.set_time(TIMELINE, duration=float(timestamp_s))
            rr.log(image_path, rr.Image(frame, color_model="RGB"))
    rr.disconnect()


def _make_blueprint(rrb, camera_names: tuple[str, ...]):
    camera_views = [
        rrb.Spatial2DView(
            origin=f"/camera/{name}",
            name=f"Camera: {name}",
        )
        for name in camera_names
    ]
    camera_panel = (
        camera_views[0]
        if len(camera_views) == 1
        else rrb.Vertical(*camera_views, row_shares=[1.0] * len(camera_views))
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            camera_panel,
            rrb.Spatial3DView(
                origin="/trajectory",
                name="End-effector trajectory",
            ),
            rrb.TimeSeriesView(
                origin="/tau_ext",
                name="||tau_ext|| [N.m]",
                plot_legend=rrb.PlotLegend(visible=True),
            ),
            column_shares=[1.0, 1.2, 1.1],
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


def _select_cameras(
    requested: str | None,
    available: list[str],
    path: Path,
) -> tuple[str, ...]:
    if not available:
        raise RuntimeError(f"Episode {path.name} has no complete camera streams")
    if requested is not None:
        requested_names = tuple(name for name in requested.split("/") if name)
        if not requested_names:
            raise RuntimeError("Camera selection cannot be empty; use --cam name[/name...]")
        unknown = [name for name in requested_names if name not in available]
        if unknown:
            raise RuntimeError(
                f"Camera(s) {unknown!r} are not in {path.name}; available={available}"
            )
        if len(set(requested_names)) != len(requested_names):
            raise RuntimeError(f"Camera selection contains duplicates: {requested!r}")
        return requested_names
    return tuple(available)


def _select_camera(requested: str | None, available: list[str], path: Path) -> str:
    """Compatibility wrapper returning the first selected camera."""

    return _select_cameras(requested, available, path)[0]


def _select_tau_ext(requested: str, teleop, path: Path) -> tuple[str, ...]:
    available = [name for name in TAU_EXT_DATASETS if name in teleop]
    if requested == "auto":
        preferred = tuple(name for name in ("tau_ext_cal", "tau_ext_pred") if name in teleop)
        if preferred:
            return preferred
        if available:
            return (available[0],)
        raise RuntimeError(
            f"Episode {path.name} has no tau_ext dataset; "
            f"expected one of {[f'teleop/{name}' for name in TAU_EXT_DATASETS]}"
        )
    if requested not in available:
        raise RuntimeError(
            f"tau_ext source {requested!r} is not in {path.name}; "
            f"available={available}"
        )
    return (requested,)


def _validate_episode_arrays(
    path: Path,
    timestamp_us: np.ndarray,
    ee_pose: np.ndarray,
    cameras: tuple[CameraData, ...],
    tau_ext: dict[str, np.ndarray],
) -> None:
    sample_count = timestamp_us.size
    expected = {
        "teleop/timestamp_us": (sample_count,),
        "teleop/ee_pose_follower": (sample_count, 4, 4),
    }
    actual = {
        "teleop/timestamp_us": timestamp_us.shape,
        "teleop/ee_pose_follower": ee_pose.shape,
    }
    for name, values in tau_ext.items():
        expected[f"teleop/{name}"] = (sample_count, 7)
        actual[f"teleop/{name}"] = values.shape
    invalid = [
        f"{name}: expected {expected[name]}, got {shape}"
        for name, shape in actual.items()
        if shape != expected[name]
    ]
    if sample_count == 0:
        invalid.append("teleop timeline is empty")
    for camera in cameras:
        if camera.time_s.shape != (camera.frames.shape[0],):
            invalid.append(
                f"camera {camera.name} timestamp/frame mismatch: "
                f"{camera.time_s.shape} versus {camera.frames.shape}"
            )
        if camera.frames.ndim != 4 or camera.frames.shape[-1] not in (3, 4):
            invalid.append(
                f"camera {camera.name} frames must be NxHxWx3/4, got {camera.frames.shape}"
            )
    if invalid:
        raise RuntimeError(f"Invalid episode {path.name}: " + "; ".join(invalid))
    for name, values in (
        ("teleop/ee_pose_follower", ee_pose),
        *[(f"teleop/{name}", values) for name, values in tau_ext.items()],
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"Episode {path.name} contains non-finite {name} values")
    if np.any(np.diff(timestamp_us) <= 0):
        raise RuntimeError(f"Episode {path.name} teleop timestamps are not strictly increasing")
    for camera in cameras:
        if camera.time_s.size == 0 or np.any(np.diff(camera.time_s) <= 0):
            raise RuntimeError(
                f"Episode {path.name} camera {camera.name} timestamps are empty or not increasing"
            )


def main() -> int:
    args = parse_args()
    episode_path = resolve_episode(args.runs_dir, args.episode)
    data = load_episode(
        episode_path,
        args.episode,
        args.camera,
        args.tau_ext_source,
    )
    camera_summary = ", ".join(
        f"{camera.name}:{camera.time_s.size}" for camera in data.cameras
    )
    print(
        f"Loading {episode_path.name}: teleop={data.teleop_time_s.size}, "
        f"cameras={camera_summary}"
    )
    print(
        "External torque source(s): "
        + ", ".join(f"teleop/{name}" for name in data.tau_ext_names)
    )
    log_episode(data, args.save)
    if args.save is not None:
        print(f"Saved Rerun recording: {args.save.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
