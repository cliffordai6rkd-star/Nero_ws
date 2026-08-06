import os
import inspect
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

OFFICIAL_AUTO_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _import_lerobot_dataset():
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                raise ImportError(
                    "Official LeRobot is required for LeRobot v3 IO. "
                    "Install a LeRobot release that provides lerobot.datasets.LeRobotDataset."
                ) from exc
    return LeRobotDataset


def _repo_id_from_root(root: Path, repo_id: Optional[str] = None) -> str:
    if repo_id:
        return repo_id
    return root.name or "local_lerobot_dataset"


def _feature_spec_for_lerobot(features: Mapping[str, Mapping], use_videos: bool) -> OrderedDict:
    converted = OrderedDict()
    for key, spec in features.items():
        if key in OFFICIAL_AUTO_FEATURES:
            continue
        spec_dict = dict(spec)
        if spec_dict.get("dtype") == "video" and not use_videos:
            spec_dict["dtype"] = "image"
        if "video_info" in spec_dict and "info" not in spec_dict:
            spec_dict["info"] = spec_dict.pop("video_info")
        converted[key] = spec_dict
    return converted


def _scalar_value(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar-compatible value, got shape {array.shape}.")
    return array.reshape(-1)[0].item()


class LeRobotV3Writer:
    """Small project wrapper around official lerobot.datasets.LeRobotDataset.create."""

    def __init__(
        self,
        root: str,
        fps: int,
        features: Mapping[str, Mapping],
        video_keys: Sequence[str],
        robot_type: str = "unknown",
        codec: str = "h264",
        image_color_space: str = "rgb",
        repo_id: Optional[str] = None,
    ):
        LeRobotDataset = _import_lerobot_dataset()
        self.root = _path(root)
        self.image_color_space = str(image_color_space).lower()
        if self.image_color_space not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported image_color_space={image_color_space!r}")

        if self.root.exists() and not any(self.root.iterdir()):
            self.root.rmdir()
        elif self.root.exists():
            raise FileExistsError(
                f"Output directory already exists and is not empty: {self.root}. "
                "Choose a new --output-dir or remove the old converted dataset first."
            )

        use_videos = bool(video_keys)
        self.features = _feature_spec_for_lerobot(features, use_videos=use_videos)
        vcodec = "h264" if codec in {"mp4v", "avc1"} else codec
        create_kwargs = dict(
            repo_id=_repo_id_from_root(self.root, repo_id),
            root=self.root,
            fps=int(fps),
            features=self.features,
            robot_type=str(robot_type or "unknown"),
            use_videos=use_videos,
            vcodec=vcodec,
        )
        create_params = inspect.signature(LeRobotDataset.create).parameters
        create_kwargs = {
            key: value
            for key, value in create_kwargs.items()
            if key in create_params
        }
        self.dataset = LeRobotDataset.create(**create_kwargs)

    def add_frame(self, frame: Mapping[str, object]) -> None:
        lerobot_frame = {}
        for key, value in frame.items():
            if key in OFFICIAL_AUTO_FEATURES or key not in self.features:
                continue
            spec = self.features[key]
            if spec.get("dtype") in {"video", "image"} and self.image_color_space == "bgr":
                value = cv2.cvtColor(np.asarray(value), cv2.COLOR_BGR2RGB)
            elif tuple(spec.get("shape", ())) == (1,):
                value = _scalar_value(value)
            lerobot_frame[key] = value

        if "next.done" in self.features and "next.done" not in lerobot_frame:
            lerobot_frame["next.done"] = False
        lerobot_frame.setdefault("task", "")
        self.dataset.add_frame(lerobot_frame)

    def save_episode(self, task: Optional[str] = None) -> None:
        episode_buffer = getattr(self.dataset, "episode_buffer", None)
        if episode_buffer is not None and episode_buffer.get("size", 0) > 0:
            if "next.done" in episode_buffer:
                episode_buffer["next.done"][-1] = True
            if task is not None and "task" in episode_buffer:
                episode_buffer["task"] = [task for _ in episode_buffer["task"]]

        save_params = inspect.signature(self.dataset.save_episode).parameters
        if task is not None and "task" in save_params:
            self.dataset.save_episode(task=task)
        else:
            self.dataset.save_episode()

    def finalize(self) -> None:
        self.dataset.finalize()


class LeRobotV3Dataset:
    """Read LeRobot v3 datasets through the official LeRobotDataset API."""

    def __init__(
        self,
        root: str,
        repo_id: Optional[str] = None,
        local_files_only: bool = True,
        video_backend: Optional[str] = None,
    ):
        LeRobotDataset = _import_lerobot_dataset()
        self.root = _path(root)
        if local_files_only:
            missing = [
                rel_path
                for rel_path in ("meta/info.json", "data")
                if not (self.root / rel_path).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Local LeRobot dataset is incomplete at {self.root}. Missing: {missing}"
                )

        self.dataset = LeRobotDataset(
            repo_id=_repo_id_from_root(self.root, repo_id),
            root=self.root,
            video_backend=video_backend,
            download_videos=True,
        )
        self.info = getattr(self.dataset.meta, "info", None)
        self.features = OrderedDict((key, dict(value)) for key, value in self.dataset.features.items())
        self.video_keys = list(getattr(self.dataset.meta, "video_keys", []))
        self.fps = int(self.dataset.fps)
        self.length = len(self.dataset)
        self.episode_data_index = self._build_episode_data_index()
        self.episode_tasks = self._build_episode_tasks()

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        return None

    def _build_episode_data_index(self) -> Dict[str, List[int]]:
        episodes = getattr(self.dataset.meta, "episodes", None)
        if episodes is None:
            raise KeyError("LeRobot dataset metadata does not expose episodes.")
        starts = []
        stops = []
        for episode in episodes:
            if "dataset_from_index" not in episode or "dataset_to_index" not in episode:
                raise KeyError(
                    "Episode metadata is missing official dataset_from_index/dataset_to_index fields."
                )
            starts.append(int(episode["dataset_from_index"]))
            stops.append(int(episode["dataset_to_index"]))
        return {"from": starts, "to": stops}

    def _build_episode_tasks(self) -> List[str]:
        episodes = getattr(self.dataset.meta, "episodes", None)
        if episodes is None:
            return []
        return ["" if episode.get("task") is None else str(episode.get("task")) for episode in episodes]

    def get_column(self, name: str) -> List:
        hf_dataset = self.dataset.hf_dataset
        if name not in hf_dataset.column_names:
            raise KeyError(name)
        return hf_dataset[name]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.dataset[idx]
