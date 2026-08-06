from typing import Dict, List, Mapping, Optional, Sequence

import copy
import logging as log
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusion_policy.common.lerobot_v3_io import LeRobotV3Dataset
from diffusion_policy.common.memory_budget import (
    compute_effective_budget_bytes,
    estimate_array_nbytes,
    format_gb,
)

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import create_indices, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.image_result_cache import (
    build_cache_metadata,
    open_or_build_image_result_cache,
    read_image_result,
    use_disk_result_cache,
)
from diffusion_policy.dataset.img_randomer import Image_randomer

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer,  get_image_identity_normalizer


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_torch_from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def _stack_fixed_shape(values, dtype):
    arrays = []
    for value in values:
        arr = np.asarray(_to_numpy(value), dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def _nearest_indices(sorted_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_timestamps, target_timestamps, side="left")
    right = np.clip(right, 0, len(sorted_timestamps) - 1)
    left = np.clip(right - 1, 0, len(sorted_timestamps) - 1)
    choose_right = np.abs(sorted_timestamps[right] - target_timestamps) < np.abs(
        sorted_timestamps[left] - target_timestamps
    )
    return np.where(choose_right, right, left)


def _coerce_image(image_value, expected_shape: Sequence[int]) -> np.ndarray:
    image_np = _to_numpy(image_value)
    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np[0]
    if image_np.ndim != 3:
        raise ValueError(f"Expected 3-D image tensor, got shape {image_np.shape}")

    # Normalize to HWC before resize.
    if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
        image_hwc = np.transpose(image_np, (1, 2, 0))
    else:
        image_hwc = image_np

    target_h, target_w = expected_shape[1], expected_shape[2]
    if image_hwc.shape[0] != target_h or image_hwc.shape[1] != target_w:
        image_hwc = cv2.resize(image_hwc, (target_w, target_h))

    image_chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
    image_max = float(image_chw.max()) if image_chw.size else 0.0
    if image_max > 1.0:
        image_chw /= 255.0
    if image_chw.shape != tuple(expected_shape):
        raise ValueError(
            f"Image shape mismatch. Expected {tuple(expected_shape)}, got {image_chw.shape}"
        )
    return image_chw


class HirolLeRobotV3Dataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        window_sampling_strategy: str = "idx",
        image_feature_map: Optional[Mapping[str, str]] = None,
        lowdim_feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
        action_feature_fields: Optional[Sequence[str]] = None,
        action_layout: str = "per_step",
        timestamp_key: str = "timestamp",
        timestamp_step_sec: Optional[float] = None,
        timestamp_tolerance_sec: Optional[float] = None,
        local_files_only: bool = True,
        preload_images: bool = False,
        memory_limit_gb: Optional[float] = None,
        memory_reserve_gb: float = 2.0,
        load_result_add="ram",
        image_randomer_config: Optional[Mapping] = None,
    ):
        super().__init__()
        if window_sampling_strategy not in {"idx", "timestamp"}:
            raise ValueError(
                f"Unsupported window_sampling_strategy={window_sampling_strategy!r}. "
                "Expected 'idx' or 'timestamp'."
            )
        if action_layout not in {"per_step", "prechunked"}:
            raise ValueError(
                f"Unsupported action_layout={action_layout!r}. "
                "Expected 'per_step' or 'prechunked'."
            )

        self.shape_meta = shape_meta
        self.dataset_path = os.path.expanduser(dataset_path)
        self.horizon = int(horizon)
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = int(n_latency_steps)
        self.action_layout = action_layout
        self.window_sampling_strategy = window_sampling_strategy
        self.timestamp_key = timestamp_key
        self.timestamp_step_sec = timestamp_step_sec
        self.timestamp_tolerance_sec = timestamp_tolerance_sec
        if self.action_layout == "prechunked":
            self.sequence_length = int(n_obs_steps or 1)
            self.sampler_pad_after = 0
        else:
            self.sequence_length = self.horizon + self.n_latency_steps
            self.sampler_pad_after = self.pad_after
        self.anchor_position = max(0, min(self.sequence_length - 1, (n_obs_steps or 1) - 1))
        self.image_data: Dict[str, np.ndarray] = {}
        self.load_result_add = load_result_add
        self.load_result_cache_path = None
        load_result_on_disk = use_disk_result_cache(load_result_add)
        self.image_randomer_config = image_randomer_config
        self.image_randomer = (
            Image_randomer(dict(image_randomer_config))
            if image_randomer_config is not None
            else None
        )


        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = [key for key, attr in obs_shape_meta.items() if attr.get("type") == "rgb"]
        self.lowdim_keys = [key for key, attr in obs_shape_meta.items() if attr.get("type") == "low_dim"]

        self.image_feature_map = dict(image_feature_map or {})
        for key in self.rgb_keys:
            self.image_feature_map.setdefault(key, f"observation.images.{key}")

        self.lowdim_feature_groups = {
            key: list(values)
            for key, values in (lowdim_feature_groups or {}).items()
        }
        for key in self.lowdim_keys:
            self.lowdim_feature_groups.setdefault(key, [f"observation.{key}"])

        self.action_feature_fields = list(action_feature_fields or ["action"])

        self.lerobot_dataset = LeRobotV3Dataset(
            self.dataset_path,
            local_files_only=local_files_only,
        )
        self.dataset_length = len(self.lerobot_dataset)

        self.timestamps = self._load_column(self.timestamp_key, dtype=np.float64).reshape(-1)
        self.episode_index = self._load_episode_index()
        self.episode_ends = self._build_episode_ends(self.episode_index)
        self.episode_ranges = self._build_episode_ranges(self.episode_ends)
        self.episode_step_sec = self._build_episode_step_sec(
            self.timestamps,
            self.episode_ranges,
            explicit_step_sec=timestamp_step_sec,
        )

        self.lowdim_data = {
            key: self._concat_columns(self.lowdim_feature_groups[key], dtype=np.float32)
            for key in self.lowdim_keys
        }
        self.action_data = self._concat_columns(self.action_feature_fields, dtype=np.float32)

        for key in self.lowdim_keys:
            expected = tuple(obs_shape_meta[key]["shape"])
            if self.lowdim_data[key].shape[1:] != expected:
                raise ValueError(
                    f"Lowdim feature {key!r} has shape {self.lowdim_data[key].shape[1:]}, "
                    f"expected {expected}. Source fields: {self.lowdim_feature_groups[key]}"
                )

        expected_action_shape = tuple(shape_meta["action"]["shape"])
        self._validate_action_shape(expected_action_shape)

        effective_budget_bytes = compute_effective_budget_bytes(
            memory_limit_gb=memory_limit_gb,
            memory_reserve_gb=memory_reserve_gb,
        )
        estimated_preload_bytes = self._estimate_image_preload_bytes()
        if effective_budget_bytes is not None:
            log.info(
                "HirolLeRobotV3Dataset RAM budget: effective=%s, estimated image-preload=%s",
                format_gb(effective_budget_bytes),
                format_gb(estimated_preload_bytes),
            )
            if preload_images and (not load_result_on_disk) and estimated_preload_bytes > effective_budget_bytes:
                log.warning(
                    "Disabling LeRobot image preload because estimated footprint %s exceeds RAM budget %s.",
                    format_gb(estimated_preload_bytes),
                    format_gb(effective_budget_bytes),
                )
                preload_images = False

        if load_result_on_disk:
            image_shapes = {
                key: tuple(self.shape_meta["obs"][key]["shape"])
                for key in self.rgb_keys
            }
            metadata = build_cache_metadata(
                source_type="hirol_lerobot_v3",
                dataset_path=dataset_path,
                dataset_length=self.dataset_length,
                rgb_keys=self.rgb_keys,
                image_shapes=image_shapes,
                extra={
                    "image_feature_map": self.image_feature_map,
                },
            )

            def build_frame(frame_idx):
                sample = self.lerobot_dataset[frame_idx]
                frame_data = {}
                for key in self.rgb_keys:
                    feature_name = self.image_feature_map[key]
                    if feature_name not in sample:
                        raise KeyError(
                            f"Feature {feature_name!r} missing from LeRobot sample. "
                            f"Available keys: {list(sample.keys())}"
                        )
                    frame_data[key] = _coerce_image(sample[feature_name], image_shapes[key])
                return frame_data

            self.image_data, self.load_result_cache_path = open_or_build_image_result_cache(
                load_result_add=load_result_add,
                dataset_path=dataset_path,
                metadata=metadata,
                build_frame_fn=build_frame,
                desc="Build LeRobot decoded image cache",
                logger=log,
            )
            self.lerobot_dataset.close()
        elif preload_images:
            self.image_data = self._preload_images()
            self.lerobot_dataset.close()

        val_mask = get_val_mask(
            n_episodes=len(self.episode_ends),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.val_mask = val_mask
        self.train_mask = train_mask
        self.indices = create_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.sampler_pad_after,
            episode_mask=self.train_mask,
        )

    def _get_hf_dataset(self):
        return None

    def _load_column(self, column_name: str, dtype) -> np.ndarray:
        try:
            values = self.lerobot_dataset.get_column(column_name)
        except KeyError as exc:
            raise KeyError(f"Column {column_name!r} not found in LeRobot v3 dataset.") from exc
        return _stack_fixed_shape(values, dtype=dtype)

    def _load_episode_index(self) -> np.ndarray:
        episode_data_index = getattr(self.lerobot_dataset, "episode_data_index", None)
        if episode_data_index is not None and "from" in episode_data_index and "to" in episode_data_index:
            starts = _to_numpy(episode_data_index["from"]).astype(np.int64).reshape(-1)
            stops = _to_numpy(episode_data_index["to"]).astype(np.int64).reshape(-1)
            episode_index = np.empty((self.dataset_length,), dtype=np.int64)
            for ep_idx, (start, stop) in enumerate(zip(starts, stops)):
                episode_index[start:stop] = ep_idx
            return episode_index

        raise KeyError(
            "LeRobot v3 dataset does not expose episode_index or episode_data_index; "
            "cannot build episode-aware window sampling."
        )

    def _concat_columns(self, column_names: Sequence[str], dtype) -> np.ndarray:
        arrays = [self._load_column(column_name, dtype=dtype) for column_name in column_names]
        if len(arrays) == 1:
            return arrays[0].astype(dtype, copy=False)
        return np.concatenate(arrays, axis=-1).astype(dtype, copy=False)

    def _validate_action_shape(self, expected_action_shape: Sequence[int]) -> None:
        expected_action_shape = tuple(expected_action_shape)
        source_shape = self.action_data.shape[1:]

        if self.action_layout == "per_step":
            if source_shape != expected_action_shape:
                raise ValueError(
                    f"Per-step action shape mismatch. Got {source_shape}, "
                    f"expected {expected_action_shape}. "
                    f"Source fields: {self.action_feature_fields}"
                )
            return

        required_chunk_steps = self.n_latency_steps + self.horizon
        if len(source_shape) < 1 or source_shape[1:] != expected_action_shape:
            raise ValueError(
                f"Prechunked action shape mismatch. Got {source_shape}, expected "
                f"(chunk_steps, {', '.join(map(str, expected_action_shape))}). "
                f"Source fields: {self.action_feature_fields}"
            )
        if source_shape[0] < required_chunk_steps:
            raise ValueError(
                f"Prechunked action has {source_shape[0]} steps, but horizon={self.horizon} "
                f"and n_latency_steps={self.n_latency_steps} require at least "
                f"{required_chunk_steps}."
            )

    def _sample_action(self, sequence_indices: np.ndarray) -> np.ndarray:
        if self.action_layout == "per_step":
            action = self.action_data[sequence_indices, ...]
            if self.n_latency_steps > 0:
                action = action[self.n_latency_steps :]
        else:
            anchor_idx = int(sequence_indices[self.anchor_position])
            chunk_start = self.n_latency_steps
            chunk_end = chunk_start + self.horizon
            action = self.action_data[anchor_idx, chunk_start:chunk_end, ...]

        expected_shape = (self.horizon,) + tuple(self.shape_meta["action"]["shape"])
        if action.shape != expected_shape:
            raise RuntimeError(
                f"Sampled action shape mismatch. Got {action.shape}, expected {expected_shape}. "
                f"action_layout={self.action_layout!r}"
            )
        return np.array(action, dtype=np.float32, copy=True)

    def _estimate_image_preload_bytes(self) -> int:
        total = 0
        for key in self.rgb_keys:
            expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
            total += self.dataset_length * estimate_array_nbytes(expected_shape, np.float32)
        return total

    def _preload_images(self) -> Dict[str, np.ndarray]:
        image_data = {
            key: np.empty(
                (self.dataset_length,) + tuple(self.shape_meta["obs"][key]["shape"]),
                dtype=np.float32,
            )
            for key in self.rgb_keys
        }
        log.info(
            "Preloading %d LeRobot frames for %d RGB keys into memory...",
            self.dataset_length,
            len(self.rgb_keys),
        )
        for frame_idx in tqdm(range(self.dataset_length), desc="Preload LeRobot images"):
            sample = self.lerobot_dataset[frame_idx]
            for key in self.rgb_keys:
                feature_name = self.image_feature_map[key]
                if feature_name not in sample:
                    raise KeyError(
                        f"Feature {feature_name!r} missing from LeRobot sample. "
                        f"Available keys: {list(sample.keys())}"
                    )
                expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
                image_data[key][frame_idx] = _coerce_image(sample[feature_name], expected_shape)
        log.info("Finished LeRobot image preload.")
        return image_data

    @staticmethod
    def _build_episode_ends(episode_index: np.ndarray) -> np.ndarray:
        if episode_index.size == 0:
            return np.zeros((0,), dtype=np.int64)
        change_points = np.nonzero(np.diff(episode_index))[0] + 1
        return np.concatenate([change_points, [episode_index.shape[0]]]).astype(np.int64)

    @staticmethod
    def _build_episode_ranges(episode_ends: np.ndarray) -> List[range]:
        episode_ranges: List[range] = []
        start = 0
        for end in episode_ends:
            episode_ranges.append(range(start, int(end)))
            start = int(end)
        return episode_ranges

    @staticmethod
    def _build_episode_step_sec(
        timestamps: np.ndarray,
        episode_ranges: Sequence[range],
        explicit_step_sec: Optional[float],
    ) -> List[float]:
        if explicit_step_sec is not None:
            return [float(explicit_step_sec) for _ in episode_ranges]

        step_sizes = []
        for episode_range in episode_ranges:
            episode_timestamps = timestamps[episode_range.start : episode_range.stop]
            diffs = np.diff(episode_timestamps)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size == 0:
                step_sizes.append(1.0)
            else:
                step_sizes.append(float(np.median(diffs)))
        return step_sizes

    def _sample_indices_to_sequence(self, sample_idx: int) -> np.ndarray:
        # buffer数据全局   sample进行窗口采样
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[sample_idx]
        sequence_indices = np.empty((self.sequence_length,), dtype=np.int64)
        last_valid_idx = max(buffer_start_idx, buffer_end_idx - 1)
        for position in range(self.sequence_length):
            if position < sample_start_idx:
                sequence_indices[position] = buffer_start_idx
            elif position >= sample_end_idx:
                sequence_indices[position] = last_valid_idx
            else:
                sequence_indices[position] = buffer_start_idx + (position - sample_start_idx)
        return sequence_indices

    def _retime_sequence_indices(self, sequence_indices: np.ndarray) -> np.ndarray:
        anchor_global_idx = int(sequence_indices[self.anchor_position])
        anchor_episode_idx = int(self.episode_index[anchor_global_idx])
        episode_range = self.episode_ranges[anchor_episode_idx]
        episode_timestamps = self.timestamps[episode_range.start : episode_range.stop]
        anchor_timestamp = float(self.timestamps[anchor_global_idx])
        step_sec = self.episode_step_sec[anchor_episode_idx]

        target_timestamps = anchor_timestamp + (
            np.arange(self.sequence_length, dtype=np.float64) - self.anchor_position
        ) * step_sec
        episode_local_indices = _nearest_indices(episode_timestamps, target_timestamps)
        if self.timestamp_tolerance_sec is not None:
            deltas = np.abs(episode_timestamps[episode_local_indices] - target_timestamps)
            nearest_edge = np.where(target_timestamps <= episode_timestamps[0], 0, len(episode_timestamps) - 1)
            episode_local_indices = np.where(
                deltas <= self.timestamp_tolerance_sec,
                episode_local_indices,
                nearest_edge,
            )
        return episode_range.start + episode_local_indices.astype(np.int64)

    def _load_frame_feature(
        self,
        frame_idx: int,
        feature_name: str,
        expected_shape: Sequence[int],
        frame_cache: Dict[int, Dict],
    ) -> np.ndarray:
        if frame_idx not in frame_cache:
            frame_cache[frame_idx] = self.lerobot_dataset[frame_idx]
        sample = frame_cache[frame_idx]
        if feature_name not in sample:
            raise KeyError(
                f"Feature {feature_name!r} missing from LeRobot sample. "
                f"Available keys: {list(sample.keys())}"
            )
        return _coerce_image(sample[feature_name], expected_shape)

    def _augment_image_sequence(self, images: np.ndarray) -> np.ndarray:
        if self.image_randomer is None:
            return images
    
        augmented_images = []
        for image_chw in images:
            image_hwc = np.transpose(image_chw, (1, 2, 0))
            image_hwc = np.clip(image_hwc * 255.0, 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(image_hwc)
    
            augmented = self.image_randomer(image_pil)
            if torch.is_tensor(augmented):
                augmented = augmented.detach().cpu().numpy()
            augmented = np.asarray(augmented, dtype=np.float32)
    
            augmented_images.append(augmented)
    
        return np.stack(augmented_images, axis=0)

    def get_validation_dataset(self) -> "HirolLeRobotV3Dataset":
        val_set = copy.copy(self)
        val_set.indices = create_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.sampler_pad_after,
            episode_mask=self.val_mask,
        )
        val_set.image_randomer = None
        val_set.image_randomer_config = None
        return val_set


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.action_data,**kwargs)
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.lowdim_data[key],**kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer() 
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.action_data.copy())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence_indices = self._sample_indices_to_sequence(idx)
        if self.window_sampling_strategy == "timestamp":
            sequence_indices = self._retime_sequence_indices(sequence_indices)
          
        obs_indices = sequence_indices[: self.n_obs_steps]
        frame_cache: Dict[int, Dict] = {}
        obs_dict = {}

        for key in self.rgb_keys:
            if key in self.image_data:
                images = read_image_result(self.image_data, key, obs_indices)
            else:
                expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
                feature_name = self.image_feature_map[key]
                images = np.stack(
                    [
                        self._load_frame_feature(
                            frame_idx=int(frame_idx),
                            feature_name=feature_name,
                            expected_shape=expected_shape,
                            frame_cache=frame_cache,
                        )
                        for frame_idx in obs_indices
                    ],
                    axis=0,
                )

            obs_dict[key] = self._augment_image_sequence(images)


        for key in self.lowdim_keys:
            obs_dict[key] = self.lowdim_data[key][obs_indices, ...].astype(np.float32, copy=False)

        action = self._sample_action(sequence_indices)

        return {
            "obs": dict_apply(obs_dict, _safe_torch_from_numpy),
            "action": _safe_torch_from_numpy(action),
        }
# 一个torch
# batch = 
#     "obs": 
#         "ee_cam_color": Tensor[B, 2, 3, 224, 224],
#         "third_person_cam_color": Tensor[B, 2, 3, 224, 224],
#         "side_cam_color": Tensor[B, 2, 3, 224, 224],
#         "state_ee": Tensor[B, 2, 15]
#     "action": Tensor[B, 16, 8],
