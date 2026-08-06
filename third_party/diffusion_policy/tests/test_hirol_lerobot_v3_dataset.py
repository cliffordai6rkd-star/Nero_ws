import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset


class _FakeLeRobotV3Dataset:
    columns = {}
    episode_data_index = {"from": [0], "to": [4]}

    def __init__(self, *args, **kwargs):
        self.episode_data_index = type(self).episode_data_index

    def __len__(self):
        return len(type(self).columns["timestamp"])

    def get_column(self, name):
        return type(self).columns[name]

    def close(self):
        return None


def _columns(action):
    return {
        "timestamp": [float(index) / 10.0 for index in range(4)],
        "observation.wrench_ext": [
            np.full((8, 6), index, dtype=np.float32)
            for index in range(4)
        ],
        "action.ee_pose": [value for value in action],
    }


def _shape_meta():
    return {
        "obs": {
            "wrench_ext": {
                "shape": [8, 6],
                "type": "low_dim",
            }
        },
        "action": {"shape": [7]},
    }


class HirolLeRobotV3DatasetActionLayoutTest(unittest.TestCase):
    def _create_dataset(self, action, *, action_layout, horizon, n_latency_steps=0):
        _FakeLeRobotV3Dataset.columns = _columns(action)
        with patch(
            "diffusion_policy.dataset.hirol_lerobot_v3_dataset.LeRobotV3Dataset",
            _FakeLeRobotV3Dataset,
        ):
            return HirolLeRobotV3Dataset(
                shape_meta=_shape_meta(),
                dataset_path="unused",
                horizon=horizon,
                pad_before=1,
                pad_after=max(0, horizon - 1),
                n_obs_steps=2,
                n_latency_steps=n_latency_steps,
                val_ratio=0.0,
                action_feature_fields=["action.ee_pose"],
                action_layout=action_layout,
                preload_images=False,
            )

    def test_prechunked_action_uses_last_observation_as_anchor(self):
        action = np.zeros((4, 8, 7), dtype=np.float32)
        for frame_idx in range(4):
            action[frame_idx, :, 0] = frame_idx * 10 + np.arange(8)

        dataset = self._create_dataset(
            action,
            action_layout="prechunked",
            horizon=8,
        )

        self.assertEqual(len(dataset), 4)
        first = dataset[0]
        last = dataset[3]
        self.assertEqual(tuple(first["obs"]["wrench_ext"].shape), (2, 8, 6))
        self.assertEqual(tuple(first["action"].shape), (8, 7))
        np.testing.assert_array_equal(first["action"][:, 0].numpy(), np.arange(8))
        np.testing.assert_array_equal(last["action"][:, 0].numpy(), 30 + np.arange(8))

        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(tuple(batch["obs"]["wrench_ext"].shape), (2, 2, 8, 6))
        self.assertEqual(tuple(batch["action"].shape), (2, 8, 7))

    def test_prechunked_action_applies_latency_inside_chunk(self):
        action = np.zeros((4, 5, 7), dtype=np.float32)
        action[:, :, 0] = np.arange(5)
        dataset = self._create_dataset(
            action,
            action_layout="prechunked",
            horizon=3,
            n_latency_steps=2,
        )

        sample = dataset[0]
        np.testing.assert_array_equal(sample["action"][:, 0].numpy(), [2, 3, 4])

    def test_per_step_action_behavior_is_unchanged(self):
        action = np.zeros((4, 7), dtype=np.float32)
        action[:, 0] = np.arange(4)
        dataset = self._create_dataset(
            action,
            action_layout="per_step",
            horizon=3,
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["action"].shape), (3, 7))
        np.testing.assert_array_equal(sample["action"][:, 0].numpy(), [0, 0, 1])

    def test_prechunked_action_rejects_short_chunk(self):
        action = np.zeros((4, 7, 7), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "require at least 8"):
            self._create_dataset(
                action,
                action_layout="prechunked",
                horizon=8,
            )


if __name__ == "__main__":
    unittest.main()
