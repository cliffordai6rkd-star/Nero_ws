#!/usr/bin/env python3

import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import torch
from torch.utils.data import DataLoader
from diffusion_policy.dataset.hirol_lerobot_v3_dataset import  HirolLeRobotV3Dataset

def test_batch_performance():
    # Dataset configuration
    # image_shape = [3, 480, 640]
    image_shape = [3,224,224]
    shape_meta = {
        'obs': {
            'state': {
                'shape': [15],
                'type': 'low_dim'
            },
            'ee_cam_color': {
                'shape': image_shape,
                'type': 'rgb'
            },
            'third_person_cam_color': {
                'shape': image_shape,
                'type': 'rgb'
            },
            'side_cam_color': {
                'shape': image_shape,
                'type': 'rgb'
            },
        },
        'action': {
            'shape': [15]
        }
    }

    dataset_path = "data/train_episode/moving_bread/moving_bread_hirol_lerobotv3"

    print("Creating HirolLobotv3Dataset...")
    dataset = HirolLeRobotV3Dataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        n_obs_steps=3,
        val_ratio=0.1
    )

    print(f"Dataset length: {len(dataset)}")

    # Test different DataLoader configurations
    configurations = [
        {"batch_size": 128, "num_workers": 2, "prefetch_factor": 4},
        {"batch_size": 128, "num_workers": 4, "prefetch_factor": 8},
        {"batch_size": 128, "num_workers": 8, "prefetch_factor": 8},
        {"batch_size": 64, "num_workers": 8, "prefetch_factor": 8},
    ]

    for i, config in enumerate(configurations):
        print(f"\n=== Testing Configuration {i+1} ===")
        print(f"Config: {config}")

        dataloader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            **config
        )

        # Warm up
        print("Warming up...")
        for j, batch in enumerate(dataloader):
            if j >= 2:  # Just 2 batches for warmup
                break

        # Measure performance
        print("Measuring batch loading performance...")
        batch_times = []

        start_time = time.perf_counter()
        for j, batch in enumerate(dataloader):
            batch_end_time = time.perf_counter()
            if j > 0:  # Skip first batch (includes initialization overhead)
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                print(f"Batch {j}: {batch_time:.3f}s")

            batch_start_time = time.perf_counter()

            if j >= 10:  # Test 10 batches
                break

        # Statistics
        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            min_time = min(batch_times)
            max_time = max(batch_times)
            print(f"\nResults for config {i+1}:")
            print(f"Average batch time: {avg_time:.3f}s")
            print(f"Min batch time: {min_time:.3f}s")
            print(f"Max batch time: {max_time:.3f}s")
            print(f"Batches under 0.5s: {sum(1 for t in batch_times if t < 0.5)}/{len(batch_times)}")

if __name__ == "__main__":
    test_batch_performance()