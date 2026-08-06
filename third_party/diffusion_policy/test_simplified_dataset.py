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
from diffusion_policy.dataset.hirol_dataset import HirolDataset

def test_simplified_dataset():
    """Test the simplified dataset with direct memory access"""

    # Dataset configuration
    image_shape = [3, 480, 640]
    shape_meta = {
        'obs': {
            'state': {'shape': [8], 'type': 'low_dim'},
            'ee_cam_color': {'shape': image_shape, 'type': 'rgb'},
            'third_person_cam_color': {'shape': image_shape, 'type': 'rgb'},
            'side_cam_color': {'shape': image_shape, 'type': 'rgb'},
        },
        'action': {'shape': [8]}
    }

    dataset_path = "/home/zyx/dataset/dp/fr3/0920/water_pouring_1_step_0_skip_abs_jps.zarr"

    print("Creating simplified HirolDataset...")
    start_time = time.perf_counter()

    dataset = HirolDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        n_obs_steps=3,
        val_ratio=0.1
    )

    init_time = time.perf_counter() - start_time
    print(f"Dataset initialization: {init_time:.2f}s")
    print(f"Dataset size: {len(dataset)}")

    # Test single sample performance
    print("\nTesting single sample performance...")
    sample_times = []

    for i in range(10):
        start = time.perf_counter()
        sample = dataset[i]
        sample_time = time.perf_counter() - start
        sample_times.append(sample_time * 1000)  # Convert to ms

        if i == 0:
            # Check sample structure
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Obs keys: {list(sample['obs'].keys())}")
            print(f"Action shape: {sample['action'].shape}")
            for key in sample['obs']:
                print(f"Obs {key} shape: {sample['obs'][key].shape}")

    avg_sample_time = sum(sample_times) / len(sample_times)
    print(f"\nSingle sample performance:")
    print(f"Average time: {avg_sample_time:.2f}ms")
    print(f"Min time: {min(sample_times):.2f}ms")
    print(f"Max time: {max(sample_times):.2f}ms")

    # Test batch loading
    print("\nTesting batch loading...")
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    batch_times = []
    print("Warming up...")

    for i, batch in enumerate(dataloader):
        if i == 0:
            print("First batch loaded (warmup)")
            batch_start = time.perf_counter()
            continue

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        print(f"Batch {i}: {batch_time:.3f}s")

        if i >= 5:  # Test 5 batches
            break

        batch_start = time.perf_counter()

    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"\nBatch loading performance:")
        print(f"Average batch time: {avg_batch_time:.3f}s")
        print(f"Min batch time: {min(batch_times):.3f}s")
        print(f"Max batch time: {max(batch_times):.3f}s")
        print(f"Batches under 0.5s: {sum(1 for t in batch_times if t < 0.5)}/{len(batch_times)}")

        # Calculate theoretical vs actual
        theoretical_time = avg_sample_time * 128 / 1000  # Convert to seconds
        speedup = theoretical_time / avg_batch_time
        print(f"\nPerformance analysis:")
        print(f"Theoretical serial time: {theoretical_time:.3f}s")
        print(f"Actual parallel time: {avg_batch_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    test_simplified_dataset()