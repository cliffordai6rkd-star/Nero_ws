#!/usr/bin/env python3

import time
import os
import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler

def test_storage_performance(dataset_path, num_samples=100):
    """Compare disk vs memory storage performance"""

    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return

    print("=" * 60)
    print("PERFORMANCE TEST: Disk vs Memory Storage")
    print("=" * 60)

    # Test 1: Disk-based storage (original)
    print("\n1. Testing DISK-based storage...")
    start_time = time.time()
    disk_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    disk_load_time = time.time() - start_time
    print(f"   Disk buffer creation time: {disk_load_time:.3f}s")

    # Create sampler for disk buffer
    disk_sampler = SequenceSampler(
        replay_buffer=disk_buffer,
        sequence_length=16,
        keys=['state', 'action']  # Use minimal keys for testing
    )

    # Test disk sampling performance
    print(f"   Testing {num_samples} samples from disk...")
    disk_sample_times = []
    for i in range(num_samples):
        start_time = time.time()
        _ = disk_sampler.sample_sequence(i)
        sample_time = time.time() - start_time
        disk_sample_times.append(sample_time)

    disk_avg_time = np.mean(disk_sample_times)
    print(f"   Disk sampling - Avg: {disk_avg_time:.6f}s, Total: {sum(disk_sample_times):.3f}s")

    # Test 2: Memory-based storage (optimized)
    print("\n2. Testing MEMORY-based storage...")
    start_time = time.time()
    memory_buffer = ReplayBuffer.copy_from_store(
        src_store=zarr.DirectoryStore(dataset_path),
        store=zarr.MemoryStore()
    )
    memory_load_time = time.time() - start_time
    print(f"   Memory buffer creation time: {memory_load_time:.3f}s")

    # Create sampler for memory buffer
    memory_sampler = SequenceSampler(
        replay_buffer=memory_buffer,
        sequence_length=16,
        keys=['state', 'action']  # Use minimal keys for testing
    )

    # Test memory sampling performance
    print(f"   Testing {num_samples} samples from memory...")
    memory_sample_times = []
    for i in range(num_samples):
        start_time = time.time()
        _ = memory_sampler.sample_sequence(i)
        sample_time = time.time() - start_time
        memory_sample_times.append(sample_time)

    memory_avg_time = np.mean(memory_sample_times)
    print(f"   Memory sampling - Avg: {memory_avg_time:.6f}s, Total: {sum(memory_sample_times):.3f}s")

    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    speedup = disk_avg_time / memory_avg_time if memory_avg_time > 0 else float('inf')

    print(f"Dataset size: {len(disk_sampler)} samples")
    print(f"Buffer creation - Disk: {disk_load_time:.3f}s, Memory: {memory_load_time:.3f}s")
    print(f"Avg sampling time - Disk: {disk_avg_time:.6f}s, Memory: {memory_avg_time:.6f}s")
    print(f"Speedup: {speedup:.2f}x faster with memory storage")

    if speedup > 1:
        print(f"✅ Memory storage is {speedup:.2f}x faster for data sampling!")
    else:
        print(f"❌ Memory storage is not faster (might be due to small dataset or other factors)")

    # Estimate total training speedup
    total_disk_time = sum(disk_sample_times) + disk_load_time
    total_memory_time = sum(memory_sample_times) + memory_load_time
    total_speedup = total_disk_time / total_memory_time
    print(f"Overall speedup (including loading): {total_speedup:.2f}x")

if __name__ == "__main__":
    # Update this path to your actual dataset
    dataset_path = "/home/zyx/dataset/dp/fr3/0920/water_pouring_1_step_0_skip_abs_jps.zarr"

    if not os.path.exists(dataset_path):
        print("Please update the dataset_path variable in the script to point to your actual dataset")
        print("Current path:", dataset_path)
        exit(1)

    test_storage_performance(dataset_path, num_samples=50)