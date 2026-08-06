from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer
import time
import logging as log

@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    # 构建采样窗口  滑动采样
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        obs_data = None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.obs_data = obs_data
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        n_data = buffer_end_idx - buffer_start_idx
        result = dict()
        for key in self.keys:
            shape = None
            if key not in self.key_first_k:
                # start2_time = time.perf_counter()
                shape = (self.sequence_length,) + self.replay_buffer[key].shape[1:]
                sample = self.replay_buffer[key][buffer_start_idx:buffer_end_idx]
                # print(f'sample k data time {time.perf_counter() - start2_time} for {key} with no key first')
                data = self._handle_sample_edge_keys(sample, key, sample_start_idx,
                    sample_end_idx, self.sequence_length,
                    shape, self.replay_buffer[key].dtype)
            else:
                obs_horizon = self.key_first_k[key]
                k_data = min(obs_horizon, n_data)
                shape = self.obs_data[key].shape[1:] if self.obs_data and key in self.obs_data else self.replay_buffer[key].shape[1:]
                shape = (obs_horizon,) + shape
                dtype = self.obs_data[key].dtype if self.obs_data and key in self.obs_data else self.replay_buffer[key].dtype
                try:
                    # start2_time = time.perf_counter()
                    if self.obs_data and key in self.obs_data:
                        sample = self.obs_data[key][buffer_start_idx:buffer_start_idx+k_data]
                        # print(f'key sample dtype: {sample.dtype}')
                    else: 
                        sample = self.replay_buffer[key][buffer_start_idx:buffer_start_idx+k_data]
                    # print(f'sample k data time {time.perf_counter() - start2_time} for {key}')
                except Exception as e:
                    import pdb; pdb.set_trace()
                data = self._handle_sample_edge_keys(sample, key, sample_start_idx,
                            sample_end_idx, obs_horizon, shape, dtype)
            assert data.shape == shape, f'{key} sample data dim wrong, expected: {shape} but get {data.shape}'
            result[key] = data
            # print(f'{key} used time: {time.perf_counter() - start_time}')
        return result

    def _handle_sample_edge_keys(self, sample, key, sample_start_idx, sample_end_idx, length, shape, dtype):
        data = sample
        # front/back padding
        if (sample_start_idx > 0) or (sample_end_idx < length):
            # log.info(f'edge case triggered for {key}, {sample_start_idx}. {sample_end_idx}. {length}')            
            data = np.zeros(shape=shape, dtype=dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < length:
                data[sample_end_idx:] = sample[-1]
            if sample_start_idx < length:
                sample_end_idx = length if sample_end_idx > length else sample_end_idx
                data[sample_start_idx:sample_end_idx] = sample[:(sample_end_idx-sample_start_idx)]
            # Ensure the array is contiguous to avoid PyTorch collation issues
            # data = np.ascontiguousarray(data)
            # Debug: print edge case handling
            # print(f"SAMPLER DEBUG: Edge case for {key}, original sample shape: {sample.shape}, final data shape: {data.shape}")
            # print(f"  sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}, length={length}")
            # print(f"  data contiguous: {data.flags.c_contiguous}, owndata: {data.flags.owndata}")
        return data
    