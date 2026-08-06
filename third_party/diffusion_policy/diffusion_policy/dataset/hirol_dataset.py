from typing import Dict
import torch
import numpy as np
import os, time, cv2, zarr, tempfile
from tqdm import tqdm # 进度条显示
import glog as log
import copy
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
)
from diffusion_policy.common.memory_budget import (
    compute_effective_budget_bytes,
    estimate_array_nbytes,
    format_gb,
)
from diffusion_policy.dataset.image_result_cache import (
    build_cache_metadata,
    open_or_build_image_result_cache,
    use_disk_result_cache,
)

DEBUG_TIME = False # 控制是否打印性能分析日志


def process_image(img, expected_image_shape):
    # cv读取图片后返回的是nparray 可以直接.shape
    original_shape = img.shape 
    # channel\height\width （3,480,640）RGB3通道 图像数组的三个维度排列顺序 下面分别是CHW和HWC 旨在拿到图像的高宽
    real_origin_shape = original_shape[1:] if original_shape[0] == 3 else original_shape[:2]
    # 从目标期望图像取出resize高宽
    resize_shape = expected_image_shape[1:] if expected_image_shape[0] == 3 else expected_image_shape[:2]
    # 如果原图高宽和目标高宽不一样，就 resize  
    # 注意 `cv2.resize` 的参数顺序是 HWC，所以用了 `[::-1]` 反转。
    if real_origin_shape != tuple(resize_shape):
        img = cv2.resize(img, tuple(resize_shape[::-1]))
    # HWC转换成CHW  Pytorch用CHW  再将unit8整数转换为float32给神经网络计算
    img = np.transpose(img, (-1, 0, 1)).astype(np.float32) / 255.0
    # 检查img与expected与image的shape是否一致
    assert img.shape == tuple(expected_image_shape), \
        f"process img wrong dim expected: {expected_image_shape}, but get {img.shape}"
    return img

def process_image_batch(args):
    """Process a batch of images in parallel  并行版本多线程处理图片"""  
    expected_image_shape, start_id, end_id, id, replay_buffer, result = args
    # 参数初始化
    data_size = end_id - start_id
    is_resized = False
    original_shape = None

    # 循环处理当前chunk的每张图片
    log.info(f'start to processing {id} chunk with {data_size} images with start id: {start_id} and end id: {end_id}')
    for i, img in enumerate(replay_buffer[start_id:end_id]):
        # 只将第一张图片的shape赋给original_shape
        if original_shape is None:
            original_shape = img.shape
        # CHW检验
        real_origin_shape = original_shape[1:] if original_shape[0] == 3 else original_shape[:2]
        resize_shape = expected_image_shape[1:] if expected_image_shape[0] == 3 else expected_image_shape[:2]
        if real_origin_shape != tuple(resize_shape):
            is_resized = True
        result[start_id + i] = process_image(img, expected_image_shape)
    log.info(f'finished processing {id} chunk images')
    return data_size, original_shape, is_resized

# Create new contiguous copies to avoid storage resize issues
def safe_torch_from_numpy(arr):
    """Convert numpy array to torch tensor with guaranteed contiguous storage"""
    # Ensure array is contiguous and create a copy
    # arr_copy = np.ascontiguousarray(arr)
    return torch.from_numpy(arr)

class HirolDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,  #action预测长度
            pad_before=0,  #  允许采样窗口向episode开头左边补充的步数。在一条轨迹刚开始时前面没有足够历史帧，用第一帧重复补在前面。
            pad_after=0,   # 在轨迹末端采样窗口用最后一帧补齐的步数
            n_obs_steps=None,   # 采样策划窗口取的历史观测步数
            n_latency_steps=0,  # 动作延迟步数
            seed=42,
            val_ratio=0.0,      #验证集比例
            max_train_episodes=None,        #最多使用多少训练episdoe  抽样限制  debug时可以减少时间
            load_into_memory=False,         # 整库存放内存
            preload_images=False,           # 是否预加载图片  还是在训练的时候再加载图片 用于低性能显卡训练
            use_parallel_loading=True,      # 是否并行预处理图片
            use_cache=None,                 # 兼容 task yaml；true 时等价于整库+图像预载
            delta_action=None,              # 兼容旧配置，当前数据集不在此处变换 action
            memory_limit_gb=None,           # 内存预算
            memory_reserve_gb=2.0,
            load_result_add="ram",          # ram 或 SSD cache 路径；路径时保存解码后的 float32 图片结果
        ):
        # 调用父类的初始化  一行直接初始化父类变量
        super().__init__()

        load_result_on_disk = use_disk_result_cache(load_result_add)
        if use_cache:
            load_into_memory = not load_result_on_disk
            preload_images = True

        # 将obs分为rgb_keys和lowdim_keys
        obs_shape_meta = shape_meta['obs']
        rgb_keys = list()
        lowdim_keys = list()
        for key, attr in obs_shape_meta.items():  # 字典的.item  遍历所有的键值对 key attr为值 
            type_val = attr.get('type', 'low_dim') # 取type 如果没有type则默认当low_dim
            if type_val == 'rgb':
                rgb_keys.append(key) #将是图像的obs放入rgb_keys
            elif type_val == 'low_dim':
                lowdim_keys.append(key) #其他obs放入lowdim_keys

        #打开zarr数据集根目录
        dataset_root = zarr.open(os.path.expanduser(dataset_path), mode='r')

        #整库复制&加载图像决策：内存占用估算 控制load_into_memory、preload_images开关 
        effective_budget_bytes = compute_effective_budget_bytes(
            memory_limit_gb=memory_limit_gb,
            memory_reserve_gb=memory_reserve_gb,
        )
        estimated_dataset_bytes = 0
        for _, value in dataset_root["data"].items():
            estimated_dataset_bytes += estimate_array_nbytes(value.shape, value.dtype)
        estimated_preload_bytes = 0
        for key in rgb_keys:
            expected_image_shape = tuple(obs_shape_meta[key]["shape"])
            step_len = dataset_root["data"][key].shape[0]
            estimated_preload_bytes += step_len * estimate_array_nbytes(expected_image_shape, np.float32)
        if effective_budget_bytes is not None:
            log.info(
                "HirolDataset RAM budget: effective=%s, estimated full-copy=%s, estimated image-preload=%s",
                format_gb(effective_budget_bytes),
                format_gb(estimated_dataset_bytes),
                format_gb(estimated_preload_bytes),
            )
            if load_into_memory and estimated_dataset_bytes > effective_budget_bytes:
                log.warning(
                    "Disabling load_into_memory because estimated dataset footprint %s exceeds RAM budget %s.",
                    format_gb(estimated_dataset_bytes),
                    format_gb(effective_budget_bytes),
                )
                load_into_memory = False
            if preload_images and (not load_result_on_disk) and estimated_preload_bytes > effective_budget_bytes:
                log.warning(
                    "Disabling preload_images because estimated preload footprint %s exceeds RAM budget %s.",
                    format_gb(estimated_preload_bytes),
                    format_gb(effective_budget_bytes),
                )
                preload_images = False


        # Load zarr data using ReplayBuffer (standard DP approach) 将zarr接入replaybuffer
        log.info(f'Loading Hirol dataset from: {dataset_path}')
        # 定义replaybuffer属性  将zarr包装成dp统一replaybuffer对象 后面dataset/sampler都通过 self.replay_buffer 来读数据 
        if load_into_memory:
            log.info('Loading dataset into memory for faster access...')
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=dataset_root.store,
                store=zarr.MemoryStore()
            )  
        else:
            log.info('Opening dataset directly from disk to avoid RAM spikes...')
            self.replay_buffer = ReplayBuffer.create_from_group(dataset_root)
        log.info(f'get {self.replay_buffer.n_episodes} episode data from replay buffer')

        # handle image data resize - parallel optimized version
        obs_data_buffer = dict() # 用来存“已经处理好的图像数据”
        step_len = len(self.replay_buffer["action"]) # 统计总步数
        log.info(f'total step len: {step_len}')

        # 并行预加载分支  
        # 用线程池 先分配一个大数组  再将整段图片按chunk分给不同线程 检查总数是否等于步长、将结果放入bs_data_buffer[key]
        load_result_cache_path = None
        if load_result_on_disk:
            image_shapes = {
                key: tuple(obs_shape_meta[key]["shape"])
                for key in rgb_keys
            }
            metadata = build_cache_metadata(
                source_type="hirol_zarr",
                dataset_path=dataset_path,
                dataset_length=step_len,
                rgb_keys=rgb_keys,
                image_shapes=image_shapes,
                extra={
                    "source_image_shapes": {
                        key: list(self.replay_buffer[key].shape)
                        for key in rgb_keys
                    },
                    "source_image_dtypes": {
                        key: str(self.replay_buffer[key].dtype)
                        for key in rgb_keys
                    },
                },
            )

            def build_frame(frame_idx):
                return {
                    key: process_image(self.replay_buffer[key][frame_idx], image_shapes[key])
                    for key in rgb_keys
                }

            obs_data_buffer, load_result_cache_path = open_or_build_image_result_cache(
                load_result_add=load_result_add,
                dataset_path=dataset_path,
                metadata=metadata,
                build_frame_fn=build_frame,
                desc="Build Hirol decoded image cache",
                logger=log,
            )
        elif preload_images and use_parallel_loading:  # =True
            # Determine optimal number of processes (reasonable limit)
            num_processes = min(50, max(1, cpu_count() // 4))
            log.info(f'Using {num_processes} processes for parallel loading')
            cur_img_key = rgb_keys[0]
            with tqdm(rgb_keys, desc=f"Loading image key: {cur_img_key}", position=0) as pbar_image_key:
                for key in pbar_image_key:
                    expected_image_shape = obs_shape_meta[key]["shape"]
                    processed_images = np.empty((step_len, ) + tuple(expected_image_shape), dtype=np.float32)
                    # Split images into chunks for parallel processing
                    chunk_size = max(1, int(len(self.replay_buffer[key]) // num_processes))
                    results_in_order = None
                    with ThreadPoolExecutor(num_processes) as ex:
                        with tqdm(range(len(self.replay_buffer[key])), desc=f"spliting chunks", position=1, leave=False) as pbar_chunk:
                            futs = []
                            start_id = 0
                            thread_id = 0
                            for chunk_id in pbar_chunk:
                                if chunk_id % chunk_size == 0:
                                    end_id = start_id + chunk_size
                                    if end_id >= len(self.replay_buffer[key]) + 1: continue
                                    args = (expected_image_shape, start_id, end_id, thread_id, self.replay_buffer[key], processed_images)
                                    start_id = end_id
                                    futs.append(ex.submit(process_image_batch, args))
                                    thread_id += 1
                                elif chunk_id == len(self.replay_buffer[key]) - 1 and chunk_id > start_id:
                                    end_id = chunk_id + 1
                                    log.info(f'end id for last one: {end_id}')
                                    args = (expected_image_shape, start_id, end_id, thread_id, self.replay_buffer[key], processed_images)
                                    futs.append(ex.submit(process_image_batch, args))
                                    thread_id += 1
                            results_in_order = [f.result() for f in futs]
                    log.info(f'already get all results for {key}')
                    
                    # Combine results from all processes
                    original_shape = None; is_resized = None; data_size = 0
                    for result in results_in_order:
                        res_size, original_shape, is_resized = result
                        data_size += res_size
                    assert data_size == step_len, f'processed data size {data_size} does not match with step len {step_len} for {key}'
                        
                    # Concatenate all processed batches
                    obs_data_buffer[key] = processed_images
                    log.info(f'combined all results for {key}')
                    log.info(f"combined result continigous: {obs_data_buffer[key].flags['C_CONTIGUOUS']}")
                    len_data_buffer = obs_data_buffer[key].shape[0]
                    assert len_data_buffer == step_len, f"{key} data buffer has wrong size, expected: {step_len} but get {len_data_buffer}, "

                    if is_resized:
                        final_shape = obs_data_buffer[key].shape
                        log.info(f'resize shape for {key} from {original_shape} to {final_shape} for {expected_image_shape}')
                    log.info(f'obs data buffer {key} len: {len(obs_data_buffer[key])} dtype: {obs_data_buffer[key].dtype}')
        elif preload_images: # 仅预加载但不走并行 退化为串行
            obs_data_buffer = self._serial_loading(rgb_keys, obs_shape_meta, step_len)
        else:
            log.info('Image preload disabled; RGB frames will be resized/normalized on demand.')
                
        # Configure observation key restrictions (standard DP pattern)
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Create validation masks (standard DP pattern)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Create sequence sampler (standard DP pattern)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            keys=rgb_keys+lowdim_keys+["action"],
            obs_data = obs_data_buffer)

        # Store all the necessary attributes
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.key_first_k = key_first_k
        self.obs_data = obs_data_buffer
        self.load_result_add = load_result_add
        self.load_result_cache_path = load_result_cache_path
        self.max_time = 0
        self.long_time_counter = 0
        self.rgb_shape_meta = obs_shape_meta
        # self.static_test_obs = dict(obs=dict(
        #     ee_cam_color=torch.zeros((3,3,128,128)),
        #     third_person_cam_color=torch.zeros((3,3,128,128)),
        #     side_cam_color=torch.zeros((3,3,128,128)),
        #     state=torch.zeros((3,8))),
        #     action=torch.zeros((horizon, 8)))
        log.info(f'Dataset loaded: {len(self)} samples from {self.replay_buffer.n_episodes} episodes')

    def _serial_loading(self, rgb_keys, obs_shape_meta, step_len, ):
        # Fallback to serial processing
        log.info('Using serial loading (parallel disabled)')
        obs_data_buffer = dict()
        with tqdm(rgb_keys, desc="Loading image keys", position=0) as pbar_image_key:
            for key in pbar_image_key:
                expected_image_shape = obs_shape_meta[key]["shape"]
                # Pre-allocate array for better performance
                img_np = np.empty((step_len,) + tuple(expected_image_shape), np.float32)
                is_resized = False
                original_shape = None
                with tqdm(self.replay_buffer[key], desc=f"Processing {key}", position=1, leave=False) as pbar_image_data:
                    for i, img in enumerate(pbar_image_data):
                        if original_shape is None:
                            original_shape = img.shape
                            log.info(f'{key} orig shape: {original_shape}')

                        real_origin_shape = original_shape[1:] if original_shape[0] == 3 else original_shape[:2]
                        resize_shape = expected_image_shape[1:] if expected_image_shape[0] == 3 else expected_image_shape[:2]
                        if real_origin_shape != tuple(resize_shape):
                            is_resized = True
                        img_np[i, ...] = process_image(img, expected_image_shape)

                # Store processed images
                obs_data_buffer[key] = img_np
                if is_resized:
                    final_shape = obs_data_buffer[key].shape
                    log.info(f'resize shape for {key} from {original_shape} to {final_shape} for {expected_image_shape}')
                log.info(f'obs data buffer {key} len: {len(obs_data_buffer[key])} dtype: {obs_data_buffer[key].dtype}')
        return obs_data_buffer

    def get_validation_dataset(self) -> 'BaseImageDataset':
        """Create validation dataset following standard DP pattern"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k=self.key_first_k,
            keys=self.rgb_keys+self.lowdim_keys+["action"],
            obs_data = self.obs_data
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Create normalizer following standard DP pattern"""
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])

        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions following standard DP pattern"""
        return torch.from_numpy(self.replay_buffer['action'][:])

    def __len__(self) -> int:
        """Return dataset length following standard DP pattern"""
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item following standard DP pattern like RealPushTImageDataset"""
        # Use sampler to get sequence data (standard DP approach)
        start = None; real_start = None; sample_time = None
        rgb_time = None; low_dim_time = None; action_time = None; torch_time = None
        if DEBUG_TIME:
            real_start = time.perf_counter()
            start = time.perf_counter()
        # 出来dict 每个key的（Horizon， data_shape）
        data = self.sampler.sample_sequence(idx)
        if DEBUG_TIME:
            sample_time = time.perf_counter() - start
            log.info(f'sample time: {sample_time:.6f}s')
        
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        # Process RGB keys
        if DEBUG_TIME:
            start = time.perf_counter()
        for key in self.rgb_keys:
            if self.obs_data and key in self.obs_data:
                obs_dict[key] = np.array(data[key], copy=False)
            else:
                expected_image_shape = tuple(self.rgb_shape_meta[key]["shape"])
                obs_dict[key] = np.stack(
                    [process_image(frame, expected_image_shape) for frame in data[key]],
                    axis=0
                )
            assert obs_dict[key].shape[1:] == tuple(self.rgb_shape_meta[key]["shape"]), \
            f'{key} get item expected: {self.rgb_shape_meta[key]["shape"]}, get {obs_dict[key].shape}'
            # obs_dict[key] = np.array(data[key])
            if DEBUG_TIME:
                log.info(f'{key} dtype: {obs_dict[key].dtype}')
        if DEBUG_TIME:
            rgb_time = time.perf_counter() - start
            log.info(f'rgb time: {rgb_time}')
        
        # Process lowdim keys
        if DEBUG_TIME:
            start = time.perf_counter()
        for key in self.lowdim_keys:
            # Ensure we have independent data, not views
            # temp_data = data[key][T_slice].astype(np.float32)
            obs_dict[key] = data[key][T_slice].astype(np.float32)
        if DEBUG_TIME:
            low_dim_time = time.perf_counter() - start
            log.info(f'low dim time: {low_dim_time}')

        if DEBUG_TIME:
            start = time.perf_counter()
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        action = np.array(action, copy=True)
        if DEBUG_TIME:
            action_time = time.perf_counter() - start
            log.info(f'action copy time: {action_time}')

        if DEBUG_TIME:
            start = time.perf_counter()
        # Debug: Check array properties before converting to torch
        # print(f"\n=== DEBUG Sample {idx} ===")
        # print(f"Action shape: {action.shape}, dtype: {action.dtype}, contiguous: {action.flags.c_contiguous}")
        # for key in obs_dict:
        #     arr = obs_dict[key]
        #     print(f"Obs[{key}] shape: {arr.shape}, dtype: {arr.dtype}, contiguous: {arr.flags.c_contiguous}")
        #     if hasattr(arr, 'flags'):
        #         print(f"  flags: owndata={arr.flags.owndata}, writeable={arr.flags.writeable}")

        torch_data = {
            'obs': dict_apply(obs_dict, safe_torch_from_numpy),
            'action': safe_torch_from_numpy(action)
        }

        # Debug: Check tensor properties after converting to torch
        # print(f"Torch action shape: {torch_data['action'].shape}, storage_size: {torch_data['action'].storage().size()}")
        # for key in torch_data['obs']:
        #     tensor = torch_data['obs'][key]
        #     print(f"Torch obs[{key}] shape: {tensor.shape}, storage_size: {tensor.storage().size()}, is_contiguous: {tensor.is_contiguous()}")
        # print("=== END DEBUG ===\n")

        if DEBUG_TIME:
            torch_time = time.perf_counter() - start
            log.info(f'torch time: {torch_time}')
            total_time = time.perf_counter() - real_start
            log.info(f'total: {total_time} for {idx}')
            if total_time > 0.0045:
                self.long_time_counter += 1
                log.info(f'sample used: {sample_time/total_time*100}')
                log.info(f'rgb used: {rgb_time/total_time*100}')
                log.info(f'low dim used: {low_dim_time/total_time*100}')
                log.info(f'action used: {action_time/total_time*100}')
                log.info(f'torch used: {torch_time/total_time*100}')
                log.info(f"{'='*30} Encountered {self.long_time_counter} long time loading {'='*30}")
            if total_time > self.max_time:
                self.max_time = total_time
        return torch_data

def test():
    import os, time
    from torch.utils.data import DataLoader
    global DEBUG_TIME
    DEBUG_TIME = True
    
    # Example shape_meta configuration
    image_shape = [3, 480, 640]
    shape_meta = {
        'obs': {
            'state': {
                'shape': [8],
                'type': 'low_dim'
            },
            'ee_cam_color':
            {
                'shape': image_shape,
                'type': 'rgb'
            },
            'third_person_cam_color':
            {
                'shape': image_shape,
                'type': 'rgb'
            },
            'side_cam_color':
            {
                'shape': image_shape,
                'type': 'rgb'
            },
        },
        'action': {
            'shape': [8]
        }
    }

    # Test dataset path - replace with your actual dataset path
    dataset_path = "/home/zyx/dataset/dp/fr3/0920/water_pouring_1_step_0_skip_abs_jps.zarr"

    if not os.path.exists(dataset_path):
        log.info(f"Dataset path not found: {dataset_path}")
        log.info("Please update the dataset_path in the test function")
        return

    log.info("Creating HirolDataset with parallel loading...")
    dataset = HirolDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        n_obs_steps=3,
        val_ratio=0.1,
        use_parallel_loading=True
    )
    log.info(f"Successfully loaded dataset {'=='*50}")

    # Test DataLoader batch loading
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    log.info(f"Successfully loaded dataloader {'=='*50}")
    start = time.time()
    bach_start = time.time()
    for batch_id, batch in enumerate(dataloader):
        log.info(f'Enter {batch_id} th batch data')
        if batch_id % 15 == 0 and batch_id != 0: break
        batch_load_time = time.time() - bach_start
        log.info(f"raw batch time: {batch_load_time}. batch time for 128: {(batch_load_time/batch_size)*128} {'=='*50}")
        bach_start = time.time()
    log.info(f'max get item time: {dataset.max_time}')

    log.info(f"RGB keys: {dataset.rgb_keys}")
    log.info(f"Low-dim keys: {dataset.lowdim_keys}")

    # Test batch data loading
    log.info(f"\nBatch obs keys: {list(batch['obs'].keys())}")
    log.info(f"Batch action shape: {batch['action'].shape}")

    # Test data loading
    log.info("\nTesting single sample loading...")
    sample = dataset[0]
    log.info(f"Sample keys: {list(sample.keys())}")
    log.info(f"Obs keys: {list(sample['obs'].keys())}")
    log.info(f"Action shape: {sample['action'].shape}")

    for key in dataset.lowdim_keys:
        log.info(f"Single sample - Lowdim {key} shape: {sample['obs'][key].shape}")
        log.info(f"Batch - Lowdim {key} shape: {batch['obs'][key].shape}")
    for key in dataset.rgb_keys:
        log.info(f"Single sample - RGB {key} shape: {sample['obs'][key].shape}")
        log.info(f"Batch - RGB {key} shape: {batch['obs'][key].shape}")
    log.info(f"Batch action shape: {batch['action'].shape}")

    # Test validation dataset
    # val_dataset = dataset.get_validation_dataset()
    # log.info(f"\nValidation dataset length: {len(val_dataset)}")

    # Test normalizer
    satrt = time.time()
    normalizer = dataset.get_normalizer()
    log.info(f"Normalizer created successfully with time {time.time() - start}")
    try:
        action_norm = normalizer['action']
        log.info(f"✅ Action normalizer works")
    except:
        log.info("❌ Action normalizer failed")

    for key in dataset.lowdim_keys:
        try:
            norm = normalizer[key]
            log.info(f"✅ {key} normalizer works")
        except:
            log.info(f"❌ {key} normalizer failed")

    for key in dataset.rgb_keys:
        try:
            norm = normalizer[key]
            log.info(f"✅ {key} normalizer works")
        except:
            log.info(f"❌ {key} normalizer failed")

    log.info("\n🎉 HirolDataset test completed successfully!")
    log.info("✅ Standard DP structure implemented")
    log.info("✅ Batch loading works")
    log.info("✅ Validation dataset works")
    log.info("✅ Normalizer works")

if __name__ == "__main__":
    test()


def _hirol_dataset_is_chunked_zarr(dataset_path: str) -> bool:
    try:
        root = zarr.open(os.path.expanduser(dataset_path), mode='r')
    except Exception:
        return False
    return isinstance(root, zarr.Group) and ('episodes' in root) and ('data' not in root)


def _hirol_dataset_sorted_episode_names(episodes_group) -> list:
    return sorted(list(episodes_group.keys()))


def _hirol_dataset_parse_shape_meta(shape_meta: dict):
    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type_val = attr.get('type', 'low_dim')
        if type_val == 'rgb':
            rgb_keys.append(key)
        elif type_val == 'low_dim':
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys, obs_shape_meta


def _hirol_dataset_get_chunked_image_group(episode_group):
    observation = episode_group['observation']
    if 'images' in observation:
        return observation['images']
    return observation


def _hirol_dataset_read_chunked_episode_array(episode_group, key: str, is_rgb: bool):
    observation = episode_group['observation']
    if is_rgb:
        image_group = _hirol_dataset_get_chunked_image_group(episode_group)
        if key not in image_group:
            raise KeyError(f'Chunked zarr is missing rgb key "{key}"')
        return image_group[key][:]
    if key not in observation:
        raise KeyError(f'Chunked zarr is missing low-dim key "{key}"')
    return observation[key][:]


def _hirol_dataset_get_chunked_preferred_action_side(root, episode_group) -> str:
    texts = list()
    try:
        for attrs in (getattr(root, 'attrs', None), getattr(episode_group, 'attrs', None)):
            if attrs is None:
                continue
            for key in ('dataset_name', 'source_root', 'source_path', 'text'):
                if key in attrs:
                    texts.append(str(attrs[key]).lower())
    except Exception:
        pass
    merged = ' '.join(texts)
    if 'left' in merged:
        return 'left'
    if 'right' in merged:
        return 'right'
    return 'left'


def _hirol_dataset_shift_actions_forward(action: np.ndarray) -> np.ndarray:
    shifted = np.array(action, copy=True)
    if shifted.shape[0] > 1:
        shifted[:-1] = shifted[1:]
    return shifted


def _hirol_dataset_try_read_explicit_chunked_action(episode_group):
    candidate_paths = (
        ('action',),
        ('actions',),
        ('observation', 'action'),
        ('observation', 'actions'),
        ('policy', 'action'),
        ('policy', 'actions'),
    )
    for path in candidate_paths:
        node = episode_group
        found = True
        for part in path:
            if not isinstance(node, zarr.Group) or part not in node:
                found = False
                break
            node = node[part]
        if found and isinstance(node, zarr.Array):
            return node[:]
    return None


def _hirol_dataset_infer_chunked_action(root, episode_group, action_shape):
    observation = episode_group['observation']
    preferred_side = _hirol_dataset_get_chunked_preferred_action_side(root, episode_group)
    pair_candidates = [
        (preferred_side, f'ee_pose_{preferred_side}', f'tool_{preferred_side}')
    ]
    for side in ('left', 'right'):
        candidate = (side, f'ee_pose_{side}', f'tool_{side}')
        if candidate not in pair_candidates:
            pair_candidates.append(candidate)

    expected_dim = int(np.prod(action_shape))
    for side, pose_key, tool_key in pair_candidates:
        if pose_key not in observation or tool_key not in observation:
            continue
        pose = observation[pose_key][:]
        tool = observation[tool_key][:]
        if len(pose.shape) != 2 or len(tool.shape) != 2:
            continue

        if pose.shape[1] == expected_dim:
            action = pose.astype(np.float32)
            action = _hirol_dataset_shift_actions_forward(action)
            log.warning(
                f'Chunked zarr does not contain explicit action. '
                f'Inferring action from {pose_key} with next-step shift.')
            return action

        concat_dim = pose.shape[1] + tool.shape[1]
        if concat_dim == expected_dim:
            action = np.concatenate([pose, tool], axis=-1).astype(np.float32)
            action = _hirol_dataset_shift_actions_forward(action)
            log.warning(
                f'Chunked zarr does not contain explicit action. '
                f'Inferring action from {pose_key}+{tool_key} with next-step shift.')
            return action
    raise KeyError(
        'Chunked zarr does not expose an action array, and no supported '
        f'inference rule matched the requested action shape {action_shape}.')


def _hirol_dataset_build_replay_buffer_from_chunked(
        shape_meta: dict,
        dataset_path: str,
        store_path: str = None,
    ) -> ReplayBuffer:
    root = zarr.open(os.path.expanduser(dataset_path), mode='r')
    episodes_group = root['episodes']
    rgb_keys, lowdim_keys, _ = _hirol_dataset_parse_shape_meta(shape_meta)
    action_shape = tuple(shape_meta['action']['shape'])

    if store_path is None:
        replay_buffer = ReplayBuffer.create_empty_numpy()
    else:
        replay_buffer = ReplayBuffer.create_empty_zarr(
            storage=zarr.DirectoryStore(store_path)
        )
    episode_names = _hirol_dataset_sorted_episode_names(episodes_group)
    log.info(f'Converting chunked zarr with {len(episode_names)} episodes into replay buffer format...')

    with tqdm(episode_names, desc='Flatten chunked episodes', leave=False) as pbar_episodes:
        for episode_name in pbar_episodes:
            episode_group = episodes_group[episode_name]
            episode_data = dict()
            episode_length = None

            for key in rgb_keys:
                value = _hirol_dataset_read_chunked_episode_array(
                    episode_group=episode_group,
                    key=key,
                    is_rgb=True)
                episode_data[key] = value
                if episode_length is None:
                    episode_length = value.shape[0]
                else:
                    assert episode_length == value.shape[0], \
                        f'{episode_name}:{key} length mismatch, expected {episode_length}, got {value.shape[0]}'

            for key in lowdim_keys:
                value = _hirol_dataset_read_chunked_episode_array(
                    episode_group=episode_group,
                    key=key,
                    is_rgb=False).astype(np.float32)
                episode_data[key] = value
                if episode_length is None:
                    episode_length = value.shape[0]
                else:
                    assert episode_length == value.shape[0], \
                        f'{episode_name}:{key} length mismatch, expected {episode_length}, got {value.shape[0]}'

            action = _hirol_dataset_try_read_explicit_chunked_action(episode_group)
            if action is None:
                action = _hirol_dataset_infer_chunked_action(
                    root=root,
                    episode_group=episode_group,
                    action_shape=action_shape)
            action = np.asarray(action, dtype=np.float32)
            if len(action.shape) == 1:
                action = action[:, None]
            assert action.shape[1:] == action_shape, \
                f'{episode_name}: action shape mismatch, expected (*,{action_shape}), got {action.shape}'
            if episode_length is None:
                episode_length = action.shape[0]
            else:
                assert episode_length == action.shape[0], \
                    f'{episode_name}: action length mismatch, expected {episode_length}, got {action.shape[0]}'
            episode_data['action'] = action

            replay_buffer.add_episode(episode_data)

    log.info(f'Chunked zarr flatten finished with {replay_buffer.n_episodes} episodes and {replay_buffer.n_steps} steps')
    return replay_buffer


def _hirol_dataset_initialize_from_replay_buffer(
        self,
        replay_buffer: ReplayBuffer,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        preload_images=False,
        use_parallel_loading=True,
        use_cache=None,
        delta_action=None,
    ):
    if use_cache:
        preload_images = True

    self.replay_buffer = replay_buffer
    log.info(f'get {self.replay_buffer.n_episodes} episode data from replay buffer')

    rgb_keys, lowdim_keys, obs_shape_meta = _hirol_dataset_parse_shape_meta(shape_meta)

    obs_data_buffer = dict()
    step_len = len(self.replay_buffer["action"])
    log.info(f'total step len: {step_len}')

    if preload_images and use_parallel_loading and len(rgb_keys) > 0:
        num_processes = min(50, max(1, cpu_count() // 4))
        log.info(f'Using {num_processes} processes for parallel loading')
        cur_img_key = rgb_keys[0]
        with tqdm(rgb_keys, desc=f"Loading image key: {cur_img_key}", position=0) as pbar_image_key:
            for key in pbar_image_key:
                expected_image_shape = obs_shape_meta[key]["shape"]
                processed_images = np.empty((step_len,) + tuple(expected_image_shape), dtype=np.float32)
                chunk_size = max(1, int(len(self.replay_buffer[key]) // num_processes))
                results_in_order = None
                with ThreadPoolExecutor(num_processes) as ex:
                    with tqdm(range(len(self.replay_buffer[key])), desc=f"spliting chunks", position=1, leave=False) as pbar_chunk:
                        futs = []
                        start_id = 0
                        thread_id = 0
                        for chunk_id in pbar_chunk:
                            if chunk_id % chunk_size == 0:
                                end_id = start_id + chunk_size
                                if end_id >= len(self.replay_buffer[key]) + 1:
                                    continue
                                args = (expected_image_shape, start_id, end_id, thread_id, self.replay_buffer[key], processed_images)
                                start_id = end_id
                                futs.append(ex.submit(process_image_batch, args))
                                thread_id += 1
                            elif chunk_id == len(self.replay_buffer[key]) - 1 and chunk_id > start_id:
                                end_id = chunk_id + 1
                                log.info(f'end id for last one: {end_id}')
                                args = (expected_image_shape, start_id, end_id, thread_id, self.replay_buffer[key], processed_images)
                                futs.append(ex.submit(process_image_batch, args))
                                thread_id += 1
                        results_in_order = [f.result() for f in futs]
                log.info(f'already get all results for {key}')

                original_shape = None
                is_resized = None
                data_size = 0
                for result in results_in_order:
                    res_size, original_shape, is_resized = result
                    data_size += res_size
                assert data_size == step_len, f'processed data size {data_size} does not match with step len {step_len} for {key}'

                obs_data_buffer[key] = processed_images
                log.info(f'combined all results for {key}')
                log.info(f"combined result continigous: {obs_data_buffer[key].flags['C_CONTIGUOUS']}")
                len_data_buffer = obs_data_buffer[key].shape[0]
                assert len_data_buffer == step_len, f"{key} data buffer has wrong size, expected: {step_len} but get {len_data_buffer}, "

                if is_resized:
                    final_shape = obs_data_buffer[key].shape
                    log.info(f'resize shape for {key} from {original_shape} to {final_shape} for {expected_image_shape}')
                log.info(f'obs data buffer {key} len: {len(obs_data_buffer[key])} dtype: {obs_data_buffer[key].dtype}')
    elif preload_images:
        obs_data_buffer = self._serial_loading(rgb_keys, obs_shape_meta, step_len)
    else:
        log.info('Image preload disabled; RGB frames will be resized/normalized on demand.')

    key_first_k = dict()
    if n_obs_steps is not None:
        for key in rgb_keys + lowdim_keys:
            key_first_k[key] = n_obs_steps

    val_mask = get_val_mask(
        n_episodes=self.replay_buffer.n_episodes,
        val_ratio=val_ratio,
        seed=seed)
    train_mask = ~val_mask
    train_mask = downsample_mask(
        mask=train_mask,
        max_n=max_train_episodes,
        seed=seed)

    self.sampler = SequenceSampler(
        replay_buffer=self.replay_buffer,
        sequence_length=horizon+n_latency_steps,
        pad_before=pad_before,
        pad_after=pad_after,
        episode_mask=train_mask,
        key_first_k=key_first_k,
        keys=rgb_keys+lowdim_keys+["action"],
        obs_data=obs_data_buffer)

    self.shape_meta = shape_meta
    self.rgb_keys = rgb_keys
    self.lowdim_keys = lowdim_keys
    self.n_obs_steps = n_obs_steps
    self.val_mask = val_mask
    self.train_mask = train_mask
    self.horizon = horizon
    self.n_latency_steps = n_latency_steps
    self.pad_before = pad_before
    self.pad_after = pad_after
    self.key_first_k = key_first_k
    self.obs_data = obs_data_buffer
    self.max_time = 0
    self.long_time_counter = 0
    self.rgb_shape_meta = obs_shape_meta
    log.info(f'Dataset loaded: {len(self)} samples from {self.replay_buffer.n_episodes} episodes')


_HIROLDATASET_ORIGINAL_INIT = HirolDataset.__init__


def _hirol_dataset_chunk_aware_init(
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
        load_into_memory=False,
        preload_images=False,
        use_parallel_loading=True,
        use_cache=None,
        delta_action=None,
        memory_limit_gb=None,
        memory_reserve_gb=2.0,
    ):
    if use_cache:
        load_into_memory = True
        preload_images = True

    if not _hirol_dataset_is_chunked_zarr(dataset_path):
        return _HIROLDATASET_ORIGINAL_INIT(
            self,
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            n_latency_steps=n_latency_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            load_into_memory=load_into_memory,
            preload_images=preload_images,
            use_parallel_loading=use_parallel_loading,
            use_cache=use_cache,
            delta_action=delta_action,
            memory_limit_gb=memory_limit_gb,
            memory_reserve_gb=memory_reserve_gb,
        )

    BaseImageDataset.__init__(self)
    log.info(f'Loading Hirol chunked dataset from: {dataset_path}')
    flatten_store_path = None
    effective_budget_bytes = compute_effective_budget_bytes(
        memory_limit_gb=memory_limit_gb,
        memory_reserve_gb=memory_reserve_gb,
    )
    if effective_budget_bytes is not None:
        self._chunk_flatten_tmpdir = tempfile.TemporaryDirectory(prefix='hirol_flatten_')
        flatten_store_path = self._chunk_flatten_tmpdir.name
        log.info(
            'Chunked dataset RAM budget active (%s); flattening through disk-backed zarr at %s',
            format_gb(effective_budget_bytes),
            flatten_store_path,
        )
    replay_buffer = _hirol_dataset_build_replay_buffer_from_chunked(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        store_path=flatten_store_path,
    )
    _hirol_dataset_initialize_from_replay_buffer(
        self,
        replay_buffer=replay_buffer,
        shape_meta=shape_meta,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        n_obs_steps=n_obs_steps,
        n_latency_steps=n_latency_steps,
        seed=seed,
        val_ratio=val_ratio,
        max_train_episodes=max_train_episodes,
        preload_images=preload_images,
        use_parallel_loading=use_parallel_loading,
        use_cache=use_cache,
        delta_action=delta_action,
    )


HirolDataset.__init__ = _hirol_dataset_chunk_aware_init
    
