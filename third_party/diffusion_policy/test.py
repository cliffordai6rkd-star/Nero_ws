import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset


# =========================
# 1. 配置要检查的数据集
# =========================
# 这里的 shape_meta 必须和训练 YAML 里的 shape_meta 保持一致。
# Dataset 会根据这个结构决定哪些字段是图像、哪些字段是 low_dim。
shape_meta = {
    "obs": {
        "ee_cam_color": {
            "shape": [3, 224, 224],
            "type": "rgb",
        },
        "third_person_cam_color": {
            "shape": [3, 224, 224],
            "type": "rgb",
        },
        "side_cam_color": {
            "shape": [3, 224, 224],
            "type": "rgb",
        },
        "state_ee": {
            "shape": [15],
            "type": "low_dim",
        },
    },
    "action": {
        "shape": [8],
    },
}

# state_ee 的最后一维是 15，把它拆成更有语义的几段来查看。
state_components = {
    "ee_position": (0, 3),
    "ee_quaternion": (3, 7),
    "joint_positions": (7, 14),
    "gripper_width": (14, 15),
}

# action 的最后一维是 8，也按训练配置拆开查看。
action_components = {
    "ee_position": (0, 3),
    "ee_quaternion": (3, 7),
    "gripper_width": (7, 8),
}


# =========================
# 2. 通用打印函数
# =========================
def show_tensor(name, x):
    """打印一个 tensor/ndarray 的基本统计，快速发现 shape、NaN、Inf 和数值范围问题。"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    print(f"\n{name}")
    print("  shape:", tuple(x.shape))
    print("  dtype:", x.dtype)
    print("  min:", x.min().item())
    print("  max:", x.max().item())
    print("  mean:", x.mean().item())
    print("  std:", x.std().item())
    print("  has_nan:", torch.isnan(x).any().item())
    print("  has_inf:", torch.isinf(x).any().item())


def show_components(prefix, x, components):
    """把 state/action 按 component 切开，分别查看每一段的统计。"""
    for name, (start, end) in components.items():
        show_tensor(f"{prefix}.{name}", x[..., start:end])


def show_normalizer_stats(name, normalizer, components):
    """查看 normalizer 用全数据拟合出来的 min/max/mean/std。"""
    stats = normalizer.get_input_stats()
    print(f"\n========== {name} normalizer input stats ==========")
    for comp_name, (start, end) in components.items():
        print(f"\n{name}.{comp_name}")
        for stat_name in ["min", "max", "mean", "std"]:
            value = stats[stat_name][start:end].detach().cpu().numpy()
            print(f"  {stat_name}: {value}")


def normalize_in_chunks(name, data, field_normalizer, components, chunk_size=4096):
    """
    分块做全局归一化检查。

    这样不用一次性把所有数据都转成一个大 tensor，也能用进度条看到检查进度。
    这里检查的是 LeRobotV3 每个原始 frame 上的 low_dim/action 全局归一化结果。
    """
    normalized_chunks = []
    total = len(data)
    for start in tqdm(range(0, total, chunk_size), desc=f"Normalize {name} globally"):
        end = min(start + chunk_size, total)
        chunk = data[start:end]
        normalized_chunks.append(field_normalizer.normalize(chunk).cpu())

    normalized = torch.cat(normalized_chunks, dim=0)
    show_tensor(f"global_normalized.{name}", normalized)
    show_components(f"global_normalized.{name}", normalized, components)
    return normalized


# =========================
# 3. 实例化 Dataset
# =========================
# 这一段对应训练时 Hydra 创建 HirolLeRobotV3Dataset 的过程。
# 重点确认 feature_map 和训练 YAML 一致，否则 shape 对但语义可能错。
dataset = HirolLeRobotV3Dataset(
    shape_meta=shape_meta,
    dataset_path="data/train_episode/moving_bread/moving_bread_hirol_lerobotv3",
    horizon=16,
    pad_before=1,
    pad_after=7,
    n_obs_steps=2,
    n_latency_steps=0,
    val_ratio=0.2,
    window_sampling_strategy="idx",
    image_feature_map={
        "ee_cam_color": "observation.images.ee_cam_color",
        "third_person_cam_color": "observation.images.third_person_cam_color",
        "side_cam_color": "observation.images.side_cam_color",
    },
    lowdim_feature_groups={
        "state_ee": ["observation.state"],
    },
    action_feature_fields=[
        "action.ee_pose",
        "action.gripper_width",
    ],
    preload_images=False,
    load_result_add="ram",
)

print("dataset len:", len(dataset))
print("dataset raw frame length:", dataset.dataset_length)
print("first index row [buffer_start, buffer_end, sample_start, sample_end]:", dataset.indices[0])
print("first sequence indices:", dataset._sample_indices_to_sequence(0))


# =========================
# 4. 检查 dataset[0]
# =========================
# 这里检查单个训练样本经过 __getitem__ 后的形态。
# 预期 obs 图像是 [n_obs_steps, C, H, W]，state_ee 是 [n_obs_steps, 15]，
# action 是 [horizon, 8]。
sample = dataset[0]
print("\n========== dataset[0] ==========")
print("sample keys:", sample.keys())
print("sample obs keys:", sample["obs"].keys())

for key, value in sample["obs"].items():
    show_tensor(f"sample.obs.{key}", value)
show_tensor("sample.action", sample["action"])

show_components("sample.obs.state_ee", sample["obs"]["state_ee"], state_components)
show_components("sample.action", sample["action"], action_components)


# =========================
# 5. 检查 DataLoader batch
# =========================
# 这里模拟训练时 DataLoader 拼 batch 后的数据。
# num_workers=0 是为了调试时错误栈更清楚。
dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(dataloader))
print("\n========== DataLoader first batch ==========")
print("batch keys:", batch.keys())
print("batch obs keys:", batch["obs"].keys())

for key, value in batch["obs"].items():
    show_tensor(f"batch.obs.{key}", value)
show_tensor("batch.action", batch["action"])

show_components("batch.obs.state_ee", batch["obs"]["state_ee"], state_components)
show_components("batch.action", batch["action"], action_components)


# =========================
# 6. 拟合 normalizer
# =========================
# 训练代码里也是通过 dataset.get_normalizer(mode="gaussian") 得到归一化器。
# gaussian 模式会对每一维做 (x - mean) / std。
normalizer = dataset.get_normalizer(mode="gaussian")

show_normalizer_stats("state_ee", normalizer["state_ee"], state_components)
show_normalizer_stats("action", normalizer["action"], action_components)


# =========================
# 7. 检查一个 batch 的归一化结果
# =========================
# 这一步对应 policy.compute_loss 里：
#   nobs = normalizer.normalize(batch["obs"])
#   naction = normalizer["action"].normalize(batch["action"])
nobs = normalizer.normalize(batch["obs"])
naction = normalizer["action"].normalize(batch["action"])

print("\n========== normalized first batch ==========")
for key, value in nobs.items():
    show_tensor(f"nobs.{key}", value)
show_tensor("naction", naction)

show_components("nobs.state_ee", nobs["state_ee"], state_components)
show_components("naction", naction, action_components)


# =========================
# 8. 全局 raw-frame 归一化检查
# =========================
# 这里检查全数据上的 state_ee/action 归一化后是否整体接近 mean=0/std=1。
# 注意：这里用的是原始 frame 级别数据，不是 DataLoader 采样出来的带 padding 窗口。
all_nstate = normalize_in_chunks(
    name="state_ee",
    data=dataset.lowdim_data["state_ee"],
    field_normalizer=normalizer["state_ee"],
    components=state_components,
)

all_naction = normalize_in_chunks(
    name="action",
    data=dataset.action_data,
    field_normalizer=normalizer["action"],
    components=action_components,
)


# =========================
# 9. 额外检查四元数模长
# =========================
# 四元数本身应该接近单位长度。这里分别检查原始值和归一化后的值。
# 原始四元数 norm 接近 1 是合理的；归一化后的 quaternion 不再表示真实四元数，
# 它只是网络输入/输出空间里的数值。
raw_state_quat_norm = torch.linalg.norm(torch.from_numpy(dataset.lowdim_data["state_ee"][:, 3:7]), dim=-1)
raw_action_quat_norm = torch.linalg.norm(torch.from_numpy(dataset.action_data[:, 3:7]), dim=-1)
nstate_quat_norm = torch.linalg.norm(all_nstate[:, 3:7], dim=-1)
naction_quat_norm = torch.linalg.norm(all_naction[:, 3:7], dim=-1)

print("\n========== quaternion norm ==========")
show_tensor("raw.state_ee.ee_quaternion.norm", raw_state_quat_norm)
show_tensor("raw.action.ee_quaternion.norm", raw_action_quat_norm)
show_tensor("normalized.state_ee.ee_quaternion.norm", nstate_quat_norm)
show_tensor("normalized.action.ee_quaternion.norm", naction_quat_norm)
