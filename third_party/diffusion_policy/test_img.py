import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


# =========================
# 1. 调试配置
# =========================
# 这个脚本专门用来做“单 batch 图像链路 debug”。
# 它不会启动训练，也不会走 workspace / wandb / optimizer。
#
# 目标是把下面几层图像数据都打印出来：
# 1) Dataset + DataLoader 输出的原始 batch 图像，范围通常应在 [0, 1]
# 2) normalizer 之后的 nobs 图像，观察它是否仍然保持 [0,1]，还是被改成别的范围
# 3) 进入 obs_encoder 之前的图像
# 4) 经过 obs_encoder 里的图像 transform（尤其是 imagenet_norm）之后的图像
# 5) 最终 obs_encoder 输出的 feature
#
# 这样可以直接验证：
# “当前图像在进入 ImageNet Normalize 之前，到底是不是保持了 [0,1] 的正确前提”

DATASET_PATH = "data/train_episode/moving_bread/moving_bread_hirol_lerobotv3"
BATCH_SIZE = 3
HORIZON = 16
N_OBS_STEPS = 2
PAD_BEFORE = 1
PAD_AFTER = 7


# =========================
# 2. shape_meta
# =========================
# 这里保持和训练配置一致，用来构造 Dataset 与 obs_encoder。
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

RGB_KEYS = [
    "ee_cam_color",
    "third_person_cam_color",
    "side_cam_color",
]


# =========================
# 3. 通用打印函数
# =========================
def show_tensor(name, x):
    """打印 tensor/ndarray 的基本统计，方便检查数值分布是否合理。"""
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


def show_rgb_per_channel(name, x):
    """额外打印 RGB 三个通道的均值和标准差，帮助判断通道分布是否被拉偏。"""
    x = x.detach().cpu()
    channel_mean = x.mean(dim=(0, 2, 3))
    channel_std = x.std(dim=(0, 2, 3))
    print(f"  channel_mean: {channel_mean.numpy()}")
    print(f"  channel_std: {channel_std.numpy()}")


def flatten_obs_time(obs_dict, n_obs_steps):
    """
    模拟 policy.compute_loss 里送入 obs_encoder 的处理：
    [B, T, ...] -> [B*T, ...]，只取前 n_obs_steps 个 observation。
    """
    return dict_apply(
        obs_dict,
        lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:])
    )


# =========================
# 4. 构造 Dataset 与 DataLoader
# =========================
dataset = HirolLeRobotV3Dataset(
    shape_meta=shape_meta,
    dataset_path=DATASET_PATH,
    horizon=HORIZON,
    pad_before=PAD_BEFORE,
    pad_after=PAD_AFTER,
    n_obs_steps=N_OBS_STEPS,
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

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(dataloader))
normalizer = dataset.get_normalizer(mode="gaussian")
nobs = normalizer.normalize(batch["obs"])


# =========================
# 5. 构造 obs_encoder
# =========================
# 这里不需要真的用预训练权重来 debug 数值分布。
# 我们只关心“图像进入 encoder 前后的范围”，因此 weights=None 就足够。
#
# 另外这里故意把 random_crop 关掉，避免每次 debug 因为随机裁剪导致统计抖动。
# 这不会影响判断“图像是否被双重归一化”。
obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=get_resnet(name="resnet18", weights=None),
    resize_shape=None,
    crop_shape=(216, 216),
    random_crop=False,
    use_group_norm=True,
    share_rgb_model=True,
    imagenet_norm=True,
)
obs_encoder.eval()


# =========================
# 6. 先看 raw batch 和 nobs
# =========================
print("\n========== raw batch image stats ==========")
for key in RGB_KEYS:
    show_tensor(f'batch["obs"]["{key}"]', batch["obs"][key])

print("\n========== normalized image stats (after dataset normalizer) ==========")
for key in RGB_KEYS:
    show_tensor(f'nobs["{key}"]', nobs[key])


# =========================
# 7. 模拟 policy 中送入 obs_encoder 的输入
# =========================
# 训练代码里在 compute_loss/predict_action 前会把 obs reshape 成 [B*T, C, H, W]。
encoder_input_raw = flatten_obs_time(batch["obs"], N_OBS_STEPS)
encoder_input_norm = flatten_obs_time(nobs, N_OBS_STEPS)

print("\n========== encoder input image stats ==========")
for key in RGB_KEYS:
    show_tensor(f'encoder_input_raw["{key}"]', encoder_input_raw[key])
    show_rgb_per_channel(f'encoder_input_raw["{key}"]', encoder_input_raw[key])
    show_tensor(f'encoder_input_norm["{key}"]', encoder_input_norm[key])
    show_rgb_per_channel(f'encoder_input_norm["{key}"]', encoder_input_norm[key])


# =========================
# 8. 单独验证 ImageNet Normalize 的输入前提
# =========================
# ImageNet Normalize 假设输入在 [0, 1]。
# 这里用同一组 mean/std，分别处理：
# 1) raw 图像（理想基线）
# 2) 当前 normalizer 输出给 encoder 的图像（真实训练路径）
imagenet_only = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

print("\n========== explicit ImageNet normalization check ==========")
for key in RGB_KEYS:
    img_raw = encoder_input_raw[key].clone()
    img_norm = encoder_input_norm[key].clone()

    img_raw_imagenet = imagenet_only(img_raw)
    img_norm_imagenet = imagenet_only(img_norm)

    show_tensor(f"{key}: raw_input -> imagenet_norm", img_raw_imagenet)
    show_rgb_per_channel(f"{key}: raw_input -> imagenet_norm", img_raw_imagenet)

    show_tensor(f"{key}: encoder_input_norm -> imagenet_norm", img_norm_imagenet)
    show_rgb_per_channel(f"{key}: encoder_input_norm -> imagenet_norm", img_norm_imagenet)


# =========================
# 9. 看 obs_encoder 内部 transform 前后
# =========================
# 这里直接复用 encoder 里的 key_transform_map。
# 由于我们在上面把 random_crop=False，这里的比较是稳定的。
print("\n========== obs_encoder transform check ==========")
transformed_inputs = {}
for key in RGB_KEYS:
    before = encoder_input_norm[key].clone()
    after = obs_encoder.key_transform_map[key](before.clone())
    transformed_inputs[key] = after

    show_tensor(f"{key}: before key_transform_map", before)
    show_rgb_per_channel(f"{key}: before key_transform_map", before)

    show_tensor(f"{key}: after key_transform_map", after)
    show_rgb_per_channel(f"{key}: after key_transform_map", after)


# =========================
# 10. 最终跑一次 obs_encoder.forward
# =========================
# 这一步不训练，只是确认：
# 1) 当前输入能不能正常进 encoder
# 2) encoder 输出 feature 的范围大概怎样
# 3) 低维输入与图像特征拼接后有没有异常值
with torch.no_grad():
    obs_feature = obs_encoder(encoder_input_norm)

print("\n========== obs_encoder output feature ==========")
show_tensor("obs_feature", obs_feature)


# =========================
# 11. 给一个简短结论提示
# =========================
print("\n========== how to read this output ==========")
print("1. 先看 encoder_input_norm 的图像范围：如果它在 [0,1]，说明 encoder 看到的是符合 ImageNet Normalize 前提的输入。")
print("2. 对比 raw_input -> imagenet_norm 和 encoder_input_norm -> imagenet_norm：如果两者很接近，说明图像归一化链路基本正确。")
print("3. 如果 after key_transform_map 的统计接近 raw_input -> imagenet_norm，说明 encoder 内部图像变换工作正常。")
