"""
通用的WandB调试日志记录工具
支持记录图像、状态、动作等训练调试信息
"""
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def log_debug_info(
    step_log: Dict[str, Any],
    obs_dict: Dict[str, torch.Tensor],
    gt_action: torch.Tensor,
    pred_action: torch.Tensor,
    epoch: int,
    max_samples: int = 4
) -> None:
    """
    记录训练调试信息到WandB

    Args:
        step_log: 日志字典，会添加调试信息
        obs_dict: 观测字典，包含图像和状态
        gt_action: 真实动作 [batch, time, action_dim]
        pred_action: 预测动作 [batch, time, action_dim]
        epoch: 当前epoch
        max_samples: 最大记录样本数
    """
    if wandb.run is None:
        return

    try:
        batch_size = gt_action.shape[0]
        log_batch_size = min(max_samples, batch_size)

        # 1. 记录图像
        _log_images(step_log, obs_dict, log_batch_size)

        # 2. 记录状态信息
        _log_states(step_log, obs_dict, log_batch_size)

        # 3. 记录动作对比
        _log_actions(step_log, gt_action, pred_action, epoch, log_batch_size)

        # 4. 记录统计信息
        _log_statistics(step_log, gt_action, pred_action)

    except Exception as e:
        print(f"WandB debug logging error: {e}")


def _log_images(step_log: Dict, obs_dict: Dict, log_batch_size: int) -> None:
    """记录图像观测"""
    wandb_images = []

    # 可能的图像键名
    image_keys = ['ee_cam_color', 'side_cam_color', 'third_person_cam_color', 'image']

    for img_key in image_keys:
        if img_key in obs_dict:
            images = obs_dict[img_key][:log_batch_size]  # [batch, time, C, H, W]

            for i in range(images.shape[0]):
                # 取最后一帧图像
                img = images[i, -1].cpu().numpy()  # [C, H, W]

                # 处理图像格式
                if img.shape[0] == 3:  # RGB
                    img = img.transpose(1, 2, 0)  # CHW -> HWC

                # 确保像素值在合理范围内
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                wandb_images.append(
                    wandb.Image(img, caption=f"{img_key.replace('_', ' ').title()} - Sample {i}")
                )

    if wandb_images:
        step_log['debug/sample_images'] = wandb_images


def _log_states(step_log: Dict, obs_dict: Dict, log_batch_size: int) -> None:
    """记录状态信息"""
    state_keys = ['state', 'robot_state', 'joint_pos']

    for state_key in state_keys:
        if state_key in obs_dict:
            states = obs_dict[state_key][:log_batch_size, -1].cpu().numpy()  # 最后一个时间步

            # 创建状态表格
            columns = [f"dim_{j}" for j in range(states.shape[1])]
            step_log[f'debug/{state_key}_table'] = wandb.Table(
                columns=["sample"] + columns,
                data=[[i] + states[i].tolist() for i in range(states.shape[0])]
            )
            break


def _log_actions(
    step_log: Dict,
    gt_action: torch.Tensor,
    pred_action: torch.Tensor,
    epoch: int,
    log_batch_size: int
) -> None:
    """记录动作对比图"""
    gt_np = gt_action[:log_batch_size].cpu().numpy()
    pred_np = pred_action[:log_batch_size].cpu().numpy()

    # 创建动作对比图
    n_samples = min(4, log_batch_size)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Action Comparison - Epoch {epoch}', fontsize=16)

    for i in range(n_samples):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        time_steps = np.arange(gt_np[i].shape[0])
        n_dims = min(6, gt_np[i].shape[1])  # 最多显示6个维度

        colors = plt.cm.tab10(np.linspace(0, 1, n_dims))

        for dim in range(n_dims):
            ax.plot(time_steps, gt_np[i, :, dim],
                   label=f'GT dim{dim}', linestyle='-',
                   color=colors[dim], alpha=0.8, linewidth=2)
            ax.plot(time_steps, pred_np[i, :, dim],
                   label=f'Pred dim{dim}', linestyle='--',
                   color=colors[dim], alpha=0.8, linewidth=2)

        ax.set_title(f'Sample {i}', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    step_log['debug/action_comparison'] = wandb.Image(fig)
    plt.close(fig)

    # 每个样本的MSE
    sample_mses = torch.mean((gt_action[:log_batch_size] - pred_action[:log_batch_size]) ** 2, dim=[1, 2])
    for i, mse in enumerate(sample_mses):
        step_log[f'debug/sample_{i}_mse'] = float(mse.cpu())


def _log_statistics(step_log: Dict, gt_action: torch.Tensor, pred_action: torch.Tensor) -> None:
    """记录统计信息"""
    gt_np = gt_action.cpu().numpy()
    pred_np = pred_action.cpu().numpy()

    # 整体统计
    step_log['debug/gt_action_mean'] = float(np.mean(gt_np))
    step_log['debug/gt_action_std'] = float(np.std(gt_np))
    step_log['debug/gt_action_min'] = float(np.min(gt_np))
    step_log['debug/gt_action_max'] = float(np.max(gt_np))

    step_log['debug/pred_action_mean'] = float(np.mean(pred_np))
    step_log['debug/pred_action_std'] = float(np.std(pred_np))
    step_log['debug/pred_action_min'] = float(np.min(pred_np))
    step_log['debug/pred_action_max'] = float(np.max(pred_np))

    # 按维度统计
    for dim in range(min(6, gt_np.shape[-1])):  # 最多记录6个维度
        step_log[f'debug/gt_action_dim{dim}_mean'] = float(np.mean(gt_np[:, :, dim]))
        step_log[f'debug/pred_action_dim{dim}_mean'] = float(np.mean(pred_np[:, :, dim]))
        step_log[f'debug/action_dim{dim}_mse'] = float(np.mean((gt_np[:, :, dim] - pred_np[:, :, dim]) ** 2))