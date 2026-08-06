#!/usr/bin/env python3

"""
测试图片调试功能的简单脚本
"""

import torch
import numpy as np
import wandb
from omegaconf import OmegaConf

# 模拟导入
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace

def test_image_debug_functionality():
    """测试图片调试功能"""
    
    # 创建模拟配置
    cfg = OmegaConf.create({
        'training': {
            'debug_images': True,
            'debug_images_count': 2,
            'debug_images_freq': 1,
            'debug_images_keys': None,
            'seed': 42,
            'device': 'cpu'
        },
        'policy': {
            '_target_': 'test'
        },
        'optimizer': {
            'learning_rate': 1e-4
        }
    })
    
    # 创建workspace实例
    workspace = TrainDiffusionTransformerHybridWorkspace.__new__(TrainDiffusionTransformerHybridWorkspace)
    workspace.cfg = cfg
    
    # 创建模拟batch数据
    batch_size = 4
    batch = {
        'obs': {
            'agentview_image': torch.rand(batch_size, 3, 84, 84),  # RGB图像
            'robot0_eye_in_hand_image': torch.rand(batch_size, 3, 84, 84),  # RGB图像
            'robot_state': torch.rand(batch_size, 10, 14),  # 非图像数据
        },
        'action': torch.rand(batch_size, 10, 7)
    }
    
    # 测试不同的图像数据范围
    test_cases = [
        ("range_0_1", torch.rand(batch_size, 3, 84, 84)),  # [0, 1]
        ("range_neg1_1", torch.rand(batch_size, 3, 84, 84) * 2 - 1),  # [-1, 1]
        ("range_0_255", torch.rand(batch_size, 3, 84, 84) * 255),  # [0, 255]
    ]
    
    print("Testing image debug functionality...")
    
    for test_name, img_tensor in test_cases:
        print(f"\n测试案例: {test_name}")
        print(f"  - 图像形状: {img_tensor.shape}")
        print(f"  - 数值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # 更新batch数据
        batch['obs']['test_image'] = img_tensor
        
        # 模拟wandb运行 (不实际初始化wandb)
        mock_wandb_run = type('MockWandbRun', (), {
            'log': lambda self, data, step: print(f"    -> 模拟上传到wandb: {len(data)} 个图像组, step={step}")
        })()
        
        # 调用图片调试函数
        try:
            workspace.log_batch_images_to_wandb(batch, mock_wandb_run, step=0, prefix="test")
            print(f"    -> ✅ 成功处理 {test_name}")
        except Exception as e:
            print(f"    -> ❌ 处理失败 {test_name}: {e}")
    
    print("\n测试灰度图像...")
    batch['obs']['grayscale_image'] = torch.rand(batch_size, 1, 84, 84)
    try:
        workspace.log_batch_images_to_wandb(batch, mock_wandb_run, step=1, prefix="gray")
        print("    -> ✅ 灰度图像处理成功")
    except Exception as e:
        print(f"    -> ❌ 灰度图像处理失败: {e}")
    
    print("\n测试debug_images=False...")
    cfg.training.debug_images = False
    workspace.log_batch_images_to_wandb(batch, mock_wandb_run, step=2, prefix="disabled")
    print("    -> ✅ debug_images=False 时正确跳过")
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    test_image_debug_functionality()