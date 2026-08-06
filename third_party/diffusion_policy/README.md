# Nero Force-Aware Diffusion Policy

本仓库面向 USB 插入等接触丰富机器人操作，研究如何把低频任务意图、当前接触状态和未来物理交互统一到分层控制流程中。

核心思想是区分两类力信息：

- 当前 `wrench_ext` 用于帮助高层策略判断接触状态，并生成接触敏感的末端位姿参考。
- 未来接触力由后续动力学模型预测，作为底层控制器需要显式跟踪的物理目标。

当前仓库已经完成高层 force-aware Diffusion Transformer、LeRobot v3 数据接入和训练工作区。PINN 未来力预测及 OSC-QP 硬件闭环仍属于后续开发阶段。

## 快速训练

在仓库根目录执行以下完整命令：

```bash
cd /home/rei/mnt/code/lcx/diffusion_policy
conda activate dp

export INSERT_USB_DATASET_PATH=/home/rei/mnt/code/lcx/nero_ws/runs/insert_usb_lerobotv3_dp
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m

python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace
```

其中 `DINOV3_MODEL_PATH` 必须替换成包含 `config.json` 和 `model.safetensors` 的本地 DINOv3 文件夹。默认使用 `cuda:0`、batch size 8、梯度累积 4 次、300 epochs 和在线 W&B 日志。

## 系统结构

```text
Nero 数据采集
  10 Hz wrist RGB                         100 Hz robot state / wrench_ext
          |                                          |
          +-------------- 时间对齐与窗口化 ----------+
                             |
               image [To, 3, 192, 256]
               wrench [To, 8, 6]
                             |
         +-------------------+-------------------+
         |                                       |
  Frozen DINOv3 ViT-S/16                  GRU force encoder
  dense patch tokens                     8 sequential tokens
         |                                       |
         +---- Cross Attention: Q=force, K/V=image
                             |
                  To * 8 fused force tokens
                             |
                Diffusion Transformer expert
                             |
             future EE pose chunk [8, 7]
                             |
        position mean + sign-aligned quaternion mean
                             |
                 low-frequency pose target
                             |
          future force predictor / PINN (planned)
                             |
                  OSC-QP controller (planned)
```

高层策略严格只使用 wrist image 和 `wrench_ext`。关节位置、速度、加速度和力矩不输入高层 Diffusion Policy，而是保留给后续接触动力学预测模块。

## 数据契约

当前 insert_usb 数据集位于：

```text
/home/rei/mnt/code/lcx/nero_ws/runs/insert_usb_lerobotv3_dp
```

数据集包含 121 个 episode、17,178 帧，统一为 10 FPS。

| 字段 | 单帧形状 | 对齐方式 | 用途 |
| --- | ---: | --- | --- |
| `observation.images.wrist` | `[192, 256, 3]` | 10 Hz 主时间线 | DINOv3 视觉输入 |
| `observation.wrench_ext` | `[8, 6]` | 历史窗口 | 高层接触状态输入 |
| `observation.joint` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.velocity` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.acceleration` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.torque` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `action.ee_pose` | `[8, 7]` | 未来窗口 | DP 训练目标，格式为 xyz + xyzw |
| `action.joint` | `[8, 7]` | 未来窗口 | 备用动作定义与消融 |

DataLoader 使用 `action_layout: prechunked`。每个 observation anchor 直接读取已经转换好的未来 `[8,7]` 末端位姿 chunk，不再从后续图像帧重复拼接 action。

## Force-Aware Diffusion Transformer

### 视觉编码

默认 backbone 为：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

DINOv3 在第一阶段完全冻结，并始终保持 eval 状态。相机图像在 LeRobot 中是 `[192,256,3]`，进入模型后为 `[3,192,256]`。

ViT-S/16 对该分辨率生成：

```text
192 / 16 = 12
256 / 16 = 16
12 * 16 = 192 patch tokens
```

模型通过 `interpolate_pos_encoding=True` 将预训练位置编码插值到 `12 x 16` 网格，不需要把图像拉伸或裁剪到 `224 x 224`。

### 力时序编码

每张图像对应一个 `[8,6]` wrench 历史窗口。默认使用 GRU 并保留全部 8 个 hidden state，因此每个 observation 产生 8 个 force query token，而不是压缩成一个 token。

可配置的时序编码器包括：

```text
gru | lstm | transformer | none
```

### Cross Attention

融合方向固定为：

```text
Q = force history tokens
K = DINOv3 patch tokens
V = DINOv3 patch tokens
```

cross-attention 输出保留 force residual。视觉信息只能通过该融合路径进入 diffusion expert，不存在独立的 global image token 旁路。

默认 `n_obs_steps=2`，所以 diffusion context 长度为：

```text
2 observations * 8 force tokens = 16 tokens
```

### Action Expert

Diffusion Transformer 对 `[8,7]` 末端位姿序列进行去噪：

- 训练 diffusion steps：100。
- 默认 DDIM 推理 steps：8。
- action 格式：`[x, y, z, qx, qy, qz, qw]`。
- 推理沿用 `start = To - 1`，当 `To=2` 时选择预测索引 1 到 7。
- 跟踪 target 的位置采用算术平均。
- quaternion 先进行半球符号对齐，再求均值并单位化。

`predict_action()` 返回：

| 键 | 形状 | 含义 |
| --- | ---: | --- |
| `action_pred` | `[B, 8, 7]` | 完整预测 chunk |
| `action` | `[B, 7, 7]` | 按 DP 时序语义选择的执行 chunk |
| `action_target` | `[B, 7]` | 用于控制器跟踪的聚合位姿 |

## Contact-Aware Curriculum Masking

接触判定使用未归一化、物理单位下的原始 `wrench_ext`，默认计算 `Fx/Fy/Fz` 的模长，并在 8 点历史窗口上取最大值。

当前全量数据统计下，默认阈值 `2 N` 会标记约 16.4% 的帧。这个值只是数据驱动的初始配置，正式实验应结合回放或接触标注确认。

仅在训练阶段执行视觉 mask；验证和推理始终使用完整图像。支持两种范围：

```text
current_observation | full_context
```

mask 概率由真实 optimizer update 驱动，而不是 DataLoader batch 数。支持：

```text
constant | linear | cosine | exponential | piecewise
```

默认使用 cosine schedule，从概率 1.0 逐渐降到 0.0，使模型从 force-dominant learning 过渡到 vision-force joint learning。

## 图像增强状态

DataLoader 已具备 resize、random crop、rotation 和 color jitter 接口，但当前 insert_usb 配置没有启用一般图像增强。

DINOv3 的 ImageNet 标准化、位置编码插值和接触课程 mask 不属于常规图像增强。腕部相机几何关系对插入任务很重要，因此在验证时序一致的数据增强前，不默认启用 crop 或 rotation。

## 本地环境

本项目不依赖 Docker。训练环境使用 Python 3.10，并与 `nero_ws` 对齐 NumPy、HDF5、OpenCV、SciPy、MuJoCo 和 Pinocchio 的版本范围。

```bash
conda env create -f conda_environment.yaml
conda activate dp
python -m pip install --no-deps lerobot==0.4.0
```

若环境已经创建，可在仓库根目录执行：

```bash
python -m pip install -e ".[training,test]"
```

`setup.py` 是项目元数据和依赖的唯一来源；`pyproject.toml` 只定义 PEP 517 构建后端和 pytest 配置。

## DINOv3 权重

推荐下载完整的 Hugging Face 格式目录：

```text
dinov3-vits16-pretrain-lvd1689m/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

从 Hugging Face 下载：

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir /path/to/dinov3-vits16-pretrain-lvd1689m
```

训练前设置本地路径：

```bash
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m
```

当前配置设置了 `dino_local_files_only: true`，训练过程中不会联网下载权重。Meta 原始 `.pth` 文件不能直接交给 `AutoModel.from_pretrained()`，需要 Hugging Face 格式的完整目录。

## 训练

### 默认训练

默认数据路径已经写入配置，也可以通过环境变量覆盖。以下命令启动完整训练：

```bash
cd /home/rei/mnt/code/lcx/diffusion_policy
conda activate dp

export INSERT_USB_DATASET_PATH=/home/rei/mnt/code/lcx/nero_ws/runs/insert_usb_lerobotv3_dp
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m

python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace
```

### 不使用在线 W&B

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  logging.mode=offline
```

### Debug smoke test

该命令只运行少量 train/validation batch，用于先检查本地 DINOv3 权重、DataLoader 和显存：

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  training.debug=true \
  logging.mode=offline \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  dataloader.prefetch_factor=null \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  val_dataloader.prefetch_factor=null
```

### Hydra 参数覆盖

常用参数覆盖示例：

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  policy.contact_threshold=2.5 \
  policy.image_mask_scope=current_observation \
  policy.mask_schedule.schedule_type=linear \
  policy.force_temporal_encoder=gru \
  dataloader.batch_size=4
```

主要配置文件：

```text
diffusion_policy/config/train_force_aware_diffusion_workspace.yaml
diffusion_policy/config/task/insert_usb_force_aware.yaml
```

训练工作区单独维护：

- `global_step`：完成的 micro-batch 数。
- `optimizer_step`：真正执行的 optimizer update 数。
- `epoch`：已完成的 epoch 数。

梯度累积结束后才更新 optimizer、LR scheduler、EMA 和课程 mask 调度。以上状态都写入 checkpoint，恢复训练不会重复已完成的 epoch。

## 测试

```bash
pytest -q tests/test_force_aware_diffusion.py
python tests/test_force_aware_workspace.py
python tests/test_hirol_lerobot_v3_dataset.py
```

测试覆盖：

- 接触判定使用原始物理单位。
- 各类 mask schedule 的端点和中间值。
- `current_observation` 与 `full_context`。
- GRU、LSTM、Transformer 和无时序编码器。
- DINOv3 完全冻结且无梯度。
- `To * 8` context token 契约。
- Diffusion loss、sampling 和 action shape。
- quaternion 符号对齐、单位化和均值 target。
- prechunked action 的 anchor 与 latency 行为。
- 梯度累积、optimizer step 和课程调度同步。

## 仓库结构

```text
diffusion_policy/
├── config/
│   ├── task/insert_usb_force_aware.yaml
│   └── train_force_aware_diffusion_workspace.yaml
├── dataset/hirol_lerobot_v3_dataset.py
├── model/
│   ├── diffusion/context_transformer_for_diffusion.py
│   └── vision/
│       ├── contact_curriculum.py
│       └── force_aware_obs_encoder.py
├── policy/force_aware_diffusion_transformer_policy.py
└── workspace/train_force_aware_diffusion_workspace.py

PINN/                         future force prediction development
tests/                        data, model and workspace tests
conda_environment.yaml        local Python 3.10 environment
setup.py                      package metadata and dependencies
```

## 当前开发状态

| 模块 | 状态 |
| --- | --- |
| 10 Hz 均匀主时间线与历史低维窗口 | 已完成 |
| 未来 `[8,7]` EE pose action 转换 | 已完成 |
| LeRobot v3 prechunked DataLoader | 已完成 |
| Frozen DINOv3 dense patch encoder | 已完成，等待本地真实权重训练 |
| GRU force tokens 与 cross-attention | 已完成 |
| Contact-aware curriculum masking | 已完成 |
| Diffusion Transformer action expert | 已完成 |
| quaternion-safe chunk target | 已完成 |
| optimizer-step-aware workspace 与 resume | 已完成 |
| 一般图像增强 | 接口存在，当前未启用 |
| Nero 硬件在线 env runner | 未接入 |
| PINN 未来高频接触力预测 | 开发中 |
| OSC-QP 力矩闭环 | 规划中 |

第一阶段的目标是先验证：在相同 demonstration 数据上，force-aware cross-attention 和接触课程 mask 是否相对纯视觉策略改善接触阶段的动作稳定性。确认高层策略有效后，再接入未来力预测和底层控制闭环。
