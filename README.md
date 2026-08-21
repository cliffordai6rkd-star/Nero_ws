## 遥操启动
```
python -m nero_collection.cli --config configs/master_slave_can.yaml
```
## 数采流程
```
按下r 机械臂进入零力拖动模式并开始录制
按下空格 停止录制并使能
输入y保存数据 输入n删除数据
q ctrl c退出
```

## 在线 torque 推理与力反馈来源

`configs/master_slave_can.yaml` 的 `tau_ext_inference.feedback_source` 支持：

```yaml
tau_ext_inference:
  enabled: true
  feedback_source: tau_f  # 可选 tau_f / tau_free
  tau_f:
    checkpoint_path: ../../PINN/outputs/tau_f_sequence/lstm_causal_derived/checkpoints/epoch_083_val_loss_0.003087.pt
    input_keys: [q, dq, ddq, delta_q, tau, tau_id]
  tau_next:
    checkpoint_path: null
```
- `tau_f` 使用 `tau_id_filtered + tau_f_pred - tau_follower`。
- `tau_free` 使用 `tau_free_pred - tau_follower`，配置块内部名称为 `tau_next`。
- `input_keys` 是可选的 checkpoint 契约校验项，允许 `q/dq/ddq/delta_q/tau/tau_id` 的有序子集；省略时直接读取 checkpoint 的 `model.inputs`。100 Hz 源帧先按 `source_butterworth_filter` 过滤，再按 stride 取 50 Hz；`ddq` 由过滤后的 `q/dq` 经因果 Kalman 得到，`tau_id` 是直接的 `RNEA(q,dq,ddq)` 输出。`tau_id_filtered` 仅用于 `tau_f` 残差公式。`tau_free` 使用这两个动力学输入时必须同时启用同采样率的 `tau_f` 分支，以共享同一帧结果。
- `tau_ext_inference.inverse_dynamics` 保存计算 `tau_ext_cal` 所需的 URDF/RNEA 参数；`realtime_plot` 只显示七轴 `tau_ext_cal/tau_ext_pred` 及其 L1 范数。新 H5 不再写入 `wrench_*` 数据集。

## 轨迹规划器
`joint_pose_coverage.yaml`从人类遥操数据集路径生成轨迹:
```
  joint_range_source_directory: ../runs/next_background_data
```
生成命令
```
python scripts/free_space.py --config configs/joint_pose_coverage.yaml
 generate --overwrite
```
仿真预检
```
python scripts/free_space.py --config configs/joint_pose_coverage.yaml
  simulate --reuse-existing --playback-speed 100
```
在`config: joint_pose_coverage.yaml`里将:
```
hardware:
  approved: true
```
真机执行:
```
python scripts/free_space.py \
  --config configs/joint_pose_coverage.yaml \
  collect
```

## h5数据真机重播replay:
- 离线检查
```
python scripts/replay_h5_hardware.py runs/next_background_data/episode_0000_20260813_174333.h5 --dry-run
```
- 真机执行
```
python scripts/replay_h5_hardware.py runs/next_background_dataruns/next_background_data/episode_0000_20260813_174333.h5 --approve-hardware
```
## todo:
- 轨迹规划器数据采集:
  - 数据重播
    ```
    python scripts/free_space.py --config configs/joint_pose_coverage.yaml collect
  <!-- - 范围覆盖 -->
- 历史数据replay
  ```
  python scripts/replay_h5_hardware.py runs/next_background_data/episode_0000_20260813_174333.h5
  ```
- 重新采集自由空间遥操数据
  - 10min 训练集
  - 5min 验证集
- wipe_board双相机30hz遥操数据
- insert_usb双相机30hz遥操数据

## 推理命令

## LeRobot v3 双相机纯 DP 训练

已提供只使用 `observation.images.wrist`、`observation.images.side` 和
`action.ee_pose` 的训练配置。训练实现在独立仓库 `/mnt/code/lcx/diffusion_policy`，
不使用本项目的 `third_party/diffusion_policy`，也不改动 FDP 训练入口：

```bash
cd /mnt/code/lcx/diffusion_policy
export PURE_DP_DATASET_PATH=/mnt/code/lcx/PINN/data/train_episode/wipe_board_lbv3
export DINOV3_MODEL_PATH=/mnt/code/lcx/model/dinov3-vitb16-pretrain-lvd1689m

python train.py --config-dir=diffusion_policy/config \
  --config-name=train_pure_diffusion_transformer_workspace
```

数据中的 `action.ee_pose` 是逐帧 `[7]`，DataLoader 将它组装为 `[8,7]` 训练窗口。
两路相机共享冻结的 DINOv3，Transformer 负责预测 diffusion noise。

```bash
python -m inference.cli --config inference/configs/nero_direct_ik.yaml --check
```

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_direct_ik.yaml \
  --simulation-config calibration/config.yaml \
  --mode q \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_direct_ik.npz \
  --scene-output /tmp/nero_direct_ik.xml
```

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --mode mit \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_contact_wm_mit.npz \
  --scene-output /tmp/nero_contact_wm_mit.xml
```

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --mode q \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_contact_wm_q.npz \
  --scene-output /tmp/nero_contact_wm_q.xml
```

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --mode tau \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_contact_wm_tau.npz \
  --scene-output /tmp/nero_contact_wm_tau.xml
```

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --mode osc_qp \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_contact_wm_osc_qp.npz \
  --scene-output /tmp/nero_contact_wm_osc_qp.xml
```
