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

## 夹爪零点标定

先停止数采进程，再运行交互式标定脚本：

```bash
python scripts/calibrate_gripper.py follower
```

夹爪失能后，手动轻轻闭合到机械零位并按 Enter 写入零点；随后脚本移动到
`0 m`、张开到配置的 `max_width_m` 并打印反馈，最后将夹爪失能。标定主夹爪
时运行 `python scripts/calibrate_gripper.py leader`。

## 在线 torque 推理与力反馈来源

`configs/master_slave_can.yaml` 的 `tau_ext_inference.feedback_source` 支持：

```yaml
tau_ext_inference:
  enabled: true
  feedback_source: tau_other  # 可选 tau_other / tau_free
  tau_other:
    checkpoint_path: ../../PINN/outputs/tau_other_sequence/lstm_causal_derived/checkpoints/epoch_083_val_loss_0.003087.pt
    input_keys: [q, dq, delta_q]
  tau_next:
    checkpoint_path: null
```
- `tau_other` 使用 `tau_g + tau_other_pred - tau_follower`，其中 `tau_g=RNEA(q,0,0)`。
- `tau_free` 使用 `tau_free_pred - tau_follower`，配置块内部名称为 `tau_next`。
- `tau_other` 的 checkpoint 输入严格为 `q/dq/delta_q`，目标为 `tau_measured-tau_g`，其中 `tau_measured=observation.torque`；`tau_id` 单独记录完整的 `RNEA(q,dq,ddq)` 结果。100 Hz 源帧先按 `source_butterworth_filter` 过滤，再按 stride 取 50 Hz。`tau_free` 使用动力学输入时必须同时启用同采样率的 `tau_other` 分支，以共享同一帧状态。
- `tau_ext_inference.inverse_dynamics` 保存计算 `tau_ext_cal` 所需的 URDF/RNEA 参数；`realtime_plot` 累积显示当前 episode 的七轴 `tau_ext_cal/tau_ext_pred` 及其 L1 范数，开始新 episode 时清空。新 H5 不再写入 `wrench_*` 数据集。

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
  --mode mtc \
  --observation-mode recorded \
  --max-steps 200 \
  --output /tmp/nero_contact_wm_mtc.npz \
  --scene-output /tmp/nero_contact_wm_mtc.xml
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

标定夹爪零点
```
python scripts/calibrate_gripper.py follower
```
