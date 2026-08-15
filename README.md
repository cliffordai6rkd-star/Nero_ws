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
    input_keys: [q, dq, delta_q, tau]
  tau_next:
    checkpoint_path: null
```
- `tau_f` 使用 `tau_id_filtered + tau_f_pred - tau_follower`。
- `tau_free` 使用 `tau_free_pred - tau_follower`，配置块内部名称为 `tau_next`。
- `input_keys` 是可选的 checkpoint 契约校验项，允许 `q/dq/delta_q/tau` 的有序子集；省略时直接读取 checkpoint 的 `model.inputs`。epoch 83 使用四路输入，其中 `tau` 与生成监督标签时复用同一个 checkpoint 因果滤波值，不会重复滤波。

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