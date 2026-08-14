# Nero 完整推理

`predictor.enabled` 是执行链路开关：

- `true`：保持原链路，DP -> predictor/F expert -> OSC-QP -> `tau_command`；
- `false`：不加载 `pinn_checkpoint`，DP action -> IK -> `q`，runtime 通过
  `command_joint_positions()` 下发关节位置。

关闭 predictor 时的数据流为：

```text
image + wrench history -> configured DP execution -> future action chunk
future action chunk -> first/mean/all or linear-target pose plan -> frame transform -> damped IK
IK q -> joint-limit + per-cycle joint-step safety -> follower arm
```

`predictor.action_chunk_mode` 同时作用于纯 IK 与 predictor/OSC-QP 链路：`first` 使用
chunk 第一帧；`mean` 对位置求均值，并在统一四元数半球后对姿态求均值；`all` 按时间
顺序执行完整 chunk。`all` 推进时，QP pose horizon 会同步裁成当前尚未执行的剩余
chunk；PINN condition 是否同步裁剪由 `action_condition_fill` 决定。

`predictor.inference_mode: open_loop` 使用严格的“采样 -> 同步推理 -> 执行”状态机：先采满
checkpoint 声明的 `n_obs_steps` 个 image/wrench 观测，冻结该窗口并同步执行一次 DP，然后
完整执行所选 action 计划。执行期间到达的图像不会写入下一轮观测；计划结束后清空旧窗口，
重新采满一批才再次推理。相机 freshness 只在采样阶段作为准入条件，动作执行期间不因图像
超过 `maximum_state_age_s` 而中断，但机械臂控制状态检查与控制超时始终有效。`asynchronous` 保留旧 worker
模式；当它与 `all` 组合时，后台结果只缓存为下一计划，不会中断正在执行的 chunk。
在线纯开环批次以实际相机时间戳为锚点，将 CAN 派生的 wrench 历史按时间戳最近邻对齐；
相机先到时等待 CAN 时间线追上，CAN 先到时等待相机。任一匹配的绝对误差超过
`runtime.maximum_observation_alignment_gap_s`（默认 30 ms）时整批不会送入模型，而会继续采样。
每张图像必须匹配 checkpoint 的 `obs_encoder.wrench_history_steps` 个互不重复、时间递增的
CAN 派生样本；例如 checkpoint 声明为 8 时，模型输入严格为每张图像 8 帧 CAN，不会用
重复最近邻样本补成 8 帧。完整批次的 CAN 数量为 `n_obs_steps * wrench_history_steps`。
`predictor.action_step_s: null` 默认读取 checkpoint 的 `task.dataset.timestamp_step_sec`
（当前 checkpoint 为 0.1 秒）；显式正数可以覆盖它。该间隔只控制高层 action waypoint
推进，低层 IK 或 PINN/OSC-QP 仍在每个有效机械臂状态周期运行。

`predictor.action_execution_mode: minimum_jerk_target` 提供单目标平滑轨迹接口。目标仍由
`action_chunk_mode` 决定：`first` 取 chunk 第一帧，`mean` 取聚合位姿，`last` 取最后一帧；
`all` 在该模式下非法。pipeline 从安装 action 时的实际末端位姿出发，用
`s(u)=10u^3-15u^4+6u^5` 做五次时间缩放：位置沿直线推进，xyzw 四元数沿最短半球做
SLERP。该曲线在起点和终点的速度、加速度均为零。轨迹在
`action_interpolation_duration_s` 内分成 `action_interpolation_steps` 段；持续时间为
`null` 时沿用 `action_step_s`，后者也为 `null` 时沿用 checkpoint 观测步长。轨迹执行完成
前不会被异步 DP 结果中断；旧 `linear_target` 配置会自动归一化到 minimum-jerk 实现，
`action_execution_mode: chunk` 完全保持原有行为。

```yaml
predictor:
  action_chunk_mode: first  # 或 mean/last
  action_execution_mode: minimum_jerk_target
  action_interpolation_duration_s: 2.0
  action_interpolation_steps: 200
```

`predictor.action_condition_fill` 控制高频 PINN 如何消费低频 DP action：

- `auto`：`world_model_v5` 使用 `hold`，其他 predictor 使用 `chunk`；
- `hold`：DP 新结果安装时锁存当前安全 action，并重复填满整个 PINN future horizon；
- `chunk`：输入尚未执行的 DP action chunk，长度不足时保持最后一帧补齐。

V5 默认 hold 的锁存值只在 DP 真正更新时改变，不随高频 PINN 周期或 QP action chunk
游标推进而改变。因此 80 Hz V5 在两次低频 DP 更新之间始终收到相同的 `[1,F,7]`
action condition。

严格保持一个 DP action 时应使用 `action_chunk_mode: first`（或 `mean`）配合 `hold`，
保证 QP 和 V5 使用同一个目标。如果 `action_chunk_mode: all` 需要执行完整 action chunk，
应把 `action_condition_fill` 改为 `chunk`；`all + hold` 会让 QP waypoint 与 V5 condition
逐渐不一致，只适合有意进行该消融时使用。

`ik` 配置收敛容差、阻尼、迭代步长和关节限位 margin；IK 不收敛时拒绝下发。
`safety.maximum_joint_position_step_rad` 再限制每个控制周期相对当前 `q` 的最大变化。

DP 采样器由部署配置覆盖，不需要修改 checkpoint：

```yaml
dp_sampling:
  method: ddim             # ddim 或 ddpm
  num_inference_steps: 8
```

切换采样器只替换 inference scheduler；训练 checkpoint 中的 `num_train_timesteps`、
beta schedule、prediction type、模型权重和 normalizer 保持不变。DDIM 通常用于少步快速
推理；DDPM 通常应配置更多 inference steps，且不能超过 checkpoint 的训练扩散步数。

## H5 离线推理与 MuJoCo 播放

`scripts/infer_h5_direct_ik.py` 从 `runs` episode 读取与在线推理相同的观测契约，按最近
时间戳将相机帧对齐到 `teleop` 状态，然后逐相机帧同步执行 DP -> action chunk -> IK。
同步入口保证每个 H5 帧只进入一次 observation history，结果不会随离线读取速度变化；
配置 `open_loop + all` 时也会按照 H5 timestamp 推进完整 chunk，而不会每帧重新推理。

MuJoCo viewer 会同步显示三类诊断信息：机器人使用当帧实际 `q_command`；绿色小方块
显示 DP checkpoint 原始输出的完整 action chunk（未经过 safety clip 或
`first/mean/all` 选择）；橙色大方块显示该 `q_command` 经 MuJoCo FK 得到的实际末端
位置。与该次推理对齐的 RGB 相机帧显示在 viewer 右上角。这样可直接区分“policy 的
整段预测在跳”与“选取/IK/关节步长后下发结果在跳”。

```bash
python scripts/infer_h5_direct_ik.py \
  runs/insert_usb --episode 1 \
  --config inference/configs/nero_direct_ik.yaml \
  --playback-speed 1.0
```

无图形桌面时可只执行 MuJoCo forward/safety 检查：

```bash
python scripts/infer_h5_direct_ik.py \
  runs/insert_usb/episode_0001_20260728_151517.h5 \
  --max-frames 20 --no-viewer
```

默认在 H5 旁保存 `_direct_ik.npz`。v2 格式新增 `dp_action_chunk [N,H,7]`，保存每帧
最新的完整原始 DP chunk；`action [N,7]` 是执行模式选取并经 action safety 处理后的目标；
`q_command [N,7]` 是关节步长限制后实际送入播放/真机接口的命令。此外还包含原始
`q_observed`、完整 IK 解 `q_ik`、时间戳对齐索引和 IK 残差，同时保存生成的
`.scene.xml`。默认配置为 `configs/nero_direct_ik.yaml`，其中不包含
`pinn_checkpoint`，并通过 `dp_checkpoint.dino_model_path` 使用
`/mnt/code/lcx/model/dinov3-vitb16-pretrain-lvd1689m` 的本地 backbone，不访问网络。

`predictor.mode` 提供三种可切换的状态/力预测后端：

- `wrench_gru`：原 V1，GRU 直接预测 future wrench；
- `world_model_v3`：V3 联合预测 future `q/v/a/tau`，再经过 Nero 接触力链得到
  future wrench。V3 内部生成的 wrench 仅用于训练正则，部署时不会作为控制目标。
- `world_model_v4`：V4 只预测 future `q/tau`，再由 checkpoint 内保存的 Nero
  因果状态估计契约重建 future `v/a`，最后经过同一接触力链得到 future wrench。
- `world_model_v5`：V5 从历史末段 `[q,tau,0]` 出发，用条件 Flow ODE 联合生成
  future `q/tau/contact logit`，再因果重建 `v/a` 并经过同一 Nero 接触力链。

V3 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> action_target (xyz + xyzw)
q/v/a/tau/wrench history + future action -> world_model_v3 -> future q/v/a/tau
future q/v/a/tau -> RNEA tau_id + frozen tau_f - tau -> tau_ext_cal
tau_ext -> damped J(q)^T inverse -> future target wrench_ext
latest action pose + target f_ext + measured f_ext -> OSC-QP -> tau_command
```

V4 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> action_target (xyz + xyzw)
DP physical action -> V4 checkpoint action normalization
q/tau history + normalized future action -> world_model_v4 -> future q/tau
[observed q history, predicted future q]
  -> causal q mean/LPF -> difference -> dq LPF -> difference -> ddq LPF
future q/dq/ddq/tau -> RNEA tau_id + frozen tau_f - tau -> tau_ext_cal
tau_ext -> damped J(q)^T inverse -> future target wrench_ext
latest action pose + target f_ext + measured f_ext -> OSC-QP -> tau_command
```

V5 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> future action chunk
q/tau history + normalized future action -> state/action GRU + Cross-Attention
recent [q,tau,0] -> checkpoint-configured 8-step Heun Flow ODE
Flow -> future q/tau/contact logit
future q -> checkpoint causal estimator -> future v/a
future q/v/a/tau -> Nero physical chain -> raw future wrench_ext
sigmoid(contact logit) >= probability_threshold -> hard-gated target wrench_ext
latest action pose + target f_ext + measured f_ext -> OSC-QP -> tau_command
```

V1 运行时仍为：

```text
current q/v/a/tau + future action -> wrench_gru -> future target wrench_ext
```

`configs/nero_pipeline.yaml` 包含执行开关、DP/PINN checkpoint 路径、安全限制、IK、OSC-QP 参数，
以及 `runtime.collection_config`。后者指向 `configs/master_slave_can.yaml`，复用其中的：

- 从臂 CAN、固件和 rest pose；
- 腕部相机；
- `tau_f` checkpoint；
- Pinocchio 逆动力学；
- `tau_ext -> wrench_ext` 阻尼 Jacobian 映射。

`predictor.enabled: false` 时 `pinn_checkpoint` 可从 YAML 中完全省略。开启 predictor 时，
DP/PINN 的网络参数、输入维度、horizon、action condition 模式、末端 frame
和归一化器必须保存在 checkpoint 的 `cfg`/`config` 中，推理配置不会重复声明它们。
`predictor.mode` 只选择 checkpoint 的运行时输出契约，并不重复声明网络结构；切换 mode
时必须同时把 `pinn_checkpoint.path` 换成相应的 V1/V3/V4/V5 checkpoint。

完整在线观测链为：

```text
follower pyAgxArm latest SDK cache
  -> 由 teleop.command.sample_rate_hz 驱动的固定 tick
  -> 同一 tick 的 q/dq/ddq/tau
  -> tau-ext estimate_aligned(timestamp,q,dq,tau,q_cmd)
  -> tau_id - tau - tau_f_pred
  -> tau_ext
  -> damped J(q)^T inverse
  -> measured wrench_ext
```

实机推理使用独立的连续状态流，DP 前向和动作执行不会暂停 CAN 派生量计算：

```text
isolated hardware process
  pyagx getters     -> fixed-rate latest SDK state stream
inference process  <- ordered state history -> causal KF + dual fixed windows
                   -> tau_ext_cal_raw -> one tau_ext filter -> wrench_cal
                   -> bounded wrench history (4096 canonical samples)
main runtime       <- latest state for control
                   <- drain all accumulated wrench samples for DP history
```

每个状态样本的 `q/dq/ddq/tau/tau_f/wrench` 都使用同一个固定 tick
`timestamp_us`；`acquired_timestamp_us` 只用于控制侧检查状态是否过期。PyAgx SDK、CAN parser、
状态采样和机械臂指令只存在于独立硬件子进程。主循环忙于 DP 推理或 minimum-jerk 动作执行
时，固定频率状态线程仍持续调用 SDK getter，进程内有界历史保留中间状态；主循环随后按时间顺序消费这些状态，
推进 tau-f/tau-next 两个 50 帧滑动窗口，并把期间产生的每一帧 wrench 补回 open-loop 的带时间戳状态历史。相机帧仍由
`CameraManager` 连续采集，DP 在形成完整窗口时再按相机时间戳与状态历史对齐；因此
不会重复消费同一个 tick 来填充 checkpoint 所需的历史（例如一张图像对应 8 个互不重复的状态
样本）。

实机 backend 默认启用该流；`mock` backend 默认保留同步读取，仅用于测试和离线回放。
episode reset、startup recovery 和 shutdown 会先停止并清空连续流，再重置两个 torque 窗口、因果 Kalman
与 DP 历史，避免复位运动污染下一段窗口输入。状态流异常或 tau-ext/wrench
计算异常会记录为 stream fault，主循环随后 fail-closed，而不是继续使用旧状态。

相机预览与状态流完全隔离：每台真实 V4L2 相机由独立 `spawn` 子进程持续采集，并将配置中
`visualize: true` 的最新 RGB 帧直接投递给 GUI 子进程。GUI 子进程拥有唯一的 OpenCV 窗口
和事件循环，即使主推理进程正在执行模型，预览仍可刷新。数据与预览队列都有界且非阻塞，
消费者跟不上时只丢弃旧帧；相机帧不会被预览进程重新用于 DP 对齐，
因此不会阻塞或改变 `q/dq/ddq_kf_causal/tau_ext/wrench` 的计算时间线。数采和推理都复用同一个
`configs/master_slave_can.yaml` 相机配置；当前 `wrist` 预览默认开启。

在线 runtime 读取 `runtime.collection_config` 的 `teleop.command.sample_rate_hz`，与数采
`MasterSlaveTeleop` 使用同一固定频率最新缓存语义。外力矩 RNEA 不使用普通状态字段 `ddq_follower`，
而是由与 PINN offline forward pass 一致的因果 Kalman 从同一 tick 的 `q/dq` 估计 `ddq_kf_causal`。
episode 复位会重启固定频率状态流，并清空两个固定窗口、Kalman 状态和两条 tau_ext 滤波状态，等待新的有限
`q/dq/tau/q_cmd` 有效后才恢复推理。collection runtime 要求 `tau_ext_inference.enabled: true`
并至少配置实际使用的 checkpoint。主从遥操作通过
`tau_ext_inference.feedback_source: tau_f | tau_free` 选择反馈分支：`tau_f` 对应
`tau_ext_cal`，`tau_free` 对应内部历史名 `tau_next` 的 `tau_ext_pred`。两路同时配置时仍会同时推理和记录，
选择项只影响主臂反馈；每路独立执行 history-ready 和 prediction-age 判定，所选分支未就绪时反馈为零。
模型输入不在运行时代码中固定为三路：加载器按 checkpoint 的 `model.inputs` 顺序组装
`q/dq/delta_q/tau` 的任意有序子集。配置中的 `input_keys` 仅作为可选的严格校验，省略时采用
checkpoint 自带列表。对于 `causal_rnea_residual_v1`，在线侧还会恢复并校验 checkpoint 中的
Kalman、`dq_sign`、RNEA state source 和 torque filter 契约；滤波后的同一份 `tau` 同时进入模型和
`tau_f=tau_filtered-tau_id_filtered` 残差链，避免二次滤波造成训练/部署偏差。

DP 的 observation 也按 checkpoint 时间契约取样：只把新相机 timestamp 加入图像时间线，
以训练配置的 `timestamp_step_sec=0.1` 选择 10 Hz image anchors；每个 anchor 从带时间戳的
wrench ring buffer 中按 `0.1 / wrench_history_steps` 选择对应高频历史。控制循环重复使用
同一相机帧时不会再重复写入 DP observation history。

启用 `wrench_visualization` 后，独立窗口显示四张在线曲线：第一行是 raw tau_ext 经 Jacobian
反解后的 wrench 合力和六个分量，第二行是一次 tau_ext 滤波后映射、再经过 DP
checkpoint 接触门控后的物理量合力和六个分量。启用 `tau_ext_filter` 时，
`observation_protection` 只负责启动预热，不再重复低通 wrench；关闭 `tau_ext_filter` 后才启用其后备 wrench 滤波。接触门控参数（阈值、force dims、历史
reducer）从 DP checkpoint 恢复；当前 checkpoint 使用 0.6 N、前三个线性力分量和 mean
历史 reducer。DP 模型内部仍在归一化之后执行同一个门控，避免把物理零值提前到
normalizer 之前。

推理循环不配置固定 Hz，也不会 `sleep`：

- `open_loop` 下 DP 仅在 action 计划完成后同步推理一次；`asynchronous` 下才使用独立
  worker，并且仅在出现新的训练频率 image anchor 时提交，避免对同一观测反复采样。
- `wrench_gru` 每个控制周期使用最新机器人状态和完整 future action chunk 推理，
  `forward_step()` 的 GRU 循环状态会跨周期携带。
- `world_model_v3/v4/v5` 独立维护 checkpoint 指定长度的历史
  `q/v/a/tau/wrench` 物理量窗口；
  episode 开头按训练数据的边界规则复制首帧左填充，并按 checkpoint
  状态估计配置中的 `sampling_dt`（旧 V3 回退到 `loss.sampling_dt`）对低维状态做固定
  时间网格插值。归一化输出先恢复为物理量；V3 直接送入接触力链，V4 先从物理 q
  因果重建 v/a。
- 每次 PINN 完成后立即更新完整的 future wrench 目标，随后 OSC-QP 求解一次。
- DP 尚未完成新推理时，OSC-QP 持续跟踪上一次 action；不会等待 DP。

因此实际控制更新速度由 PINN 推理与 OSC-QP 求解的真实耗时自然决定，不由 YAML 中的
名义 100 Hz 决定。第一周期使用 `osc_qp.dt_s` 作为启动值；之后 QP 的预测离散步长会用
相邻 `InferenceInput.timestamp_s` 的实测周期更新。

`timing.enabled: true` 时使用 `time.perf_counter()` 统计并按
`timing.report_interval_s` 汇总打印实际控制频率、完整周期最新耗时、PINN/OSC-QP/DP
平均推理耗时。汇总打印不会在每个控制周期执行，减少终端 I/O 对实时性的影响。

OSC-QP 的第一步输出不会直接下发。`torque_filter` 在最终力矩限幅内执行：

```text
QP first_tau
  -> torque clip
  -> causal median spike rejection
  -> first-order low-pass
  -> hard per-axis slew-rate limit
  -> final torque clip
  -> MIT t_ff
```

`median_window` 必须为正奇数；窗口越大会抑制更长尖峰，但也增加相位延迟。
`lowpass_cutoff_hz: null` 可关闭低通，`rate_limit_nm_s: null` 可关闭硬变化率限制。
滤波器以当前实测关节力矩作为首周期状态，并在每次 `reset()` 时清空历史。
`InferenceOutput.tau_unfiltered` 保留限幅后的 QP 原始输出，
`InferenceOutput.tau_command` 是实际准备下发的滤波后输出。

DP checkpoint 需遵循 diffusion-policy workspace 格式，并提供
`policy.predict_action(obs)["action"]` future chunk 和 `action_target`。
每个 action 固定为 `[x,y,z,qx,qy,qz,qw]`。OSC-QP 跟踪 future chunk；
若 chunk 短于 PINN/QP horizon，尾部用最后一个目标保持，长于 horizon 则截断。

PINN 的 action condition 完全由 checkpoint 内的 `model` 配置恢复：

- `action_condition_mode: absolute_pose`：直接传 DP 的绝对 future pose，与当前
  `wrench_sequence_v1.yaml` 训练预处理一致。
- `action_condition_mode: relative_pose`：在线用历史最后一个关节状态做 Pinocchio FK，
  并在 PINN 所在 GPU 上批量计算
  `[p_future-p_current, q_current⁻¹⊗q_future]`；相对四元数统一为 `qw >= 0`。
- `action_current_frame_name` 指定 DP/PINN action 所属 frame。当前数据对应 `link7`；
  OSC-QP 仍控制 `robot.frame_name: gripper_base`。推理启动时缓存固定的
  `T_link7_gripper_base`，将每个 action pose 转成 OSC 控制 frame 后再优化。

这些选项不会出现在 `inference/configs/nero_pipeline.yaml`；修改 PINN 训练逻辑后必须
使用包含相应配置和权重的新 checkpoint。旧 checkpoint 未声明
`action_condition_mode` 时按训练器历史默认值 `relative_pose` 兼容。

PINN checkpoint 支持 `/mnt/code/lcx/PINN` 的原生格式：
`checkpoint["config"] + checkpoint["model"] + checkpoint["normalizer"]`。也支持配置内含
`policy._target_` 或 `model._target_` 的通用格式。恢复出的模型
应实现 `predict_force(inputs)`（也兼容 `predict(inputs)` 或 `forward(inputs)`）。
`predictor.mode: wrench_gru` 对应原生 `WrenchSequenceGRUV1.forward_step()`，会自动使用 checkpoint 的 `model.inputs`
选择 `q/v/a/tau`，从 checkpoint normalizer 恢复归一化，并以 checkpoint 的
`model.action_key` 接收 future action condition。旧 V1 checkpoint 若未声明
`action_key`，推理端不会额外注入 condition。输入均带 batch 维；
输出为 6D wrench，或包含 `f_ext/wrench/force/force_target/target_wrench` 之一的字典。
wrench 统一采用“环境作用于工具”的 `[Fx,Fy,Fz,Mx,My,Mz]`。DP/PINN 使用
collection checkpoint 训练时的 wrench 坐标系；如果 collection 配置为 `local`，
runtime 会在送入 OSC-QP 前将 measured/target wrench 旋转至
`LOCAL_WORLD_ALIGNED`。

`predictor.mode: world_model_v3` 对应原生 `DeterministicWorldModelV3.predict()`。
推理端使用 collection 配置中的同一个 tau_f checkpoint、Pinocchio URDF、重力、锁定关节、
末端 frame、参考坐标系和阻尼系数。tau_f 对“实测历史 + V3 预测未来”做一次无状态序列推理，
因此不会修改 runtime 中用于当前实测 wrench 的在线 tau_f GRU 隐状态。V3 的
`history_horizon` 必须不小于 tau_f checkpoint 的训练 horizon（当前均为 50）。

`predictor.mode: world_model_v4` 对应原生 `DeterministicWorldModelV4.predict()`。
V4 checkpoint 的 `model.state_estimator` 保存 `sampling_dt`、q 均值窗口和 q/dq/ddq
三个低通截止频率。推理端反归一化 q/tau 后调用 checkpoint 模型自带的
`reconstruct_future_state()`，训练与部署因此使用同一份 Torch 因果递推实现；实测历史
v/a 仍用于 tau_f 历史，预测未来 v/a 只从预测 q 得到。当前打包训练数据没有每个子采样
timestamp，所以该固定网格保证的是训练/部署约定一致，不等价于逐点复现原始异步 CAN
滤波状态。

启用新 V4 checkpoint 的 `contact_gate` 后，模型还会输出 `[1,F,1]` 的
`contact_probability`。推理端先用预测 q/v/a/tau 计算 raw Nero wrench，再从 checkpoint
恢复 `probability_threshold`，逐 future step 对模型预测概率做二分类；非接触点的
六维 target wrench 整体置零。物理力阈值和连续帧确认只用于离线制作训练 label，
在线门控不读取实测 wrench、不执行时间滞回，也不把 contact 状态作为模型输入。
没有 `contact_gate` 的旧 V4 checkpoint 保持原始未门控行为。

`predictor.mode: world_model_v5` 对应原生
`StateToStateFlowWorldModelV5.predict()`。在线不提供 contact 输入；模型在 Flow source
的第 15 维填零，并按 checkpoint 的 `flow_inference_steps` 和 `flow_solver` 从真实历史
q/tau 积分到 future q/tau/contact。当前默认是 8 步 Heun。Flow contact logit 经 sigmoid
后沿用上述 `probability_threshold` 硬门控，物理阈值、滞回帧数和 contact label 仍只属于
离线训练数据处理。

推荐 V5 在线配置：

```yaml
predictor:
  enabled: true
  mode: world_model_v5
  action_chunk_mode: first
  action_condition_fill: auto  # V5 自动采用 hold
```

检查配置并真实恢复两个 checkpoint：

```bash
python scripts/nero_inference.py --config inference/configs/nero_pipeline.yaml --check
```

Mock 端到端运行（仍会真实加载三个 checkpoint）：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_pipeline.yaml \
  --run --backend mock --duration 10
```

真机只读运行会连接并使能从臂、启动相机和执行完整推理，但不发送 OSC-QP 命令：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_pipeline.yaml \
  --run
```

确认急停、关节范围、wrench 符号和控制限制后，必须显式添加下列参数才会下发命令。
predictor 开启时通过 MIT 下发 `tau_command`，关闭时通过位置接口下发 IK `q`：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_pipeline.yaml \
  --run --enable-command
```

启用命令后，runtime 在第一个推理 episode 前使用
`runtime.collection_config` 中对应 `arm_pair` 的 `follower.rest_q` 复位从臂；复位的插值速度、
超时、误差阈值和平均采样数继续复用该 collection config 的 `teleop.command.reset_*`
配置。`runtime.maximum_inference_steps` 是单个 episode 的最大控制步数：达到上限后结束
当前 episode、复位、清空 DP/PINN/滤波历史，再自动开始下一次推理；设为 `null` 可关闭
按步数切分。

运行期间按键：

- `i`：结束当前 episode，复位并开始下一个 episode；
- `q` 或 `Ctrl-C`：结束当前 episode，复位到 `follower.rest_q`，保持 follower enabled，
  然后断开进程并退出；
- `--duration` 到期执行与退出相同的最终复位流程。

推理、PINN 或 OSC-QP 抛出 exception 时，runtime 会优先停止控制循环并立即执行同一套
`follower.rest_q` 复位和使能确认，再清理模型 worker 并退出。只有硬件复位本身失败时，
会尽最大努力切回 follower、读取当前关节位置并发送当前位置保持、再次确认 enable；
异常路径不会主动调用 `disable()`，原始 exception 仍会继续抛出以保留报错堆栈。

不带 `--enable-command` 时保持只读语义，不会为了 episode 边界移动或复位机械臂。
`--check` 永远不会连接机械臂。
