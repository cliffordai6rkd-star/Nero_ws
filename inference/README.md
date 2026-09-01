# Nero 完整推理

> 开发模块、公共契约和新增 DP/PI0 action expert 的指南见
> [`README_DEVELOPMENT.md`](README_DEVELOPMENT.md)。本文主要记录运行时配置、真机启动、H5 回放和 MuJoCo 使用方式。

## Modular inference components

在线运行时正在收敛到以下稳定边界：

```text
NeroObservationSampler -> ObservationProcessor -> HighLevelPolicy
                       -> ActionScheduler -> WorldModel
                       -> ActionResolver/SafetyGuard -> RobotController
```

公共契约位于 `inference/core/`。`InferenceBase` 只负责生命周期和阶段编排，
模型适配器位于 `inference/policies/`，WM 位于 `inference/world_models/`，控制器
适配器位于 `inference/control/`。现有 `NeroInferenceRuntime` 已经使用独立的
`NeroObservationSampler` 和 `NeroPipelineOutputController`，因此 DP、Contact WM
和 direct-IK 的旧入口仍保持兼容，后续可以逐步替换 `pipeline.py` 内部阶段。

当前迁移已将 DP 的观测历史/时间戳对齐放入 `inference/stages/DPObservationBuffer`，
action chunk 的时间戳推进放入 `ActionPlanExecutor`，DP forward 通过
`policies/dp.DiffusionPolicyAdapter` 调用。WM 未确定时继续使用 `NullWorldModel`；
动作解析和通用限幅可分别注入 `DirectActionResolver`/`BasicSafetyGuard`。

`architecture` 配置块用于选择新的编排层，默认关闭以保持旧 YAML 行为：

```yaml
architecture:
  enabled: false
  policy_type: legacy_pipeline
  world_model_type: none
```

WM 尚未确定时使用 `NullWorldModel`，它不会在基类中引入额外的分支。

相机的 `output_size` 只影响模型输入；开启 `visualize` 后预览默认使用裁剪后的
采集分辨率。需要单独缩放预览时配置 `preview_output_size`：

```yaml
cameras:
  - name: wrist
    width: 640
    height: 480
    output_size: [256, 192]
    preview_output_size: null
    visualize: true
```

推理时的 `tau_ext` 图复用数采的 `RealtimeJointPlotter`，数据由
`inference/diagnostics/tau_ext.py` 适配，不会阻塞状态采样和控制线程。
`realtime_plot.norm` 默认为 `l1` 以保持历史曲线；需要数学上的欧氏范数时设置为 `l2`。

`predictor.enabled` 是执行链路开关：

- `true`：加载 contact/SWM predictor，并按 `execution.mode` 选择 MTC、q 或 tau；
- `false`：不加载 `pinn_checkpoint`，DP action -> IK -> `q`，runtime 通过
  `command_joint_positions()` 下发关节位置。

顶层 `action` 声明 DP checkpoint 的 7 维输出语义：

- `action: eepose`（默认）：`[x,y,z,qx,qy,qz,qw]`，纯 DP 分支通过 IK 执行；
- `action: joint`：绝对七轴 `q`，纯 DP 分支直接执行，不调用 IK。启用 WM 时，整段
  joint action 会先用 URDF 对 `robot.action_frame_name` 做 FK，再以相同的
  `[x,y,z,qx,qy,qz,qw]` pose 契约输入 WM。一般情况下 WM 的 target/current pose
  使用同一 action frame；`contact_world_model_opd_sweep/.../step_00100000.pt` 因训练
  数据契约特殊，推理端临时固定复现 `target=link7, current=gripper_tcp`。

关闭 predictor 时的数据流为：

```text
image + wrench history -> configured DP execution -> future action chunk
future action chunk -> first/mean/all or linear-target pose plan -> frame transform -> damped IK
IK q -> joint-limit + per-cycle joint-step safety -> follower arm
```

`predictor.action_chunk_mode` 同时作用于纯 IK 与 predictor 链路：`first` 使用
chunk 第一帧；`mean` 对位置求均值，并在统一四元数半球后对姿态求均值；`all` 按时间
顺序执行完整 chunk。PINN condition 是否同步裁剪由 `action_condition_fill` 决定。

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
推进，低层 IK 或 PINN/control transport 仍在每个有效机械臂状态周期运行。

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

在线 runtime 的键盘控制：`s` 请求并执行一个完整的有效推理/控制周期，执行后自动
回到等待状态；再次按 `s` 执行下一周期，`c` 恢复连续运行，`i` 重置 episode，`q`
退出。也可以用 `--single-step` 启动后直接处于等待状态：

```bash
python -m inference.cli \
  --config inference/configs/nero_contact_wm.yaml \
  --run --single-step --backend pyagxarm --enable-command
```

单步请求会保留到下一张有效相机帧和 CAN 状态都可用，因此一次 `s` 不会因为传感器暂时
没有新数据而消耗掉；它只对应一次 `pipeline.step()` 和一次硬件命令。

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

`predictor.mode` 提供多类可切换的状态/力预测后端：

- `wrench_gru`：原 V1，GRU 直接预测 future wrench；
- `world_model_v3`：V3 联合预测 future `q/v/a/tau`，再经过 Nero 接触力链得到
  future wrench。V3 内部生成的 wrench 仅用于训练正则，部署时不会作为控制目标。
- `world_model_v4`：V4 只预测 future `q/tau`，再由 checkpoint 内保存的 Nero
  因果状态估计契约重建 future `v/a`，最后经过同一接触力链得到 future wrench。
- `world_model_v5`：V5 从历史末段 `[q,tau,0]` 出发，用条件 Flow ODE 联合生成
  future `q/tau/contact logit`，再因果重建 `v/a` 并经过同一 Nero 接触力链。

当前 `inference/configs/nero_contact_wm.yaml` 和 `nero_swm.yaml` 均收束到
`model/carswm/epoch_0001200.pt`，统一使用原生 `SWMInferencePipeline`。该 checkpoint
的输入是 50 个 100 Hz 的 `q/dq/delta_q/tau` 历史和 8 个 25 Hz、7 维绝对关节 action
token；输出 future `q/dq/delta_q/tau` 及三阶段 contact state。`delta_q` 始终按实际下发的
`q_cmd - q` 构造，输入归一化和 flow 推理步数直接读取 checkpoint 元数据。

旧 `ContactWMInferencePipeline` 仍保留用于历史 `q_tau_contact` checkpoint，但不再用于
上述 `carswm` 文件。使用这两个配置时，通过 `execution.mode` 选择三种执行方式：

- `mtc`：计算 `tau_qv = Kp(q_cmd-q)-Kd*dq+g(q)`，再与 WM 第一帧
  `tau_pred` 按 `mtc_alpha` 融合。`mtc_q_cmd_source: wm_state` 使用 `q_hat`；
  `wm_delta` 使用 `q_hat + delta_q_hat`。固件继续使用配置的 `mit_kp/mit_kd`、
  零速度目标，`torque_target` 只发送 `tau_cmd - tau_pd` 的残差，避免重复叠加
  固件已经计算的位置/速度反馈。
- `q`：对预测 q 做关节限位/单周期步长保护后调用 `command_joint_positions()`。
- `tau`：仅下发限幅/滤波后的 WM `tau_pred`。硬件使用 MIT 报文作为传输
  envelope，但 `q/dq` 反馈增益固定为零，因此不会叠加 `kp/kd`、重力或其它控制项。

`execution.mode: mtc` 只发送残差 `torque_target`；
`tau_command` 是按当前采样的 q/dq 计算出的融合总力矩（触发发送限幅时记录限幅后的值）。

原生 SWM（`predictor.mode: swm` 或 `swm_opd`）使用 PINN checkpoint 中声明的
`q/dq/delta_q/tau` 四路 100 Hz 状态历史，以及 25 Hz 的 7 维绝对关节 action chunk。
运行时不会在这四路输入外追加滤波；checkpoint 内的 normalizer 和模型预处理负责训练时
的变换。`delta_q` 始终按实际下发的 `q_cmd - q` 构造，首个状态样本没有上一条命令时为零。
SWM 的 `dataloader.expert_fps` 是动作推进时钟，必须与 DP checkpoint 的动作时间步一致；
不一致会在启动时拒绝，避免动作 chunk 被错误地按另一频率消费。SWM 输出可以通过
`execution.mode` 选择：

- `q`：下发限幅后的预测 q；
- `mtc`：按 `mtc_alpha` 融合 q/v/gravity 候选与预测 tau，并通过固定 MIT 增益发送残差 feedforward；
- `tau`：仅下发经过限幅/滤波的预测 `tau_pred`，硬件传输时 `kp=kd=0`；

最小配置形态如下：

```yaml
dp_checkpoint:
  path: ../../../model/dp/insert_usb/200ep/optimizer_step=00095000.ckpt
  image_encoder: imagenet
  device: cuda:0
  use_ema: true

pinn_checkpoint:
  path: ../../model/carswm/epoch_0001200.pt
  device: cuda:0
  use_ema: true

action: joint
predictor:
  enabled: true
  mode: swm_opd
  inference_mode: asynchronous
  action_chunk_mode: all
  action_execution_mode: chunk
  action_step_s: null

execution:
  mode: q  # mtc / tau
```

在线启动时需让当前工作区和 PINN 源码都可导入：

```bash
conda run -n nero_dp bash -lc '\
  cd /home/rei/mnt/code/lcx/nero_ws && \
  PYTHONPATH=.:/home/rei/mnt/code/lcx/PINN python -m inference.cli \
    --config inference/configs/nero_swm.yaml \
    --run --backend pyagxarm --enable-command'
```

例如先只检查 checkpoint 和契约：

```bash
python -m inference.cli --config inference/configs/nero_contact_wm.yaml --check
```

## H5 -> MuJoCo 独立仿真分支

仿真分支不会初始化 CAN、相机或真机 runtime。它读取一条 H5 episode，在状态时钟上运行
同一个 DP/contact-WM pipeline，并把输出交给带七个 torque motor 的 MuJoCo 模型动态积分。
`q` 是软件 PD 位置伺服，`mtc` 是固定固件增益下的融合总力矩，`tau` 直接使用
`tau_command`；三种模式都不会直接写 `data.qpos`。

默认每 10 ms 调一次策略、每 1 ms 做 MuJoCo 子步。SWM 接收 50 个 100 Hz 的
q/dq/delta_q/tau 历史和 DP 的 8 个 7 维绝对关节 action token。默认 `recorded` 观测模式用 H5 的 q/dq/ddq/tau
作为策略输入，同时记录仿真状态；要检查闭环漂移可使用 `hybrid_closed_loop`，此时图像和
wrench 仍来自 H5，而 q/dq/tau 改用仿真上一周期状态。

先用无窗口模式验证模型和 scene：

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --dp-checkpoint /path/to/dp.ckpt \
  --pinn-checkpoint /path/to/contact_wm_student.pt \
  --mode mtc --max-steps 200 --output /tmp/nero_mujoco_mtc.npz
```

`--dp-checkpoint` and `--pinn-checkpoint` are optional overrides.  The checked-in
contact/SWM YAML uses paths relative to the repository config directory (for
example `../../model/carswm/epoch_0001200.pt`); replace only the DP path when
using a different policy checkpoint.  The CLI validates both checkpoint files
and the local DINO directory before allocating the models, so a stale path fails
with a short actionable message.  For a pure DP -> IK replay,
use `inference/configs/nero_direct_ik.yaml`; the runner automatically selects
the `q` MuJoCo servo when `predictor.enabled: false`.

其余模式只需替换 `--mode` 为 `q` 或 `tau`。需要图形窗口时加 `--viewer`；在
无显示服务器的机器上保持默认的 headless 模式。输出 NPZ 包含 recorded/simulated q、dq、
ddq、command/applied torque、q/dq/tau target、DP/PINN 更新标记和 MuJoCo 接触计数，生成的
执行器 MJCF 可用 `--scene-output` 保存。

H5 loader 要求 `teleop/timestamp_us`、`q_follower`、`dq_follower`、`tau_follower`、
外力矩（优先 `wrench_ext`，旧录音兼容 `wrench_cal`/`wrench_pred`）以及所选 camera 的
`frames/timestamp_us`。如果旧录音没有 `ddq_follower`，默认用 timestamp-aware 的因果
`dq` 后向差分补出 ddq；严格复现实验可通过 `derive_ddq_if_missing=False` 禁止该回退。arm
名称同时支持 `teleop` 属性和旧格式的 `metadata/arm_names_json`。相机对状态是因果对齐的
（只使用当前 tick 之前的最新帧），direct-IK 的 wrench 窗口按每个历史图像重新构造为
`[n_obs_steps, wrench_history_steps, 6]`。当前环境若出现 h5py/NumPy ABI 错误，需要先修复
Python 环境，代码会明确报告该依赖错误。

离线 runner 要求 `predictor.inference_mode: open_loop`，这样 DP 在新图像窗口到达时同步
更新，结果不依赖主机线程调度；只有明确接受非确定性时才传 `--allow-asynchronous`。

V3 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> action_target (xyz + xyzw)
q/v/a/tau/wrench history + future action -> world_model_v3 -> future q/v/a/tau
future q/v/a/tau -> RNEA tau_g + frozen tau_other - tau -> tau_ext_cal
tau_ext -> damped J(q)^T inverse -> future target wrench_ext
legacy wrench reference (no longer executable; migrate to contact/SWM q, mtc or tau)
```

V4 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> action_target (xyz + xyzw)
DP physical action -> V4 checkpoint action normalization
q/tau history + normalized future action -> world_model_v4 -> future q/tau
[observed q history, predicted future q]
  -> causal q mean/LPF -> difference -> dq LPF -> difference -> ddq LPF
future q/dq/ddq/tau -> RNEA tau_g + frozen tau_other - tau -> tau_ext_cal
tau_ext -> damped J(q)^T inverse -> future target wrench_ext
legacy wrench reference (no longer executable; migrate to contact/SWM q, mtc or tau)
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
legacy wrench reference (no longer executable; migrate to contact/SWM q, mtc or tau)
```

V1 运行时仍为：

```text
current q/v/a/tau + future action -> wrench_gru -> future target wrench_ext
```

历史 `configs/nero_pipeline.yaml` 已随旧控制链删除。当前配置仍通过
`runtime.collection_config` 指向 `configs/master_slave_can.yaml`，复用其中的：

- 从臂 CAN、固件和 rest pose；
- 腕部相机；
- `tau_other` checkpoint；
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
  -> tau_g + tau_other_pred - tau
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

每个状态样本的 `q/dq/ddq/tau/tau_other/wrench` 都使用同一个固定 tick
`timestamp_us`；`acquired_timestamp_us` 只用于控制侧检查状态是否过期。PyAgx SDK、CAN parser、
状态采样和机械臂指令只存在于独立硬件子进程。主循环忙于 DP 推理或 minimum-jerk 动作执行
时，固定频率状态线程仍持续调用 SDK getter，进程内有界历史保留中间状态；主循环随后按时间顺序消费这些状态，
推进 tau-other/tau-next 两个 50 帧滑动窗口，并把期间产生的每一帧 wrench 补回 open-loop 的带时间戳状态历史。相机帧仍由
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
每个底层状态帧先复制，并由共享的 `source_butterworth_filter` 对 `q/dq/tau/q_cmd` 做因果滤波；
随后 `tau_other` 和 `tau_next` 分别按
`teleop.command.sample_rate_hz / observation_sample_rate_hz` 的整数 stride 取第 0、N、2N…帧。
两路不再按时间戳执行最近邻或 Unix epoch 固定相位重采样；无法得到整数 stride 的配置会在启动时报错。
episode 复位会重启固定频率状态流，并清空两个固定窗口、Kalman 状态和两条 tau_ext 滤波状态，等待新的有限
`q/dq/tau/q_cmd` 有效后才恢复推理。collection runtime 要求 `tau_ext_inference.enabled: true`
并至少配置实际使用的 checkpoint。主从遥操作通过
`tau_ext_inference.feedback_source: tau_other | tau_free` 选择反馈分支：`tau_other` 对应
`tau_ext_cal`，`tau_free` 对应内部历史名 `tau_next` 的 `tau_ext_pred`。两路同时配置时仍会同时推理和记录，
选择项只影响主臂反馈；每路独立执行 history-ready 和 prediction-age 判定，所选分支未就绪时反馈为零。
`tau_other` checkpoint 的输入严格为 `q/dq/delta_q`，不接收测量力矩、`ddq` 或 `tau_id`。
其目标契约为 `tau_other=tau_measured-tau_g`，其中 `tau_measured=observation.torque`、
`tau_g=RNEA(q,0,0)`；在线反馈使用 `tau_g + tau_other_pred - tau_follower`。`tau_id` 仍单独由
因果 Kalman 得到 `ddq` 后计算完整的 `RNEA(q,dq,ddq)`，只用于记录和其他需要动力学输入的分支。

DP 的 observation 也按 checkpoint 时间契约取样：只把新相机 timestamp 加入图像时间线，
以训练配置的 `timestamp_step_sec=0.1` 选择 10 Hz image anchors；每个 anchor 从带时间戳的
wrench ring buffer 中按 `0.1 / wrench_history_steps` 选择对应高频历史（每个 tick 取不晚于
anchor 的最新状态）。控制循环重复使用
同一相机帧时不会再重复写入 DP observation history。多相机 checkpoint 以
`runtime.camera`（当前为 `wrist`）作为 master anchor；其余相机严格选择 timestamp
不晚于该 anchor 的最新帧，复现 wipe-board H5 转换中的 `resample: previous`，不会把
未来 side 帧错配给 wrist。

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
- 每次 PINN 完成后立即更新 world-model 参考；q、MTC 或 tau 控制链直接消费最新参考。
- DP 尚未完成新推理时，控制链持续执行上一次 action；不会等待 DP。

因此实际控制更新速度由 PINN 推理与传输耗时自然决定，不由额外的 QP 离散步长决定。
力矩滤波器使用相邻 `InferenceInput.timestamp_s` 的实测周期更新。

`timing.enabled: true` 时使用 `time.perf_counter()` 统计并按
`timing.report_interval_s` 汇总打印实际控制频率、完整周期最新耗时、PINN/DP
平均推理耗时。汇总打印不会在每个控制周期执行，减少终端 I/O 对实时性的影响。

`tau` 链的第一帧预测不会绕过滤波直接下发。`torque_filter` 在最终力矩限幅内执行：

```text
WM tau_pred
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
`InferenceOutput.tau_unfiltered` 保留限幅后的 WM 原始输出，
`InferenceOutput.tau_command` 是实际准备下发的滤波后输出。

DP checkpoint 需遵循 diffusion-policy workspace 格式，并提供
`policy.predict_action(obs)["action"]` future chunk 和 `action_target`。
每个 action 固定为 `[x,y,z,qx,qy,qz,qw]`。若 chunk 短于 PINN horizon，尾部用最后一个
目标保持，长于 horizon 则截断。

PINN 的 action condition 完全由 checkpoint 内的 `model` 配置恢复：

- `action_condition_mode: absolute_pose`：直接传 DP 的绝对 future pose，与当前
  `wrench_sequence_v1.yaml` 训练预处理一致。
- `action_condition_mode: relative_pose`：在线用历史最后一个关节状态做 Pinocchio FK，
  并在 PINN 所在 GPU 上批量计算
  `[p_future-p_current, q_current⁻¹⊗q_future]`；相对四元数统一为 `qw >= 0`。
- `action_current_frame_name` 指定 DP/PINN action 所属 frame。当前数据对应 `link7`；
  控制链使用 `robot.frame_name: gripper_base`。推理启动时缓存固定的
  `T_link7_gripper_base`，将每个 action pose 转成控制 frame。

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
runtime 会在进入控制链前将 measured/target wrench 旋转至
`LOCAL_WORLD_ALIGNED`。

`predictor.mode: world_model_v3` 对应原生 `DeterministicWorldModelV3.predict()`。
推理端使用 collection 配置中的同一个 tau_other checkpoint、Pinocchio URDF、重力、锁定关节、
末端 frame、参考坐标系和阻尼系数。tau_other 对“实测历史 + V3 预测未来”做一次无状态序列推理，
因此不会修改 runtime 中用于当前实测 wrench 的在线 tau_other GRU 隐状态。V3 的
`history_horizon` 必须不小于 tau_other checkpoint 的训练 horizon（当前均为 50）。

`predictor.mode: world_model_v4` 对应原生 `DeterministicWorldModelV4.predict()`。
V4 checkpoint 的 `model.state_estimator` 保存 `sampling_dt`、q 均值窗口和 q/dq/ddq
三个低通截止频率。推理端反归一化 q/tau 后调用 checkpoint 模型自带的
`reconstruct_future_state()`，训练与部署因此使用同一份 Torch 因果递推实现；实测历史
v/a 仍用于 tau_other 历史，预测未来 v/a 只从预测 q 得到。当前打包训练数据没有每个子采样
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

真机只读运行会连接并使能从臂、启动相机和执行完整推理，但不发送控制命令：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_pipeline.yaml \
  --run
```

确认急停、关节范围、wrench 符号和控制限制后，必须显式添加下列参数才会下发命令。
predictor 开启时按 `execution.mode` 下发（`tau` 模式使用零增益 MIT 传输），关闭时通过位置接口下发 IK `q`：

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

推理、PINN 或控制传输抛出 exception 时，runtime 会优先停止控制循环并立即执行同一套
`follower.rest_q` 复位和使能确认，再清理模型 worker 并退出。只有硬件复位本身失败时，
会尽最大努力切回 follower、读取当前关节位置并发送当前位置保持、再次确认 enable；
异常路径不会主动调用 `disable()`，原始 exception 仍会继续抛出以保留报错堆栈。

不带 `--enable-command` 时保持只读语义，不会为了 episode 边界移动或复位机械臂。
`--check` 永远不会连接机械臂。
