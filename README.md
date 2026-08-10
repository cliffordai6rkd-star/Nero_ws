# Nero Dual-Arm Teleop Collection

这个仓库提供 Nero 松灵 PyAgxArm 的一套主从臂遥操数采入口：一台主臂/遥操臂，一台从臂/执行臂。当前实现的是 CAN 协议下的 `master_slave` 模式；`meta_quest3_vr` 和 `keyboard_3d_mouse` 已在配置层预留，但还没有控制器实现。

## 当前能力

- 100 Hz 软件双边主从控制：两臂均保持固件 `follower_mode`，逻辑主臂使用 MIT 零刚度拖动、低阻尼和重力前馈，逻辑从臂可选择 MIT 阻抗或 position 跟踪。
- 从臂关节外力矩反馈：动力学残差经过启动零偏、低通、死区、增益、限幅、变化率限制和启动渐入后反射到主臂。该实时触觉反馈链与下述 checkpoint 摩擦预测链相互独立。
- 主从夹爪遥操，以及关节状态、控制命令、力矩、电流、末端位姿、夹爪和 RGB 图像的 H5 v7 采集。
- 在线双模型外力矩推理：同步计算 `tau_ext_cal/tau_ext_pred` 及对应的 `wrench_cal/wrench_pred`，支持实时显示和 H5 保存。
- 动力学辨识：Fourier 激励轨迹、MuJoCo 预检、真机采集、H5 导入、惯性/摩擦/零偏辨识、URDF/manifest 输出和独立验证。
- 自由空间自动采集：基于真实 H5 已到达路径生成 30 分钟低速/静态覆盖轨迹，执行全帧 MuJoCo 预检、逐帧真机安全联锁和 30,000 帧自动分片。
- 7 轴滚动时域 OSC-QP 力/位姿控制接口。该模块目前可独立调用和 benchmark，尚未接入正式数采机械臂执行链。

当前 180,000 帧自由空间轨迹的 MuJoCo 全量预检已通过；正式 30 分钟真机采集仍应在现场重新核对配置/轨迹指纹、工作区和急停。当前真实配置启用一路腕部 V4L2 相机。多相机写入能力保留，但未启用的相机不会出现在 H5 中。

## OSC-QP 力/位姿控制器

`nero_collection.control` 提供 7 轴滚动时域 OSC-QP 接口。默认控制周期为
`0.01 s`（100 Hz），预测 10 步（100 ms）；每次只执行 `result.first_tau`，下一周期
使用新状态重新求解。`result.tau` 是完整的 `(10, 7)` 未来关节力矩序列。

### Action 定义

OSC-QP 没有接收一个固定长度的扁平 action 向量。上层输入的是当前关节状态和
`OSCTargetTrajectory` 末端任务参考：

| 输入 | shape | 含义 |
| --- | --- | --- |
| `q` | `(7,)` | 当前七轴关节角，rad |
| `dq` | `(7,)` | 当前七轴关节速度，rad/s |
| `target.poses` | `(H,4,4)` | 预测窗口内的末端目标齐次变换矩阵 |
| `target.wrenches` | `(H,6)` | 目标环境作用于工具的 wrench，`[Fx,Fy,Fz,Mx,My,Mz]` |
| `target.twists` | `(H,6)`，可选 | 目标末端线速度和角速度；省略时为零 |
| `target.accelerations` | `(H,6)`，可选 | 目标末端线加速度和角加速度；省略时为零 |
| `measured_wrench` | `(6,)`，可选 | 当前实测环境作用于工具的 wrench，用于目标力反馈修正 |
| `previous_tau` | `(7,)`，可选 | 上一周期实际下发的关节力矩，用于惩罚力矩突变 |

QP 内部优化的 action/决策变量是未来 `H` 步的七轴关节加速度 `(H,7)` 和末端
wrench `(H,6)`。默认 `H=10`，因此共有 `10 * (7 + 6) = 130` 个决策变量。求解后按

```text
tau = M(q) * ddq + h(q, dq) - J(q)^T * wrench
```

得到未来关节力矩 `(H,7)`。机械臂真正执行的底层 action 是 `result.first_tau`，即第一步
七维关节力矩，单位 N·m；它不是关节位置或速度 action。当前模块只提供求解结果，尚未连接
正式数采程序的机械臂下发接口。

```python
import numpy as np

from nero_collection.control import (
    OSCQPController,
    OSCTargetTrajectory,
    PinocchioDynamicsModel,
)

model = PinocchioDynamicsModel(
    "urdf/nero/nero_with_gripper.urdf",
    frame_name="gripper_base",
)
controller = OSCQPController(model)
target = OSCTargetTrajectory.constant(
    pose=target_pose_4x4,
    wrench=target_environment_on_tool_wrench_6d,
    horizon_steps=controller.config.horizon_steps,
)
result = controller.optimize_mpc(
    q=q_7d,
    dq=dq_7d,
    target=target,
    measured_wrench=measured_environment_on_tool_wrench_6d,
    previous_tau=last_commanded_tau_7d,
)
tau_now = result.first_tau
tau_horizon = result.tau
```

位姿输入使用 `4x4` 齐次变换矩阵；位姿误差、twist、acceleration 和 wrench 在
`LOCAL_WORLD_ALIGNED` 中按“前三维线性分量、后三维角分量”的顺序计算。wrench 的符号约定是
“环境施加到末端”；如果输入的是机器人施加到环境的目标力，需要先取反。默认同时约束
URDF 关节位置、速度和力矩，并支持配置加速度、wrench 上下界与摩擦锥。真实机械臂首次
运行前必须把 `torque_limit`、目标力范围和接触坐标系按硬件安全值显式收紧。

修改预测窗口、约束或动力学模型后，可重复测量包含 Pinocchio、QP 构造、OSQP setup
和求解在内的端到端频率：

```bash
python scripts/benchmark_osc_qp.py --horizon 10 --iterations 200
```

## 快速运行

首次配置环境：

```bash
git submodule update --init
python scripts/setup_env.py
conda activate nero
```

`diffusion_policy` 作为 `third_party/diffusion_policy` 子模块固定训练和推理使用的模型源码。
`scripts/setup_env.py` 会把 Nero 和该子模块安装进同一个 Conda 环境，并固定兼容 LeRobot 0.4.0
的 PyTorch 2.7.1、torchvision 0.22.1、TorchCodec 0.5 和 CUDA 12.6 wheel。这样从
`third_party/diffusion_policy` 训练得到的 checkpoint 可以在 `nero_ws` 进程中用同一份
模型类和依赖加载。

克隆新工作区时也可以一次拉取所有子模块：

```bash
git clone --recurse-submodules https://github.com/cliffordai6rkd-star/nero_ws.git
```

更新 DP 版本必须显式提交 Nero 仓库中的子模块指针：

```bash
git -C third_party/diffusion_policy fetch origin
git -C third_party/diffusion_policy checkout <目标提交>
git add third_party/diffusion_policy
```

真实数采 CLI 会在连接机械臂之前自动按照 YAML 的通道和 bitrate 调用
`scripts/setup_can.py`，需要时终端会提示输入 `sudo` 密码。也可以单独执行：

```bash
python scripts/setup_can.py
```

如果只配置某一个接口：

```bash
python scripts/setup_can.py can0
```

配置并检查 can0/can1 是否有 CAN 帧：

```bash
python scripts/check_can_links.py
```

只检查 can1：

```bash
python scripts/check_can_links.py can1
```

`scripts/setup_env.py` 会安装当前项目、`python-can` 和 AgileX `pyAgxArm` SDK。真实机械臂运行前，先确认：

```bash
python -c "import pyAgxArm; print('pyAgxArm OK')"
```

真实机械臂数采。Nero 主从夹爪与各自机械臂共用 can0/can1，不需要独立串口或夹爪 server：

```bash
python scripts/nero_teleop_collect.py --config configs/master_slave_can.yaml
```

正式启动机械臂前，可以独立检查配置中启用的 V4L2 相机：

```bash
python scripts/visualize_cameras.py -c configs/master_slave_can.yaml
```

脚本在一个窗口内并排显示所有 `enabled: true` 的相机，并在各画面左上角标注
配置中的 `name`。终端按 `q`、`Esc` 或 `Ctrl-C` 退出；运行期间会输出每路实际
FPS，退出时输出 p99 和最大帧间隔。当前相机配置示例中的 `side` 和 `wrist` 均请求
`MJPG 640x480@25`，写入 H5 前缩放为 `256x192`。该脚本不会连接机械臂或 CAN。
USB 重插或更换设备后应先确认序列号、实际视频节点和实测帧率。

如果 can0/can1 已由 systemd 或其他进程配置，可以跳过自动步骤：

```bash
python scripts/nero_teleop_collect.py --config configs/master_slave_can.yaml --skip-can-setup
```

`--backend mock` 会自动跳过 CAN 配置。

主臂模式切换验证：

```bash
python scripts/check_master_modes.py --config configs/master_slave_can.yaml
```

如果需要进一步验证主臂能否 `enable` 或进入 `follower_mode`，显式加参数：

```bash
python scripts/check_master_modes.py --enable
python scripts/check_master_modes.py --include-follower-mode
```

单臂实时打印关节角：

```bash
python scripts/print_arm_q.py --channel can1
```

如果要直接读取配置里的从臂：

```bash
python scripts/print_arm_q.py --config configs/master_slave_can.yaml --role follower
```

检查 Nero 真实关节使能/驱动状态：

```bash
python scripts/check_nero_enable_status.py --channel can0
python scripts/check_nero_enable_status.py --channel can1
```

没有硬件时跑通 H5 写入：

```bash
python scripts/nero_teleop_collect.py \
  --config configs/master_slave_can.yaml \
  --backend mock \
  --episode-limit 1 \
  --dry-run-duration 2.0
```

当前环境里的 `h5py` 如果报 `numpy.dtype size changed`，说明 numpy/h5py 二进制版本不匹配。建议在采集环境里重装依赖：

```bash
python -m pip install --upgrade --force-reinstall "numpy>=1.23,<3" "h5py>=3.11" PyYAML
```

## 软件双边主从控制

正式数采不再使用固件封装的 `leader_mode` 主从逻辑。两台物理臂始终保持固件 `follower_mode`；上位机始终向逻辑主臂发送 `move_mit()`，逻辑从臂可选择 MIT 或 position：

| 软件角色 | MIT 行为 |
| --- | --- |
| 逻辑主臂 | `kp=0`、低 `kd`、重力前馈，并叠加从臂外力矩反馈 |
| 逻辑从臂 | `mit` 时使用相对镜像目标、关节阻抗和重力前馈；`position` 时直接发送限幅后的相对镜像位置 |

相对映射为 `q_follower_des = q_follower_0 + position_scale * (q_leader - q_leader_0)`，再经过关节限位与 `joint_step_limit_rad` 限幅。启动时程序将两臂切到固件 follower、使能并共同移动到 `follower.rest_q`；按 `r` 或 `t` 只建立软件 reference 和启动双边 MIT，不再切换主臂固件角色。

### 单臂固件模式检查脚本

`scripts/check_master_modes.py` 是独立的固件诊断工具，不属于正式数采控制链。它一次只连接配置中的一台机械臂；`--role leader` 选择当前 pair 的物理主臂，`--role follower` 选择从臂。

默认执行以下检查流程；它会切换模式，但不会使能机械臂，也不会下发运动指令：

```text
connect
  -> 尝试读取 normal joint q
  -> set_normal_mode
  -> set_leader_mode
  -> 读取 leader joint q
  -> disconnect
```

检查配置中的主臂：

```bash
python scripts/check_master_modes.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --role leader
```

加上 `--enable` 后，会在 `set_normal_mode` 和 `set_leader_mode` 之间调用 `enable()`。加上 `--include-follower-mode` 后，还会执行一次 `leader_mode -> follower_mode -> leader_mode` 往返切换：

```text
connect
  -> set_normal_mode
  -> enable                         # 仅当指定 --enable
  -> set_leader_mode -> 读取主臂 q
  -> set_follower_mode              # 仅当指定 --include-follower-mode
  -> set_leader_mode -> 再次读取 q
  -> disconnect
```

例如，完整检查配置中的从臂能否使能并往返切换模式：

```bash
python scripts/check_master_modes.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --role follower \
  --enable \
  --include-follower-mode
```

### Leader 模式阻抗探测

`scripts/probe_leader_impedance.py` 用于检查 V120 在收到 MIT 命令后是否仍保持
`leader_mode`。脚本默认保留机械臂当前使能状态；仅在机械臂尚未使能时才显式添加
`--enable`，该选项在首次使能失败时可能按 adapter 的既有逻辑执行 reset 和清错。
先运行不带授权参数的只读检查：

```bash
python scripts/probe_leader_impedance.py \
  --config configs/master_slave_can.yaml \
  --pair main
```

确认机械臂远离关节限位、急停可触达后，发送一次零增益 MIT 命令：

```bash
python scripts/probe_leader_impedance.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --zero-probe
```

只有输出 `ZERO-PROBE PASS` 且 `after zero MIT: role=leader` 时，才继续在第七轴进行
短时小阻尼测试：

```bash
python scripts/probe_leader_impedance.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --zero-probe \
  --joint 7 \
  --damping-kd 0.05 \
  --duration-s 5
```

测试期间缓慢拖动指定关节，对比零阻尼时的手感，并观察打印的模式、电流和力矩。
也可以在第七轴持续发送严格限幅的前馈力矩，以更直接地检查 leader 模式是否执行
`t_ff`：

```bash
python scripts/probe_leader_impedance.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --zero-probe \
  --joint 7 \
  --torque-ff 0.1 \
  --duration-s 2
```

`--torque-ff` 单位为 N·m，可用负值检查反方向，绝对值被限制在 `3.0 N·m`；它与
`--damping-kd` 互斥。测试开始前应轻扶机械臂，不要靠近 joint7 限位。
脚本在正常结束、异常和 `Ctrl-C` 时都会尝试发送零增益 MIT 命令。如果零增益命令使
模式变成 follower 或模式无法确认，脚本会停止，不会执行非零阻尼测试。

也可以只切换模式并持续观察单臂关节角：

```bash
# 主臂进入 leader_mode，读取 leader joint q
python scripts/print_arm_q.py \
  --config configs/master_slave_can.yaml \
  --role leader \
  --set-leader-mode \
  --source leader

# 从臂进入 follower_mode，读取 normal joint q；该命令不会 enable 从臂
python scripts/print_arm_q.py \
  --config configs/master_slave_can.yaml \
  --role follower \
  --set-follower-mode \
  --source normal
```

运行模式检查前应停止数采程序，避免两个进程同时占用同一 CAN 机械臂。`--enable` 和主臂上的 `--include-follower-mode` 可能改变机械臂的受力/可拖动状态，只能在工作区清空、急停可用时执行。检查脚本退出时只断开连接，不主动调用 `disable()`。

## 终端交互

启动后程序会：

1. `log.info` 打印机械臂启动、连接和输入设备检查。
2. 打印两臂当前角色，将主臂和从臂都置为固件 `follower_mode` 并使能，使上位机可以向两臂发送 MIT 命令。
3. 两臂几乎同时执行 `move_j(follower.rest_q)`，然后分别等待完成并检查平均关节误差。
4. 复位完成后两臂保持 follower，终端等待 `r`、`t` 或 `q`。
5. 按 `t` 建立相对镜像 reference，并启动软件双边 MIT；逻辑主臂使用零位置刚度、低阻尼、重力补偿和从臂反射力矩，固件角色仍保持 `follower_mode`。
6. 按 `r` 进入遥操录制；如果已经按过 `t`，则沿用当前 reference，不切换固件角色。
7. 每次进入控制都会先验证两臂支持 `move_mit()`，再配置持久 MIT motion mode。
8. 遥操过程中按空格停止，随后按 `y` 保存或按 `n` 丢弃。
9. 当 `reset_after_episode: true` 时，无论按 `y` 还是 `n`，两臂都会退出双边控制并共同复位。

当 `tau_ext_inference.enabled: true` 时，每个对齐后的 follower 样本只执行一次双模型推理。控制命令先经过关节步长限制，再按 ZOH 语义作为该时刻的 `q_cmd`；`tau_f` 与 `tau_next` checkpoint 共享同一份 `[q, dq, q_cmd-q]` 特征。系统从两个 checkpoint 分别恢复 LSTM/GRU 结构、输入顺序、训练 horizon 和归一化统计，并以独立固定窗口推理。

在线加速度不读取 `ArmState.ddq`，而是使用与 PINN `offline_tau_labels.py` forward pass 相同的变步长常加速度 Kalman Filter：状态为 `[q,dq,ddq]`，观测为实测 `[q,dq]`，过程噪声为连续 white-jerk，并在时间间隔超过 `max_gap_s` 时复位。RTS 是使用未来观测的离线 backward smoother，只用于构造训练标签，不能进入在线控制。RNEA 保留实测 `q/dq`，只使用 `ddq_kf_causal`：

```text
tau_id_raw      = RNEA(q, dq, ddq_kf_causal)
tau_id_filtered = causal_lowpass(tau_id_raw)
tau_ext_cal_raw = tau_id_filtered + tau_f_pred - tau_follower_filtered
tau_next_target = tau_next_checkpoint_target_filter(tau_follower_filtered)
tau_ext_pred_raw= tau_next_pred - tau_next_target
tau_ext_cal     = lowpass(moving_average(tau_ext_cal_raw, 21), 20 Hz)
tau_ext_pred    = lowpass(moving_average(tau_ext_pred_raw, 21), 20 Hz)
```

`tau_f` checkpoint 必须声明 `matched_causal_torque_filter_v1` target contract 以及训练
标签使用的 cutoff/median window。启动时会验证它们与在线实测 torque 滤波配置完全
一致；旧 checkpoint 或参数不匹配会直接拒绝启用。`tau_follower` 和 `tau_id` 使用两个
独立但同参数、同时间戳、同 episode reset 的滤波状态，避免重复过滤实测 torque。
`tau_next` checkpoint 的 target filter 只在共享源滤波之后作用于 `tau_ext_pred` 分支的
测量 torque；当前配置复现三点因果中值。`tau_next_pred` 不会再次过滤，`tau_f` 分支也
不会经过这一级，因此不改变 `10 Hz/median1` matched contract。

`tau_ext_filter` 与 PINN 数据推理保持一致：每条 raw residual 只经过一次 21 点因果滑动均值和 20 Hz 一阶因果低通，启动窗口以第一个真实 residual 填充。`tau_ext_pred` 用于主臂双边力反馈，override 路径不会再进入双边控制器的旧 residual 低通。两条处理后的外力矩分别经同一个 Jacobian 阻尼最小二乘映射得到 `wrench_cal/wrench_pred`；原始对照保留为 `*_raw`。绘图独立进程的左列依次显示 `tau_ext_cal/tau_ext_pred`，右列显示 `wrench_cal/wrench_pred`，每次开始录制都会清空全部滤波状态和历史。

两个 checkpoint 按“每个 50 帧训练窗口从零状态开始”训练。在线模式保持相同语义：episode 开始后先采满 50 个真实样本，约 0.5 秒预热期间不执行模型前向且力反馈保持为零；完整窗口到齐后才开始预测，每个窗口都重新创建零循环状态。第 51 帧预测使用第 2-51 帧；新 episode 或控制器重启时清空两个窗口和 Kalman 状态。

程序运行时的终端提示和日志均为英文。

按 `q` 或 `Ctrl-C` 退出。

## 自由空间自动采集

### Tau refinement 三段规划器

[configs/tau_refinement_coverage.yaml](/home/rei/mnt/code/lcx/nero_ws/configs/tau_refinement_coverage.yaml)
针对 `runs/tau_refinement/episode_0000_20260806_144955.h5` 生成严格 100 Hz、30 分钟的补充轨迹。
它不完整重播 590 秒人类时间线，也不外推到未采工作区；所有目标姿态均来自人类已到达数据，目标之间的
最小加加速度连接逐条经过 MuJoCo 检查。

- 810 秒：从 XYZ 体素中选择 128 个代表姿态，以约 `0.10 rad/s` 峰值低速连接。
- 630 秒：选择 96 个分布式姿态，低速移动并保持，补充静止和近静止样本。
- 360 秒：执行 96 次 `0.08-0.30 rad` 高层目标切换；实际 `q_cmd` 使用有界 minimum-jerk
  过渡，而不是向硬件发送不连续位置阶跃。

已生成的轨迹包含 180,000 帧，程序预检结果为工作空间越界 0、自碰撞 0、异常环境接触 0；
`hardware.approved` 仍保持 `false`，必须先在本机 MuJoCo 窗口和真实工作站环境中目视检查：

```bash
python scripts/free_space.py \
  --config configs/tau_refinement_coverage.yaml \
  simulate --reuse-existing --playback-speed 100
```

确认轨迹和现场工作空间后，将该配置中的 `hardware.approved` 改为 `true`，再执行真机采集：

```bash
python scripts/free_space.py \
  --config configs/tau_refinement_coverage.yaml \
  collect
```

长时间切分逻辑保持不变。丢弃开头 0.2 秒后共保存 179,980 帧，自动生成 6 个 H5：前五个各
30,000 帧，最后一个 29,980 帧。每个 H5 metadata 都记录全局轨迹索引、planner、三段名称和样本数。

### 原经验路径规划器

[configs/free_space_coverage.yaml](/home/rei/mnt/code/lcx/nero_ws/configs/free_space_coverage.yaml)
定义了一条严格 100 Hz、总计 1800 秒的无接触覆盖轨迹。不需要主臂示教边界，也不区分
train/validation。规划器读取 `trajectory.reference_h5_path` 中的 `teleop/q_follower`，用 URDF 正运动学
计算每帧 `gripper_base` 的 XYZ，并从已经实机到达的路径生成低速和静止轨迹。当前参考数据是
`runs/nero_refinement/episode_0000_20260726_194959.h5`。

该流程位于常规数采入口的上层：自由空间采集器负责轨迹生成、MuJoCo 预检、轨迹执行和逐帧安全联锁，
底层复用常规数采的机械臂适配器、`EpisodeBuffer`、H5 v7 写入器、在线 `tau_f` 推理以及
episode 编号规则。机械臂、checkpoint、状态处理和最终输出目录均读取 `collection_config` 指向的
`configs/master_slave_can.yaml`。

硬件采集每写入 30,000 帧就结束并保存当前 episode，然后在保持最后一个关节目标期间创建下一
episode。保存和显式垃圾回收的暂停不会计入后续 100 Hz deadline。当前 180,000 帧轨迹按
`output.discard_initial_s: 0.2` 丢弃开头 20 帧，因此会依次生成 6 个 H5，样本数为
`30000, 30000, 30000, 30000, 30000, 29980`。
每个分片都会重置在线滤波器和 GRU/LSTM 状态，并在 `episode_metadata_json` 中记录分片序号及其对应的
全局轨迹索引范围。分片保存后会先等待异步状态对齐器重新产出完整有限状态，再恢复 100 Hz 调度；
短暂缺帧不会直接终止轨迹，超过 `hardware.max_timestamp_gap_s` 仍会触发安全停止。

30 分钟轨迹包含：

- 1368 秒：保留实测路径的时序和连通性，以最高约 `0.20 rad/s` 低速重定时遍历。
- 432 秒：在 XYZ 体素中选取 96 个广泛分布的实测姿态，低速连接并分别停留 1 秒。

参考 H5 识别出的末端范围为 X `[-0.502, -0.055] m`、Y `[-0.314, 0.375] m`、
Z `[0.137, 0.504] m`，凸包体积约 `0.06068 m³`。规划器会校验参考 H5 的所有姿态均满足 URDF
软限位和 MuJoCo 安全检查；不会在未采集过的关节空间边界上自行扩张。

### 1. 生成轨迹

```bash
python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  generate --overwrite

python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  summary
```

生成阶段会读取并校验参考 H5，输出
`calibration/data/empirical_workspace_low_static.npz`。NPZ 保存源 H5 路径、SHA-256 和识别出的工作空间；
`summary` 显示两段时长、关节范围、速度、加速度、XYZ 范围和近静止样本数，不连接机械臂。

### 2. MuJoCo 全量预检

每次修改 YAML 或重新生成 NPZ 后，都必须重新运行完整预检：

```bash
python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  simulate --reuse-existing --headless
```

该命令对 180,000 帧逐帧执行 MuJoCo 前向计算，检查软限位、`dq/ddq`、工作空间、自碰撞和地面碰撞，
并生成 `calibration/data/free_space_preflight.json`。

需要快速目视检查整条轨迹时，去掉 `--headless`。下面的命令会复用已生成的 NPZ，在全帧安全检查后
打开原生 MuJoCo 窗口，以 100 倍速播放一次；30 分钟轨迹约 18 秒播放完成：

```bash
python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  simulate --reuse-existing --playback-speed 100
```

窗口只用于观察，关闭窗口可提前结束播放；真机联锁始终以完整报告为准。只有报告中的 `passed` 为
`true`，且配置和轨迹的 SHA-256 均未变化，真机采集才会启动。

### 3. 一条命令生成、预检并采集

检查预检报告和轨迹摘要后，将 `configs/free_space_coverage.yaml` 中的
`hardware.approved` 设为 `true`。运行前清空机械臂工作空间、移除外部接触和负载，并准备急停。

上层 `run` 命令生成轨迹、执行完整预检，通过后调用现有数采执行逻辑：

```bash
python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  run --overwrite --headless
```

复用已有轨迹并跳过终端 `MOVE` 输入：

```bash
python scripts/free_space.py \
  --config configs/free_space_coverage.yaml \
  run --reuse-existing --headless --yes
```

`--yes` 只跳过人工输入，不绕过 MuJoCo 预检、配置/轨迹指纹或逐帧硬件安全检查。仍可分别使用
`generate`、`simulate` 和 `collect` 调试各阶段。采集过程中按 `Ctrl-C` 可终止；若触发跟踪误差、
软限位、力矩、时间戳或截止时间联锁，未完成轨迹不会写入最终 H5。

### 4. H5 输出和检查

采集器会再次核对配置指纹和每条 NPZ 的 SHA-256；配置或轨迹在仿真后发生变化都会拒绝运动。
真机按 100 Hz 发送从臂位置命令，并逐帧检查跟踪误差、关节软限位、力矩阈值、时间戳间隔和循环截止时间。
自由空间采集器只负责轨迹执行、安全联锁和 30,000 帧 episode 分片；字段计算、H5 schema 和保存全部
复用常规 `EpisodeBuffer` 数采逻辑。输出与常规数采相同，保留 `q_cmd`、
`ddq_kf_causal`、两种模型预测、两种外力矩和轨迹来源元数据。
H5 目录和文件前缀继承 `collection_config` 的 `output` 配置，自动续接常规数采的 episode 编号。轨迹
seed、配置 SHA-256、轨迹 SHA-256、参考 H5 SHA-256 和工作空间范围保存在
`/metadata/episode_json`。
每次轨迹开始后的 `output.discard_initial_s` 秒仍会执行控制、安全检查、状态滤波和在线推理，但不会
写入 H5、更新实时图或保存相机帧。当前配置丢弃前 `0.2` 秒（代码默认值为 `2.0` 秒）；因此轨迹总时长包含该预热段。

采集后可检查最新 H5 的格式、样本数和轨迹元数据：

```bash
python scripts/inspect_free_space_h5.py runs/insert_usb
```

## 配置

入口配置在 [configs/master_slave_can.yaml](/home/rei/mnt/code/lcx/nero_ws/configs/master_slave_can.yaml)。关键字段：

- `teleop.mode`: `master_slave`、`meta_quest3_vr`、`keyboard_3d_mouse` 三种模式名已预留。
- `teleop.protocol`: 当前实现 `can`。
- `teleop.backend`: `pyagxarm` 或 `mock`。
- `teleop.master_slave.arm_pairs`: 主从 pair 列表；默认只需要一个 pair，里面有 `leader` 和 `follower` 两台臂，并在 CAN 下配置 `channel`、`interface`、`bitrate`、`firmware`、`rest_q`。当前官方 `pyAgxArm` Nero 配置主要靠 `channel` 区分 CAN 接口，默认不需要 `can_id`。
- `teleop.command.sample_rate_hz`: 双边控制和数采目标频率。
- `teleop.command.maximum_can_frame_gap_s`: 单个 CAN 状态流允许的最大相邻帧间隔。超过该值立即终止状态采样，且该异常帧不会用于计算 `dq/ddq`；当前配置为 `0.03 s`。
- `teleop.command.control_watchdog_timeout_s`: 双边控制相邻周期或单周期状态读取/计算允许的最大耗时。超过该值会在继续下发命令前触发安全异常；当前配置为 `0.05 s`。
- `output.discard_initial_s`: 常规 episode 或整条自由空间轨迹开始后的预热时长。代码默认 `2.0` 秒，当前 YAML 显式设为 `0.2` 秒。期间控制、安全检查、状态滤波和在线推理继续运行，但不保存 H5 数据、不更新实时图，也不保存相机帧。自由空间内部的 30,000 帧分片不会重复丢弃预热段。
- `teleop.command.state_alignment_delay_s`: 异步 CAN 状态对齐延迟。主时间线使用 `now - delay`，分别从四组关节状态和七路力矩历史中取最近值。
- `teleop.command.control_mode`: 只决定逻辑从臂的下发方式。`mit` 使用关节阻抗并启用 `follower_kp/kd`；`position` 使用位置命令。逻辑主臂在两种模式下都保持 MIT 零刚度拖动和力反馈。
- `teleop.command.joint_step_limit_rad`: 每周期从臂目标最大变化量，用于限制映射跳变。
- `teleop.command.reset_on_start`: 启动后是否让两臂共同回到 `follower.rest_q` 并分别自检。
- `teleop.command.reset_after_episode`: `y` 保存或 `n` 丢弃后是否共同复位两臂；当前配置为 `true`。
- `teleop.command.bilateral_mit.leader_kd`: 逻辑主臂拖动阻尼。过大时拖动沉重，过小时容易振荡；主臂 `kp` 省略并使用固定默认零值。
- `teleop.command.bilateral_mit.follower_kp/kd`: 仅 MIT 从臂使用。`kp` 决定跟踪刚度，`kd` 抑制超调和振荡。
- `teleop.command.bilateral_mit.leader_gravity_scale/follower_gravity_scale`: 两臂 Pinocchio 重力前馈比例。
- `teleop.command.bilateral_mit.force_feedback_sign`: 每轴反射力矩方向，只允许 `+1/-1`，首次实机必须逐轴确认。
- `teleop.command.bilateral_mit.force_feedback_gain`: 反射力矩比例，越大反馈越明显，也越容易不稳定。
- `teleop.command.bilateral_mit.force_feedback_deadband_nm`: 忽略模型误差和小噪声的死区。
- `teleop.command.bilateral_mit.force_feedback_limit_nm`: 实际反馈到主臂的逐轴上限。
- `teleop.command.bilateral_mit.force_feedback_lowpass_hz`: 外力矩反馈低通截止频率；越低越稳但触感延迟更明显。

YAML 中省略的主臂零刚度、位置比例、关节/力矩安全限幅、反馈变化率、启动斜坡、关节限位裕量和复位插值参数继续使用代码中的保守默认值，通常不需要在实机调参时修改。
- `gripper.teleop_enabled`: 是否在遥操期间读取 can0 主夹爪开合宽度并控制 can1 从夹爪。只控制和记录开合宽度，不记录夹爪力或控制模式。
- `gripper.scale`、`offset_m`: 开合宽度映射 `follower_width = scale * leader_width + offset_m`。
- `gripper.min_width_m`、`max_width_m`: 从夹爪命令行程限制，单位 m。
- `gripper.force_n`: 从夹爪命令力，单位 N。首次实机测试应从较小值开始。
- `gripper.command_rate_hz`、`deadband_m`: 从夹爪 CAN 指令频率和最小开合宽度变化。
- `gripper.keepalive_s`: 主夹爪不变时重复发送从夹爪目标的间隔，避免单次命令丢失后不再恢复。
- `realtime_plot.enabled`: 是否打开实时关节数据窗口；代码默认 `false`，当前 YAML 已启用。
- `realtime_plot.window_s`: 滑动时间窗口长度；当前配置为 `20.0` 秒。
- `realtime_plot.update_rate_hz`: 图形刷新频率。它只影响显示，不改变 `teleop.command.sample_rate_hz`。
- `realtime_plot.inverse_dynamics.urdf_path`: 用于计算 `tau_id` 的 Pinocchio 模型。
- `realtime_plot.inverse_dynamics.manifest_path`: 辨识输出的 manifest；当前 `tau_f` 只使用 RNEA 的 `tau_id`，不再把 manifest 中的库仑摩擦、黏性摩擦或关节偏置加进残差。
- `realtime_plot.inverse_dynamics.delay_s`: 兼容旧配置的字段，配置应保持 `0.0`。状态时间线延迟由 `teleop.command.state_alignment_delay_s` 单独控制。
- `realtime_plot.inverse_dynamics.locked_joint_names`: 从完整 URDF 裁剪夹爪关节，使 Pinocchio 模型只保留七个机械臂关节。
- `realtime_plot.inverse_dynamics.gravity_m_s2`: Pinocchio RNEA 使用的基坐标系重力向量。
- `realtime_plot.wrench_mapping.frame_name`: 两条 `tau_ext_*` 雅可比映射使用的末端 URDF frame，当前为 `gripper_base`。
- `realtime_plot.wrench_mapping.reference_frame`: wrench 的表达坐标系；`local` 为末端局部系，`local_world_aligned` 为原点位于末端、轴与基座对齐的坐标系。
- `realtime_plot.wrench_mapping.damping`: 求解 `tau_ext_* = J(q)^T wrench_*` 的阻尼最小二乘系数，用于降低奇异位形附近的数值放大。
- `tau_ext_inference.tau_f` / `tau_next`: 两个 PINN 序列模型的独立 checkpoint 配置；两者缺一不可。
- `checkpoint_path`: 可指向具体 `.pt`，也可指向 checkpoint 目录；目录模式启动时选择文件名中 `val_loss` 最低的文件。相对路径按 YAML 所在目录解析。
- `device`: 每个模型可独立选择 `cpu` 或 `cuda:0`；CPU 推理线程数固定为 `1`。
- `horizon` / `input_keys` / `output_key`: checkpoint 一致性检查。当前两个模型均要求 `horizon=50`、输入 `[q,dq,delta_q]`；输出分别为 `tau_f` 和 `tau`。
- `tau_f` target contract: checkpoint 必须包含 `target_contract=matched_causal_torque_filter_v1` 及 `target_filter.cutoff_hz/median_window`；当前为 `10 Hz/1`。
- `tau_ext_inference.state_estimator.*`: 在线因果 Kalman 的观测噪声、white-jerk 过程噪声、初始协方差和断流复位阈值，必须与 PINN 离线标签配置中的 forward filter 一致。
- `dynamics_processing.*`: 控制 H5 中测得力矩的因果处理；`ddq_kf_causal` 始终由 `tau_ext_inference.state_estimator` 单独产生，不使用旧的 `d(dq)/dt` 作为 RNEA 加速度。
- `cameras[*].backend`: 当前真实相机使用 `v4l2`；每台相机在独立子进程和独立 PyAV/FFmpeg 解码线程中持续抓帧，数采/推理主进程只按需取得最新帧。
- `cameras[*].serial_number` / `device`: 可按 USB 序列号解析 `/dev/video*`，也可显式指定设备节点。当前 `wrist=CC1WC520122`，`external=1111111111111111`。
- `cameras[*].pixel_format`、`width`、`height`、`fps`: 当前双相机配置示例均请求 `MJPG 640x480@25`。H5 使用相机逐帧真实时间戳，不把请求帧率当作实际帧率，也不假定多相机硬件同步。
- `cameras[*].visualize`: 在数采和推理中启用独立进程预览。预览队列只保留最新帧，不反压相机读取、H5 保存或控制计算。
- `cameras[*].exposure_dynamic_framerate`: 可选 V4L2 布尔控制；设为 `false` 可禁止自动曝光通过降低 FPS 延长曝光。显式配置该项需要系统安装 `v4l-utils`。相机启动后会在 3 秒窗口内记录 requested、driver-reported 和 measured FPS，实测低于请求值 90% 时输出 warning。
- `cameras[*].crop`: `[y0,y1,x0,x1]`；`output_size` 是写入 H5 前的 `[width,height]`。OpenCV BGR 会转换为连续内存的 RGB `uint8`。
- `cameras[*].buffer_size`: 设为 `1` 以减少旧帧延迟。每台相机只向数采循环交付最新且尚未交付的帧。
- `cameras[*].frame_timeout_s`: 已启动相机允许的最长无帧时间；连续读取失败或超过该时间会触发数采安全异常。当前腕部相机配置为 `0.5 s`。
- `robot_states`: 支持 `q`、`velocity`、`acceleration`、`ee_pose`、`torque`、`tau_id`、`tau_id_filtered`、`current`。`tau_id_filtered` 已在推理器内部完成与 torque 匹配的滤波，保存时不会再次处理。

CAN 帧间隔、双边控制周期、相机无帧或 MIT 前馈力矩越界任一安全异常发生后，常规数采会停止遥操命令和相机线程，将两臂切回固件 `follower_mode`，重建 CAN 状态时间线，尝试回到 `rest_q`，随后断开连接并以退出码 `1` 结束。该故障处理路径严格禁止调用机械臂 `disable()`；即使复位失败，也只继续断连退出。

未知的 arm 字段会自动放入 `config_kwargs`，传给 `create_agx_arm_config(...)`，所以如果你们本地 PyAgxArm 需要额外 CAN 参数，可以直接加到对应 arm 配置里。

MIT 模式要求安装的 PyAgxArm 提供 Nero `move_mit()`。启动实机前应清空工作区并确保急停可用；先保持较低 `kp`，确认 7 个关节的方向和阻尼均正确后再逐步增加。程序会校验 SDK 公布的参数范围，但参数在范围内不代表对当前负载一定安全。

如果启动时报 `does not expose Nero move_mit()`，升级 PyAgxArm：

```bash
python -m pip install --upgrade "git+https://github.com/agilexrobotics/pyAgxArm.git"
```

## H5 布局

生成文件遵循 `/home/rei/mnt/code/lcx/data/train_episode/wipe_board/wipe_board` 的风格：

- 根属性：常规双边数采使用 `format=factr_multimodal_episode/v9`，并包含 `saved_at_us`
- `/config_yaml`: 本次采集配置原文
- `/teleop/timestamp_us`: 延迟后的固定主时间线；重复目标时刻不重复写入
- `/teleop/q_follower`、`q_cmd`：从臂实测关节角和实际发送给从臂的关节命令
- `/teleop/q_leader`、`dq_leader`、`ddq_leader`：软件逻辑主臂状态
- `/teleop/dq_follower`：V120 `motor_state_*` 官方 velocity 转换到 q/URDF 关节坐标系并按主时间线对齐后的七维速度；转换符号为 `[-1,-1,-1,-1,-1,+1,-1]`
- `/teleop/ddq_follower`：滤波后的官方 velocity 按真实 motor 时间差求一阶导、再低通后的七维加速度
- `/teleop/ee_pose_follower`
- `/teleop/tau_follower`、`current_follower`
- `/teleop/ddq_kf_causal`：与 PINN forward-KF 一致的在线因果加速度
- `/teleop/tau_id`：`RNEA(q_follower,dq_follower,ddq_kf_causal)`
- `/teleop/tau_id_filtered`：与 `tau_follower` 同参数独立因果低通后的 `tau_id`
- `/teleop/tau_f_pred`：摩擦/残差 checkpoint 的在线预测
- `/teleop/tau_next_pred`：自由空间总力矩 checkpoint 的在线预测
- `/teleop/tau_ext_cal_raw`、`tau_ext_pred_raw`：未经 residual 后处理的两条外力矩
- `/teleop/tau_ext_cal`、`tau_ext_pred`：各自的 raw residual 经过一次 `21 点因果滑动均值 -> 20 Hz 一阶低通`；`tau_ext_pred` 同时用于双边力反馈
- `/teleop/wrench_cal_raw`、`wrench_pred_raw`：由 raw residual 映射的诊断 wrench
- `/teleop/wrench_cal`、`wrench_pred`：分别由处理后的两条外力矩经 `tau_ext_* = J(q)^T wrench_*` 阻尼最小二乘映射得到，frame 为闭合夹爪指尖中心 `gripper_tcp`，shape 为 `(N,6)`，分量顺序为 `Fx,Fy,Fz,Mx,My,Mz`
- `/teleop/gripper_follower`、`gripper_cmd`：从夹爪实际开合宽度和实际发送给从夹爪的命令，单位均为 m
- `/cameras/<name>/frames` 和 `/cameras/<name>/timestamp_us`，仅在相机实际接入时写入

单个主从 pair 时，`q_follower`、`q_leader`、`q_cmd` 的 shape 是 `(N, 7)`，`ee_pose_follower` 的 shape 是 `(N, 4, 4)`。多 pair 时 joint vector 按 `/teleop` 的 `arm_names` 属性顺序拼接。

`joint_12`、`joint_34`、`joint_56` 和 `joint_7` 保留原始 q 历史；七个 `motor_state_*` 分别维护官方 velocity 历史。保存和控制使用的 `q[k]` 始终是原始关节角，`dq` 不再由 q 差分：

```text
dq[k]         = lowpass(motor_state_velocity[k] * joint_sign)
ddq_raw[k]    = (dq[k] - dq[k-1]) / dt_motor
ddq[k]        = lowpass(ddq_raw[k])
```

第一帧用当前 q、已转换到关节坐标系的官方 velocity 和零加速度初始化。之后每个 motor CAN 帧先低通 velocity，再按相邻 motor 帧的真实时间差计算普通状态字段 `ddq_follower`；它保留作观测诊断，不参与外力矩 RNEA。外力矩链路使用独立的 `ddq_kf_causal`。主时间线使用 `now - state_alignment_delay_s`，从 q 组历史和七个 motor 历史选择最近样本；力矩和电流来自相同 motor 样本。同一主时间格的快照、双模型窗口和 Kalman 只推进一次。实时数采和推理不进行需要未来样本的插值；超过配置上限的 CAN 间隔由 watchdog 拒绝。

H5 额外保存 `control_timestamp_us`、`state_acquired_timestamp_follower_us`、`q_source_timestamp_follower_us`、`motor_source_timestamp_follower_us` 和 `state_source_skew_follower_us`。`teleop` 属性中的 `input_frame_count` 与 `duplicate_input_frame_count` 用于核对控制循环重复读取；PINN 特征映射未声明这些字段时不会把它们作为训练输入。

当前 `dynamics_processing.enabled: false` 且 `robot_states.torque.lowpass: true`，因此 H5 与双模型公式中的
`tau_follower` 使用一条 10 Hz 因果一阶 IIR，`tau_id_filtered` 使用参数相同但状态独立的
另一条 IIR。若启用 `dynamics_processing`，则改用其 `torque_median_window` 和
`torque_lowpass_hz`；参数必须仍与 checkpoint target contract 一致，所有模式的
`zero_phase` 均为 `false`。

第一个完整对齐状态即可产生 `ddq_kf_causal`；两个 torque prediction 和两条 `tau_ext_*` 在前 49 帧写零，第 50 个真实样本形成完整窗口后才有效。Kalman 启动帧加速度初始化为零，之后所有推理结果与 `teleop/timestamp_us` 逐帧等长对齐。
每个推理数据集都记录实际 checkpoint、horizon、输入顺序、输出字段、网络结构和归一化模式属性。

## Rerun episode 可视化

当前在线实时窗口已经使用新双链路字段。旧的 Rerun episode 脚本仍面向 v7 `/teleop/wrench_ext`，不属于 v9 数采写入契约；读取新文件时应选择 `/teleop/wrench_cal` 或 `/teleop/wrench_pred`。

```bash
python scripts/visualize_h5_rerun.py \
  runs/insert_usb \
  --episode 52
```

默认优先选择 `wrist` 相机。如果目标 H5 实际包含其他相机，可用 `--camera <name>` 指定；也可以保存为 Rerun recording 后单独打开：

```bash
python scripts/visualize_h5_rerun.py \
  runs/insert_usb \
  --episode 52 \
  --save runs/insert_usb/episode_0052.rerun.rrd

python scripts/open_rerun_recording.py runs/insert_usb/episode_0052.rerun.rrd
```

如果环境中曾安装同名但无关的 `rerun` 包，需要先执行 `python -m pip uninstall rerun`，再安装项目声明的 `rerun-sdk`。

## 真实 SDK 接入点

- PyAgxArm 适配器：[nero_collection/arms/pyagx.py](/home/rei/mnt/code/lcx/nero_ws/nero_collection/arms/pyagx.py)
- 主从控制逻辑：[nero_collection/teleop/master_slave.py](/home/rei/mnt/code/lcx/nero_ws/nero_collection/teleop/master_slave.py)
- H5 写入：[nero_collection/h5_writer.py](/home/rei/mnt/code/lcx/nero_ws/nero_collection/h5_writer.py)
- V4L2 相机入口：[nero_collection/cameras.py](/home/rei/mnt/code/lcx/nero_ws/nero_collection/cameras.py)
