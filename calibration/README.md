# Nero 动力学参数辨识

本目录为nero机械臂动力学参数动态辨识流程。采样轨迹是满足位置、速度、加速度和安全约束的多关节有限 Fourier 激励轨迹。原始 URDF 永远不会被覆盖；辨识结果默认写到同目录的 `*_identified.urdf`。

## 配置参数说明

配置文件是 `calibration/config.yaml`。所有长度为 7 的数组都严格按照 `model.joint_names` 的顺序解释，当前即 `[joint1, joint2, joint3, joint4, joint5, joint6, joint7]`。角度单位为 rad，角速度为 rad/s，角加速度为 rad/s²，力矩为 N·m。修改数组时不能按 URDF 文件中的视觉 link 顺序或 CAN ID 自行重排。

### 模型和硬件入口

| 参数 | 含义 |
| --- | --- |
| `collection_config` | 现有 NERO 采集配置。动态采集从这里取得 follower endpoint、V112、CAN 通道、bitrate 和 PyAgxArm backend。 |
| `pair` | `collection_config` 中要使用的主从臂 pair 名称；辨识只驱动该 pair 的 follower。 |
| `model.urdf_path` | 辨识先验和轨迹限位来源。几何、关节轴、零位及坐标系必须正确。 |
| `model.locked_joint_names` | 构建 7 DoF Pinocchio 模型时锁定的非机械臂关节。当前夹爪关节锁在 Pinocchio neutral 位，辨识后的 joint7 是包含这些锁定末端刚体的聚合惯性。 |
| `model.joint_names` | 硬件反馈数组与 Pinocchio 活动关节的共同顺序。启动前会严格比较，不一致直接拒绝运行。 |
| `model.gravity_m_s2` | 基坐标系中的重力向量。方向错误会直接污染质量、质心和零偏估计。 |

### Fourier 激励

| 参数 | 单位 | 含义 |
| --- | --- | --- |
| `excitation.sample_rate_hz` | Hz | 轨迹离散频率和真机位置命令频率。轨迹文件的时间间隔必须与它一致。 |
| `excitation.duration_s` | s | 每次完整激励的时长。建议使 `duration_s * fundamental_hz` 为整数，减少周期首尾不连续。 |
| `excitation.fundamental_hz` | Hz | Fourier 基频。最高激励频率为 `harmonics * fundamental_hz`。 |
| `excitation.harmonics` | - | 每个关节使用的有限 Fourier 谐波数。增加谐波可提高激励丰富度，也会提高速度、加速度和跟踪要求。 |
| `excitation.optimization_trials` | - | 随机候选轨迹数量。更多候选通常能降低回归矩阵条件数，但生成耗时更长。 |
| `excitation.amplitude_rad` | rad | 七轴相对中心的最大允许振幅。优化器可能进一步缩小实际振幅以满足速度和加速度约束。 |
| `excitation.max_velocity_rad_s` | rad/s | 命令轨迹的逐轴最大解析速度，用来约束 Fourier 系数。V112 下不会用无效固件速度字段验证实机速度。 |
| `excitation.max_acceleration_rad_s2` | rad/s² | 命令轨迹的逐轴最大解析加速度。它是轨迹生成约束，不是实测加速度保护。 |
| `excitation.joint_limit_margin_rad` | rad | 从 URDF 上下限两侧各缩进的安全距离。生成轨迹和采集时的实测 `q` 都必须位于缩进后的范围。 |
| `excitation.max_tracking_error_rad` | rad | 真机逐轴 `abs(q - q_cmd)` 上限。任意一轴超过即中止本次采集。 |
| `excitation.start_move_speed_rad_s` | rad/s | 从当前姿态移动到 Fourier 首点时，minimum-jerk 过渡的峰值速度限制。 |
| `excitation.profiles[*].name` | - | 轨迹点名称，也是 `--profile` 的选择值。 |
| `excitation.profiles[*].role` | - | `train` 表示参与辨识，`validation` 表示只用于验证。 |
| `excitation.profiles[*].seed` | - | 当前点的 Fourier 随机种子。每个 profile 必须不同。 |
| `excitation.profiles[*].center_rad` | rad | 当前点的七轴周期运动中心，必须逐点通过 MuJoCo 和现场检查。 |
| `excitation.profiles[*].repetitions` | - | 真机采集该轨迹的重复次数。 |
| `excitation.profiles[*].trajectory_path` | - | 生成的命令轨迹 NPZ 路径。 |
| `excitation.profiles[*].dataset_path` | - | 真机采集数据 NPZ 路径。 |

`max_velocity_rad_s` 和 `max_acceleration_rad_s2` 约束的是数学命令轨迹；V112 没有可靠的关节速度反馈，因此它们不能证明实际机械臂没有超速或振荡。实际保护还依赖位置跟踪误差、力矩反馈、现场观察和硬件急停。

### 三个训练中心、一个验证中心及修改方法

当前正式流程在 `excitation.profiles` 中定义三个训练中心和一个独立验证中心：

| profile | 用途 | 当前中心 `[J1..J7]` rad |
| --- | --- | --- |
| `train_a` | 第一条训练轨迹 | `[0.30, -0.30, 0.30, 1.55, 0.30, 0.30, 0.30]` |
| `train_b` | 低位训练轨迹，补充重力、质量和质心方向 | `[-0.10, -0.02, -0.10, 1.68, -0.08, 0.12, -0.18]` |
| `train_c` | 高位但 J6 反向的训练轨迹，补充腕部和动态耦合 | `[0.30, -0.28, 0.30, 1.42, 0.30, -0.30, 0.30]` |
| `validation` | 中央构型，不参与参数拟合，只评价跨构型预测能力 | `[0.00, -0.15, 0.00, 1.55, 0.00, 0.05, 0.00]` |

任何 `center_rad`、amplitude、速度或加速度修改都会使之前的安全批准失效。即使配置中的 `safety.approved` 已经是 `true`，新生成的四个 profile 仍必须重新通过 MuJoCo、contact 输出和 `--plan-only` 检查后才能真机采集。

修改某个中心时只编辑对应 profile 的 `center_rad`。数组顺序始终是 `[joint1, ..., joint7]`，不能修改 `model.joint_names` 来改变含义。`seed`、`trajectory_path` 和 `dataset_path` 在四个 profile 间必须保持不同；`role: validation` 的数据不能改成训练输入。

轨迹生成器首先执行逐轴必要条件：

```text
URDF_lower + joint_limit_margin + amplitude
    <= center_rad <=
URDF_upper - joint_limit_margin - amplitude
```

按当前 URDF、`joint_limit_margin_rad=0.10` 和当前 `amplitude_rad`，数学上允许的中心范围为：

| 关节 | center 最小值 rad | center 最大值 rad |
| --- | ---: | ---: |
| J1 | -2.4253 | 2.4253 |
| J2 | -1.4800 | 1.4800 |
| J3 | -2.4700 | 2.4700 |
| J4 | -0.7600 | 1.8900 |
| J5 | -2.4700 | 2.4700 |
| J6 | -0.5200 | 0.7400 |
| J7 | -1.2908 | 1.2908 |

这些只是单关节限位，不代表无自碰撞或无环境碰撞。选择中心时遵循以下原则：

1. 先以已经安全的 `train_a` 为基准，每次只让若干关节移动约 `0.1~0.2 rad`；
2. 训练 B 应重点改变 J2、J4、J7 的构型，同时保持末端在实际工作区内；
3. train_c 应补充 A、B 尚未覆盖的腕部方向或动态组合，不能只是重复 A；
4. validation 应与 A、B、C 都不同，但仍处于未来在线接触估计会使用的区域；
5. 不要仅把同一中心换 seed 当作新中心；它能改变速度/加速度组合，但不能充分增加重力姿态信息；
6. 每次修改后只重新生成该 profile，并完整查看 MuJoCo。

例如修改 `train_b.center_rad` 后应运行：

```bash
python -m calibration.generate_excitation \
  --config calibration/config.yaml \
  --profile train_b
```

这里不要加 `--reuse-existing`，否则会继续加载修改前的 NPZ。只有中心和轨迹配置均未改变、只是重新查看 scene 时才使用：

```bash
python -m calibration.generate_excitation \
  --config calibration/config.yaml \
  --profile train_b \
  --reuse-existing
```

四个中心都确认后，最后一次运行不带 `--profile`。脚本会按配置顺序处理全部 profile，并打印三个训练轨迹堆叠后的联合 rank/condition：

```bash
python -m calibration.generate_excitation \
  --config calibration/config.yaml \
  --reuse-existing
```

### MuJoCo 仿真预览和世界场景

`generate_excitation` 保存轨迹后会把 URDF 转成 MJCF，合入 `calibration/mujoco/scene_template.xml`，输出与轨迹同名的 `*.scene.xml`，随后用 `mujoco.viewer.launch_passive()` 打开原生 MuJoCo 桌面窗口并播放。scene 参照本机 `franka_fr3/fr3_scene.xml`，只定义 FR3 同款的 statistic、headlight、haze、黑色 skybox、checker floor 和单个 directional light；不定义固定 camera、坐标轴实体或工作区边框。原始 URDF 不会被修改。

| 参数 | 单位 | 含义 |
| --- | --- | --- |
| `simulation.scene_template_path` | - | FR3 风格的 MuJoCo 世界场景模板，包含 `option`、`visual`、`asset` 和 `worldbody`。 |
| `simulation.end_effector_body` | - | 用于生成末端轨迹和工作区检查的 MuJoCo body，当前为 `gripper_base`。 |
| `simulation.floor_z_m` | m | MuJoCo world 坐标系中的 floor 高度。应根据机械臂真实安装基准设置。 |
| `simulation.workspace_min_m` | m | 数值审查工作区的 `[x_min, y_min, z_min]`，不渲染边框。 |
| `simulation.workspace_max_m` | m | 数值审查工作区的 `[x_max, y_max, z_max]`。末端超出时轨迹点变红并打印越界样本数。 |
| `simulation.display_rate_hz` | Hz | 原生 viewer 动画刷新频率；保存的轨迹仍保持 `sample_rate_hz`。 |
| `simulation.playback_speed` | 倍速 | 默认动画倍速。`1.0` 为实时，命令行 `--playback-speed` 可临时覆盖。 |
| `simulation.collision_sample_stride` | 样本 | 每隔多少个轨迹样本执行一次 MuJoCo contact 检查。越小检查越密、生成耗时越长。 |
| `simulation.ignored_contact_pairs` | - | 经全轨迹统计和人工检查确认的固定 MuJoCo geom 重叠。必须写出两个准确 geom 名称，不能使用 link 通配或全局关闭碰撞。 |

NERO 的 visual mesh 是 MuJoCo 不支持的 Collada DAE，因此转换时明确丢弃 DAE visual，并使用 URDF 中同一 link 的 STL collision mesh 显示和计算 contact。`model.locked_joint_names` 中的夹爪关节只在临时转换副本中改成 fixed：这既匹配 Pinocchio reduced model，也避免空 `gripper_link` 作为 MuJoCo 可动零质量 body；原 URDF 不会改变。关节图距离不超过 2 的相邻 body 被排除，避免把关节连接面和夹爪正常接触报成自碰撞；base 与 floor 的预期接触也被排除。输出中的 `non-neighbor self-contact pairs` 和 `unexpected world-contact pairs` 都必须为 0。这仍不包含真实线缆、相机和未写入 MJCF 的周边障碍，必须结合现场查看。`workspace_min_m/max_m` 只是人工配置的审查边界，不是自动获得的真实环境模型。

当前 allowlist 仅包含 `link5_collision_0 <-> gripper_flange_collision_0`。依据是四个不同 profile 的 301/301 个检查样本全部命中，最小距离始终约 `-0.028506 m` 且只产生微米级变化，符合 MuJoCo STL 凸包固定重叠特征。报告会始终打印该忽略项；若 URDF、mesh 或转换方式改变，必须删除 allowlist 并重新统计，不能沿用旧结论。

生成命令相关选项：

- 默认行为：保存 NPZ 和完整 MJCF scene、打印限位与 contact 统计、打开 MuJoCo 原生窗口并播放一次；
- `--loops N`：播放 N 次；
- `--playback-speed X`：按 X 倍速播放；
- `--hold-seconds S`：播放结束后保持最终画面 S 秒；
- `--profile NAME`：只处理指定 profile，可重复使用；不提供时按配置顺序处理三个训练点和一个验证点；
- `--reuse-existing`：已有 profile NPZ 直接复用，不存在的 profile 仍正常生成；用于保留已检查的点并补齐其他点；
- `--no-visualize`：仍生成 MJCF 并执行工作区/contact 检查，但不打开桌面窗口。

### 真机安全限额

| 参数 | 单位 | 触发条件和行为 |
| --- | --- | --- |
| `safety.approved` | - | 真机运动总开关。对采集命令而言，`false` 时只允许 `--plan-only`，采集器在连接机械臂前拒绝继续；轨迹生成和离线辨识不受影响。只有完成现场轨迹审查后才可人工改为 `true`。 |
| `safety.max_abs_torque_nm` | N·m | 七轴 PyAgxArm 实际力矩反馈的绝对值上限。任意一轴 `abs(tau)` 超限立即抛出错误并停止发送后续轨迹点。 |
| `safety.max_timestamp_gap_s` | s | 相邻 adapter observation 时间戳的最大间隔。间隔非正或超过此值说明采集/通信时间线不可信，立即中止。 |

以下参数也属于真机安全边界，虽然它们位于 `excitation` 段：

- `joint_limit_margin_rad`：同时检查计划位置和实测位置；
- `max_tracking_error_rad`：监测失步、控制延迟或异常跟随；
- `max_velocity_rad_s`、`max_acceleration_rad_s2`：限制生成的命令轨迹；
- `start_move_speed_rad_s`：限制进入激励首点的过渡运动。

配置中的默认数值只是本项目当前的审查起点，不代表 NERO 厂商额定安全值，也不代表适合当前安装方式。首次真机运行前至少应完成以下检查：

1. 保持 `safety.approved: false`，分别生成训练和验证轨迹；
2. 运行 `collect_dynamics --plan-only`，检查逐轴位置最小值、最大值、速度和加速度；
3. 在 MuJoCo 原生窗口中完整观看轨迹，确认 `workspace violations: 0`、`non-neighbor self-contact pairs: 0` 和 `unexpected world-contact pairs: 0`；
4. 根据实际机械限位、线缆、桌面、夹爪和周边障碍缩小 `amplitude_rad`，必要时降低速度和加速度；
5. 根据无接触低速运行日志和硬件额定值确定逐轴 `max_abs_torque_nm`，不要直接照搬默认数组；
6. 清空末端负载和工作区、确认机械臂安装牢固并保持硬件急停可立即操作，然后才设置 `approved: true`。

软件越限后会停止后续命令并退出采集，但不会自动 disable 一台承受重力的机械臂；退出路径保持现有项目的“不断使能、防止掉臂”策略。因此这些软件阈值不能替代硬件急停，也不能保证碰撞一定能在造成损伤前被检测到。

### 离线预处理

| 参数 | 含义 |
| --- | --- |
| `preprocess.state_method` | `spline` 或 `fourier`。两种方法都只使用实测 `q(t)` 和时间戳，并解析计算 `dq`、`ddq`；都不依赖 V112 velocity。 |
| `preprocess.spline_smoothing_rad2` | 每轴平滑样条的残差预算系数，实际传给样条的 `s` 为该值乘样本数。过小会放大编码器噪声到加速度，过大会抹掉真实运动。只在 `spline` 模式生效。 |
| `preprocess.fourier_harmonics` | 实测角度 Fourier 拟合的谐波数，只在 `fourier` 模式生效。可以高于命令轨迹谐波数以容纳跟踪动态，但不宜拟合高频噪声。 |
| `preprocess.torque_lowpass_hz` | 实测力矩离线四阶零相位低通截止频率。若它达到实际 Nyquist 频率的 95%，代码会跳过低通并只保留中值滤波。该滤波非因果，只用于离线辨识。 |
| `preprocess.torque_median_window` | 力矩低通前的奇数长度中值窗口，用于去除单点尖峰。 |
| `preprocess.outlier_z` | 基于 MAD 的异常阈值。角度拟合残差或力矩滤波残差任意一轴超过阈值时，整行样本被剔除。 |
| `preprocess.endpoint_trim_s` | 每段轨迹首尾剔除时长，降低样条端点导数和过渡段误差。 |
| `preprocess.validation_fraction` | 未提供独立 `--validation-data` 时，按完整 `trajectory_id` 留出的内部验证比例。最终报告仍推荐使用独立 seed 的验证文件。 |
| `preprocess.split_seed` | 内部整轨迹训练/验证划分的随机种子。 |
| `preprocess.min_samples` | 每段轨迹在滤波和异常剔除前后都必须满足的最少样本数。 |
| `preprocess.coulomb_velocity_scale_rad_s` | 库仑摩擦 `sign(dq)` 的平滑尺度，模型使用 `tanh(dq / scale)`，避免零速处不连续。 |

### 参数估计和物理约束

| 参数 | 含义 |
| --- | --- |
| `identification.svd_relative_tolerance` | 相对最大奇异值的截断阈值，用于确定可辨识基参数秩。过小会保留噪声方向，过大会丢失有效参数组合。 |
| `identification.huber_delta` | Huber IRLS 和物理恢复的鲁棒损失转折点，作用于归一化残差。 |
| `identification.max_irls_iterations` | 基参数鲁棒加权最小二乘最大迭代数。 |
| `identification.ridge` | 可辨识基参数线性求解的微小正则项，用于改善数值稳定性。 |
| `identification.physical_prior_weight` | 物理恢复对原 URDF 先验的权重。越大越保守，越小越依赖实验数据。 |
| `identification.mass_bounds_kg` | 每个 reduced-model 刚体质量的正值上下界；不是整机总质量范围。 |
| `identification.max_abs_com_m` | 每个刚体质心三个坐标相对关节/子 link 坐标系的绝对上限。 |
| `identification.max_coulomb_nm` | 七轴非负库仑摩擦系数上限。 |
| `identification.max_viscous_nm_per_rad_s` | 七轴非负黏性摩擦系数上限。 |
| `identification.max_abs_bias_nm` | 七轴常值力矩零偏的绝对上限。零偏不会写进 URDF，而是保存在 manifest。 |
| `identification.max_physical_evaluations` | 质量、质心和惯量物理一致恢复允许的最大残差函数评估次数。91 维数值 Jacobian 需要远多于几百次评估。 |
| `identification.physical_tolerance` | 物理恢复的 `ftol/xtol/gtol` 终止容差。实测数据默认使用 `1e-7`，无需追求无噪声问题的机器精度。 |
| `identification.physical_optimizer_backend` | `jax` 使用 GPU 自动微分 Jacobian，`scipy` 使用 CPU 数值差分。RTX 4060 主机正式辨识推荐 `jax`。 |

质量通过对数变量保证为正；惯量通过正定二阶矩的 Cholesky 参数化保证正定和三角不等式。`mass_bounds_kg`、`max_abs_com_m` 及摩擦/零偏上限是辨识参数边界，不是机械臂运动安全限额。

## 1. 生成训练和验证轨迹

先在有桌面显示的 NERO 环境中安装/更新项目依赖，确保包含 `mujoco>=3.3`：

```bash
conda activate nero
python -m pip install -e '.[gpu-identification]'
```

MuJoCo 原生 viewer 需要有效的图形桌面和 `DISPLAY`。SSH 环境应使用 X11 转发或在机械臂工作站的本地图形终端运行；无图形环境只能加 `--no-visualize` 生成和检查 MJCF。

```bash
python -m calibration.generate_excitation --config calibration/config.yaml
```

脚本按 `profiles` 顺序生成并播放 `train_a`、`train_b`、`train_c` 和 `validation`。每个 profile 行中的 `standalone rank/condition` 只评价该轨迹自身；生成 B 时联合参考 A，生成 C 时联合参考 A+B。终端最后的 `combined training regressor` 才是 A+B+C 堆叠矩阵的联合 rank/condition，正式判断训练激励应优先看该值。单独重看某点可运行 `--profile train_b --reuse-existing`。

## 2. 真机采集

先保持 `calibration/config.yaml` 中 `safety.approved: false`，只查看计划：

```bash
python -m calibration.collect_dynamics --config calibration/config.yaml --plan-only
```

在真实工作区确认轨迹、清空末端负载并准备急停后，把 `safety.approved` 改为 `true`：

```bash
python -m calibration.collect_dynamics --config calibration/config.yaml
```

采集器只连接和使能一次机械臂，然后依次移动到四个 profile，按各自 `repetitions` 采集并写入各自 `dataset_path`。单点重采使用 `--profile train_a`。

采集器复用 `configs/master_slave_can.yaml` 中的 follower、V112、CAN 通道和 PyAgxArm 初始化。每个样本保存同一次 adapter observation 的本机时间戳、实测 `q`、torque/current 和 `q_cmd`。V112 的 velocity 字段不会进入数据或安全判断。

## 3. 辨识并生成新 URDF

当前物理恢复配置使用 `physical_optimizer_backend: jax`。首次运行前确认 JAX 看见 RTX 4060：

NERO 环境为 Python 3.10，因此使用 `jax[cuda12]==0.6.2`；本机 NVIDIA 580 驱动虽然报告 CUDA 13.0，但向后兼容该 CUDA 12 wheel。不要在这个环境安装要求 Python 3.11 的 JAX CUDA 13 插件。

```bash
python -c "import jax; print(jax.devices())"
```

输出必须包含 `CudaDevice` 或 `platform='gpu'`，不能只有 CPU。

```bash
python -m calibration.identify_dynamics --config calibration/config.yaml
```

辨识脚本自动读取所有 `role: train` 的 dataset，并把 `role: validation` 的 dataset 作为未参与拟合的验证集。

处理流程：

1. 从实测 `q(t)` 做平滑样条或有限 Fourier 拟合，并解析求取 `dq`、`ddq`；不做直接二次差分。
2. 对实测 torque 做中值与零相位低通，按角度拟合残差和力矩尖峰做 MAD 异常剔除。
3. 构造 `computeJointTorqueRegressor`，追加库仑摩擦、黏性摩擦和七轴零偏列。
4. 对加权回归矩阵做 SVD，在可辨识子空间中用 Huber IRLS 估计。
5. 以原 URDF 为先验恢复每节质量、质心和惯量。质量使用对数参数，惯量使用正定二阶矩 Cholesky 参数，因此正定性和三角不等式始终成立。
6. 生成 `urdf/nero/nero_with_gripper_identified.urdf`、摩擦/零偏 manifest、RMSE/NRMSE 报告和验证残差图。

## 4. 独立验证

也可以在辨识完成后用另一份从未参与拟合的数据单独验证：

```bash
python -m calibration.validate_dynamics \
  --data calibration/data/dynamics_validation.npz \
  --identified-urdf urdf/nero/nero_with_gripper_identified.urdf \
  --manifest calibration/results/dynamics_manifest.yaml
```

报告包含原模型和新模型逐关节 RMSE、NRMSE、最大绝对残差以及残差曲线。URDF 只保存惯性参数；库仑摩擦、黏性摩擦和力矩零偏保存在 manifest，在线估计器需要同时加载两者。

## 数据格式

动态 NPZ 包含：

- `timestamp_us`: `(N,)` SDK 聚合关节角反馈的 CAN 帧时间戳；SDK 未提供时才回退到 adapter 读取完成时间；
- `q_can_timestamp_us`、`q_acquired_timestamp_us`: `(N,)` 原始关节角 CAN 时间和 getter 主机返回时间；
- `q`: `(N, 7)` 实测关节角，rad；
- `q_cmd`: `(N, 7)` 同步轨迹命令，rad；
- `tau`: `(N, 7)` PyAgxArm 实际力矩反馈，N·m；
- `motor_timestamp_us`: `(N, 7)` 七轴电机状态各自的 CAN 帧时间戳；预处理先据此把力矩逐轴重采样到 `timestamp_us`；旧数据缺少该字段时回退到关节角时间线；
- `motor_acquired_timestamp_us`: `(N, 7)` 七轴 motor getter 各自返回的主机 Unix 时间；CAN 时间缺失、倒退或停滞时逐轴使用该时间；
- `current`: `(N, 7)` 实际电流反馈，SDK 不提供时为 NaN；
- `trajectory_id`: `(N,)` 完整轨迹/重复段编号；
- `metadata_json`: 固件、CAN、关节顺序、单位和数据来源。
