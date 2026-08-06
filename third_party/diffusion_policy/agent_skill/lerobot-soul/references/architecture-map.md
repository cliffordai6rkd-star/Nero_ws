# LeRobot Architecture Map

## Core idea

LeRobot 不是围绕单一机器人实现展开，而是围绕一组稳定抽象展开：

- `Robot` / `Teleoperator`: 真实设备抽象
- `Camera`, `MotorsBus`: 设备子系统
- `PreTrainedConfig` + policy model: 学习策略抽象
- `ProcessorStep` + `DataProcessorPipeline`: 表示空间转换和特征契约
- `EnvConfig`: 仿真或真实环境抽象
- CLI scripts: orchestration 层

大多数 bug 都来自这些抽象层之间的边界错位，而不是某个单函数本身。

## High-value directories

### Runtime and orchestration

- `src/lerobot/scripts/`
  - 统一 CLI 入口
  - 常见排查入口：`lerobot_record.py`, `lerobot_teleoperate.py`, `lerobot_eval.py`, `lerobot_train.py`

- `src/lerobot/configs/parser.py`
  - CLI 参数、`.path`/`.type`、插件加载、`draccus` 解析入口

### Devices

- `src/lerobot/robots/`
  - 机器人实现
  - 基类：`robot.py`
  - 配置：`config.py`
  - 工厂：`utils.py`

- `src/lerobot/teleoperators/`
  - 遥操作设备实现
  - 基类：`teleoperator.py`
  - 配置：`config.py`
  - 工厂：`utils.py`

- `src/lerobot/cameras/`
  - 相机配置和适配层

- `src/lerobot/motors/`
  - 电机总线、校准和具体驱动

### Data and feature contracts

- `src/lerobot/processor/`
  - 最关键的中间层
  - 负责 teleop/robot/policy/env 之间的表示转换
  - 重点看：`pipeline.py`, `factory.py`, `converters.py`

- `src/lerobot/datasets/`
  - `LeRobotDataset`、feature aggregation、视频/图像写入、dataset utils

### Learning and evaluation

- `src/lerobot/policies/`
  - policy config、model、processor

- `src/lerobot/envs/`
  - 环境配置、registry 和 gym 适配

- `src/lerobot/async_inference/`
  - policy server / robot client 异步推理链路

## How the system usually flows

### Teleoperation / recording

1. CLI 解析配置
2. `register_third_party_plugins()` 导入外部扩展
3. `make_robot_from_config()` / `make_teleoperator_from_config()`
4. 设备连接与校准
5. processor pipeline 映射 teleop action / robot observation
6. dataset feature contract 聚合
7. dataset frame 写入与编码

### Training / evaluation

1. CLI 解析 policy/env/dataset 配置
2. policy config 和 model 创建
3. pre/post processors 创建或加载
4. dataset/env features 映射到 policy features
5. forward / loss / rollout / eval

## Common failure boundaries

### Config registration boundary

症状：

- `unknown choice`
- 指定了 `--robot.type` 但解析失败
- `Policy type 'x' is not available`

常查：

- 对应 `Config.register_subclass(...)`
- 相关模块是否被 import
- plugin 包命名和发现机制
- 工厂函数是否支持该类型

### Device contract boundary

症状：

- 设备连接成功但 `get_observation()` / `send_action()` 出错
- calibration 文件读取异常
- 真实动作奇怪但代码无异常

常查：

- `observation_features` / `action_features`
- `connect`, `configure`, `calibrate`, `disconnect`
- 电机名、相机关联、joint 顺序、单位、归一化、边界

### Processor contract boundary

症状：

- record/eval/train 之间 action 或 observation 键不一致
- shape mismatch
- dataset 保存成功但训练/评估时报 schema 错
- 中间步骤 silently 变形

常查：

- `transform_features(...)`
- `aggregate_pipeline_dataset_features(...)`
- `combine_feature_dicts(...)`
- `step_through()` 与 hook 调试

### Policy and dataset boundary

症状：

- processor 加载失败
- normalization/stats 问题
- NaN / exploding loss
- pretrained config 与当前 dataset 不兼容

常查：

- `src/lerobot/policies/factory.py`
- policy config 的 `validate_features()`
- dataset metadata / stats / delta timestamps

## OOP-safe fix rules

- 设备差异落到设备类、配置类或 processor step，不要落到通用 CLI。
- schema 差异落到 feature contract 或 adapter，不要靠临时 rename 掩盖。
- 新类型优先走 registry + factory + plugin 机制。
- 机器人专属安全逻辑优先放在该机器人模块或 processor step，不污染通用 policy 代码。
- 若一个修复会破坏多机器人兼容性，默认认为修复位置不对。

## Native docs worth loading

- `docs/source/debug_processor_pipeline.mdx`
- `docs/source/processors_robots_teleop.mdx`
- `docs/source/integrate_hardware.mdx`
- `docs/source/bring_your_own_policies.mdx`
