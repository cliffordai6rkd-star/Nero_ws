---
name: lerobot-soul
description: Architecture-aware debugging, triage, and repair skill for this LeRobot workspace. Use when a developer hits a bug, regression, strange runtime behavior, hardware-control issue, processor/dataset/policy mismatch, training or evaluation failure, plugin-registration problem, or when implementing a fix that must preserve LeRobot's OOP abstractions, ChoiceRegistry-based configuration model, processor pipelines, and multi-robot extensibility.
---

# Lerobot Soul

## Overview

用这个 skill 时，先把异常定位到 LeRobot 的哪一层，再在正确抽象层修复，不要用脚本级特判破坏整个框架。

## Working Mode

按下面顺序工作：

1. 先复现，再修复。优先拿到最小可复现命令、配置或测试。
2. 先定层，再读代码。优先判断问题属于 CLI/配置、工厂注册、硬件接入、processor、dataset、policy、env、async 哪一层。
3. 先查基类和工厂，再查具体设备或模型实现。LeRobot 绝大多数行为都从 `ChoiceRegistry`、工厂函数和 processor pipeline 分发。
4. 在最低且正确的抽象层修复。不要把机器人特例塞进通用脚本；优先新增/修正子类、配置类、processor step 或插件注册。
5. 修完后做定向验证。优先跑最小相关 `pytest`、相关 CLI 或 example，而不是盲跑全仓库。
6. 如果发现 skill 内容已经落后于仓库结构，立刻更新本 skill，并重新生成架构快照。

## Fast Triage

先把症状映射到层：

- `unknown choice`, `type not available`, CLI 参数解析异常
  - 先查 `src/lerobot/configs/parser.py`
  - 再查对应 `*Config.register_subclass(...)`
  - 再查 `register_third_party_plugins()`

- 设备无法实例化、类找不到、插件没生效
  - 先查 `src/lerobot/utils/import_utils.py`
  - 再查 `src/lerobot/robots/utils.py` 或 `src/lerobot/teleoperators/utils.py`
  - 再查包命名是否符合 `lerobot_robot_*` / `lerobot_policy_*` 规则

- 机器人/遥操作器连接、校准、动作异常
  - 先查基类契约 `Robot` / `Teleoperator`
  - 再查具体实现的 `connect`, `calibrate`, `configure`, `send_action`, `get_observation`
  - 再查 motors/cameras 子系统和 calibration 文件

- 录制、回放、评估数据键错位，或 feature mismatch
  - 先查 `observation_features` / `action_features`
  - 再查 processor pipeline 的 `transform_features(...)`
  - 再查 `aggregate_pipeline_dataset_features(...)` 和 `combine_feature_dicts(...)`

- 训练/评估形状不匹配、NaN、processor 加载问题
  - 先查 `src/lerobot/policies/factory.py`
  - 再查 pre/post processor 构建与 dataset stats
  - 再用 pipeline hooks 或 `step_through()` 看中间态

- 奇怪但没有报错的真实机器人行为
  - 先怀疑 calibration、坐标系、关节顺序、processor safety/bounds、单位或归一化
  - 再看对应 example 和文档，确认设计意图不是被误改

## Architectural Invariants

在这个仓库里，默认遵守这些约束：

- 配置与运行时实现分离。新增设备或策略时，优先新增 `Config` 子类和实现类，而不是在 CLI 脚本里堆 `if/elif`。
- 所有可选实现尽量通过 `draccus.ChoiceRegistry` 注册，再由工厂或插件发现。
- 机器人、遥操作器、相机、环境、策略都是可替换组件；修复时不能假设只有单一机器人。
- 表示空间转换应放在 processor pipeline，而不是散落在 record/train/eval 脚本内。
- dataset feature contract 必须和实际 pipeline 输出保持一致；修 observation/action 键时要同步检查 dataset 侧。
- 通用脚本是 orchestration 层，不应该承载设备专属业务逻辑，除非仓库已有明确模式。
- 真实机器人安全优先。涉及动作幅度、IK、速度、边界时，先保守再放开。

## Default Reading Order

按需加载，不要一次读太多：

- 先读 `references/architecture-map.md`
  - 需要快速理解仓库骨架、主要目录、关键基类和入口时读

- 再读 `references/debug-playbook.md`
  - 需要按症状排查、挑选测试、定位命令或判断修复落点时读

- 再读 `references/architecture-snapshot.md`
  - 需要确认当前仓库最新目录、机器人列表、teleoperator 列表、policy 列表、测试分布时读

如果问题集中在 processor：

- 读仓库原文档 `docs/source/debug_processor_pipeline.mdx`
- 读仓库原文档 `docs/source/processors_robots_teleop.mdx`

如果问题集中在扩展新硬件或策略：

- 读仓库原文档 `docs/source/integrate_hardware.mdx`
- 读仓库原文档 `docs/source/bring_your_own_policies.mdx`

## Repair Strategy

修复时优先采用下面策略：

- 先复用现有抽象
  - `Robot`, `Teleoperator`, `PreTrainedConfig`, `ProcessorStep`, `EnvConfig`

- 先复用现有入口
  - `lerobot-record`, `lerobot-teleoperate`, `lerobot-eval`, `lerobot-train`

- 先复用现有 examples/tests
  - 用最接近问题域的 example 复现
  - 用最接近模块的 test 做回归验证

- 避免这些坏味道
  - 为某个机器人在通用脚本中加不可扩展分支
  - 在 policy/model 层硬编码某个 dataset 键名而不更新 feature contract
  - 用局部 rename 或 silent fallback 掩盖配置或 schema 不一致
  - 跳过 plugin/register 机制，直接在脚本里手动 import 私有实现

## Keep This Skill Fresh

如果在真实使用过程中发现以下情况，立即更新本 skill：

- 新增或删除主要子系统、机器人、遥操作器、策略、环境
- 工厂函数、注册机制、CLI 入口、processor 工作流发生变化
- 某类 bug 的最佳排查路径已经改变
- 新增了更合适的测试或 example 作为复现入口

更新流程：

1. 修改 `SKILL.md` 或 `references/*.md`
2. 运行 `scripts/refresh_architecture_snapshot.py`
3. 若 UI 元数据已过时，重新生成 `agents/openai.yaml`
4. 运行 skill 校验
