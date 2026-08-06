# LeRobot Debug Playbook

## First response

拿到问题后先补齐这四件事：

1. 最小复现命令、配置、测试或 example
2. 出错层级
3. 期望行为和实际行为
4. 最近改动涉及的模块

如果用户只给现象，不要先改代码；先把问题压缩到最小路径。

## Symptom to suspect map

### CLI / parser / registry

症状：

- `--robot.type` / `--policy.type` 无法解析
- `--config_path`、`--policy.path`、`.type` / `.path` 组合异常

先查：

- `src/lerobot/configs/parser.py`
- 对应 `config_*.py`
- `register_third_party_plugins()` 的调用点

优先命令：

- `rg -n "register_subclass\\(" src/lerobot`
- `rg -n "register_third_party_plugins" src/lerobot`

### Robot / teleoperator / camera / motor

症状：

- 无法连接设备
- 校准文件不加载
- 命令已发送但动作异常
- observation/action 键缺失

先查：

- 基类契约
- 设备实现类
- 该设备对应 tests 和 examples

优先命令：

- `rg -n "class .*\\(Robot\\)|class .*\\(Teleoperator\\)" src/lerobot`
- `rg -n "def connect|def calibrate|def configure|def send_action|def get_observation" src/lerobot/robots src/lerobot/teleoperators`

### Processor / feature contract

症状：

- shape mismatch
- action/observation key drift
- record 正常但 train/eval 失败
- 中间转换怀疑出错

先查：

- `src/lerobot/processor/pipeline.py`
- `src/lerobot/processor/factory.py`
- 相关 `processor_*.py`
- `docs/source/debug_processor_pipeline.mdx`

排查动作：

- 给 pipeline 挂 before/after hooks
- 用 `step_through()` 看每一步 transition
- 检查 `transform_features(...)` 是否和真实输出一致

### Dataset / recording

症状：

- 数据集建出来了，但训练读不通
- 视频写入、特征合并、frame 构建异常

先查：

- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/datasets/pipeline_features.py`
- `src/lerobot/datasets/utils.py`

重点检查：

- `aggregate_pipeline_dataset_features(...)`
- `combine_feature_dicts(...)`
- `build_dataset_frame(...)`

### Policy / training / evaluation

症状：

- policy 类型可用但实例化失败
- pretrained/config/processor 不兼容
- loss 异常、NaN、动作维度错

先查：

- `src/lerobot/policies/factory.py`
- 对应 policy 的 `configuration_*.py`, `modeling_*.py`, `processor_*.py`
- `examples/training/train_policy.py`

## Verification matrix

按模块挑最接近的验证：

- processor 相关
  - `pytest -sv tests/processor`

- dataset 相关
  - `pytest -sv tests/datasets`

- robots / teleoperators / cameras / motors
  - `pytest -sv tests/robots`
  - `pytest -sv tests/teleoperators`
  - `pytest -sv tests/cameras`
  - `pytest -sv tests/motors`

- training / policies / envs
  - `pytest -sv tests/policies`
  - `pytest -sv tests/training`
  - `pytest -sv tests/envs`

- 端到端 smoke
  - 参考 `Makefile`
  - 只在依赖完备且必要时跑对应 target

## Preferred reproduction sources

优先级从高到低：

1. 现有 failing test
2. 最接近的 `tests/...`
3. 最接近的 `examples/...`
4. 对应 CLI 最小命令
5. 手工临时代码

如果能用 tests/examples 复现，就不要先写新的临时脚本。

## Repair heuristics

- 看见大量 `if robot.type == ...`
  - 先怀疑修复层级错误

- 看见 schema/key 问题
  - 先改 feature contract / adapters / pipeline

- 看见类型不可用
  - 先查 register/import/plugin，不要先改用户命令

- 看见真实机器人动作怪异
  - 先停留在安全思路：校准、边界、单位、坐标系、关节顺序

## Skill maintenance rule

每次遇到下列任一情况后，顺手更新本 skill：

- 新模块或新机器人已经成为常用路径
- 旧的最佳排查命令失效
- 新测试目录或 example 更适合作为复现模板
- 架构快照和真实仓库不一致

更新后运行：

- `python3 agent_skill/lerobot-soul/scripts/refresh_architecture_snapshot.py`
- `python3 /home/hph/.codex/skills/.system/skill-creator/scripts/quick_validate.py agent_skill/lerobot-soul`
