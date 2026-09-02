# Inference 开发层指南

本文面向需要扩展 Nero 推理功能的开发者。它描述模块边界、数据契约、生命周期和新增模型的方式；运行参数、真机启动命令、H5/MuJoCo 使用说明请先看 [`README.md`](README.md)。

## 1. 总体目标

推理代码按职责拆成六个阶段：

```text
ObservationSampler
        |
        v
ObservationProcessor
        |
        v
HighLevelPolicy (DP / PI0 / VLA)
        |
        v
ActionChunkScheduler
        |
        +--> WorldModel (当前可使用 NullWorldModel)
        |
        v
ActionResolver -> SafetyGuard -> RobotController
```

`InferenceBase` 只编排顺序和生命周期，不应该知道具体模型、相机 SDK、CAN API、IK 或机器人型号。模型差异放到 `inference/policies/`，WM 差异放到 `inference/world_models/`，机器人差异放到 `inference/control/`。

当前旧 Nero 入口正在按这个边界迁移。`inference/stages/` 已提供两个可复用的
执行阶段：`DPObservationBuffer` 负责图像/CAN 历史和严格的因果时间戳对齐，
`ActionPlanExecutor` 负责按 observation timestamp 推进 DP chunk 和 open-loop
计划。`NeroInferencePipeline` 暂时保留同名私有方法作为兼容委托；新增代码应直接
依赖这些 stage，不要再向 pipeline 添加新的观测缓存或 action 游标字段。

当前仓库仍保留 `NeroInferencePipeline` 和 `NeroInferenceRuntime` 作为旧 DP/WM 兼容入口。
对于 `architecture.enabled=true, policy_type=lerobotdp,
world_model_type=contact_wm`，runtime 已将其作为 timestamp 双异步模式的唯一组合入口：
它直接连接 DP worker、Contact WM worker 和 100 Hz control worker，不再经过
`InferenceBase/ModularInferenceRunner` 的同步 scheduler。`predictor.enabled=false` 时
省略 Contact WM worker，控制 worker 直接按时间戳消费 DP action plan；DP 与控制的异步
边界保持不变。其它历史 modular policy 仍可通过显式
`modular_inference`/`modular_builder` 运行；不要把这些兼容路径与 Contact WM 的三线程
执行模型混用。

## 2. 公共数据契约

契约位于 [`core/contracts.py`](core/contracts.py)，不依赖 torch、机器人 SDK 或 PINN 包。

### `Observation`

传给 processor、policy 和 WM 的规范观测：

| 字段 | 形状/含义 |
| --- | --- |
| `timestamp_us` | 用于对齐和 action chunk 推进的状态时间戳 |
| `acquired_timestamp_us` | 采样完成时间戳 |
| `q`, `dq`, `ddq`, `tau`, `tau_ext` | 七轴向量；`tau_ext` 是校准后的外部关节力矩 |
| `wrench_ext` | 六维 `[Fx,Fy,Fz,Mx,My,Mz]` |
| `images` | `{camera_name: HxWx3 uint8}`，供模型使用的帧 |
| `image_timestamps_us` | 每个相机帧的时间戳 |
| `q_cmd` | 可选的上一帧关节命令 |
| `metadata` | 扩展字段；例如 `gripper`、`tau_result`、处理后的 wrench |

`Observation` 会校验有限值和形状。不要在 policy 中传入未经转换的 legacy dict；需要转换时在 sampler 或 processor 中完成。旧状态流可用 [`core/adapters.py`](core/adapters.py) 的 `observation_from_state_sample()`。

### `ActionChunk`

高层模型输出的未来动作序列，统一为 `[H, D]`：

```python
ActionChunk(
    values=actions,
    semantic="joint",       # joint / eepose / pose / torque
    frame_name="link7",     # pose 动作所属坐标系；joint 可为 None
    timestamp_us=obs.timestamp_us,
    step_s=0.1,
)
```

模型可以输出单帧 `[D]`，契约会自动转成 `[1, D]`。模型原始输出、归一化和 batch 维度处理必须在 policy adapter 内完成，不能让 `InferenceBase` 猜测。

### `ControlTarget` 和 `InferenceCycle`

`ActionResolver`/`WorldModel` 将动作转换为机器人空间的 `ControlTarget`（`q`、`dq`、`torque` 或 `pose`）。每次循环最后产生 `InferenceCycle`，其中包含 observation、当前 action、target、command、更新标记和耗时，可直接给诊断 sink 使用。

## 3. `InferenceBase` 生命周期

接口位于 [`core/base.py`](core/base.py)，模块协议位于 [`core/interfaces.py`](core/interfaces.py)。典型用法：

```python
inference = InferenceBase(
    sampler=sampler,
    policy=policy,
    processor=processor,
    world_model=world_model,
    action_resolver=action_resolver,
    safety_guard=safety_guard,
    controller=controller,
    diagnostics=(tau_plotter,),
)

inference.start()
inference.reset_episode()
while running:
    cycle = inference.step()  # 可能返回 None，表示本次没有有效观测
inference.close()
```

生命周期约束：

1. `start()` 依次启动 sampler、policy、controller 和 diagnostics，并且是幂等的。
2. `reset_episode()` 必须在 `start()` 后调用；它清空 action scheduler，并重置所有有状态模块。
3. `step()` 只做一次采样和一次编排，不负责 sleep。频率由 sampler、硬件状态流或上层 loop 决定。
4. `close()` 关闭 controller、policy、sampler 和 diagnostics；关闭异常会汇总后抛出。

`ActionChunkScheduler` 使用 observation timestamp 推进 chunk，不理解 joint/eepose 语义。动作语义和 frame 转换属于 resolver 的职责。

## 4. 模块职责和禁止事项

### ObservationSampler

负责相机、CAN/状态流、时间戳对齐和 freshness 检查，输出一个完整 `Observation`。[`NeroObservationSampler`](core/nero_sampler.py) 已经复用连续状态流，并保留 `tau_ext` 和高清预览路径。

不要在 sampler 中加载模型、执行 IK 或发送机器人命令。

### ObservationProcessor

负责模型专属的数据处理：历史窗口、归一化前的字段整理、图像变换、任务文本拼接、夹爪字段补充等。它可以维护 episode 状态，但必须实现 `reset_episode()`。

不要在 processor 中做模型 forward 或控制安全限制。

### HighLevelPolicy

只负责高层 action expert 的加载和推理。最小接口：

```python
class HighLevelPolicy(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def reset_episode(self) -> None: ...
    def predict(self, observation: Observation) -> ActionChunk | None: ...
```

### WorldModel

接收当前 observation 和 scheduler 选出的 action，返回 `ControlTarget | None`。WM 尚未确定时使用 `NullWorldModel`，不要在基类中增加 `if wm_type == ...` 分支。

### ActionResolver

负责 action 语义、坐标系转换、FK/IK、chunk 选择和轨迹插值。它是 DP/PI0 与 Nero 机械臂之间的唯一动作解释层。
通用 `DirectActionResolver` 只处理 joint/torque action；末端 pose 必须注入带 IK 或 FK
知识的 resolver。需要实验性逻辑时使用 `CallableActionResolver`，不要把转换分支加回
`InferenceBase`。

### SafetyGuard

负责 stale observation、位移/旋转步长、关节限位、力矩和目标 wrench 等安全约束。模型 adapter 不得自行下发安全裁剪后的机器人命令。
`BasicSafetyGuard` 提供跨机器人都成立的有限值、关节步长和 torque 限幅；Nero legacy
pipeline 的完整 pose/wrench 安全策略仍由旧 pipeline 保持，迁移时应逐项替换并补回归。

### RobotController

只把 `ControlTarget` 翻译成硬件、仿真或 mock API。[`control/base.py`](control/base.py) 提供 `ArmRobotController` 和 `CallableRobotController`；真机默认应支持 read-only 模式。

### DiagnosticSink

用于 tau_ext、wrench、相机预览、频率和耗时等旁路信息。诊断不能阻塞主控制路径。推理时的 `||tau_ext||` 绘图使用 [`diagnostics/tau_ext.py`](diagnostics/tau_ext.py)，相机的 `preview_frame` 不应复用于模型输入。

## 5. 新增 DP action expert

LeRobot 0.4 导出的目录 checkpoint（`config.json` + `model.safetensors`）使用
[`policies/lerobotdp.py`](policies/lerobotdp.py) 单文件接口。它读取 checkpoint
声明的 `observation.state`、`observation.images.wrist` 和
`observation.images.side`，并将 LeRobot 的逐步 `select_action()` 结果整理为
`ActionChunk([8,7], semantic="joint", step_s=0.04)`。环境需要安装可选依赖
`lerobot==0.4.0`；旧 Hydra `.pt/.ckpt` 仍由 `policies/dp/` 兼容层处理。

```python
from inference.policies.lerobotdp import LeRobotDiffusionPolicy

policy = LeRobotDiffusionPolicy.from_pretrained(
    "/home/rei/mnt/code/lcx/nero_ws/model/dp/"
    "pretrained_model-20260901T082955Z-1-001/pretrained_model",
    device="cuda:0",
)
```

DP 包中的 [`policies/dp/adapter.py`](policies/dp/adapter.py) 提供轻量
`DiffusionPolicyAdapter`，适用于暴露 `predict_action(model_input)` 的 diffusion-policy 模型：

```python
policy = DiffusionPolicyAdapter(
    model=dp_model,
    input_builder=build_dp_input,
    semantic="eepose",       # 或 joint
    frame_name="link7",
    action_steps=8,
    step_s=0.1,
)
```

旧 Nero pipeline 也通过该 adapter 的 `predict_raw()` 调用 DP。joint action 的
标准 diffusion 采样和 quaternion 旁路位于 `policies/dp/adapter.py` 的
`predict_diffusion_action()`；pipeline 不再维护这段模型分支，只负责把已经对齐的
image/wrench batch 放到 checkpoint device，并将原始映射转换为旧输出契约。

DP 特有的 checkpoint 恢复、image/wrench history、normalizer 和 scheduler 逻辑应封装在 DP policy 或 processor 中。不要把这些字段加到 `InferenceBase`。

如果模型的 forward API 不是 `predict_action()`，新增一个小 adapter：

```python
import numpy as np

from inference.core.contracts import ActionChunk


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim >= 3:
        value = value[0]
    return value


class MyDPPolicy:
    def __init__(self, model, input_builder):
        self.model = model
        self.input_builder = input_builder

    def start(self):
        self.model.eval()

    def reset_episode(self):
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation):
        model_input = self.input_builder(observation)
        raw = self.model(model_input)
        return ActionChunk(
            values=to_numpy(raw),
            semantic="eepose",
            frame_name="link7",
            timestamp_us=observation.timestamp_us,
        )

    def close(self):
        pass
```

适配器只需满足 `HighLevelPolicy`；它不需要继承 `InferenceBase`。

本项目第一个按训练契约验证的 DP checkpoint 是 push-button 任务，但实现属于
通用 [`DiffusionPolicy`](policies/dp/policy.py)，而不是任务类。它对应
`/home/rei/mnt/code/lcx/diffusion_policy/diffusion_policy/config/train_dp_push_button.yaml`
和 `/home/rei/mnt/code/lcx/model/dp/latest.ckpt`：

```python
from inference.policies import DiffusionPolicy

policy = DiffusionPolicy.from_checkpoint(
    device="cuda:0",
    checkpoint_path="/home/rei/mnt/code/lcx/model/dp/latest.ckpt",
    dino_model_path="/home/rei/mnt/code/lcx/model/dinov3-vitb16-pretrain-lvd1689m",
)
```

该 checkpoint 的契约是双相机 `side/wrist`、每路
`[3,192,256]`、`n_obs_steps=2`、`horizon=9`、输出 `[8,7]` 绝对关节动作、动作时间步长
`0.04s`。对应的部署参数模板是 [`configs/nero_push_button.yaml`](configs/nero_push_button.yaml)。

后续 DP 任务只需替换 checkpoint 或部署配置；不应复制新的
`PushButtonPolicy`/`InsertUSBPolicy` 任务类。

## 6. 新增 PI0 / VLA action expert

PI0 的本地模型和 websocket 服务属于两个 backend，但都可以复用同一个 `Pi0PolicyAdapter`：

```text
inference/policies/pi0.py
inference/policies/pi0_backend.py
    LocalPi0Backend       # openpi 本地 policy
    WebsocketPi0Backend   # openpi_client 远程 policy
inference/processors/pi0.py
```

backend 只提供模型服务生命周期：

```python
class Pi0Backend(Protocol):
    def start(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, model_input: dict) -> dict | np.ndarray: ...
    def close(self) -> None: ...
```

`Pi0PolicyAdapter.predict()` 负责调用 backend、取出 `actions`、去掉 batch 维度并构造 `ActionChunk`。PI0 的 `state`、相机 key、任务文本和图像 resize 放到 `Pi0ObservationProcessor` 或 `observation_builder`。

当前 `Observation.q` 是七轴机械臂状态，不包含夹爪。PI0 如果需要夹爪，使用 `Observation.metadata["gripper"]`；不要把夹爪值静默拼进 `q`，除非同步修改公共契约和所有校验。

注册新 policy：

```python
# inference/policies/__init__.py
from inference.policies.pi0 import Pi0PolicyAdapter

POLICY_REGISTRY.register("pi0", Pi0PolicyAdapter)
```

注册表位于 [`factory.py`](factory.py)。注册 key 必须稳定、小写且唯一。

## 7. 配置约定

模块化配置声明放在 YAML 的 `architecture`：

```yaml
architecture:
  enabled: true
  policy_type: dp               # dp / diffusion_policy / pi0 / callable
  world_model_type: none          # none / callable / future implementation
```

模型 checkpoint 的网络结构、shape metadata 和 normalizer 仍应从 checkpoint 恢复；部署 YAML 只覆盖设备、路径、采样步数、动作语义等运行参数。

新增配置字段时同步修改：

1. `inference/config.py` 中的 dataclass；
2. `load_inference_config()` 的校验和路径解析；
3. 至少一个示例 YAML；
4. `--check` 输出和对应测试。

`NeroInferenceRuntime` 对 `policy_type: dp`/`lerobotdp` 已提供内置模块化 builder，会自动创建 sampler、policy、joint resolver、安全 guard 和 controller；TAVLA 等仍需通过 `modular_builder` 注入官方模型与处理器。旧入口在 `architecture.enabled: false` 时保持兼容。

## 8. 测试要求

新模型先用 fake backend 和 fake sampler 测通公共契约，不要一开始连接真机或加载大 checkpoint。最低测试应覆盖：

- `[D]`、`[1,H,D]`、`[B,H,D]` 输出是否统一为 `[H,D]`；
- `ActionChunk.semantic`、`frame_name` 和 timestamp 是否正确；
- `reset_episode()` 是否清空模型内部历史；
- backend 推理失败时是否不会发送旧 action；
- scheduler 在时间戳推进后的 action index；
- resolver/safety/controller 是否收到正确 target；
- PI0 所需夹爪字段缺失时是否明确报错；
- 高分辨率 `preview_frame` 不会改变模型输入的 `output_size`。

可参考 [`tests/test_inference_core.py`](../tests/test_inference_core.py) 中的 `CallablePolicy`、fake sampler 和 fake controller 测试模式。提交前运行：

```bash
pytest -q tests/test_inference_core.py
pytest -q tests/test_inference_runtime.py
```

完整测试可能受本地 `h5py`/NumPy ABI、JAX GPU extra 等环境依赖影响；遇到这类失败时应区分环境问题和本次模块改动。

## 9. 维护规则

1. 新模型只新增 policy/processor/backend，不复制一份 inference loop。
2. 新机器人只新增 controller 或 resolver，不修改 DP/PI0 adapter。
3. 新 WM 只实现 `WorldModel`，WM 未确定时保留 `NullWorldModel`。
4. 不在 `InferenceBase` 中加入模型名称分支、torch 类型或机器人 SDK 类型。
5. 所有跨模块数据通过 `Observation`、`ActionChunk`、`ControlTarget` 传递；临时信息放 `metadata` 并记录来源。
6. 任何会影响控制的逻辑必须有 deterministic/mock 测试，再接入旧 runtime。
7. 旧 `NeroInferencePipeline` 和 `NeroInferenceRuntime` 在迁移完成前保持可用，不做一次性删除。

推荐的提交顺序是：先加契约测试，再加 adapter/backend，再加 registry 和配置，最后接入 modular builder；这样每一步都可以单独回归。
