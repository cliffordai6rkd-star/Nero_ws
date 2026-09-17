# π0 LoRA → CaRS-WM → q

独立入口 `scripts/run_pi0_wm.py`，独立配置 `inference/configs/pi0_wm.yaml`。
不导入旧 DP runtime、TimestampFastSlowRuntime 或 ContactWMInferencePipeline。
不会加载 DP checkpoint，不实现 MTC、MIT 力矩、WM τ 前馈或 MPC。

## 文件

| 文件 | 用途 |
| --- | --- |
| `core.py` | 原生 action 计划、固定步数调度、单请求常驻 worker、延迟接管 |
| `wm.py` | 当前 PINN checkpoint/EMA/normalizer/contract、真实历史和预处理 |
| `pi.py` | 官方 websocket 客户端、物理 EE pose 和相机观测 |
| `runtime.py` | 启动测速、100 Hz 位置下发与保持 |
| `visualization.py` | 独立进程 FK、完整最新/执行轨迹、选择样本和执行点 |
| `config.py` | 独立 YAML 校验 |
| `../../scripts/serve_pi0_wm.py` | 使用实际 openpi TrainConfig 和官方 policy loader 启动 server |
| `../../scripts/mock_pi0_wm_server.py` | 本机 websocket mock server |
| `../../tests/test_pi0_wm.py` | 步数调度、跨计划窗口、迟到、滤波、当前模型 stride 等测试 |

## 计划与执行

- π0 完整返回值保留在机器人端，默认消费前 50 个 25 Hz token。每 4 次外部下发推进一 token。`pi_loop_id/action_step/substep` 与 WM 轮次独立。
- WM action 从 `floor(control_step/4) + action_start_offset` 起取连续原生 token。尾部可以跨入已返回的新计划；交接固定在旧计划的第 50 个 token 之后，新计划从 token 0 生效，不重复末尾值、不重采样。
- π0 的固定提前请求点为 `consume - action_horizon - action_start_offset - ceil((峰值耗时+余量)*25)`，额外留出一个边界 token。每个计划仅触发一次正常请求；每个 worker 最多一个在途请求（含未领取的结果），不积压观测。
- 若 π0 错过交接，新计划在返回后的下一原生 action 边界生效，并记录 gap 和新旧七维参考跳变量。缺少完整 action 窗口时停止发起 WM；有效预测尾部用尽后保持上次实际成功下发的 q。已用于 WM 的计划边界不移动。
- 启动先预热，再分别测 π0 和 WM。WM 默认测速至少 1 秒且至少 10 个样本；快照准备、线程交付、输出反归一化/CPU 拷贝、GPU 同步、结果可见性均计入。相机预览和 MuJoCo 在测速前启动，采样数/Flow steps/solver/设备不在测速后切换。
- `P=ceil((WM峰值+margin)*100)`，`T=E-P`。启动输出峰值、P/T/E。`P>E` 或 `P+E>H_external` 直接报错；不自行修改执行长度或模型 horizon。
- 当前轮执行到 T 时请求下一轮，执行计数继续增长。新结果只能在当前轮已执行 E 步后接管；设 `d=接管控制步-请求快照控制步`，首个命令取 `prediction[d]`，包括推理完成后等待接管的步数。必须 `d+E<=H_external`。
- 偶发迟到消耗当前预测仍有效的尾部，之后保持位置；过期结果丢弃并请求新结果。实际接管才递增 WM 轮次并清零执行计数。历史、滤波器和 π0 索引一直连续。
- `wm.selected_sample` 固定选择一条完整轨迹，不对样本均值控制。τ 仍参与训练定义的输入/输出，绝不参与下发。
- MuJoCo 通过现有 `MujocoKinematicVisualizer` 生命周期、`MujocoKinematicFK` 和轨迹绘制函数实现。紫色为最新预测样本，蓝色为最新选中样本，绿色为执行轨迹，黄色为执行点，红色为实测末端。标签包含 request/计划/执行 loop ID。机器人主体来自实测 q，只调用 `mj_forward`，无动力学 stepping。队列容量 1，FK 和显示均在子进程；无 16 步截断。

## 训推契约与已核实数据

当前本地 `../PINN/outputs/cwm_insert_usb_100hz_80step/checkpoints/latest.pt` 是
`carswm_v9` / schema 10：历史 50、action 20、future 80、stride 1，100/25 Hz，
输入 q/dq/delta_q/τ，输出 q/τ，action offset 1。结构从 checkpoint 读取，
由当前模型的 `validate_checkpoint()` 验证；没有旧 v3 限制。
`sample()` 自行下采样状态并展开结果，adapter 不对 action stride 抽取，也不二次展开。

EE pose 是 base→link7 的 `xyz + quaternion_xyzw`（米，四元数规范为 w≥0），不是七关节角。
相机传递 uint8 HWC RGB，图像缩放和模型变换由实际训练配置的 server 完成。
server 用官方 `create_trained_policy()` 从 checkpoint/assets 恢复 normalization，执行训练的输入和逆输出变换。
机器人只接收物理绝对 EE action，再用 WM 自己的 normalizer 归一化。

WM 预处理不能简单把 `source_already_filtered` 理解为硬件已滤波。
实现读取 `wm.preprocessing.source_h5` 的原始字段属性和 `timeline` 的实际导出滤波记录，
然后追加 checkpoint 中**未在数据集执行**的训练滤波，保持连续因果状态。
当前示例数据：

| 模态 | 实际处理 |
| --- | --- |
| q | H5 原始反馈，无滤波；训练已声明预处理，因此不额外追加 |
| dq | SDK 电机速度，PyAgxArm 已修正关节符号；导出 20 Hz 一阶因果低通，再训练 15 Hz 一阶低通 |
| delta_q | 当时实际成功下发并保持的限幅后 q_cmd − q；当前数据无额外滤波 |
| τ | 实测电机力矩；导出 20 Hz 一阶低通，再训练 15 Hz 一阶低通 |

这也说明当前转换元数据与 YAML 中“q/delta_q 已做 15 Hz 二阶滤波”的声明不一致；
部署复现实际进入训练的数据，不凭声明新增滤波。硬件 dq 已修正符号，不能再次取负。
显式 `backward_difference` 来源可用因果后向差分；其它未知/非因果差分、原始 H5 已滤波等来源拒绝启动。
切换数据/checkpoint 时必须提供对应的 H5/导出元信息，不能借用此例的预处理证据。
这里检查所指定 H5 的属性；未逐一审计训练集所有 episode 的来源属性。

真实下发初始化沿用 follower + enable 的位置命令链路，保持当前姿态，不自动回 rest 位。
q 范围来自 Nero URDF，单步限幅默认 0.02 rad；`held` 只在命令成功后更新。
默认 dry-run 只读取硬件，不 enable、不发送预测命令。真实硬件 dry-run 的初始 held q
以初始实测 q 为静止保持假设，不能用它验证另一控制器同时运动时的 delta_q 语义。
模拟机械臂使用模拟反馈；MuJoCo 显示状态从不作为 WM 历史。

## 配置与依赖

完整示例见 `../configs/pi0_wm.yaml`。主要可改项：

```yaml
pi0:
  host: your-server
  port: 8000
  consume_steps: 50
  # interface.training_config 必须填实际 Nero EE-pose LoRA TrainConfig 名
wm:
  num_samples: 1
  selected_sample: 0
  flow_steps: 8
  solver: heun
control:
  hz: 100
  execute_steps: 20  # 始终是 200 ms，不乘 temporal_stride
calibration:
  wm_seconds: 1.0
  minimum_samples: 10
  wm_margin_s: 0.02
```

这是片段；请编辑完整配置。还需核实 CAN/USB 绑定、相机设备和训练裁剪/分辨率。
示例 `training_config: null` 是有意保留的未核验项，真实客户端会报错。
本地 openpi checkout 没有 Nero LoRA 配置或 checkpoint，不能宣称已核验远端训练变换。
server 启动脚本检查实际 LoRA model variant、repack state/action/image 映射，拒绝 ALOHA 关节适配器，
并打印真实 transforms/asset ID。两端 interface metadata 必须完全匹配。
这不能替代对实际训练坐标系、图像裁剪和自定义输出变换的核对。

机器人端使用现有 Nero Python 环境和同级 PINN 源码。官方 openpi-client 的包元数据要求
NumPy<2，与本仓库 NumPy 2.2 约束冲突；不要为装客户端降级机器人环境。
本实现已通过直接使用**官方源码** websocket 客户端验证，不复制/重写该客户端：

```bash
uv pip install --python .venv/bin/python 'websockets>=15,<16' 'msgpack>=1,<2'
export PYTHONPATH="$PWD/../openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 启动命令

在**已安装实际训练配置的 openpi 服务端环境**中，配置文件需与机器人端 interface 一致：

```bash
python /path/to/Nero_ws/scripts/serve_pi0_wm.py \
  --config /path/to/Nero_ws/inference/configs/pi0_wm.yaml \
  --train-config ACTUAL_NERO_EE_LORA_CONFIG \
  --checkpoint /path/to/actual_lora_checkpoint --port 8000
```

模型和 normalization assets 必须来自该真实训练 checkpoint，不使用 π0 base 或 ALOHA 示例替代。
机器人端从 Nero_ws 根目录运行：

```bash
# 无 server、无硬件；包含独立 MuJoCo FK worker 的全 mock 验证
.venv/bin/python scripts/run_pi0_wm.py --mock --headless --steps 260

# 已配置真实 server / WM / 相机 / CAN，只观察与预测，不下发
.venv/bin/python scripts/run_pi0_wm.py --config inference/configs/pi0_wm.yaml --dry-run

# 显式启用真实机械臂 q 命令（本轮没有运行此命令）
.venv/bin/python scripts/run_pi0_wm.py --config inference/configs/pi0_wm.yaml --enable-commands
```

若要验证官方 websocket 协议，将完整 YAML 复制为一个测试配置，改成
`hardware.backend: mock`、两相机 `backend: mock` / `visualize: false`、
`pi0.interface.training_config: mock_nero_ee_lora`、`wm.device: cpu`，并按机器能力设 Flow steps。
复制配置到别处时使用绝对路径。随后两个终端运行：

```bash
.venv/bin/python scripts/mock_pi0_wm_server.py --config /absolute/mock.yaml --port 8000
.venv/bin/python scripts/run_pi0_wm.py --config /absolute/mock.yaml --dry-run --headless --steps 260
# 如果只验证协议和调度，可额外传 --mock-wm；此选项禁止真实下发。
```

## 验证记录与边界

- 新测试覆盖：提前触发、当前计数不被请求重置、包含等待时间的延迟偏移、`d+E` 边界、跨 chunk 原生窗口、迟到尾部/保持/恢复、单请求 worker、实际限幅后 delta_q、因果滤波等价性、完整样本选择。
- 直接调用当前 PINN 小模型测试 `temporal_stride=2`：5 个 action token 全部保留，外部 future=8 只展开一次。
- 实际 v9 checkpoint 在 CPU 上通过官方 websocket mock server 联合 dry-run：1 sample、1 Euler step、实测 WM 峰值约 116 ms，P=14/T=6/E=20，260 控制步完成 13 轮 WM、2 个 π0 chunk；WM overrun=0，过期结果=0，日志确认跨 chunk 窗口。CPU 运行记录到 9 次控制周期超期（包含启动标定），不代表已达到硬实时。
- DP/旧 runtime/旧 async 调度相关测试通过；原离线可视化的 3 个测试缺少现代码所需 action_index，旧 MuJoCo sampler 测试使用当前 PINN 已删除参数。这 4 个失败均发生于未修改文件，也可独立复现；未为本分支更改旧行为。
- **未验证**：实际远端 LoRA checkpoint/训练配置、正式 GPU/Flow steps 的部署延迟、真实相机预览窗口、实体 CAN/机械臂 100 Hz 下发、现场 MuJoCo GUI。已验证 headless FK 进程与模拟采集；未自动启用真实运动。

```bash
.venv/bin/python -m pytest -q tests/test_pi0_wm.py
```
