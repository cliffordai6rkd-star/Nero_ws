# π0 LoRA → CaRS-WM → q

## 多轮交互推理

```yaml
control:
  inference_runs: 5
  wait_for_start: true
```

程序启动先复位到 rest_q，然后等待运行脚本的终端按键（无需回车）：

- s：开始一轮；若当前轮暂停，则从新观测重新规划后恢复本轮。
- d：暂停推理并在当前位置保持，不复位、不消耗轮数。
- i：结束当前轮并复位，等待 s 开始下一轮；暂停时也有效。
- Ctrl+C：退出程序，沿用异常退出的保持逻辑。

每轮最多执行 control.maximum_steps（或 --steps）个控制周期，暂停/校准/复位时间
不计入；到上限自动结束并复位。完成 inference_runs 轮后退出。空闲时按 i 只复位，
不消耗轮数。相机在轮间保持运行；首次 s 后进行校准，后续复用校准参数。
暂停/结束后旧推理结果丢弃，后台任务完成前不会并发启动新的模型任务；恢复会重新填充
历史和建立 π0 action 计划，不续播过期 chunk。
无交互终端时须设置 wait_for_start=false、inference_runs=1，使用原单轮自动启动行为。

相机预览右上角以白字显示当前执行的 WM 样本/帧的预测阶段：free motion、alignment、
contact。alignment 对应模型的 precontact_or_transition 类，不表示独立检测到几何对准。
暂停、等待有效预测、WM 关闭或模型没有兼容的三阶段输出时隐藏文字。叠字只在 GUI
进程中进行，不写入 π0 输入或保存图像。模拟 WM 显示 free motion。

独立入口 `scripts/run_pi0_wm.py`，独立配置 `inference/configs/pi0_wm.yaml`。
不导入旧 DP runtime、TimestampFastSlowRuntime 或 ContactWMInferencePipeline。
不会加载 DP checkpoint；支持 q 位置控制和可选 MTC/MIT 力矩前馈，不实现 MPC。

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

- π0 完整返回值保留，动作 i 的时间锚点固定为请求观测控制步 + i×4。`consume_steps` 是每段最多消费的 token 数，启动校准根据 chunk 长度与推理余量自动缩短，预留过期前缀。
- WM action 从 `floor(control_step/4) + action_start_offset` 起取原生 token。每个目标 token 根据原始观测锚点选择源动作，非整步相位使用 previous 保持；新计划生效时跳过已过期动作，不把第 0 帧重新标成生效时刻。
- 提前请求以最后一段已提交计划的结束时间为基准，预留 WM action 窗口和 π0 推理时间。必要时在该计划生效前请求后继，worker 仍最多一个在途请求。
- 若 π0 迟到，新计划在返回后的下一 action 边界恢复，只使用剩余有效后缀；完全过期则丢弃并重新请求。不补齐或重复尾帧。缺少完整 action 窗口时停止发起 WM，有效预测尾部用尽后保持已下发 q。已用于 WM 的计划边界不移动。
- 此修复对齐源动作时间，不保证不同随机预测之间自然连续，也不对 chunk 做额外平滑。
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

边缘端不需要训练 episode 的 H5 文件，也不需要
`world_model_timeline.json`。实时 history 完全由机械臂反馈构造：`q` 是当前关节位置，
`dq` 固定使用当前硬件反馈的电机速度，`tau` 是实测
电机力矩，`delta_q` 是最近一次成功下发并保持的 q_cmd 减当前 q。`held` 只在命令成功后更新，
因此不会把预测值误当成实际命令。

预处理契约来自 checkpoint 保存的 `dataloader` 配置和
`normalize_dataloader_filters(data)`。数据集创建时已经执行的
`dataset_preprocessed_operations` 不会重复执行；checkpoint 声明的剩余因果操作会在实时
history 上连续执行。如果 checkpoint 明确声明 q、dq、delta_q、tau 没有训练滤波，运行时
`operations` 为空，原始实时值直接进入 history。dq 来源固定为硬件反馈速度，不读取或校验 checkpoint 的 `dq_source` 声明，也不由 q 差分计算。硬件侧已经
完成的符号修正不会再次取负。更换 checkpoint 后仍需检查其 dataloader/filter 配置。

当前本地 `cwm_insert_usb_100hz_80step` checkpoint 的 dataloader 声明 q、delta_q 已在数据集
阶段完成 15 Hz 二阶低通，dq、tau 需要实时执行 15 Hz 一阶低通；因此部署 operations 只包含
dq/tau 的一阶低通。checkpoint 无需保存 `dq_source` 字段。

真实下发初始化沿用 follower + enable 的位置命令链路，随后先复位到
`hardware.endpoint.rest_q`，复位完成后才开始推理校准。在运行终端按 `i`（无需回车）
会停止推理调度、复位到同一 `rest_q`，然后退出；校准等待期间也支持此按键。
复位使用数采的 `move_j` 方式和默认参数：30 Hz 插值、1 rad/s、每步不超过 0.05 rad，
通过 5 次反馈均值确认误差不超过 0.02 rad，未到位时限时微调。
Ctrl+C、异常及步数结束仍保持最后成功下发的目标，不自动复位。
物理 dry-run 不执行启动或按键复位。操作键需要焦点位于启动脚本的终端。
每次 π0 结果返回时，独立线程打印 `PI0_CHUNK {"q": [...]}`，仅包含此时 WM 最新观测的
七个关节位置（rad），不打印其他观测量、动作或诊断元数据。
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

WM 可选择顺序开环执行：

```yaml
wm:
  enable: true
  inference_mode: openloop  # prefetch 为原提前请求模式；省略时默认 prefetch。
```

openloop 每轮使用最新观测推理，结果返回后在下一个可下发时刻从预测第 0 帧执行
`control.execute_steps` 个 100 Hz 时间步，然后才请求下一轮。推理等待期间保持最后
位置目标，MTC 前馈按原逻辑退回重力补偿，不继续消费上一轮剩余预测，也不提前请求。
q/tau 使用同一播放索引。此模式有意将轨迹从接管时刻重放，不使用预取模式的延迟丢帧；
execute_steps 只需不超过模型预测长度，不要求推理延迟加执行长度落在预测窗口内。
等待推理受 calibration.request_timeout_s 限制。启动耗时校准仍保留，π0 的独立
chunk 调度不变。command_hz 较低时，每段保证从一个实际下发时刻开始，其余帧按分频跳过。

这里的开环仅指 WM 段内不重新查询，不是冻结 π0 的时间轴，也不是逐条等候关节到位。
等待期间 π0 action 时间继续前进；下一轮 WM 选取新的当前 action 窗口，不回放等待期间
已经过期的 action。每轮播放时长是 execute_steps/100 秒，推理时间另计；例如播放
100ms、推理80ms，仅约56%的时间在播放。预测16帧而 execute_steps=10 时，其余6帧丢弃。

π0 使用 `pi0.anchor_camera`（默认 wrist）帧的主机时间戳作为 action 原点。
根据实际采样时刻插值到控制步时间轴，保留不足一帧的相位；状态取该帧时刻之前最近的
观测，图像时间未被状态历史覆盖时等待下一周期。校准的 π0 延迟预算包含图像已有年龄。
这修正了“收到请求时刻代替图像时刻”的误差，但主机帧时间不是硬件曝光时间，且没有
实现两台相机的硬件同步。prefetch 的 WM q_pred[d] 保持下一周期目标语义，不改成 d-1。

`control.hz: 100` 是状态采样、WM 历史和预测索引频率；`control.command_hz: 50`
单独控制命令下发频率，适用于 q、mtc 及纯 π0 模式。未填写时默认与 hz 相同。
command_hz 必须为 hz 的整数分频，例如 100/50/25/20/10 Hz。跳过的命令不排队，
下次下发使用当时对应的预测点；held_q 只在成功下发后更新。execute_steps 仍按
100 Hz 计数，maximum_step_rad 仍是每次实际下发的限幅，不因降频自动放宽。
MTC 的速度/加速度和力矩变化率使用实际下发间隔；watchdog_timeout_s 必须大于
1/command_hz 并留出抖动余量。启动复位、模式切换及退出保持不受此分频限制，复位仍用
独立的数采复位频率。降低 command_hz 不会降低 WM 推理计算量。

`control.mode` 默认 `q`（位置下发），设为 `mtc` 可启用 WM q/tau 同步 MIT 控制。
`mtc` 要求 `wm.enable: true` 且模型预测 tau，纯 π0 模式需使用 `q`。参数在
`control.mtc` 下：逐关节 kp/kd、tau_scale、速度/加速度、前馈/总力矩与变化率限值，
以及控制周期 watchdog 和重力模型 URDF。标量限值会展开成七关节。

MTC 从同一 WM 结果、样本和延迟索引提取 q/tau。位置参考先做速度/加速度及关节范围
限制，并按剩余距离提前制动，避免固定目标下的参考超调。目标突然反向或进入保持时，
不越过/远离目标优先于加速度连续性；速度参考由最终位置参考差分得到。前馈为
`g(q_measured) + tau_scale * (tau_WM - g(q_WM))`，参考偏离预测时降低残差权重。
这是预测力矩前馈，不是实测力矩误差闭环。前馈做幅值/变化率限制，同时根据当前反馈
估算 PD+前馈总力矩，必要时缩小 kp/kd；这不是固件总力矩的硬限幅保证。
预测耗尽时保持位置并将前馈逐渐退回重力补偿；控制周期超时则退出。启动复位完成后
进入 MIT，位置参考重新锚定到实测 q，前馈从切模前实测 torque（经过幅值限制）开始，
通过 `control.mtc.startup_blend_s`（默认 1 秒，必须为正）过渡到模型前馈，且仍受力矩
变化率限制。反馈力矩无效时拒绝切模；只有下发成功才推进过渡状态。
`tau_scale=0` 仅关闭预测残差，仍保留完整重力补偿。模型重力与实机的匹配需要单独验证。
按 i、异常或正常退出时先用最后成功位置目标恢复位置模式，再复位或保持。
默认 kp/kd 来自数采从臂，tau_scale=0.1 是待实机验证的初始设置，不能视为已验证增益。

只向 MIT 下发力矩可使用以下配置（其余配置保留）：

```yaml
wm:
  enable: true
control:
  mode: tau
  command_hz: 100
  mtc:
    maximum_tracking_error_rad: 0.25
    startup_blend_s: 1.0
```

`tau` 与 `mtc` 共用 `control.mtc` 参数。上位机每个 100 Hz 控制周期读取反馈，
计算 `kp*(q_ref-q) + kd*(dq_ref-dq) + g(q) + w*tau_scale*(tau_WM-g(q_WM))`。
其中 w 是参考偏离预测时的残差权重；重力加预测残差先受 feedforward_limit_nm 限制，
再与 PD 相加。启动过渡、total_torque_limit_nm 和 maximum_torque_rate_nm_s
均作用于最终完整力矩。MIT 下发的 p_des/v_des/kp/kd 全为零，只有 t_ff 有效。
WM 的推理周期仍独立，支持 prefetch/openloop；等待预测时保持最后参考并继续反馈控制。
实测 q 与最终参考偏差超过 maximum_tracking_error_rad、反馈过期或周期超时会进入
原有退出流程，恢复位置模式。tau 模式拒绝降低 command_hz，以免将上位机反馈环一起降频。
Python watchdog 无法在进程卡死时执行；固件命令超时保护仍需在实机确认。
此模式没有实机稳定性验证，mock 仅验证下发协议与流程，不模拟力矩动力学。

`wm.enable` 默认 `true`。设为 `false` 可切换为纯 π0 开环执行：

```yaml
wm:
  enable: false
```

该模式不加载 WM checkpoint、不启动 WM worker，也不进行 WM 校准。π0 仍输出
`action.ee_pose` 语义的绝对 base→link7 位姿 `[x,y,z,qx,qy,qz,qw]`（默认响应键为 `actions`）。
后台线程使用 `mujoco.mujoco_model_path` 的机械臂模型进行逐点 IK，首点使用请求时的
实测关节角作初值，后续点使用前一点的解。该模型仍是必需的，即使可视化已关闭。
求解限制在模型与硬件共同允许的关节范围内；不收敛则退出，不发送失败解。
π0+IK 的总耗时计入 π0 校准，继续使用观测锚定的 chunk 时间轴和过期前缀跳过逻辑。
动作按 25 Hz 更新关节目标，按 command_hz 下发并沿用单步限幅；缺少有效计划时保持上次目标。
开环指 chunk 内不使用 WM 预测或重新规划；仍读取反馈做状态新鲜度及关节范围检查。
启动复位、终端按 `i` 复位退出、dry-run 行为与 WM 模式相同。`--mock-wm` 仅适用于 WM 开启时。

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

### CaRS-WM 离线加速 benchmark

`wm.acceleration` 默认四项均关闭，旧配置因此保持 eager 行为。需要启用加速时：

```yaml
wm:
  acceleration:
    enabled: true
    cache_condition_kv: true
    compile: true
    compile_mode: reduce-overhead
```

不连接相机、π0 服务或机械臂的 benchmark 使用同一 checkpoint、条件、显式源噪声和
`float32`，比较 eager、K/V 缓存、compile、缓存+compile 四种路径，并分别测量当前
部署 Flow steps/solver 与 64 步 Heun：

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_pi0_wm.py \
  --config inference/configs/pi0_wm.yaml --iterations 30 --warmup 3
```

输出包含 GPU 计算和 WMAdapter 端到端的测量次数、p50、p95、最大值、峰值显存、
compile+首次 worker warmup 时间，以及 q/tau/contact 输出的最大绝对误差。没有可用
checkpoint 或 CUDA 时脚本明确退出或报告 GPU 项未提供，不使用 mock 耗时宣称加速。

## 验证记录与边界

- 新测试覆盖：提前触发、当前计数不被请求重置、包含等待时间的延迟偏移、`d+E` 边界、跨 chunk 原生窗口、迟到尾部/保持/恢复、单请求 worker、实际限幅后 delta_q、因果滤波等价性、完整样本选择。
- 直接调用当前 PINN 小模型测试 `temporal_stride=2`：5 个 action token 全部保留，外部 future=8 只展开一次。
- 实际 v9 checkpoint 在 CPU 上通过官方 websocket mock server 联合 dry-run：1 sample、1 Euler step、实测 WM 峰值约 116 ms，P=14/T=6/E=20，260 控制步完成 13 轮 WM、2 个 π0 chunk；WM overrun=0，过期结果=0，日志确认跨 chunk 窗口。CPU 运行记录到 9 次控制周期超期（包含启动标定），不代表已达到硬实时。
- DP/旧 runtime/旧 async 调度相关测试通过；原离线可视化的 3 个测试缺少现代码所需 action_index，旧 MuJoCo sampler 测试使用当前 PINN 已删除参数。这 4 个失败均发生于未修改文件，也可独立复现；未为本分支更改旧行为。
- **未验证**：实际远端 LoRA checkpoint/训练配置、正式 GPU/Flow steps 的部署延迟、真实相机预览窗口、实体 CAN/机械臂 100 Hz 下发、现场 MuJoCo GUI。已验证 headless FK 进程与模拟采集；未自动启用真实运动。

```bash
.venv/bin/python -m pytest -q tests/test_pi0_wm.py
```
