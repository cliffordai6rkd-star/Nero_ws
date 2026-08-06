# Dual\-Timescale Intent and Interaction\-State Fusion for Contact\-Rich Robotic Manipulation

> Learning models decide what the robot should do, while physics\-based controllers decide how the robot should physically interact with the environment\.
> 
> 

# Pipeline：

思路1

若控制器部分由于urdf精度问题无法实现，考虑如下思路2

问题导向：

现在的研究大致分为那几种力融入vla？

还有哪些问题没有解决？

我们的pipeline可以解决哪些问题？

## I\. Introduction

---

接触丰富操作（contact\-rich manipulation）本质上包含两个相互耦合但不同的层面：任务级动作意图生成和物理交互控制。前者决定机器人应该朝何处运动、如何接近目标以及如何完成任务阶段切换；后者决定机器人在接触过程中应以怎样的力与环境交互，并保证运动目标与接触力目标之间的协调。

近年来，Vision\-Language\-Action（VLA）模型通过大规模视觉语言预训练和机器人数据微调，在机器人操作任务中展现出较强的任务理解与动作生成能力。对于抓取、放置、移动物体等以视觉定位和轨迹跟踪为主的任务，策略主要学习从视觉语义到运动轨迹的映射，即理解目标在哪里以及末端应该移动到哪里。

然而，在接触丰富任务中，任务成功并不只取决于末端是否到达目标位姿，还取决于接触建立后的力、摩擦以及约束状态是否满足要求。例如，精密装配中的毫米级位置偏差可能导致侧向接触力快速增加并造成卡滞；擦拭任务则需要在沿目标轨迹运动的同时保持稳定法向压力。由于这些物理交互状态通常难以仅通过视觉和低维机器人状态准确推断，机器人进入接触阶段后，图像观测可能变化很小，但接触力、摩擦状态和约束关系已经显著改变。因此，单纯依赖动作空间预测容易导致接触建立不稳定、动作犹豫以及执行精度下降。

近期 force\-aware manipulation policies 尝试将力或触觉信息引入策略学习，使模型能够利用当前物理反馈改善动作生成。一些方法将力或触觉作为额外观测输入，增强策略对当前接触状态的感知能力；另一些方法进一步利用力反馈或预测 wrench 对动作进行 refinement。然而，这些物理信息通常仍主要服务于动作生成或动作修正，其最终目标仍是得到更合理的动作轨迹。对于接触丰富任务而言，机器人还需要预测执行当前运动意图后可能产生的未来物理交互状态，并在底层控制中协调运动目标与接触力目标。因此，关键问题在于如何将接触力建模为未来物理状态，并进一步作为底层控制器需要显式跟踪的物理目标。

本文希望将接触力从策略输入中的被动观测信号，进一步建模为可预测的未来物理状态，并将其作为底层力矩优化中的显式物理目标，使高层动作参考不再仅被转化为位姿跟踪命令，而是进一步与未来接触力目标共同参与控制决策。具体而言，本文提出：高层 diffusion policy 以 DINOv3 提取的视觉特征和在线接触力估计量作为输入，通过 cross\-attention 生成对当前接触状态敏感的低频末端位姿参考；同时，本文在接触阶段引入Contact\-aware Curriculum Training，使策略在训练早期强化对接触力信号的利用能力，并逐步过渡到视觉\-力联合决策。随后，dynamics\-constrained PINN 根据机器人动力学历史观测和高层位姿参考，预测未来高频接触力轨迹；最后，底层 OSC\-QP 控制器将低频位姿参考和高频预测力目标统一到任务空间力矩优化中，使控制器在跟踪末端运动意图的同时显式调节接触力。

通过这种方式，当前接触力主要用于增强高层动作参考对接触状态的感知能力，而预测得到的未来接触力则作为底层控制器的物理目标参与力矩优化。与仅利用力反馈进行动作生成、动作修正或阻抗控制执行的方法不同，本文关注的是将未来接触力轨迹作为控制时域内的显式力目标，使机器人能够在高层动作参考保持相对稳定的情况下，通过高频力矩调节响应接触状态变化。

> 本文主要贡献如下：
> 
> 

1. 提出一种预测式接触力引导的力矩控制框架，将当前接触力用于低频动作参考生成，将未来预测接触力作为底层 OSC\-QP 中的显式力目标，从而连接高层动作意图和力矩级物理交互控制。 

2. 设计一种基于 DINOv3 的视觉\-力融合 diffusion policy，通过 cross\-attention 融合图像空间表征与在线接触力估计，并引入接触感知的课程式模态遮蔽训练，以增强高层策略在接触阶段对力信号的利用能力。 

3. 构建一种 contact\-physics\-constrained high\-frequency force forecasting module，根据机器人本体动力学历史和高层位姿参考预测未来接触力轨迹，并将其作为 OSC\-QP 控制器中的力参考，实现运动跟踪与接触力调节的统一优化。

4. 搭建基于factr2的无传感器接触力估计方法的nero主从臂阻抗控制遥操平台  

## Ⅱ\. Related Works

### A\. Vision\-Language\-Action Models for Robot Manipulation

近年来，Vision\-Language\-Action（VLA）模型通过大规模视觉语言预训练和机器人数据微调，在机器人操作任务中展现出较强的任务理解与动作生成能力。该类方法通常将视觉观测、语言指令和机器人本体状态映射为未来动作序列，在抓取、放置、移动物体等以视觉定位和轨迹跟踪为主的任务中取得了显著进展。

然而，接触丰富操作不仅需要生成合理的运动轨迹，还需要感知、预测并调节机器人与环境之间的物理交互状态。对于插入、擦拭、打磨和装配等任务，接触力、摩擦状态和约束关系可能在图像变化很小的情况下快速变化。因此，仅从观测到动作的预测范式难以显式描述动作执行后产生的未来物理交互过程。

### B\. Force/Tactile/Torque\-aware Policy Learning

为提升机器人在接触任务中的物理交互感知能力，已有研究开始将 force、tactile 或 torque 信号引入策略学习框架。一类方法将力或触觉作为额外观测模态，与视觉、语言和机器人状态共同输入策略网络，使模型能够利用当前物理反馈改善动作生成。例如，ForceVLA 将六维力/力矩信号编码为 force token，并与视觉语言特征进行融合；TacVLA、AT\-VLA、VTLA 等工作进一步探索触觉 token、视觉\-触觉联合表示以及自适应触觉融合机制，使策略能够根据任务阶段利用接触反馈。

近期 torque\-aware VLA 进一步研究如何将关节力矩信号引入 VLA 架构。TA\-VLA 系统分析了 torque signals 在 VLA 中的设计空间，并提出 joint action\-torque diffusion objective，使模型在动作预测过程中学习 torque 相关的物理反馈。 这类方法表明，force / tactile / torque 信号能够增强策略对接触过程的表征能力，但其主要目标仍然是改善动作生成或动作预测。

另一类工作关注不同时间尺度下的力/触觉反馈融合。Reactive Diffusion Policy（RDP）采用 slow\-fast 结构：低频 latent diffusion policy 预测高层动作 chunk，高频 asymmetric tokenizer 利用触觉反馈实现快速 reactive behavior。 ManipForce 则提出采集高频 force\-torque 和 RGB 数据，并通过 Frequency\-Aware Multimodal Transformer 融合低频视觉与高频力/力矩信号，用于 force\-guided policy learning。 FAWAM 进一步将 force 信息用于 perception、prediction 和 execution，联合建模 future wrench 和 action，并利用预测 wrench 辅助 residual action correction。

这些方法从不同角度证明了物理反馈对于接触任务的重要性。然而，它们中的 force / tactile / torque 信号通常仍主要服务于 action generation、action decoding 或 residual action refinement。相比之下，本文关注的不是利用物理信号进一步修正动作输出，而是在高层动作参考给定的条件下预测未来高频接触力轨迹，并将其作为底层力矩优化中的显式力目标。

此外，多模态机器人策略还面临一个常见问题：当视觉特征具有较强表达能力时，模型可能在训练中优先依赖视觉模态，而忽略较低维但对接触阶段更关键的力或触觉信号。已有工作常通过模态 dropout、随机遮蔽或多模态数据增强提高策略对不同模态的鲁棒性。与一般的随机模态遮蔽不同，本文采用接触感知的课程式模态遮蔽训练，仅在接触力非零的样本上逐步遮蔽图像输入，使模型在训练早期强化对接触力信号的利用，并随着训练推进逐渐恢复视觉\-力联合决策。

### C\. Force\-executed and Hybrid Force\-Motion Imitation Learning

除了将物理信号用于策略输入或辅助预测，近期研究也开始将 force / torque 与底层控制器结合，形成 force\-executed imitation learning 框架。ForceMimic 提出 ForceCapture 示教系统，并通过 HybridIL 学习 wrench\-position parameters，执行阶段结合 hybrid force\-position control primitive 完成接触丰富任务。 FILIC 则采用 dual\-loop force\-guided imitation learning 结构：外环 Transformer imitation policy 生成 target pose，内环 impedance torque controller 根据 force feedback 执行 compliant force\-informed manipulation；对于没有力/力矩传感器的机器人，FILIC 还利用关节力矩和 Jacobian\-based inversion 估计末端力。 最近的 Force Policy 进一步提出 global\-local vision\-force policy，在接触后利用高频局部 force feedback 估计 interaction frame，并执行 hybrid force\-position control。

这些方法表明，将 force feedback 与底层力控或阻抗控制结合，可以显著提升接触任务中的稳定性和安全性。然而，它们通常依赖当前 force estimation、预设的 hybrid force\-position control primitive 或局部 impedance regulation。本文与这类方法的关键区别在于：本文不只利用当前力反馈执行控制，而是在低频高层位姿参考给定的条件下，预测未来高频接触力轨迹，并将其作为控制时域内的力目标引入 OSC\-QP 力矩优化。这样，底层控制器不仅响应当前接触状态，还能够利用预测的未来物理交互趋势进行力矩调节。

### D\. Predictive Contact Dynamics and Force\-guided Torque Control

机器人控制领域长期研究力位混合控制、Operational Space Control（OSC）以及基于 Quadratic Programming（QP）的优化控制方法，用于在任务空间中同时处理末端运动目标、接触力目标、关节力矩限制和机器人动力学约束。这类方法通常直接在力矩层面进行优化，适合擦拭、打磨、插入和精密装配等需要协调运动目标与接触力目标的任务。

进一步地，contact\-implicit MPC 将接触动力学直接嵌入预测控制问题，使机器人能够在不显式预设接触模式的情况下处理接触建立、断开和切换。相关方法通常利用互补约束、可微接触模型或 contact dynamics approximation 来描述混合接触过程。 C3 等方法进一步将 contact\-implicit MPC 与 consensus / complementarity optimization 结合，用于处理接触丰富系统中的局部优化问题。

然而，纯模型控制方法通常需要较强的接触几何、接触模式或模型先验，而传统力位控制也往往依赖人工设计的运动参考和力参考。本文试图在学习策略和模型控制之间建立连接：高层 diffusion policy 生成低频末端位姿参考，轻量化 contact\-physics\-constrained force predictor 根据机器人动力学历史和高层参考预测未来高频接触力轨迹，底层 OSC\-QP 则将该预测力轨迹作为控制时域内的显式力目标，从而在任务级动作意图与力矩级物理交互控制之间建立可优化的接口。

## III\. Method

### A\. Framework Overview

本文提出一种。该框架的核心思想是：将接触力从策略输入中的被动观测信号，进一步建模为控制时域内可预测的未来物理状态，并将其作为底层力矩优化中的显式力目标。

整体框架包含三个模块。首先，高层 diffusion policy 使用 DINOv3 提取图像中的语义与空间几何特征，并通过 cross\-attention 与在线接触力估计特征进行融合，生成对当前接触状态敏感的低频末端位姿参考。为了避免模型在训练早期过度依赖视觉信息，本文进一步引入接触感知的课程式模态遮蔽训练，在接触阶段先强化模型对力信号的利用，再逐步过渡到视觉\-力联合决策。其次，轻量化接触动力学预测模块根据机器人本体动力学历史和高层位姿参考，预测未来高频接触力轨迹。最后，底层 OSC\-QP 控制器将低频位姿参考与高频预测接触力目标统一到任务空间力矩优化中，使机器人能够在跟踪末端运动意图的同时调节接触力。

这一设计区分了当前力和未来力的作用：当前在线估计力用于增强高层策略对接触状态的感知能力，而未来预测力则作为控制器需要跟踪的物理目标，用于引导力矩级控制决策。

### B\. DINOv3\-based Force\-aware Motion Reference Generation

在高层策略部分，本文采用 DINOv3 作为视觉表征骨干网络，用于提取图像中的语义与空间几何特征。选择 DINOv3 的原因在于，接触丰富操作不仅依赖物体类别或全局语义，还高度依赖局部空间结构，例如目标表面位置、工具与物体的相对关系、潜在接触区域以及插入/擦拭/打磨过程中需要保持的几何约束。相比于仅面向分类或全局识别的视觉特征，DINOv3 通过大规模自监督学习获得较强的 dense visual representation，能够在不依赖任务特定标注的情况下提供较稳定的 patch\-level 空间表征，因此更适合作为接触任务中低频动作参考生成的视觉前端。Meta 对 DINOv3 的介绍也强调其单一 backbone 能产生高质量 dense features，并支持检测、深度估计等 dense prediction 任务；DINOv3 论文同样指出其 dense features 在多种视觉任务上具有较强表现。

具体而言，图像观测首先经过 DINOv3 编码器得到视觉特征，在线估计的当前接触力经过力编码器得到物理交互特征：

$\mathbf{z}_{I,t}
=
E_{\mathrm{DINO}}
\left(
\mathbf{I}_t
\right),
\quad
\mathbf{z}_{f,t}
=
E_f
\left(
\hat{\mathbf{f}}_{c,t}^{\mathrm{est}}
\right)$

随后，本文通过 cross\-attention 融合视觉特征和接触力特征：

$\mathbf{z}_{t}^{\mathrm{vf}}
=
\mathrm{CrossAttn}
\left(
\mathbf{z}_{I,t},
\mathbf{z}_{f,t}
\right)$

融合后的视觉\-力表征被输入 diffusion policy，用于生成未来一段时间内的低频末端位姿参考：

$\mathbf{x}_{ee,t:t+H_l}^{\mathrm{ref}}
=
\pi_{\theta}
\left(
\mathbf{z}_{t}^{\mathrm{vf}}
\right)$

其中，It 表示当前图像观测，fc表示在线估计得到的当前接触力，zI,t表示 DINOv3 提取的视觉特征，zf,t表示接触力特征，zt 表示 cross\-attention 融合后的视觉\-力表征，xee表示高层策略输出的低频末端位姿参考。

该设计中，DINOv3 负责提供稳定的视觉语义与空间结构信息，在线接触力估计负责提供当前物理交互状态，cross\-attention 则使策略能够根据当前接触状态动态调整对视觉区域的关注。例如，在接触建立前，策略可以更多依赖视觉特征判断目标位置和接近方向；在接触建立后，策略可以结合力特征判断是否过压、欠压或出现异常接触，从而生成更合适的末端位姿参考。

#### B\.1 Contact\-aware Curriculum Training

在训练高层 diffusion policy 时，本文进一步一种contact\-aware curriculum modality masking，用于增强模型对接触力信号的利用能力。接触丰富任务中，视觉观测通常能够提供目标位置、物体边界和潜在接触区域等空间信息，但在接触建立后，图像变化可能较小，而接触力能够更加直接地反映过压、欠压、卡滞或滑动等关键物理状态。因此，如果直接同时输入图像和接触力信号，模型在训练早期可能更倾向于依赖视觉特征完成动作拟合，从而削弱接触力对动作参考生成的作用。

为此，本文在接触阶段样本上采用渐进式视觉遮蔽训练。具体而言，当在线估计接触力的幅值大于阈值 ϵf 时，认为当前样本处于接触相关阶段，并按照随训练步数递减的概率对图像观测进行遮蔽：

$p_{\mathrm{mask}}(n,t)
=
\mathbb{I}
\left(
\left\|
\hat{\mathbf{f}}_{c,t}^{\mathrm{est}}
\right\|_2
>
\epsilon_f
\right)
p_0
\left(
1
-
\frac{n}{N}
\right)_+$

其中，n表示当前训练步数，N表示课程学习持续的总步数，p0表示初始图像遮蔽概率，ϵf 表示判断接触阶段的力阈值，\(⋅\)表示max⁡\(⋅,0\)。当样本处于非接触阶段时，图像不进行遮蔽；当样本处于接触阶段时，训练初期以较高概率遮蔽图像，随后随着训练步数增加逐渐降低遮蔽概率。

遮蔽后的图像输入定义为：

$\tilde{\mathbf{I}}_t
=
m_t \mathbf{I}_t
+
\left(
1-m_t
\right)
\mathbf{I}_{\mathrm{mask}},
\quad
m_t
\sim
\mathrm{Bernoulli}
\left(
1-p_{\mathrm{mask}}(n,t)
\right)$

其中，It表示原始图像观测，It\~表示遮蔽后的图像输入，Imask表示全零图像或 learnable mask token 图像，mt 为二值遮蔽变量。当 mt=0 时，模型无法利用当前图像信息，只能主要依赖接触力特征生成动作参考；当 mt=1时，模型同时利用图像和接触力信息。

在该训练策略下，高层 diffusion policy 的输入变为遮蔽后的图像观测和在线估计接触力：

$\mathbf{x}_{ee,t:t+H_l}^{\mathrm{ref}}
=
\pi_{\theta}
\left(
\tilde{\mathbf{I}}_t,
\hat{\mathbf{f}}_{c,t}^{\mathrm{est}}
\right)$

其中，xee表示高层策略生成的低频末端位姿参考，fc表示在线估计得到的当前接触力。

该课程学习策略的目的不是削弱视觉信息，而是在训练早期显式强化模型对接触力信号的关注，使策略学习接触力与末端位姿参考之间的关联。例如，当接触力过大时，策略应生成更加保守或远离接触面的位姿参考；当接触力不足时，策略可以继续引导末端接近目标表面。随着训练过程推进，图像遮蔽概率逐渐降低，模型从 force\-dominant learning 过渡到 vision\-force joint learning，最终学习如何结合 DINOv3 提取的空间几何特征和在线接触力估计，生成对当前接触状态敏感的低频动作参考。

该设计与后续高频接触力预测模块形成互补：课程式模态遮蔽强化了高层策略对当前接触状态的响应能力，而 dynamics\-constrained PINN 则进一步根据本体动力学历史和高层位姿参考预测未来高频接触力轨迹。前者提升低频动作参考的接触敏感性，后者为底层 OSC\-QP 控制器提供控制时域内的力目标。

### C\. Contact\-physics\-constrained High\-frequency Force Forecasting

在高层 diffusion policy 生成低频末端位姿参考后，机器人仍然需要在接触过程中进行高频、细粒度的力矩调节。然而，高层位姿参考通常以较低频率更新，难以及时描述接触力、摩擦状态和约束关系的快速变化。为此，本文设计一个轻量化的接触力预测模块，用于在高层位姿参考给定的条件下，根据机器人近期本体动力学历史预测未来高频接触力轨迹。

与高层视觉\-力策略不同，该模块不直接处理图像信息，而专注于建模机器人本体动力学状态、历史关节力矩和未来运动参考之间的关系。具体而言，给定历史窗口内的关节位置、速度、加速度和观测关节力矩，以及高层策略输出的低频末端位姿参考，接触力预测模块输出未来一段控制时域内的高频接触力序列：

$\hat{\mathbf{f}}_{c,t:t+H_f}^{\mathrm{pred}}
=
g_{\phi}
\left(
\mathbf{q}_{t-K:t},
\dot{\mathbf{q}}_{t-K:t},
\ddot{\mathbf{q}}_{t-K:t},
\boldsymbol{\tau}_{t-K:t}^{\mathrm{obs}}
\mid
\mathbf{x}_{ee,t:t+H_l}^{\mathrm{ref}}
\right)$

其中，K表示历史观测窗口长度，Hf表示高频接触力预测时域，fc表示预测的未来接触力序列。由于Hf对应高频力预测，而Hl对应低频动作意图预测，二者可以处于不同时间尺度。

该模块的输出是底层 OSC\-QP 控制器在控制时域内需要跟踪的接触力参考，将高层动作参考可能引发的未来物理交互状态显式化，使底层控制器能够在执行位姿跟踪的同时提前考虑接触力变化。

预测模块可以采用如下动力学残差作为物理约束项：

$J_{\mathrm{dyn}}
=
\left\|
\mathbf{M}\left(\mathbf{q}\right)\ddot{\mathbf{q}}
+
\mathbf{h}\left(\mathbf{q},\dot{\mathbf{q}}\right)
-
\boldsymbol{\tau}^{\mathrm{obs}}
-
\mathbf{J}_{c}\left(\mathbf{q}\right)^{\top}
\hat{\mathbf{f}}_{c}
\right\|_2^2$

$J_{\mathrm{normal}}
=
\left\|
\mathrm{ReLU}\left(-\hat{f}_{n}\right)
\right\|_2^2
$

$J_{\mathrm{comp}}
=
\left\|
\mathrm{ReLU}\left(-\phi\left(\mathbf{q}\right)\right)
\right\|_2^2
+
\left\|
\hat{f}_{n}
\,
\phi\left(\mathbf{q}\right)
\right\|_2^2$

$J_{\mathrm{fric}}
=
\left\|
\mathrm{ReLU}
\left(
\left\|
\hat{\mathbf{f}}_{t}
\right\|_2
-
\mu
\hat{f}_{n}
\right)
\right\|_2^2
$

$J_{\mathrm{nonpen}}
=
\left\|
\mathrm{ReLU}
\left(
-
\phi\left(\mathbf{q}_k\right)
-
h
\mathbf{J}_{n}
\left(
\mathbf{q}_k
\right)
\mathbf{v}_{k+1}
\right)
\right\|_2^2$

其中，式1为动力学残差约束、式2为法向力非负约束、式3为接触非穿透（法向互补）约束、式4为摩擦锥约束，式5为接触速度与接触状态互补约束

总损失可以写为：

$J_{\mathrm{phys}}
=
J_{\mathrm{dyn}}
+
\alpha_{n}J_{\mathrm{normal}}
+
\alpha_{c}J_{\mathrm{comp}}
+
\alpha_{f}J_{\mathrm{fric}}
+
\alpha_{v}J_{\mathrm{vel}}$

其中各约束权重可当作超参数进行调整。

接触力预测模块不仅学习从本体动力学历史到未来接触力的统计映射，还受到机器人动力学和接触物理约束的正则化。因此，预测得到的未来接触力轨迹更适合作为底层 OSC\-QP 控制器中的力参考，而不是仅作为动作生成网络的辅助预测结果。

### D\. Predictive Force\-guided Torque Optimization

在得到低频末端位姿参考和未来高频接触力轨迹后，本文采用基于 Operational Space Control 的二次规划控制器，将运动跟踪目标和预测力目标统一到力矩优化中。高层 diffusion policy 输出的 xee描述机器人在任务层面的运动意图，而接触力预测模块输出的 fc则描述执行该运动意图时可能产生的未来物理交互状态。底层控制器的目标是在满足机器人动力学和力矩约束的条件下，同时跟踪末端位姿参考并调节接触力。

需要强调的是，预测接触力本身不是最终控制输出。本文将预测得到的未来接触力轨迹作为 OSC\-QP 中接触力变量的参考目标，引导控制器求解关节力矩。具体地，在每个控制周期内，控制器求解如下优化问题：

$\begin{aligned}
\boldsymbol{\tau}_{t}^{*}
=
\arg\min_{\boldsymbol{\tau},\,\ddot{\mathbf{q}},\,\boldsymbol{\lambda}}
\quad &
J_{\mathrm{motion}}
\left(
\ddot{\mathbf{q}},
\mathbf{x}_{ee,t:t+H_l}^{\mathrm{ref}}
\right)
+
J_{\mathrm{force}}
\left(
\boldsymbol{\lambda},
\hat{\mathbf{f}}_{c,t:t+H_f}^{\mathrm{pred}}
\right)
+
J_{\mathrm{reg}}
\left(
\boldsymbol{\tau}
\right)
\\
\mathrm{s.t.}
\quad &
\mathbf{M}
\left(
\mathbf{q}
\right)
\ddot{\mathbf{q}}
+
\mathbf{h}
\left(
\mathbf{q},
\dot{\mathbf{q}}
\right)
=
\boldsymbol{\tau}
+
\mathbf{J}_{c}
\left(
\mathbf{q}
\right)^{\top}
\boldsymbol{\lambda}
\\
&
\boldsymbol{\tau}_{\min}
\le
\boldsymbol{\tau}
\le
\boldsymbol{\tau}_{\max}
\\
&
\boldsymbol{\lambda}
\in
\mathcal{C}_{\mathrm{friction}}
\end{aligned}$

通过该优化形式，低频末端位姿参考和高频预测接触力被统一到同一个力矩控制问题中。高层策略负责提供任务级运动意图，接触力预测模块负责提供控制时域内的物理交互目标，OSC\-QP 则在机器人动力学约束下求解满足二者的关节力矩。与仅使用当前力反馈的阻抗控制不同，该方法利用预测的未来接触力轨迹作为力参考，使控制器能够在接触状态快速变化时进行更主动的力矩调节。

# Development Progress

### PINN

* [x] transformer： p\(action\_k\+1 \| f\(qk,vk,ak,tauk\)=lambda\_K\+1\) 

* [ ] phik标注：sam3重建三维点云，sdf计算phik

* [ ] 给图像打标签后finetune dinov3

* [x] pinocchio 根据公式计算合力与ati读数是否相等   验证ati拿到的到底是末端合力还是接触力

* [ ] Loss       

### Diffusion Action Policy

* [x] 改dataloader，新增 ft 数据流

ft数据从另一个lerobotv3文件读，并通过绝对时间戳对齐ft与img

对齐方法：img：30HZ，FT：120HZ

对于每张时间戳为 t\_img 的图像，先在同一 episode 的 FT 数据中截取 \[t\_img\-0\.033333s, t\_img） 这段历史力觉信号，然后在该区间内生成 4个均匀时间点，将六维FT 数据分别线性插值到这 4 个时间点；无法被有效插值的位置填零并由 ft\_mask 标记。最终每张图像对应一个固定形状为 \[4, 6\] 的 FT 序列，obs\_n\_step张 observation 图像则得到 \[obs\_n\_step, 4, 6\]。

* [x] 增加 encoder 配置：

\- image encoder: DINOv3      ps：dinov3模型申请被拒绝了，现在目前用的是dinov2

\- FT encoder: LSTM

\- fusion: cross attention

* [x] 接入 DP Policy

让现有 DiffusionUnetImagePolicy 能使用新的 obs encoder

* [ ] 加一个训练输入噪声mask调度器，随着训练步数逐步减小模型输入image部分的噪声    

### QP Controller：







