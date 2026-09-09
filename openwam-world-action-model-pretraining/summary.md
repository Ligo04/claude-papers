# OpenWAM 论文摘要

> **OpenWAM: An Open, Modular Exploration Towards Systematic World–Action Model Pretraining**
> arXiv:2609.07398v1 [cs.RO] · 2026-09-07 · 43 页
> NUS / 清华 / 北大 / 港大 / 浙大 / 港中文 / 上交
> 代码：https://github.com/OpenWAM-Official/OpenWAM

---

## 一、背景语境

具身智能面临一个**根本性的数据不对称**：

| 数据类型 | 规模 | 提供的信息 |
|---|---|---|
| 世界变化的视频 | 极其丰富 | 视觉动态、物理演化 |
| 带可执行动作标签的机器人轨迹 | 相对稀缺 | 可执行的控制信号 |

三代方法应对这一不对称的方式不同：

1. **纯示范学习**（ACT、Diffusion Policy）——只从机器人示范中同时学视觉规律与控制，数据瓶颈最严重。
2. **VLA（Vision–Language–Action）**——从预训练视觉语言模型继承**语义与语言知识**，动作学习从"懂语义"开始。
3. **WAM（World–Action Model）**——从视频生成预训练继承**视觉动态的生成式先验**，动作学习从"懂世界怎么变"开始。

WAM 的赌注在于：视频生成被显式训练来建模时间演化，因此天然是"学习动作如何与物理变化交互"的起点。

## 二、问题陈述

WAM 的核心假设不是"用视频模型初始化策略"或"给视频生成器接个动作头"，而是**世界预测与动作生成可以协同学习**：世界建模提供状态、动力学、可能未来的结构化知识来指导动作；动作学习则把模型的注意力聚焦到对控制真正重要的变化上。

但要实现这种协同并不容易，且现有系统全是**单体式（monolithic）**的——生成骨干、视觉表征、模型架构、信息流、推理过程、训练数据配比彼此紧耦合，使得"哪个设计选择起了作用、为什么起作用"无法回答。

由此提出三个中心问题：

- **Q1**：WAM 应该继承什么世界知识？（继承）
- **Q2**：如何在世界学习与动作学习之间建立协同？（交互）
- **Q3**：这种协同如何跨域扩展？（整合）

## 三、主要贡献

### 1. OpenWAM-Infra：模块化基础设施

把 WAM 设计空间因子化为**三类可互换模块 + 一条组合规则** `C(E, S, M)`：

- **视觉编码器 E**（始终冻结）：2 个重建式（Wan2.2-VAE、FLUX.2-VAE）+ 2 个表征式（DINOv3、V-JEPA 2.1）+ 可选 S-VAE 降维适配器
- **流骨干 S**：世界流 W（5 个视频骨干，1.3B→14B）、动作流 A（独立 ActionDiT 或共享）、可选理解流 U（Qwen3-VL）
- **可见性注意力掩码 M**：4 种跨模态模式（isolated / video-sees-action / action-sees-video / mutual）

组合出 **6 个架构变体**，覆盖 Single-System（Vanilla、MoE）、Dual-System（joint self-attn、joint cross-attn、IDM）、Tri-System（joint self-attn）三个族系。

配套**四个解耦组件**：单一 trainer（一套联合 flow-matching 目标）、单一 policy server（推理模式与去噪调度正交）、单一评测协议（8 个仿真基准 + 真机走同一 WebSocket 接口）、自包含 checkpoint（权重 + 完整重建配置 + 动作归一化统计）。

### 2. OpenWAM-Study：三条设计原则

在 RoboTwin2.0 上做受控实验，每个决策都在前一个决策所创造的条件下检验：

- **Finding 1（继承）**：世界知识经由**足够强的生成骨干**和**紧凑且信息丰富的视觉隐空间**传递最有效。重建式编码器不是唯一选择——表征式编码器配合维度压缩同样高性能。
- **Finding 2（交互）**：协同需要**训练时显式的 world→action 信息流**，以及**推理时同步联合去噪**。参数量和"名义上的联合预测"本身都不产生协同。
- **Finding 3（整合）**：具身预训练**主要扩展 OOD 泛化**而非 ID 拟合。机器人轨迹保住动作接地，第一视角人类视频拓宽迁移，**单阶段共训**最有效整合两者。预训练规模下，**mutual 双向可见性**一致更优。

### 3. OpenWAM-α：规模化的开源预训练模型

把上述配方组合起来：Wan2.2-VAE（冻结）+ Wan2.2-TI2V-5B 世界流 + 1B ActionDiT 动作流，30 层全部作为桥接层做 joint self-attention，mutual 掩码，同步去噪 10 步。

在 **518.5M 帧（≈6,369 小时）** 第一视角人类 + 机器人数据上单阶段共训，通过 **80 维统一动作空间**打通 21 种具身。128 张 H200 训 7 天。

## 四、关键结果

### 消融数字（RoboTwin2.0-Full 平均成功率 %）

**视频骨干规模**（Dual-System joint self-attn）：
| 骨干 | 参数 | 平均成功率 |
|---|---|---|
| Wan2.1-VACE | 1.3B | 90.14 |
| Cosmos-Predict2.5 | 2B | 91.64 |
| Wan2.2-TI2V | 5B | 92.39 |
| Wan2.1-I2V | 14B | 93.79 |

14B 最强，但 5B 仅落后 1.40 点而参数少近 3 倍 → 采用 5B 作为默认。

**视觉表征先验**（骨干权重随机初始化以隔离表征贡献）：
| 编码器 | 平均成功率 |
|---|---|
| DINOv3 w/o SVAE | 76.42 |
| V-JEPA 2.1 w/o SVAE | 80.91 |
| FLUX.2-VAE | 84.26 |
| V-JEPA 2.1 w/ SVAE | 88.46 |
| DINOv3 w/ SVAE | 90.18 |
| Wan2.2-VAE | 90.30 |

关键：决定隐空间质量的不是"重建式 vs 表征式"的类别之分，而是隐变量本身的**操作性质**——时间与 token 双维度的紧凑性 + 丰富的世界信息。

**架构容量**（Table 1）：
| 族系 | 变体 | Clean | Randomized | 平均 |
|---|---|---|---|---|
| Single | Vanilla | 85.20 | 85.80 | 85.50 |
| Single | MoE | 86.22 | 83.04 | 84.63 |
| Dual | **Joint Self-Attention** | 92.34 | 92.38 | **92.36** |
| Dual | Joint Cross-Attention | 87.86 | 88.64 | 88.25 |
| Dual | Detached Cross-Attention | 92.06 | 91.64 | 91.85 |
| Dual | IDM | 87.76 | 88.14 | 87.95 |
| Tri | Joint Self-Attention | 92.84 | 92.36 | **92.60** |

容量从 single→dual→tri 单调提升；采用 dual joint self-attn 以隔离 VLM 理解特征的干扰。

**训练时信息流**（Table 2）：
| 掩码 | Clean | Randomized | 平均 |
|---|---|---|---|
| Isolated | 88.08 | 86.74 | 87.41 |
| Video Sees Action | 87.92 | 87.34 | 87.63 |
| **Action Sees Video** | 92.98 | 91.80 | **92.39** |
| Mutual | 92.50 | 91.80 | 92.15 |

分界干净得惊人：动作流能否读到世界特征决定约 5 点差距。反向 action→world 通路在 from-scratch 下几乎无影响。

**推理时去噪调度**：同步（sync）93.0 最优；variance-shift（α∈{4,8,16,32}）与 linear-offset（o∈{0.2..0.8}）两族、两个引领方向共 16 种异步调度**全都不如同步**。支持"表征学习假说"而非"测试时隐式 planning-then-IDM"。

**具身预训练**（600 小时等预算，RoboTwin2.0-Clean2Random）：
| 策略 | ID (Clean2Clean) | OOD (Clean2Random) |
|---|---|---|
| From Scratch | 87.00 | 14.50 |
| Robot Only | 88.50 (+1.50) | 23.80 (+9.30) |
| Ego → Robot（两阶段） | 87.10 (+0.10) | 26.50 (+12.00) |
| **Ego + Robot 共训** | 87.68 (+0.68) | **26.62 (+12.12)** |

ID 增益微弱，OOD 增益巨大（+12 点）。Robot-only 的 ID 最强，混合策略的 OOD 更好。

**信息流偏好反转**（Figure 11，Mutual − Action-Sees-Video 百分点）：
| 设置 | 基线绝对值 | 差值 |
|---|---|---|
| From Scratch · RoboTwin-Full | 92.39 | −0.24 |
| Pretrained · RoboTwin-Full | 92.34 | +0.16 |
| Pretrained · C2R (ID) | 86.98 | +0.70 |
| Pretrained · C2R (OOD) | 25.70 | +0.72 |

预训练把世界–动作交互变成**真正双向的交换**：预测的未来帧为动作生成提供视觉引导，预测的动作反过来告知帧中机械臂运动的合成。数据稀缺时两条流建立不起这种对应，充足预训练数据填上了这个缺口。

### OpenWAM-α 仿真基准（8 个基准，Figure 13）

| 基准 | OpenWAM-α | 最强基线 | 位置 |
|---|---|---|---|
| LIBERO | **99.3** | Qwen-RM / Being-H0.7 99.2 | SOTA |
| EBench（移动双臂） | **49.4** | Qwen-RM 45.6 | SOTA，领先约 4 点 |
| RoboCasa-GR1（灵巧手） | **60.5** | Being-H0.7 49.2 | SOTA |
| RoboCasa365（移动单臂） | **38.2** | Qwen-RM 35.9 | SOTA |
| VLABench | **58.9** | π0.5 48.1 | SOTA |
| RoboDojo（双臂） | 11.92 | Qwen-RM 19.34 | 最强 WAM |
| RoboTwin2.0-Full | 93.60 | Qwen-RM 93.85 | 顶层，微小差距 |
| RoboTwin2.0-C2R | 69.0 | Qwen-RM 77.1 | 最强 WAM |
| LIBERO-Plus | 69.2 | Qwen-RM 89.0 | **明显掉队** |

**LIBERO-Plus 的例外值得注意**：损失集中在 camera 与 noise 扰动上（33.8 / 39.8），background 与 layout 也有可见缺口。论文给出两层归因：

- **数据视角**：OpenWAM-α 的单臂数据仅约 1,100 小时（DROID 固定平台固定视角 + InternData-A1 的单臂部分），而 ABot-M0.5 有 8,000+ 小时、Being-H0.7 有 5,700+ 小时更多样的单臂数据。反过来，OpenWAM-α 的混合中富含第一视角、双臂、灵巧手数据，对应基准上就强。
- **架构视角**：Fast-WAM、ABot-M0.5、OpenWAM-α 共享同一预测目标——Wan2.2-VAE 重建式隐空间上的像素级未来视频，三者在 camera/noise 上呈现**相同的特征签名**。旁证：ImageWAM 只预测单帧未来（更像编辑而非 rollout，无跨帧误差累积）故仍强；Being-H0.7 用 V-JEPA 2.1 编码（时序但语义级而非像素重建）故对扰动明显更鲁棒。

### VLA vs WAM（Figure 16）

**ID 上 WAM 拟合更好**：最干净的对照是 StarVLA vs Fast-WAM（两者都无具身预训练）——LIBERO 97.6 vs 96.6、RoboTwin C2R Clean 77.8 vs 46.5、RoboTwin-Full 91.85 vs 88.25，WAM 全胜。机制：视频隐变量监督为参数优化注入了额外信息源，VLA 只有动作监督、没有中间隐变量目标。

**OOD 上 VLA 泛化更好**：同一对照完全反转——LIBERO-Plus 51.5 vs 74.1（VLA 大幅领先）。机制正是 ID 优势的另一面：长时程未来预测在分布偏移下意味着**更重的误差累积**，而纯动作 VLA 从不背这个负担。

**两个缺口都能用数据补偿**——数据仍是模型开发的第一优先级。

### 真机实验（三种具身）

- **单臂**（Franka-Research-3，6 任务）：OpenWAM-α 82.5% vs LingBot-VA 77.5% vs π0.5 55.0%。其中"Put Chili in Drawer"和"Put Jenga in Drawer"两项 20/20 满分。
- **双臂**（RoboDojo-Real，3 具身 18 任务）：OpenWAM-α 榜首，Score/SR **37.6 / 24.4**，第二名 π0.5 为 22.9 / 12.8。ARX X5 的 store_in_safe 任务达到 100.0 / 100.0。
- **灵巧手**（Wuji 手 + Tianji 臂，4 任务）：**该平台与其 9D 末端 + 21 DoF 灵巧手动作空间从未出现在预训练混合中**，OpenWAM-α 在全部任务与 ID/OOD 设置上都明显超过 π0.5。例如 Put Away Clothes：ID 100.0/100%，OOD 91.7/85%（π0.5 为 75.0/45%）。

## 五、量化配置速查

| 项 | 值 |
|---|---|
| 视觉编码器 | Wan2.2-VAE（冻结），因果编码：首帧单独 + 后续每 4 帧一组 |
| 世界流 | Wan2.2-TI2V-5B DiT，token 宽 3072，3D RoPE (帧,高,宽) |
| 动作流 | 1B ActionDiT，token 宽 1024，1D RoPE (chunk 索引) |
| 桥接 | 全部 30 对层，共享注意力空间 24 heads × 128 dim |
| 文本编码器 | umT5（冻结），4096 维 context |
| 动作空间 | 80 维：2×34 维镜像臂块（EEF 位置 3 + 6D 旋转 6 + 夹爪 1 + 灵巧手 24）+ 12 保留位 |
| 目标 | 联合 flow matching，λ_v = λ_a = 1.0，钟形 w(·) 峰值在中等噪声 |
| 时间步采样 | t = 1 − f_ρ(u), u~U[0,1], ρ=5（偏向高噪声），两流独立采样 |
| 推理 | 同步对角线，N=10 步，首帧隐变量每步后重新钉住 |
| 延迟 | RTX 5090 上约 170 ms/chunk（编译 + CUDA graph + 速度缓存 + 跳过 VAE 解码，加速 2.0–2.6×） |
| 预训练数据 | 518.5M 帧 / 6,369 小时（原始池 1.33B 帧 / 14,348 小时） |
| 数据配比 | 第一视角 30.1% · InternData-A1 30.0% · AgiBotWorld-Beta 18.6% · RoboCOIN 14.3% · DROID 7.0% |
| 算力 | 16 节点 × 8 H200 = 128 GPU，≈7 天，全局 batch 3,072 clips，155,862 步（1 epoch） |
| 优化器 | AdamW，lr 1e-4，cosine + 5% warmup，wd 0.01，β=(0.9,0.95)，grad clip 1.0，bf16，ZeRO-2 |
| 输入 | 33 帧，video stride 4，分辨率 384×320，多视角启用 |
