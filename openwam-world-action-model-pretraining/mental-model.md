# 心智模型：怎么给 OpenWAM 归类

## 一、这是什么类型的问题？

**不是**一篇"我提出了新架构/新损失/新算法"的论文。它是一篇**方法论论文 + 基础设施论文 + 规模化验证**的三合一：

```
论文类型        本文对应部分       核心产出
────────────────────────────────────────────────
系统/基础设施    OpenWAM-Infra     可组合的模块化框架
经验科学         OpenWAM-Study     3 个受控实验得出的 Finding
规模化验证       OpenWAM-α         一个开源预训练模型 + 3 个 Takeaway
```

判断一篇论文属于哪类，最快的方法是问"如果结论错了，什么会崩？"
- 架构论文：错了就是这个架构不好用
- **本文**：错了就是**整个 WAM 社区的设计直觉是错的**——因为它给的是"该怎么选"的配方，而非"我选了什么"

这也决定了阅读方式：**别只看 OpenWAM-α 的分数表**，那只是配方的验证。真正的内容是 Section 4 的三组受控实验，以及 Section 5.3 里从分数反推出的三个 Takeaway。

## 二、假设的前置知识

### 必须有的

| 概念 | 为什么必须 | 不熟的话去看 |
|---|---|---|
| **Diffusion / Flow Matching** | 整个训练目标和推理过程都建在上面，式(2) 是 flow matching 的速度回归形式 | Rectified Flow、π0 论文 |
| **DiT（Diffusion Transformer）** | 所有骨干都是 DiT，AdaLN 注入时间步、patch embedding、RoPE 都是 DiT 惯用手法 | Peebles & Xie 2023 |
| **注意力掩码** | Finding 2 的全部内容就是 2×2 掩码块的开关 | 任何 transformer 教材 |
| **Action Chunking** | 动作流预测的是 chunk 而非单步，这是 ACT 引入的范式 | ACT (Zhao et al. 2023) |
| **VLA 范式** | 5.3.2 整节在和 VLA 对比 | RT-2、OpenVLA、π0 |

### 有帮助但非必须

- **VAE 隐空间**：理解"为什么 48 维和 768 维不能互换"
- **DeepSpeed ZeRO**：只影响读训练工具那一小节
- **具身基准生态**（LIBERO / RoboTwin / RoboCasa 等）：不熟也能读，但会失去对分数量级的直觉

### 不需要的

- 强化学习：本文纯监督学习（flow matching），完全不涉及 RL
- 经典世界模型（Dreamer 系）的规划与 model-based RL：只在 Related Work 提了一句血缘

## 三、在研究地图上的位置

```
                        视觉/语言预训练
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
      视觉语言模型（VLM）              视频生成模型
              │                               │
      继承语义与语言知识              继承视觉动态生成先验
              ↓                               ↓
    ┌─────────────────┐           ┌─────────────────────┐
    │   VLA 范式       │           │   WAM 范式           │
    │ RT-2, OpenVLA   │           │ DreamZero, Fast-WAM │
    │ π0, π0.5        │           │ LingBot-VA, Motus   │
    │ Qwen-RobotManip │           │ Cosmos Policy       │
    └─────────────────┘           └─────────────────────┘
              │                               │
     开放生态已建立                    ← 开放生态缺失 ─┐
     OpenVLA / StarVLA                                │
     XPolicyLab                              ★ OpenWAM 填这个空缺
              │                                       │
              └───────────┬───────────────────────────┘
                          ↓
              "谁更强？"→ Takeaway 3：各有优势区，
                          缺口都能靠数据补，
                          结合两者是值得追的方向
```

**横向的方法论血缘**（同样把模型设计当经验科学做的工作）：
```
Cambrian-1              → 多模态：视觉表征、模态容量、数据组成
Beyond Language Modeling → 多模态：统一预训练
Physics of Multimodal Pretraining → 知识流动、协同 vs 竞争、模态统一时机
Physics of Language Models → LLM 架构设计
Action Chunking 分析系列 / Generative Control Policies → 机器人学习
Large Behavior Models / LBM co-train / StarVLA-α / OpenHLM → 数据与系统级受控比较
                          ↓
                    ★ OpenWAM：把这套方法论搬到 WAM
```

## 四、怎么在脑子里存这篇论文

### 一句话版本

> 把 WAM 的设计空间因子化成 `C(E, S, M)`，逐个变量做受控实验，得出"骨干要强 + 隐空间要紧凑 + 训练时必须 world→action + 推理时同步去噪 + ego/robot 单阶段共训 + 预训练规模下用 mutual"这套配方，再用它训一个 6,400 小时的开源模型验证配方扛得住规模。

### 三个数字版本

- **5 点**：动作流能否看到世界流的差距（92.39 vs 87.63）——Finding 2 的全部
- **12 点**：具身预训练在 OOD 上的增益（14.50 → 26.62），而 ID 只有 1.5 点——Finding 3 的全部
- **14 点**：给表征式编码器加降维适配器的收益（DINOv3 76.42 → 90.18）——Finding 1 里最反直觉的部分

### 三张图版本

- **Figure 2**（模块化基础设施）：记住 `C(E, S, M)` 和六个架构变体的谱系
- **Figure 9**（去噪调度消融）：记住"16 条异步曲线全在同步之下"这个画面
- **Figure 10 + 11**（预训练效果与偏好反转）：记住 ID 柱子几乎不动、OOD 柱子翻倍，以及 Mutual 的箭头在预训练后从红转绿

### 该和什么放在一起记

| 一起记 | 为什么 |
|---|---|
| **StarVLA / StarVLA-α** | VLA 侧的对应物。OpenWAM 明确说自己填的是 WAM 侧的同一空缺 |
| **Cambrian-1** | 同一套方法论的多模态版本，读过它就懂 OpenWAM 在做什么 |
| **Fast-WAM / LingBot-VA / DreamZero / Cosmos Policy** | 被 OpenWAM 收编为架构变体的那些单体系统 |
| **π0 / π0.5 / Qwen-RobotManip** | 5.3.2 里作为 VLA 代表的对照面 |
| **Being-H0.7 / ImageWAM / ABot-M0.5** | LIBERO-Plus 归因论证里的三个关键旁证，理解"像素级预测的鲁棒性天花板"必须记住它们 |

## 五、这篇论文改变了什么直觉

| 原直觉 | 论文修正 |
|---|---|
| 表征式编码器（语义强）应该比重建式（只压缩）更适合世界建模 | 类别之分无关，**维度紧凑性**才是关键；配了降维适配器后两类打平 |
| WAM 有效是因为它在测试时"先想象未来再反推动作" | 更像是**视频预测作为辅助任务塑造了表征**；显式的 planning-then-IDM 反而更差（IDM 87.95，异步调度全败） |
| 参数量上去了、名义上联合预测了，协同就会出现 | 需要**专属动作容量 + 明确方向的信息通路**，Single-System Vanilla 只有 85.50 |
| 具身预训练能全面提升性能 | 主要买的是 **OOD 泛化**（+12），ID 几乎不动（+1.5） |
| 小规模消融选出的最优架构，放大后还是最优 | Mutual vs 单向的偏好**符号会随预训练翻转**——架构选择依赖数据规模 |
| WAM 继承了视频先验，应该比 VLA 更能泛化 | **反了**：WAM 的 ID 拟合更好，VLA 的 OOD 泛化更好（长时程预测 = 更重的误差累积） |
