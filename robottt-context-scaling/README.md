# RoboTTT: Context Scaling for Robot Policies

**论文**：*RoboTTT: Context Scaling for Robot Policies*（2026-07-15，NVIDIA GEAR + Stanford + UT Austin）
**作者**：Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, Linxi "Jim" Fan
**项目主页**：research.nvidia.com/labs/gear/robottt

RoboTTT 把机器人视觉运动策略（VLA）的上下文长度从当前 SOTA 的个位数帧扩展到 8K 个时间步（30Hz 下约 4-5 分钟），比现有方法长三个数量级，且推理延迟不随上下文增长。做法是把 Test-Time Training（TTT）——一种在训练和推理阶段都用梯度下降更新"快权重"（fast weights）的序列建模机制——接入 GR00T N1.7 的 DiT action head，配合 sequence action forcing 和 truncated backpropagation through time（TBPTT）的训练配方使长上下文训练在显存上可行。在此基础上，论文首次报告了机器人闭环性能随预训练上下文长度稳定提升、且无饱和迹象的缩放律，并解锁了一次性人类视频模仿、在线自我改进（DAgger Distillation）、抗扰动等新能力。

## 难度评级

**高级（Advanced）**。需要预先理解 VLA/扩散或 flow-matching action head、attention 与循环序列模型（RNN/线性注意力）的关系、元学习（梯度的梯度）、以及模仿学习中的 DAgger 范式。公式不多，但机制耦合较深（架构 + 训练配方 + 两种在线学习应用互相支撑），建议按顺序完整学习而非跳读。

## 学习材料导航

按以下顺序学习效果最佳：

1. **[summary.md](summary.md)** — 背景、问题陈述、核心贡献与关键量化结果，5-10 分钟建立全局印象。
2. **[insights.md](insights.md)** — 逐层拆解论文的关键洞察与设计动机（最重要）。
3. **[method.md](method.md)** — 模型架构、训练配方、两种在线学习应用的技术细节与复现要点。
4. **[qa.md](qa.md)** — 15 道自测题（基础/中级/高级），检验理解深度。
5. **[mental-model.md](mental-model.md)** — 这项工作在研究版图中的定位，如何在脑中归类它。
6. **[reflection.md](reflection.md)** — 开放问题、脆弱假设、可能的扩展方向。
7. **code/** — 代码复现目录（当前为空，官方代码尚未发布）。
8. **index.html** — 论文可视化浏览页面（如已生成）。

## 关键要点

- **架构**：TTT 层插在 GR00T N1.7 DiT action head 的注意力层之后；attention 处理单时间步内信息，TTT 处理跨时间步信息。16 个 DiT 层各加一层 TTT（约 10M 参数/层，538M→690M），fast model 为两层 GeLU MLP，16 个 register tokens 跨时间步携带视觉语言信息，tanh 门控（α 初始化 0.001）保护预训练能力。
- **训练配方**：sequence action forcing（逐 action chunk 独立采样噪声水平，否则训练不稳定）+ TBPTT（梯度按 segment 截断、fast weights 跨 segment 传递，显存只取决于 segment 长度而非总序列长度）。
- **上下文缩放律**：预训练上下文从 128 增至 8K，RoboTTT 闭环性能持续上升至 71.5%（8K），比 1K 高 63%，比最佳短上下文基线高 57%，无饱和；对照基线 GDN（Gated DeltaNet，无测试时梯度下降）则无此趋势。
- **新能力**：一次性人类视频模仿（Circuit 未见配置 65% 得分、6/10 成功，GDN 为 0/10）；DAgger Distillation 在线自我改进（平均提升 33%，标准 DAgger 只提升 9%）；抗扰动鲁棒性；十阶段五分钟 Gear Bot 任务上唯一能完整成功的方法。
- **主结果**：平均任务完成分数 79%，比单步基线 GR00T N1.7（42%）高 87%，比最强基线 GDN（56%）高 41%。

## 预计学习时长

约 2-3 小时（含通读原文、完成自测题、整理心智模型）。若只求快速掌握核心思想，读完 summary.md + qa.md 基础题约 30 分钟。

## 目录结构

```
robottt-context-scaling/
├── README.md          本文件：导航与要点速览
├── summary.md          背景、贡献、关键结果
├── qa.md                15 道自测题
├── mental-model.md      研究定位与知识归类
├── reflection.md        开放问题与扩展方向
├── fulltext.txt         论文全文（文本提取）
├── paper.pdf            论文原始 PDF
├── meta.json             元数据
├── images/               论文图片资源
└── code/                 代码复现（待补充）
```
