# UNITAS：3D-Native World Action Model

> arXiv:2610.12099v1 · 2026-10-08 · cs.RO · 27 pages

UNITAS 重新定义 world action model 的接口：观察、机器人动作和动作导致的场景变化不再分别停留在图像、关节向量或隐式 latent 中，而是在每个 interaction window 里对齐到同一个 metric 3D frame。模型用 action flow 表示人手/夹爪点轨迹，用 scene flow 表示环境点的位移，用 world-aligned 3D positional embedding 固定视觉 token 的物理位置，再用 physical-time trajectory tokenizer 把不同帧率、不同 horizon 的轨迹压成每点一个 token。这样同一个 1.7B 模型既能直接输出可执行 action chunk，也能在 simulator mode 下预测 action-conditioned scene change。

## 难度与阅读路线

- 难度：Advanced。
- 性质：architecture-based empirical system。
- 建议顺序：先看本页与 summary.md；再读 method.md 的三条流和 token；用 insights.md 理解为什么“metric frame”是接口；最后读 qa.md。
- 预计时间：第一次通读 60–90 分钟；结合公式、附录和 demo 约 2 小时。

## 五个关键结论

1. UNITAS 的核心不是“加 depth”，而是让 observation、action flow、scene flow 共享可度量的 3D 坐标和点身份。
2. action flow 与 executable action chunk 可以并行采样；scene flow 以 action flow 为条件，因此模型既能控制也能预测后果。
3. 同一模型有 policy mode 和 simulator mode：前者跳过未来点流、低延迟直接出动作；后者保留点流，支持动作条件下的场景预测。
4. 1.7B UNITAS 在 LIBERO 平均成功率 99.8%、LIBERO-Plus 89.1%、RoboTwin Clean/Randomized 92.94%/93.04%、VLABench SR 56.2%。
5. 真实实验只有两个任务、每任务 20 次：Organize Books 90%，Pack the Ball 80%；论文结论应保持在已评测范围内。

## 文件结构

- README.md：导航、关键结论和证据边界。
- summary.md：背景、方法、数据和量化结果。
- insights.md：为什么 3D-native 表征改变了 WAM 的接口。
- method.md：公式、模块、算法流程、训练配置和复现风险。
- mental-model.md：把 UNITAS 放进 WAM/VLA/3D manipulation 研究地图。
- reflection.md：扩展方向、脆弱假设与失败模式。
- qa.md：5 个基础、5 个中级、5 个高级问题。
- code/world_aligned_point_flow_demo.py：标准库教学 demo。
- index.html：无外部依赖的交互式探索页。
- paper.pdf、paper.txt：论文原件与文本抽取。
- images/：205 个嵌入图像与重命名关键图。
- sources.md、source-status.md、meta.json：来源、证据等级和元数据。

