# S4R：Scaling for Rigid-Body Interpenetration Resolution

这份中文学习节点围绕论文 [S4R（arXiv:2609.20524）](https://arxiv.org/abs/2609.20524)、作者博客 [To Scale or Not to Scale](https://frank-zy-dou.github.io/blog/to-scale-or-not-to-scale/) 和实现仓库 [Dynamics-Modeling/Penetration_Solving](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/Penetration_Solving) 建立。论文要解决的是：一组静态刚体三角网格已经严重互穿，怎样用较小布局位移把它们变成可以交给物理模拟器的 full-size、无互穿初态。S4R 先围绕固定参考中心把每个物体缩到 `s_min=0.01`，再沿尺度路径逐步恢复到 `s=1`；每个尺度增量解最小范数接触 QP，最后在 full scale 做网格级评估和有界 tail refinement。

**难度**：高级；**性质**：几何优化 + 碰撞检测 + 系统实证；**建议用时**：90–120 分钟。需要理解 signed clearance、closest-point witness、凸 QP、数值 continuation、BVH/FCL 和实验协议。

## 阅读顺序

1. [summary.md](summary.md)：问题、方法、贡献和论文数字。
2. [mental-model.md](mental-model.md)：把 scale、接触和验收放进同一张研究地图。
3. [method.md](method.md)：公式、算法、实现细节和复现风险。
4. [insights.md](insights.md)：为什么有效、何时失败，以及博客带来的概念区分。
5. [code-reading.md](code-reading.md)：仓库当前实现的入口、状态契约和论文差异。
6. [reflection.md](reflection.md)：可检验的延伸问题。
7. [qa.md](qa.md)：15 道分层自测题。
8. [index.html](index.html)：使用论文真实数据的交互式结果/机制探索器。
9. [code/scale_continuation_demo.py](code/scale_continuation_demo.py)：仅依赖 Python 标准库的 2-D 概念 demo。

## 三个关键结论

- S4R 是**静态几何修复器**，不是带 gravity、friction、inertia 的动力学接触求解器。
- 缩小物体制造 clearance；整场景同比缩放不会改变相互穿插关系。中间尺度是求解路径，最终必须按原始 full-size mesh 验收。
- 论文中的 `Pen=0` 是统一 evaluator 在有界求解预算内得到的经验结果；冻结 witness、缓存和线性化都没有给出每步可行性定理。

论文结果与当前 GitHub `main`（核对时 HEAD `91cbaf2672a6f587075ad75ceed770717d5e6e85`）的 README 数字分开记录。论文主要表格使用 `N≤5000`；当前 README 还给出 `N=3000` 的版本化 demo 数字，不能直接当作同一实验复现。

## 目录

```text
s4r-scaling-rigid-body-interpenetration/
├── paper.pdf                 # arXiv PDF
├── meta.json                 # 论文元数据与恰好两个语义标签
├── summary.md                # 结构化摘要
├── method.md                 # 方法和伪代码
├── insights.md               # 深入理解
├── mental-model.md           # 研究地图
├── code-reading.md           # 当前仓库代码核对
├── reflection.md             # 延伸问题
├── qa.md                     # 15 道自测题
├── index.html                # 无外部依赖交互探索器
├── code/                     # 可运行概念 demo
└── images/                   # 从论文 PDF 提取的图
```

## Demo

```bash
python3 code/scale_continuation_demo.py
python3 code/scale_continuation_demo.py --steps 40 --scale-min 0.02
```

这个 demo 用圆盘和循环投影近似接触 QP，只展示“缩小—恢复—full-size 检查”的思想；它没有实现论文的三角网格、FCL、OSQP、SOI 事件、frozen-witness cache 或 GPU Warp oracle。

## 本轮验证边界

已验证：PDF SHA256 与下载文件一致；`meta.json`/`index.json` 可解析且两处均只有两个标签；概念 demo 可运行并在 full-size evaluator 输出 `pen=0`；Python 编译检查通过；交互 HTML 的内联 JavaScript 通过 `node --check` 和结构化控件检查。未在本机完整安装上游 `python-fcl`/CUDA 环境运行仓库 benchmark，也未把作者 README 数字当作本机收据。
