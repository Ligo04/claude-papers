# CuACD 中文精读

CuACD 研究怎样把三角网格快速变成一组凸碰撞体。它沿用 CoACD 的碰撞感知凹度，把切割、凸包和搜索中的几何计算改成 warp 协作，并用设备端 heap 处理动态大小的中间数组。论文报告 78–104× 的分解加速；本学习包同时记录公开实现与“全程无主机同步”叙述之间的差异。

- 论文：CuACD: A Fully GPU-Resident Approximate Convex Decomposition
- 作者：Ruoxi Shi、Xinyue Wei、Fanbo Xiang、Zexiang Xu、Hao Su
- 版本：arXiv:2609.28731v1，提交日期 2026-09-23；学习日期 2026-10-14。
- 难度：Advanced。性质：系统设计 + 计算几何 + 实验；无需深度学习基础。
- 语义标签：凸分解、Warp协作。
- 阅读前提：凸包、网格拓扑、CUDA warp、采样误差；method.md 会解释必要概念。

## 怎样学习

1. 10 分钟：交互页“概览”与 summary.md，先分清输入、输出和计时范围。
2. 25 分钟：insights.md 与 mental-model.md，理解为什么 warp 和 heap 必须配合。
3. 35 分钟：method.md、交互页“方法”，跟随切割 → 凸包 → 评分的过程。
4. 20 分钟：source-study.md，核对论文、CLI、Python API 的差异。
5. 20 分钟：运行教学 demo，再完成 qa.md 的 15 道题。

预计完整学习 90–120 分钟。交互页包含全部中文笔记和问答，可单文件离线打开。

## 记住四点

- 加速来自系统组织；质量收益与凹边候选平面有关，两者要分开判断。
- 论文默认根候选预算为 30 + 4×16 = 94，深度 2；不是每个轴 30 个再加 64。
- 0.23 s 是 V-HACD 数据集的平均分解时间，不含 PaMO，不是任意网格或完整资产导入耗时。
- 所查源码保留主机控制、同步与初始 CPU 凸包。GPU 内部几何阶段融合的价值仍成立，但不能宣称已验证全程无 CPU 参与。

## 文件导航

| 文件 | 内容 |
|---|---|
| index.html | 单文件交互学习页：概览、方法、结果、源码边界、完整精读、问答 |
| summary.md / metrics.json | 实验与可复核原始表格数据 |
| insights.md / mental-model.md | 核心理解与研究定位 |
| method.md | 公式、组件、伪代码、参数与复现风险 |
| source-study.md | 固定 revision 的源码核查 |
| qa.md / reflection.md | 15 问答与研究延伸 |
| code/warp_geometry_primitives_demo.py | 标准库 CPU 教学：ballot 压缩、分段排序、二维凸包预筛选 |
| code/demo_result.json | 教学 demo 实际输出 |
| sources.md / source-status.md | 证据定位与验证边界 |
| paper.pdf / paper.txt | 原文及保留布局的全文 |
| images/ | 原图与矢量图渲染；output/playwright/ 保存页面验收截图 |
| sources/upstream/ | 只读源码摘录；manifest 固定 SHA 和逐文件校验和 |

运行：

```bash
python3 code/warp_geometry_primitives_demo.py --self-test --json code/demo_result.json
```

教学代码不调用 CUDA，不实现完整三维 ACD；本包未运行官方 GPU benchmark，也未接入 IsaacLab、Newton 或 ManaEnv。
