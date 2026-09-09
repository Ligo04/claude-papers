# MeshFEM：非线性有限元的块加速求解器

这篇 SIGGRAPH 2026 技术论文把非线性有限元/几何处理中的 Newton 线性代数链路作为一个整体优化：局部元素能量让 Hessian 呈现“稀疏的稠密小块”结构，MeshFEM 因而从块稀疏模式构建、并行 Hessian 装配、符号分析，到数值分解和求解都做专门加速。公开摘要还报告了注入曲面参数化与带接触弹性仿真两个应用方向。由于 ACM 正文在本环境不可下载，本学习包将摘要事实与官方代码观察分开标注。

## 学习路线

1. 先看 [`summary.md`](summary.md)，建立问题、贡献和证据边界。
2. 再看 [`mental-model.md`](mental-model.md) 和 [`method.md`](method.md)，把“元素能量 → 块 Hessian → Newton 求解器”串起来。
3. 阅读 [`insights.md`](insights.md)，重点理解为什么块结构会同时影响装配、符号分析、数值分解和 solve。
4. 运行 [`code/block_hessian_demo.py`](code/block_hessian_demo.py)，观察同一个 Hessian 的标量/块表示和 Newton 步。
5. 打开 [`index.html`](index.html)，用滑块探索维度与块连接数对存储量的影响。
6. 用 [`qa.md`](qa.md) 自测；最后读 [`reflection.md`](reflection.md) 讨论可扩展方向。

## 难度与时间

- 难度：Advanced；需要稀疏矩阵、Newton 法、有限元局部装配和 Cholesky 的基础。
- 建议时间：首次通读 60–90 分钟；再花 30 分钟运行 demo 并对照源码。

## 关键结论

- 不能只优化 Hessian-vector product：在直接 Newton 方法中，模式、装配、symbolic factorization、numeric factorization、solve 都可能成为瓶颈。
- 维度为 `d` 的节点变量产生 `d × d` 稠密块；保留块边界可以减少索引和调度开销，并让小矩阵 kernel 有机会发挥作用。
- 变化不频繁的接触稀疏模式适合缓存/复用，但缓存必须有预算和过期策略，否则会用内存换来失控的重分析成本。
- 本目录不包含论文原图和精确表格数字，因为正文下载受限；这不是对论文结果的臆测替代。

## 目录结构

```text
meta.json          元数据、摘要、来源状态和两个语义标签
source-status.md   可访问来源与全文缺口
summary.md         背景、贡献、结果和证据等级
insights.md        核心洞见、权衡、局限和工程含义
method.md          算法链路、伪代码、实现陷阱
mental-model.md    研究地图与心智模型
reflection.md      延伸问题和脆弱假设
qa.md              5/5/5 分级自测题
code/              NumPy-only 可运行概念演示
images/            因 ACM 403 暂无原图，见其中 README
index.html         无外部依赖的交互式块稀疏探索器
```
