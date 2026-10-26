# DiffGIPC 学习资料

DiffGIPC 研究接触丰富的逆问题。软体使用 FEM，近刚体使用每个物体 12 个自由度的 affine 表示；二者在同一个逐帧伴随系统中求解，让损失能跨碰撞、摩擦和关节传回形状、初始状态与控制参数。GPU 执行进一步降低反复计算轨迹和梯度的成本。

原文：[arXiv:2610.19091v1](https://arxiv.org/abs/2610.19091v1)，2026-10-13 提交，共 30 页。学习资料核对日期为 2026-10-19。

- 难度为高级。论文属于系统设计与实验研究，核心方法涉及隐式微分、轨迹伴随和混合坐标的链式法则。
- 两个语义标签为「可微接触仿真」「混合坐标伴随」。
- 建议先花 20 分钟阅读 `mental-model.md` 和 `insights.md`，再花 40–60 分钟推导 `method.md`，最后用 20–30 分钟运行演示并完成问答。这是学习时间估计。

阅读后应能解释：为什么 12 个运动自由度能保留详细表面；为什么每帧只解一个系统仍能传递跨物体、跨时间的梯度；为什么形状导数必须包含质量、接触映射和摩擦历史的变化；为什么 4133.3× 需要结合基准与表示方式理解。

在网页阅读器中，点击左侧同名文件查看材料；`index.html` 是交互入口。

| 文件 | 内容 |
| --- | --- |
| `summary.md` | 背景、贡献、实验结果及数值口径 |
| `insights.md` | 核心直觉、推导含义、局限与工程判断 |
| `method.md` | 公式、算法、实现细节与复现风险 |
| `mental-model.md` | 前置知识和研究位置 |
| `qa.md` | 基础、中级、高级各 5 道自测题 |
| `reflection.md` | 可检验的扩展问题 |
| `evidence.md` | 页码、表格、数据差异与验证边界 |
| `index.html` | 离线交互页面，使用论文表格数据 |
| `code/contact_trajectory_adjoint_demo.py` | 可独立运行的教学演示 |
| `code/demo_results.json` | 本机演示的实际输出 |
| `data.json` | 人工核对后的论文表格数据及来源 |
| `images/` | PDF 提取图片和关键图表的整页渲染 |
| `sources/` | arXiv HTML、提取记录和来源说明 |
| `paper.pdf` / `paper.txt` / `meta.json` | 原文、保留版式的全文和元数据 |

运行演示：

```bash
python3 /home/ligo/claude-papers/papers/diffgipc-mixed-coordinate-contact/code/contact_trajectory_adjoint_demo.py
```

演示仅依赖 NumPy，检验接触耦合、时间递推、affine 点映射、质量矩阵和直接损失的导数。演示场景为合成数据，未复现三维 FEM、摩擦、CCD、CUDA PCG 或论文运行时间。

核对时，论文写明代码将发布，作者主页的 DiffGIPC 条目仅链接 arXiv。尚未确认可用的官方 DiffGIPC 源码入口。已有 libuipc 或 StiffGIPC 的源码不等于本文实现已经发布。
