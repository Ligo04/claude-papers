# 论文与公开源码核对

核对日期：2026-10-14。仓库 revision：`13a00e25477f53a3533938b0325e2d575e3fd68b`。

这是静态阅读结果，没有编译或运行官方 GPU 实现；不能据此推断性能退化幅度，也未证明此 revision 与论文 benchmark 使用的版本一致。源码摘录保存在 sources/upstream，逐文件 SHA256 见 sources/source-manifest.json。

## 1. “没有中间主机同步”与公开实现不同

论文 §1、§3.2 声称输入上传后、最终读回前没有主机同步。公开入口呈现两处明确边界：

- [Python 入口 cuacd/__init__.py:402–410](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/cuacd/__init__.py#L402)：调用 `scipy.spatial.ConvexHull(verts)` 构造初始 hull，随后才把原网格和初始 hull 传入 native 函数。这是 CPU 初始化，并不在论文所指上传之后，但限制了“整个 API 全 GPU”这种更宽泛解释。
- [外层循环 csrc/lookahead.c:320–334](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/csrc/lookahead.c#L320)：C 主机代码执行迭代；逐轮 CC 分解路径读回部件数后同步。
- [计数与终止 csrc/lookahead.c:440–478](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/csrc/lookahead.c#L440)：读回 `n_cutting` 后 `cuStreamSynchronize`，CPU 判断是否退出；再读回错误标志并同步。这些不是只在 debug 分支执行。
- [树层扩展 csrc/lookahead.c:553–583](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/csrc/lookahead.c#L553)：主机按层循环，读回 `next_n` 并同步，再决定后续工作量。

因此可确认的是：大量几何状态和内部操作在设备上；所查公开实现仍保留主机控制及中间同步。“GPU-resident geometry”与“host-free execution”应分开表述。这里观察到的是论文叙述与特定公开 revision 的差异，不能推断作者实验数据造假或系统失效。

## 2. Python API 默认不等于论文默认

| 项目 | 论文 Table 3 / 正文 | CLI 源码 | Python API 源码 |
|---|---|---|---|
| width | 30 | 30 | 60 |
| n_concave_edges | 16 | 16 | 32 |
| depth / width2 | 2 / 5 | 2 / 5 | 2 / 5 |
| merge_hulls | 启用合并后处理 | True | False |
| 末尾 decompose_components | 有连通分量处理 | 启用 | False |
| 逐轮 CC-Split | 启用 | 启用 | 默认没有禁用 |

证据：[API 函数签名 367–374](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/cuacd/__init__.py#L367)、[CLI 参数 227–262](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/cuacd/cli.py#L227)。最后两行不能混淆：API 末尾开关为 False，不意味着逐轮 CC-Split 被关闭。复现应显式传参，而非只调用无参数示例。

API 在 [424–426](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/cuacd/__init__.py#L424) 检查 `width + 4*n_concave_edges < 512`，因为后续有固定候选缓冲区。论文默认最大 94；API 默认最大 188。实际凹边不足时有效候选可以更少。

## 3. 几何与集成边界

官方 README 给出最大包围盒边长归一到 2 的步骤，返回后再反变换。CLI 自行完成尺度处理；Python 使用者必须保持一致。τ 的单位跟随归一化坐标。

论文和 README 说明默认 pool 为可用显存的 70%，但所查 [`csrc/heap.c:170–176`](https://github.com/eliphatfs/cuacd/blob/13a00e25477f53a3533938b0325e2d575e3fd68b/csrc/heap.c#L170) 实际采用 `free_bytes * 0.80`，并设最低 64 MiB。这里应区分文档默认与实现默认，不能直接用文档估算进程占用。数据通过 NumPy 进入；docs/status.md 将 `__cuda_array_interface__` 标为未实现。因而不能把这个 API 当作已验证的 GPU tensor 零拷贝入口。

docs/status.md 的测试通过、缺失功能等记录只属于作者仓库声明。本包没有运行这些测试。文档内部 CLI 合并选项描述也可能滞后；参数默认以所查实际源码为准。

## 4. 对仿真流程的阅读建议

将其视为值得评估的碰撞体预处理器，先选代表资产，固定尺度、阈值、预处理与 convex 数量/顶点上限；分别验收分解耗时、接触几何质量、引擎 cooking 与实际运行时成本。未修改任何 ManaEnv、IsaacLab 或 Newton 工作树。
