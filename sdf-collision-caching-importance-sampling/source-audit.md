# 原文与公开源码核对

核对日期为 2026-10-19。PDF 为 12 页，SHA-256：

```text
8a4c0e63114699443099307030647527a52affec763111ac788614f2737fb1b4
```

项目页 PDF 与固定源码仓库 `docs/paper.pdf` 的哈希一致。源码固定为 `08539dd3ad5fa9b41fdd160c730747de0b510b4e`，使用了其中 README、LICENSE、`collide.cpp/.h`、`optimize.cpp`、`shapes.cpp`、`solver.cpp`。本地副本保留原始版权头。没有编译或运行原作者的 C++ demo，也没有取得和运行论文 GPU benchmark。

## 主要证据入口

- [原论文](https://savant117.github.io/sdf-dcd-demo/paper.pdf)，第 3–6 页为 Algorithm 1 和方法，第 7–9 页为表 1–5，第 10 页为穿透与动能曲线。
- [项目页](https://savant117.github.io/sdf-dcd-demo/)，核实完整作者、项目链接和 demo 范围。
- [固定版本 README](https://github.com/savant117/sdf-dcd-demo/blob/08539dd3ad5fa9b41fdd160c730747de0b510b4e/README.md)，解释原文没有完全指定的实现细节。
- [固定版本 collide.cpp](https://github.com/savant117/sdf-dcd-demo/blob/08539dd3ad5fa9b41fdd160c730747de0b510b4e/source/collide.cpp)，对应目标、缓存、激活条件和随机种子。
- [固定版本 optimize.cpp](https://github.com/savant117/sdf-dcd-demo/blob/08539dd3ad5fa9b41fdd160c730747de0b510b4e/source/optimize.cpp)，对应实际优化停止条件。
- [固定版本 shapes.cpp](https://github.com/savant117/sdf-dcd-demo/blob/08539dd3ad5fa9b41fdd160c730747de0b510b4e/source/shapes.cpp)，对应 Laplacian、曲率阈值与种子准备。

## 论文算法与 demo 的区别

| 项目 | 论文 | 固定版本公开 demo |
| --- | --- | --- |
| 计算后端 | 表 1、2 为 GPU，其余为单线程 CPU | 单线程 CPU，可编译为 WebAssembly；README 明示场景规模缩小 |
| 响应求解器 | CPU 顺序冲量，GPU AVBD | 顺序冲量，包含 split impulse 修正 |
| 时间步 | 默认 16 ms | `1/60 s`，分 4 个子步，每子步重新检测 |
| 激活 | 两个原始场都小于零 | 另含默认 0.02 的 contact skin |
| 深度 | 式 5 取更负的单场值 | 内部响应使用 `φ_A+φ_B`，非重叠情况另算表面间隙 |
| 法向 | 根据更深的场选梯度及符号 | 显式归一化，缓存点优先匹配历史法向方向 |
| 排序 | 按 `g` 保留较优点 | 对最终候选集合重算，并加入新点替换门槛 0.01 |
| 凸物体对 | Algorithm 1 保留随机探索 | 默认在缓存接触完整时跳过额外随机探索，SGD 除外 |
| NM simplex | 围绕种子的 10% 重叠盒大小四面体 | 轴向 simplex；缓存点步长另做限制 |
| 缓存 | 几何候选的跨帧重用 | 同时保存响应冲量，影响 solver warm start |
| DISPLACE | 正文写为 `φ(x)+sin(kx)sin(ky)sin(kz)` | 用 `k` 做幅值，固定频率 7，README 明示差异 |

这些额外处理可能影响稳定性与耗时。纸面公式、公开 demo 行为与论文 benchmark 需要分别复现。

## 原文中需要保留的不一致

1. 第 4.2 节正文写 Box-stacking 平均 `0.6 ms`，但表 3 的 NM 为 `1.15 ms`，Ellipsoid 为 `0.46 ms`。这里按表格列出，不把正文近似数替换到数据中。
2. 第 4.5 节写 NM 在全部失真下 `t_coll≤1.00 ms`，但表 5 的 TWIST `k=2.5` 是 `1.14 ms`，`k=5` 是 `1.01 ms`。这里保留表值。
3. 图 8 图注将 TWIST 写为 `k=0.20`、DISPLACE 写为 `k=5.0`，与表 5 的参数范围不一致。这里不替作者猜测修订值。
4. 式 7 需按 Laplacian 定义累计三个坐标方向。公开源码如此实现，学习笔记补出了显式求和。
5. 式 2 的根号内是 `ε`。平滑宽度与 `sqrt(ε)` 同量纲，正文把 `ε=0.1` 描述为约十倍 `τ=0.01` 的平滑区域，不能直接作为距离缩放公式。

## 统计与归因边界

表 1–5 的全部行转录在 [sources/metrics.json](sources/metrics.json)。`null` 保留“未报告”或 `FAIL` 状态，不补成零。表中 SDF query 数照原报告保存；公开 demo 的计数器实际在一次双场目标评估时递增，不能不加说明地跨实现比较。

图 6、7 仅做定性读取，原页面截图在 `images/penetration-and-energy.png`。没有生成假冒逐帧曲线、方差、置信区间或图线数字化数据。穿透测试的 1,000 表面样本 oracle 是离散参考，动能参考也不是解析真值。

消融表明确测试了缓存和平滑。所读版本没有给出独立关闭排斥项的等预算消融，因而不能将组合方法的全部稳定性改善单独归因于排斥。

## 元数据校正

技能解析器把会议页眉识别为标题，把 Pacific Graphics 识别为作者，并把摘要延伸到版权与引言。学习包已按 PDF 标题、项目页作者及正文人工校正 `meta.json`；原输出保留在 `sources/parser-output.json`。

本地收录日期为 2026-10-19。没有用 PDF 内部生成时间冒充论文发表日期。
