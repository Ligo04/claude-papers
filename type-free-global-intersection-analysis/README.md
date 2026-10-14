# Type-Free Global Intersection Analysis with Linear Displacement Fields：中文学习包

这篇 ACM SIGGRAPH Asia 2026 论文研究“已经穿透之后怎样把薄壳解开”：它把每条 edge–face (EF) 相交 contour 当作种子，在候选线性分离方向上执行 VF/EE/FV ray casting，用双侧乘积复形上的 cluster culling 保留真正连到 contour 的 response support，再以最小位移代价选择方向，并把 hit 转成隐式 Newton/PCG 求解器可直接使用的 penalty pairs。论文的关键转变是把 GIA 的离散 contour 分类和 ICM 的局部 contour-shortening 都换成同一个几何搜索接口；它提高了复杂嵌套穿透下的实用收敛性，但不提供严格的全局解缠保证。

- **难度**：Advanced；几何计算、碰撞检测、隐式积分和数值优化交叉。
- **论文性质**：算法/系统设计 + 实证评估。
- **建议学习时间**：第一次通读 60–90 分钟；配合公式和代码 3–5 小时。
- **两个标签**：`几何解相交`、`方向优化`。

## 阅读顺序

1. [summary.md](summary.md)：先抓问题、贡献和数字结果。
2. [mental-model.md](mental-model.md)：建立“contour → direction → support → penalty”心智模型。
3. [method.md](method.md)：按公式和伪代码重建方法。
4. [insights.md](insights.md)：理解它为何能绕开 GIA/ICM 的失败模式。
5. [source-code.md](source-code.md)：把论文段落映射到仓库当前实现。
6. [code/linear_displacement_demo.py](code/linear_displacement_demo.py)：运行一个二维教学代理；它不是官方复现。
7. [index.html](index.html)：在浏览器里用滑块查看真实实验数字、候选方向和代价排序。
8. [qa.md](qa.md) 与 [reflection.md](reflection.md)：检查理解并提出后续研究问题。

## 目录

```text
paper.pdf                  原始 17 页 PDF
meta.json                  标题、作者、来源和两个语义标签
summary.md                 背景、贡献、结果
mental-model.md            学习地图
method.md                  公式、流程、伪代码和复现风险
insights.md                关键洞见、比较和限制
source-code.md             GitHub 当前代码的映射与状态
validation.md              本轮静态验证和未运行边界
qa.md                      5 基础 + 5 中级 + 5 高级问题
reflection.md              延伸方向与可证伪假设
code/                      可运行的最小概念 demo
images/                    从论文抽取的图像
index.html                 离线交互式学习页
```

## 来源固定点

- 论文 PDF：[chengzhuuwu.github.io/files/untangling_siggraph_asia_2026.pdf](https://chengzhuuwu.github.io/files/untangling_siggraph_asia_2026.pdf)
- 代码仓库：[github.com/ChengzhuUwU/Untangling26](https://github.com/ChengzhuUwU/Untangling26)
- 本次代码阅读固定到 `main` 的 `d361b71e4ae97b2604c3b704fe31299e3b5dfb2f`（2026-09-19）。仓库会继续变化，代码描述以该 SHA 为准。

论文 PDF 的自动 metadata 不可靠；标题、作者、摘要和实验数字以首页正文、`pdftotext -layout` 和仓库 README/ResearchLog 交叉核对。论文中的结果是作者报告，代码验证状态见 [validation.md](validation.md)。
