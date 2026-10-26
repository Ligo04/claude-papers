# PhysLDM 中文学习资料

PhysLDM 把体积软体的未来运动写成条件轨迹生成问题。空间 VAE 压缩每帧局部形变，ST-VAE 压缩整段时间，最后由条件 rectified-flow 模型生成潜空间轨迹。全局刚体运动使用专门通道，细小形变保留在规范坐标中。学习重点是表示压缩、时间精度、多模态预测，以及代理模型可微性的适用范围。

原文为 [arXiv:2610.07609v3](https://arxiv.org/abs/2610.07609v3)，45 页，最新修订日期 2026-10-18。本资料于 2026-10-20 核对。[作者项目页](https://yucrazing.github.io/physldm/)提供视频与可视化。

- 难度为高级；论文属于架构设计与实验研究，涉及三阶段训练、时空表示、flow matching 和高阶自动微分。
- 语义标签为「变形体仿真」「潜空间扩散」。
- 建议用 20 分钟阅读直觉和摘要，40–60 分钟阅读方法与实验边界，再用 20–30 分钟运行演示、做问答。时间为学习安排估计。

先记住三个区别：`2.48 mm` 是完整轨迹的重建误差，`28.2 mm` 是 Objaverse held-out 的未来预测误差；一次生成 64 帧包含多步 flow 积分；“训练不显式使用本构公式”不意味着模型自动适应任意真实动力学。

在网页阅读器左侧选择同名文件。`index.html` 是交互入口。

| 文件 | 阅读目的 |
| --- | --- |
| `mental-model.md` | 从一只落地软物体理解输入、输出和三类模型 |
| `summary.md` | 核心结果、实验范围和量化指标 |
| `insights.md` | 方法为何有用、哪些结论需要限定 |
| `method.md` | 架构、张量尺寸、损失、采样与高阶梯度 |
| `qa.md` | 基础、中级、高级各 5 道自测题 |
| `comparison-with-diffgipc.md` | 与本次会话上一篇 DiffGIPC 的概念对照 |
| `reflection.md` | 可检验的扩展与复现方案 |
| `evidence.md` | 公式、表格页码和验证边界 |
| `index.html` / `data.json` | 交互数据页与核对后的原文数值 |
| `code/multimodal_trajectory_flow_demo.py` | 合成多模态轨迹、解析 flow 和 DCC 演示 |
| `code/demo_results.json` | 演示实际输出 |
| `paper.pdf` / `paper.txt` / `meta.json` | 原文、完整文本和元数据 |
| `images/` / `sources/` | 提取图片、图表页面和来源记录 |

运行演示：

```bash
python3 /home/ligo/claude-papers/papers/physldm-latent-deformable-simulation/code/multimodal_trajectory_flow_demo.py
```

演示只依赖 NumPy，使用解析混合分布解释平均轨迹与采样轨迹的区别，未训练 PhysLDM。尚未复现官方模型、GPU 吞吐或真实数据实验；核对官方论文与项目页时，未确认代码、权重和完整数据的下载入口。
