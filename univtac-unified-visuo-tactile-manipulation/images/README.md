# 图像索引

图像由 `extract-images.py` 从 13 页 PDF 中抽取。多面板图会被保存为多个 `page_*` 资源；同时生成整页预览，保留图注和面板关系。

## 命名版关键图

| 文件 | 对应内容 |
|---|---|
| `figure-1-encoder-pipeline.png` | Figure 1：三路径 Encoder 与策略集成 |
| `figure-2-tactile-reconstruction.png` | Figure 2：RGB、深度和 marker 位置重建 |
| `figure-3-benchmark-tasks.png` | Figure 3：八项 UniVTAC Benchmark 任务 |
| `figure-4-6-experiment-results.png` | Figure 4–6：数据规模、真机帧和实验设置 |
| `figure-7-real-world-rollouts.png` | Figure 7：真机 tactile / vision rollout 对照 |
| `figures-contact-sheet.png` | 五张整页预览的缩略导航 |

原始 `page_*` 文件保留了 PDF 的图像资源；命名版是便于阅读的整页 raster 预览，不是额外实验数据。
