# 图像索引

图像由 `extract-images.py` 从 24 页 PDF 中提取。PDF 的多面板图片会被编码成多个独立资源，因此目录里同时保留了原始的 `page_*` 文件和便于阅读的命名版/整页预览。

## 命名版关键图

| 文件 | 对应内容 |
|---|---|
| `figure-1-rendered-vs-dlss5.png` | Figure 1：渲染输入与 DLSS 5 输出 |
| `figure-2-streaming-latency-memory.png` | Figure 2：分块与逐帧流式的延迟/内存 |
| `figure-3-backbone-tradeoff.png` | Figure 3：性能—质量权衡 |
| `figure-4-5-appearance-distribution.png` | Figure 4–5：分布迁移与外观模型子集 |
| `figure-6-tone-structure-control.jpeg` | Figure 6：结构/色调强度 |
| `figure-7-manual-masking.png` | Figure 7：手工区域掩码 |
| `figure-8-automatic-mask.png` | Figure 8：自动 skin 掩码 |
| `figure-9-qualitative-comparison.png` | Figure 9：与离线基础模型的定性比较 |
| `figure-10-ray-budget-comparison.png` | Figure 10：4× 光线预算对照 |
| `figure-11-low-quality-input.png` | Figure 11：低质量输入的限制 |

整页预览由 `pdftoppm` 生成，只为保留图注和多面板布局；它们不是额外实验结果。原始 PDF 与文本核验文件位于上一级目录。
