# GPC 代码 Demo

三个自包含、可独立运行的教学脚本（仅依赖 `numpy`），分别对应 GPC 的三个阶段。

| 脚本 | 对应阶段 | 演示内容 |
|------|---------|---------|
| `fsq_quantization_demo.py` | Stage 1 · Skill Quantization | FSQ 量化公式 `⌊(L/2)tanh(z)⌉`、隐式码本 `9^40`、STE 直通梯度、与 VQ-VAE 显式码本对比 |
| `token_grouping_autoregressive_demo.py` | Stage 2 · Generative Controller | Token 分组（复现 Table 3 权衡）、分组往返无损、自回归 + nucleus(top-p) 采样带来的行为多样性 |
| `cola_layer_demo.py` | Stage 3 · Task Adaptation | CoLA 前向（式 7，DoRA+FiLM）、`<1%` 参数占比、任务条件化使输出随任务观测 `c` 变化 |

## 运行

```bash
pip install numpy
python fsq_quantization_demo.py
python token_grouping_autoregressive_demo.py
python cola_layer_demo.py
```

## 说明

- 这些 demo 的目标是**讲清核心机制**，不是复现论文性能。Transformer 用玩具打分函数模拟，重点展示自回归*流程*与因果依赖，而非真实建模能力。
- `fsq_quantization_demo.py` 里的 STE 恒等式 `z_ste = z_cont + (z_disc - z_cont).detach()` 是 PyTorch 实现的关键，demo 用 NumPy 手动演示其前向/反向语义。
- `cola_layer_demo.py` 中 `B` 给了小的非零初始化以便展示条件化效果；论文中 `B` 初始化为 0，使微调起点严格等于预训练主干。
