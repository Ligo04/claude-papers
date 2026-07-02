# KPM-FR 代码 Demo

四个自包含、可独立运行的教学脚本（仅依赖 `numpy` 或标准库），对应 KPM-FR 的核心机制。

| 脚本 | 对应章节 | 演示内容 |
|------|---------|---------|
| `fr_1d_differentiation_demo.py` | §3.3 空间离散 | Gauss-Lobatto 点、Lagrange 微分矩阵 D、外插 R、g₂ 校正矩阵 C、`Q=D-CR`；验证高阶精度与谱类收敛；公共通量 F* 校正为全局连续 |
| `bgk_halftime_flux_demo.py` | §3.5 动理学通量 | Boltzmann-BGK 平衡态、矩恢复宏观量、半时刻分布 `f(h)=A f_s+B g`（式 11）、闭式矩通量、黏性(τ)统一处理；**全程只算矩、不存分布** |
| `memory_footprint_comparison.py` | 内存论点 | 42 字节/点 vs LBM 家族；不同网格总内存；复现赛车 demo（531M 点 ≈ 21 GiB）为何塞得进 RTX 4090 |
| `predictor_corrector_pipeline_demo.py` | §3.6 + §4 | 预测矩 vs 朴素传梯度（16→5 标量）的带宽节省；两 kernel 循环的数据流；守恒验证 |

## 运行

```bash
pip install numpy
python fr_1d_differentiation_demo.py
python bgk_halftime_flux_demo.py
python memory_footprint_comparison.py
python predictor_corrector_pipeline_demo.py
```

## 说明与免责

- 这些 demo 的目标是**讲清核心机制**，不是复现论文的性能或数值精度。
- `bgk_halftime_flux_demo.py` 用 1D 连续速度模型 + Gauss-Hermite 求积近似速度空间积分 `⟨·⟩`，用于展示矩恢复与半时刻通量的**结构**；真实 KPM-FR 在 3D 张量积单元上、用预积分的解析矩表达式。
- `predictor_corrector_pipeline_demo.py` 的 1D 平流用一阶迎风玩具通量，**带明显数值耗散**（真实 KPM-FR 用高阶 FR，耗散极低）。可验证的两点是：**跨单元载荷压缩**与**质量/动量严格守恒**。预测矩标量数（5）为示意，非论文精确值。
- `memory_footprint_comparison.py` 的每点字节数由论文给出的相对比例（42B、18%、34%、53%）反推。
