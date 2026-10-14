# 方法拆解

## 1. 输入与参数

三角网格 `M={X,F}`，点 `p=(f,b)` 由三角形索引和重心坐标表示。HKTex 是 `S` 个核及一个底色：

```text
κ_s = (p*_s, θ_s, η_s, τ_s, ς_s, c_s)
```

位置是面内连续点；`θ,η` 决定 ALBO；`τ,ς` 控制尺度和边界；`c` 是颜色。

## 2. 从 LBO 到热核

正文第 4 页印作 `L=-MW`，第 5 页的热方程与 `exp(-tL)` 也存在符号约定不一致。实现时应从一致的广义谱问题 `WΦ=MΦΛ` 出发：对半正定刚度矩阵 W，定义 `A=M^-1 W`，使用 `du/dt=-Au` 与 `exp(-tA)`；这是笔记采用的标准化记号，不能把正文印刷形式直接当作代码。M 是质量矩阵，各向异性通过改变局部度量进入 W。

点源热扩散的截断谱表达是：

```text
h_t(p,p*) = Σ_{k=1..K} exp(-t λ_k^(η,θ))
             φ_k^(η,θ)(p) φ_k^(η,θ)(p*) m*
```

归一化除以源点温度，使中心值为 1。归一化本身并不能一般地证明所有点的值均不超过 1，谱截断/插值后仍需数值检查。双调和距离门控为

```text
g(p,p*) = exp(-d_BH(p,p*)² / (2 σ_BH²))
```

然后用 sigmoid 软阈值把 `g·h̄_t` 映射为最终权重 `α(p)`。这个分层结构把“扩散形状”“局部支撑”“边界锐度”解耦。

## 3. 连续参数查询

论文预计算 7 个角度 × 7 个各向异性节点的 ALBO 特征基和特征值。查询 `(θ,η)` 时把 `η` 映射为 `s=ln(1+η)`，在半极坐标网格中找包围单元，再对四个角点做双线性插值。为避免谱基的排列、旋转和符号歧义，构建网格时依次执行：

1. 质量加权相关矩阵 + Hungarian assignment，对齐特征向量排列；
2. 质量加权 Procrustes，通过 SVD 消除特征子空间旋转；
3. 按与参考基的内积修正逐列符号。

## 4. 连续空间评价与 KNN

所有顶点谱量 `A` 都用三角形重心插值 `b_p(A)`。将 `Z=Φ_iso Λ_iso^-1` 作为双调和嵌入时，须排除零特征值对应的常数模态或使用伪逆，避免除零。在该空间做 GPU FAISS KNN，只为 `K_s` 个候选源评价谱核。论文简写为 `O(P×K_s)`；显式保留谱维数 K 时约为 `O(P×K_s×K)`，另有 KNN 检索成本。高斯门控仅衰减而非严格截断，有限 KNN 仍是近似。

## 5. 颜色与训练目标

```text
K(p) = top-K_c kernels by α_s(p)
C(p) = c_base + Σ_{s∈K(p)} α_s(p)c_s / Σ_{s∈K(p)} α_s(p)
```

有 UV GT 时，在网格表面均匀采点，最小化 `||C(p)-C_GT(p)||²`。有多视图时，从相机采样命中网格的射线，最小化渲染像素误差；Mitsuba3 + Dr.Jit + PRB 让误差回传到核参数。

## 6. 流形优化与密度控制

```text
g ← ∇_{p*} L
v ← project_to_tangent(g, T_{p*}M)
p_new ← Exp(p*, -lr · v)
momentum_new ← ParallelTransport(momentum, p* → p_new)
```

剪枝记录命中率 `ν` 与能量 `ω`；低于阈值的核删除。稠密化按 L1 颜色误差分配到贡献核：小核原地 clone，大核沿主轴用指数映射 split，并用增长比率限制单步新增数量。

## 7. 伪代码

```text
precompute isotropic LBO (64 dims)
for angle in 7 angles:
  for anisotropy in 7 log-spaced values:
    eigendecompose ALBO (256 dims)
    align basis to previous/reference basis

initialize sources by area oversampling + farthest-point sampling
initialize colors, theta, eta, tau, sharpness

repeat:
  sample surface points or camera rays
  interpolate spectral quantities at target/source points
  candidates = FAISS_KNN(biharmonic_embedding)
  alpha, color = evaluate_local_heat_kernels(candidates)
  loss = photometric_or_rendering_error(color)
  Adam-update non-position parameters with projected ranges
  Riemannian-SGD-update positions with Exp and parallel transport
  accumulate hit/energy/error statistics
  prune weak kernels; clone or split high-error kernels
```

## 8. 复现风险

- 需要可靠的流形网格与 LBO/ALBO 特征分解；特征值近重合时谱对齐要特别小心。
- straightest-geodesic、GPU FAISS、Triton、Mitsuba3/Dr.Jit 的版本与 CUDA 组合会影响结果。
- 论文把许多剪枝、稠密化和优化超参数放在 supplementary；只按正文无法得到完全一致的训练轨迹。
- 质量因子：作者以参数补偿解释 `m*=1` 的效果；但按式 (3)–(4) 字面推导，分子分母含相同源质量，非零时会精确约掉。因此归一化核中不需要用学习补偿这个因子；未归一化扩散或其他质量用途应另行核对。
- 式 (7) 在所有权重为零时分母为零，实际实现须明确底色回退或 epsilon 策略；残差颜色不必属于 [0,1]，最终材质范围需另作处理。
- 全局 Procrustes 旋转若混合不同特征值的模态，仅重排特征值通常不能保持原热算子。本文将其视为作者采用的近似对齐；应检查旋转是否限制在近简并子空间，或同步变换谱算子。这里是公式层面的复现疑问，未审计官方实现。
- 表 1 的 Storage 对所有方法计入拓扑/UV/网络开销；复现比较时必须保持相同数据格式、压缩方式和预算定义。
