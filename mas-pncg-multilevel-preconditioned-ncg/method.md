# 方法详解

## 算法概览

```
┌─────────────────────────────────────────────────────┐
│                  MAS-PNCG Pipeline                   │
│                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ 接触检测  │───→│ Woodbury 更新 │───→│ 预条件梯度 │  │
│  │ C_curr   │    │ Level-0 子域  │    │ z_{k+1}   │  │
│  └──────────┘    └──────────────┘    └─────┬─────┘  │
│       │                                     │       │
│       │          ┌──────────────┐    ┌──────┴──────┐│
│       │          │ 全局重启判断  │←───│2D 子空间优化 ││
│       │          │ r_k > δ ?    │    │ (μ,ν) 求解  ││
│       │          └──────┬───────┘    └──────┬──────┘│
│       │                 │                   │       │
│       │          ┌──────┴───────┐    ┌──────┴──────┐│
│       └─────────→│ MAS 完全重建  │    │ 逐子域 CCD  ││
│      (if restart)│ (周期性)     │    │ α_d 步长    ││
│                  └──────────────┘    └─────────────┘│
└─────────────────────────────────────────────────────┘
```

## 组件分解

### 1. IPC 能量函数

E(x) = ½(x - x̃)ᵀM(x - x̃) + h²Ψ(x) + B(x) + D(x)
         惯性              弹性      接触barrier  摩擦

其中 x̃ = xᵗ + hvᵗ + h²M⁻¹f_ext 是惯性预测位置。

接触 barrier：B(x) = κ Σ_{c∈C} b(d_c(x))，当距离 d < d̂ 时激活。

### 2. Sparse-Input Woodbury Level-0 更新（核心创新）

**关键观察**：接触 barrier Hessian 的 SPD 投影后近似为 rank-1 项之和

∇²b(d_c(x)) ≈ k·nnᵀ

其中 k = b''(d_c) 是标量刚度，n = ∇d_c 是接触法线。

**更新策略**：

维护全局接触集 C_curr，与上次重启时的 C_base 对比：

| 接触状态 | 条件 | 更新向量 u | 显著性 ΔS |
|---------|------|----------|----------|
| 新接触 | c ∉ C_base | √k_curr · n_curr | k_curr |
| 法线旋转 | n_curr · n_base < ε_rot (25°) | √k_curr · n_curr | k_curr |
| 刚度增加 | k_curr > k_base | √(k_curr - k_base) · n_curr | k_curr - k_base |
| 刚度减少 | k_curr ≤ k_base | 忽略（保持较硬近似） | ΔS = 0 |

**Top-K 截断**：每个子域 d 保留显著性最高的 K 个更新向量（K=8），形成更新矩阵 U_d ∈ ℝ^{N_d×K}。

**Woodbury 求解器**（Capacitance System）：

对局部预条件方向 z_d = B̃_d · g_d：

1. **Base Solve**: z_base = B_d · g_d（使用缓存的基础逆）
2. **Update Projection**: W_d = B_d · U_d，r = U_dᵀ · z_base
3. **Capacitance Solve**: (I_K + U_dᵀ · W_d) · λ = r
4. **Correction**: z_d = z_base - W_d · λ

B̃_d = B_d - B_d · U_d · (I_K + U_dᵀ · B_d · U_d)⁻¹ · U_dᵀ · B_d

### 3. MAS 层次结构

Level-0（细粒度）：用 Woodbury 增量更新
Level-1 到 L（粗层级）：在全局重启时重建，平时冻结

全局预条件梯度：

z_{k+1} = Σ_d S_dᵀ z_d + Σ_l C_(l)ᵀ M_(l)⁻¹ C_(l) g_{k+1}

### 4. 最优 2D 子空间最小化

搜索方向：p_{k+1} = -μ·z_{k+1} + ν·p_k

在子空间内最小化二次模型 Q(p) = gᵀp + ½pᵀH̃p：

| z^T H̃ z_{k+1}    -z^T H̃ p_k | |μ|   |z^T g_{k+1}  |
| -p^T H̃ z_{k+1}    p^T H̃ p_k | |ν| = |-p^T g_{k+1} |

H̃ 通过 Woodbury 更新近似，矩阵-向量积 H̃v 可高效计算。

系数 μ 起到"自然步长"的作用（α=1.0 作为初始试步长），只需 CCD 约束。

### 5. 全局重启判断

监控正交性损失：

r_k = |g_{k+1}ᵀ z_k| / (g_{k+1}ᵀ z_{k+1})

当 r_k > δ (δ=0.3) 时触发 MAS 完全重建（含粗层级）。

实验中平均每 3-10 次 PNCG 迭代触发一次重启。由于 Woodbury 更新代价极低，重启之间的轻量迭代积累出巨大的计算节省。

### 6. 逐子域保守 CCD（CCCD）

标准 CCD：α_global = min_c TOI(c) → 过于保守

CCCD：每个子域独立计算 α_d

CCD 算法流程：
1. 计算单调区域 [0, α_mon]
2. 在 [0, min(1.0, α_mon)] 内用符号变化检测碰撞
3. 二分搜索定位碰撞点（或达到最小阈值 α_l）
4. 每个子域独立执行：x_{k+1} = x_k + Σ_d S_dᵀ α_d S_d p_{k+1}

**改进的下界**：直接用相对位移约束替代标准 ACCD 的绝对位移约束，提供更紧的估计。

## 完整算法伪代码

```
Algorithm MAS-PNCG:
Input: xᵗ, vᵗ, M, barrier d̂, tol ε, restart δ
Output: x^{t+1}

x₀ ← xᵗ
x̃ ← xᵗ + hvᵗ + h²M⁻¹f_ext
Restart ← True

for k = 0 to IterMax:
    C ← ComputeConstraintSet(x_k, d̂)
    
    if Restart:
        (P_base, H̃) ← RebuildMAS(x_k, C)
        p_{k+1} ← p_base
    else:
        P_{k+1} ← WoodburyUpdate(P_base, C)
    
    g_{k+1} ← ∇E(x_k, x̃, C)
    z_{k+1} ← P_{k+1} g_{k+1}
    v ← H̃ z_{k+1}
    
    if Restart:
        μ ← (z_{k+1}ᵀ g_{k+1}) / (z_{k+1}ᵀ v)
        ν ← 0
    else:
        (μ, ν) ← Solve2DSubspace(z_{k+1}, p_k, g_{k+1}, H̃)
    
    p_{k+1} ← -μ z_{k+1} + ν p_k
    {α_d} ← ConservativeCCD(x_k, p_{k+1})
    x_{k+1} ← x_k + Σ_d S_dᵀ α_d S_d p_{k+1}
    
    if ‖z_{k+1}‖ ≤ ε and Restart:
        break
    
    r_k ← |g_{k+1}ᵀ z_k| / (g_{k+1}ᵀ z_{k+1})
    Restart ← (r_k > δ)
    z_k ← z_{k+1}

return x_{k+1}
```

## 实现陷阱

| 陷阱 | 描述 | 解决方案 |
|------|------|---------|
| SPD 保证 | 接触 Hessian 在法线方向切线分量为负 | 投影到法线方向，消除负特征值 |
| 法线旋转不稳定 | 接触法线大幅旋转导致 Woodbury 更新方向错误 | ε_rot = 25° 阈值检测，按新接触处理 |
| Top-K 溢出 | 子域内更新向量超过 K 个 | 按显著性 ΔS 排序截断，保证 capacitance 系统规模 |
| 刚度减少处理 | 直接减去刚度会破坏 SPD | 忽略刚度减少，保持较硬近似 |
| 正交性退化 | Woodbury 误差累积 | Powell 重启策略，r_k > 0.3 时触发 |
| CCD 数值锁定 | 全局步长被单个最硬接触限制 | 逐子域 CCCD，各区域独立推进 |

## 超参数敏感性

| 参数 | 默认值 | 影响 | 建议 |
|------|-------|------|------|
| K (Top-K) | 8 | 子域 capacitance 系统大小 | 增大提高精度但增加求解代价 |
| ε_rot | 25° | 法线旋转判定阈值 | 太小→频繁重建；太大→方向误差 |
| δ (重启阈值) | 0.3 | 触发全局重启的频率 | 太小→重启过频；太大→收敛退化 |
| d̂ (barrier 距离) | 10⁻³ - 10⁻⁴ | 接触激活范围 | 场景相关，影响接触集大小 |
| ε (收敛容差) | 10⁻⁵ | 终止条件 | 更松→更快但可能有伪影 |
| α_l (CCD 最小步长) | 场景相关 | CCCD 精度下界 | 太大→过保守；太小→迭代多 |
