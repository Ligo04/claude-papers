# 方法详解

## 系统架构

```
┌─────────────────────────────────────────────────┐
│                   Isaac Lab (future)             │
│          (环境包装器、任务定义、训练管线)           │
├─────────────────────────────────────────────────┤
│                     Newton                       │
│        (资产导入、碰撞检测、可视化)                │
│  ┌───────────────────────────────────────────┐   │
│  │              Kamino Solver                 │   │
│  │   (动力学表述、约束求解、GPU 并行化)        │   │
│  └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│                  NVIDIA Warp                     │
│    (Python → CUDA kernel 编译、PyTorch/JAX 互操作) │
├─────────────────────────────────────────────────┤
│                     CUDA                         │
└─────────────────────────────────────────────────┘
```

## 第一阶段：极大坐标动力学建模 (Section 3)

### 状态表示

每个刚体 i 携带独立状态：
- **位姿**: q_i = (r_i, q_i) ∈ ℝ³ × S³ (位置 + 四元数)
- **速度**: u_i = (v_i, ω_i) ∈ ℝ⁶ (线速度 + 角速度)

### 质量矩阵

```
M = diag(M₁, ..., M_nb)

M_i = | m_i·I₃      0        |
      |   0     R_i·I^b_i·R_i^T |
```

块对角结构！每个刚体的惯性独立，不存在交叉耦合项。

### 运动方程 (Newton-Euler)

```
M·u̇ = h(q,u) + J(q)^T·λ
```

- h: 重力 + 陀螺力 + 外力
- J: 约束 Jacobian
- λ: 约束反力（待求解）

### 三类约束的统一

| 约束类型 | 数学形式 | 描述 |
|----------|----------|------|
| 双边关节约束 | f(q) = 0 | 锁定自由度（含闭环！）|
| 单边关节限位 | 0 ≤ λ ⊥ g(q) ≥ 0 | 互补条件 |
| 摩擦接触 | Signorini-Coulomb-Newton + De Saxcé | 法向互补 + 库仑摩擦锥 |

**关键点**: 闭环约束与普通关节约束形式相同 (f(q) = 0)，无需特殊处理。

### 时间积分

半隐式 Euler：
1. 隐式求解约束反力 λ
2. 显式积分位姿（SO(3) 上的指数映射）

也支持 Moreau-Jean 中点格式（在半步配置处评估约束，对闭环系统更稳定）。

### KKT 系统

```
| M    J^T | | u⁺  |   | Δt·h + M·u⁻ |
| J    -R  | | -λ  | = | -v*          |
```

- v*: Baumgarte 稳定化 + 冲击偏置 + PD 控制项
- R: 正则化（关节电枢、阻尼、隐式 PD）

## 第二阶段：Proximal-ADMM 求解器 (Section 4)

### 对偶问题

消去速度变量（Schur 补），得到纯约束反力空间的优化问题：

```
λ* = argmin_{x ∈ K} ½ x^T·D·x + x^T·(v_f + Γ(v⁺(x)))
```

- **D = J·M⁻¹·J^T + R**: Delassus 矩阵（约束空间的逆表观惯性）
- **v_f = J·(u⁻ + Δt·M⁻¹·h) + v***: 自由速度向量
- **Γ(·)**: De Saxcé 修正（确保滑动接触不产生法向穿透力）
- **K = ∏_k K_μk**: 可行锥（库仑摩擦锥的笛卡尔积）

### PADMM 算法

```
输入: D, v_f, K, η, ρ
1. 构建 D_{η,ρ} = D + (η+ρ)I          ← 每时间步一次
2. 初始化 x⁰, y⁰, z⁰ (warm-start)
3. for i = 1,...,N_max:
4.   s_i ← Γ(z_{i-1})                  ← De Saxcé 修正
5.   x_i ← -D_{η,ρ}⁻¹·(v_f + s_i      ← 线性求解 (核心瓶颈)
            - η·x_{i-1} - ρ·y_{i-1}
            - z_{i-1})
6.   y_i ← Π_K(x_i - ρ⁻¹·z_{i-1})     ← 解析锥投影
7.   z_i ← z_{i-1} - ρ·(x_i - y_i)     ← 对偶变量更新
8.   Nesterov 加速 (y_i, z_i)
9.   if ||r_p||∞, ||r_d||∞, ||r_c||∞ < ε:
10.    break
11. return λ* ← y_N
```

**结构性质**: D_{η,ρ} 在同一时间步的所有 ADMM 迭代中保持不变 → 只需一次分解/预处理。

### 加速技术

| 技术 | 作用 |
|------|------|
| Nesterov 加速 | 外推 y, z 变量，加速收敛；残差增大时自适应重启 |
| Warm-starting | 缓存上一时间步的 λ*，接触按几何对 + 位置匹配 |
| 对角预条件 | Jacobi 预条件 P = diag(D)^{-1/2}，改善谱性质 |
| 固定迭代预算 | 线性求解器固定 9 轮，确保 CUDA Graph 捕获 |

## 第三阶段：GPU 并行化 (Section 5)

### 线性系统求解（核心瓶颈）

**方案 A: 稠密 block-Cholesky (小系统)**
```
1. 显式组装 D_{η,ρ}
2. Block-LLT 分解（每时间步一次）
3. 每 ADMM 迭代: 前向-后向替换
```

**方案 B: 稀疏矩阵-free Conjugate Residual (大系统)**
```
1. J 以 BSR (Block-Sparse-Row) 格式存储
2. D·v 通过链式乘法: J^T → M⁻¹ → J 实现，不显式组装 D
3. Warm-started CR 迭代器，固定 9 轮
```

**内存对比**:
- 稠密: O(n²) — Delassus 矩阵存储主导
- 稀疏: O(n) — 每约束固定非零块数

### 异构世界并行化

问题：不同并行世界可能包含不同拓扑的机器人 → 约束维度不同 → 无法直接向量化。

解决方案：
```
┌─ World 0: DR Legs (31 bodies, 6 loops) ─┐
├─ World 1: BDX (21 bodies, 0 loops)      ├─→ 统一 CUDA kernel
├─ World 2: Olaf (36 bodies, 5 loops)     │    通过全局偏移数组
└─ World 3: DR Legs (31 bodies, 6 loops) ─┘    映射到各自数据
```

每个 world 的刚体/约束通过全局偏移数组索引，同一 kernel 处理不同大小的子问题。

## 实现细节

### RL 训练配置 (DR Legs)

```
仿真频率:     250 Hz (Moreau-Jean 中点积分)
策略频率:     50 Hz (decimation = 5)
并行环境:     4096
控制方式:     隐式 PD (Kp=15, Kd=0.6)
策略网络:     MLP 512×512×512, ELU
观测维度:     94 (朝向、速度、步态相位、关节位置、动作历史)
动作维度:     12 (归一化关节位置目标)
训练算法:     PPO (RSL-RL 框架)
吞吐量:       ~3600 env steps/s
总训练时间:   ~31 小时 (单 RTX Pro 6000)
```

### 伪代码：前向动力学一步

```python
def step(state, action, dt):
    # 1. 计算外力
    h = gravity + gyroscopic + external_forces(action)

    # 2. 组装约束 Jacobian
    J = build_constraint_jacobian(state.q)  # BSR 格式

    # 3. 构建 Delassus 系统
    if small_system:
        D = J @ inv(M) @ J.T + R
        factor = cholesky(D + (eta + rho) * I)
    else:
        delassus_op = lambda v: J @ (inv(M) @ (J.T @ v)) + R @ v

    # 4. 计算自由速度
    v_free = J @ (state.u + dt * inv(M) @ h) + v_star

    # 5. PADMM 求解
    x, y, z = warm_start(prev_lambda)
    for _ in range(n_iterations):
        s = de_saxce_correction(z)
        rhs = -(v_free + s - eta*x - rho*y - z)
        x = solve(factor, rhs)  # 或 iterative_solve(delassus_op, rhs)
        y = cone_projection(x - z/rho, friction_cones)
        z = z - rho * (x - y)
        nesterov_accelerate(y, z)

    lambda_star = y

    # 6. 更新速度和位姿
    state.u = state.u + dt * inv(M) @ (h + J.T @ lambda_star)
    state.q = integrate_se3(state.q, state.u, dt)

    return state
```

## 关键实现陷阱

1. **四元数归一化**: 积分后需要重新归一化，否则旋转漂移
2. **接触 warm-start 匹配**: 接触点可能出现/消失，需要按几何对和空间位置匹配缓存的反力
3. **CUDA Graph 捕获**: 要求确定性的 kernel 执行序列 → 固定迭代预算
4. **Baumgarte 稳定化**: 约束漂移的数值稳定化，参数需要调整
5. **De Saxcé 修正**: 非关联库仑摩擦的正确处理，防止滑动接触产生穿透力
