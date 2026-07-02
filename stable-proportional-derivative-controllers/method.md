# 方法详解：SPD 的推导、算法与实现

## 1. 从传统 PD 说起

离散时间域下的 PD 控制器：

$$\tau_n = -k_p(q_n - \bar q_n) - k_d\dot q_n \tag{1}$$

- $q_n, \dot q_n$：第 $n$ 步的位置与速度
- $\bar q_n$：目标位置
- $k_p$：比例增益（"弹簧硬度"）
- $k_d$：微分增益（"阻尼系数"）

**失稳机制**：要快速消除偏差 → $k_p$ 必须大 → 控制力随仿真累积发散 → 必须缩小 $\Delta t$。

## 2. SPD 公式（核心）

把控制力改用**下一时刻**状态：

$$\tau_n = -k_p(q_{n+1} - \bar q_{n+1}) - k_d\dot q_{n+1} \tag{2}$$

$q_{n+1}, \dot q_{n+1}$ 未知 → 一阶泰勒展开（截断到前两项）：

$$q_{n+1} = q_n + \Delta t\,\dot q_n$$
$$\dot q_{n+1} = \dot q_n + \Delta t\,\ddot q_n$$

代入 (2)：

$$\boxed{\tau_n = -k_p(q_n + \Delta t\,\dot q_n - \bar q_{n+1}) - k_d(\dot q_n + \Delta t\,\ddot q_n)} \tag{3}$$

注意第二项里出现了 $\ddot q_n$ —— 这正是要通过牛顿第二定律求解的未知量，因此 (3) 是**隐式方程**。

### 带参考速度的版本（减少跟踪滞后）

实践中常用非零参考速度版本以减小跟踪延迟：

$$\tau_n = -k_p(q_n + \Delta t\,\dot q_n - \bar q_{n+1}) - k_d(\dot q_n + \Delta t\,\ddot q_n - \bar{\dot q}_{n+1}) \tag{5}$$

实验显示，SPD + 参考速度时，跟踪误差随增益**近似线性下降**（见 summary 的误差表）。

## 3. 1D 玩具例子（理解稳定性的钥匙）

单自由度系统 $m\ddot q_0 = \tau_0$，目标维持 $q$ 在 $\bar q$，初始 $q_0=p, \dot q_0=\dot p$：

$$m\ddot q_0 = \tau_0 = -k_p(p + \Delta t\dot p - \bar q) - k_d(\dot p + \Delta t\ddot q_0) \tag{6,7}$$

移项解出加速度：

$$\ddot q_0 = -\frac{k_p(p + \Delta t\dot p - \bar q) + k_d\dot p}{m + k_d\Delta t} \tag{8}$$

**关键观察**：分母 $m + k_d\Delta t$。取极端情形 $k_p\to\infty$ 且 $k_d = k_p\Delta t$：

$$\ddot q_0 = -\frac{p + 2\Delta t\dot p - \bar q}{\Delta t^2}$$

虽然 $k_p=\infty$，加速度**有限**。一步前向欧拉后：

$$\dot q_1 = -\dot p - \frac{p - \bar q}{\Delta t}, \qquad q_1 = p + \Delta t\dot p$$

再一步：

$$q_2 = \bar q$$

→ **任意大增益 + 任意步长，两步精确收敛，永远稳定。**

## 4. 广义坐标下的多体系统

铰接刚体的运动方程（广义坐标）：

$$M(q)\ddot q + C(q,\dot q) = \tau_{int} + \tau_{ext} \tag{9}$$

- $M(q)$：质量矩阵（非线性依赖 $q$）
- $C(q,\dot q)$：科氏力/离心力（非线性依赖 $q, \dot q$）
- $\tau_{ext}$：外力（重力等）
- $\tau_{int}$：内部控制力矩，由 SPD 给出

SPD 内部力：

$$\tau_{int} = -K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d(\dot q_n + \ddot q_n\Delta t) \tag{10}$$

代入 (9) 并对 $\ddot q_n$ 求解：

$$\boxed{\ddot q_n = (M + K_d\Delta t)^{-1}\big(-C - K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d\dot q_n + \tau_{ext}\big)} \tag{11}$$

$K_p, K_d$ 为对角矩阵。求出 $\ddot q_n$ 后用**显式欧拉**积分到下一步。

### 为什么不用全隐式积分？

全隐式积分（对整个 (9) 用下一帧状态）需要**线性化** $M$ 和 $C$：
- 要算更高阶导数；
- 要小心处理旋转表示的**奇异性**（欧拉角万向锁、四元数约束等）；
- 推导和实现都很复杂。

SPD **完全不需要线性化** —— 只把控制力用下一帧状态，$M, C$ 仍取当前值。这大大简化推导/实现并节省计算。

### 稳定性特征对比

| | 全隐式积分 | SPD |
|---|---|---|
| 稳定性 | 无条件稳定 | 条件稳定 $K_d\ge K_p\Delta t$ |
| 阻尼 | 常过阻尼（毁掉时序关键动作） | 不过阻尼 |
| 计算 | 贵 | 便宜 |

## 5. 算法流程（伪代码）

```python
# 每个仿真步：
def spd_step(q, qdot, q_des_next, qdot_des_next, Kp, Kd, dt,
             mass_matrix_fn, coriolis_fn, external_force_fn):
    M = mass_matrix_fn(q)                 # 当前质量矩阵
    C = coriolis_fn(q, qdot)              # 当前科氏/离心力
    tau_ext = external_force_fn(q, qdot)  # 重力等外力

    # SPD 核心：把 Kd*dt 加到质量矩阵上
    A = M + Kd * dt                       # (M + K_d Δt)

    # 右端项：注意用 q + qdot*dt（预测的下一帧位置）
    p_term = -Kp @ (q + qdot * dt - q_des_next)
    d_term = -Kd @ qdot
    # 若用参考速度版本(5)，d_term 改为 -Kd @ (qdot - qdot_des_next)... 见下注

    b = -C + p_term + d_term + tau_ext

    qddot = solve(A, b)                   # 解线性方程组，得加速度

    # 显式欧拉积分
    qdot_next = qdot + qddot * dt
    q_next    = q + qdot * dt             # 注意：用旧速度（前向欧拉）
    # 也可用 semi-implicit: q_next = q + qdot_next * dt

    # 实际施加的控制力矩（如需输出给黑盒仿真器/机器人）
    tau = -Kp @ (q + qdot*dt - q_des_next) - Kd @ (qdot + qddot*dt)
    return q_next, qdot_next, tau
```

**注**：参考速度版本 (5) 中阻尼项变为 $-K_d(\dot q_n + \Delta t\ddot q_n - \bar{\dot q}_{n+1})$，移项后右端多一个 $+K_d\bar{\dot q}_{n+1}$，左端的 $A$ 不变。

## 6. 约束仿真扩展（替代罚函数法）

罚函数法 = 用 PD/弹簧跟踪约束，因此同样有稳定性问题。SPD 可优雅替代。

### 粒子在曲线上（particle-on-curve）

先用 (3) 预测 $q_{n+1}$，找曲线上最近点 $\bar q_{n+1}$，方向 $d = (\bar q_{n+1}-q_{n+1})/|\cdot|$ 与曲线正交。约束力：

$$\tau = -k_p(q_{n+1} - \bar q_{n+1}) - k_d\, d d^T \dot q_{n+1} \tag{12}$$

注意阻尼项是 $d d^T\dot q_{n+1}$ —— **只阻尼沿约束违反方向 $d$ 的速度分量**（投影矩阵 $dd^T$）。修正后的运动方程：

$$(M + k_d\Delta t\, dd^T)\ddot q = -k_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - k_d\, dd^T\dot q_n \tag{13}$$

### 距离约束（distance constraint）

两粒子保持距离 $r_0$（无质量杆），$d=(q_1-q_2)/|q_1-q_2|$：

$$\tau_1 = -k_p(q_1^{n+1} - q_2^{n+1} - r_0 d) - k_d\, dd^T(\dot q_1^{n+1} - \dot q_2^{n+1}), \quad \tau_2 = -\tau_1 \tag{14,15}$$

修正运动方程（$D = \begin{bmatrix} I & -I \\ -I & I\end{bmatrix}$）：

$$(M + k_d\Delta t\, dd^T D)\ddot q_n = -k_p(D(q_n + \Delta t\dot q_n) - r_0 d) - k_d\, dd^T D\dot q_n \tag{16}$$

### 实验：3D Tinkertoy 螺旋

- 实验一：2400 段平滑螺旋，绿粒子在线约束下滑，紫粒子用距离约束当摆。
- 实验二：仅 24 段**折线**螺旋（含非可微尖角），用 $k_p=10^{16}, k_d=1.5k_p\Delta t$。尖角处有震荡但快速衰减、不累积，仿真稳定。这是任何需要解析导数的方法都做不到的极端测试。

## 7. 实现陷阱

1. **稳定条件 $K_d \ge K_p\Delta t$ 必须遵守**（$\alpha\ge1$）。$\alpha<1$ 不保证稳定。1D 下 $k_d=k_p\Delta t$（$\alpha=1$）收敛最快。
2. **$K_d\Delta t$ 加到的是质量矩阵，不是直接加到力上**——别写错成在右端加项。
3. **求解的是加速度 $\ddot q$，不是力**——控制与仿真在 SPD 中是交织的。若需要显式力矩输出（黑盒仿真器/真实机器人），要在求出 $\ddot q$ 后回代式 (10)，这是论文提到的"额外算力"来源。
4. **目标用的是 $\bar q_{n+1}$（下一帧目标），不是 $\bar q_n$**——跟踪 mocap 时要取下一帧的参考姿态。
5. **旋转表示**：广义坐标含旋转关节时，$q+\dot q\Delta t$ 对四元数/旋转群不是简单加法，需在切空间正确处理（论文用广义坐标回避了显式线性化，但实现时仍要注意旋转的指数映射）。
6. **关节限位用超大增益**：论文用 $k_p=10^9$ 强制肘关节限位——这正是 SPD 允许的，传统 PD 做不到。

## 8. 超参数敏感性

- **全局增益**：SPD 最大卖点之一是 42-DOF 角色只需调一个全局 $k_p$。传统 PD 因稳定性差必须每个 DOF 单独调（根关节要松、手要紧，否则根跟不上或手发散）。
- **$k_d/(k_p\Delta t)$ 比值**：论文表 3 中此比值在 1~96 之间变化（如篮球反应用 16，足球重球用 96），用来调节反应的"软硬"。
- **时间变化增益（关键帧插值）**：用高阶多项式 gain profile（如 $(1-\alpha)^{24}k_t$）让关键帧处增益陡增、中间帧增益低 → 既满足关键帧又保持柔顺次级运动。线性 blend 则产生僵硬运动。

## 9. 复现风险

- 没有公开官方代码（论文无 GitHub/code 链接），需自行实现。
- 多维稳定性只有"近似对角质量矩阵"的启发式论证，无严格证明；小增益区域理论上未证稳定（但经验上从未失稳）。
- 泰勒一阶截断的精度依赖步长——步长过大时预测位置 $q+\dot q\Delta t$ 误差变大，虽稳定但跟踪精度下降。
- 单边约束（脚-地接触）会产生拉力，直接套用会出错，需额外处理。
