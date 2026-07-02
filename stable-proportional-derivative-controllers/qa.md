# 自测题（15 题：基础 5 / 进阶 5 / 高级 5）

---

## 基础题（Basic）

### Q1. 传统 PD 控制器的核心矛盾是什么？

<details>
<summary>答案</summary>

跟踪精度与仿真效率之间的**不该有的耦合**。要精确跟踪就要大比例增益 $k_p$，但大 $k_p$ 会让控制力随仿真推进数值发散，为保稳定只能大幅缩小时间步长 $\Delta t$（论文对比工作用到 0.67 ms），导致仿真效率崩塌。动画师被迫在"准"和"快"之间二选一。

</details>

---

### Q2. SPD 与传统 PD 在公式上的唯一关键区别是什么？

<details>
<summary>答案</summary>

控制力用**下一时刻状态** $q_{n+1}, \dot q_{n+1}$ 计算，而非当前状态 $q_n, \dot q_n$：

$$\tau_n = -k_p(q_{n+1} - \bar q_{n+1}) - k_d\dot q_{n+1}$$

由于下一时刻状态未知，用一阶泰勒展开 $q_{n+1}=q_n+\Delta t\dot q_n$、$\dot q_{n+1}=\dot q_n+\Delta t\ddot q_n$ 把它表示出来并求解。

</details>

---

### Q3. SPD 的稳定条件是什么？

<details>
<summary>答案</summary>

$K_d \ge K_p\Delta t$（写成 $k_d=\alpha k_p\Delta t$ 时要求 $\alpha\ge1$）。当 $\alpha=1$（即 $k_d=k_p\Delta t$）时在 1D 情形收敛最快（两步精确到位）。$\alpha<1$ 时不保证稳定。

</details>

---

### Q4. 论文展示了 SPD 的哪三个应用？

<details>
<summary>答案</summary>

1. **运动跟踪 + 仿真**：跟踪 mocap 数据并对环境扰动（踢球/打排球/接篮球）产生物理响应。
2. **关键帧插值**：用时间变化增益的两个 SPD 跟踪相邻关键帧，生成自然的次级运动（toy mouse 尾巴）。
3. **约束动力学**：替代罚函数法，用稳定 PD 跟踪约束（3D Tinkertoy 螺旋）。

</details>

---

### Q5. SPD 与全隐式积分（如 Baraff-Witkin）的灵感关系和关键区别是什么？

<details>
<summary>答案</summary>

**灵感来源**：Baraff & Witkin 1998《Large Steps in Cloth Simulation》用全隐式积分稳定布料仿真。

**关键区别**：Baraff-Witkin 处理的是**被动布料**，对整个系统全隐式；角色有**主动控制力**，所以 SPD **只对控制力**用下一帧状态，积分本身仍用前向欧拉。因此 SPD 不需要线性化整个动力学系统，比全隐式更高效。

</details>

---

## 进阶题（Intermediate）

### Q6. 在 1D 玩具例子中，为什么 $k_p\to\infty$ 时加速度仍然有限？

<details>
<summary>答案</summary>

加速度公式 $\ddot q_0 = -\dfrac{k_p(p+\Delta t\dot p-\bar q)+k_d\dot p}{m+k_d\Delta t}$。当 $k_d=k_p\Delta t$ 且 $k_p\to\infty$ 时，分子分母都含 $k_p$，相除约掉，得有限值 $\ddot q_0 = -\dfrac{p+2\Delta t\dot p-\bar q}{\Delta t^2}$。

直觉：分母里的 $k_d\Delta t = k_p\Delta t^2$ 把无穷大的分子"中和"了。这正是稳定的根源——力是按"刚好让下一帧到位"设计的，不会过冲。

</details>

---

### Q7. 在多体系统中，SPD 的"魔法"体现在哪个矩阵运算上？为什么它带来稳定性？

<details>
<summary>答案</summary>

体现在 $\ddot q_n = (M + K_d\Delta t)^{-1}(\ldots)$ 里的 **$(M + K_d\Delta t)^{-1}$**。

阻尼项 $K_d\Delta t$ 被加进了质量矩阵。这相当于做了**隐式阻尼**：当 $K_p$（从而 $K_d\ge K_p\Delta t$）很大时，$K_d$ 主导 $M$，使 $M+K_d\Delta t$ 近似为对角矩阵，把多维问题退化为各自稳定的 1D 问题。隐式阻尼是无条件稳定的来源，因此大增益不再发散。

</details>

---

### Q8. 为什么论文不直接用全隐式积分，而要发明 SPD？列出至少三个理由。

<details>
<summary>答案</summary>

1. **避免线性化**：全隐式要线性化非线性的 $M(q)$ 和 $C(q,\dot q)$，需计算更高阶导数并处理旋转表示的奇异性，推导/实现都很复杂。SPD 完全不需要。
2. **计算更高效**：不必线性化整个系统，只需对控制力做半隐式。
3. **不过阻尼**：全隐式积分常过度阻尼运动，毁掉接篮球这类时序关键动作；SPD 不过阻尼，保留运动的"脆"感。
4. （代价）SPD 只有条件稳定 $K_d\ge K_p\Delta t$，全隐式无条件稳定——这是 SPD 换来高效/不过阻尼付出的代价。

</details>

---

### Q9. 实验中 PD 在哪个增益水平开始失效，SPD + 参考速度表现如何？

<details>
<summary>答案</summary>

双摆实验（$\Delta t=1/60$ s）中，PD 在 $k_p\ge3\times10^3$ 时仿真**直接发散**（误差表中标 NA），因为不稳定控制力让仿真崩溃。

SPD 始终稳定。**SPD + 跟踪参考速度**时，跟踪误差随增益增大**近似线性下降**（如 DOF1 在 $k_p=3\times10^4$ 时 $\|e\|_2$ 降到 0.0016）。这一线性下降在 SPD 跟踪零速度时不出现，作者推测是因为消除了仿真运动与参考运动间的跟踪延迟。

</details>

---

### Q10. SPD 用于关键帧插值时，时间变化增益（gain profile）起什么作用？线性 blend 与高阶多项式有何区别？

<details>
<summary>答案</summary>

用两个 SPD 分别跟踪相邻的两个关键帧，gain profile 决定运动的时序特性。

- **线性 blend** 两个增益 → 产生**僵硬**的运动。
- **高阶多项式**（如 $k_{p_i}=(1-\alpha)^{24}k_t$、$k_{p_{i+1}}=(\alpha^{24}+0.005)k_t$，$\alpha$ 为到第 $i$ 帧的归一化距离）→ 关键帧处增益陡增、中间帧增益低。这种鲜明对比既能**精确满足关键帧**，又让中间运动**保持柔顺**、产生自然次级运动。

此法适合有突然加速度变化的运动，对本身平滑的运动效果不明显。

</details>

---

## 高级题（Advanced）

### Q11. 推导从 SPD 公式 (3) 到多体加速度公式 (11) 的关键步骤。

<details>
<summary>答案</summary>

从 SPD 内部力 (10)：
$$\tau_{int} = -K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d(\dot q_n + \ddot q_n\Delta t)$$

代入运动方程 (9) $M\ddot q_n + C = \tau_{int} + \tau_{ext}$：
$$M\ddot q_n + C = -K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d\dot q_n - K_d\ddot q_n\Delta t + \tau_{ext}$$

把含 $\ddot q_n$ 的项（$M\ddot q_n$ 和 $K_d\Delta t\ddot q_n$）移到左边：
$$(M + K_d\Delta t)\ddot q_n = -C - K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d\dot q_n + \tau_{ext}$$

两边左乘 $(M+K_d\Delta t)^{-1}$ 即得 (11)。关键是认出 $\tau_{int}$ 里的 $-K_d\ddot q_n\Delta t$ 含未知加速度，移项后变成质量矩阵增量。

</details>

---

### Q12. 在约束扩展中，为什么阻尼项要用投影矩阵 $dd^T$ 而不是直接用 $\dot q_{n+1}$？

<details>
<summary>答案</summary>

因为约束只对**沿违反方向 $d$ 的速度分量**起作用，垂直于 $d$ 的速度（如粒子沿曲线滑动的速度）是合法的、不应被阻尼掉。

$dd^T$ 是到方向 $d$ 上的**投影算子**：$dd^T\dot q$ 提取速度在 $d$ 方向的分量。所以 $-k_d dd^T\dot q_{n+1}$ 只阻尼违反约束的速度分量。如果直接用 $-k_d\dot q_{n+1}$ 会错误地阻尼掉粒子沿曲线的正常运动（如下滑速度），破坏物理正确性。这也是式 (12) 与式 (2) 的关键差异。

</details>

---

### Q13. 论文对多维稳定性只给了"近似对角"的启发式论证，而非严格证明。这个论证的逻辑链是什么？它在什么情形下可能不成立？

<details>
<summary>答案</summary>

**逻辑链**：
1. 1D 情形严格证明：$k_d\ge k_p\Delta t$ 时 $q$ 不过冲、稳定。
2. 高刚度系统中 $K_p$ 很大 → 由下界 $K_d\ge K_p\Delta t$，$K_d$ 也很大。
3. SPD 把 $K_d\Delta t$ 加到 $M$ 上；当 $K_d$ 主导 $M$ 时，$M+K_d\Delta t$ 近似为**对角矩阵**（$K_d$ 本身是对角的）。
4. 对角化 → 各 DOF 解耦 → 1D 稳定性结论逐维成立。

**可能不成立的情形**：当 $K_p$（从而 $K_d$）**较小**时，$M$（含非对角耦合项）不再被 $K_d\Delta t$ 淹没，$M+K_d\Delta t$ 远非对角，DOF 间强耦合，1D 分析失效。论文承认此时严格分析困难，只能说"小增益经验上不引起不稳定"。质量矩阵病态（如长链轻连杆的老鼠尾巴）也会加剧问题。

</details>

---

### Q14. SPD 为什么不适合脚-地接触这类单边约束？这指向什么未来方向？

<details>
<summary>答案</summary>

单边约束（unilateral）只允许"推"不允许"拉"——地面只能支撑脚向上，不能把脚吸住。但 SPD 本质是弹簧，会产生**双向力**：当预测脚穿透地面时它推回（正确），但当脚抬离地面时弹簧会产生**朝向地面的拉力**（错误，地面不会拉脚）。

**未来方向**：把 SPD 适配到单边约束。论文指出这能极大提升仿真性能——因为处理碰撞通常需要缩小时间步长，而稳定的 SPD 约束可以避免这种缩步，从而在接触处也保持大步长。

</details>

---

### Q15. 为什么说 SPD 让"自动调参（gain tuning via optimization）"成为可能？联系它对下游 RL 工作（如 DeepMimic）的意义。

<details>
<summary>答案</summary>

传统 PD 的稳定性对增益组合**极其敏感**，搜索空间"参差不齐（jagged）"——某些增益组合直接让仿真爆炸，使优化/搜索算法难以收敛。

SPD 在高增益 + 大步长下都稳定 → 搜索空间平滑得多 → 优化更易收敛到最优解。这是论文提出的未来方向。

**对下游 RL 的意义**：DeepMimic 等仿人控制把"策略网络输出目标姿态 → SPD 转成关节力矩"作为执行管线。SPD 的两个性质恰好契合 RL：① **大步长**提升训练采样效率；② **高增益稳定跟踪**让策略能精确执行目标姿态而不必担心仿真发散。可以说 SPD 是连接"学到的策略"和"稳定物理执行"的关键桥梁，使物理 RL 仿人控制在工程上可行。

</details>

---

> 做完这 15 题，你应该能：① 一句话讲清 SPD 的核心改动；② 手推 1D 和多体的加速度公式；③ 解释稳定性的数学根源；④ 说清它与隐式积分的取舍；⑤ 理解它为何成为现代物理 RL 控制的标准组件。
