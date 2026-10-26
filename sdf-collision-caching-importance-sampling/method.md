# 方法与实现推导

## 模块与数据

每个碰撞物体对维护一组世界坐标候选。候选至少有位置 `x`、带符号深度 `d`、法向 `n`；公开 demo 额外存激活状态、历史冲量、排序值和 Ellipsoid 状态。

```text
物体变换与 AABB
       │
       ▼
计算重叠搜索域 V
       │
       ├── 历史种子：速度预测后局部优化
       │
       └── 新种子：重叠域内的曲率特征点，必要时均匀抽样
                         │
                         ▼
          用已经找到的点构造排斥目标 g
                         │
                         ▼
        求局部极小点，查询原始 φ 与梯度
                         │
                         ▼
            距离去重，筛选激活接触
                  │              │
                  ▼              ▼
          返回接触给动力学      排序保留下帧缓存
```

图中“筛选激活接触”表示交给求解器的步骤。Algorithm 1 的候选集合可以包含未激活点，不能简单把整个缓存解释成一定接触的点。

## 搜索域与目标

第 3.1 节定义 `V=B_A∩B_B`，其中 `B` 是世界坐标 AABB。若无重叠，论文的 DCD 不继续搜索。

\[
f(x)=\max(\phi_A(x),\phi_B(x)),\qquad
s(x)=\frac{\phi_A+\phi_B+\sqrt{(\phi_A-\phi_B)^2+\epsilon}}2,
\]

\[
g(x)=s(x)-\alpha\sum_{x_i\in\mathcal M}\|x-x_i\|,
\qquad x^*\approx \operatorname*{argmin}_{x\in V}g(x).
\]

这里写 `≈` 是为了明确实际运行的是有限迭代的局部搜索。它不声称求得全局极小点。排斥集合是已经找到的候选，因此搜索按点顺序更新。

如需实现梯度，令 `r=sqrt((φ_A-φ_B)^2+ε)`，`t=(φ_A-φ_B)/r`，则在可微处

\[
\nabla s=\tfrac12(1+t)\nabla\phi_A+\tfrac12(1-t)\nabla\phi_B,
\]

\[
\nabla g=\nabla s-\alpha\sum_i\frac{x-x_i}{\|x-x_i\|}.
\]

当 `ε=0` 且两个场相等，或 `x=x_i`，需要单独定义分支；不能直接除以零。Nelder–Mead 不用这个梯度，但最终接触法向仍需要场梯度。

## 时间预测

刚体点速度为 `u(p)=v_cm+ω×(p-c)`。两者都运动时，式 6 用两者点速度的平均值；一方静止时使用运动方的完整点速度：

\[
p' =p+\Delta t\,\frac{u_A(p)+u_B(p)}2.
\]

这是种子的预测，不是把接触永久绑定到一个表面。它仍要在新的场中重新优化和重新判断激活状态。

## 曲率特征的准备

在物体局部表面生成候选，估计

\[
\Delta\phi(x)\approx\sum_{i=1}^3
\frac{[\nabla\phi(x+\delta e_i)-\nabla\phi(x-\delta e_i)]\cdot e_i}{2\delta}.
\]

保留 `|Δφ|>κ` 的点。上式显式写出三个方向的求和；论文式 7 只显示带 `i` 的项，公开源码 `Shape::laplacian` 用循环累加三个方向。运行时把点变换到世界坐标，在 `V` 内选取。公开 demo 先在局部坐标剔除再做世界坐标复查，并使用 reservoir sampling。

## 一次更新时间步的伪代码

```python
V = intersection(aabb(A), aabb(B))
if V.is_empty():
    clear_cache(A, B)
    return []

M = []
old_cache = cache(A, B)
for old_point in old_cache:
    seed = advect_with_body_velocities(old_point)
    x = local_minimize(g_using(M), seed, domain=V)
    M.append(query_contact_candidate(x))

count = n_rand + max(0, n_cache - len(old_cache))
for _ in range(count):
    seed = sample_curvature_features_in(V)
    x = local_minimize(g_using(M), seed, domain=V)
    M.append(query_contact_candidate(x))

M = remove_spatial_duplicates(M, distance=tau)
cache(A, B) = sorted(M, key=objective_against_final_M)[:n_cache]
return [c for c in M if c.phi_A < 0 and c.phi_B < 0]
```

伪代码的最后一行展开了激活接触语义。Algorithm 1 原样返回候选 `M`，激活判断在第 3.4 节。曲率点为空的均匀回退与最终集合重算排序来自公开 demo 对实现细节的解释。公开 demo 还有跳过凸物体对随机搜索等额外分支，这里没有加进论文主体算法。

## 接触数据

若两个场均为负：

\[
(d,n)=
\begin{cases}
(\phi_A,\nabla\phi_A),&\phi_A<\phi_B,\\
(\phi_B,-\nabla\phi_B),&\text{否则}.
\end{cases}
\]

这是论文式 5。精确 SDF 的梯度在可微处为单位长度；失真场与离散插值场不保证这一点。公开实现显式归一化法向，并用历史法向保持方向连续。深度、符号、法向方向和响应求解器的预期应一起核对。

## 论文参数及尺度

| 参数 | 论文值 | 作用与敏感性 |
| --- | ---: | --- |
| `n_cache` | 4 | 长期保留候选上限；增大通常增加查询和约束预算 |
| `n_rand` | 1 | 每步新探索；不含补齐空缓存的额外样本 |
| `ε` | 0.1 | 影响两个场近等值处的平滑偏差 |
| `α` | 0.01 | 排斥强度；过大会牺牲深度、偏向边界 |
| `τ` | 0.01 | 局部优化容差；也用作空间去重阈值 |
| `Δt` | 0.016 s | 主体实验时间步；不能替换成任意采样周期 |
| `κ` | 4 | 高曲率阈值，基于单位包围盒模型 |
| `δ` | 0.001 | 曲率有限差分步长 |
| 体素网格 | `100³` | 三线性插值查询 |

若几何统一放大 `s` 倍，且 SDF 数值按真实距离同比变化，要保持相似的无量纲行为，应令 `τ、δ` 乘 `s`，`ε` 乘 `s²`，`κ` 除以 `s`。固定点数时 `α` 保持无量纲。这是由公式量纲推导的缩放规律，不是论文参数 sweep 的实测结论。改变缓存点数仍会改变排斥项总强度。

式 2 的根号中是 `ε`，不是 `ε²`。因此距离平滑尺度是 `sqrt(ε)`；不能仅因为 `0.1/0.01=10` 就把 ε 解释成“十倍距离容差”。

## 实现与复现风险

1. 对局部 SDF 查询正确应用刚体逆变换，输出梯度正确旋转回世界坐标；非均匀缩放需要额外处理。
2. 用两个原始场判断接触；不要用 `g<0` 或 `s<0` 的反命题代替。
3. AABB 很薄时，轴向 simplex 在投影后可能退化。二维教学演示对边界种子增加轻微内移，属于额外数值选择。
4. 区分每个优化器的停止规则。公开 NM 实际还检查 simplex 直径或函数值跨度，并非只比较连续两个最佳点位置。
5. 去重、最终目标重算、排序和缓存点优先级会改变接触身份；逐帧检查跳变比只看点数有用。
6. 单位归一化、体素梯度插值和 curvature seeds 的准备时间都应记录。预处理成本不应混进每步检测耗时，也不应消失在总成本报告里。
7. 查询计数要说明是目标评估次数、场值调用次数还是含梯度的接口调用。公开 demo 的一个 `queries++` 对应两个物体的场查询。
8. GPU 主要按物体对并行；单个物体对内的排斥更新依赖此前候选。论文不能据此推导所有采样点都可无同步并行求解。

可运行版本见 [code/contact_manifold_demo.py](code/contact_manifold_demo.py)。它展示二维几何搜索，不计算摩擦冲量、刚体积分或论文 benchmark。
