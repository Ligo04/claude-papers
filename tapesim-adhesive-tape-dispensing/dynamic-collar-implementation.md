# 动态 collar：从论文规则到三角网格状态更新

依据 [TapeSim v2 III-C](https://arxiv.org/html/2609.28766v2#S3.SS3)。这里的三角网格代码是本包教学实现，输入位置由程序指定，没有隐式薄壳/IPC 运动求解。初始化曲线、接口记录与内存布局均不能当成作者源码。

## 1. 先分清四份数据

| 数据 | 含义 | 何时变化 |
| --- | --- | --- |
| `rest_positions` | 平坦、无应力的材料网格 | 正常释放不改变；切割需另行处理 |
| `cluster_local[i]` | 初始卷绕形状在 cluster 坐标中的位置 | 本文单向释放过程中保留 |
| `x[i],v[i]` | 实时世界位置与速度 | 每次运动求解更新 |
| `archived_interfaces` | 卷绕后保存的 β 或 bond reference | 用于恢复；不被活动历史覆盖 |

`cluster_local` 通常是弯曲卷绕形状，不能拿它替换平坦 rest shape。前者定义刚体如何携带材料，后者定义材料弹性能的参考形状。

## 2. 元素归属与顶点自由度必须分开

令 `C_E` 为仍归 cluster 的三角形集合。教学实现令

$$V_R=\bigcup_{e\in C_E}\mathrm{corners}(e)$$

中的顶点仍由刚体坐标驱动，其他顶点具有独立薄壳坐标。这样，某个三角形恢复为薄壳后，它的边界角点仍可能由邻接的 cluster 元素约束。

这是本包采用的一个明确的边界组织方案。论文给出元素恢复和坐标映射机制，但没有完整公开所有边界存储细节。其他等价组织方式仍需检查重复 DOF、位置连续性和质量装配。

因此应分别维护：

```text
cluster_elements        哪些元素的内部弹性/界面被省略
rigid_vertices          哪些顶点仍由共享刚体坐标驱动
free_vertices           哪些材料已离开原卷
independent_vertices    哪些顶点有独立的 3D 坐标
collar_vertices         independent_vertices - free_vertices
```

collar 内的壳元素可以跨到刚体边界上。不能要求每个 shell 元素的全部角点都已有独立自由度。

## 3. 一次完整更新

### A. 固定本步状态，完成运动求解

`C_E`、bond 集合、黏附历史在单次运动优化内固定。先解决位置，再在步边界处理离散状态切换。否则 Newton 的目标函数和未知数布局会在迭代过程中改变。

### B. 更新界面事件

求解后更新 cohesive β，或评估 bond 创建与断裂。Bond 前沿候选除了几何门槛，还需要本步 release event 和无剩余 live support。

“没有 live support”与“本步发生释放事件”不是同一个判断。前者是持续状态，后者是事件。只保留一个布尔值会丢掉两者的区别。

### C. 相对旧自由区选候选

```python
old_free = free_vertices.copy()
candidates = one_ring(old_free) & collar_vertices
```

`one_ring` 来自胶带材料三角网格的拓扑边，不是碰撞邻居或空间近邻。若把卷内层间接触边混进图里，前沿就可能跨层跳过材料顺序。

对每个候选计算

$$\tilde x_i=R_py_i+t_p,\qquad g_i=\|x_i-\tilde x_i\|.$$

`g_i` 衡量相对刚体携带形状的变形，不是点到卷芯中心的距离，也不是点到其他胶带表面的接触距离。

```python
accepted = {
    i for i in candidates
    if g[i] > delta
    and (family == "Coh" or (released_this_step[i] and live_support_count[i] == 0))
}
new_free = old_free | accepted
```

Coh 版本的几何门槛不要求 β 已降到零。这里的 `free` 表示离开原卷的材料分类，不等于“与场景中任何东西都没有黏附”。

### D. 同步提交前沿，再计算距离

所有候选都相对 `old_free` 判断，最后一次性生成 `new_free`。不要在 BFS 中边释放边扩大候选集合。

在教学 3 列网格中，正确规则一次接受下一材料行的 3 个顶点；故意写成原地递归版本，会在同一更新中接受旧 collar 的 6 个顶点。宽度方向多个顶点可以同时释放，论文没有要求全局每步只释放一个顶点。

### E. 按元素角点距离恢复 collar

用 `new_free` 作为多源 BFS 起点，在材料图上计算最短边数 `dist[i]`。对原 cluster 元素，恢复条件为

$$e\in C_E\quad\land\quad\min_{i\in\mathrm{corners}(e)}dist[i]<N.$$

注意是元素任一角点满足，且是严格小于 `<N`。`dist=N` 的元素在这次更新中仍归 cluster。这里的 `N=10` 是图边数，物理长度会随网格变化。

```python
thawed = {e for e in old_cluster_elements
          if min(dist[i] for i in corners[e]) < N}
cluster_elements = old_cluster_elements - thawed
new_independent = old_rigid_vertices - vertices(cluster_elements)
```

cluster 元素只减少，自由材料只增加。释放材料重新黏到目标物体上时，也不会重新回到原 cluster。

## 4. 质量按元素贡献分配

教学代码采用 lumped mass。每个三角形有

$$m_e=\rho t A_e^0,$$

每个角点收到 `m_e/3`。若元素归 cluster，这三份进入 cluster 所有的质量；否则进入 shell 节点的质量贡献。每个元素只走一条分支，故

$$M=\sum_{e\in C_E}m_e+\sum_{e\notin C_E}m_e$$

保持不变。切割移除材料是另一件事，不属于这里的无损重分类。

关键在边界。某个顶点可能同时接收 cluster 元素和 shell 元素的质量贡献，但仍受刚体约束。shell 来源的质量也要通过该顶点实际的 `x(q)` 映射进入系统，不能仅因名字叫 shell mass 就创建另一份独立位置。

完整质量矩阵应从相同动能出发进行装配；本教学代码没有实现全局 `JᵀMJ` 或隐式惯性矩阵，只核对元素归属、质量和重分类瞬间的动量/动能。

## 5. 恢复速度与改变 cluster 质量中心

新获得独立自由度的顶点继承

$$v_i=v_p+\omega_p\times(x_i-c_p).$$

这里 `v_p` 是旧 cluster 质量中心的速度。不能设零，也不能只复制平移速度，否则会丢掉角速度贡献。

释放一批元素后，剩余 cluster 的质量中心从 `c_old` 变成 `c_new`。若继续用“质量中心速度”存线速度，同一瞬时刚体速度场需要

$$v_{p,new}=v_{p,old}+\omega_p\times(c_{new}-c_{old}).$$

角速度保持不变。把 `c_new` 与旧的 `v_p` 直接组合，会改变所有剩余 cluster 顶点的速度。教学程序独立检查了这项参考点变换。

也可以把速度存为固定局部原点的 twist。那就应使用相应的运动学定义，而不是再套质量中心公式。两种存储选择要在代码中明确。

## 6. 恢复界面时保持两份历史

需要区分：卷绕后归档的接口状态，与正在接触求解中变化的活动状态。

- 新恢复元素关联的卷内接口，从归档恢复 β 或原 bond reference。
- 已经活动的接口继续自己的历史，不能每次前沿更新都被重新拷回归档值。
- 本步已经断开的活动 bond 不因为“恢复历史”而立刻复活。
- 同一源接触对是否被 bond 替换，应与 bond live 状态一致。

教学实现每个元素挂一个合成接口记录，用来检查“只恢复新区域、保留原参考形状、活动记录不被反复重置”。真实实现应以稳定 point–triangle pair ID 管理接口，数量与元素数不必一一对应。接触检测、支持计数和 reference gap 的生成不在本例范围内。

## 7. 可核对的教学结果

网格为 31 个材料行、每行 3 个顶点，共 93 顶点、120 个三角形。行间距 2 mm、宽 19 mm；这些网格规模是教学设置。材料厚度与密度采用论文表 I 的 0.18 mm 和 1300 kg/m³。

以 `N=10` 为例，每次同步推进一材料行，恢复 4 个三角形，3 个顶点获得独立自由度。`N=3` 仅用于对照图距离范围。程序核对了 `<N` 边界、旧自由区快照、事件/支撑门槛、归档历史及重分类瞬间的质量、线动量、角动量和动能。

```bash
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/triangular_collar.py
```

80 项检查通过。结果与全部教学网格状态在 `code/collar-mesh-validation.json`。位置输入由程序指定，守恒检查只覆盖重分类瞬间，不能据此声称薄壳积分、接触动力学或真实放带已验证。

## 8. 放入真实求解器时还缺什么

需要接上统一薄壳/刚体隐式势能、完整导数映射、真实 support/release 事件、转动轨迹 CCD、碰撞源对过滤/恢复、质量矩阵与稀疏结构更新。前沿造成 DOF 布局变化时，还要明确索引、预条件器和 GPU 数据结构的更新时机。

该方法减少的是卷内独立变形自由度。细碰撞几何和 cluster 与外部场景的接触仍要处理。应先用固定输入与严格参考解核对状态和轨迹，再讨论运行时间。
