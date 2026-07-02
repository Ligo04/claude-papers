# 代码演示说明

三个自包含、可独立运行的 Python 演示，把 IsaacIPC / GMCP 的核心思想变成可看的数字和曲线。

**依赖**：仅需 `numpy`。运行：`python3 <文件名>`

---

## 1. `dual_mesh_mapper_demo.py` —— 双网格映射器（论文 §3.2）

复现把视觉顶点用「重心坐标 + 法向偏移」绑定到仿真三角形，并每帧重建。

- **演示 1**：预计算正确性 + **刚性不变性**（刚体变换下重建误差 = 0）。
- **演示 2**：误差随仿真网格细化的收敛——h 每减半，误差比 ≈ **4.16**，干净地验证论文所说的 **O(h²)**（渲染保真度随网格细化二次提升）。

核心函数：`embed_vertex`（公式 1-3，解 2×2 Gram 系统）、`reconstruct_vertex`（公式 4）。

## 2. `gmcp_barrier_demo.py` —— GMCP 数值要素（论文 §4.2-§4.5）

逐一拆解 GMCP barrier 势 `Ψ = Σ κ_q·w_k·γ_k·B(g_k,ε_k)`（公式 10）的五个要素：

1. **IPC barrier B(g,ε)**：g→0 时能量/力→∞，实现 feasible 无穿透。
2. **C¹ Hermite 阶跃 H(x,δ)**（公式 13）：master 特征权重 γ 的平滑过渡，端点导数 = 0（验证 C¹）。
3. **自适应支撑半径 ε=min(0.9·g_ref, ε_max)**（公式 17）：保证 g_ref>ε，静止态无伪力。
4. **线性化间隙步长 α=min_k 0.9·g_k/(−∇g_k·Δx)**（公式 18-19）：牛顿步无穿透。
5. **mortar 积分权重 w_k**：在非均匀压力 + 非均匀采样下，等权法误差 7.09%，mortar 权重误差 0%——展示 w_k 的价值。

## 3. `mortar_face_sampling_demo.py` —— GMCP 面样本流水线（论文 §4.1）

复现面样本构造：投影 → **Sutherland-Hodgman 裁剪** → 扇形三角化 → **2D Gauss 求积** → 有符号间隙。

- 验证 **Σw_k = 重叠多边形面积**（相对误差 0），证明求积权重正是 mortar 积分权重的几何来源。
- 演示有符号间隙 `g_k = n_s·(x_m,k − x_s,k)`（公式 6）。

核心函数：`sutherland_hodgman`、`triangle_gauss_3pt`、`fan_triangulate_and_quadrature`。

---

> 这些演示是**教学性简化**，不是 libuipc 的真实实现。它们隔离单个概念以便理解；真实 GMCP 在 GPU 上、3D 全耦合、配合牛顿求解器运行。libuipc 源码见 https://github.com/spiriMirror/libuipc 。
