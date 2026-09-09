# 方法详解

## 算法总览

```
输入网格 M = (V, F)
    ↓
[初始化] 每个面 → 一个线性算子 Q + 一个最小基元
    ↓
[优先队列] 计算所有相邻面对的合并代价
    ↓
[贪心合并] 取最小代价对 → 合并 → 更新邻接代价
    ↓ (重复直到达到目标数量 N)
[后处理] 移除被其他基元完全包含的冗余基元
    ↓
输出: N 个凸基元
```

## 步骤 1：线性算子初始化

对每个面 Fᵢ，计算：
- 法线 nᵢ = normal(Fᵢ)
- 切线 tᵢ = tangent(Fᵢ)
- 线性算子 Qᵢ = area(Fᵢ) · (nᵢnᵢᵀ + ε·tᵢtᵢᵀ)

其中 nᵢnᵢᵀ 是法线外积（3×3 矩阵），ε·tᵢtᵢᵀ 是切线正则化项，防止共面退化。

### 为什么用外积？

nn^T 的物理含义：
- 它是一个投影矩阵（投影到法线方向）
- 特征值为 [1, 0, 0]（单个面时）
- 多个面合并后，特征值反映各方向的"法线累积强度"

## 步骤 2：特征分解 → 基元方向

对合并后的 Q = W·Λ·W⁻¹：
- w₂（最大特征值）→ 主法线方向（OBB 最短轴）
- w₁（中间特征值）→ 次要方向
- w₀（最小特征值）→ 圆柱/胶囊的轴向

这三个正交向量定义了基元的局部坐标系。

## 步骤 3：基元拟合

以特征向量为坐标轴，对每种基元类型计算参数：

### OBB（定向包围盒）
```
对每个轴 i ∈ {x, y, z}:
  uᵢ = max(wᵢᵀ · p), ∀p ∈ points
  lᵢ = min(wᵢᵀ · p), ∀p ∈ points
  hᵢ = max(½(uᵢ - lᵢ), 0.001)  // 半径长
  cᵢ = ½(uᵢ + lᵢ)              // 中心
体积 = 8·hx·hy·hz
```

### 球体
```
center = OBB 中心
r = max(‖c - p‖₂), ∀p ∈ points
体积 = 4/3·π·r³
```

### 圆柱体（3 个轴各算一个，取最小）
```
对轴 a:
  r = max(‖(I - aaᵀ)(p - p_cyl)‖₂)  // 到轴距离
  h = max(aᵀ(p - p_cyl)) - min(aᵀ(p - p_cyl))
体积 = π·r²·h
```

### 胶囊体
```
类似圆柱，但高度考虑球端：
  h(p) = aᵀ(p - p_cap) - sqrt(r² - r(p)²)
  height = max(h) - min(h)
体积 = π·r²·h + 4/3·π·r³
```

## 步骤 4：最优基元选择

对每对相邻基元，计算所有可能基元类型的合并代价：

C(p₀, p₁) = k(type) · V(merge(p₀, p₁)) - (V(p₀) + V(p₁))

其中 k(type) 是类型权重：
- 盒子/球/胶囊: 1.0（物理引擎原生支持）
- 圆柱: 1.05（与胶囊相比略微惩罚）
- 梯形棱柱: 1.4
- 棱台: 2.1（最高惩罚）

选择代价最低的基元类型作为合并结果。

## 步骤 5：贪心合并循环

```python
priority_queue = []
for (f0, f1) in adjacent_faces:
    P* = best_primitive(f0 + f1)
    cost = V(P*) - V(f0) - V(f1)
    if cost <= volume_threshold:
        priority_queue.push((cost, P*, f0, f1))

while not empty(priority_queue) and |P| > N:
    (cost, P*, f0, f1) = priority_queue.pop_min()
    merge(f0, f1) → P*
    update_adjacent_costs()
```

## 步骤 6：后处理

移除冗余基元：如果基元 A 的所有点都在基元 B 内部，则删除 A。

## 数据结构

- **循环链表森林**：管理面-基元的归属关系，O(1) 合并
- **并查集**：快速查询面的归属，用于优先队列中过期边的剪枝

## 伪代码（Algorithm 1 完整版）

```
Input:  Mesh = (V, F), target N, volume threshold M
Output: Primitives P, |P| ≤ N

// 预处理：去重叠顶点
1. 对每个面计算 nᵢ, tᵢ
2. Qᵢ = area(Fᵢ)(nᵢnᵢᵀ + ε·tᵢtᵢᵀ)
3. Pᵢ = Prim(Qᵢ)
4. 初始化优先队列 pq
5. for 每对相邻面 (f0, f1):
     P* = Prim(f0 + f1)
     if V(P*) - V(f0) - V(f1) > M: continue
     pq.push(cost, (P*, f0, f1))
6. while !pq.empty() and |P| > N:
     (P*, f0, f1) = pq.pop()
     更新 f0, f1 的体积和基元
     更新相邻基元的代价
7. 后处理：移除被包含的基元
8. return 唯一基元集合
```

## 复杂度分析

- 时间：O(|F| log |F|)（优先队列操作）
- 空间：O(|F|)（每面一个 3×3 矩阵 + 链表节点）
- 实际性能：百万面网格约 5 秒
