# 心智模型：一条 contour 如何变成 solver 约束

## 这是什么类型的问题？

这是**带离散证据筛选的非光滑方向优化**：输入是已有 EF 穿透，输出是能让物理求解器逐步分离的 VF/EE penalty pairs。它不是单纯碰撞检测，也不是一次性的 mesh projection。

## 一张纸上的五个对象

1. **Contour seed**：把相邻 EF pairs 连成的穿透组件，提供两侧 anchors。
2. **Direction \(r\)**：单位球上的候选分离方向，决定每条线性 ray 的 travel distance。
3. **Hit support \(S(r)\)**：VF/EE/FV 命中经过双侧 adjacency 和 anchor culling 后留下的 response band。
4. **Depth/cost**：同一 contour 共享 \(D_\mathcal C=\max\delta_h\)，以面积/质量加权形成 \(J(r)\)。
5. **Penalty pair**：固定 barycentric weights 和 characteristic normal 后，成为 Newton 系统的 affine constraint。

```text
EF pairs
  └─ adjacency flood-fill → contour C + roles A0/A1
       └─ candidate directions r
            └─ VF / EE / reverse-FV ray evidence
                 └─ dual-sided hit graph + contour anchors
                      └─ valid support S(r), depth D_C, objective J
                           └─ choose/refine r*
                                └─ VF/EE penalty templates
                                     └─ implicit Newton + PCG
```

## 它在研究地图上的位置

- **碰撞检测**回答“哪里相交”；本文接在 EF narrow phase 之后。
- **解缠响应**回答“沿什么几何方向把当前相交带分开”；本文的核心。
- **物理时间步**回答“在惯性、弹性、弯曲和 contact penalty 共同作用下，下一状态是什么”；论文把 response pair 放回这一层。
- **后处理/形变修复**可能直接改顶点，但不能表达本文的 solver-compatible 能量语义。

## 需要的先验

线性代数（3×3 solve、Cramer）、三角网格 primitive adjacency、BVH ray traversal、连通分量/CSR、隐式积分的 Newton/PCG、基本的 penalty contact。读者不需要先掌握所有 GIA contour taxonomy，但应知道“EF contour 是一组相邻 edge–face intersections”。

## 一个最小例子

把一条穿过矩形的线看成 source，把矩形边界当 destination。不同方向会有不同最大 ray depth；若 support 权重相同，选择 `max depth` 最小的方向。代码 demo 保留了这个排序直觉，却故意不实现三角形 barycentric、VF/EE/FV 和 topological culling，所以它只能帮助建立方向成本直觉。
