# 心智模型：一张“块结构 → 全链路收益”的地图

## 这是什么类型的问题？

这是“非线性优化外层 + 稀疏直接线性代数内核”的系统设计问题。有限元元素提供局部性，几何自由度提供小块，Newton 法把这些局部贡献反复送进线性求解器。

## 需要的先验知识

- 有限元：元素、局部梯度/Hessian、装配；
- 数值优化：Newton、线搜索、非凸 Hessian；
- 稀疏矩阵：CSC、消去树、ordering、symbolic/numeric factorization；
- 并行程序：thread-local accumulation、锁、内存带宽。

## 研究地图

```text
元素能量
   └─局部 stencil
       └─块稀疏 Hessian
           ├─装配：排序/合并/线程局部
           ├─模式：ordering + 缓存
           └─数值：块 supernodal Cholesky
                └─Newton / 接触 / 几何处理应用
```

## 如何分类这篇工作

不要把它只归类为“一个更快的 Cholesky”。更准确的标签是：

1. **表示层**：显式保留有限元的块稀疏结构；
2. **编译/分析层**：减少模式构建和重复 symbolic 工作；
3. **执行层**：并行装配、块 kernel、supernodal factorization；
4. **优化层**：让非凸 Newton 外层能稳定消费这个求解器。

它的主要洞见是跨层协同，而不是某个孤立公式。
