# 摘要

## 背景与问题

宽相碰撞检测需要从大量 AABB 中快速找出潜在相交对。普通 BVH 适合剪枝，但串行构建和每线程显式遍历栈不适合 GPU。LBVH 用 Morton code 把三维空间近似映射到一维顺序，使树拓扑可以高度并行地构建。

## 文档覆盖的实现

1. `Bound<float>` 的 AABB 相交、合并与场景归一化。
2. 每轴 10 bit、总计 30 bit 的 Morton 编码。
3. 按 Morton code 排序后，用 Karras 2012 的最长公共前缀规则确定每个内部节点的范围和切分点。
4. 每叶一个线程、用原子会合完成内部 AABB refit。
5. 两遍 count/scan/fill 查询方案，以及实际代码采用的预分配单遍 shared-memory 缓冲方案。
6. 计算子树大小、pre-order 位置、escape 目标，写成 32-byte `StacklessNode`。
7. 查询命中内部节点时 `idx += 1`，命中叶或 miss 时 `idx = escape`。

## 结果与证据边界

文档没有给出定量性能表。仓库 README 声明 `stacklessbvh_lite` 可通过仿真数据集 ground truth；源码中的 ground-truth 比较会对自查询结果排序并放入集合，所以它能证明碰撞集合一致，但不能证明输出没有重复方向。当前学习过程没有运行完整数据集基准。

