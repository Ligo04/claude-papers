# 关键洞察与代码差异

## 为什么 Morton code 足以构建一棵树

Morton 排序不是在优化 SAH；它只提供一个保留局部性的线性顺序。Karras 构建把相邻 key 的最长公共前缀当作隐式 radix tree 的层级：公共前缀越长，空间上通常越接近，越应该在更深层合并。

代码把 `(mortonCode, sortedIndex)` 合成 64-bit key。后 32 bit 的排序位置只用于打破 Morton code 相同的情况，使每个 key 唯一；它不是对象 ID。稳定排序保证相同 Morton code 的原始相对次序可复现。

## `fence` 的真正含义

内部节点编号与它覆盖的 Morton 连续区间有特殊关系：区间的一个端点就是内部节点编号，`fence` 保存另一个端点。因此 `max(nodeIdx, node.fence)` 是该子树最大的 Morton 叶位置。经典自查询对查询位置 `tid` 剪掉最大位置不大于 `tid` 的整棵子树，从而只保留一个方向。

## 原子“第二个孩子负责向上”

refit 和子树大小计算都采用同一模式：第一个到父节点的孩子写入数据并退出；第二个孩子拿到第一个孩子留下的标志/大小，合并两个孩子，再继续走向祖先。这让每个内部节点只由一个线程完成最终计算，不需要层级 kernel launch。

需要特别审查发布顺序。当前代码先写 child bound，再 `atomicOr`，而 `__threadfence()` 位于第二个到达者读取 sibling bound 之前。传统 CUDA 写法通常要求发布者在递增/交换完成标志前执行 fence；单靠消费者 fence 不能直观建立发布关系。

## stackless 是控制流编码，不是免费午餐

pre-order 保证左孩子紧邻父节点，所以命中内部节点只需 `idx + 1`。`escape` 指向“当前子树处理完后应该继续的节点”。代价是：

- 每个节点增加一个跳转地址；
- 构建要额外计算子树大小和 pre-order 位置；
- 当前参考实现让每个节点走到根，工作量约为 `O(N * depth)`；
- DFS pre-order 的不同 query 线程仍可能分歧，去掉本地栈不等于访存天然合并。

## 文档与当前源码的偏差

1. 文档第 3--4 节给出 two-pass count/scan/fill，但当前两个参考实现都没有对应 kernel。
2. 文档算法 10 与 `culbvh.cu` 的有栈 shared-buffer 查询高度对应；后者还用模板按实测树深选择栈大小。
3. 文档算法 17 与 `stacklessbvh_lite.cu` 对应，但代码加入每 100 次节点访问强制回到块同步点，保证活跃线程定期参与 flush。
4. `StacklessLite` 自查询仅过滤 `node.data == objIdx`，没有 `nodeIdx > tid` 之类的方向规则，因此对称碰撞对会写两次。
5. `query_compare_ground_truth` 先把结果对排序并放入集合，所以它会掩盖对称重复。
6. `subtreeSizes` 被传入 stackless query kernel，但查询阶段没有使用它。
7. 头文件把 cross-query 描述成“against another BVH”，实际 API 接受的是另一组 AABB；它只为查询 AABB 计算 Morton 顺序，没有构建第二棵 BVH。

## 适用边界

这套方法适合动态仿真宽相：拓扑重建可以并行，AABB 只变形时可以 refit。它不保证 SAH 质量；对象移动很远但只 refit 时，树仍正确但剪枝质量会持续下降。触碰边界也算相交，因为 `Bound::intersects` 使用闭区间语义。

