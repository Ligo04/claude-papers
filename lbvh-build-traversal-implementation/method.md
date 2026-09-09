# 实现逐段拆解

## 1. 核心数据结构

`Bound<float>` 保存 6 个 float，即 `min/max` 三轴。相交判断等价于“任一轴严格分离则不相交”的否定，因此边界相等算碰撞。

经典内部节点 `LBVHNode<float>` 保存两个 child bound、左右 child reference、parent reference 和 `fence`。child reference 的最高位表示叶；parent reference 的最高位表示当前节点是父节点的右孩子。两种 bit 语义依赖字段上下文。

无栈节点为 32 bytes：24-byte AABB、4-byte `data`、4-byte `escape`。`data == -1` 表示内部节点，否则是原始 primitive ID。

## 2. Morton 编码

先对所有输入 AABB 做 reduce 得到场景 AABB。每个线程取 primitive AABB 中心，在场景包围盒中归一化到 `[0,1]`，乘 1024、截断并 clamp 到 `[0,1023]`。

`mortonExpandBits` 通过乘法和掩码把一个 10-bit 整数扩成每个有效 bit 之间隔两个 0 的 30-bit 模式。三个轴分别左移 2、1、0 位后交错，得到 30-bit Morton code。

随后 `stable_sort_by_key(morton, objIDs)`：Morton code 和 `objIDs` 一起重排。排序后 leaf index 是 Morton 顺序，`objIDs[leaf]` 才是原始输入 ID。

## 3. Karras 内部节点构建

对 `N-1` 个内部节点各启动一个线程。`mergeIdx(code, idx)` 形成唯一 64-bit key，`commonUpperBits(a,b)=clz(a xor b)` 得到 radix 前缀长度。

### 3.1 DetermineRange

根节点 `idx=0` 特判覆盖 `[0,N-1]`。其余节点比较与左右相邻 key 的公共前缀，选择前缀更长的方向 `d`。用 2、4、8... 指数搜索得到范围长度上界，再二分出仍满足前缀大于 `minDelta` 的最远端点。

### 3.2 FindSplit

计算区间首尾 key 的公共前缀 `deltaNode`，在区间内二分寻找最后一个与首 key 有更长前缀的位置 `gamma`。左子树覆盖 `[first,gamma]`，右子树覆盖 `[gamma+1,last]`。

若某一侧区间只有一个 leaf，就把 leaf index 的 MSB 置 1；否则引用对应内部节点。与此同时写好内部节点或 leaf 的 parent 信息。

`fence` 保存当前内部节点覆盖区间的另一个端点。因为区间连续，它支持自查询按 Morton 位置整棵剪枝。

## 4. 自底向上 refit

每个 leaf 线程从 `aabbs[objIDs[leaf]]` 出发，把自己的 bound 写进父节点相应槽位。

第一次到达父节点的线程通过原子操作留下状态并退出；第二次到达者说明 sibling 已经到达，于是合并两个 child bound，再把合并结果带到祖先。根节点由最后存活的线程完成。

经典实现顺便在原子标志中传播左右子树需要的 DFS 栈深，最后从 `flags[0]` 拷回 `maxStackSize`，查询时选择相应模板实例。

## 5. 有栈单遍查询

一个 query AABB 对应一个 CUDA 线程。线程栈初始压入 root：

- 内部节点：先用两个 child bound 测试，命中的 child reference 入栈；
- leaf：写出 `(treeOriginalID, queryOriginalID)`；
- 自查询：leaf 只接受 `leafMortonIndex > queryMortonIndex`，内部节点通过 `fence` 剪掉只含较小位置的子树。

每个 block 共享 512 个 `int2` 槽位。命中先用 shared atomic 申请槽位；缓冲满时把当前 leaf 重新压栈并退出内循环。block 同步后仅线程 0 对全局 counter 做一次 `atomicAdd`，再由全 block 协作拷贝结果。这样把“每个碰撞一次全局原子”变成“每次 block flush 一次全局原子”。

调用者仍必须预分配 `resSize`。counter 可以大于容量，host 返回 `min(counter,resSize)` 并打印 overflow；它不是文档 two-pass 所说的精确分配。

## 6. 文档中的 two-pass 方案

理想流程是：

1. 每个 query 遍历一次，只写 `counts[q]`；
2. exclusive scan 得到每个 query 的不冲突写区间；
3. 按 `offsets[N-1]+counts[N-1]` 精确分配；
4. 再遍历一次填充结果。

优势是无需猜最大结果数、fill 阶段无需全局 append counter。代价是树遍历两次、scan 一次，并且动态分配往往引入同步。当前仓库没有实现这条查询路径。

## 7. 无栈布局构建

### 7.1 子树大小

leaf 大小为 1；内部节点大小为 `1 + leftSize + rightSize`。仍采用“第二个孩子继续向上”的原子会合模式，结果写在内部节点索引位置。

### 7.2 pre-order 位置

每个节点独立向根回溯：

- 当前是左孩子：位置增加 1，只跨过父节点；
- 当前是右孩子：位置增加 `1 + size(leftSibling)`，跨过父节点及完整左子树。

累加到根后得到该节点在 pre-order 数组中的唯一位置。

### 7.3 escape 祖先

向上回溯时遇到的第一个“当前路径是左孩子”的祖先，其右孩子就是当前子树处理结束后的下一棵子树。代码先暂存这个祖先，等所有 pre-order 位置算完，再把祖先的右孩子解析为实际 pre-order index。若沿途始终是右孩子，则 escape 为 `-1`。

### 7.4 写线性节点

内部节点 bound 是两个 child bound 的并，`data=-1`；leaf bound 来自原始 AABB，`data=objIDs[leaf]`。两类节点分别并行写入同一个 `2N-1` 数组。

## 8. 无栈查询

状态只剩一个 `nodeIdx`：

```text
hit internal -> nodeIdx + 1
hit leaf     -> report, then escape
miss         -> escape
escape == -1 -> done
```

实际 kernel 每个线程最多连续访问 100 个节点便回到 block 同步点，shared buffer 按与有栈版本类似的方式 flush。`sharedActiveCount` 让已经遍历完成的线程保持参与同步，直到整个 block 的查询都结束。

## 9. refit 与重建

`LBVHStacklessLite::refit()` 只重新合并内部 AABB，再按已有 pre-order/escape 映射重写线性节点；拓扑、子树大小和 escape 不变。它保证包围关系仍正确，但对象跨越原 Morton 分区后树质量可能变差。当前 `rootBounds` 成员没有在 refit 中重算，因此 `bounds()` 返回值可能陈旧。

## 10. 复杂度与资源

- Morton/ID/parent/flag 等数组：`O(N)`。
- 排序：实现由 Thrust 完成，通常是 GPU radix 路径；抽象复杂度可按 `O(N log N)` 比较排序上界理解。
- radix tree：`N-1` 线程，每线程做指数搜索和二分。
- refit/size：总节点数据为 `O(N)`，关键路径 `O(depth)`。
- pre-order 位置：每节点走到根，平衡树约 `O(N log N)` 工作，退化树最坏 `O(N^2)`。
- 查询：平均取决于剪枝质量；最坏每 query 访问整棵树。
- 有栈遍历状态：每线程 `O(depth)`；无栈遍历状态：每线程 `O(1)`，代价转移到每节点 escape 和预处理数组。
