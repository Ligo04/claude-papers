# 心智模型

把实现分成三个互不混淆的坐标系：

1. **原始对象 ID**：调用者输入 AABB 的下标。
2. **Morton 叶位置**：排序后叶数组的下标，决定树结构、范围和 `fence`。
3. **pre-order 节点位置**：无栈数组的下标，决定 `idx+1` 与 `escape`。

`objIDs[mortonLeaf] -> originalID` 把第二个坐标系映射回第一个；`internalPreorder` 和 `leafPreorder` 把原始 radix-tree 节点映射到第三个。大部分阅读错误都来自把这三个索引混成一个。

其次，把 `escape` 看成编译后的控制流：

- 命中内部节点：顺序执行下一条指令，即 `idx + 1`。
- miss：跳到当前子树之后的第一条指令，即 `escape[idx]`。
- 处理完叶：同样跳到 `escape[idx]`。
- `-1`：程序结束。

因此 stackless BVH 不是“没有遍历状态”，而是把栈隐式编码进节点顺序和跳转地址。

