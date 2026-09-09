# LBVH 构建与遍历实现笔记

这份 18 页实现文档说明了 CUDA 上从 AABB 构建 LBVH、执行有栈查询，再把树线性化为带 escape 指针的无栈布局的全过程。难度为高级，性质属于 GPU 系统实现；建议先读 `mental-model.md`，再读 `method.md`，最后用 `insights.md` 检查文档与仓库代码的差异。

## 核心结论

- 构建主干是：场景包围盒 -> 30-bit Morton code -> 稳定排序 -> Karras radix tree -> 自底向上 refit。
- `objIDs` 是 Morton 序位置到原始对象 ID 的映射；树里的叶索引不是原始对象 ID。
- 无栈版本用 pre-order 数组和 `escape` 指针把每线程遍历状态从 `O(depth)` 降到 `O(1)`。
- 文档中的 two-pass 查询没有在当前 `culbvh.cu` 或 `stacklessbvh_lite.cu` 中落地；实际代码使用预分配结果缓冲和块级 shared-memory flush。
- `StacklessLite` 自查询输出会包含对称方向的重复对，验证代码通过集合归一化后比较，因此验证的是碰撞集合，不是原始输出唯一性。

## 建议学习时间

- 快速理解：45 分钟
- 逐 kernel 对照源码：2--3 小时

## 文件

- `summary.md`：问题、贡献和实现范围
- `method.md`：逐阶段算法与源码映射
- `insights.md`：关键设计、代价、偏差和风险
- `mental-model.md`：理解这套代码所需的概念框架
- `qa.md`：15 个分层问答
- `reflection.md`：可继续改进的方向
- `code/escape_pointer_demo.py`：可运行的 7 节点无栈遍历演示
- `index.html`：交互查看 pre-order/escape 状态转移

