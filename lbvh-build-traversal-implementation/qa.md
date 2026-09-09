# 分层问答

## 基础

### 1. LBVH 中的 L 指什么？

<details><summary>答案</summary>

Linear。primitive 先按 Morton code 排成一维顺序，再由该顺序隐式构建 radix tree。

</details>

---

### 2. 为什么每轴只有 10 bit？

<details><summary>答案</summary>

三个轴交错后恰好形成 30-bit key，能装入 32-bit 无符号整数，并用紧凑的位扩展技巧计算。

</details>

---

### 3. AABB 边界刚好接触是否算相交？

<details><summary>答案</summary>

算。实现只有在某轴严格分离时才返回 false。

</details>

---

### 4. `objIDs[i]` 表示什么？

<details><summary>答案</summary>

Morton 排序后第 `i` 个 leaf 对应的原始输入对象 ID。

</details>

---

### 5. 为什么二叉树有 `N-1` 个内部节点和 `2N-1` 个总节点？

<details><summary>答案</summary>

每个内部节点恰有两个孩子的满二叉树满足内部节点数等于叶数减一。

</details>

---

## 中级

### 6. Morton code 相同如何避免 radix key 冲突？

<details><summary>答案</summary>

把 Morton code 放在 64-bit 高 32 位，把排序位置放在低 32 位。公共前缀先比较空间 code，完全相同后再由位置区分。

</details>

---

### 7. `determineRange` 为什么先指数搜索再二分？

<details><summary>答案</summary>

事先不知道节点覆盖多远。指数搜索用对数次探测找到上界，再在上界内二分精确边界。

</details>

---

### 8. refit 为什么只让第二个到父节点的线程继续？

<details><summary>答案</summary>

父 AABB 必须等两个孩子都写完。第一个线程留下到达状态并退出，第二个线程合并 sibling 数据并承担向祖先传播的工作，避免每层单独 launch kernel。

</details>

---

### 9. pre-order 中为什么左孩子一定是 `parent+1`？

<details><summary>答案</summary>

先访问父，再递归完整左子树，然后右子树；因此父后第一个元素必是左孩子。

</details>

---

### 10. escape 为什么由第一个左祖先决定？

<details><summary>答案</summary>

从当前节点向上找，第一个“路径来自左子树”的祖先仍有未访问的右子树；那棵右子树是 DFS 的下一个位置。若全来自右子树，当前节点在最右路径上，遍历结束。

</details>

---

## 高级

### 11. `fence` 如何消除经典自查询重复？

<details><summary>答案</summary>

每棵子树覆盖连续 Morton 叶区间。若该区间最大 leaf position 不大于当前 query position，则所有候选都已经或将由较小位置方向处理，可以整棵跳过。

</details>

---

### 12. 为什么 `StacklessLite` 的自查询仍产生对称重复？

<details><summary>答案</summary>

它只过滤 leaf primitive ID 等于 query primitive ID，没有用 Morton 位置或 `fence` 规定唯一方向，所以 A 查询 B 和 B 查询 A 都会报告。

</details>

---

### 13. two-pass 与 shared-buffer single-pass 的主要权衡是什么？

<details><summary>答案</summary>

two-pass 精确分配且 fill 无全局 append 争用，但需遍历两次、scan 和一次分配同步；single-pass 只遍历一次，但容量由调用者猜测，仍需块级/global atomic flush。

</details>

---

### 14. 为什么无栈不一定更快？

<details><summary>答案</summary>

它减少每线程 local/register 栈，却增加节点大小、线性化构建和 escape 跳转；warp 线程仍可能沿不同 escape 路径分歧，DFS 顺序也未必提供最佳跨线程访存合并。

</details>

---

### 15. 仅 refit 何时需要退回 rebuild？

<details><summary>答案</summary>

当 primitive 已大幅跨越原 Morton 分区，内部 AABB 会变得宽且重叠，查询访问节点显著增加。应以查询节点访问数或时间退化作为 rebuild 触发条件。

</details>

