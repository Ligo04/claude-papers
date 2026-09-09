# 自测题（5 基础 + 5 中级 + 5 高级）

## 基础题

### 1. MeshFEM 主要解决什么问题？

<details>
<summary>答案</summary>

它面向非线性有限元和几何处理中的 Newton 优化，针对局部元素能量产生的块稀疏 Hessian，加速模式构建、装配、分解和线性求解的全链路。

</details>

---

### 2. 为什么 Hessian 不是普通的标量稀疏矩阵？

<details>
<summary>答案</summary>

每个节点有 `d` 个空间坐标，同一元素内的坐标分量通常相互耦合，所以每个非零节点对对应一个稠密 `d×d` 块。

</details>

---

### 3. BlockCatamari 采用哪类直接分解？

<details>
<summary>答案</summary>

官方仓库将它描述为实数 supernodal sparse Cholesky，计算 `L Lᵀ`；它不是任意对称不定系统的通用 LDLᵀ 求解器。

</details>

---

### 4. 论文摘要提到哪些应用？

<details>
<summary>答案</summary>

注入曲面参数化，以及带接触的弹性仿真。

</details>

---

### 5. 本学习包为什么没有论文原图？

<details>
<summary>答案</summary>

ACM 页面和 PDF 在当前环境返回 HTTP 403，所以没有把不可访问内容伪装成已解析全文；`images/README.md` 记录了这个缺口。

</details>

---

## 中级题

### 6. 为什么只优化 Hessian-vector product 不能覆盖本文的目标？

<details>
<summary>答案</summary>

直接 Newton 仍需建立稀疏模式、装配完整系统、做 symbolic/numeric factorization 和 solve；HVP 只覆盖其中一部分线性代数路径。

</details>

---

### 7. symbolic factorization 和 numeric factorization 有什么区别？

<details>
<summary>答案</summary>

symbolic 阶段根据稀疏模式和 ordering 规划消去结构，不依赖当前数值；numeric 阶段把当前 Hessian 数值填入并计算因子。模式不变时通常可以复用前者。

</details>

---

### 8. 线程局部装配为什么有帮助？

<details>
<summary>答案</summary>

不同元素可能同时写同一全局块。让每个线程先写自己的局部矩阵可以减少共享锁竞争，最后再合并；代价是额外内存和合并带宽。

</details>

---

### 9. 接触问题为什么需要稀疏模式缓存策略？

<details>
<summary>答案</summary>

接触活动集会偶尔改变模式。完全重做 symbolic 很贵，而永久缓存所有历史模式会耗尽内存；LRU 以预算和过期年龄折中。

</details>

---

### 10. 新元素能量接入时最先应该明确什么？

<details>
<summary>答案</summary>

元素访问的局部节点 stencil、局部梯度/Hessian 的块布局、是否总能插入对角块，以及模式在迭代间的稳定程度。

</details>

---

## 高级题

### 11. 块存储为什么可能同时减少索引开销并提高 kernel 效率？

<details>
<summary>答案</summary>

索引是在节点块层而非每个标量条目层操作；确定块位置后，一次处理连续的 `d×d` 数据。这样既减少搜索/调度次数，也为小矩阵乘加、三角 solve 提供规则数据布局。

</details>

---

### 12. 如果 Hessian 长时间不定，BlockCatamari 路径会遇到什么问题？

<details>
<summary>答案</summary>

实数 Cholesky 需要适合的正定系统。必须在外层做 Hessian projection、shift、信赖域或负曲率策略；否则 factorization 可能失败。修正本身也可能成为主要成本。

</details>

---

### 13. 为什么“缓存命中率”不是唯一评价指标？

<details>
<summary>答案</summary>

缓存命中能减少 symbolic 重建，但缓存占用、模式合并、动态条目查找也有成本。应把 pattern、symbolic、assembly、numeric、solve 的时间和内存一起记录。

</details>

---

### 14. 什么时候标量通用求解器可能胜过块专用求解器？

<details>
<summary>答案</summary>

问题很小、块大小不适合现有 kernel、模式高度动态、线程调度/转换成本过高，或系统需要通用 indefinite/复数支持时，成熟通用求解器可能更稳或更快。

</details>

---

### 15. 如何设计一个公平的 MeshFEM 复现实验？

<details>
<summary>答案</summary>

固定网格、元素能量、ordering、线程数、BLAS/MKL 和停止条件；分别计时模式构建、装配、symbolic、numeric、solve 及 Newton 总时间，并报告内存、收敛/失败次数和接触活动集变化，而不是只报一次 factorization 的速度。

</details>

---
