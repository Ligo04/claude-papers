# 自测与深入问答

基础、中级、高级各 5 题。建议先尝试作答，再展开答案。

### 01 · 基础：ACD 的输入与输出是什么？为什么不只求一个凸包？

<details>
<summary>答案</summary>

输入是一般三角网格，输出是一组近似原形状的凸体。单个凸包会填平凹槽与把手，可能挡住本应存在的通道；多个凸体保留更多结构。误差更小一般需要更多部件，运行时碰撞查询也可能更贵。

</details>

---

### 02 · 基础：这篇论文是否需要训练数据或神经网络？

<details>
<summary>答案</summary>

不需要。它是几何算法与 GPU 系统工作。V-HACD、PartNet-Mobility、Objaverse 在这里是评测输入，不是训练语料。未来可结合学习式 proposal，但不是本文实测方法。

</details>

---

### 03 · 基础：这里的 warp 是什么？

<details>
<summary>答案</summary>

CUDA 的 32-lane 线程组。不同候选提供粗粒度并行，同一组 lane 协作扫描、压缩、排序和处理局部几何。它不是 NVIDIA Warp Python 框架，也不代表所有 kernel 都严格只有 32 threads/block。

</details>

---

### 04 · 基础：论文默认根部有多少候选？

<details>
<summary>答案</summary>

最大预算为 W_top + 4 N_ce = 30 + 4×16 = 94。30 是总轴向候选预算，不能再乘 3。凹边不足时有效数量可小于上限；凹边候选只在默认前 10 轮启用。

</details>

---

### 05 · 基础：78×、80×、104× 的比较对象和范围是什么？

<details>
<summary>答案</summary>

是 RTX 4090/i9-12900K 条件下，Table 1 三个数据集上 CoACD 与 CuACD 平均分解时间的比值近似。所有方法使用统一 PaMO 输入，但计时排除了 PaMO。不是完整资产导入加速，也不是训练速度或仿真 FPS 加速。

</details>

---

### 06 · 中级：为什么体积残差和 Hausdorff 要一起用？

<details>
<summary>答案</summary>

小体积的细长错误仍可能严重影响接触。kR_v 描述凸包填入体积的尺度，双向 Hausdorff 关注最坏表面距离。搜索中用较便宜的体积项，终止检查完整指标；实现中的 Hausdorff 是采样估计，仍有漏检风险。

</details>

---

### 07 · 中级：一个 ballot mask 如何让保留的 lane 写入不同位置？

<details>
<summary>答案</summary>

对 lane ℓ，将 mask 与低于 ℓ 的位掩码相与再 popcount，得到本批中排在它前面的保留 lane 数；加上前面批次的输出数量即可得到唯一紧凑索引。仅 pred=true 的有效 lane 写入。最后不足 32 个元素的批次必须屏蔽无效 lane。

</details>

---

### 08 · 中级：设备端 heap 为什么影响整个算法组织？

<details>
<summary>答案</summary>

切割交点、封口和凸包输出都是动态长度。设备端分配可以在生产者计算出数量后直接供给消费者，从而融合阶段并减少主机参与。仅把算术换成 CUDA，而仍回 CPU 申请每一段输出，会保留关键开销。公开源码仍存在外层同步，所以要把阶段内部与全局控制分开。

</details>

---

### 09 · 中级：方向极值预筛选为什么不应损失真正的凸包顶点？

<details>
<summary>答案</summary>

见证点是输入子集，其凸包 P0 包含于完整凸包。严格位于 P0 内的点无法扩张 hull，可以删除。保留外部/边界点并最终执行完整 D&C，恢复完整凸包。保证依赖正确谓词和坐标表示；教学例子仅在二维验证该逻辑。

</details>

---

### 10 · 中级：为什么先拆连通分量再继续切会更好？

<details>
<summary>答案</summary>

空间分离的组件若被当成一个部件，会产生跨越空隙的大凸包，并让搜索用平面反复分开本已断开的部分。Table 4 中取消逐轮 CC-Split 后，平均部件数从 33.57 增至 40.87、时间 0.23 增至 0.25 s。这是实验支持，不是所有输入的必然规律。

</details>

---

### 11 · 高级：CPU 控制实验能否完全隔离 GPU 实现的作用？

<details>
<summary>答案</summary>

它移植凹边候选和参数后获得接近质量，支持候选是质量改善的重要来源；20.85 s 对 0.23 s 支持系统加速贡献。但 CPU 仍是 MCTS，GPU 是 bounded look-ahead；并非完全相同算法逐指令对照，不能把所有差异精确归因于单一因素。

</details>

---

### 12 · 高级：“Fully GPU-Resident”能否理解成整个 Python 调用零 CPU 参与？

<details>
<summary>答案</summary>

不能。固定 revision 的 Python 入口先调用 SciPy ConvexHull；native C 的外层循环读回 n_cutting、next_n，并调用 cuStreamSynchronize。可确认几何计算和大量中间状态驻留设备，不能确认论文所述上传后无中间同步适用于这个公开版本。静态差异也不能直接估算速度影响。

</details>

---

### 13 · 高级：τ=0.05 是否是每个模型 5 cm 的严格误差保证？

<details>
<summary>答案</summary>

不是。API 需要把最大包围盒边长归一到 2，阈值随此坐标表示。Hausdorff 为采样估计；Table 5 报告最大凹度 0.059。阈值是优化与停止的设定，不是未经额外验证的连续几何严格上界。实际接触任务还需独立评估。

</details>

---

### 14 · 高级：为什么“内存只有约 1.4 GiB”可能误导服务部署？

<details>
<summary>答案</summary>

论文同时说明 context 创建会预留可用 VRAM 的 70%。内部活跃内存、内部峰值和进程预留量不相同；释放给 device heap 的块也不一定归还 CUDA allocator。与训练/渲染共卡时应明确 pool 和生命周期，并实测长期复用与碎片。

</details>

---

### 15 · 高级：怎样设计一个能区分系统瓶颈与几何质量的复现实验？

<details>
<summary>答案</summary>

固定 revision、输入/PaMO/归一化与显式搜索参数，记录冷启动和稳态分解；用 profiler 测 CPU 同步、kernel 时间、内存和 occupancy。质量用更多表面样本独立检查，再测目标引擎中的通道、接触与 cooking。比较数量、误差和耗时，不用单个容易网格或不同阈值推断普遍提升。

</details>

---

