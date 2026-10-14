# 自测题

### 基础 1：HKTex 要替代的核心对象是什么？
<details>
<summary>答案</summary>

它替代依赖 UV 展开的二维纹理图，把颜色表示成直接定义在三角网格流形上的一组各向异性热核。

</details>

---

### 基础 2：一个 HKTex 核有哪些主要参数？
<details>
<summary>答案</summary>

源点位置 `p*`、方向 `θ`、各向异性 `η`、尺度阈值 `τ`、边界锐度 `ς` 和颜色 `c`；全局还有可学习底色 `c_base`。

</details>

---

### 基础 3：热核与欧氏高斯的直观关系是什么？
<details>
<summary>答案</summary>

热核是流形上热方程的点源响应；在平坦欧氏空间中具有高斯形式，在弯曲网格上沿测地几何扩散。

</details>

---

### 基础 4：HKTex 如何从多个核得到颜色？
<details>
<summary>答案</summary>

选取点 `p` 处贡献最大的 top-`K_c` 个核，用 `α_s(p)` 做分区统一的 alpha 加权颜色，并加到全局底色上。

</details>

---

### 基础 5：论文报告的 UV 拟合数据集规模是多少？
<details>
<summary>答案</summary>

来自过滤后 Objaverse 子集的 313 个网格；多视图实验使用 162 个网格。

</details>

---

### 中级 1：为什么需要各向异性 LBO？
<details>
<summary>答案</summary>

普通 LBO 的扩散在切向方向上各向同性；ALBO 的 `η` 和 `θ` 让扩散椭圆可拉伸、可旋转，更有效表达定向纹理。

</details>

---

### 中级 2：谱基对齐解决了什么问题？
<details>
<summary>答案</summary>

不同 ALBO 的特征向量会排列交换、子空间旋转和符号翻转。Hungarian、Procrustes/SVD 和符号校正使网格节点之间的基可插值。

</details>

---

### 中级 3：双调和 KNN 如何改变复杂度？
<details>
<summary>答案</summary>

先在 `Z=Φ_isoΛ_iso^-1` 中做 KNN，只对 `K_s` 个邻近源计算热核，使复杂度从 `O(P×S)` 变为 `O(P×K_s)`。

</details>

---

### 中级 4：为什么普通 Adam 不能直接更新源点坐标？
<details>
<summary>答案</summary>

普通更新会把源点带离曲面。HKTex 将梯度投影到切平面，用指数映射沿测地线走，并对动量做平行移动。

</details>

---

### 中级 5：HKTex 的主要速度代价是什么？
<details>
<summary>答案</summary>

每个交点要做双调和 KNN、局部谱查询和核聚合；表 1 中纹理渲染平均约 784.7 ms。

</details>

---

### 高级 1：双调和门控为什么既能减少振铃又能支持 KNN？
<details>
<summary>答案</summary>

它用随双调和距离衰减的高斯压制谱截断的远场振铃，而同一距离在 `Z` 嵌入中等价于欧氏距离，因此可直接检索邻域。

</details>

---

### 高级 2：统一源质量 `m*=1` 为什么可能仍然有效？
<details>
<summary>答案</summary>

作者用参数补偿解释实验结果；按式 (3)–(4)，相同源质量在归一化分子和分母中会直接约掉。这是比补偿解释更直接的代数观察；其他未归一化质量用途仍需单独检查。

</details>

---

### 高级 3：top-k 选择会带来什么可微性风险？
<details>
<summary>答案</summary>

排名切换会让候选集合离散变化，造成梯度不连续或动画闪烁；软 top-k 或固定邻域可作为改进方向。

</details>

---

### 高级 4：为什么存储比较必须把 UV 坐标算进去？
<details>
<summary>答案</summary>

UV 方法还要保存每顶点坐标、接缝复制和空白区域；只比较图像尺寸会低估真实预算。论文统一计入拓扑开销、网络权重和核参数。

</details>

---

### 高级 5：扩展到粗糙度时最先要检查什么？
<details>
<summary>答案</summary>

检查同一核支撑是否适合不同频率的 albedo/粗糙度，为粗糙度设置物理范围，并在可微 PBR 中分开材质误差和光照误差。

</details>

---
