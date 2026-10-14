# 关键洞见与批判性阅读

## 1. 真正改变的是“搜索变量”

GIA 把变量放在离散拓扑类别和 illegal region；ICM 把变量放在 contour length。本文把变量换成一条 contour 的**全局线性分离方向** \(r\)。这个变量足够小，能用 10 个 seed + 局部二阶试探搜索；又足够全局，能让同一 response band 的 hit 共享一个深度目标。于是“找到正确区域”和“选择移动方向”被同一目标函数连接起来。

这里的 global 不是“对所有点拟合任意形变场”，而是“在一个 contour 内使用同一条线性 displacement field”。这解释了它为什么比 ICM 更不容易在深腔中被局部梯度困住，也解释了连续扭转是它的结构性失败模式。

## 2. 为什么必须把 hit 当作乘积复形

一个 ray hit 不是普通点，而是 `(source primitive, destination primitive)` 的配对。只有 source 侧和 destination 侧同时相邻，两个 hit 才应连接。只在单侧 mesh adjacency 上 flood-fill 会把不同层、不同响应带粘在一起；双侧 product-complex adjacency 用离散拓扑保留“哪一对面在相交”。

锚定条件再砍掉 detached debris：连通分量必须触碰 contour 两侧的 one-ring seeds，并按 EF anchor coverage 选择物理 band。这是论文里最容易被一句“cluster culling”带过、但决定语义正确性的步骤。

## 3. Cramer 形式为何很适合方向优化

VF、EE、FV 三个几何测试都可写成 \([r\mid a\mid b]z=d\)。令 \(n=a\times b\)，Cramer 规则给

\[
\delta_h(r)=\frac{n^Td}{n^Tr}=\frac{\nu_h}{n^Tr}.
\]

节点位置固定时，\(\nu_h\) 不随方向改变；因此候选方向搜索不必反复解完整几何系统，只需更新一个分母。局部命中身份不变时，\(J(r)\) 是可微的；命中/cluster 发生离散切换时又会不连续。所以先做离散 basin sampling，再在固定 basin 中优化，是对目标函数真实形状的回应，而不是把非凸问题假装成凸问题。

## 4. 论文目标与当前代码目标不是完全相同

论文式 (2) 用 \(D_\mathcal C^2\sum A_h\) 排序。固定到仓库 `d361b71` 的代码在 `prp_is_better_response_hit_info` 中先比较 `penetration_objective`，再比较 depth、displacement cost、hit count、combo index；ResearchLog 还记录了 coverage normalization 和 effective-mass displacement cost。代码的工程目标比论文正文的简式更丰富，阅读源码时应以 comparator 和 debug fields 为准。

同样，论文的 CCR 公式使用当前变形空间 Euclidean chord distance；当前实现支持 rest-geodesic、current Euclidean、AND 和 blend 变体，默认 tau 为 0.25，并用它防止折叠层之间的“空气捷径”。这是合理的工程扩展，但不是论文表格中单独消融过的结论。

## 5. 自适应 k-ring 是速度边界，不是语义替代

从 contour anchors 做 BFS，按 \(k\) 过滤 V/E/F，并在 response frontier 未脱离当前 ring 时指数扩张。论文报告 878 条 contour 上 raw hits 平均 60,749 降到 1,360，且与 full-mesh valid hit 集合保持 100% parity。

它仍需要 cluster culling：hop distance 只能定义候选域，不能判断一组 hit 是否是同一个物理 response。当前代码的 `frontier_converged` 还是有效性条件；未闭合 frontier 的 combo 不应参与 best-result selection。

## 6. solver 兼容性来自“固定权重”，不是后处理位移

ray hit 确定 barycentric weights 后，constraint

\[
C_h(x)=n_h^T\left(\sum_j w_{h,j}x_{h,j}\right)-D_\mathcal C
\]

对节点位置是 affine 的；因此 \(\nabla^2 C_h=0\)，每个 pair 只贡献 \(\kappa A_h J_h^TJ_h\) 的常数 Gauss–Newton block。解缠不是先把 mesh 顶点硬挪一下再交还模拟器，而是把一个物理可解释的 penalty pair 注入同一个 Newton 系统。

## 7. 结果应怎样读

- **收敛**：EF 从初值到 0，说明报告的离散相交对被清掉。
- **形状**：表 5 的 \(V/V_0\) 显示过硬 penalty 会局部 pinching；bounded step + relaxation 才让形状回到平衡。
- **速度**：GPU 只消掉 ray traversal 不代表整帧线性加速；CSR 和 host assembly 仍是瓶颈。
- **泛化**：论文只在给出的 canonical、Synthetic、Round Table、Cloth Drop、Hair Card 和 Jang benchmark 上给出证据；“type-free”指不依赖预设 contour taxonomy，不代表对所有拓扑都有 theorem。

## 与前序方法的对照

| 方法 | 主要变量 | 优点 | 失败模式 |
|---|---|---|---|
| GIA | contour 类型 + illegal region | 全局区域语义清楚 | 分类/拓扑分支脆弱；错误 flood-fill 会持续制造新 EF |
| ICM | 局部 contour length | 简单、并行、易接入 | 深穿透、多层互锁掉进局部极小 |
| 本文 | 方向 \(r\) + product-complex support | 统一接口、与 solver 的位移代价一致 | 单一方向不能表达空间变形；候选域和 CCR 仍可能泄漏 |
| 学习/隐式方法 | learned/implicit proxy | 可针对特定数据加速 | 泛化、训练数据和闭壳体假设限制应用范围 |

## 可迁移的工程原则

1. 先把“有效响应语义”写成离散关系，再优化连续变量。
2. 对非光滑目标，保留多 basin 的离散 seed，不要只从一个梯度起点出发。
3. 把几何证据转换为现有 solver 已理解的局部约束，避免另造一套积分器。
4. 把“EF 清零”“位移小”“形状保真”“吞吐高”作为独立指标记录。
