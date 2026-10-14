# 反思与后续问题

## 如果继续扩展

1. **空间变化的 flow field**：把单一 \(r\) 换成低维、可正则化的 spatially varying field；先保证邻域连续，再把每个局部 response 投影回 solver constraint。
2. **可证明的 candidate locality**：用 surface geodesic、当前空间距离和 curvature-aware bound 共同决定候选域，给出“不会漏掉 restorative support”的充分条件。
3. **质量/刚度感知的冲突选择**：WIS 权重可以从单纯 penetration area 扩展为 effective mass、bending/stretch stiffness 和预估 solver work。
4. **device-resident pipeline**：把 CSR 建图、coverage、objective 和 response assembly 从 host evaluation 迁入 GPU，验证 542.3 ms GPU 路径的瓶颈是否真正下降。
5. **更完整的质量指标**：每帧同时报告 EF、triangle inversion、strain/bending energy、volume ratio、最小 clearance 和拓扑变化，而不只看 EF=0。
6. **参数分离**：在 held-out geometry 上分别消融 \(\kappa\)、\(\tau\)、k-ring growth、CCR mode、direction optimization 和 response-depth cap。

## 脆弱假设

- 一个 contour 的 support 可以被一个线性方向同时分开。
- 双侧 adjacency 连通分量与物理 response region 一致。
- 10 个 seed 足以覆盖主要能量 basin。
- 固定 barycentric weights 在一次 Newton step 中仍有合理几何意义。
- greedy contour suppression 不会抑制掉必要的互补 response。
- 用 EF=0 作为“解缠完成”不会掩盖过度拉伸、反转或错误的拓扑改变。

## 可证伪实验

- 构造多圈连续 twist，统计任何单向 \(r\) 下同时满足 \(\delta_h>0\) 的比例；预期会出现论文 Fig.14 的跨半球法向失败。
- 对同一 mesh 逐步增加远端折叠，比较 pure current Euclidean、rest geodesic、AND、blend CCR 的 false support rate 和 missed restorative hit rate。
- 固定 geometry，随机打乱 contour processing order，测 greedy suppression 对 final EF、volume ratio 和 energy 的敏感度。
- 对 59-case 做 train/validation split，禁止用 validation 调 \(\kappa\)；报告 success、iterations、volume 和 wall time 的置信区间。
- 让 full-mesh raycast 与 adaptive k-ring 在退化、非流形和近切向场景上逐 hit 对齐，而不是只比较 EF=0。

## 读者应带走的判断

这是一个把拓扑语义、几何搜索和物理 solver 接口连起来的强工程方案；论文的证据支持“在给定 benchmark 上稳定实用”，不支持“已解决通用解相交”。最有价值的下一步不是再堆更多方向，而是让 response locality、空间变化流和质量指标可验证。
