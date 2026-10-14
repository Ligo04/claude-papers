# Mental model：把 S4R 放进研究地图

## 这是什么问题？

S4R 属于 **静态几何可行化（static geometry repair）**：输入是已经有位姿的刚体网格集合，输出是尽量少移动、恢复到原始 full size、通过网格级无互穿检查的位姿。它的接口在物理仿真器之前，目标是给后续的 contact dynamics、motion planning 或 robot policy 一个合法起点。

不要把它归类为“更快的刚体动力学引擎”。它没有时间积分、质量、重力、摩擦、速度或冲量；论文也明确把它定位为 repair stage。MuJoCo/PhysX settle 是否稳定、物体是否 upright、摩擦是否让布局改变，都需要后续单独验收。

## 需要的先验

- 刚体位姿：平移 `p` 和旋转 `R`；默认路径只优化 `p`。
- 三角网格近邻查询：BVH/FCL、closest points、contact witness 和 separating direction。
- signed clearance：分离为正，穿透为负；最终 `Pen` 用严格负号判断。
- 凸二次规划：最小化平移增量的二范数，约束是接触法向上的线性分离不等式。
- numerical continuation：用易解的参数起点追踪到目标参数，而不是一次解决非凸大步。

## 一张图记住它

```text
深穿透输入
   │  fixed reference centers；缩小每个 body
   ▼
s = s_min（验证起点）
   │  预测将激活的接触；SOI 跳步；QP 修正
   │  每 M 步 full query，其余 frozen-witness 预测
   ▼
s ↑ 1（路径上可能有历史偏置，也可能堵塞）
   │  锁定 full size；tail QP + exact mesh evaluator
   ▼
可交给仿真器的几何初态 / 或明确失败状态
```

## 它在研究地图中的位置

1. **投影/PGS/全局 QP**：直接在当前重叠上推开，便宜但深穿透下的单次线性化可能留下 residual 或移动很远。
2. **罚函数/动力学修复**：把重叠变成力或能量，能和仿真统一，但静态初始化可能携带大冲量、改变布局，或者受步长与刚度影响。
3. **Barrier/IPC/CCD**：对真正无穿透的运动很强，但通常把无交叉起点作为前提；S4R 的一个用途是先把初态送进这个可用域。
4. **S4R**：沿几何尺度做 homotopy，把深问题拆成浅接触的局部 QP；重点收益来自减少 mesh query 和保持稀疏 active set。
5. **上层布局/SimReady Gate**：语言约束或关系约束决定“应该放在哪里”，S4R 只负责在 scale-space 中修复接触；之后还要 mesh verifier 和 MuJoCo settle。

## 读结果时的四个开关

- `continuation_complete`：是否真的到过 `s=1`。
- `native_converged`：solver 自己的 full-scale tail 是否停在可行状态。
- `status`：当前代码只有 full-size evaluator、有限位姿、墙/关节都满足时才叫 `converged`。
- `fcl_verified`：GPU 采样 oracle 之后，full scale exact FCL 是否复查通过。

任何一个中间尺度的“看起来分开”都不能替代这四个证据。博客的统一提醒是：**尺度空间改变的是求解路径或表示容差，原始几何任务仍要在原始单位和最终状态上验收。**
