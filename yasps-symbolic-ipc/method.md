# 方法 —— 详细技术讲解

本文件遵循论文自身的结构(第 3–8 节),但为 *学习* 重新组织:先讲数据模型与前端,再讲两个算子,然后是
微分、索引生成、代码生成和求解器。伪代码与文字并重;实现陷阱和复现风险就地标注。

---

## 0. 系统全貌(第 3 节)

给定用户定义的能量以及要最小化的属性集合,YASPS 自动:

1. 计算局部梯度 `g_local` 和局部 Hessian `H_local`(自动投影到 PSD),
2. 装配全局 `g` 和全局 `H`,
3. 通过 GPU 共轭梯度求解 `H·x = g`(相当于一次牛顿步)。

```
        前端(Python)                            后端(自动,GPU/CUDA)
 scene / mesh / primitives / attributes  ┌──────────────────────────────────────────┐
 connectivities, JOIN, UNION, energies   │  符号微分器                                │
 addMinimizeTarget(...)        ──────────▶│     → 符号 g, H(作为属性)                 │
 minimizeEnergy(tol)                      │  索引生成器                                │
                                          │     → 全局稀疏 + 局部→全局映射             │
                                          │     → 压缩的全局 Hessian 结构              │
                                          │  代码生成器 / JIT                          │
                                          │     → CUDA kernel(.o),链接               │
                                          │     → PSD 投影(EVD)kernel                │
                                          │  最小化器                                  │
                                          │     → 装配 H,g;PCG 求解 H x = g           │
                                          └──────────────────────────────────────────┘
```

**刻意排除**(因为它们与参数化耦合):**碰撞检测 / CCD** 和 **牛顿步进 / 线搜索**。YASPS 只返回线性系统的解
`x`;用户把它接入自己的步进/阻尼/终止策略。例子里用了一个专门针对 *线性* 参数化的 CCD。

---

## 1. 数据模型(第 4.1–4.4 节)

三层作用域:**scene ⊃ mesh ⊃ primitive type(图元类型)**。它们本身 *不* 携带几何——位置、质量等都是
层叠在上面的 **属性(attributes)**。

### 图元类型(Primitive type)
一袋 `n` 个实例,索引集 `[n] = {1,…,n}`。例如:`vertices`(n = NUM_VERTS)、`tets`(n = NUM_TETS)、
`affine_body`(n = 1)、`pp` 点-点碰撞对(n = 0,动态)。

```python
s0 = scene("scene0")
bunny_affine = s0.addMesh("bunny_affine")
bunny_affine.addPrimitive("vertices",    numInstances=NUM_BUNNY_VERTEX)
bunny_affine.addPrimitive("affine_body", numInstances=1)
bunny_affine.addPrimitive("tets",        numInstances=NUM_TETS)
```

### 属性(Attribute)
形式上是函数 `α_X : [n] → ℝ^{r×c}`(每实例 r×c 矩阵);堆叠为张量 `ℝ^{n×r×c}`。存为扁平 GPU 数组,大小 `n·r·c`。

```python
bunny_affine.affine_body.addAttribute("affine_matrix", rows=3, cols=3)
bunny_affine.affine_body["affine_matrix"].updateValue(np.eye(3))
bunny_affine.vertices.addConstant("rest_position", rows=3, cols=1)   # addConstant ⇒ 不参与微分(∂ = 0)
```

- `addAttribute` vs `addConstant`:**命名关乎微分,而非可变性。** `Constant` 仍可数值更新;它只是表示
  "对此求导恒为 0"。

### 属性的计算 —— 符号化 & 广播
对属性的任何表达式都产生一个 **新的符号属性**(计算图),而 *不是* 一个值。

```python
rp      = bunny_affine.vertices["rest_position"]   # 3×1
dotted  = rp.dot(rp)                               # 1×1,父级 = vertices(血统推断)
dotted2 = 2.0 * dotted                             # 1×1
dotted2.compute().value                            # 此刻才代码生成+运行,返回扁平数组
```

关键行为与 **陷阱**:
- **不物化中间量。** `dotted` 从不写入内存;只有 `dotted2` 的结果会。(写比算更贵。)→ *复现风险:* 若你
  试图 `.value` 一个中间量并期望它被缓存,它不会。
- 符号属性表示 **每实例** 表达式(形状 r×c);算子在代码生成时 **隐式广播** 到所有 `n` 个实例。
- **血统规则(lineage rule):** 一次计算里的所有属性必须共享同一血统(同一图元类型,或一个影响其下所有
  内容的祖先 scene/mesh 属性)。scene 级 `timestep_size` 能与顶点属性结合;但 `youngs_modulus`(在 `tets`)
  不能直接与 `rest_position`(在 `vertices`)结合。**唯一的例外是 `JOIN` 和 `UNION`**,它们正是为跨血统而生。

### 连接关系(Connectivity)
一个映射 `C : I_A → I_B^k`,**arity 为 k**,从图元类型 A 到 B:A 的每个实例引用 B 的一个 k 元组。
例如 `C_tet2v(i) = (j₁,j₂,j₃,j₄)`。

```python
# 顶点 → 仿射体(arity 1;所有顶点指向唯一的 body 0)
bv2abd = bunny_affine.vertices.addConnectivity("...", bunny_affine.affine_body, [0]*NUM_BUNNY_VERTEX, 1)
# tet → 4 顶点(arity 4)
affine_tet2v = bunny_affine.tets.addConnectivity("tet2v", bunny_affine.vertices, TET_INDICES, 4)
```

---

## 2. `JOIN`(第 4.5 节)—— 沿关系收集

**定义。** 对连接关系 `C : I_A → I_B^k` 和属性 `α_B : I_B → ℝ^{r×c}`:

```
JOIN_C(α_B)(i) = ( α_B(j₁(i)), …, α_B(j_k(i)) ) ∈ ℝ^{k×(rc)}   其中 (j₁,…,j_k) = C(i)
```

每个 A 实例收集它引用的 k 个 B 属性,沿大小为 k 的新前导维堆叠(末尾 r×c 展平以避免张量膨胀)。结果
**归 A 所有**,因此重新进入正常血统规则。`JOIN` 自身不遵守血统规则(那是它的职责),但它的输出遵守。

```python
# 通过 arity-1 的 bv2abd 把 affine_matrix(3×3)JOIN 到 vertices → 每实例 1×9
bva = bunny_affine.vertices.addAttribute("affine_matrix", through=bv2abd,
                                          source=bunny_affine.affine_body["affine_matrix"])
bva = bva.resize(3, 3)                       # 1×9 → 3×3
# 现在在顶点上计算世界位置 p = A·r + t
current_position = bva * bunny_affine.vertices["rest_position"] + bvt
bvp = bunny_affine.vertices.addAttribute("position", computed_attribute=current_position)
# 把这些位置 JOIN 到 tets(arity 4)→ 每实例 4×3
btp = bunny_affine.tets.addAttribute("positions", through=affine_tet2v, source=bvp)
```

**为何 JOIN 对微分友好(第 5.1 节)。** JOIN 只是拼接,所以:
- `∂JOIN_C(α_B)/∂(被收集项) = I`(恒等),且
- 若每个 `α_B` 依赖参数 β,则 `∂JOIN_C(α_B)/∂β` 是 **块对角** 的(每个被收集实例一块)。
- 丢掉零的非对角块 ⇒ **`∂JOIN_C(α_B)/∂β ≡ JOIN_C(∂α_B/∂β)`**:*微分与 JOIN 可交换。* 二阶导同理
  (JOIN 线性 ⇒ 链式法则的曲率项消失)。

---

## 3. `UNION`(第 4.6 节)—— 合并可选参数化

**动机。** 一个碰撞能量可能触及不同参数化的顶点(自由 vs ABD)。为每种写单独代码路径会把图分裂
(free-free、free-ABD、ABD-free、ABD-ABD),每个都要自己的导数 + 装配。数值预计算世界位置又会丢掉导数轨迹。

**定义。** 对带属性 `α_{Xⱼ} : [nⱼ] → ℝ^{r×c}`(同形状)的图元类型 `X₁,…,Xₘ`:

```
UNION(α_{X₁},…,α_{Xₘ}) : ⨆ⱼ [nⱼ] → ℝ^{r×c},     UNION(…)(j,i) = α_{Xⱼ}(i)
```

定义域是 **不相交(带标签)并集**:一个元素是带标签的对 `(j, i)`(哪个分支 j、哪个实例 i),运行时压缩成
单个可逆整型索引。

**`primitiveUnion` 构造。** 一个 UNION 横跨多个图元类型,似乎违反单一血统规则。YASPS 用一个新实体
**`primitiveUnion`** 解决,它位于 mesh 下一级,声明它统一了哪些图元。UNION 的属性挂在 `primitiveUnion` 上
——恢复单一血统。

```python
bunnies = s0.addMesh("bunnies")
bunnies.addPrimitiveUnion("vertices", [bunny_affine.vertices, bunny_soft.vertices])
bunnies.vertices.addAttribute("position")        # 字符串 ⇒ 两个子图元中都必须存在且同形状
bunnies.vertices.addAttribute("rest_position")

# 点-点碰撞对:构建时数量未知
bunnies.addPrimitive("pp", numInstances=0, isDynamic=True)
pp2v = bunnies.pp.addConnectivity("pp2v", bunnies.vertices, [], 2)        # 运行时由 CCD 填充
pp_positions = bunnies.pp.addAttribute("positions", through=pp2v,
                                        source=bunnies.vertices["position"])   # 每实例 2×3

def point_point(position, dHat, kappa):
    p0, p1 = position.row(0), position.row(1)
    d     = (p1 - p0).dot(p1 - p0)
    I5    = d / dHat
    lenE  = d - dHat
    return kappa * lenE * lenE * I5.log() * I5.log()

pp_energy = bunnies.pp.addAttribute("point_point", computed_attribute=point_point(pp_positions, DHAT, KAPPA))
```

**为何 UNION 对微分友好(第 5.2 节)。** `∂UNION(α₁,…,αₘ)/∂(children) = UNION(I_{X₁},…,I_{Xₘ})`,
对任意参数:`∂UNION(…)/∂β = UNION(∂α₁/∂β,…,∂αₘ/∂β)`。*微分与 UNION 可交换*,所以混合参数化的导数仍是
**一个** 符号节点——编译期无分裂。运行时 UNION 仍做路径选择,但只有一个 kernel,而非 2^k。

**填充(padding)微妙之处。** 不同分支可能依赖 *不同维度* 的参数集 `β₁,…,βₘ`(自由:3,ABD:12)。
为在一个 UNION 中合并它们,YASPS 找到最大参数维 `q_l` 并把每个导数 **填充** 到形状 `p × q_l`。这正是符号
Hessian 保守地变成 **最大尺寸**(如 ABD-ABD 24×24)的原因——也是后续压缩要撤销的东西。

---

## 4. 符号微分(第 5 节)—— 引擎

两个 **要求**:(a)对 YASPS 图求导得到另一个 YASPS 图(对 JOIN/UNION 封闭);(b)对 UNION 求导得到 *单一*
统一图(无指数分支)。两个 **性能需求**:最大化复用符号中间量;产出适合 PSD 投影的 Hessian。

### 两遍式二阶链式法则
骨架是式 (3):

```
∇²ₓ f(g(x)) = Jg(x)ᵀ · ∇²_g f(g(x)) · Jg(x)  +  Σⱼ (∂f/∂uⱼ)(g(x)) · ∇²ₓ gⱼ(x)
```

- **第一遍:** 遍历图,推导局部 Jacobian、梯度、Hessian。
- **第二遍:** 用链式法则传播它们,装配出能量的完整梯度和 Hessian。

### 边界节点拆分(第 5.3 节)—— 复用机制
链式法则可递归组合,所以 YASPS 不必只选一个切点 `f∘g`。它把注意力限制在 **边界节点** 上:
- 能量属性,
- `JOIN` 属性,
- `UNION` 属性,
- 数据属性(用户提供的叶子)。

**算法 1** 把图粗化成一个"边界图",其边标记出必须计算局部 Jacobian/Hessian 的位置。**关键在于**,因为
微分与 JOIN/UNION *可交换*(第 5.1–5.2 节),YASPS **不** 直接对边界 JOIN/UNION 节点求导——而是对该节点的
*子节点* 关于上游边界节点求导(**算法 2**),再用算子规则符号化地推断父节点导数。**算法 3** 沿边界图递归
应用链式法则,装配最终的符号梯度/Hessian。

**两层复用:** (1)某条边界边上的局部 Jacobian/Hessian 只算 *一次*,被触及该边的每个能量共享;
(2)流经共享节点(如统一的 `vertices.position` 同时喂给弹性 + 点-三角 + 点-边能量)的中间梯度被复用而非
重算。这种复用是对生成代码做公共子表达式消除抓不到的。

> **实践提示。** 式 (3) 是为标量 `f` 写的;边界节点通常不是标量。YASPS 逐分量应用链式法则再堆叠结果,以避免
> 张量表示。非微分目标的数据节点被丢弃。

### 降维投影重写(第 5.4 节)—— 大性能杠杆
当 **曲率项**(式 3 的第二个求和)在编译期可证为 **零** 时——即内层映射 `g` 是 *线性* 的(对自由顶点和
仿射体成立!)——YASPS 把局部块

```
   Jg(x)ᵀ · ∇²_g f(g(x)) · Jg(x)
⟶ Jg(x)ᵀ · Project( ∇²_g f(g(x)) ) · Jg(x)
```

重写。`Project(·)` 是一个被编译成"特征分解-并-截断"kernel 的符号算子。这把 PSD 投影从 *扩展 DoF 空间* 移到
*更小的 g 空间*。对自由+ABD 顶点的点-点势垒:投影 **6×6** 内层 Hessian 而非 **24×24** 扩展块 → 头条的 ~44× 收益。

---

## 5. 索引生成器(第 6 节)

为全局装配 `H·x = g`,你必须知道 **每个每实例局部块落在哪里**。YASPS 通过遍历 *同一张* 计算图获得这个信息,
但把 **JOIN/UNION 下降为连接关系的索引列表而非数值数据**。放置索引直接得出。因为图在编译期固定,索引计算
本身被 **GPU 并行化**,每 kernel 静态内存——这很重要,因为碰撞连接关系每步都变(动态实例)。索引生成器还
产出 **压缩的全局 Hessian 结构**(附录 B)以最小化存储与计算。

---

## 6. 代码生成器 / JIT(第 7 节)

JIT 编译器把符号图下降为 **GPU 优化的 CUDA**。

### 模块化代码生成(第 7.1 节)
- 深度优先遍历符号树,做 **公共子表达式消除**,组合成 kernel;遍历会记录读取了哪些叶子数据属性,因此
  **kernel 参数自动推断**(无需手写签名)。
- **不发射单一巨型 kernel。** "语义重要"的节点——请求的输出、`JOIN`/`UNION` 节点、以及 **命名** 属性——被
  编译成 **独立的 object 文件** 再 *链接*,类似 C 的分离编译。共享子图(如统一位置)只编译一次,被所有下游
  kernel 复用。
- 三个阶段(**算法 4–6**):(1)从请求输出做 DFS 收集重要节点 `N` 且不再深入它们,把其余压进无重复栈 `S`;
  (2)在 `S` 上发射代码串,用中间值替换重复;(3)链接此前为重要节点生成的 `.o` 文件,编译出最终可链接 kernel。
  `β = 2·α`(α 是独立 JOIN 模块)的 kernel 只需 α 的 *符号引用*;α 的实现稍后链接。

### Hessian 代码生成与局部压缩(第 7.2 节)
在模块化流水线之上,两个二阶专属优化:

1. **可选的 `J^T·H_inner·J` 物化。** 当第 5.4 节条件成立,可选择只物化 `J` 和投影后的内层 Hessian,在
   *散射(scatter)时即时* 计算 `J^T·H_inner·J`。当 `H_final` 会超过 `J + H_inner` 时省峰值内存(如秩亏局部
   ⇒ **6× 尺寸缩减**,第 10.9 节);代价是装配时的额外算术。很少需要,因为局部块通常很小。
2. **投影前的局部块压缩。** UNION 预留最大尺寸块,所以很多行/列结构为零(失活分支)——去掉它们。此外,当
   两个参与顶点共享 *相同* 底层参数(图 12/13:P1、P2 都来自 MeshA,共享 α、β),几个局部块映射到 *同一* 全局
   块,可在 **投影前聚合**,缩小 EVD 维度。(一个块 `∂²E/∂β∂α` 甚至消失——它被压进 `∂²E/∂α∂β` 在下三角的
   转置。)压缩还减少散射进无重复压缩全局 Hessian 时的原子加数量。

**按尺寸按需编译(图 11)。** 值得注意的是,YASPS **不** 按 *参数组合* 生成单独 kernel(那就是爆炸)。它按
**压缩后的 Hessian 尺寸 k** 生成:运行时每遇到一个新的尺寸-`k` 局部块,就 JIT 一个 `f_k.o`,在累加前把块压缩
到 `k×k`,并由一个协调 kernel 对 `hessian_size` 做 `switch`。实践中 kernel 数量小得多。

---

## 7. 求解器(第 8 节)

其余一切都在 GPU 上,所以线性求解也是:一个 **修改的预条件共轭梯度(PCG)** [Baraff & Witkin 1998],主要工作是
稀疏矩阵-向量乘(SpMV)。两个优化轴:

- **更便宜的 SpMV:** 通过消除重复块进一步压缩全局 Hessian(附录 B)+ 一个匹配该存储布局的定制 SpMV kernel
  (附录 C)。*这就是 YASPS 碾压 GIPC 的 CG 时间之处*——如 39.6k 顶点时每次 CG 迭代 0.075 ms vs 0.97 ms。
- **更少的 PCG 迭代:** 一个 **块 Jacobi 预条件子**(附录 D)。

---

## 8. 端到端走一遍(双 bunny 贯穿例子)

1. 构建 scene,两个 mesh(`bunny_soft`、`bunny_affine`);加图元(`vertices`、`tets`、`affine_body`)。
2. 加属性(`rest_position`、`affine_matrix`、`translation`);JOIN 它们以计算顶点世界位置。
3. 建一个 `bunnies` mesh,`addPrimitiveUnion("vertices", [...])`,UNION 这些 `position` 属性。
4. 加动态 `pp` 碰撞图元 + `pp2v` 连接关系(每步由外部 CCD 填充);JOIN 位置,定义 `point_point` 势垒能量。
5. 注册能量:`s0.addEnergy(snh_energy)`、`addEnergy(affine_energy)`、`addEnergy(pp_energy, dynamic_instances=True)`。
6. `s0.addMinimizeTarget([soft.position, affine.affine_matrix, affine.translation])`。
7. `result = s0.minimizeEnergy(tolerance=1e-6)` → 首次调用触发微分 + 索引生成 + 代码生成 + 编译;后续调用复用
   kernel。返回每个目标的(取负后的)更新方向。
8. 用户用自己的牛顿步进/线搜索施加更新并循环。

---

## 实现陷阱与复现风险(汇总)

- **中间量从不物化** —— 别指望缓存某个未命名子表达式的 `.value`。
- **血统纪律** —— 非祖先、跨图元的属性只能经由 JOIN/UNION 相遇;忘了 `primitiveUnion` 是经典错误。
- **UNION 形状匹配** —— UNION 的属性在各分支必须同形状;字符串命名的 UNION 属性必须在 *每个* 子图元类型中存在。
- **UNION ⇒ 最大尺寸 Hessian。** 若跳过块压缩 / 降维投影,在超大块(如 24×24)上的 O(n³) EVD 会悄悄主导运行时
  ——图 23 中 1857× vs 1× 的差距 *完全* 来自于此。
- **降维投影需要线性参数化。** 核实曲率项确实消失;非线性参数化会放弃这个杠杆。
- **动态实例**(`isDynamic=True`、`numInstances=0`)需要每步由 *外部* CCD 重填连接关系列表;YASPS 不提供 CD/CCD。
- **首次调用 JIT 延迟** —— 代码生成 + nvcc 编译发生在首次 `minimizeEnergy`;基准要测 *稳态* 时间,而非第一步。
- **手动能量重表述很重要** —— 通过额外的 JOIN+图元把 `Ψ(F(x))` 具体化(第 10.4 节)会改变被投影的矩阵
  (12→9、9→6),即便引擎不变也能带来 1.56×–2×。
- **超参敏感性** —— CG 相对容差(10⁻⁴)、牛顿位移/dt 阈值(10⁻² m/s)、势垒 `d̂`/`κ`、以及材料刚度都影响迭代
  次数与收敛(基准里 Stark 在碰撞上无法收敛,提醒步进/线搜索住在 YASPS *外部*)。

参见 [[insights]] 看这些机制 *为什么* 重要,[[mental-model]] 看它们如何拼在一起。
