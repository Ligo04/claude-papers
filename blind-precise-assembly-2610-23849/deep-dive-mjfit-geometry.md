# mjfit 几何构造：从轮廓到凸碰撞块

源码核对日期：2026-10-20。固定版本 `eee986cc087d14458337fdc195d5cf03abc6aa6d`，`mjfit 0.1.0`。本次增加实际生成与 MuJoCo 编译检查，尚未执行动力学或作者性能基准。

## 1. 怎样用凸物体表示带孔物体

一个有孔板整体是非凸的。若把整块板作为一个采用凸包碰撞的 mesh，孔会消失。`mjfit` 将板分成很多凸棱柱，各棱柱的并集围出孔。

若二维材料区域为 A，每个凸分块为 Cᵢ，则：

```text
A = C₁ ∪ C₂ ∪ ... ∪ Cₖ
三维板 = ∪ᵢ (Cᵢ × [z₀,z₁])
```

二维凸多边形沿直线挤出仍然凸。每个棱柱取凸包得到自身，所以 MuJoCo 不会因凸包运算在该块以外添加材料。这些 geom 可以都挂在同一个 body 上；凸块多不代表多了关节或自由度。

源文件：[`_mjcf.py:117`](sources/mjfit/src/mjfit/_mjcf.py#L117)、[`_peg.py:168`](sources/mjfit/src/mjfit/_peg.py#L168)。本目录链接附行号仅作定位提示；精确行号见文件内容或下方 GitHub 链接。

## 2. 间隙是沿边法向偏移，不是把零件整体缩放

对于逆时针轮廓，设边向量为 eᵢ，向内单位法向为 nᵢ。孔边过点 vᵢ，插销对应边由以下直线定义：

```text
孔边：   nᵢ · (x − vᵢ) = 0
销边：   nᵢ · (x − vᵢ) = c
```

新顶点是相邻两条偏移直线的交点，即 mitred offset。`offset_inward()` 通过一个 2×2 线性系统求这个交点；相邻边共线时直接沿法向移动。

例如孔宽 20 mm、高 10 mm，单边间隙 c=0.1 mm，则配对销为 19.8×9.8 mm。这是教学算例。若简单统一缩放 0.98，得到 19.6×9.8 mm，两方向单边间隙分别为 0.2 和 0.1 mm，已经不同。

如果输入的是插销轮廓，程序反过来向外偏移 c 生成孔。尖角会产生长的斜接顶点位移，内角 α 下约为 `c/sin(α/2)`。这不等于法向间隙变了。间隙过大使局部特征翻转或自交时，`offset_is_valid()` 检查边方向、面积和简单多边形条件，并拒绝无效轮廓。

源码：[`_polygon.py:144–181`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/_polygon.py#L144)；调用在 `_peg.py:146–163`。

## 3. 孔周围的材料具体怎么切

### 方板与星形可见的孔

“星形可见”是几何性质：存在内部点可以看见轮廓的所有边，连线不穿出孔。它不只指五角星；任意凸孔也满足这个性质。此实现选择顶点坐标的均值作为检查点。

对孔的每一条边：

1. 从检查点向两个端点发射射线。
2. 求射线与外方板边界的交点。
3. 收集两条射线之间可能经过的方板角。
4. 用这些点与原孔边围出一个凸楔形区域。

每条孔边对应一块，公共边严格对齐。因此十条边的五角星孔生成十块板壁，不需要选择通用三维凸分解的凹度阈值。

### 其他轮廓

| 区域 | 实现选择 |
|---|---|
| 方板，孔从所选中心可见 | `radial_wedges()`，每孔边一楔形 |
| 孔不满足该检查，或指定了外板轮廓 | 对“外板减去孔”用 `mapbox_earcut` 三角剖分 |
| 凸插销 | 一个凸棱柱 |
| 从顶点均值可见的非凸插销 | 从中心向各边做扇形三角剖分 |
| 其他简单插销 | ear clipping |
| 孔中 islands | 每个 island 本身独立分块；销的 pocket 周围用带孔三角剖分 |

孔板与插销的规则不同。不要把 `mjfit` 全部概括成三角剖分后贪心合并；销孔模块没有统一做这种合并。螺纹的非凸齿形另有凸分区逻辑。

源码：[`_polygon.py:204–341`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/_polygon.py#L204)，`_peg.py:168–203`。

## 4. 从二维凸块到三维 mesh

给定一个凸多边形的 n 个顶点 `(xᵢ,yᵢ)`，`Meshes.prism()` 生成上下两圈：

```text
(x₁,y₁,z₀), ... , (xₙ,yₙ,z₀)
(x₁,y₁,z₁), ... , (xₙ,yₙ,z₁)
```

默认将这 2n 个顶点写到 MJCF `<mesh vertex="...">`，由 MuJoCo 构造凸包；也可以导出 OBJ。一个板 body 可以引用十个这种 mesh geom，插销 body 引用另十个。

因此可以同时有细致的凹形轮廓和全部凸的接触图元。代价是碰撞对子、接触约束和内部接缝数量可能增加。

## 5. 有内部凸起时，销怎样留空

USB 插座舌片或连接器针脚可以表示成 `islands`，即孔内必须保留的实体区域。

销的外轮廓是孔向内偏移 c；每个 pocket 则是对应 island 向外偏移 c。销的截面是“外轮廓减去所有 pocket”。程序检查 island 在孔内、pocket 在销内，以及 pocket 彼此分离。

在深度方向，pocket 深度为 `plate_thickness + c`，上方保留实心背部；因此销总长必须大于这个深度。这不是任意三维内部空腔系统，island 高度也受生成器假设约束。

依据：[`_peg.py:157–203`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/_peg.py#L157)。

## 6. 圆孔的两种误差，系数不同

设理想孔半径为 R，外接正 n 边形的边与圆相切，则其顶点半径是：

```text
R_vertex = R / cos(π/n)
孔角点的形状外扩：e_shape = R × (sec(π/n) − 1)
```

`shapes.circle()` 用这个顶点半径。给定 `max_error` 时，选择满足该上界的边数；指定 n 时直接使用 n。

配对销由同一个孔多边形向内偏移 c 得到。对应顶点之间的径向间隔是 `c/cos(π/n)`，相对法向间隙多出：

```text
e_clearance = c × (sec(π/n) − 1)
```

对于固定相对方向的凸配对轮廓，这也能通过中心平移约束 `normalᵢ·translation ≤ c` 理解。前一个公式描述多边形相对理想圆的形状误差，后一个描述配对轮廓的间隙量。

例如 R=8 mm、c=0.1 mm、n=32，按公式计算：

- 孔角点相对理想圆外扩约 38.7 μm。
- 配对间隙的最大额外量约 0.484 μm。

这是公式算例，不是论文表 4 的仿真沉降误差。碰撞基准中的精确 MuJoCo 圆柱销也不能直接等同于生成器的配对多边形销。

另一个源码细节：`stadium` 的圆弧同样外接，但 `bezier()` 对曲线采样并用弦连接。其弦可切入理想曲线，误差由 `max_error` 控制；生成出的多边形被视为实际设计轮廓。因此不能说所有曲线都保持理想连续形状的外接边界。

源码：[`shapes.py:42–100`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/shapes.py#L42)、[`shapes.py:137–175`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/shapes.py#L137)。

## 7. 螺纹为什么更复杂

螺纹先定义子午截面上的二维齿形 `(r,u)`，再沿螺旋扫掠：

```text
x = r cos θ
y = r sin θ
z = u + lead·θ/(2π)
lead = n_starts × pitch
```

每个小角区间 `[θ,θ+Δθ]` 用有限顶点的凸包表示。螺栓由芯柱与外牙楔块组成；螺母由外壳和朝内的牙楔块组成。

间隙在截面上生成：两侧暴露齿边分别后退 c/2，螺母齿根相应外移。这里的 `clearance` 是生成器截面中的法向间隔定义，不要直接当成三维螺旋曲面的全局最短距离。

关键困难是凸包表面是弦面，不是螺旋曲面：

- 内螺纹的弦可能向轴线方向下垂，侵入螺栓空间。源码用近似 `sec(Δθ/2)` 的径向外推补偿。
- 扭转面四边形的凸包还可能越过理论牙侧。源码对牙侧进一步后退，处理 helicoid twist。
- 外螺纹也并非任何齿型都自动安全。陡峭牙侧、方牙等情况需要额外补偿。
- 对候选凸包面做采样和局部细化，估计剩余越界并继续退让。这是数值检查，不是一般 CAD 曲面的形式化无误差证明。
- `seam_relief` 在楔块接缝处移除少量材料，做缓坡，减少跨接缝时卡住。

因此少分段会增加游隙，增加分段又会提高 geom 与接缝数量。不能假定分段越多，动态行为就越好。

源码：[`_thread.py:619`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/_thread.py#L619)，尤其内部扫掠、`shrink`、`excess` 的处理；参考 [`docs/threads.md`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/docs/threads.md)。

## 8. 齿轮怎样构造

生成器先按模数、齿数、压力角计算节圆与基圆，采样渐开线齿廓。正常参数下每齿一个凸棱柱，再加一个连接齿根的轮毂棱柱。如果齿廓不是凸的，退回从中心做扇形三角剖分。

齿侧间隙由节圆处的 `tooth_width` 控制。设两个相同设定齿轮都取圆周齿距的 w 倍为齿厚，则在标准中心距下，剩余比例为 `1−2w`。默认 w=0.48，留下约 4% 齿距的余量。

生成的轮廓是采样近似，没有制造级齿根圆角。支持外啮合直齿轮，不等于支持斜齿、锥齿或内齿轮。齿轮接触是否稳定仍依赖求解器和约束设置，几何构造本身不能保证动力学。

源码：[`_gear.py:146–245`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/src/mjfit/_gear.py#L146)，[`docs/gears.md`](https://github.com/younghyopark/mjfit/blob/eee986cc087d14458337fdc195d5cf03abc6aa6d/docs/gears.md)。

## 9. 本次实际生成与检查

运行：

```bash
~/claude-papers/papers/blind-precise-assembly-2610-23849/.geometry-venv/bin/python \
  ~/claude-papers/papers/blind-precise-assembly-2610-23849/code/inspect_mjfit_geometry.py
```

示例参数为五角星外半径 12 mm、内半径 5.4 mm，方板边长 48 mm、厚度 8 mm，销长 20 mm、单边间隙 0.5 mm。这是教学示例，不是论文真机试验对象。

实际得到夹具 10 块、插销 10 块，MuJoCo 中共 20 个 mesh、2 个刚体，不含 world。每块均为凸多边形挤出，分块面积总和匹配板减孔和插销轮廓面积；对应边法向间隙约为 0.5 mm。MuJoCo 3.16.0 编译后，凸包体积相对棱柱体积的最大相对差为约 `8.41e−8`。

输出文件：

- [可直接加载的 MJCF](outputs/mjfit-star.xml)
- [实际凸块坐标与核验记录](outputs/mjfit-geometry-inspection.json)
- [聚焦源码测试日志](outputs/mjfit-focused-tests.log)

这证明所选示例的构造和编译几何符合检查条件，没有验证动态装配、所有可能输入、完整螺纹测试矩阵或跨引擎表现。

本次另运行 15 项现有聚焦测试，全部通过，耗时 0.79 秒：销孔凸包体积、边间隙、圆形外接、Bézier 弦误差、螺纹侧面间隙，以及 M10 螺纹 `c=100 μm / 24 segments / 无 seam relief` 的一组凸楔块内含检查。测试只覆盖这些参数，不能推成全部齿型和精度设置都已验证。
