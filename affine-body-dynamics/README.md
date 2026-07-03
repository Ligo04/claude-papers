# Affine Body Dynamics (ABD) — 学习资料

> Lan, Kaufman, Li, Jiang, Yang. *Affine Body Dynamics: Fast, Stable and Intersection-free Simulation of Stiff Materials.* ACM TOG (SIGGRAPH 2022). [arXiv:2201.10022](https://arxiv.org/abs/2201.10022)

## 这篇论文讲什么

近刚体接触仿真的经典之作。核心思想极其简洁:刚体模型的真正优势是**紧凑坐标**而非**刚性约束**——把每个物体的严格刚性(SE(3))放松为 12 自由度仿射变换 + 一个刚硬的正交惩罚势($\kappa\nu\|AA^T-I\|_F^2$),轨迹就从分段刚性(弯曲)变成分段线性,IPC 管线中最昂贵的曲线 CCD 退化为标准线性 CCD。配合 i-AABB 剔除与接触感知的并行 Hessian 装配,在保留 IPC 全部无穿插/收敛保证的前提下,比 rigid-IPC 快 2–3 个数量级(CPU),GPU 上达 10,000×;并在 28 齿轮 245 万三角形等所有现有刚体方法(rigid-IPC/Bullet/MuJoCo)全部失败的场景中稳定完成仿真。

- **难度**:中高级(核心思想一页纸能讲清,但需要 IPC/barrier 接触与隐式积分背景)
- **性质**:方法/系统类(仿真框架),数学推导轻、工程设计重
- **预计学习时间**:核心理解 1.5–2 小时;含代码演示与自测 3–4 小时

## 如何使用这套材料

建议阅读顺序:

1. **summary.md** — 背景、贡献、定量结果(10 分钟建立全景)
2. **insights.md** ⭐ — 核心洞见:为什么"放松刚性"能赢、概念转变、权衡(最重要,精读)
3. **method.md** — 方法细节:运动学/正交势/affine CCD/i-AABB/Hessian 装配/关节约束(动手前必读)
4. **mental-model.md** — 知识地图:前置知识、在 rigid body → IPC → rigid-IPC → ABD 脉络中的位置
5. **code/** — 两个可运行的 Python 演示(2D 最小 ABD 实现 + 正交势能量地形)
6. **index.html** — 交互式探索器(在浏览器打开,拖动 κ/Δt 看性能与刚度权衡,浏览真实实验数据)
7. **qa.md** — 15 道分级自测题(基础/进阶/高级)
8. **reflection.md** — 研究者视角:扩展方向与脆弱假设

## 关键要点(TL;DR)

1. 刚性约束 ⇒ 弯曲轨迹 ⇒ curved CCD 是 rigid-IPC 的瓶颈;仿射松弛 ⇒ 线性轨迹 ⇒ 线性 CCD。
2. 12 DOF/体,常数 Jacobian、常数质量矩阵、无 Coriolis 力 ⇒ 标准隐式积分直接可用。
3. stiff 正交势($\kappa>100$ GPa)看似危险,但接触 barrier 已主导系统刚度,边际代价≈0。
4. 关节约束经"虚拟四面体"视角变成线性等式约束;混合刚柔仿真同框架内直接做。
5. 复杂场景 90% 加速来自:迭代数少 + CCD 便宜;i-AABB 与并行装配再补一刀。
6. 全保证保留:无穿插、收敛、精确摩擦;可微、GPU 友好、无需凸分解和逐场景调参。

## 文件结构

```
affine-body-dynamics/
├── README.md              ← 本文件
├── meta.json              ← 元数据(标题/作者/标签)
├── FACTS.md               ← 论文事实速查表(所有数字的出处)
├── summary.md             ← 摘要:背景/贡献/结果
├── insights.md            ⭐ 核心洞见(最重要)
├── method.md              ← 方法细节与伪代码
├── mental-model.md        ← 知识地图与前置知识
├── qa.md                  ← 15 道自测题
├── reflection.md          ← 扩展方向与开放问题
├── index.html             ← 交互式探索器
├── paper.pdf              ← 论文原文
├── fulltext.txt           ← 提取文本(p1–8)
├── fulltext-p9-end.txt    ← 提取文本(p9–14)
├── code/                  ← 可运行 Python 演示
│   ├── abd_2d_falling_boxes.py
│   └── orthogonality_potential_demo.py
└── images/                ← 论文图片
```
