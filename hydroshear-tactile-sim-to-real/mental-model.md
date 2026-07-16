# 心智模型 — 如何在脑中安放 HydroShear

## 这是什么类型的问题？

**"传感器仿真保真度"问题**，具体是：给 RL 策略提供的触觉观测通道，在仿真里应该怎么造，才能让策略学到的"触觉→动作"映射在真机上仍然成立。

它不是接触求解器论文（不解动力学方程组），不是触觉硬件论文，也不是新 RL 算法论文（RL 部分全部沿用 TacSL 的 AACD + PPO）。**全部创新集中在观测模型 M 上。**

## 预备知识清单

| 领域 | 需要什么 | 深度 |
|---|---|---|
| 机器人学 | SE(3) 位姿、坐标系变换 ᴱXᴵ | 熟练 |
| 接触力学 | 库仑摩擦、stick-slip、法向/切向分解 | 概念级 |
| 几何表示 | SDF（符号距离函数）、水密网格 | 概念级 |
| 触觉传感 | GelSight 原理（弹性膜+相机+marker 光流） | 概念级 |
| RL | PPO、teacher-student 蒸馏、非对称 actor-critic | 概念级 |
| 数值方法 | 增量 return mapping（塑性力学）、拉格朗日 vs 欧拉视角 | 加分项 |

## 在研究地图中的位置

### 触觉仿真谱系（保真度 ↑ / 速度 ↓ 排列）

```
FEM (Narang等)          ── 最准，太慢，RL 不可用
  ↑
DiffTactile/Taxim 加速FEM ── 折中
  ↑
★ HydroShear             ── 路径依赖 + SE(3) + stick-slip；33-104ms
  ↑
FOTS                     ── 阴影✓ 但 SE(2)-only、物体中心参考；重实现后 ~12ms
  ↑
TacSL penalty-based      ── 瞬时穿透×速度，无阴影无历史
  ↑
Tacto 刚体接触           ── 最粗
```

HydroShear 占据"RL 可用速度区间内的保真度上界"这个生态位。

### 触觉 sim-to-real 策略谱系

```
表征选择:  触觉RGB图像(TacSL Gray) / 归一化剪切(TacSL Shear, VitaScope) /
          剪切场(FOTS, ★HydroShear) / 离散触觉(inhand translation)
迁移策略:  域随机化 + 零样本(★) / 真实数据微调 / real-to-sim 修正(SCORE 路线)
```

### 与我已读论文的关系图

```
[Stable PD]──执行器──┐
[ABD/FLASH]──接触/仿真基建──┼──> 机器人操作 sim-to-real 大图
[MagicSim]──交互基础设施──┤
[SILR-Cloth]──simulator-in-the-loop(在线)──┤
[SCORE]──real-to-sim-to-real(数据修正)──┤
[Embodied-R1.5]──上层认知/规划──┘
★HydroShear──传感器观测模型(离线RL训练)──┘
```

共同主题：**仿真在机器人学习中的角色正在从"渲染世界"细化为"渲染策略所需的最小充分信息"**。HydroShear 说明对触觉而言这个最小充分信息是"路径依赖的剪切场"，不是 RGB 像素。

## 分类心智标签

- 📂 触觉仿真 → 剪切/力场模型（非图像渲染）
- 📂 sim-to-real → 观测通道保真度（非域随机化技巧）
- 📂 接触模型 → 水弹性 + SDF + 增量摩擦 return mapping
- 📂 RL 基建 → 观测模型即插件（engine-agnostic, GPU-parallel）

## 快速回忆钩子（3 个问题）

1. 为什么叫 non-holonomic？→ 剪切场依赖位姿**路径**而非终点（式 5 递归）。
2. stick-slip 哪来的？→ 切向力累积 + 库仑锥 min-截断（式 11）+ 脱离重置（式 12）。
3. 为什么赢？→ 四个任务各卡一种基线缺陷（幅值/方向/历史/滑移），只有它全保留；93% vs 34–61%。
