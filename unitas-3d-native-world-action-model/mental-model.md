# 心智模型：把世界、动作和后果放进一个物理坐标系

## 它是什么问题？

UNITAS 是一个 3D-native world action model，目标是同时学习：

- 观察到的物理状态；
- 机器人或人手如何运动；
- 运动会怎样改变环境；
- 如何把预测结果变成可执行 action chunk。

它连接 WAM、3D manipulation、flow-matching policy 和 world-model evaluation。

## 一张图

    RGB / RGB-D + calibration + instruction + action history
                              ↓
                 world-aligned visual tokens
                              ↓
        point dynamics expert ─────── action expert
          A: action flow                  u: executable action
                              ↓
                     S: scene flow
                              ↓
             policy execution / simulator consequence

## 在研究地图中的位置

- video WAM：用未来图像或 latent 表示世界；
- VLA：把视觉、语言和动作统一到 policy；
- 3D-aware policy：用深度、点云或轨迹增强几何；
- PointWorld：给定外部动作预测 3D scene flow；
- UNITAS：从 instruction 生成 action，同时以同一 metric frame 预测动作后果。

## 怎么分类

可以把 UNITAS 看成：

    shared physical interface
      = world-aligned observation
      + point-level action flow
      + point-level scene flow
      + executable action expert
      + physical-time tokenization

## 阅读时的三个问题

1. 这个结果是 policy mode，还是 simulator/world-model mode？
2. 指标是在看 action success、scene displacement，还是 action–scene consistency？
3. 使用了 depth、RGB routing、camera calibration 和哪一种 query density？

这三个问题决定了不同实验能不能直接比较。

