# TapeSim 在解决什么

机器人拉胶带会同时改变自由胶带的形状、胶带卷的位姿和释放前沿的位置。胶带还可能从桌面剥离、黏到新物体上，或被刀片切断。全薄壳表示保留了这些能力，却要反复求解卷内大量几乎一起运动的高刚度界面；整个卷永久刚化又会失去放带能力。

TapeSim 的核心选择是把变形计算集中在确实需要变形的材料上。卷内主体为 rigid cluster；释放前沿附近为可变形 collar；已释放材料为自由薄壳。前沿推进时，从 cluster 中恢复下一段薄壳自由度和原有界面历史。

## 方法贡献

- 按释放前沿推进的刚体/薄壳分区，保留连续放带和重新黏附的能力。
- 为选定的 point–triangle 黏附对提供可断开的 virtual-tetrahedron bond，用单个形状保持能量替代该对的 barrier、摩擦和 cohesive 能量。
- 将放带、贴附和切断接到机器人场景中，用运动回放、冻结策略的真实/仿真配对案例及遥操作封箱演示评估。

这是一种仿真表示与求解组织方法。论文没有提出新的 VLA 结构，也没有用 TapeSim 训练一个可公开复用的策略。

## 四种配置

| 配置 | 胶带–胶带/胶带–卷芯界面 | 卷内表示 |
| --- | --- | --- |
| Coh-F | cohesive contact | 全薄壳 |
| Coh-R | cohesive contact | cluster + collar |
| Bond-F | releasable bond | 全薄壳 |
| Bond-R | releasable bond | cluster + collar |

Peel 的桌面黏附在四种配置中都使用同一组已标定 cohesive 参数。不能把 Bond-R 理解为场景中的所有接触都变成 bond。

## 数字及其范围

| 实验 | 主要数字 | 解释范围 |
| --- | --- | --- |
| 32 圈，固定 Newton 容差 `1e-3 m/s` | bond 家族 366 → 115 ms/step，约 3.2×；cohesive 家族约 3.4× | RTX PRO 6000 Blackwell Server Edition；物理步，不含前沿更新、控制、遥测与渲染 |
| 32 圈，可比卷芯角度范围 | Coh-F → Coh-R：307.77 → 36.73 ms，8.38×；Bond-F → Bond-R：130.43 → 29.14 ms，4.48× | 各模型选不同 tolerance，使角度范围相对 Bond-R 目标相差不超过 5% |
| 32 圈，单独容差实验 | Coh-F 的角度范围 9.7° → 38.1° | 只把容差从 `1e-2` 收紧到 `1e-3 m/s`；支持存在求解器导致的运动抑制 |
| Stick 开环回放 | Coh-F 22.96 px；Bond-F 17.03；Coh-R 17.57；Bond-R 16.28 | 5 个真实 episode，每模型重复 2 次，排除前 3 秒；图像平面卷芯 landmark，不是 3D 全状态误差 |
| Peel 配对预测 | Coh-F agreement/BA 为 69%/50%；Coh-R 为 81%/72.9%；Bond-R 为 82%/76.3% | 100 个配对案例；真实成功 69/100 |
| Stick 配对预测 | Coh-R/Bond-R 的总成功率 67.6%/69.0%，真实为 70.4%；BA 50.7%/51.7% | 71 个案例；总率接近，但个体判别弱。Bond-F 的 BA 为 64.6% |
| Wrap ≥1 圈 | Coh-R agreement/BA 为 76.4%/76.6%；Bond-R 为 72.7%/72.4% | 同一组 55 个案例，真实成功 28/55 |
| Wrap ≥2 圈 | Coh-R agreement 49/55，但 48 个是共同失败；真实成功仅 4/55 | 全部预测失败也能得到 51/55 的 agreement；不能只看 89.1% |
| 遥操作封箱 | 3846 步，128.2 秒仿真用 304.1 秒墙钟，0.422× 实时 | Coh-R、RTX 5090、步长 `1/30 s`，一段连续仿真演示 |

`8.38×`、`4.48×` 和全失败基线 `92.7%` 为本包直接计算的派生值。没有给未报告的固定容差数据补造精确小数。

## 标定与复现条件

胶带宽 19 mm、全厚度 0.18 mm、密度 1300 kg/m³；膜和弯曲的 E 均配置为 50 MPa，泊松比 0.45。每米质量由几何和密度算得 4.446 g/m。注意哪些量实测，哪些是配置值。

桌面剥离力实测 2.5 N。300×19 mm 胶带以 6 mm/s、约 180° 剥离，`Cn=Ct=0.675 N` 时仿真反力中位数为 2.54 N。默认步长 0.01 秒，collar 宽度为 10 条 mesh-graph edges。摆动实验的 bond 参数为 `κb=1e6 Pa`、断裂阈值 0.5 N，前沿门槛 `δ=5 d̂=0.9 mm`。

VLA 是内部基础模型，按任务用遥操作数据微调。训练/验证 episode 数依次为 Peel 284/16、Stick 241/14、Wrap 241/14。策略预测 32 步 action chunk，100 Hz 机器人控制、30 Hz waypoint、3 Hz 重规划。真实相机 60 Hz、仿真相机 30 Hz。这里检验的是冻结策略的结果预测，没有证据支持把这些数字当作纯仿真训练的 sim-to-real 成功率。

## 阅读结论

对这一组卷带任务，cluster 的收益更稳定；Coh 与 Bond 的任务排名不同。减少卷内自由度的依据来自运动结构，界面选择还需要结合标定与任务。可比运动实验只约束了特定角度范围，不能据此断言两种模型在全部接触、张力或断裂行为上等价。

来源：[论文 v2，III–IV 节](https://arxiv.org/html/2609.28766v2)、[作者项目页](https://genesis-embodied-ai.github.io/tapesim-website/)、[作者公开统计](https://genesis-embodied-ai.github.io/tapesim-website/assets/data/evidence.json)。
