# 运行二维接触流形演示

只需要 Python 3.10 或更新版本，无第三方依赖。

```bash
python3 /home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/code/contact_manifold_demo.py --self-check
```

默认在脚本旁的 `output/` 生成 `contacts.json` 和 `contact_manifold.svg`。查看每帧候选数、激活数、命中左右区域、目标评估次数和点坐标。一次目标评估查两个原始场；计数不包含事后的接触数据合成和种子预处理。

仅改变采样策略：

```bash
python3 /home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/code/contact_manifold_demo.py \
  --sampling uniform --output-dir /home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/code/output-uniform
```

关闭缓存，并把每帧新搜索数设为 5，以保留默认配置每步最多 5 次局部优化的预算：

```bash
python3 /home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/code/contact_manifold_demo.py \
  --cache 0 --random 5 --output-dir /home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/code/output-no-cache
```

教学场景使用带两个支脚的组合形状缓慢平移，地面保持静止。形状、运动、随机种子和输出计数都来自本地教学设置。代码实现平滑最大值、排斥、有界二维 Nelder–Mead、曲率特征种子、速度预测、去重和缓存选择。

额外数值选择是把恰在搜索域边界的种子向盒中心移动 10%，防止轴向 simplex 经投影后退化。它不是论文中已声明的处理。二维曲率评分用标量场的二阶差分近似，区别于原文对梯度的差分，但对应同一个 Laplacian 定义。

`--self-check` 检查距离场基本值、平滑上界、浅重叠反例、有界优化器的已知二次函数解和曲率种子存在性。它不验证真实刚体稳定性。

本演示没有动力学、摩擦冲量、真实刚体转动、CCD、体素输入或 GPU 并行。不要用演示的查询计数或运行速度替代论文表格；一次随机种子下命中两个区域也不意味着完备性。
