# CPU 教学演示

```bash
python3 -c 'import torch; print(torch.__version__)'
python3 /home/ligo/claude-papers/papers/supermimic-differentiable-world-model/code/differentiable_control_loop.py
```

已有 PyTorch 即可运行，只使用 CPU。默认种子为 7，输出结构化 JSON；可用 `--output /绝对路径/result.json` 保存结果。`--iterations` 改变教学训练次数，不对应论文训练时长。

程序用一维受控系统代替 G1。环境的衰减系数随上下文变化，多个目标共享同一动力学。每次循环采集实际执行的含噪动作，用 8 步自回归窗口拟合动力学，然后在冻结模型参数的条件下展开 32 步并优化策略。模型参数在下次循环继续训练。

运行会检查自动微分与中心差分一致、冻结参数不阻断输入梯度、`detach()` 会切断状态梯度，以及默认训练后真实教学环境的跟踪误差下降。另给一个解析反例，说明预测数值接近不代表 Jacobian 接近。

这个例子省略接触、观测历史、终止头和学习式 critic。环境、奖励、模型与训练数字都是教学构造，不能用于说明论文的人形机器人性能。论文中的 8、32、0.99 是唯一沿用的训练配置数字。所有模型从头创建，不使用官方网页 demo 的代码、权重或输出。
