# 反思与可延伸方向

## 如果继续做这项工作

1. 把 Astra 的审核频率、proposal prefix 长度和 token budget 做成受控变量，报告 success–latency–cost 三维曲线。
2. 在相同初始状态上对 Direct、Hybrid、policy-only 做 paired evaluation，并按 task family 分层，避免总体平均掩盖接触失败。
3. 对每次介入记录“模型改变了哪个可观测事实”：目标、接触、遮挡、顺序、完成判定还是恢复动作。
4. 引入一个独立的 low-level contact controller，再测试 Astra 是否只需提供 subgoal 或 contact mode。
5. 对 repeated trials 做预注册评测，把一次成功、连续成功和跨初始状态泛化分开报告。

## 脆弱假设

- action interface 能把语言模型的数值输出稳定地映射到真实接触。
- 视觉、本体状态和执行历史足以判断隐藏接触状态。
- policy proposal 的质量足够高，审核成本不会超过它带来的收益。
- prompt、notes、controller calibration 的变化可以和模型本身的能力分开。
- 以 50 episodes 或 5 trials 估计的成功率能代表更广分布。
- physics pauses during inference 的仿真结果可以外推到实时机器人。

## 可能失败的场景

- 需要高频手指接触重构、柔顺控制或快速反应的任务；
- 视觉遮挡造成“已抓住/已放稳”的错误判断；
- 长序列中一次小误差不断累积；
- policy 与 Astra 的 proposal 分布不匹配；
- latency 让反馈已经过时；
- evaluator 只看终态，忽略中间阶段的危险动作。

## 一个可检验的扩展假设

“最有效的 Hybrid 不是让 Astra 审核每个 proposal，而是让它只在一个可学习的 uncertainty/contact-change trigger 触发时介入。”验证时应同时报告介入率、token、success、动作平滑度和失败恢复时间。

