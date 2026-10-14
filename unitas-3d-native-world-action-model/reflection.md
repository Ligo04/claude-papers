# 反思与可延伸方向

## 如果继续做 UNITAS

1. 把 action-only、action+flow、action+both-flow 的 latency 与 manipulation success 放在同一条 Pareto 曲线上。
2. 对真实机器人补充定量 scene-flow ground truth，区分 geometry prediction error 与 robot execution error。
3. 把 world-frame 作为显式 safety interface：在生成 action chunk 前检查 reachability、collision 和 contact feasibility。
4. 研究更长 physical-time windows，验证 3 s tokenizer 是否能稳定拼接长时程任务。
5. 对 action-flow history 做 adaptive memory，处理重规划、回退和重复尝试。
6. 在更多不共享相机视角、gripper morphology 的 embodiment 上评估 cross-embodiment transfer。

## 脆弱假设

- camera calibration 和 robot-base/world frame 足够可靠；
- 20 个 gripper points 能覆盖关键接触几何；
- scene query sampling 不会漏掉关键物体；
- 3 s physical-time window 足够表示动作语义；
- policy mode 的 action chunk 不需要显式场景后果预测；
- simulation scene-flow supervision 能迁移到真实环境。

## 可能失败的场景

- 大遮挡、透明物体、反光物体导致 world grounding 错误；
- 接触变化发生在 query points 之外；
- 长时间堆叠误差让 action history 过时；
- scene flow 正确但 end-effector action 不可执行；
- 真实传感器缺 depth 且 camera calibration 漂移；
- 高 query density 让 world-model mode 无法满足闭环 latency。

## 可检验的新假设

如果在 policy mode 中只保留“与当前 action uncertainty 或 contact change 相关的 scene queries”，可以用明显低于 3000 queries 的成本保留 action–scene consistency。应同时报告成功率、Moving ADE/FDE、Acc@1cm、query 数和 wall-clock。

