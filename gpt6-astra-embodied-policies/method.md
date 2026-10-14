# 方法拆解：Direct、Hybrid 与六域接口

## 1. 组件分工

| 层 | 作用 | 论文中的实例 |
| --- | --- | --- |
| Observation | 给 Astra 可见输入和执行历史 | wrist/head RGB、front RGB、本体状态、任务说明、历史动作 |
| Astra decision | 生成、审核或重写下一段决策 | EEF target、joint/finger target、navigation primitive、subgoal、motion reference |
| Proposal source | 提供高频候选动作 | π0.5、任务特定 RL、ScaleTrack、HumanoidGPT、ScaleBFM |
| Action adapter | 检查范围并翻译到 simulator/robot | IK、operational-space controller、PD、离散 forward/turn/STOP、whole-body interface |
| Feedback | 让下一次决定基于实际结果 | 更新图像、本体状态、执行历史、成功/失败 predicate |
| Evaluator | 判定结果和统计边界 | success、partial Score、native return、SPL/nDTW、token/request accounting |

## 2. Direct 与 Hybrid

### Direct

Astra 看到 observation 后直接产生动作命令：

- gripper：bimanual EEF target 和 gripper command，执行 1–5 个控制步；
- dexterous：wrist/finger target；
- mobile manipulation：arm、gripper、base command；
- navigation：forward、turn、STOP；
- locomotion：joint PD target 或 dense motion reference。

Direct 的优点是责任链短、任务适配灵活；缺点是每次数值输出都要承担连续控制和接触稳定性。

### Hybrid

Hybrid 先接收 learned policy/controller 的 candidate：

1. Astra 判断 proposal 是否符合当前目标和物理状态；
2. 接受一个 prefix，或生成 1–5 步局部 correction；
3. 读取执行后的反馈；
4. 继续审核、恢复、切换 subgoal，或在预算耗尽时退回 policy-only。

在 RoboDojo 的 Hybrid 中，π0.5 提议 50-step joint-space sequence，Astra 可以执行前 1–15 步，或改成 1–5 步 EEF correction；两个分支互斥。这个设计让 Astra 的低频判断和 policy 的高频 motion prior 分工。

## 3. 六域的接口变化

    gripper / dexterous:
      image + proprio + instruction
            ↓
      EEF / joint / finger targets
            ↓
      IK or operational-space controller

    mobile manipulation:
      image + policy proposal + subgoal
            ↓
      accepted prefix / correction / rewritten subgoal
            ↓
      arm + base controller

    navigation:
      RGB only
            ↓
      discrete forward / turn / STOP
            ↓
      Habitat navigation primitive

    locomotion:
      observation + goal
            ↓
      joint PD target or dense reference
            ↓
      PD / frozen ScaleTrack

    humanoid loco-manipulation:
      visual + proprio + task command
            ↓
      velocity / sparse body target
            ↓
      frozen whole-body controller

## 4. 统一反馈伪代码

    state = reset(task, seed)
    history = []

    while not terminal and budget_left:
        obs = observe(state, history)

        if mode == DIRECT:
            command = astra.generate(obs, instruction)
            source = "astra"
        else:
            proposal = policy.propose(obs, instruction)
            decision = astra.review(obs, proposal, history)
            if decision.accept:
                command = decision.prefix
                source = "policy"
            else:
                command = decision.correction
                source = "astra"

        command = interface.check_bounds_and_translate(command)
        state = environment.step(command)
        history.append(obs, command, source, state)

    return evaluator.measure(history)

这段伪代码表达控制责任，不代表论文公开了统一的软件实现。实际研究按域使用不同 observation、prompt、action rate、horizon 和 evaluator。

## 5. 关键统计

- **Binary success**：例如 RoboDojo 24/50、RoboCasa 29/75、DexJoCo 5/10。
- **Partial Score**：例如 dexterous ten-task mean 61.6；分数表示 subgoal progress，不等同成功率。
- **Native return**：HumanoidBench 任务有不同阈值，不能跨任务直接比较柱高。
- **Path efficiency**：SPL/nDTW/sDTW 同时观察“到达”和“路径是否接近参考”。
- **Source share**：85.6% policy actions 与 14.4% Astra corrections 是已执行控制步占比，不是 model call 占比。
- **Resource**：token、retained requests、latency 必须和 physical/controller clock 分开。

## 6. 实现陷阱

1. 把一个 Hybrid 分数归因给 Astra，而不报告 policy/controller。
2. 把 partial Score 写成 success rate。
3. 把 model calls 或 interactions/s 写成 robot FPS。
4. 用同一平均值合并不同 observation regime、task split 和 evaluator。
5. 只统计成功轨迹，忽略 budget exhaustion、failed STOP、fall 和 unfinished subgoal。
6. 看到一次 online diagnosis 成功就称为 online learning。
7. 忽略 prompt、notes、action interface、controller calibration 改变了系统条件。

## 7. 超参数与预算敏感性

论文中的敏感变量不是单纯的 neural learning rate，而是：

- action prefix 长度和 correction 长度；
- Astra decision budget、token budget、context compaction；
- policy review cadence；
- observation history 和 camera regime；
- simulator horizon、control frequency、controller calibration；
- whole-body reference 的维度和落地方式。

复现时必须把这些配置与模型权重一起记录，否则相同模型名并不代表相同控制系统。

