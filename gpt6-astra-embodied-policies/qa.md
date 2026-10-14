# 自测题

## 基础题

### 1. 论文为什么区分 Direct 和 Hybrid？

<details>
<summary>答案</summary>

Direct 让 Astra 自己生成机器人命令；Hybrid 让 Astra 审核或修正 learned policy / whole-body controller 的 proposal。两者把不同的 physical responsibility 分给了 Astra，所以结果不能直接混为“模型成功率”。

</details>

---

### 2. RoboDojo Hybrid 的成功率是多少？

<details>
<summary>答案</summary>

在 10 个任务、每任务 5 个 paired instances 上，Hybrid 成功 24/50，即 48%。对应 native Score 是 62.60；这是该评测子集和该 Hybrid 控制路径下的结果。

</details>

---

### 3. RoboCasa Hybrid 的 38.7% 表示什么？

<details>
<summary>答案</summary>

它是 15 个任务、每任务 5 个 episodes、共 75 个 shared starts 上的 29/75 成功率。它不是 38.7 FPS，也不是所有 RoboCasa365 任务的全量 benchmark 结果。

</details>

---

### 4. 论文在哪个导航集合上达到 92%？

<details>
<summary>答案</summary>

在固定 50 episodes 的 VLN-CE RxR English-guide Val-Unseen 子集上，Astra 成功 46/50，即 92%，SPL 为 77.25。

</details>

---

### 5. locomotion 最后一次 Astra attempt 的结局是什么？

<details>
<summary>答案</summary>

30 秒 time horizon 内没有到达目标，前进 2.598 m，最终误差 6.519 m。五次 sequential attempts 都没有到达；PASSAGE 在同一 matched course 上 13.18 秒到达。

</details>

---

## 中级题

### 6. 为什么“85.6% policy actions”不能理解为 Astra 只调用了 14.4% 的次数？

<details>
<summary>答案</summary>

85.6%/14.4% 统计的是已执行 physical control steps 的来源。Astra 仍审核每个 policy proposal，审核会消耗 model calls 和 tokens；它不是 action-step share 与 request share 的同义词。

</details>

---

### 7. dexterous ten-task 的 61.6 是什么指标？

<details>
<summary>答案</summary>

它是 0–100 的 task completion Score，允许 partial credit，十个任务等权平均。它不是 61.6% binary success，也不能与 RoboDojo 的 final success rate 直接比较。

</details>

---

### 8. 为什么 in-hand rotation 能保持抓握却仍然失败？

<details>
<summary>答案</summary>

旋转需要指尖在物体运动过程中重新分配接触。一些 Astra 轨迹没有掉落，但 axial speed 很低，目标旋转没有完成；所以“support retention”和“contact reconfiguration”是两个不同能力。

</details>

---

### 9. RoboLab 中 Direct 为什么比 Hybrid 高？

<details>
<summary>答案</summary>

RoboLab 是单臂、语义 pick-and-place 等任务，Direct 已能从观察直接构造动作，达到 49/50 = 98%。该子集的 π0.5 只有 18/50 = 36%，Hybrid 审核一个不匹配的 proposal 反而会引入额外决策，结果为 46/50 = 92%。

</details>

---

### 10. 为什么 navigation 需要同时看 SR 和 SPL？

<details>
<summary>答案</summary>

SR 只回答是否到达；SPL 还反映成功路径是否高效。HM3D 上 Astra SR 82% 但 SPL 43.69，说明到达能力不错，搜索过程仍有较大 detour。

</details>

---

## 高级题

### 11. 论文的“完整控制路径”观点是什么？

<details>
<summary>答案</summary>

物理动作的意义由 observation、Astra decision、action interface、low-level controller、simulator 和 evaluator 共同决定。同一 Astra 权重更换 grasp interface、gait calibration 或 whole-body controller 后，动作空间和结果都会变。因此应报告完整系统，而不是只归因于模型。

</details>

---

### 12. 5 次 locomotion 失败是否证明 Astra 完全不会运动？

<details>
<summary>答案</summary>

不能。论文显示 Astra 能提高稳定性和前进距离，HumanoidBench 也有不少任务超过 baseline；但在 dense motion reference 的一个 matched course 上，可靠到达目标仍未实现。更准确的结论是“当前接口和延迟下的密集参考生成不可靠”。

</details>

---

### 13. 为什么作者把“physics paused during inference”单独写出来？

<details>
<summary>答案</summary>

因为 30 秒 simulated motion 不等于 30 秒 wall-clock execution。最后一次 locomotion 有 250 个同步 calls，平均 39.86 s/call；如果物理在等待推理，真实闭环时间会被 latency 主导。控制频率、仿真时长和模型决策速度必须分开计时。

</details>

---

### 14. RoboCasa 的探索性 p-value 能证明 Hybrid 更好吗？

<details>
<summary>答案</summary>

不能直接这样说。Hybrid vs π0.5 的 exact McNemar p = 0.0227，但论文明确说明未校正多重比较，也没有处理 task 内依赖和 harness 差异；Hybrid vs Direct 为 0.5235。它支持进一步研究，不构成跨条件的强因果结论。

</details>

---

### 15. 如果要把论文扩展成更可靠的系统，优先改什么？

<details>
<summary>答案</summary>

先建立 action-level provenance 和 latency budget：按 trigger 让 Astra 介入，而不是审核每个 proposal；把 contact mode、uncertainty、proposal source、controller state 和 physics/model clocks 全部记录；再用 paired repeated trials 比较 success、介入率、token、平滑度和恢复时间。这样才能判断收益来自更好的决策，还是来自更强的下游 controller。

</details>

---

