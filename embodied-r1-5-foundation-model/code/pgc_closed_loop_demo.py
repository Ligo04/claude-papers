#!/usr/bin/env python3
"""
pgc_closed_loop_demo.py
================================================================================
Embodied-R1.5 的 **Planner-Grounder-Corrector (PGC) 闭环框架**（论文 §5）模拟。

核心思想（论文最深刻的一句）：
  单个 Embodied-R1.5 实例**异步扮演全部三个角色**（规划/定位/纠错），
  框架本身是**无状态、无推理逻辑、无启发式规则**的"笨"外壳，只负责控制流
  调度与极简 FIFO Memory；**所有智能决策都来自模型内化能力**。
  → 系统天花板只由模型能力决定，随模型 scaling 自然提升。

本 demo 用一个**状态机**复现论文 Figure 3 的"做奶茶"例子：
  - Planner 把长程任务分解为子任务序列；
  - Grounder 为每个子任务自适应编排指向能力(OFG→定位功能部件, VTG→规划轨迹)；
  - Corrector 异步判定 SUCCESS / PROCESS / FAIL；
  - 失败写入 Memory → 反馈 Planner → 触发重试/重规划。
  - 复现论文细节："press the tea dispenser switch"**第一次失败、第二次成功**。

注意：这里的"模型"是**脚本化的桩(stub)**，用来展示**框架的控制流**，
      而非真实推理。真实系统里 plan()/ground()/correct() 都是模型前向。

依赖：仅 Python 标准库。运行：python pgc_closed_loop_demo.py
================================================================================
"""

from collections import deque
from dataclasses import dataclass


# ===========================================================================
# 极简 FIFO Memory（论文：minimal first-in-first-out buffer）
# ===========================================================================
class FIFOMemory:
    def __init__(self, maxlen=6):
        self.buf = deque(maxlen=maxlen)

    def push(self, record):
        self.buf.append(record)

    def recent(self):
        return list(self.buf)

    def last_status(self, subtask):
        for rec in reversed(self.buf):
            if rec["subtask"] == subtask:
                return rec["status"]
        return None


# ===========================================================================
# "模型"桩：同一个实例提供 plan / ground / correct 三个能力接口
#   —— 模拟"单模型异步扮演三角色"。真实系统中这三者都是 Embodied-R1.5 前向。
# ===========================================================================
@dataclass
class EmbodiedR15Stub:
    """脚本化的能力接口。用确定性规则模拟模型行为，便于复现论文叙事。"""

    # ---- Planner：长程任务分解 ----
    def plan(self, instruction, sop=None):
        # 论文 Figure 3 的奶茶 SOP（10 步，这里取核心几步演示）
        return [
            "Move the cup to the coaster beneath the tea dispenser",
            "Press the tea dispenser switch",
            "Move the cup to the coaster beneath the milk dispenser",
            "Grasp the stirring rod",
            "Stir the drink inside the cup",
            "Place the cup in the pickup area",
        ]

    # ---- Grounder：自适应编排指向能力（OFG / VTG）----
    def ground(self, subtask):
        """根据子任务需求，自主决定调用哪些指向接口。"""
        s = subtask.lower()
        calls = []
        if "press" in s or "switch" in s or "button" in s:
            calls = ["OFG(定位开关功能部件)", "VTG(规划按压轨迹)"]
        elif "grasp" in s or "stirring rod" in s:
            calls = ["OFG(定位把手)", "VTG(规划抓取轨迹)"]
        elif "move" in s or "place" in s:
            calls = ["REG(定位目标物)", "RRG(定位放置区域)", "VTG(规划搬运轨迹)"]
        elif "stir" in s:
            calls = ["VTG(规划搅拌环形轨迹)"]
        else:
            calls = ["REG(定位目标)"]
        return calls

    # ---- Corrector：异步判定执行状态 ----
    def correct(self, subtask, exec_result, memory):
        """SUCCESS / PROCESS / FAIL。可复用 memory 中的历史上下文做错误归因。"""
        if exec_result["still_running"]:
            return "PROCESS", None
        if exec_result["ok"]:
            return "SUCCESS", None
        # 失败：基于观测做错误归因（这里用脚本化的 reason）
        reason = exec_result.get("reason", "execution deviation detected")
        return "FAIL", reason

    # ---- Planner：失败后的自适应重规划 ----
    def replan(self, subtask, reason, memory):
        # 复用 Corrector 的归因，给出修正建议（论文：error info 反馈给 Planner）
        return f"retry '{subtask}' with correction: {reason}"


# ===========================================================================
# 低层技能执行器 + 脚本化的失败注入（复现 Figure 3 叙事）
# ===========================================================================
class ScriptedExecutor:
    """
    脚本化执行：复现论文 Figure 3——"press the tea dispenser switch"
    第一次因'位置偏移(position offset from the switch)'失败，第二次成功。
    其余子任务一次成功。
    """
    def __init__(self):
        self.attempts = {}

    def execute(self, subtask, grounding):
        self.attempts[subtask] = self.attempts.get(subtask, 0) + 1
        n = self.attempts[subtask]
        # 注入：按开关首次失败（位置偏移），重试成功
        if "Press the tea dispenser switch" == subtask and n == 1:
            return {"ok": False, "still_running": False,
                    "reason": "position offset from the switch (按压点偏移~2cm)"}
        return {"ok": True, "still_running": False}


# ===========================================================================
# PGC 闭环外壳：无推理逻辑，只做控制流调度 + Memory 管理
# ===========================================================================
def pgc_loop(instruction, model, executor, sop=None, max_retries=3, verbose=True):
    memory = FIFOMemory(maxlen=6)
    log = []

    def emit(role, msg):
        line = f"  [{role:<9}] {msg}"
        log.append(line)
        if verbose:
            print(line)

    emit("USER", f'指令: "{instruction}"')
    plan = model.plan(instruction, sop=sop)               # ← Planner
    emit("PLANNER", f"任务分解为 {len(plan)} 个子任务:")
    for i, st in enumerate(plan, 1):
        emit("PLANNER", f"   {i}. {st}")
    print()

    stats = {"subtasks": len(plan), "executions": 0, "failures": 0, "recoveries": 0}

    for i, subtask in enumerate(plan, 1):
        emit("SCHED", f"── 子任务 {i}/{len(plan)}: {subtask}")
        retries = 0
        while True:
            grounding = model.ground(subtask)             # ← Grounder
            emit("GROUNDER", f"自适应编排: {' + '.join(grounding)}")

            exec_result = executor.execute(subtask, grounding)
            stats["executions"] += 1

            status, reason = model.correct(subtask, exec_result, memory)  # ← Corrector
            memory.push({"subtask": subtask, "status": status, "obs": exec_result})

            if status == "SUCCESS":
                emit("CORRECTOR", "状态=SUCCESS ✅ → 推进下一子任务")
                break
            elif status == "PROCESS":
                emit("CORRECTOR", "状态=PROCESS ⏳ → 继续等待")
                continue
            else:  # FAIL
                stats["failures"] += 1
                emit("CORRECTOR", f"状态=FAIL ❌ 归因: {reason}")
                emit("MEMORY", "失败信息写入 Memory → 反馈 Planner")
                retries += 1
                if retries > max_retries:
                    emit("SCHED", f"超过最大重试 {max_retries} 次，放弃该子任务")
                    break
                correction = model.replan(subtask, reason, memory)  # ← Planner 重规划
                emit("PLANNER", f"重规划: {correction}")
                emit("SCHED", f"↻ 自主重试 (第 {retries+1} 次尝试)")
                stats["recoveries"] += 1
        print()

    return stats, memory


def main():
    print("=" * 78)
    print(" Embodied-R1.5 · PGC 闭环框架 demo（复现论文 Figure 3 '做奶茶'）")
    print("=" * 78)
    print(" 关键：以下 PLANNER/GROUNDER/CORRECTOR 全部由**同一个模型实例**异步提供；")
    print("       外壳(SCHED/MEMORY)无任何推理逻辑，只做调度与 FIFO Memory 管理。")
    print("=" * 78 + "\n")

    model = EmbodiedR15Stub()
    executor = ScriptedExecutor()
    stats, memory = pgc_loop(
        instruction="Can you make me a cup of milk tea?",
        model=model, executor=executor,
        sop="Milk Tea Preparation Workflow (10 steps)",
    )

    print("=" * 78)
    print(" 执行统计")
    print("=" * 78)
    print(f"   子任务总数      : {stats['subtasks']}")
    print(f"   执行次数        : {stats['executions']}  (含重试)")
    print(f"   检测到的失败    : {stats['failures']}")
    print(f"   自主纠错恢复    : {stats['recoveries']}")
    print(f"   最终            : 全部完成 ✅ (无人工干预)")
    print()
    print(" Memory (最近 FIFO 记录):")
    for rec in memory.recent():
        print(f"   - {rec['status']:<8} {rec['subtask']}")
    print()
    print("=" * 78)
    print(" 复现要点：'Press the tea dispenser switch' 首次因'位置偏移'被 Corrector 判 FAIL，")
    print("           写入 Memory→反馈 Planner→自主重规划重试→第二次 SUCCESS。")
    print("           整个过程零人工干预——这正是闭环自主纠错(论文 Figure 3)的核心。")
    print("=" * 78)


if __name__ == "__main__":
    main()
