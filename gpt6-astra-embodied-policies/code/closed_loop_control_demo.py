#!/usr/bin/env python3
"""教学 demo：把论文里的 Direct / Hybrid 责任分工变成一个可运行的 toy 闭环。

这不是 GPT-6 Astra、π0.5、RoboDojo 或 RoboCasa 的复现。
它只用确定性状态和论文公开的汇总数字，演示：
1. policy 先给 proposal；
2. review 只在观测到偏差时接受或局部修正；
3. 记录每个控制步的 source；
4. 把 step-source share 与 model-request cost 分开。
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path


@dataclass
class ToyState:
    position: float = 0.0
    target: float = 1.0
    contact_ok: bool = False


@dataclass
class StepRecord:
    step: int
    proposal: float
    command: float
    source: str
    position: float
    contact_ok: bool


def policy_proposal(state: ToyState, step: int) -> float:
    """模拟 learned policy：大多数时间方向正确，但会在接触前过冲。"""
    remaining = state.target - state.position
    if step in {2, 6}:
        return remaining + 0.30
    return 0.20 if remaining > 0 else 0.0


def astra_review(state: ToyState, proposal: float) -> tuple[str, float]:
    """模拟 Hybrid review：接受安全 proposal，否则只做局部 correction。"""
    next_position = state.position + proposal
    if next_position > state.target + 0.05:
        correction = max(0.0, state.target - state.position)
        return "astra_correction", correction
    return "policy", proposal


def run(mode: str, steps: int = 10) -> list[StepRecord]:
    state = ToyState()
    records: list[StepRecord] = []

    for step in range(steps):
        if mode == "direct":
            proposal = policy_proposal(state, step)
            command = proposal
            source = "astra"
        elif mode == "hybrid":
            proposal = policy_proposal(state, step)
            source, command = astra_review(state, proposal)
        else:
            raise ValueError("mode must be direct or hybrid")

        state.position += command
        state.contact_ok = 0.35 <= state.position <= 0.85
        records.append(
            StepRecord(
                step=step,
                proposal=round(proposal, 4),
                command=round(command, 4),
                source=source,
                position=round(state.position, 4),
                contact_ok=state.contact_ok,
            )
        )

        if state.position >= state.target:
            break

    return records


def summarize(records: list[StepRecord]) -> dict[str, object]:
    source_counts: dict[str, int] = {}
    for record in records:
        source_counts[record.source] = source_counts.get(record.source, 0) + 1

    final_position = records[-1].position if records else 0.0
    success = abs(final_position - 1.0) < 0.05 and any(r.contact_ok for r in records)
    return {
        "steps": len(records),
        "final_position": final_position,
        "success": success,
        "executed_step_sources": source_counts,
        "paper_reference": {
            "robodojo_hybrid_success_percent": 48.0,
            "robodojo_hybrid_policy_step_share_percent": 85.6,
            "robodojo_hybrid_astra_step_share_percent": 14.4,
            "note": "这些数字来自论文；toy trajectory 不是论文评测。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "hybrid"), default="hybrid")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    records = run(args.mode, args.steps)
    result = {"mode": args.mode, "records": [r.__dict__ for r in records], "summary": summarize(records)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

