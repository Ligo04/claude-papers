# -*- coding: utf-8 -*-
"""
WALL-WM —— 事件切分 vs 定长 chunk 切分（最小 Demo）
===================================================

WALL-WM 的灵魂一句：“固定 chunk 按时钟切，语义事件按具身动态切。”
本仅依赖 NumPy 的 Demo 用一个一维玩具轨迹，量化展示二者的区别：

  - 我们造一段“伸手→抓取→移动→放置”的合成机器人轨迹，每个阶段（事件）有
    自己的运动特征（速度/接触信号），事件边界处特征发生“突变”。
  - 定长 chunk：每 L 帧切一刀（按时钟）。一个 chunk 常常跨越事件边界，
    内部混入两种不同动态 → “块内不一致”高 → 退化成短时相关性拟合。
  - 事件切分：按动态突变（具身关节）切。每段内部一致、边界干净。

我们用两个可度量的指标说明为什么事件单位更好：
  (A) 块内动态不一致度（越低越好）：一个块若跨越事件边界，内部混合多种动态。
  (B) “需要历史才能消歧”的块比例（越低越好）：定长 chunk 常需历史上下文
      才能确定它属于哪个语义事件（论文 Fig.2 的观点）。

运行：
    python event_vs_chunk_segmentation_demo.py
"""

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1) 合成一段多事件机器人轨迹
# ---------------------------------------------------------------------------
# 每个事件 = (名称, 时长帧数, 该事件的“动态签名”：平均速度 + 是否接触)
EVENTS = [
    ("reach  伸手", 22, dict(vel=+1.0, contact=0)),
    ("grasp  抓取",  9, dict(vel=+0.05, contact=1)),
    ("move   移动", 26, dict(vel=-0.8, contact=1)),
    ("place  放置", 13, dict(vel=+0.03, contact=1)),
    ("release释放",  6, dict(vel=-0.2, contact=0)),
]


def synth_trajectory():
    """返回逐帧 (velocity, contact) 以及每帧真实事件 id 与事件边界。"""
    vel, con, ev_id = [], [], []
    boundaries = [0]
    for i, (_, dur, sig) in enumerate(EVENTS):
        for _ in range(dur):
            vel.append(sig["vel"] + rng.normal(0, 0.05))
            con.append(sig["contact"])
            ev_id.append(i)
        boundaries.append(len(vel))
    return np.array(vel), np.array(con), np.array(ev_id), boundaries


# ---------------------------------------------------------------------------
# 2) 两种切分
# ---------------------------------------------------------------------------
def fixed_chunks(T, L):
    """定长 chunk：每 L 帧一段（按时钟）。返回 [(start,end), ...]"""
    return [(s, min(s + L, T)) for s in range(0, T, L)]


def event_chunks(boundaries):
    """事件切分：按事件边界（按具身动态的关节）。"""
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# 3) 指标
# ---------------------------------------------------------------------------
def intra_inconsistency(segs, ev_id):
    """块内动态不一致度：每段内含多少种不同事件（>1 即跨界），按帧加权平均。
    完美事件切分 = 1.0（每段恰好一个事件）。"""
    total = 0.0
    for s, e in segs:
        n_events = len(np.unique(ev_id[s:e]))
        total += n_events * (e - s)
    return total / sum(e - s for s, e in segs)


def frac_need_history(segs, ev_id):
    """需要历史消歧的块比例：若一个块的起始帧所属事件，与该块“多数事件”不一致，
    则单看这个块无法确定它表达什么语义事件，需要历史上下文（论文 Fig.2 观点）。"""
    need = 0
    for s, e in segs:
        ids = ev_id[s:e]
        majority = np.bincount(ids).argmax()
        if ids[0] != majority or len(np.unique(ids)) > 1:
            need += 1
    return need / len(segs)


def main():
    print(__doc__)
    vel, con, ev_id, bnd = synth_trajectory()
    T = len(vel)
    print(f"合成轨迹：{T} 帧，{len(EVENTS)} 个语义事件，边界在 {bnd}")
    print(f"事件：{[e[0] for e in EVENTS]}\n")

    print("=" * 64)
    print("定长 chunk（按时钟切）—— 不同 chunk 长度 L：")
    print("-" * 64)
    print(f"  {'L':>3} | {'块内不一致度':>12} | {'需历史消歧块占比':>16}")
    for L in [8, 12, 16, 24]:
        segs = fixed_chunks(T, L)
        ic = intra_inconsistency(segs, ev_id)
        fh = frac_need_history(segs, ev_id)
        print(f"  {L:>3} | {ic:>12.3f} | {fh*100:>14.0f}%")

    print()
    print("=" * 64)
    print("事件切分（按具身动态的关节切）：")
    print("-" * 64)
    segs = event_chunks(bnd)
    ic = intra_inconsistency(segs, ev_id)
    fh = frac_need_history(segs, ev_id)
    print(f"  事件 | {'块内不一致度':>12} | {'需历史消歧块占比':>16}")
    print(f"  {len(segs):>4} | {ic:>12.3f} | {fh*100:>14.0f}%")

    print()
    print("结论：")
    print("  • 定长 chunk 的块内不一致度 > 1（很多 chunk 跨越事件边界，内部混入两种动态），")
    print("    且相当比例的 chunk 需要历史上下文才能确定其语义 → 退化成短时相关性拟合。")
    print("  • 事件切分块内不一致度 = 1.000（每段恰好一个语义事件），无需历史消歧。")
    print("  这正是 WALL-WM 的主张：在‘事物的天然关节处切分’，让语言/视频/动作对齐到")
    print("  同一个原子单位，从而保留预训练先验、增强组合性与长程泛化（论文 §1、Fig.2）。")


if __name__ == "__main__":
    main()
