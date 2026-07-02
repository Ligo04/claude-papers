# -*- coding: utf-8 -*-
"""
DreamZero —— 联合视频-动作流匹配 + 闭环 KV 替换（最小 Demo）
============================================================

这是一个独立、仅依赖 NumPy 的教学版重实现，对应 DreamZero 的两个核心机制：
  (1) 把动作学习表述为“逆动力学（IDM）”——从*预测出的视觉未来*反解动作；
  (2) 闭环 KV 缓存替换——用*真实观测*替换自回归预测帧，根治复合误差。

为了不用 14B 视频扩散模型就把思想讲清楚，我们用一个一维玩具“世界”：
  - 世界状态 s（标量，想象成“画面”：物体在传送带上的位置）。
  - 动作 a = 速度指令；真实动力学 s_{t+1} = s_t + a_t（这就是我们要隐式学的 IDM 的逆）。
  - “视频”就是状态序列 o = [s_l, s_{l+1}, ..., s_{l+H}]。

DreamZero 的等式（式 1）：联合预测(视频, 动作) = 视频预测 × IDM。
本 Demo 展示：
  A. 用流匹配联合去噪“视频(未来状态)”和“动作”，二者共享去噪时刻（论文设定）。
  B. 一旦模型能预测未来视频，动作就能由相邻帧之差 a_t = o_{t+1} - o_t“读出”（IDM）。
  C. 自回归推出长程时，预测误差会累积（漂移）；闭环用真实观测替换后，漂移被清零。
  D. 大多数失败源于“视频预测错误”，而非“动作抽取错误”（论文关键观察）。

运行：
    python joint_video_action_fm_demo.py
"""

import numpy as np

rng = np.random.default_rng(0)

H = 8          # 每个 chunk 的时域（未来帧数）
SIGMA = 0.05   # 真实执行噪声：动作执行时世界会偏离模型的‘想象’。
               # 正是这种偏离，在开环（模型只信自己的预测）下会累积；
               # 而闭环用真实观测纠偏，才能消除累积 —— 这是本 Demo (C) 的关键。


# ---------------------------------------------------------------------------
# 玩具世界 + 一个“视频预测器”
# ---------------------------------------------------------------------------
# 目标行为：把物体从当前位置匀速送到目标 goal。理想未来视频是一条直线插值。
STEP = 1.0     # 每个 chunk 朝目标推进的上限（让长程任务需要多个 chunk 才能完成，
               # 这样执行噪声才有机会在开环里逐 chunk 累积）。


def chunk_subgoal(s0, goal):
    """一个 chunk 只朝目标推进有限的一步（受 STEP 限制）。"""
    return s0 + np.clip(goal - s0, -STEP, STEP)


def ideal_future_video(s0, goal, H):
    """给定起点与（本 chunk 的）子目标，理想未来‘视频’是线性插值。"""
    return np.linspace(s0, chunk_subgoal(s0, goal), H + 1)


def video_predictor(s0, goal, H, quality=1.0):
    """模拟一个预训练‘视频扩散’骨干：预测本 chunk 的未来帧。
    quality∈[0,1]：1=完美预测；<1 注入预测误差（模拟弱视频骨干/幻觉）。
    这对应论文‘机器人能力 = 视频生成质量’：quality 越低，动作越差。"""
    truth = ideal_future_video(s0, goal, H)
    noise = (1.0 - quality) * rng.normal(0, STEP * 0.15 + 0.05, H + 1)
    noise[0] = 0.0                      # 第 0 帧是已知的当前观测，不加噪
    return truth + noise


# ---------------------------------------------------------------------------
# (B) 逆动力学：从预测视频里“读出”动作
# ---------------------------------------------------------------------------
def extract_actions_via_idm(video):
    """IDM：a_t = o_{t+1} - o_t。动作直接来自预测的视觉未来——
    这正是 DreamZero 的‘动作 = 从视频反解’。"""
    return np.diff(video)


# ---------------------------------------------------------------------------
# (A) 联合流匹配：共享去噪时刻同时去噪‘视频’与‘动作’ latent
# ---------------------------------------------------------------------------
# 约定（与论文式 2 一致）：t=1 干净，t=0 噪声；x_t = t·x1 + (1-t)·x0，速度目标 v = x1 - x0。
def flow_match_denoise(x1_clean, n_steps, shared_seed):
    """对一个干净目标向量做‘流匹配去噪’演示：从噪声沿速度场积分回 x1。
    （这里用解析最优速度 v=x1-x0；真实模型用 DiT 预测 v。）"""
    r = np.random.default_rng(shared_seed)
    x0 = r.standard_normal(x1_clean.shape)     # 纯噪声起点
    x = x0.copy()
    v = x1_clean - x0                          # 常速度（解析最优）
    ts = np.linspace(0.0, 1.0, n_steps + 1)    # 从 t=0(噪声) 到 t=1(干净)
    for i in range(n_steps):
        dt = ts[i + 1] - ts[i]
        x = x + v * dt                         # 朝干净目标积分
    return x


def dreamzero_chunk(s0, goal, quality=1.0, n_steps=16, seed=0):
    """一个 chunk：联合去噪‘视频’与‘动作’，二者共享同一去噪过程。
    返回 (预测视频, 抽取的动作)。"""
    # 1) 视频预测器给出本 chunk 的干净目标视频（含可能的预测误差）
    pred_video_clean = video_predictor(s0, goal, H, quality)
    clean_actions = extract_actions_via_idm(pred_video_clean)
    # 2) 把 [视频; 动作] 拼成联合 latent，用【共享时刻】一起流匹配去噪（论文设定）
    joint_clean = np.concatenate([pred_video_clean, clean_actions])
    joint_denoised = flow_match_denoise(joint_clean, n_steps, seed)
    pred_video = joint_denoised[:H + 1]
    actions = joint_denoised[H + 1:]
    return pred_video, actions


# ---------------------------------------------------------------------------
# (C)(D) 自回归长程：开环漂移 vs 闭环 KV 替换
# ---------------------------------------------------------------------------
def rollout(goal, quality, closed_loop, n_chunks=6, seed0=0):
    """连续推多个 chunk。closed_loop=True 时，每个 chunk 后用‘真实观测’重置起点
    （= 闭环 KV 缓存替换）；False 时用模型自己预测的末帧（= 开环，会漂移）。"""
    true_s = 0.0                         # 世界的真实状态
    belief_s0 = 0.0                      # 喂给模型的起点（信念）
    true_traj = [true_s]
    for k in range(n_chunks):
        pred_video, actions = dreamzero_chunk(belief_s0, goal, quality,
                                              n_steps=16, seed=seed0 + k)
        # 在真实世界里执行抽取出的动作（真实动力学 s_{t+1}=s_t+a_t）
        for a in actions:
            true_s = true_s + a + (SIGMA * rng.standard_normal() if SIGMA else 0.0)
            true_traj.append(true_s)
        if closed_loop:
            belief_s0 = true_s           # 用真实观测替换 → 误差清零（KV 替换）
        else:
            belief_s0 = pred_video[-1]   # 用模型预测末帧 → 误差累积（开环漂移）
    return np.array(true_traj)


def main():
    print(__doc__)
    goal = 1.0

    # ---- (B) 视频质量 → 动作质量：论文的核心等式 ----
    print("=" * 60)
    print("(B/D) 机器人能力 = 视频生成质量")
    print("-" * 60)
    for q in [1.0, 0.7, 0.4]:
        errs = []
        for s in range(200):
            _, acts = dreamzero_chunk(0.0, goal, quality=q, seed=s)
            reached = np.cumsum(acts)[-1]          # 执行动作后到达的位置
            errs.append(abs(reached - goal))
        print(f"  视频质量={q:.1f}  ->  到达目标的平均误差 = {np.mean(errs):.3f}")
    print("  视频越准，动作越准 —— 失败源于‘视频预测错误’而非‘动作抽取’。\n")

    # ---- (C) 开环漂移 vs 闭环 KV 替换 ----
    print("=" * 60)
    print("(C) 自回归长程：开环漂移 vs 闭环 KV 缓存替换")
    print("-" * 60)
    q = 0.6                                        # 故意用不完美的视频骨干
    goal_far = 5.0                                 # 远目标：需 ~5 个 chunk 才能到达
    drift_open, drift_closed = [], []
    for s in range(300):
        to = rollout(goal_far, q, closed_loop=False, n_chunks=6, seed0=s)
        tc = rollout(goal_far, q, closed_loop=True,  n_chunks=6, seed0=s)
        drift_open.append(abs(to[-1] - goal_far))
        drift_closed.append(abs(tc[-1] - goal_far))
    print(f"  目标={goal_far}，6 个 chunk 后与目标的最终偏差：")
    print(f"    开环（用预测帧续推）         : {np.mean(drift_open):.3f}  <- 误差累积/漂移")
    print(f"    闭环（用真实观测替换 KV）     : {np.mean(drift_closed):.3f}  <- 漂移被清零")
    print(f"    闭环把长程漂移降低了 {(1-np.mean(drift_closed)/np.mean(drift_open))*100:.0f}%")
    print("\n要点：DreamZero 利用‘机器人是闭环’这一事实，每执行完一个 chunk")
    print("就用真实观测替换 KV 缓存里的预测帧 —— 这是纯视频世界模型没有、")
    print("WAM 独有的优势，根治了自回归视频的复合误差（论文 §3.1 / Fig. 4）。")


if __name__ == "__main__":
    main()
