# -*- coding: utf-8 -*-
"""
T-Rex —— 时空触觉 VQ-VAE（最小 Demo）
=====================================

T-Rex 用逐指 VQ-VAE 编码触觉力*历史*（每指 16 个时间步），把连续力时间序列
变成几个*离散* token（式 2）。其论点（及消融，表 2）是：这种*时间维度*编码
能捕捉动态——滑动起始、接触瞬变——而静态逐帧 MLP 会把它们丢掉。

这个仅依赖 NumPy 的 Demo 在合成的逐指力历史上训练一个微型一维 VQ 码本，
力历史来自四种典型触觉“事件”（无接触 none、稳定抓握 grip、滑动 slip、撞击 impact）。
它展示：

  1. 量化：一段 16 步力窗口被映射到一个离散码（码本质心），再从该码重建。
  2. 语义：每个学到的码本条目专门化为一种触觉*事件*类型——离散 token 是可解释的接触基元。
  3. 为何时间维度优于静态：像 slip 这样的事件是由其*轨迹*（先升后降）定义的，
     对单帧编码器是不可见的。

实现说明
-------
VQ-VAE 的码本实质上是通过在线向量量化学到的。为了让这个教学玩具*稳健*（真实 VQ-VAE
以码本坍缩闻名），我们用该过程行为良好的核心：k-means++ 初始化、Lloyd 更新、
死码重置——即对力历史做诚实的向量量化。这里编码器是恒等的（我们直接量化原始窗口）；
唯一学习的非 VQ 部分是一个用于重建的线性解码器。其*思想*——在一段时间窗口上的离散 token——
与论文完全相同。

运行：
    python tactile_vqvae_demo.py
"""

import numpy as np

rng = np.random.default_rng(7)

T = 16          # 力历史窗口（论文用 f_{t-15:t}，即 16 步）
K = 4           # 码本大小（我们设计 4 种典型触觉事件）
EVENTS = ["none", "grip", "slip", "impact"]
EVENTS_CN = {"none": "无接触", "grip": "稳定抓握", "slip": "滑动", "impact": "撞击"}


# ---------------------------------------------------------------------------
# 合成触觉事件：每个是一段归一化的 16 步力轨迹
# ---------------------------------------------------------------------------
def make_event(kind):
    t = np.linspace(0, 1, T)
    if kind == "none":         # 无接触：接近零，微小传感器噪声
        f = np.zeros(T)
    elif kind == "grip":       # 稳定抓握：先升后*保持*高位
        f = np.clip(t * 4, 0, 1.0)
    elif kind == "slip":       # 滑动：先建立压力，然后*突然*降到约 0
        f = np.where(t < 0.6, t / 0.6, np.maximum(0.0, 1.0 - (t - 0.6) * 3))
    elif kind == "impact":     # 撞击：早期尖锐峰值后衰减
        f = np.exp(-((t - 0.25) ** 2) / 0.005) * (1 - 0.3 * t)
    else:
        raise ValueError(kind)
    return f + rng.normal(0, 0.03, T)


def sample_batch(n):
    labels = rng.integers(0, len(EVENTS), n)
    X = np.stack([make_event(EVENTS[l]) for l in labels])
    return X, labels


# ---------------------------------------------------------------------------
# 向量量化（VQ-VAE 核心的码本）
# ---------------------------------------------------------------------------
def kmeanspp_init(X, k):
    """k-means++ 播种——把初始码分散开，避免坍缩。"""
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.stack(centers)


def quantize(X, codebook):
    d = ((X[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)   # (n, K)
    idx = d.argmin(1)
    return idx


def train_codebook(X, k, iters=50):
    """Lloyd 算法 + 死码重置（稳健 VQ）。"""
    cb = kmeanspp_init(X, k)
    for _ in range(iters):
        idx = quantize(X, cb)
        for c in range(k):
            mask = idx == c
            if mask.any():
                cb[c] = X[mask].mean(0)
            else:                                # 死码：重置到一个随机样本
                cb[c] = X[rng.integers(X.shape[0])]
    return cb


def fit_decoder(codes_onehot, X):
    """把 one-hot 码映射到重建波形的线性解码器（最小二乘）。
    在 VQ-VAE 中这是学到的解码器网络。"""
    Wd, *_ = np.linalg.lstsq(codes_onehot, X, rcond=None)
    return Wd


def main():
    print(__doc__)

    Xtr, _ = sample_batch(1024)
    codebook = train_codebook(Xtr, K)

    idx_tr = quantize(Xtr, codebook)
    onehot = np.eye(K)[idx_tr]
    Wd = fit_decoder(onehot, Xtr)
    recon = ((onehot @ Wd - Xtr) ** 2).mean()
    print(f"在 {T} 步窗口上训练了 {K} 条码本。重建 MSE = {recon:.4f}\n")

    # ---- 每种事件类型映射到哪个码？ ----------------------
    print("码本专门化（每个离散 token 是否 = 一种接触事件？）")
    Xte, lab = sample_batch(4000)
    idx = quantize(Xte, codebook)
    print(f"  {'事件':10s} -> 主导码   （4 个 token 上的直方图）")
    event_to_code = {}
    for e in range(len(EVENTS)):
        hist = np.bincount(idx[lab == e], minlength=K)
        dom = hist.argmax()
        event_to_code[EVENTS[e]] = dom
        share = hist[dom] / hist.sum()
        name = f"{EVENTS_CN[EVENTS[e]]}({EVENTS[e]})"
        print(f"  {name:14s} -> 码 {dom}    {hist}   （{share*100:.0f}% 纯）")

    distinct = len(set(event_to_code.values()))
    print(f"\n  -> {distinct}/{K} 种事件映射到*不同*码"
          f"（{'干净的 1:1 专门化' if distinct == K else '存在共享'}）。")

    # ---- 为何时间维度优于静态 -------------------------
    print("\n为何*时间维度*编码重要（表 2 消融：无 VQ-VAE 力时 -7%）：")
    slip = make_event("slip")
    grip = make_event("grip")
    print(f"  末帧力 f_t | grip(抓握) = {grip[-1]:+.2f}   slip(滑动) = {slip[-1]:+.2f}")
    print("    -> 这里 grip 以高位*结束*、slip 以接近零*结束*；但若 slip 在窗口中段就")
    print("       结束、或存在传感器偏置，单一末帧就有歧义。静态编码器只看到一个数。")
    cg = quantize(grip[None], codebook)[0]
    cs = quantize(slip[None], codebook)[0]
    print(f"  完整 16 步窗口 | 时间维度 VQ 把 grip -> 码 {cg}、slip -> 码 {cs}",
          "（不同）" if cg != cs else "（相同）")
    print("    -> 时间 token 编码了整条“先升后降”的形状，")
    print("       因此滑动起始是一个一等的、可区分的信号。")

    print("\n要点：把 16 步力*历史*压成一个离散 token，让触觉专家能对接触*动态*")
    print("（滑动起始、撞击瞬变）做出反应 —— 这正是 T-Rex 触觉专家所用的高频信号。")


if __name__ == "__main__":
    main()
