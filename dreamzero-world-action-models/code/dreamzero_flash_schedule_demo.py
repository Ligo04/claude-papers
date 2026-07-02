# -*- coding: utf-8 -*-
"""
DreamZero-Flash —— 解耦噪声调度（最小 Demo）
============================================

DreamZero-Flash 的核心洞见（论文 §3.2.5、Fig. 5、Table 3）：
  少步/单步推理时，动作要去噪到‘干净’，但其条件是当前 chunk 内‘仍带噪’的视频表示
  （因为极少步时生成的视频 token 还不准）。标准训练让视频与动作共享同一去噪时刻
  t~U(0,1)，于是模型从没见过‘视频很噪、动作要干净’的配置 —— 这就是训练-测试错配。

  Flash 在训练时把视频时刻偏向高噪声：t_video = 1 - η, η~Beta(7,1)（期望 0.125），
  动作时刻保持均匀。这逼模型学会‘从带噪视觉上下文预测干净动作’，正好匹配单步推理。

本仅依赖 NumPy 的 Demo：
  1. 画出（文本直方图）两种噪声调度：标准(耦合) vs Flash(解耦)。
  2. 用一个玩具回归任务模拟训练-测试错配，复现论文结论的定性形状：
     单步推理下，Flash 远好于天真单步、接近 4 步基线（论文 74% vs 52% vs 83%）。

运行：
    python dreamzero_flash_schedule_demo.py
"""

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1) 两种噪声调度的采样分布
# ---------------------------------------------------------------------------
# 约定（论文 Fig. 5）：t 越接近 1 越‘干净’，越接近 0 越‘噪’。
def sample_video_t(mode, n):
    if mode == "coupled":          # 标准 DreamZero：视频与动作共享 U(0,1)
        return rng.uniform(0, 1, n)
    elif mode == "flash":          # Flash：t_video = 1 - Beta(7,1) → 偏向高噪(小 t)
        return 1.0 - rng.beta(7.0, 1.0, n)


def text_hist(samples, label, bins=10):
    h, edges = np.histogram(samples, bins=bins, range=(0, 1), density=True)
    peak = h.max() if h.max() > 0 else 1
    print(f"  {label}  (均值 t_video = {samples.mean():.3f})")
    for i in range(bins):
        bar = "█" * int(round(h[i] / peak * 32))
        print(f"    t∈[{edges[i]:.1f},{edges[i+1]:.1f}]  {bar}")
    print()


# ---------------------------------------------------------------------------
# 2) 训练-测试错配的玩具模型
# ---------------------------------------------------------------------------
# 任务：给定一个‘视频上下文 c’（标量），预测干净动作 a* = f(c) = sin(3c)。
# 但推理时，模型拿到的是‘带噪视频’ c_noisy = c + noise(t_video)，噪声大小随 t_video 减小而增大。
# 训练时模型见到的 (c_noisy, t_video) 分布由噪声调度决定 —— 这就是关键。
def true_action(c):
    return np.sin(3.0 * c)


def video_noise_std(t_video):
    """t_video 越小（越噪），视频上下文噪声越大。单步推理对应 t_video≈0。"""
    return 0.6 * (1.0 - t_video)


def make_features(c, t_video):
    """简单非线性特征（固定随机基），让线性回归能拟合 sin。"""
    global _W, _b
    z = np.outer(c, _W) + _b                 # (n, F)
    return np.concatenate([np.cos(z), np.sin(z),
                           t_video[:, None]], axis=1)


F = 24
_W = rng.normal(0, 3.0, F)
_b = rng.normal(0, 3.0, F)


def train(mode, n=20000):
    """按给定噪声调度采样训练数据，最小二乘拟合‘从带噪视频预测干净动作’。"""
    c = rng.uniform(-1, 1, n)
    t_v = sample_video_t(mode, n)
    c_noisy = c + rng.normal(0, 1, n) * video_noise_std(t_v)
    X = make_features(c_noisy, t_v)
    y = true_action(c)                       # 目标永远是‘干净动作’
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w


def evaluate(w, t_video_at_inference, n=5000):
    """在指定的‘推理时视频噪声水平’下评估。单步推理 → t_video 很小(很噪)。"""
    c = rng.uniform(-1, 1, n)
    t_v = np.full(n, t_video_at_inference)
    c_noisy = c + rng.normal(0, 1, n) * video_noise_std(t_v)
    X = make_features(c_noisy, t_v)
    pred = X @ w
    mse = np.mean((pred - true_action(c)) ** 2)
    # 转成一个 0~100 的‘任务进度’代理：进度 = 100 * exp(-k*mse)
    progress = 100.0 * np.exp(-3.0 * mse)
    return mse, progress


def main():
    print(__doc__)

    print("=" * 60)
    print("1) 噪声调度对比（Fig. 5）")
    print("-" * 60)
    text_hist(sample_video_t("coupled", 50000), "标准 DreamZero：视频 t ~ U(0,1)（耦合）")
    text_hist(sample_video_t("flash", 50000),   "DreamZero-Flash：视频 t = 1-Beta(7,1)（解耦，偏噪）")

    print("=" * 60)
    print("2) 训练-测试错配：单步推理（视频很噪, t_video≈0.1）下的表现")
    print("-" * 60)
    w_std = train("coupled")
    w_flash = train("flash")

    # 4 步基线：推理时视频已较干净（t_video 高，约 0.7）
    _, prog_4step = evaluate(w_std, t_video_at_inference=0.7)
    # 标准模型 + 天真单步：视频仍很噪（t_video 约 0.1），但模型没在该工况训练过
    _, prog_std_1 = evaluate(w_std, t_video_at_inference=0.1)
    # Flash 模型 + 单步：训练时就大量见过‘视频噪、动作干净’
    _, prog_flash_1 = evaluate(w_flash, t_video_at_inference=0.1)

    print(f"  标准 DreamZero, 4 步（视频较干净 t≈0.7） : 任务进度代理 = {prog_4step:5.1f}")
    print(f"  标准 DreamZero, 单步（视频仍噪 t≈0.1）    : 任务进度代理 = {prog_std_1:5.1f}  <- 错配，最差")
    print(f"  DreamZero-Flash, 单步（视频仍噪 t≈0.1）   : 任务进度代理 = {prog_flash_1:5.1f}  <- 优于天真单步")
    print()
    print(f"  本玩具复现的【定性排序】：4 步 > Flash 单步 > 天真单步")
    print(f"  （注：上面是本 Demo 的任务进度代理，量纲与论文不同，只看排序。")
    print(f"   论文 Table 3 的真实数字是：4 步 83%  >  Flash 单步 74%  >>  天真单步 52%）")
    print("  Flash 通过‘训练时就让视频偏噪、动作保持干净’，消除了单步推理的训练-测试错配，")
    print("  从而把去噪步 4→1、延迟 350ms→150ms，性能损失很小。")


if __name__ == "__main__":
    main()
