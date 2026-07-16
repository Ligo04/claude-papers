#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
variance_reduction_demo.py
============================================================
教学演示：ABC-VLA 论文中的「多次采样降方差」训练 trick
（论文 3.2 节 "Variance Reduction"，事实来源见 FACTS_ABC.md）

论文原话（精简复述，非官方代码，仅用于教学理解）：
    diffusion / flow-matching 的训练损失本质上是对三个随机变量的期望：
        L = E_{x, eps, tau} [ || eps - f_theta(x_tau, tau, cond) ||^2 ]
    其中 x 是干净动作（clean action chunk），eps 是噪声，tau 是扩散时间步。
    通常每个训练样本只采一组 (eps, tau) 来做蒙特卡洛估计 —— 这是「k=1」。

    ABC-VLA 的骨干是 4B 参数的 VLM，而挂在后面的 DiT action head 只有 42M 参数
    （骨干比头大近 100 倍）。因此“最贵”的部分是 VLM 的前向/反向传播，而不是
    DiT 头。论文的核心洞察是：
        1. VLM 前向只需算一次，把它的输出特征在 batch 维度复制 k 份；
        2. 每一份配一组独立采样的 (eps, tau)，分别过（很便宜的）DiT 头，
           算出 k 个独立损失；
        3. 反向传播时，在「conditioning 接口」（即 VLM 输出喂给 DiT 的那个点）
           把 k 份梯度做平均，再往回传给 VLM。
    这样 VLM 的反向传播成本与 k 无关（只做一次），而 DiT 头的成本随 k 线性增长，
    但因为头很小，k=8 时总的 step 耗时几乎不变（论文实测 H200 上：
    k=1 → 1.346 s/step，k=8 → 1.366 s/step，仅 +1.5% 开销）。
    好处：蒙特卡洛估计用了 k 个独立样本平均，方差按 ~1/k 量级下降，
    这是标准的「重要性均值方差缩减」结论（Var[mean of k iid] = Var[1 draw] / k）。

本脚本用一个「玩具」flow-matching 损失（不依赖真实 VLM/DiT 权重）来数值验证：
    - 对同一个 conditioning（等价于同一次 VLM 前向的输出），分别用 k=1 和 k=8
      组独立的 (eps, tau) 计算梯度估计；
    - 重复多次 outer trial，统计梯度估计的方差，验证其随 k 增大而下降；
    - 并且用论文给出的真实 wall-clock 数字（1.346s vs 1.366s）说明：
      即使 k=8 的“计算量”远大于 k=1，但因为 VLM 反向只做一次，
      实际耗时几乎不变 —— 这正是该 trick「近乎免费」的关键。

依赖：仅标准库 + numpy；若无 numpy 会退化为纯 Python 实现的等价逻辑。
若有 matplotlib 则额外画出 (a) 方差 vs k (b) train loss vs GPU-hours 示意图，
否则优雅降级为纯文本输出，不影响脚本可运行性。
"""

from __future__ import annotations

import sys
import math
import random
import logging

# ------------------------------------------------------------------
# 日志：贯穿全程打印关键步骤，方便教学时逐行对照
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("variance_reduction_demo")

# ------------------------------------------------------------------
# 依赖探测：numpy 优先，缺失则用纯 Python 列表模拟（保证脚本能独立跑通）
# ------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
    log.info("numpy 可用，使用向量化实现")
except ImportError:
    HAS_NUMPY = False
    log.info("numpy 不可用，退化为纯 Python 实现（速度较慢但结果等价）")

try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境下也能出图（保存为文件）
    import matplotlib.pyplot as plt
    # 指定支持中文的字体，避免默认 DejaVu Sans 缺字形告警/方块乱码
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
    log.info("matplotlib 可用，将生成示意图")
except ImportError:
    HAS_MPL = False
    log.info("matplotlib 不可用，跳过绘图，仅打印文本结果")


# ============================================================
# 玩具 flow-matching 损失模型
# ------------------------------------------------------------
# 为什么这样简化：真实 ABC-VLA 里 f_theta 是一个 42M 参数的 DiT，
# 输入是 VLM 特征 + 噪声动作 + 时间步。为了在教学脚本里几秒内跑完，
# 我们把 f_theta 换成一个「线性 + 已知最优解」的解析模型：
#     目标（flow-matching 速度场）：v* (x_tau, tau) = x - eps  （标准 conditional flow matching 定义）
#     预测：v_pred = cond_scale * (x - eps)  之类的线性近似（这里我们直接构造
#     一个带参数 theta 的线性预测器，损失对 theta 可求解析梯度）
# 这足以复现「对 (eps, tau) 求期望」这件事本身带来的方差，
# 而不需要真的训练一个神经网络 —— 这正是我们要演示的统计现象，
# 与 f_theta 的具体架构无关。
# ============================================================

ACTION_DIM = 8          # 玩具动作维度（模拟 30 步 chunk 里单步的简化 8 维）
CONDITIONING = 0.37      # 模拟「VLM 输出的 conditioning 标量」，一次前向只算一次
THETA_TRUE = 1.0         # 模型的“真实”最优参数（用于构造解析梯度）


def sample_one_draw(rng):
    """
    采样一组独立的 (x, eps, tau)，对应论文里 diffusion loss 期望的一次蒙特卡洛抽样。

    x   : 干净动作（clean action），这里用固定的合成向量 + 小噪声模拟数据分布
    eps : 噪声，标准正态
    tau : 扩散时间步，[0, 1] 均匀采样（连续时间 flow-matching 常见做法）
    """
    if HAS_NUMPY:
        x = np.ones(ACTION_DIM) * 0.5 + rng.normal(0, 0.05, size=ACTION_DIM)
        eps = rng.normal(0, 1.0, size=ACTION_DIM)
        tau = float(rng.uniform(0, 1))
    else:
        x = [0.5 + rng.gauss(0, 0.05) for _ in range(ACTION_DIM)]
        eps = [rng.gauss(0, 1.0) for _ in range(ACTION_DIM)]
        tau = rng.uniform(0, 1)
    return x, eps, tau


def grad_estimate_single_draw(x, eps, tau, theta, cond=CONDITIONING):
    """
    单次 (x, eps, tau) 采样下，flow-matching 损失对 theta 的梯度估计。

    损失定义（教学简化版）：
        L(theta) = || (x - eps) - theta * cond * (x_tau - eps) ||^2
    其中 x_tau = tau * x + (1 - tau) * eps 是插值噪声动作（rectified-flow 风格）。
    这是一个关于 theta 的二次函数，梯度可解析求出，
    grad = -2 * cond * (x_tau - eps) · [(x - eps) - theta * cond * (x_tau - eps)]

    这个具体形式不重要——重要的是它对每个 (eps, tau) 抽样都不同，
    所以对它取期望天然带有蒙特卡洛方差，这正是我们要缩减的对象。
    """
    if HAS_NUMPY:
        x = np.asarray(x)
        eps = np.asarray(eps)
        x_tau = tau * x + (1.0 - tau) * eps
        target = x - eps
        pred = theta * cond * x_tau
        residual = target - pred
        grad = -2.0 * cond * float(np.dot(x_tau, residual))
        return grad
    else:
        x_tau = [tau * xi + (1.0 - tau) * ei for xi, ei in zip(x, eps)]
        target = [xi - ei for xi, ei in zip(x, eps)]
        pred = [theta * cond * xt for xt in x_tau]
        residual = [t - p for t, p in zip(target, pred)]
        dot = sum(xt * r for xt, r in zip(x_tau, residual))
        grad = -2.0 * cond * dot
        return grad


def k_draw_grad_estimate(k, theta, rng):
    """
    对应论文 trick 的核心：单次 VLM 前向（这里即固定的 CONDITIONING）摊销到
    k 组独立 (eps, tau) 上，DiT 头分别计算，反向在 conditioning 接口处平均。

    返回 k 个独立梯度的平均值 —— 这就是 k-draw 梯度估计器。
    """
    grads = []
    for _ in range(k):
        x, eps, tau = sample_one_draw(rng)
        g = grad_estimate_single_draw(x, eps, tau, theta)
        grads.append(g)
    return sum(grads) / len(grads)


def variance_of_estimator(k, theta, n_trials, rng):
    """
    重复 n_trials 次「k-draw 梯度估计」，统计这些估计值之间的方差。
    方差越小，说明这一步的梯度方向越稳定 → 训练收敛更快/更稳。
    """
    estimates = [k_draw_grad_estimate(k, theta, rng) for _ in range(n_trials)]
    mean = sum(estimates) / len(estimates)
    var = sum((e - mean) ** 2 for e in estimates) / (len(estimates) - 1)
    return var, mean


def main():
    log.info("=" * 60)
    log.info("ABC-VLA 「多次采样降方差」trick —— 玩具数值验证")
    log.info("论文数据来源: arXiv 2606.27375v1, 3.2 节 Variance Reduction")
    log.info("=" * 60)

    rng = np.random.default_rng(42) if HAS_NUMPY else random.Random(42)
    theta = THETA_TRUE
    n_trials = 4000  # outer trial 次数，越大方差估计越稳（这是我们评估工具的方差，非训练本身）

    k_values = [1, 2, 4, 8, 16, 32]
    log.info("对每个 k in %s，重复 %d 次「独立训练 step」并统计梯度方差...", k_values, n_trials)

    results = []
    for k in k_values:
        var, mean = variance_of_estimator(k, theta, n_trials, rng)
        results.append((k, var, mean))
        log.info("k=%2d  梯度估计方差=%.6f  均值=%.6f", k, var, mean)

    # ------------------------------------------------------------
    # 理论核对：k 个独立同分布估计量取平均，方差应 ~ Var(k=1) / k
    # ------------------------------------------------------------
    var_k1 = results[0][1]
    print("\n" + "=" * 70)
    print("表 1：方差随 k 的下降情况（预期 ~ Var(k=1)/k）")
    print("=" * 70)
    print(f"{'k':>4} | {'实测方差':>12} | {'理论预测 Var(1)/k':>18} | {'相对误差':>10}")
    print("-" * 70)
    for k, var, mean in results:
        theory = var_k1 / k
        rel_err = abs(var - theory) / theory if theory > 0 else float("nan")
        print(f"{k:>4} | {var:>12.6f} | {theory:>18.6f} | {rel_err:>9.1%}")

    # ------------------------------------------------------------
    # 对齐论文的真实 wall-clock 开销数字（不是我们能测出来的，是论文报告的事实）
    # ------------------------------------------------------------
    h200_k1_s = 1.346   # 论文实测：H200, k=1, s/step
    h200_k8_s = 1.366   # 论文实测：H200, k=8, s/step
    overhead_pct = (h200_k8_s - h200_k1_s) / h200_k1_s * 100

    print("\n" + "=" * 70)
    print("表 2：论文实测 wall-clock 开销（H200 GPU，真实数据，非本脚本测量）")
    print("=" * 70)
    print(f"{'配置':>10} | {'s/step':>8} | {'相对 k=1 开销':>14}")
    print("-" * 70)
    print(f"{'k=1':>10} | {h200_k1_s:>8.3f} | {'baseline':>14}")
    print(f"{'k=8':>10} | {h200_k8_s:>8.3f} | {overhead_pct:>13.1f}%")

    print("\n结论：")
    print(f"  1) 方差缩减：k=8 相对 k=1 的梯度估计方差下降到约 "
          f"{results[3][1] / var_k1:.3f} 倍（理论预期 1/8={1/8:.3f}），与 ~1/k 量级吻合。")
    print(f"  2) 开销几乎免费：k 从 1 增到 8（理论算力 8 倍），但因为最贵的 VLM 反向传播")
    print(f"     只做一次（只有 42M 参数的 DiT 头随 k 线性变贵），实测 H200 上每 step")
    print(f"     仅从 {h200_k1_s}s 增加到 {h200_k8_s}s（+{overhead_pct:.1f}%）。")
    print(f"     这正是论文强调的核心卖点：用几乎不变的算力换取严格更低方差的梯度估计。")

    # ------------------------------------------------------------
    # 可选绘图
    # ------------------------------------------------------------
    if HAS_MPL:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

            # 图 (a): 方差 vs k（log-log，应接近斜率 -1 的直线）
            ks = [r[0] for r in results]
            vars_ = [r[1] for r in results]
            theory = [var_k1 / k for k in ks]
            ax = axes[0]
            ax.loglog(ks, vars_, "o-", label="实测方差 (empirical)", color="#2E86AB")
            ax.loglog(ks, theory, "--", label="理论 Var(1)/k", color="#A23B72")
            ax.set_xlabel("k (每个 conditioning 摊销的独立采样数)")
            ax.set_ylabel("梯度估计方差")
            ax.set_title("方差 vs k：多次采样的方差缩减效果")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)

            # 图 (b): train loss vs GPU-hours 示意图（k=1 vs k=8，定性示意，非真实训练曲线）
            # 为什么这样画：论文的核心叙事是——k=8 每 step 几乎不增加耗时，
            # 但每 step 的梯度质量更高（方差更低)，所以同样 GPU-hours 下收敛更快、更稳。
            # 这里用解析衰减曲线 + 噪声抖动来定性示意这个效果，明确标注为"示意"而非真实曲线。
            gpu_hours = np.linspace(0.1, 20, 200) if HAS_NUMPY else [0.1 + i * 0.1 for i in range(200)]
            rng2 = np.random.default_rng(7)

            def synth_loss_curve(gpu_hours, noise_scale, step_cost):
                # steps ~ gpu_hours / step_cost；loss 随 steps 指数衰减 + 噪声抖动
                if HAS_NUMPY:
                    steps = np.asarray(gpu_hours) / step_cost
                    base = 0.15 + 0.9 * np.exp(-steps / 6.0)
                    noise = rng2.normal(0, noise_scale, size=len(gpu_hours)) * np.exp(-steps / 8.0)
                    return base + noise
                else:
                    out = []
                    for gh in gpu_hours:
                        steps = gh / step_cost
                        base = 0.15 + 0.9 * math.exp(-steps / 6.0)
                        out.append(base)
                    return out

            # k=1: 每 step 耗时 1.346s（换算为「单位 GPU-hour 内能跑的 step 数」的相对基准），噪声更大
            # k=8: 每 step 耗时 1.366s（几乎相同），但梯度方差低 → 用更小的噪声抖动来示意更稳的收敛
            loss_k1 = synth_loss_curve(gpu_hours, noise_scale=0.05, step_cost=h200_k1_s / 3600.0)
            loss_k8 = synth_loss_curve(gpu_hours, noise_scale=0.015, step_cost=h200_k8_s / 3600.0)

            ax2 = axes[1]
            ax2.plot(gpu_hours, loss_k1, label="k=1 draw（方差大，抖动明显）", color="#F18F01", alpha=0.85)
            ax2.plot(gpu_hours, loss_k8, label="k=8 draws（方差小，更稳更快收敛）", color="#2E86AB", alpha=0.85)
            ax2.set_xlabel("GPU-hours")
            ax2.set_ylabel("train loss（示意，非真实训练曲线）")
            ax2.set_title("Train loss vs GPU-hours 示意图\n(k=8 每 step 耗时几乎不变，但梯度更稳)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out_path = "variance_reduction_demo.png"
            fig.savefig(out_path, dpi=130)
            log.info("图已保存到: %s", out_path)
        except Exception as exc:  # 绘图失败不应影响脚本主流程/退出码
            log.warning("绘图过程出现异常，已优雅降级为纯文本输出: %s", exc)
    else:
        log.info("matplotlib 不可用，已跳过绘图部分。")

    print("\n[提示] 本脚本是论文核心 trick 的教学简化实现，非官方代码，")
    print("       仅用于直观理解「摊销 VLM 前向 + 多组独立 (eps,tau) + 反向平均」")
    print("       为何能在几乎不增加计算开销的情况下降低梯度方差。")


if __name__ == "__main__":
    main()
