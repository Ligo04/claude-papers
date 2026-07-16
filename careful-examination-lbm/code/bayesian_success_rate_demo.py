"""
bayesian_success_rate_demo.py
================================
复现论文核心统计工具之一：用【贝叶斯 Beta-Bernoulli 后验】来表达成功率(SR)的不确定性，
并画出论文里那种"小提琴图"式的后验分布——而不是只报一个点估计。

论文动机（Sec IV-A4）：
  - 计算策略成功率 = 估计底层 Bernoulli 分布的参数 p。
  - 用【均匀先验 Beta(1,1)】，观测到 s 次成功 / f 次失败后，后验是 Beta(1+s, 1+f)。
  - 用后验分布（而非置信区间）能展示 p 的【整个不确定性】，且避免"CI 重叠 => 无差异"的误判。

本 demo 展示：为什么"50 次 rollout"和"200 次 rollout"给出的确定性天差地别，
以及为什么把任务难度调到 SR~50% 时区分两个策略最难（方差最大）。

只依赖 numpy（画图用纯 ASCII，零第三方绘图依赖，方便任何环境直接跑）。
"""

import numpy as np


def beta_posterior_stats(successes: int, failures: int):
    """均匀先验 Beta(1,1) 下的后验 Beta(1+s, 1+f) 的均值与 95% 可信区间。"""
    a, b = 1 + successes, 1 + failures
    mean = a / (a + b)
    # 用后验采样近似 95% 可信区间（也可用 scipy.stats.beta.ppf，但这里避免依赖）
    samples = np.random.default_rng(0).beta(a, b, size=200_000)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, lo, hi, samples


def ascii_violin(samples: np.ndarray, label: str, width: int = 50):
    """把后验样本渲染成一行 ASCII "小提琴"：密度越高的位置字符越密。"""
    hist, edges = np.histogram(samples, bins=width, range=(0, 1))
    peak = hist.max() if hist.max() > 0 else 1
    # 用 3 档字符表示密度
    chars = " .:-=+*#%@"
    line = "".join(chars[min(len(chars) - 1, int(h / peak * (len(chars) - 1)))] for h in hist)
    mean = samples.mean()
    mean_col = int(mean * width)
    marker = " " * mean_col + "|"
    print(f"{label:>28} |{line}|  mean={mean:.3f}")
    print(f"{'':>28} |{marker[:width]}|  (| = 后验均值)")


def prob_A_better_than_B(sA, fA, sB, fB, n=500_000):
    """蒙特卡洛估计 P(p_A > p_B)：两个后验各自采样比较。这就是贝叶斯版的'谁更好'。"""
    rng = np.random.default_rng(42)
    pa = rng.beta(1 + sA, 1 + fA, n)
    pb = rng.beta(1 + sB, 1 + fB, n)
    return (pa > pb).mean()


def demo_sample_size_effect():
    print("=" * 90)
    print("实验 1：同样的经验成功率 (0.6)，样本量从 10 → 50 → 200，后验如何收窄")
    print("        论文真机用 50 次/任务/条件，仿真用 200 次。看确定性的差别。")
    print("=" * 90)
    for n in (10, 50, 200):
        s = int(round(0.6 * n))
        f = n - s
        mean, lo, hi, samples = beta_posterior_stats(s, f)
        ascii_violin(samples, f"n={n} ({s}成功/{f}失败)")
        print(f"{'':>28}   95% 可信区间 = [{lo:.3f}, {hi:.3f}]，宽度 = {hi - lo:.3f}\n")


def demo_variance_at_half():
    print("=" * 90)
    print("实验 2：为什么把任务调到 SR≈50% 时，区分两个策略最'需要'大样本？")
    print("        Bernoulli 方差 p(1-p) 在 p=0.5 最大 → 后验最宽 → 最难分开。")
    print("=" * 90)
    n = 50
    for p in (0.1, 0.5, 0.9):
        s = int(round(p * n))
        _, lo, hi, samples = beta_posterior_stats(s, n - s)
        print(f"  真实 p={p:.1f}: 后验95%区间宽度 = {hi - lo:.3f}  (方差 p(1-p)={p*(1-p):.3f})")
    print("  => p=0.5 处后验最宽。论文刻意把任务调难到~50%，正是要在'最有信息量'但'最难分'的工作点上比较。\n")


def demo_policy_comparison():
    print("=" * 90)
    print("实验 3：贝叶斯地回答'微调LBM 是否比 单任务基线 更好？'")
    print("        对应论文用假设检验+CLD 做的事，这里用 P(p_LBM > p_baseline) 直观表达。")
    print("=" * 90)
    # 模拟论文风格数据：微调LBM 33/50 成功，单任务基线 26/50 成功
    sL, nL = 33, 50
    sB, nB = 26, 50
    p_better = prob_A_better_than_B(sL, nL - sL, sB, nB - sB)
    mL, *_ = beta_posterior_stats(sL, nL - sL)
    mB, *_ = beta_posterior_stats(sB, nB - sB)
    print(f"  微调 LBM   : {sL}/{nL} 成功, 后验均值 SR ≈ {mL:.3f}")
    print(f"  单任务基线 : {sB}/{nB} 成功, 后验均值 SR ≈ {mB:.3f}")
    print(f"  P(LBM 的真实成功率 > 基线的真实成功率) = {p_better:.3f}")
    if p_better > 0.95:
        print("  => >0.95：有较强证据认为 LBM 更好（贝叶斯语义下的'显著'）。")
    else:
        print("  => 未达 0.95：证据不足，可能需要更多 rollout（呼应论文'统计功效'警告）。\n")


if __name__ == "__main__":
    demo_sample_size_effect()
    demo_variance_at_half()
    demo_policy_comparison()
    print("核心 takeaway：报告一个成功率数字是不够的——必须报告它的【不确定性】，")
    print("并在比较策略时用后验/假设检验，而不是肉眼看柱状图谁高。🐾")
