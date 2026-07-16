"""
cld_significance_demo.py
=========================
复现论文用来标注统计显著性的【Compact Letter Display (CLD)】算法（Sec IV-A5）。

问题：有 k 个策略，做 k(k-1)/2 次成对显著性检验后，得到"哪些对可区分/不可区分"的
      一堆布尔关系。CLD 把这些关系压缩成字母标签：
      - 不共享任何字母的两个策略 => 统计显著不同；
      - 共享至少一个字母 => 不可区分。

本 demo：
  1) 用【Bonferroni 校正】的双比例 z 检验做成对比较（近似论文对二值SR的检验思路）；
  2) 用一个标准的贪心 CLD 算法把"不可区分图"转成字母标签；
  3) 打印一张可读的显著性表。

只依赖 numpy + 标准库。
"""

import numpy as np
from math import sqrt, erf


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def two_proportion_pvalue(s1, n1, s2, n2) -> float:
    """双比例 z 检验的双尾 p 值（成功数 s / 试验数 n）。"""
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - normal_cdf(abs(z)))


def pairwise_significance(policies, alpha=0.05):
    """
    返回 (names, significant_matrix)。
    significant[i][j] = True 表示 i,j 在 Bonferroni 校正后统计显著不同（可区分）。
    """
    names = [p[0] for p in policies]
    k = len(policies)
    n_tests = k * (k - 1) // 2
    alpha_adj = alpha / n_tests  # Bonferroni：维持全局 alpha
    sig = np.zeros((k, k), dtype=bool)
    print(f"  成对检验数 = {n_tests}, Bonferroni 校正后单次阈值 alpha' = {alpha_adj:.4f}\n")
    for i in range(k):
        for j in range(i + 1, k):
            _, s1, n1 = policies[i]
            _, s2, n2 = policies[j]
            pval = two_proportion_pvalue(s1, n1, s2, n2)
            is_sig = pval < alpha_adj
            sig[i][j] = sig[j][i] = is_sig
            mark = "显著不同 ✓" if is_sig else "不可区分"
            print(f"  {names[i]:>14} ({s1}/{n1}={s1/n1:.2f}) vs "
                  f"{names[j]:>14} ({s2}/{n2}={s2/n2:.2f}) : p={pval:.4f}  {mark}")
    return names, sig


def compute_cld(names, sig):
    """
    贪心 CLD：字母代表一个'不可区分的极大团(clique)'。
    两个策略若【不显著】=> 可以共享字母；若显著不同 => 不能共享任何字母。
    做法：不可区分关系 = (not sig)。求其极大团覆盖，每个团分配一个字母。
    """
    k = len(names)
    # 不可区分图的邻接（含自环）
    connected = ~sig
    np.fill_diagonal(connected, True)

    letters = [set() for _ in range(k)]
    letter_id = 0
    covered_edges = set()

    # 需要被同一字母覆盖的"不可区分对"（含单点）
    def edges_of(nodes):
        e = set()
        for a in nodes:
            for b in nodes:
                if a <= b:
                    e.add((a, b))
        return e

    all_needed = set()
    for i in range(k):
        for j in range(i, k):
            if connected[i][j]:
                all_needed.add((i, j))

    # 贪心找极大团直到覆盖所有"不可区分对"
    while covered_edges != all_needed:
        # 从任一未覆盖对出发，贪心扩成极大团
        remaining = all_needed - covered_edges
        seed = min(remaining)
        clique = {seed[0], seed[1]}
        # 尝试加入更多点，保持两两不可区分
        for v in range(k):
            if v in clique:
                continue
            if all(connected[v][u] for u in clique):
                clique.add(v)
        # 分配字母
        ch = chr(ord('a') + letter_id)
        letter_id += 1
        for v in clique:
            letters[v].add(ch)
        covered_edges |= edges_of(clique)

    return ["".join(sorted(s)) for s in letters]


def main():
    print("=" * 84)
    print("Compact Letter Display 演示：4 个策略在某任务上的成功率显著性分组")
    print("（数字模拟论文风格：微调LBM 最好，预训练LBM 与单任务接近）")
    print("=" * 84)
    # (名字, 成功数, 总数)  —— 每策略 50 次 rollout
    policies = [
        ("Finetuned-LBM", 40, 50),   # SR 0.80
        ("Pretrained-LBM", 27, 50),  # SR 0.54
        ("Single-Task", 25, 50),     # SR 0.50
        ("Scratch-Small", 12, 50),   # SR 0.24
    ]
    names, sig = pairwise_significance(policies)
    labels = compute_cld(names, sig)

    print("\n" + "=" * 84)
    print("CLD 结果（不共享字母 = 95% 置信下显著不同）：")
    print("=" * 84)
    for (nm, s, n), lab in zip(policies, labels):
        bar = "#" * int(s / n * 40)
        print(f"  {nm:>16} SR={s/n:.2f} [{bar:<40}] 字母: {lab}")
    print("\n读法示例：Finetuned-LBM='a' 与 Single-Task='c' 不共享字母 => 显著更好；")
    print("          Pretrained-LBM 与 Single-Task 若共享字母 => 统计上不可区分")
    print("          （正呼应论文'零样本预训练LBM 不稳定胜过单任务'的发现）。🐾")


if __name__ == "__main__":
    main()
