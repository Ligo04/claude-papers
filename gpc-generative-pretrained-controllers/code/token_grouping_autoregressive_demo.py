"""
Token 分组 + 自回归采样 demo —— GPC Stage 2
=============================================

演示两件事：
  1. Token 分组（grouping）：把 d 个 L 进制 token 每 G 个打包成 1 个 L^G 词表的大 token。
     复现论文 Table 3 的词表大小 / 序列长度权衡。
  2. 自回归采样：GPT 式因果分解 (式 5) + nucleus(top-p) 采样 (式 6 训练目标)。

纯 NumPy，直接运行:  python token_grouping_autoregressive_demo.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. Token 分组：复现 Table 3
# ----------------------------------------------------------------------
def grouping_table(L=9, d=40):
    """对不同分组因子 G，算出词表大小 N_vocab=L^G 与序列长度 N_token=d/G。"""
    print(f"Token 分组权衡 (L={L}, d={d})  —— 对照论文 Table 3")
    print(f"{'G':>3} | {'N_vocab = L^G':>16} | {'N_token = d/G':>13} | 备注")
    print("-" * 60)
    for G in [1, 2, 4, 5, 8]:
        if d % G != 0 and G != 8:
            note = "(d 不整除 G, 仅示意)"
        else:
            note = ""
        n_vocab = L ** G
        n_token = int(np.ceil(d / G))
        flag = ""
        if n_vocab > 1e7:
            flag = "  ← 词表爆炸, 超现代 LLM, 内存不可行"
        elif G == 5:
            flag = "  ← 甜点: APD 最高, 定性质量最佳"
        print(f"{G:>3} | {n_vocab:>16,} | {n_token:>13} |{flag}{note}")
    print()


def group_tokens(per_dim_tokens, L=9, G=5):
    """把 (d,) 的 L 进制 token 序列打包成 (d/G,) 的 L^G 进制 token。

    每组 G 个 L 进制位 → 一个大 token: sum_i tok_i * L^i
    """
    d = len(per_dim_tokens)
    assert d % G == 0, "为简化, 要求 G 整除 d"
    grouped = []
    for start in range(0, d, G):
        chunk = per_dim_tokens[start:start + G]
        val = 0
        for i, t in enumerate(chunk):
            val += int(t) * (L ** i)          # L 进制打包
        grouped.append(val)
    return np.array(grouped)                    # (d/G,) 每个 ∈ [0, L^G)


def ungroup_tokens(grouped, L=9, G=5):
    """逆操作: 把 L^G 进制大 token 拆回 G 个 L 进制 token。"""
    out = []
    for val in grouped:
        for i in range(G):
            out.append((val // (L ** i)) % L)
    return np.array(out)


# ----------------------------------------------------------------------
# 2. 自回归 + nucleus(top-p) 采样
# ----------------------------------------------------------------------
def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def nucleus_sample(probs, top_p=0.9, rng=None):
    """Nucleus (top-p) 采样 (Holtzman 2020):
    只在累积概率达到 top_p 的最小 token 集合里采样, 抑制低概率离群, 保多样性。"""
    rng = rng or np.random.default_rng()
    order = np.argsort(probs)[::-1]                 # 概率从高到低
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    # 保留累积概率首次 >= top_p 的位置(含)之前的 token
    k = np.searchsorted(cum, top_p) + 1
    keep = order[:k]
    renorm = probs[keep] / probs[keep].sum()
    return rng.choice(keep, p=renorm)


class TinyAutoregressiveController:
    """一个玩具版 GPT 式生成控制器: 给定状态 s 和已生成 token, 预测下一个 token 的分布。
    这里用随机但 *因果* 的打分函数模拟 transformer, 重点展示自回归 *流程* 而非真实建模。
    """
    def __init__(self, vocab, d_prime, seed=0):
        self.vocab = vocab            # L^G
        self.d_prime = d_prime        # 每步要生成的 token 数
        self.rng = np.random.default_rng(seed)
        # 玩具"权重": 让 logits 依赖于状态和已生成前缀(因果)
        self.W = self.rng.normal(0, 1, size=(8, vocab))

    def next_logits(self, state, prefix):
        """p(z̃^j | s, z̃^{<j}) 的 logits。只用 state + 已生成 prefix (因果)。"""
        ctx = np.zeros(8)
        ctx[:len(state)] = state[:8]
        # 让前缀影响 logits: 简单地把前缀和加进上下文
        if len(prefix) > 0:
            ctx[0] += 0.3 * np.sum(prefix) / self.vocab
        return ctx @ self.W

    def sample_step(self, state, top_p=0.9):
        """自回归生成一个决策步的完整 token 序列 z̃_t = (z̃^0,...,z̃^{d'-1})。"""
        prefix = []
        for j in range(self.d_prime):
            logits = self.next_logits(state, prefix)      # 只看 state + prefix
            probs = softmax(logits)
            tok = nucleus_sample(probs, top_p, self.rng)  # top-p 采样
            prefix.append(tok)
        return np.array(prefix)


# ----------------------------------------------------------------------
# 演示
# ----------------------------------------------------------------------
def main():
    L, d, G = 9, 40, 5
    print("=" * 62)
    grouping_table(L, d)

    print("=" * 62)
    print(f"[分组往返验证]  L={L}, d={d}, G={G} → d'={d//G} 个 token, 词表 {L**G:,}")
    per_dim = np.random.default_rng(1).integers(0, L, size=d)   # d 个 L 进制 token
    grouped = group_tokens(per_dim, L, G)
    recovered = ungroup_tokens(grouped, L, G)
    print(f"  原始 d 个 token (前10): {per_dim[:10]}")
    print(f"  分组后 d'={len(grouped)} 个大 token: {grouped}")
    print(f"  逆分组无损还原? {np.array_equal(per_dim, recovered)}")

    print("\n" + "=" * 62)
    print("[自回归 + nucleus 采样]  相同状态多次采样 → 行为多样(对照 CVAE 的确定性)")
    ctrl = TinyAutoregressiveController(vocab=L**G, d_prime=d // G, seed=7)
    state = np.array([0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2, 0.0])
    seqs = [ctrl.sample_step(state, top_p=0.9) for _ in range(5)]
    for i, s in enumerate(seqs):
        print(f"  采样 {i}: {s}")
    n_unique = len({tuple(s) for s in seqs})
    print(f"  5 次采样得到 {n_unique} 种不同技能 token 序列 —— 这就是 GPC 的行为多样性来源。")
    print("\n  (把 top_p 调到很小 → 趋于确定性, 类似 CVAE 坍缩到单一行为。)")


if __name__ == "__main__":
    main()
