"""
FSQ (Finite Scalar Quantization) 量化 demo —— GPC Stage 1 的核心
====================================================================

对照论文式 (4):   ẑ = ⌊ (L/2) · tanh(z) ⌉

本 demo 用纯 NumPy 演示：
  1. FSQ 如何把连续潜向量量化到规则网格（隐式码本 L^d）；
  2. 隐式码本为何"零存储、零坍缩"；
  3. STE（直通估计器）如何让梯度穿过不可导的取整；
  4. 与 VQ-VAE 最近邻查表的对比（为何 FSQ 不需要显式码本）。

只依赖 numpy，直接运行：  python fsq_quantization_demo.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. FSQ 前向量化
# ----------------------------------------------------------------------
def fsq_quantize(z, L=9):
    """把连续潜向量 z 的每一维独立量化到 L 个标量级。

    z : (..., d) 连续潜向量（编码器输出）
    L : 每维的量化级数（论文用 9）

    返回 ẑ，每维取值落在 {-⌊L/2⌋, ..., 0, ..., ⌊L/2⌋}（共 L 个整数格点）。
    这些整数格点就是"隐式码本"的坐标——我们从不显式存储它。
    """
    # tanh 把每维压到 (-1, 1)，再乘 L/2 把 (-1,1) 展开到约 (-L/2, L/2)
    scaled = (L / 2.0) * np.tanh(z)
    # 四舍五入到最近整数 → 落到网格格点
    z_hat = np.round(scaled)
    return z_hat


def fsq_code_index(z_hat, L=9):
    """把每维的整数格点编码成一个 0..L-1 的 token id，
    再把 d 维组合成一个隐式码本索引（仅示意，L^d 通常大到无法真的用一个整数存）。"""
    # 把 {-4,...,4} 平移到 {0,...,8}
    per_dim = (z_hat + (L // 2)).astype(int)          # (..., d)
    return per_dim                                     # 每维一个 L 进制 token


# ----------------------------------------------------------------------
# 2. STE：让梯度穿过 round()
# ----------------------------------------------------------------------
def fsq_straight_through(z, L=9):
    """演示 STE 的核心恒等式（用 numpy 手动模拟前向/反向）。

    在 PyTorch 里写法是:
        z_disc = torch.round((L/2)*torch.tanh(z))
        z_ste  = z_cont + (z_disc - z_cont).detach()
    前向: z_ste == z_disc（量化值）
    反向: d z_ste / d z_cont == 1（梯度直通，绕过 round 的零梯度）

    这里 z_cont 指 (L/2)*tanh(z) 之后的连续值，便于展示。
    """
    z_cont = (L / 2.0) * np.tanh(z)
    z_disc = np.round(z_cont)
    # 前向值 = z_disc；"反向"我们约定梯度 = 对 z_cont 的梯度（即 1）
    z_ste_forward = z_cont + (z_disc - z_cont)   # 数值上 == z_disc
    grad_wrt_z_cont = np.ones_like(z_cont)       # STE 约定的反向梯度
    return z_ste_forward, grad_wrt_z_cont


# ----------------------------------------------------------------------
# 3. 对比：VQ-VAE 的最近邻查表（需要显式码本）
# ----------------------------------------------------------------------
def vqvae_quantize(z, codebook):
    """VQ-VAE 式量化：在显式码本里找最近邻（式 2）。

    z        : (N, d)
    codebook : (K, d)  —— 必须显式存储、训练更新，可能坍缩
    返回每个 z 对应的码本索引 k* 和量化向量。
    """
    # ||z - e_k||^2 对所有 k
    dists = ((z[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)  # (N, K)
    k_star = dists.argmin(axis=1)                                   # (N,)
    return k_star, codebook[k_star]


# ----------------------------------------------------------------------
# 演示
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    L, d = 9, 40                      # 论文配置：每维 9 级、40 维
    print("=" * 66)
    print(f"FSQ 配置: L={L} 级/维, d={d} 维  →  隐式码本大小 = L^d = {L}^{d}")
    print(f"          ≈ {float(L)**d:.3e}  （天文数字，但零存储！）")
    print("=" * 66)

    # 编码器输出一个 batch 的连续潜向量
    z = rng.normal(0, 1.5, size=(4, d))

    z_hat = fsq_quantize(z, L)
    tokens = fsq_code_index(z_hat, L)

    print("\n[1] 前 8 维的量化演示（样本 0）:")
    print(f"    连续 z      : {np.round(z[0, :8], 2)}")
    print(f"    (L/2)tanh(z): {np.round((L/2)*np.tanh(z[0, :8]), 2)}")
    print(f"    量化 ẑ      : {z_hat[0, :8].astype(int)}   (格点, 落在 -4..4)")
    print(f"    token id    : {tokens[0, :8]}             (平移到 0..8)")

    # STE 演示
    _, grad = fsq_straight_through(z, L)
    print("\n[2] STE: round() 处处梯度为 0，但 STE 让 d z_ste/d z_cont = 1")
    print(f"    直通梯度全为 1 ? {np.allclose(grad, 1.0)}")

    # "零坍缩"：随机采多组潜向量，看用到的格点覆盖率
    big = rng.normal(0, 1.5, size=(20000, d))
    used = fsq_code_index(fsq_quantize(big, L), L)
    per_dim_levels_used = [len(np.unique(used[:, j])) for j in range(d)]
    print("\n[3] '无坍缩'检查: 每维平均用到 "
          f"{np.mean(per_dim_levels_used):.1f}/{L} 个级")
    print("    —— FSQ 的格点永远存在，不存在 VQ-VAE 的'死码'概念。")

    # 对比 VQ-VAE 显式码本
    K = 512
    codebook = rng.normal(0, 1, size=(K, d))          # 需显式存储 K×d 参数
    k_star, _ = vqvae_quantize(z, codebook)
    print(f"\n[4] 对比 VQ-VAE: 需显式码本 {K}×{d} = {K*d} 个可训练参数,")
    print(f"    还需 EMA 更新/死码重初始化等启发式; 本 batch 命中码 = {k_star}")
    print("    FSQ 用 0 个码本参数就得到了 L^d 个隐式码。\n")

    print("要点: FSQ 把'学习一个离散集合'(VQ-VAE 的难题) 变成")
    print("      '编码器把连续特征摆到固定网格上'(普通连续优化) —— 这就是它稳定的原因。")


if __name__ == "__main__":
    main()
