"""
OpenWAM 核心机制之三：可见性注意力掩码与信息流
================================================

论文 §3.1 / §4.2.2，式 (1)。

掩码在视频/动作两个模态上因子化为 2×2 块：

        ┌                    ┐
   M =  │  M_V←V    M_V←A   │      M_X←Y = 模态 X 的 token 能否 attend 模态 Y
        │  M_A←V    M_A←A   │
        └                    ┘

两个**模态内**块被固定：
  M_V←V  首帧因果：噪声帧互相可见、且都能看干净首帧；干净首帧只看自己
                  → 保护干净条件不被噪声污染
  M_A←A  双向：一个 chunk 内的噪声动作 token 互相可见 → 预测的动作彼此告知

两个**跨模态**块给出 4 种模式，这就是 Finding 2 的全部内容：

  模式                M_A←V   M_V←A   RoboTwin2.0-Full 平均
  isolated              ✗       ✗           87.41
  video-sees-action     ✗       ✓           87.63   ← 有通路，但方向错了
  action-sees-video     ✓       ✗           92.39
  mutual                ✓       ✓           92.15

约 5 点的鸿沟只由一个二值问题决定：**动作 token 能不能读到视频 token**。

这个 demo 用**多层可达性传播**严格证明为什么"方向错了等于没有"：即使堆
再多层，video-sees-action 下动作 token 也永远拿不到世界信息，因为动作从不
attend 视频。

运行：python3 visibility_mask_information_flow.py
"""
import numpy as np

# ============================================================================
# 掩码构造
# ============================================================================

MODES = {
    #                     M_A←V  M_V←A
    "isolated":          (False, False),
    "video-sees-action": (False, True),
    "action-sees-video": (True,  False),
    "mutual":            (True,  True),
}


def first_frame_causal(n_frames):
    """
    M_V←V：首帧因果注意力。
      - 索引 0 是干净的条件帧（当前观测的编码）
      - 噪声帧 1..n-1 互相可见，且都能 attend 干净首帧
      - 干净首帧**只 attend 自己** → 干净条件不被噪声污染
    返回 [n_frames, n_frames] 的布尔矩阵，M[q, k] = query q 能否 attend key k。
    """
    M = np.zeros((n_frames, n_frames), dtype=bool)
    M[1:, :] = True          # 噪声帧看所有帧（含干净首帧）
    M[0, 0] = True           # 干净首帧只看自己
    return M


def bidirectional(n):
    """M_A←A：双向注意力，chunk 内动作 token 全互见。"""
    return np.ones((n, n), dtype=bool)


def build_mask(n_frames, n_actions, mode):
    """
    按式 (1) 组装完整掩码。token 顺序：[视频帧 0..T'-1 | 动作步 0..H-1]。
    """
    a_sees_v, v_sees_a = MODES[mode]
    N = n_frames + n_actions
    M = np.zeros((N, N), dtype=bool)

    fv, fa = slice(0, n_frames), slice(n_frames, N)
    M[fv, fv] = first_frame_causal(n_frames)      # M_V←V  固定
    M[fa, fa] = bidirectional(n_actions)          # M_A←A  固定
    M[fa, fv] = a_sees_v                          # M_A←V  可配
    M[fv, fa] = v_sees_a                          # M_V←A  可配

    # 干净首帧的行永不 attend 噪声未来帧或动作（论文 §5.1.1 明确要求）
    M[0, 1:] = False
    return M


# ============================================================================
# 多层可达性传播
# ============================================================================
# 单层注意力里，query q 只能直接拿到 M[q, k] 为真的那些 k 的信息。
# 但堆 L 层后，信息可以经中间 token 间接传播：
#     R_1 = M
#     R_{l+1} = R_l ∨ (R_l · M)      （布尔矩阵乘）
# 这正好是布尔邻接矩阵的传递闭包，L 层后的可达性 = M 的 L 步可达。

def reachability(M, layers):
    """返回 L 层堆叠后 token 之间的信息可达矩阵。"""
    R = M.copy()
    for _ in range(layers - 1):
        R = R | (R.astype(np.uint8) @ M.astype(np.uint8)).astype(bool)
    return R


def action_receives_world(M, n_frames, n_actions, layers):
    """
    L 层后，动作 token 能否（直接或间接）获取任何视频 token 的信息？
    这就是 Finding 2 关心的那个二值问题。
    """
    R = reachability(M, layers)
    return bool(R[n_frames:, :n_frames].any())


def world_receives_action(M, n_frames, n_actions, layers):
    """L 层后，视频 token 能否获取动作信息？（反向通路）"""
    R = reachability(M, layers)
    return bool(R[:n_frames, n_frames:].any())


def render_mask(M, n_frames, n_actions, title):
    """把掩码画成 ASCII。行 = query，列 = key。█ 可见，· 屏蔽。"""
    N = n_frames + n_actions
    head = "     " + "".join("V" if k < n_frames else "A" for k in range(N))
    lines = [f"  {title}", head]
    for q in range(N):
        tag = ("V0*" if q == 0 else f"V{q} ") if q < n_frames else f"A{q-n_frames} "
        row = "".join("█" if M[q, k] else "·" for k in range(N))
        lines.append(f"  {tag:<3}│{row}")
    return "\n".join(lines)


# ============================================================================
# 自检
# ============================================================================

def demo():
    T_prime, H = 4, 3        # 4 个隐帧（含 1 个干净首帧）、3 个动作步
    L_BRIDGE = 30            # OpenWAM-α：全部 30 对层都是桥接层

    print("=" * 78)
    print("OpenWAM 可见性注意力掩码与信息流")
    print("=" * 78)
    print(f"\n配置：{T_prime} 个视频隐帧（V0 = 干净首帧，标 *）+ {H} 个动作步")
    print(f"      OpenWAM-α 的桥接层数 = {L_BRIDGE}（全部配对层）")

    # --- 检查 1：模态内块的固定语义 ---
    Mvv = first_frame_causal(T_prime)
    assert Mvv[0, 0] and not Mvv[0, 1:].any(), "干净首帧只能 attend 自己"
    assert Mvv[1:, :].all(), "噪声帧应能 attend 所有帧（含干净首帧）"
    Maa = bidirectional(H)
    assert Maa.all(), "动作 chunk 内应全互见"
    print("\n[检查 1] M_V←V 首帧因果（干净条件不被噪声污染）、M_A←A 双向  ✓")

    # --- 检查 2：四种模式的掩码结构 ---
    print("\n" + "-" * 78)
    print("四种跨模态模式的完整掩码（行 = query，列 = key）")
    print("-" * 78)
    masks = {}
    for mode in MODES:
        M = build_mask(T_prime, H, mode)
        masks[mode] = M
        a_sees_v, v_sees_a = MODES[mode]
        # 结构断言
        assert M[T_prime:, :T_prime].all() == a_sees_v or not a_sees_v
        assert bool(M[T_prime:, :T_prime].any()) == a_sees_v, f"{mode}: M_A←V 块不符"
        assert bool(M[1:T_prime, T_prime:].any()) == v_sees_a, f"{mode}: M_V←A 块不符"
        assert not M[0, 1:].any(), f"{mode}: 干净首帧的行必须全屏蔽（除自己）"
        print()
        print(render_mask(M, T_prime, H, mode))

    # --- 检查 3：多层可达性 —— 为什么"方向错了等于没有" ---
    print("\n" + "-" * 78)
    print(f"堆叠 {L_BRIDGE} 层后的信息流（可达性传递闭包）")
    print("-" * 78)
    paper_scores = {
        "isolated": 87.41, "video-sees-action": 87.63,
        "action-sees-video": 92.39, "mutual": 92.15,
    }
    print(f"\n{'模式':<20}{'动作←世界':>12}{'世界←动作':>12}{'论文成功率':>12}")
    print("-" * 78)
    for mode, M in masks.items():
        a_w = action_receives_world(M, T_prime, H, L_BRIDGE)
        w_a = world_receives_action(M, T_prime, H, L_BRIDGE)
        print(f"{mode:<20}{'✓' if a_w else '✗':>12}{'✓' if w_a else '✗':>12}"
              f"{paper_scores[mode]:>11.2f}%")

    # 核心断言：video-sees-action 下，堆多少层动作都拿不到世界信息
    for layers in (1, 2, 5, 30, 100):
        assert not action_receives_world(masks["video-sees-action"], T_prime, H, layers), \
            f"video-sees-action 下动作永远不该获取世界信息（测试层数 {layers}）"
        assert not action_receives_world(masks["isolated"], T_prime, H, layers), \
            f"isolated 下动作永远不该获取世界信息（测试层数 {layers}）"
        assert action_receives_world(masks["action-sees-video"], T_prime, H, layers), \
            f"action-sees-video 下动作第 1 层就该获取世界信息（测试层数 {layers}）"
        assert action_receives_world(masks["mutual"], T_prime, H, layers)

    print(f"\n[检查 3] video-sees-action 即使堆到 100 层，动作仍拿不到世界信息  ✓")
    print("         因为动作 token 从不 attend 视频 token——反向通路帮不上忙。")

    # --- 检查 4：可达性完全预测了实验分组 ---
    print("\n" + "-" * 78)
    print("可达性预测 vs 论文实测")
    print("-" * 78)
    can, cannot = [], []
    for mode, M in masks.items():
        (can if action_receives_world(M, T_prime, H, L_BRIDGE) else cannot).append(mode)

    lo, hi = max(paper_scores[m] for m in cannot), min(paper_scores[m] for m in can)
    print(f"\n  动作拿不到世界信息：{cannot}")
    print(f"      成功率 {min(paper_scores[m] for m in cannot):.2f}% – {lo:.2f}%")
    print(f"  动作拿得到世界信息：{can}")
    print(f"      成功率 {hi:.2f}% – {max(paper_scores[m] for m in can):.2f}%")
    print(f"\n  两组之间的鸿沟：{hi - lo:.2f} 个百分点，组内差异 ≤ 0.24")
    assert hi - lo > 4.0, "两组之间应有约 5 点的鸿沟"
    assert lo < hi, "拿不到世界信息的一组必须全部低于拿得到的一组"
    print("\n[检查 4] 一个二值问题完全解释了实验分组  ✓")
    print("         这也是一个实现陷阱：只检查'有没有跨模态注意力'不够，方向才是关键。")

    # --- 检查 5：mutual vs 单向的偏好随预训练翻转 ---
    print("\n" + "=" * 78)
    print("论文 Figure 11：mutual vs action-sees-video 的偏好随预训练翻转")
    print("=" * 78)
    flips = [
        # 设置,                          action-sees-video 绝对值, Mutual − ASV
        ("From Scratch · RoboTwin-Full",  92.39, -0.24),
        ("Pretrained  · RoboTwin-Full",   92.34, +0.16),
        ("Pretrained  · C2R (ID)",        86.98, +0.70),
        ("Pretrained  · C2R (OOD)",       25.70, +0.72),
    ]
    print(f"\n{'设置':<32}{'ASV 基线':>10}{'Mutual − ASV':>15}")
    print("-" * 78)
    for name, base, delta in flips:
        arrow = "→ Mutual 更好" if delta > 0 else "→ 单向更好"
        print(f"{name:<32}{base:>10.2f}{delta:>+14.2f}   {arrow}")

    scratch = [d for n, _, d in flips if n.startswith("From Scratch")]
    pretrained = [d for n, _, d in flips if n.startswith("Pretrained")]
    assert all(d < 0 for d in scratch), "from-scratch 应偏好单向"
    assert all(d > 0 for d in pretrained), "预训练后应偏好 mutual"
    # ID 与 OOD 的幅度相当 → 排除域偏移伪影
    id_delta = [d for n, _, d in flips if "C2R (ID)" in n][0]
    ood_delta = [d for n, _, d in flips if "C2R (OOD)" in n][0]
    assert abs(id_delta - ood_delta) < 0.1, "ID 与 OOD 的翻转幅度应相当（排除域偏移伪影）"

    print(f"\n  翻转在 ID (+{id_delta:.2f}) 与 OOD (+{ood_delta:.2f}) 上幅度相当")
    print("    → 排除域偏移伪影（若是域偏移，两者应不同）")
    print("  两个协议（RoboTwin-Full 与 C2R）同向翻转")
    print("    → 排除评测协议伪影")
    print("\n  留下的是真实机制：预训练让世界–动作交互变成真正双向的交换——")
    print("  预测的未来帧为动作生成提供视觉引导，预测的动作反过来告知")
    print("  那些帧中机械臂运动该怎么合成。数据稀缺时两条流建立不起这种")
    print("  对应，反向通路只是噪声；充足数据填上缺口，mutual 的优势才兑现。")
    print("\n[检查 5] 架构选择依赖数据规模：小规模选出的最优放大后可能不再最优  ✓")

    print("\n" + "=" * 78)
    print("全部检查通过 ✓")
    print("=" * 78)


if __name__ == "__main__":
    demo()
