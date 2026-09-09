"""
OpenWAM 核心机制之二：80 维统一动作空间
=========================================

论文 §3.4 / §5.1.2，式 (4)。

问题：跨具身训练一个模型，需要**一个动作头**面对宽度和语义都不同的身体。
RoboTwin2.0 是 14D 关节空间或 20D 双臂末端空间，LIBERO 是 10D 末端空间，
Wuji 灵巧手平台是 9D 末端 + 21 DoF 手指。

解法：定义一个槽位语义**固定**的 80 维空间：

    ┌──────────── 左臂 34D ───────────┬──────────── 右臂 34D ───────────┬─ 保留 12D ─┐
    │ EEF位置3 6D旋转6 夹爪1 灵巧手24 │ EEF位置3 6D旋转6 夹爪1 灵巧手24 │ 底盘等      │
    └─────────────────────────────────┴─────────────────────────────────┴─────────────┘

每个数据集声明索引映射 π：
    u = Scatter_π(Norm(a))        原生动作 → 统一空间
    a = Norm⁻¹(Gather_π(u))       统一空间 → 原生动作

因为槽位语义固定，不同具身**共享的结构落在同一坐标上**。式(2) 的 validity
mask m 精确标记被映射的槽位：未映射坐标训练时不接收梯度，推理时留在自己的
解析噪声路径上。

这是"从未见过的具身也能微调上去"的机制基础——论文在 Wuji 手 + Tianji 臂
（9D 末端 + 21 DoF，预训练混合中完全不存在）上全面超过 π0.5。

运行：python3 unified_action_space.py
"""
import numpy as np

# ============================================================================
# 槽位布局（论文 §3.4）
# ============================================================================
ARM_BLOCK = 34          # 每条臂：3 + 6 + 1 + 24
N_ARMS = 2              # 左右镜像
RESERVED = 12           # 具身专属通道（移动底盘等）
UNIFIED_DIM = N_ARMS * ARM_BLOCK + RESERVED    # = 80

# 单条臂块内部的偏移
OFF_EEF_POS = 0         # 末端位置 xyz
OFF_EEF_ROT = 3         # 6D 旋转表示
OFF_GRIPPER = 9         # 夹爪开合
OFF_DEXHAND = 10        # 灵巧手 24 DoF
LEN = {"eef_pos": 3, "eef_rot": 6, "gripper": 1, "dexhand": 24}

LEFT, RIGHT = 0, 1
RESERVED_BASE = N_ARMS * ARM_BLOCK             # = 68


def slot(arm, part, i=0):
    """返回 (arm, part) 的第 i 个分量在 80 维空间中的绝对索引。"""
    base = arm * ARM_BLOCK + {"eef_pos": OFF_EEF_POS, "eef_rot": OFF_EEF_ROT,
                              "gripper": OFF_GRIPPER, "dexhand": OFF_DEXHAND}[part]
    assert 0 <= i < LEN[part], f"{part} 只有 {LEN[part]} 个分量，收到 i={i}"
    return base + i


def slots(arm, part):
    """返回 (arm, part) 全部分量的索引列表。"""
    return [slot(arm, part, i) for i in range(LEN[part])]


def reserved_slot(i):
    """第 i 个保留槽位（0..11）。语义由具身自己声明。"""
    assert 0 <= i < RESERVED
    return RESERVED_BASE + i


# ============================================================================
# 具身适配器：声明 π 映射与归一化统计
# ============================================================================

class Embodiment:
    """
    一个具身的动作空间适配器。

    pi: 长度为 native_dim 的列表，pi[j] = 原生第 j 维应落到统一空间的哪个槽位。
    mean/std: 归一化统计，随 checkpoint 一起保存（论文强调缺失是硬错误）。
    """

    def __init__(self, name, pi, mean=None, std=None):
        self.name = name
        self.pi = np.asarray(pi, dtype=np.int64)
        self.native_dim = len(pi)

        assert self.pi.min() >= 0 and self.pi.max() < UNIFIED_DIM, "槽位索引越界"
        assert len(set(pi)) == len(pi), f"{name}: π 必须是单射，不能两个原生维映到同一槽位"

        self.mean = np.zeros(self.native_dim) if mean is None else np.asarray(mean, float)
        self.std = np.ones(self.native_dim) if std is None else np.asarray(std, float)
        assert np.all(self.std > 0), "归一化 std 必须为正"

    # ---- 式 (4) ----
    def to_unified(self, a_native):
        """u = Scatter_π(Norm(a))。未映射的槽位填 0。"""
        a_native = np.atleast_2d(a_native)
        normed = (a_native - self.mean) / self.std
        u = np.zeros((a_native.shape[0], UNIFIED_DIM))
        u[:, self.pi] = normed                      # scatter
        return u

    def from_unified(self, u):
        """a = Norm⁻¹(Gather_π(u))。"""
        u = np.atleast_2d(u)
        gathered = u[:, self.pi]                    # gather
        return gathered * self.std + self.mean

    def validity_mask(self):
        """
        式 (2) 的 m：精确标记被映射的槽位。
        训练时未映射坐标不接收梯度；推理时它们留在解析噪声路径上。
        """
        m = np.zeros(UNIFIED_DIM)
        m[self.pi] = 1.0
        return m


# ---------------------------------------------------------------------------
# 三个真实具身的映射（按论文描述的原生动作空间构造）
# ---------------------------------------------------------------------------

# LIBERO：10D 单臂末端空间 = 位置 3 + 旋转 6 + 夹爪 1
LIBERO = Embodiment(
    "LIBERO (单臂 10D 末端)",
    pi=slots(LEFT, "eef_pos") + slots(LEFT, "eef_rot") + slots(LEFT, "gripper"),
    mean=np.zeros(10),
    std=np.array([0.05, 0.05, 0.05, 1, 1, 1, 1, 1, 1, 1.0]),   # 位置量级远小于旋转
)

# RoboTwin2.0：20D 双臂末端空间 = 2 × (位置 3 + 旋转 6 + 夹爪 1)
ROBOTWIN = Embodiment(
    "RoboTwin2.0 (双臂 20D 末端)",
    pi=(slots(LEFT, "eef_pos") + slots(LEFT, "eef_rot") + slots(LEFT, "gripper")
        + slots(RIGHT, "eef_pos") + slots(RIGHT, "eef_rot") + slots(RIGHT, "gripper")),
)

# Wuji 手 + Tianji 臂：9D 末端 + 21 DoF 灵巧手 —— 预训练混合中**不存在**的具身
# 9D 末端 = 位置 3 + 旋转 6（无独立夹爪，抓握由灵巧手完成）
WUJI_DEX = Embodiment(
    "Wuji手+Tianji臂 (9D末端 + 21DoF灵巧手)",
    pi=(slots(LEFT, "eef_pos") + slots(LEFT, "eef_rot")
        + [slot(LEFT, "dexhand", i) for i in range(21)]),   # 只占 24 个手指槽的前 21 个
)

# EBench 式移动双臂：双臂 20D + 底盘 3D（用保留槽位）
EBENCH_MOBILE = Embodiment(
    "EBench 式移动双臂 (20D 双臂 + 3D 底盘)",
    pi=(slots(LEFT, "eef_pos") + slots(LEFT, "eef_rot") + slots(LEFT, "gripper")
        + slots(RIGHT, "eef_pos") + slots(RIGHT, "eef_rot") + slots(RIGHT, "gripper")
        + [reserved_slot(i) for i in range(3)]),            # 底盘 vx, vy, ωz
)

ALL = [LIBERO, ROBOTWIN, WUJI_DEX, EBENCH_MOBILE]


# ============================================================================
# 自检
# ============================================================================

def demo():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("OpenWAM 80 维统一动作空间")
    print("=" * 78)

    assert UNIFIED_DIM == 80, f"统一空间必须是 80 维，实际 {UNIFIED_DIM}"
    assert ARM_BLOCK == sum(LEN.values()), "臂块 34 = 3 + 6 + 1 + 24"
    print(f"\n布局：{N_ARMS} × {ARM_BLOCK}D 镜像臂块 + {RESERVED}D 保留 = {UNIFIED_DIM}D  ✓")
    print(f"      臂块内部：位置 {LEN['eef_pos']} + 6D旋转 {LEN['eef_rot']}"
          f" + 夹爪 {LEN['gripper']} + 灵巧手 {LEN['dexhand']} = {ARM_BLOCK}")

    # --- 检查 1：往返一致性（式 4 的可逆性）---
    print("\n" + "-" * 78)
    print(f"{'具身':<40} {'原生维':>7} {'占用槽位':>9} {'往返误差':>12}")
    print("-" * 78)
    for emb in ALL:
        a = rng.normal(size=(64, emb.native_dim))
        a_rt = emb.from_unified(emb.to_unified(a))
        err = float(np.max(np.abs(a - a_rt)))
        assert err < 1e-10, f"{emb.name} 往返误差 {err} 过大，式(4) 必须精确可逆"
        n_used = int(emb.validity_mask().sum())
        assert n_used == emb.native_dim, "占用槽位数必须等于原生维度（π 是单射）"
        print(f"{emb.name:<40} {emb.native_dim:>7} {n_used:>9} {err:>12.2e}")
    print("\n[检查 1] 全部具身 Scatter/Gather 往返精确可逆  ✓")

    # --- 检查 2：共享结构落在同一坐标 ---
    print("\n" + "-" * 78)
    print("共享结构对齐：不同具身的'左臂末端位置'必须落在同一槽位")
    print("-" * 78)
    shared = slots(LEFT, "eef_pos")
    for emb in (LIBERO, ROBOTWIN, WUJI_DEX, EBENCH_MOBILE):
        assert list(emb.pi[:3]) == shared, f"{emb.name} 的左臂末端位置未对齐到槽位 {shared}"
        print(f"  {emb.name:<40} → 槽位 {shared}")
    print("\n[检查 2] 槽位语义固定，共享结构自动对齐  ✓")
    print("         这是跨具身单一动作头可行的根本原因。")

    # --- 检查 3：validity mask 隔离未映射坐标 ---
    print("\n" + "-" * 78)
    print("validity mask：未映射坐标不接收梯度")
    print("-" * 78)
    emb = LIBERO
    m = emb.validity_mask()
    a = rng.normal(size=(8, emb.native_dim))
    u_true = emb.to_unified(a)
    u_pred = u_true + rng.normal(size=u_true.shape)          # 模型在全部 80 维上都有输出

    # 式 (2)：动作项被 m ⊙ 限制
    per_dim_loss = m * np.mean((u_pred - u_true) ** 2, axis=0)
    unmapped = np.setdiff1d(np.arange(UNIFIED_DIM), emb.pi)
    assert np.all(per_dim_loss[unmapped] == 0.0), "未映射坐标必须零损失（因而零梯度）"
    assert np.all(per_dim_loss[emb.pi] > 0.0), "已映射坐标必须有非零损失"
    print(f"  {emb.name}")
    print(f"    已映射槽位 {len(emb.pi):>2} 个 → 平均损失 {per_dim_loss[emb.pi].mean():.4f}")
    print(f"    未映射槽位 {len(unmapped):>2} 个 → 平均损失 {per_dim_loss[unmapped].mean():.4f}（严格为 0）")
    print("\n[检查 3] m ⊙ 精确隔离未映射坐标  ✓")

    # --- 检查 4：归一化后的数值永不到达机器人 ---
    print("\n" + "-" * 78)
    print("归一化边界：反归一化必须还原物理单位")
    print("-" * 78)
    emb = LIBERO
    # 一个物理上有意义的动作：末端沿 x 移动 2 cm，夹爪闭合
    a_phys = np.zeros((1, 10))
    a_phys[0, 0] = 0.02          # +2 cm
    a_phys[0, 9] = -1.0          # 夹爪闭合
    u = emb.to_unified(a_phys)
    print(f"  物理动作      x = {a_phys[0,0]:+.4f} m, 夹爪 = {a_phys[0,9]:+.2f}")
    print(f"  归一化后      槽位{slot(LEFT,'eef_pos',0)} = {u[0, slot(LEFT,'eef_pos',0)]:+.4f}"
          f", 槽位{slot(LEFT,'gripper')} = {u[0, slot(LEFT,'gripper')]:+.4f}")
    a_back = emb.from_unified(u)
    print(f"  反归一化回来  x = {a_back[0,0]:+.4f} m, 夹爪 = {a_back[0,9]:+.2f}")
    assert np.allclose(a_back, a_phys), "反归一化必须精确还原物理单位"
    # x 的 std 是 0.05，所以 0.02 m 归一化后应是 0.4
    assert abs(u[0, 0] - 0.4) < 1e-12, "归一化应按 per-dim std 缩放"
    print("\n[检查 4] 归一化值只在模型侧存在，服务端反归一化后才返回机器人  ✓")
    print("         论文：'normalized values therefore never reach a robot'")

    # --- 检查 5：未见具身的槽位覆盖 ---
    print("\n" + "-" * 78)
    print("未见具身的适配：Wuji 手 + Tianji 臂")
    print("-" * 78)
    emb = WUJI_DEX
    m = emb.validity_mask()
    used_left_arm = int(m[LEFT * ARM_BLOCK:(LEFT + 1) * ARM_BLOCK].sum())
    used_right_arm = int(m[RIGHT * ARM_BLOCK:(RIGHT + 1) * ARM_BLOCK].sum())
    used_reserved = int(m[RESERVED_BASE:].sum())
    print(f"  原生动作空间：9D 末端 + 21 DoF 灵巧手 = {emb.native_dim}D")
    print(f"  占用左臂块 {used_left_arm}/{ARM_BLOCK}（位置3 + 旋转6 + 灵巧手21，无独立夹爪）")
    print(f"  占用右臂块 {used_right_arm}/{ARM_BLOCK}（单臂平台）")
    print(f"  占用保留位 {used_reserved}/{RESERVED}")
    assert used_left_arm == 30 and used_right_arm == 0 and used_reserved == 0
    # 关键：末端位置与旋转跟其他具身对齐，所以预训练学到的这部分知识可以迁移
    assert list(emb.pi[:9]) == slots(LEFT, "eef_pos") + slots(LEFT, "eef_rot")
    print(f"\n  末端位置与旋转的 9 个槽位与 LIBERO / RoboTwin **完全对齐**")
    print(f"  → 预训练在这些槽位上学到的动作知识可直接迁移，")
    print(f"     只有 21 个灵巧手槽位是微调时新学的。")
    print("\n[检查 5] 未见具身通过共享槽位复用预训练知识  ✓")

    # --- 论文实测：未见具身上的真机结果 ---
    print("\n" + "=" * 78)
    print("论文 Table 8：Wuji 手 + Tianji 臂真机结果（该平台不在预训练混合中）")
    print("=" * 78)
    results = [
        # 任务,               π0.5 ID,        OpenWAM-α ID,   π0.5 OOD,       OpenWAM-α OOD
        ("Stack Toy Tower",    "53.3 / 10%",  "70.0 / 40%",  "40.0 / 10%",  "56.7 / 30%"),
        ("Collect Shuttlecocks","31.4 / 20%",  "82.9 / 60%",  "31.5 / 20%",  "52.6 / 26.7%"),
        ("Put Away Clothes",   "86.7 / 60%",  "100.0 / 100%","75.0 / 45%",  "91.7 / 85%"),
        ("Twist off Bottle Cap","80.0 / 30%",  "100.0 / 70%", "65.0 / 30%",  "90.0 / 80%"),
    ]
    print(f"\n{'任务':<24}{'π0.5 (ID)':>14}{'OpenWAM-α (ID)':>17}"
          f"{'π0.5 (OOD)':>14}{'OpenWAM-α (OOD)':>18}")
    print("-" * 78)
    for row in results:
        print(f"{row[0]:<24}{row[1]:>14}{row[2]:>17}{row[3]:>14}{row[4]:>18}")
    print("\n  格式：Score / SR。OpenWAM-α 在全部任务与全部设置上超过 π0.5。")
    print("  统一动作空间的固定槽位语义让'完全没见过的具身与动作维度'也能可靠适配。")

    print("\n" + "=" * 78)
    print("全部检查通过 ✓")
    print("=" * 78)


if __name__ == "__main__":
    demo()
