"""
MotionBricks 核心概念演示：Multi-Head VQ Tokenizer + Masked Motion Generation

演示内容：
1. Multi-head vector quantization（产品量化）的运动编码
2. 临界阻尼弹簧的根轨迹生成
3. Masked token modeling 的运动插值

运行：python multihead_tokenizer_demo.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ============================================================
# Part 1: Multi-Head Vector Quantization
# ============================================================

@dataclass
class MultiHeadVQTokenizer:
    """K 个独立码本的联合量化器，模拟 MotionBricks 的多头 tokenizer。

    单码本容量 = codebook_size
    多头组合容量 = codebook_size ^ num_heads
    """

    num_heads: int = 4
    codebook_size: int = 64
    dim_per_head: int = 8
    total_dim: int = 32

    def __post_init__(self):
        np.random.seed(42)
        self.codebooks = [
            np.random.randn(self.codebook_size, self.dim_per_head).astype(np.float32)
            for _ in range(self.num_heads)
        ]
        for i, cb in enumerate(self.codebooks):
            self.codebooks[i] = cb / np.linalg.norm(cb, axis=1, keepdims=True)

    def encode(self, x: np.ndarray) -> list[np.ndarray]:
        """将连续运动向量编码为 K 组离散索引。"""
        chunks = np.array_split(x, self.num_heads, axis=-1)
        indices = []
        for head_idx, (chunk, codebook) in enumerate(zip(chunks, self.codebooks)):
            dists = np.linalg.norm(chunk[:, None, :] - codebook[None, :, :], axis=-1)
            indices.append(np.argmin(dists, axis=-1))
        return indices

    def decode(self, indices: list[np.ndarray]) -> np.ndarray:
        """从 K 组离散索引重建连续运动向量。"""
        chunks = [self.codebooks[i][idx] for i, idx in enumerate(indices)]
        return np.concatenate(chunks, axis=-1)

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """编码 + 解码，返回重建向量和索引。"""
        indices = self.encode(x)
        reconstructed = self.decode(indices)
        return reconstructed, indices


def demo_tokenizer():
    """演示多头量化 vs 单头量化的容量差异。"""
    print("=" * 60)
    print("Part 1: Multi-Head VQ Tokenizer")
    print("=" * 60)

    tokenizer = MultiHeadVQTokenizer(num_heads=4, codebook_size=64, dim_per_head=8)

    print(f"  码本数 K = {tokenizer.num_heads}")
    print(f"  每个码本大小 = {tokenizer.codebook_size}")
    print(f"  组合容量 = {tokenizer.codebook_size}^{tokenizer.num_heads} = {tokenizer.codebook_size ** tokenizer.num_heads:,}")
    print(f"  等效单码本需要 {tokenizer.codebook_size ** tokenizer.num_heads:,} 条目")
    print()

    num_frames = 100
    motion_data = np.random.randn(num_frames, tokenizer.total_dim).astype(np.float32)
    motion_data = motion_data * 0.3

    reconstructed, indices = tokenizer.quantize(motion_data)
    error = np.mean(np.linalg.norm(motion_data - reconstructed, axis=-1))
    print(f"  {num_frames} 帧运动数据重建误差: {error:.4f}")

    unique_combos = set()
    for frame in range(num_frames):
        combo = tuple(idx[frame] for idx in indices)
        unique_combos.add(combo)
    print(f"  唯一 token 组合数: {len(unique_combos)} / {num_frames}")
    print()

    return motion_data, reconstructed


# ============================================================
# Part 2: Critically Damped Spring (根轨迹生成)
# ============================================================

def critically_damped_spring(
    x0: np.ndarray,
    v0: np.ndarray,
    x_target: np.ndarray,
    omega: float,
    dt: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """临界阻尼弹簧求解器，用于 Smart Locomotion 的根轨迹生成。

    方程：x'' + 2ω x' + ω²(x - x_target) = 0
    临界阻尼条件：ζ = 1（无振荡，最快收敛）
    """
    positions = [x0.copy()]
    velocities = [v0.copy()]

    x = x0.copy()
    v = v0.copy()

    for _ in range(num_steps):
        displacement = x - x_target
        acceleration = -2.0 * omega * v - omega * omega * displacement
        v = v + acceleration * dt
        x = x + v * dt
        positions.append(x.copy())
        velocities.append(v.copy())

    return np.array(positions), np.array(velocities)


def demo_spring():
    """演示 4 级根轨迹生成的前 2 级（线性插值 + 弹簧）。"""
    print("=" * 60)
    print("Part 2: Root Trajectory Generation (Spring Model)")
    print("=" * 60)

    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 3.0])
    num_steps = 200
    dt = 0.01

    # Level 0: 线性插值（naive）
    t = np.linspace(0, 1, num_steps + 1)
    naive_traj = start[None, :] + t[:, None] * (goal - start)[None, :]

    # Level 1: 临界阻尼弹簧
    spring_traj, spring_vel = critically_damped_spring(
        x0=start, v0=np.array([0.0, 0.0]), x_target=goal, omega=3.0, dt=dt, num_steps=num_steps
    )

    # 速度对比
    naive_speed = np.linalg.norm(np.diff(naive_traj, axis=0), axis=1) / dt
    spring_speed = np.linalg.norm(np.diff(spring_traj, axis=0), axis=1) / dt

    print(f"  起点: {start}, 终点: {goal}")
    print(f"  线性插值终点误差: {np.linalg.norm(naive_traj[-1] - goal):.6f}")
    print(f"  弹簧模型终点误差: {np.linalg.norm(spring_traj[-1] - goal):.6f}")
    print(f"  线性插值速度方差: {np.var(naive_speed):.6f} (恒速)")
    print(f"  弹簧模型速度方差: {np.var(spring_speed):.4f} (加速→减速)")
    print()

    return naive_traj, spring_traj, naive_speed, spring_speed


# ============================================================
# Part 3: Masked Motion In-Betweening
# ============================================================

def demo_masked_inbetweening():
    """演示 masked token modeling 的运动插值。

    模拟：给定首尾关键帧（已知 token），生成中间帧。
    简化版用余弦插值 + 噪声模拟神经网络输出。
    """
    print("=" * 60)
    print("Part 3: Masked Motion In-Betweening")
    print("=" * 60)

    num_frames = 64
    num_joints = 3  # 简化为3个关节角度

    np.random.seed(123)
    gt_motion = np.zeros((num_frames, num_joints))
    for j in range(num_joints):
        freq = 0.5 + j * 0.3
        phase = j * np.pi / 4
        gt_motion[:, j] = np.sin(2 * np.pi * freq * np.linspace(0, 1, num_frames) + phase)

    # 设置关键帧约束（模拟 T₃ 约束）
    keyframe_indices = [0, 15, 31, 47, 63]
    mask = np.ones(num_frames, dtype=bool)  # True = masked (需要生成)
    for idx in keyframe_indices:
        mask[idx] = False  # 关键帧不遮盖

    # 简化的插值生成（实际用 Transformer + masked prediction）
    generated = np.zeros_like(gt_motion)
    generated[~mask] = gt_motion[~mask]

    for seg_start_idx in range(len(keyframe_indices) - 1):
        i_start = keyframe_indices[seg_start_idx]
        i_end = keyframe_indices[seg_start_idx + 1]
        seg_len = i_end - i_start
        for f in range(1, seg_len):
            alpha = f / seg_len
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            generated[i_start + f] = (
                (1 - smooth_alpha) * gt_motion[i_start] + smooth_alpha * gt_motion[i_end]
            )

    error = np.mean(np.abs(gt_motion - generated))
    masked_error = np.mean(np.abs(gt_motion[mask] - generated[mask]))
    print(f"  总帧数: {num_frames}")
    print(f"  关键帧数: {len(keyframe_indices)} ({100 * len(keyframe_indices) / num_frames:.1f}%)")
    print(f"  遮盖帧数: {np.sum(mask)} ({100 * np.sum(mask) / num_frames:.1f}%)")
    print(f"  全局平均误差: {error:.4f}")
    print(f"  遮盖帧平均误差: {masked_error:.4f}")
    print()

    return gt_motion, generated, keyframe_indices, mask


# ============================================================
# Visualization
# ============================================================

def plot_all(motion_data, reconstructed, naive_traj, spring_traj, naive_speed,
             spring_speed, gt_motion, generated, keyframe_indices, mask):
    """生成综合可视化图表。"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MotionBricks 核心概念演示", fontsize=14, fontweight="bold")

    # (1) Tokenizer 重建
    ax = axes[0, 0]
    dims_to_show = [0, 1, 2]
    for d in dims_to_show:
        ax.plot(motion_data[:, d], alpha=0.6, label=f"原始 dim{d}")
        ax.plot(reconstructed[:, d], "--", alpha=0.8, label=f"重建 dim{d}")
    ax.set_title("Multi-Head VQ: 编码→解码重建")
    ax.set_xlabel("帧")
    ax.set_ylabel("值")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (2) 根轨迹对比
    ax = axes[0, 1]
    ax.plot(naive_traj[:, 0], naive_traj[:, 1], "b--", label="Level 0: 线性插值", alpha=0.6)
    ax.plot(spring_traj[:, 0], spring_traj[:, 1], "r-", label="Level 1: 临界阻尼弹簧", linewidth=2)
    ax.plot(*naive_traj[0], "go", markersize=10, label="起点")
    ax.plot(*naive_traj[-1], "r*", markersize=12, label="终点")
    ax.set_title("Root Trajectory: 线性 vs 弹簧")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # (3) 速度曲线
    ax = axes[1, 0]
    time = np.arange(len(naive_speed)) * 0.01
    ax.plot(time, naive_speed, "b--", label="线性插值（恒速）", alpha=0.6)
    ax.plot(time, spring_speed, "r-", label="弹簧模型（加速→减速）", linewidth=2)
    ax.set_title("速度曲线对比")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("速度")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Masked in-betweening
    ax = axes[1, 1]
    frames = np.arange(gt_motion.shape[0])
    joint_idx = 0
    ax.plot(frames, gt_motion[:, joint_idx], "b-", label="Ground truth", alpha=0.4, linewidth=2)
    ax.plot(frames, generated[:, joint_idx], "r--", label="生成结果", linewidth=1.5)
    ax.scatter(
        keyframe_indices,
        gt_motion[keyframe_indices, joint_idx],
        c="green", s=80, zorder=5, label="关键帧约束"
    )
    for idx in keyframe_indices:
        ax.axvline(x=idx, color="green", alpha=0.15, linestyle=":")
    ax.set_title("Masked In-Betweening (Joint 0)")
    ax.set_xlabel("帧")
    ax.set_ylabel("关节角度")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("motionbricks_demo.png", dpi=150, bbox_inches="tight")
    print("图表已保存: motionbricks_demo.png")
    plt.show()


if __name__ == "__main__":
    motion_data, reconstructed = demo_tokenizer()
    naive_traj, spring_traj, naive_speed, spring_speed = demo_spring()
    gt_motion, generated, keyframe_indices, mask = demo_masked_inbetweening()

    print("=" * 60)
    print("生成可视化...")
    print("=" * 60)
    plot_all(
        motion_data, reconstructed,
        naive_traj, spring_traj,
        naive_speed, spring_speed,
        gt_motion, generated, keyframe_indices, mask,
    )
