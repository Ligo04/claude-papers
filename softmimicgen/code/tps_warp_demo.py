"""
SOFTMIMICGEN Core Concept Demo: Thin Plate Spline (TPS) Non-Rigid Registration
================================================================================

This demo illustrates the key mechanism of SOFTMIMICGEN:
1. Represent deformable objects as point sets
2. Compute a TPS warp field between source and target configurations
3. Use the warp field to transform robot end-effector trajectories

Run: python tps_warp_demo.py
Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# =============================================================================
# TPS (Thin Plate Spline) Registration in 2D
# =============================================================================

def tps_kernel(r: np.ndarray) -> np.ndarray:
    """TPS kernel function: U(r) = r^2 * log(r) for 2D."""
    result = np.zeros_like(r)
    mask = r > 0
    result[mask] = r[mask] ** 2 * np.log(r[mask])
    return result


def compute_tps_warp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    reg_lambda: float = 0.01,
) -> dict:
    """
    Compute TPS warp from source to target point set.

    The TPS model: f(x) = Ax + b + sum_i w_i * U(||x - x_i||)
    where U(r) = r^2 * log(r) is the 2D TPS kernel.

    Args:
        source_pts: (N, 2) source control points
        target_pts: (N, 2) target control points
        reg_lambda: regularization strength (controls smoothness)

    Returns:
        dict with TPS parameters for applying the warp
    """
    n = source_pts.shape[0]

    # Build TPS system matrix
    # K[i,j] = U(||x_i - x_j||)
    dists = np.linalg.norm(source_pts[:, None] - source_pts[None, :], axis=-1)
    K = tps_kernel(dists)

    # P = [1, x, y] for affine part
    P = np.hstack([np.ones((n, 1)), source_pts])

    # Build the linear system:
    # | K + lambda*I  P | | w |   | target |
    # | P^T           0 | | a | = | 0      |
    top = np.hstack([K + reg_lambda * np.eye(n), P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bottom])

    rhs = np.vstack([target_pts, np.zeros((3, 2))])

    # Solve
    params = np.linalg.solve(L, rhs)
    weights = params[:n]       # (N, 2) - non-linear part
    affine = params[n:]        # (3, 2) - [b; A] affine part

    return {
        "control_points": source_pts,
        "weights": weights,
        "affine": affine,
        "reg_lambda": reg_lambda,
    }


def apply_tps_warp(warp: dict, points: np.ndarray) -> np.ndarray:
    """Apply TPS warp to a set of points."""
    ctrl = warp["control_points"]
    w = warp["weights"]
    a = warp["affine"]

    # Distances to control points
    dists = np.linalg.norm(points[:, None] - ctrl[None, :], axis=-1)  # (M, N)
    K = tps_kernel(dists)

    # f(x) = a[0] + a[1]*x + a[2]*y + sum_i w_i * U(||x - ctrl_i||)
    P = np.hstack([np.ones((points.shape[0], 1)), points])
    warped = K @ w + P @ a

    return warped


def compute_tps_jacobian(warp: dict, point: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of TPS warp at a single point.
    Used by SOFTMIMICGEN to transform end-effector orientations.

    J_f(p) captures local deformation around point p.
    """
    ctrl = warp["control_points"]
    w = warp["weights"]
    a = warp["affine"]

    # Affine part Jacobian: d/dx [a0 + a1*x + a2*y] = [a1, a2]^T
    J = a[1:].T  # (2, 2)

    # Non-linear part Jacobian
    for i in range(ctrl.shape[0]):
        diff = point - ctrl[i]
        r = np.linalg.norm(diff)
        if r > 1e-10:
            # d/dx [r^2 * log(r)] = 2*log(r)*diff + diff
            # = diff * (2*log(r) + 1)
            dU_dx = diff * (2 * np.log(r) + 1)
            J += np.outer(w[i], dU_dx)

    return J


# =============================================================================
# Simulation: Deformable Object + Robot Trajectory
# =============================================================================

def generate_rope_points(n_points: int = 15, bend: float = 0.0) -> np.ndarray:
    """Generate a rope as a series of 2D points with controllable bend."""
    t = np.linspace(0, 1, n_points)
    x = t * 2.0
    y = bend * np.sin(np.pi * t)
    return np.column_stack([x, y])


def generate_trajectory(
    rope: np.ndarray, n_waypoints: int = 8
) -> np.ndarray:
    """
    Generate a robot end-effector trajectory that follows the rope.
    The trajectory stays slightly above the rope with an offset.
    """
    # Sample points along the rope
    indices = np.linspace(0, len(rope) - 1, n_waypoints).astype(int)
    traj = rope[indices].copy()
    traj[:, 1] += 0.3  # offset above the rope
    return traj


# =============================================================================
# Visualization
# =============================================================================

def plot_softmimicgen_demo():
    """
    Full SOFTMIMICGEN pipeline demo:
    1. Source scene: straight rope + human demonstration trajectory
    2. Target scene: bent rope
    3. TPS registration between rope configurations
    4. Warped trajectory for the target scene
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Source scene ---
    source_rope = generate_rope_points(15, bend=0.0)
    source_traj = generate_trajectory(source_rope)

    ax = axes[0]
    ax.set_title("Step 1: Source Demonstration\n(Human teleop)", fontsize=13, fontweight="bold")
    ax.plot(source_rope[:, 0], source_rope[:, 1], "o-", color="#2196F3",
            markersize=6, linewidth=2, label="Source rope (nodes)")
    ax.plot(source_traj[:, 0], source_traj[:, 1], "s--", color="#F44336",
            markersize=8, linewidth=2, label="EE trajectory (source)")
    for i in range(len(source_traj) - 1):
        ax.annotate("", xy=source_traj[i + 1], xytext=source_traj[i],
                     arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5))
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Target scene + registration ---
    target_rope = generate_rope_points(15, bend=0.6)

    # Compute TPS warp from source to target rope
    warp = compute_tps_warp(source_rope, target_rope, reg_lambda=0.1)

    ax = axes[1]
    ax.set_title("Step 2: Non-Rigid Registration\n(TPS warp field)", fontsize=13, fontweight="bold")
    ax.plot(source_rope[:, 0], source_rope[:, 1], "o-", color="#2196F3",
            markersize=6, linewidth=2, alpha=0.4, label="Source rope")
    ax.plot(target_rope[:, 0], target_rope[:, 1], "o-", color="#4CAF50",
            markersize=6, linewidth=2, label="Target rope (nodes)")

    # Draw correspondence arrows
    for s, t in zip(source_rope, target_rope):
        ax.annotate("", xy=t, xytext=s,
                     arrowprops=dict(arrowstyle="->", color="#9C27B0",
                                     lw=1.0, alpha=0.5))

    # Visualize warp field on a grid
    grid_x, grid_y = np.meshgrid(np.linspace(-0.2, 2.2, 12),
                                  np.linspace(-0.8, 0.8, 8))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    warped_grid = apply_tps_warp(warp, grid_pts)
    displacements = warped_grid - grid_pts
    ax.quiver(grid_pts[:, 0], grid_pts[:, 1],
              displacements[:, 0], displacements[:, 1],
              color="#FF9800", alpha=0.3, scale=8)

    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Warped trajectory ---
    warped_traj = apply_tps_warp(warp, source_traj)

    ax = axes[2]
    ax.set_title("Step 3: Warped Trajectory\n(Ready for execution)", fontsize=13, fontweight="bold")
    ax.plot(target_rope[:, 0], target_rope[:, 1], "o-", color="#4CAF50",
            markersize=6, linewidth=2, label="Target rope")
    ax.plot(source_traj[:, 0], source_traj[:, 1], "s--", color="#F44336",
            markersize=8, linewidth=2, alpha=0.3, label="Original trajectory")
    ax.plot(warped_traj[:, 0], warped_traj[:, 1], "D-", color="#9C27B0",
            markersize=8, linewidth=2, label="Warped trajectory")

    for i in range(len(warped_traj) - 1):
        ax.annotate("", xy=warped_traj[i + 1], xytext=warped_traj[i],
                     arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.5))

    # Show Jacobian-based orientation at a few points
    for idx in [0, 3, 6]:
        J = compute_tps_jacobian(warp, source_traj[idx])
        # Original direction (pointing right)
        orig_dir = np.array([0.15, 0.0])
        # Transformed direction
        new_dir = J @ orig_dir
        new_dir = new_dir / np.linalg.norm(new_dir) * 0.15
        ax.annotate("", xy=warped_traj[idx] + new_dir, xytext=warped_traj[idx],
                     arrowprops=dict(arrowstyle="->", color="#E91E63", lw=2))

    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("softmimicgen_tps_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Demo saved to softmimicgen_tps_demo.png")


def demo_registration_cost():
    """
    Demonstrate source demonstration selection based on registration cost.
    SOFTMIMICGEN selects the source segment with lowest registration cost.
    """
    # Target rope configuration
    target = generate_rope_points(15, bend=0.4)

    # Multiple source configurations
    sources = {
        "Source A (straight)": generate_rope_points(15, bend=0.0),
        "Source B (slight bend)": generate_rope_points(15, bend=0.3),
        "Source C (opposite bend)": generate_rope_points(15, bend=-0.5),
    }

    print("=" * 60)
    print("Source Demonstration Selection via Registration Cost")
    print("=" * 60)
    print(f"Target: rope with bend=0.4\n")

    costs = {}
    for name, src in sources.items():
        warp = compute_tps_warp(src, target, reg_lambda=0.1)
        warped_src = apply_tps_warp(warp, src)
        cost = np.mean(np.linalg.norm(warped_src - target, axis=1))
        costs[name] = cost
        print(f"  {name}: registration cost = {cost:.4f}")

    best = min(costs, key=costs.get)
    print(f"\n  >> Selected: {best} (lowest cost)")
    print("=" * 60)


if __name__ == "__main__":
    print("SOFTMIMICGEN TPS Warp Demo")
    print("-" * 40)

    # Demo 1: Registration cost-based source selection
    demo_registration_cost()

    # Demo 2: Full pipeline visualization
    print("\nGenerating visualization...")
    plot_softmimicgen_demo()
