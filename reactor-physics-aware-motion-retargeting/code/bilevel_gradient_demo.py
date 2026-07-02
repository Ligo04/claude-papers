"""
Bilevel Gradient Approximation Demo

Demonstrates ReActor's simplified gradient estimate that avoids computing
the inverse Hessian of the policy parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Ground truth: optimal retargeting parameters
theta_optimal = np.array([0.5, 0.3])

def motion_loss(theta):
    """L_motion: penalizes deviation from original motion"""
    return 0.5 * np.sum(theta**2)

def motion_gradient(theta):
    """∇_θ L_motion"""
    return theta

def tracking_performance(theta):
    """Simulated policy performance (better near optimal theta)"""
    diff = theta - theta_optimal
    return -0.3 * np.sum(diff**2)

def policy_gradient_term(theta, lambda_weight=0.1):
    """Approximation of (∂π*/∂θ) · (∂L_motion/∂π)"""
    direction = theta_optimal - theta
    advantage = tracking_performance(theta)
    return lambda_weight * direction * np.abs(advantage)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

theta1 = np.linspace(-1, 1.5, 100)
theta2 = np.linspace(-1, 1.5, 100)
T1, T2 = np.meshgrid(theta1, theta2)

motion_loss_grid = 0.5 * (T1**2 + T2**2)
tracking_grid = -0.3 * ((T1 - theta_optimal[0])**2 + (T2 - theta_optimal[1])**2)
total_loss_grid = motion_loss_grid - tracking_grid

# Plot 1: Motion fidelity loss
ax1 = axes[0]
contour1 = ax1.contour(T1, T2, motion_loss_grid, levels=15, cmap='Blues')
ax1.clabel(contour1, inline=True, fontsize=8)
ax1.plot(0, 0, 'r*', markersize=15, label='Original motion')
ax1.set_xlabel('θ₁ (joint offset 1)')
ax1.set_ylabel('θ₂ (joint offset 2)')
ax1.set_title('Motion Fidelity Loss\n(prefers θ = 0)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Tracking performance
ax2 = axes[1]
contour2 = ax2.contour(T1, T2, -tracking_grid, levels=15, cmap='Reds')
ax2.clabel(contour2, inline=True, fontsize=8)
ax2.plot(theta_optimal[0], theta_optimal[1], 'g*', markersize=15, label='Optimal tracking')
ax2.set_xlabel('θ₁ (joint offset 1)')
ax2.set_ylabel('θ₂ (joint offset 2)')
ax2.set_title('Tracking Difficulty\n(prefers θ = [0.5, 0.3])')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Plot 3: Combined objective with gradient flow
ax3 = axes[2]
contour3 = ax3.contour(T1, T2, total_loss_grid, levels=15, cmap='viridis')
ax3.clabel(contour3, inline=True, fontsize=8)
ax3.plot(0, 0, 'r*', markersize=12, label='Original motion', alpha=0.7)
ax3.plot(theta_optimal[0], theta_optimal[1], 'g*', markersize=12, label='Optimal tracking', alpha=0.7)

theta_current = np.array([-0.5, -0.5])
trajectory = [theta_current.copy()]
learning_rate = 0.15
lambda_weight = 0.1

for _ in range(20):
    grad_motion = motion_gradient(theta_current)
    grad_policy = policy_gradient_term(theta_current, lambda_weight)
    theta_current = theta_current - learning_rate * (grad_motion + grad_policy)
    trajectory.append(theta_current.copy())

trajectory = np.array(trajectory)
ax3.plot(trajectory[:, 0], trajectory[:, 1], 'ko-', linewidth=2, markersize=4, label='Bilevel optimization', alpha=0.7)
ax3.plot(trajectory[0, 0], trajectory[0, 1], 'mo', markersize=10, label='Start')
ax3.plot(trajectory[-1, 0], trajectory[-1, 1], 'co', markersize=10, label='Converged')

ax3.set_xlabel('θ₁ (joint offset 1)')
ax3.set_ylabel('θ₂ (joint offset 2)')
ax3.set_title(f'Bilevel Optimization (λ={lambda_weight})\nBalances fidelity & tracking')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.tight_layout()
plt.savefig('bilevel_gradient.png', dpi=150, bbox_inches='tight')
print("✓ Saved bilevel_gradient.png")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print(f"1. Motion fidelity loss pulls toward θ=0 (original motion)")
print(f"2. Tracking performance is best at θ={theta_optimal} (robot-specific)")
print(f"3. Bilevel optimization finds compromise between both")
print(f"4. Final converged point: θ={trajectory[-1]}")
print("="*60)

plt.show()
