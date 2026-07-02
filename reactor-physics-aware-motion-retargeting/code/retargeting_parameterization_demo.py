"""
Retargeting Parameterization Demo

Visualizes how ReActor's sparse parameterization adapts human motion
to robot morphology using constant per-joint offsets.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simple 2-link arm example
class Arm:
    def __init__(self, l1, l2, name):
        self.l1 = l1  # Upper arm length
        self.l2 = l2  # Forearm length
        self.name = name
    
    def forward_kinematics(self, q1, q2):
        """Compute end effector position from joint angles"""
        x1 = self.l1 * np.cos(q1)
        y1 = self.l1 * np.sin(q1)
        x2 = x1 + self.l2 * np.cos(q1 + q2)
        y2 = y1 + self.l2 * np.sin(q1 + q2)
        return np.array([[0, x1, x2], [0, y1, y2]])

# Human and robot arms with different proportions
human = Arm(l1=1.0, l2=0.8, name="Human")
robot = Arm(l1=0.7, l2=0.5, name="Robot")

# Motion sequence: reaching motion
t = np.linspace(0, 2*np.pi, 50)
q1_human = 0.3 * np.sin(t) + 0.5
q2_human = 0.4 * np.cos(t) + 0.3

# Nominal retargeting (just scale)
scale = (robot.l1 + robot.l2) / (human.l1 + human.l2)
q1_nominal = q1_human * scale
q2_nominal = q2_human * scale

# Learned offsets (constant per joint)
delta_q1 = 0.15  # Learned offset for joint 1
delta_q2 = -0.20  # Learned offset for joint 2
q1_adapted = q1_nominal + delta_q1
q2_adapted = q2_nominal + delta_q2

# Visualization
fig = plt.figure(figsize=(16, 5))

# Plot 1: Human motion
ax1 = fig.add_subplot(131)
for i in [0, 12, 24, 36, 48]:
    pos = human.forward_kinematics(q1_human[i], q2_human[i])
    ax1.plot(pos[0], pos[1], 'o-', linewidth=2, markersize=6, alpha=0.6, color=plt.cm.Blues(i/48))
ax1.set_xlim(-0.5, 2.0)
ax1.set_ylim(-0.5, 2.0)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Human Motion (Source)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X position')
ax1.set_ylabel('Y position')

# Plot 2: Nominal retargeting (scaled)
ax2 = fig.add_subplot(132)
for i in [0, 12, 24, 36, 48]:
    pos_human = human.forward_kinematics(q1_human[i], q2_human[i])
    pos_nominal = robot.forward_kinematics(q1_nominal[i], q2_nominal[i])
    ax2.plot(pos_human[0], pos_human[1], 'o-', linewidth=1, markersize=4, alpha=0.3, color='gray', label='Human' if i==0 else '')
    ax2.plot(pos_nominal[0], pos_nominal[1], 'o-', linewidth=2, markersize=6, alpha=0.6, color=plt.cm.Reds(i/48))
ax2.set_xlim(-0.5, 2.0)
ax2.set_ylim(-0.5, 2.0)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('Nominal Retargeting (Scaled)\nNo learned offsets', fontsize=12, fontweight='bold')
ax2.set_xlabel('X position')
ax2.legend(['Human (reference)', 'Robot (nominal)'], fontsize=8)

# Plot 3: Adapted retargeting (with learned offsets)
ax3 = fig.add_subplot(133)
for i in [0, 12, 24, 36, 48]:
    pos_human = human.forward_kinematics(q1_human[i], q2_human[i])
    pos_adapted = robot.forward_kinematics(q1_adapted[i], q2_adapted[i])
    ax3.plot(pos_human[0], pos_human[1], 'o-', linewidth=1, markersize=4, alpha=0.3, color='gray')
    ax3.plot(pos_adapted[0], pos_adapted[1], 'o-', linewidth=2, markersize=6, alpha=0.6, color=plt.cm.Greens(i/48))
ax3.set_xlim(-0.5, 2.0)
ax3.set_ylim(-0.5, 2.0)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.set_title(f'Adapted Retargeting\nΔq₁={delta_q1:+.2f}, Δq₂={delta_q2:+.2f}', fontsize=12, fontweight='bold')
ax3.set_xlabel('X position')
ax3.legend(['Human (reference)', 'Robot (adapted)'], fontsize=8)

plt.tight_layout()
plt.savefig('retargeting_parameterization.png', dpi=150, bbox_inches='tight')
print("✓ Saved retargeting_parameterization.png")

# Compute tracking errors
pos_human_all = np.array([human.forward_kinematics(q1_human[i], q2_human[i])[:, -1] for i in range(len(t))])
pos_nominal_all = np.array([robot.forward_kinematics(q1_nominal[i], q2_nominal[i])[:, -1] for i in range(len(t))])
pos_adapted_all = np.array([robot.forward_kinematics(q1_adapted[i], q2_adapted[i])[:, -1] for i in range(len(t))])

error_nominal = np.mean(np.linalg.norm(pos_human_all - pos_nominal_all, axis=1))
error_adapted = np.mean(np.linalg.norm(pos_human_all - pos_adapted_all, axis=1))

print("\n" + "="*60)
print("TRACKING ERRORS")
print("="*60)
print(f"Nominal retargeting error: {error_nominal:.4f}")
print(f"Adapted retargeting error: {error_adapted:.4f}")
print(f"Improvement: {(1 - error_adapted/error_nominal)*100:.1f}%")
print("="*60)
print("\nKEY INSIGHT:")
print("Constant per-joint offsets (Δq₁, Δq₂) adapt the motion to")
print("robot's morphology while preserving motion characteristics.")
print("="*60)

plt.show()
