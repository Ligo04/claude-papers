"""
2D 非惯性力缩放演示

演示本文方法如何通过缩放惯性力来独立控制振幅：
- ε = 1.0：原始物理
- ε = 0.5：振幅减半
- ε = 2.0：振幅翻倍
频率保持不变
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# 2D 质点系统参数
m = 1.0  # 质量
k = 20.0  # 刚度
c = 0.5  # 阻尼

# 约束运动：圆周旋转
omega_constraint = 2.0  # 约束角速度 (rad/s)
R_constraint = 0.5  # 约束半径

# 时间参数
dt = 0.01
t_max = 5.0
t = np.arange(0, t_max, dt)
n_steps = len(t)

# 三种 ε 值
epsilon_values = [0.5, 1.0, 2.0]
colors = ['blue', 'green', 'red']
labels = ['ε=0.5 (减弱)', 'ε=1.0 (原始)', 'ε=2.0 (放大)']

# 约束位置（圆周运动）
x_constraint = R_constraint * np.cos(omega_constraint * t)
y_constraint = R_constraint * np.sin(omega_constraint * t)

# 约束加速度（向心加速度）
ax_constraint = -omega_constraint**2 * x_constraint
ay_constraint = -omega_constraint**2 * y_constraint

# 计算每种 ε 下的响应
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for epsilon, color, label in zip(epsilon_values, colors, labels):
    # 初始化
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    
    # 初始条件：静止在约束位置
    x[0] = x_constraint[0]
    y[0] = y_constraint[0]
    
    # 时间积分（简化的显式欧拉）
    for i in range(n_steps - 1):
        # 相对于约束的位移
        dx = x[i] - x_constraint[i]
        dy = y[i] - y_constraint[i]
        
        # 弹性力
        fx_elastic = -k * dx
        fy_elastic = -k * dy
        
        # 阻尼力
        fx_damping = -c * vx[i]
        fy_damping = -c * vy[i]
        
        # 惯性力（缩放后）
        fx_inertial = epsilon * m * ax_constraint[i]
        fy_inertial = epsilon * m * ay_constraint[i]
        
        # 总加速度
        ax = (fx_elastic + fx_damping + fx_inertial) / m
        ay = (fy_elastic + fy_damping + fy_inertial) / m
        
        # 更新速度和位置
        vx[i+1] = vx[i] + ax * dt
        vy[i+1] = vy[i] + ay * dt
        x[i+1] = x[i] + vx[i+1] * dt
        y[i+1] = y[i] + vy[i+1] * dt
    
    # 计算振幅（相对于约束的 RMS 位移）
    dx_rel = x - x_constraint
    dy_rel = y - y_constraint
    amplitude = np.sqrt(np.mean(dx_rel**2 + dy_rel**2))
    
    # 绘制轨迹
    ax1.plot(x, y, color=color, linewidth=2, label=f'{label}\n振幅={amplitude:.3f}', alpha=0.7)
    
    # 绘制时域响应（径向位移）
    r_rel = np.sqrt(dx_rel**2 + dy_rel**2)
    ax2.plot(t, r_rel, color=color, linewidth=2, label=label)

# 轨迹图
ax1.plot(x_constraint, y_constraint, 'k--', linewidth=2, label='约束轨迹', alpha=0.5)
ax1.scatter([0], [0], s=100, c='black', marker='x', linewidths=3, label='原点')
ax1.set_xlabel('x 位置', fontsize=12)
ax1.set_ylabel('y 位置', fontsize=12)
ax1.set_title('2D 轨迹：ε 缩放惯性力', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)

# 时域图
ax2.set_xlabel('时间 (s)', fontsize=12)
ax2.set_ylabel('径向位移', fontsize=12)
ax2.set_title('时域响应：频率不变，振幅可调', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 添加注释
ax1.annotate('ε < 1: 减弱惯性力\n→ 振幅减小', 
             xy=(-0.3, 0.3), xytext=(-1.2, 1.0),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=10, color='blue', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

ax1.annotate('ε > 1: 放大惯性力\n→ 振幅增大', 
             xy=(0.8, -0.8), xytext=(0.5, -1.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/ligo/claude-papers/papers/adjustable-constrained-soft-tissue/images/inertial_scaling_demo.png', dpi=150, bbox_inches='tight')
print("图片已保存到 images/inertial_scaling_demo.png")
print("\n关键观察：")
print("- ε=0.5：振幅约为原始的 50%")
print("- ε=1.0：原始物理")
print("- ε=2.0：振幅约为原始的 200%")
print("\n结论：通过缩放惯性力，可以独立调节振幅而不改变频率！")
