"""
1D 频率-振幅耦合可视化

演示在标准物理下，频率和振幅是耦合的：
- 提高刚度 k → 频率增大，振幅减小
- 降低刚度 k → 频率减小，振幅增大
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 物理参数
m = 1.0  # 质量
c = 0.1  # 阻尼系数
v0 = 1.0  # 初始速度（脉冲加速度）

# 三种刚度
k_values = [10.0, 20.0, 40.0]  # 低、中、高刚度
colors = ['blue', 'green', 'red']
labels = ['低刚度 (k=10)', '中刚度 (k=20)', '高刚度 (k=40)']

# 时间轴
t = np.linspace(0, 10, 1000)

# 计算每种刚度下的响应
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

for k, color, label in zip(k_values, colors, labels):
    # 阻尼固有频率
    omega_d = np.sqrt(4*m*k - c**2) / (2*m)
    
    # 位移响应（Eq. 12 from paper）
    z = (v0 / omega_d) * np.exp(-c/(2*m) * t) * np.sin(omega_d * t)
    
    # 频率和振幅
    freq = omega_d / (2 * np.pi)
    amplitude = v0 / omega_d
    
    # 绘制时域响应
    ax1.plot(t, z, color=color, label=f'{label}\n频率={freq:.2f} Hz, 振幅={amplitude:.3f}')
    
    # 绘制频率-振幅关系
    ax2.scatter(freq, amplitude, s=200, color=color, marker='o', edgecolors='black', linewidths=2)

# 时域图
ax1.set_xlabel('时间 (s)', fontsize=12)
ax1.set_ylabel('位移 z(t)', fontsize=12)
ax1.set_title('1D 弹簧质点系统的时域响应', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# 频率-振幅关系图
ax2.set_xlabel('频率 (Hz)', fontsize=12)
ax2.set_ylabel('振幅', fontsize=12)
ax2.set_title('频率-振幅耦合：振幅 ∝ 1/ω_d', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 添加理论曲线
freq_theory = np.linspace(0.3, 1.5, 100)
amplitude_theory = v0 / (2 * np.pi * freq_theory)
ax2.plot(freq_theory, amplitude_theory, 'k--', linewidth=2, label='理论曲线: A = v₀/(2πν)')
ax2.legend(fontsize=10)

# 添加注释
ax2.annotate('提高刚度 k\n→ 频率增大\n→ 振幅减小', 
             xy=(1.0, 0.16), xytext=(1.2, 0.35),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/ligo/claude-papers/papers/adjustable-constrained-soft-tissue/images/coupling_demo.png', dpi=150, bbox_inches='tight')
print("图片已保存到 images/coupling_demo.png")
print("\n关键观察：")
print("- 低刚度 (k=10)：频率低 (0.50 Hz)，振幅大 (0.318)")
print("- 中刚度 (k=20)：频率中 (0.71 Hz)，振幅中 (0.225)")
print("- 高刚度 (k=40)：频率高 (1.01 Hz)，振幅小 (0.159)")
print("\n结论：仅通过调整 k 无法独立控制频率和振幅！")
