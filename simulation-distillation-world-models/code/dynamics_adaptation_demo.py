"""
SimDist 动力学适应演示

演示核心概念：
1. 仿真动力学 vs 真实动力学的差异
2. 监督式动力学微调过程
3. 适应前后的预测精度对比

使用简单的弹簧-阻尼系统作为示例。
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# === 物理系统 ===

class SpringDamperSystem:
    """弹簧-阻尼系统（模拟机器人关节动力学）"""

    def __init__(self, k=2.0, c=0.5, m=1.0, dt=0.05):
        self.k = k  # 弹簧刚度
        self.c = c  # 阻尼系数
        self.m = m  # 质量
        self.dt = dt

    def step(self, state, action):
        """state = [position, velocity], action = force"""
        x, v = state
        a = (action - self.k * x - self.c * v) / self.m
        v_new = v + a * self.dt
        x_new = x + v_new * self.dt
        return np.array([x_new, v_new])

    def rollout(self, state, actions):
        trajectory = [state]
        current = state.copy()
        for a in actions:
            current = self.step(current, a)
            trajectory.append(current)
        return np.array(trajectory)


# === 潜在动力学模型（神经网络简化为线性模型）===

class LatentDynamicsModel:
    """简化的潜在动力学模型 f_θ"""

    def __init__(self, state_dim=2, action_dim=1):
        self.A = np.eye(state_dim) + np.random.randn(state_dim, state_dim) * 0.1
        self.B = np.random.randn(state_dim, action_dim) * 0.1

    def predict(self, state, action):
        action = np.atleast_1d(action)
        return self.A @ state + self.B @ action

    def predict_trajectory(self, state, actions):
        trajectory = [state]
        current = state.copy()
        for a in actions:
            current = self.predict(current, a)
            trajectory.append(current)
        return np.array(trajectory)

    def finetune(self, states, actions, targets, lr=0.01, epochs=100):
        """
        监督式动力学微调（SimDist 的核心适应机制）
        最小化 L_real = Σ‖f_θ(z_t, a_t) - z_{t+1}‖²
        """
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for s, a, t in zip(states, actions, targets):
                pred = self.predict(s, a)
                error = pred - t
                total_loss += np.sum(error ** 2)

                # 梯度下降更新 A 和 B
                a_arr = np.atleast_1d(a)
                grad_A = np.outer(error, s)
                grad_B = np.outer(error, a_arr)
                self.A -= lr * grad_A
                self.B -= lr * grad_B

            losses.append(total_loss / len(states))
        return losses


def main():
    # 仿真系统（用于预训练）
    sim_system = SpringDamperSystem(k=2.0, c=0.5, m=1.0)

    # 真实系统（参数不同 — 这就是 sim-to-real gap）
    real_system = SpringDamperSystem(k=3.5, c=0.8, m=1.2)

    # 生成动作序列
    T = 60
    actions = np.sin(np.linspace(0, 4 * np.pi, T)) * 2.0

    # 仿真轨迹
    init_state = np.array([1.0, 0.0])
    sim_traj = sim_system.rollout(init_state, actions)
    real_traj = real_system.rollout(init_state, actions)

    # === Phase 1: 在仿真数据上预训练动力学模型 ===
    model = LatentDynamicsModel()

    # 用仿真数据预训练
    sim_states = sim_traj[:-1]
    sim_targets = sim_traj[1:]
    pretrain_losses = model.finetune(sim_states, actions, sim_targets, lr=0.005, epochs=200)

    # 预训练后在真实数据上的预测
    pred_before = model.predict_trajectory(init_state, actions)

    # === Phase 2: 用少量真实数据微调（SimDist 的核心）===
    # 只用前 15 步真实数据（模拟 "15 分钟真实数据"）
    n_real = 15
    real_states_few = real_traj[:n_real]
    real_targets_few = real_traj[1:n_real + 1]
    real_actions_few = actions[:n_real]

    finetune_losses = model.finetune(
        real_states_few, real_actions_few, real_targets_few,
        lr=0.003, epochs=150
    )

    # 微调后的预测
    pred_after = model.predict_trajectory(init_state, actions)

    # === 可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1：位置预测对比
    ax = axes[0, 0]
    time = np.arange(T + 1) * sim_system.dt
    ax.plot(time, real_traj[:, 0], 'k-', linewidth=2, label='Real System')
    ax.plot(time, sim_traj[:, 0], 'b--', linewidth=1.5, label='Sim System')
    ax.plot(time, pred_before[:, 0], 'r:', linewidth=1.5, label='Model (Before Adapt.)')
    ax.plot(time, pred_after[:, 0], 'g-', linewidth=1.5, label='Model (After Adapt.)')
    ax.axvline(n_real * sim_system.dt, color='orange', linestyle='--', alpha=0.7,
               label=f'Adaptation data boundary ({n_real} steps)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.set_title('Position Prediction: Before vs After Dynamics Adaptation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图 2：预测误差
    ax = axes[0, 1]
    error_before = np.abs(pred_before[:, 0] - real_traj[:, 0])
    error_after = np.abs(pred_after[:, 0] - real_traj[:, 0])
    ax.fill_between(time, error_before, alpha=0.3, color='red', label='Before Adaptation')
    ax.fill_between(time, error_after, alpha=0.3, color='green', label='After Adaptation')
    ax.plot(time, error_before, 'r-', linewidth=1)
    ax.plot(time, error_after, 'g-', linewidth=1)
    ax.axvline(n_real * sim_system.dt, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Prediction Error|')
    ax.set_title('Prediction Error Reduction via Dynamics Finetuning')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图 3：微调损失曲线
    ax = axes[1, 0]
    ax.semilogy(pretrain_losses, 'b-', alpha=0.7, label='Pretrain (Sim Data)')
    ax.semilogy(range(len(pretrain_losses), len(pretrain_losses) + len(finetune_losses)),
                finetune_losses, 'g-', linewidth=2, label='Finetune (Real Data)')
    ax.axvline(len(pretrain_losses), color='orange', linestyle='--',
               label='Switch to Real Data')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('Training Loss: Pretrain → Finetune')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图 4：相空间轨迹
    ax = axes[1, 1]
    ax.plot(real_traj[:, 0], real_traj[:, 1], 'k-', linewidth=2, label='Real')
    ax.plot(pred_before[:, 0], pred_before[:, 1], 'r:', linewidth=1.5, label='Before')
    ax.plot(pred_after[:, 0], pred_after[:, 1], 'g-', linewidth=1.5, label='After')
    ax.plot(init_state[0], init_state[1], 'ko', markersize=10)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase Space: Dynamics Adaptation Corrects Trajectory Shape')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("SimDist: Supervised Dynamics Adaptation\n"
                 f"(Only {n_real} real-world steps needed for significant improvement)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/simulation-distillation-world-models/code/dynamics_adaptation_demo.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # 打印统计
    mse_before = np.mean((pred_before - real_traj) ** 2)
    mse_after = np.mean((pred_after - real_traj) ** 2)
    print(f"Dynamics Adaptation Results:")
    print(f"  MSE before adaptation: {mse_before:.4f}")
    print(f"  MSE after adaptation:  {mse_after:.4f}")
    print(f"  Improvement: {(1 - mse_after/mse_before)*100:.1f}%")
    print(f"  Real data used: {n_real} steps (out of {T} total)")
    print(f"\nKey insight: SimDist reduces real-world adaptation to supervised learning.")
    print(f"No reward labels, no credit assignment, no policy gradients needed.")


if __name__ == "__main__":
    main()
