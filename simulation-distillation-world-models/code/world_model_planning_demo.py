"""
SimDist 世界模型规划演示

演示核心概念：
1. 潜在空间动力学预测
2. MPPI 采样规划
3. 奖励/价值评估轨迹

这是一个简化的 2D 版本，展示 SimDist 的规划机制。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

np.random.seed(42)

# === 简化的 2D 世界模型 ===

class SimpleWorldModel:
    """模拟 SimDist 的世界模型结构（2D 简化版）"""

    def __init__(self, goal, dynamics_noise=0.1):
        self.goal = np.array(goal)
        self.dynamics_noise = dynamics_noise

    def dynamics(self, state, action):
        """潜在动力学模型 f_θ：预测下一状态"""
        next_state = state + action + np.random.randn(2) * self.dynamics_noise
        return next_state

    def predict_trajectory(self, state, actions):
        """分块预测 T 步未来状态（SimDist 用 Transformer 一次性预测）"""
        trajectory = [state]
        current = state.copy()
        for a in actions:
            current = self.dynamics(current, a)
            trajectory.append(current)
        return np.array(trajectory)

    def reward(self, state):
        """奖励模型 R_θ：基于距离目标的距离"""
        dist = np.linalg.norm(state - self.goal)
        return np.exp(-0.5 * dist)

    def value(self, state):
        """价值模型 V_θ：长时域回报估计"""
        dist = np.linalg.norm(state - self.goal)
        return np.exp(-0.3 * dist)

    def evaluate_trajectory(self, trajectory, gamma=0.95):
        """评估轨迹回报：R = Σγ^t·r_t + γ^T·V(s_T)"""
        total_return = 0
        for t, state in enumerate(trajectory[:-1]):
            total_return += (gamma ** t) * self.reward(state)
        total_return += (gamma ** len(trajectory)) * self.value(trajectory[-1])
        return total_return


# === MPPI 规划器 ===

class MPPIPlanner:
    """Model Predictive Path Integral 规划器"""

    def __init__(self, world_model, horizon=8, num_samples=200, temperature=0.1):
        self.model = world_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature

    def plan(self, current_state, base_policy_mean=None):
        """
        MPPI 规划：
        1. 采样候选动作序列（部分由基础策略热启动）
        2. 用世界模型预测轨迹
        3. 按回报加权得到最优动作
        """
        if base_policy_mean is None:
            base_policy_mean = np.zeros((self.horizon, 2))

        # 采样候选动作序列
        action_samples = base_policy_mean[None, :, :] + \
            np.random.randn(self.num_samples, self.horizon, 2) * 0.3

        # 预测所有候选轨迹
        trajectories = []
        returns = []
        for i in range(self.num_samples):
            traj = self.model.predict_trajectory(current_state, action_samples[i])
            trajectories.append(traj)
            ret = self.model.evaluate_trajectory(traj)
            returns.append(ret)

        returns = np.array(returns)
        trajectories = np.array(trajectories)

        # 重要性加权
        weights = np.exp((returns - returns.max()) / self.temperature)
        weights /= weights.sum()

        # 加权平均得到最优动作序列
        optimal_actions = np.sum(
            weights[:, None, None] * action_samples, axis=0
        )

        return optimal_actions, trajectories, weights


# === 可视化 ===

def visualize_planning(ax, state, goal, trajectories, weights, optimal_traj, title):
    """可视化 MPPI 规划过程"""
    # 绘制候选轨迹（按权重着色）
    top_k = 30
    top_indices = np.argsort(weights)[-top_k:]

    for idx in top_indices:
        traj = trajectories[idx]
        alpha = min(weights[idx] * top_k * 2, 0.8)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=alpha, linewidth=0.5)

    # 绘制最优轨迹
    ax.plot(optimal_traj[:, 0], optimal_traj[:, 1], 'r-', linewidth=2.5, label='Optimal Plan')
    ax.plot(optimal_traj[0, 0], optimal_traj[0, 1], 'ko', markersize=10, label='Current State')

    # 绘制目标
    goal_circle = Circle(goal, 0.15, color='green', alpha=0.3)
    ax.add_patch(goal_circle)
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


def visualize_adaptation():
    """演示动力学适应前后的规划差异"""
    goal = [3.0, 3.0]
    start = np.array([0.5, 0.5])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 场景 1：仿真动力学（无噪声）
    sim_model = SimpleWorldModel(goal, dynamics_noise=0.02)
    planner = MPPIPlanner(sim_model, horizon=10, num_samples=300)
    actions, trajs, weights = planner.plan(start)
    optimal_traj = sim_model.predict_trajectory(start, actions)
    visualize_planning(axes[0], start, goal, trajs, weights, optimal_traj,
                       "Pretrained (Sim Dynamics)\nLow noise, accurate predictions")

    # 场景 2：真实世界动力学（高噪声，未适应）
    real_model_before = SimpleWorldModel(goal, dynamics_noise=0.25)
    planner2 = MPPIPlanner(real_model_before, horizon=10, num_samples=300)
    actions2, trajs2, weights2 = planner2.plan(start)
    optimal_traj2 = real_model_before.predict_trajectory(start, actions2)
    visualize_planning(axes[1], start, goal, trajs2, weights2, optimal_traj2,
                       "Before Adaptation (Real Dynamics)\nHigh noise, inaccurate predictions")

    # 场景 3：适应后（噪声降低）
    real_model_after = SimpleWorldModel(goal, dynamics_noise=0.08)
    planner3 = MPPIPlanner(real_model_after, horizon=10, num_samples=300)
    actions3, trajs3, weights3 = planner3.plan(start)
    optimal_traj3 = real_model_after.predict_trajectory(start, actions3)
    visualize_planning(axes[2], start, goal, trajs3, weights3, optimal_traj3,
                       "After Adaptation (Finetuned)\nReduced noise, improved planning")

    plt.suptitle("SimDist: MPPI Planning with World Model Dynamics Adaptation",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/simulation-distillation-world-models/code/planning_demo.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Planning demo saved to code/planning_demo.png")


def visualize_value_transfer():
    """演示价值函数的排序迁移特性"""
    goal = np.array([3.0, 3.0])

    # 创建状态网格
    x = np.linspace(0, 4, 50)
    y = np.linspace(0, 4, 50)
    X, Y = np.meshgrid(x, y)

    # 仿真中的价值函数
    sim_model = SimpleWorldModel(goal, dynamics_noise=0.02)
    V_sim = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            V_sim[i, j] = sim_model.value(np.array([X[i, j], Y[i, j]]))

    # 真实世界的"真实"价值（加入偏移但保持排序）
    V_real = V_sim * 0.7 + 0.1  # 缩放+偏移，但排序不变

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].contourf(X, Y, V_sim, levels=20, cmap='viridis')
    axes[0].plot(goal[0], goal[1], 'r*', markersize=15)
    axes[0].set_title('Sim Value Function V_sim(s)', fontsize=11)
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].contourf(X, Y, V_real, levels=20, cmap='viridis')
    axes[1].plot(goal[0], goal[1], 'r*', markersize=15)
    axes[1].set_title('Real Value (scaled, shifted)\nRanking preserved!', fontsize=11)
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # 排序一致性可视化
    n_pairs = 500
    states_i = np.random.rand(n_pairs, 2) * 4
    states_j = np.random.rand(n_pairs, 2) * 4
    v_sim_i = np.array([sim_model.value(s) for s in states_i])
    v_sim_j = np.array([sim_model.value(s) for s in states_j])
    v_real_i = v_sim_i * 0.7 + 0.1 + np.random.randn(n_pairs) * 0.02
    v_real_j = v_sim_j * 0.7 + 0.1 + np.random.randn(n_pairs) * 0.02

    sim_ranking = v_sim_i > v_sim_j
    real_ranking = v_real_i > v_real_j
    concordance = (sim_ranking == real_ranking).mean()

    axes[2].scatter(v_sim_i - v_sim_j, v_real_i - v_real_j, alpha=0.3, s=10)
    axes[2].axhline(0, color='k', linewidth=0.5)
    axes[2].axvline(0, color='k', linewidth=0.5)
    axes[2].set_xlabel('V_sim(s_i) - V_sim(s_j)')
    axes[2].set_ylabel('V_real(s_i) - V_real(s_j)')
    axes[2].set_title(f'Rank Concordance: {concordance:.1%}\n(Key insight: ranking transfers!)', fontsize=11)

    plt.suptitle("SimDist: Value Function Transfer — Ranking > Calibration",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/simulation-distillation-world-models/code/value_transfer_demo.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Value transfer demo saved. Rank concordance: {concordance:.1%}")


if __name__ == "__main__":
    visualize_adaptation()
    visualize_value_transfer()
    print("\nDemos complete. Key takeaways:")
    print("1. MPPI samples many trajectories and weights them by predicted return")
    print("2. Dynamics adaptation reduces prediction noise → sharper planning")
    print("3. Value function ranking transfers across sim-to-real gap")
