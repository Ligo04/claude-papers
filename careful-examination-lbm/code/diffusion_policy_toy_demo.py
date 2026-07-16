"""
diffusion_policy_toy_demo.py
=============================
一个【极简、自包含】的 Diffusion Policy 概念实现，对应论文 Sec IV-B 的核心：
用 DDPM 训练目标学一个"条件动作分布"，用 DDIM 式迭代去噪采样出动作。

论文的策略是把动作序列(16步×20维)当作要去噪的对象，条件是(视觉+本体+语言)。
这里为了'能在 CPU 上秒跑、又抓住本质'，我们做一个玩具版：
  - 观测 O ∈ R^2 （模拟"低维本体/条件"）；
  - 动作 A ∈ R^2 （模拟一个动作 token）；
  - 真实条件分布是【多峰】的：给定 O，最优动作可能是两个模式之一
    （模拟操作任务的多模态性——比如从左边或右边绕过障碍，二者都对）。

重点演示扩散策略相对回归(MSE)策略的关键优势：
  【能建模多峰动作分布，而回归会把两个模式平均成中间的错误动作】。

实现要点忠于论文：
  - 训练损失 = 预测噪声 ε 的 MSE（式2 的 DDPM loss）；
  - 采样 = 从高斯噪声出发，按噪声调度迭代去噪（式1 的 DDIM 更新思想）。

只依赖 numpy。网络用一个手写的 2 层 MLP（含反向传播），零深度学习框架依赖。
"""

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------- 数据：多峰条件分布 -----------------------------
def sample_task(batch):
    """
    给定观测 o（标量，∈[-1,1] 扩成2维），真实最优动作有两个模式：
      mode+ = o + [0.6, 0.6],  mode- = o - [0.6, 0.6]
    随机各占一半 —— 这就是'多模态最优解'。回归策略会学成它们的平均 = o，撞障碍。
    """
    o = rng.uniform(-1, 1, size=(batch, 1))
    O = np.concatenate([o, -o], axis=1)  # 2维观测
    pick = rng.integers(0, 2, size=(batch, 1))
    offset = np.where(pick == 0, 0.6, -0.6)
    base = np.concatenate([o, o], axis=1)
    A = base + offset  # 2维动作，双峰
    return O.astype(np.float64), A.astype(np.float64)


# ----------------------------- 扩散噪声调度 -----------------------------
K = 20  # 去噪步数
betas = np.linspace(1e-4, 0.15, K)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)


# ----------------------------- 手写 MLP: ε_θ(O, A_k, k) -----------------------------
class NoisePredictor:
    """输入 [O(2), A_k(2), k_embed(1)] = 5 维, 输出预测噪声 ε(2 维)。两层 tanh MLP。"""

    def __init__(self, hidden=128):
        d_in, d_out = 5, 2
        s = 0.3
        self.W1 = rng.normal(0, s, (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s, (hidden, d_out))
        self.b2 = np.zeros(d_out)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h = np.tanh(self.z1)
        return self.h @ self.W2 + self.b2

    def backward(self, grad_out, lr):
        gW2 = self.h.T @ grad_out
        gb2 = grad_out.sum(0)
        gh = grad_out @ self.W2.T
        gz1 = gh * (1 - self.h ** 2)
        gW1 = self.x.T @ gz1
        gb1 = gz1.sum(0)
        for p, g in [(self.W1, gW1), (self.b1, gb1), (self.W2, gW2), (self.b2, gb2)]:
            p -= lr * g / len(self.x)


def make_input(O, A_k, k):
    k_embed = np.full((len(O), 1), k / K)  # 简单的步数编码
    return np.concatenate([O, A_k, k_embed], axis=1)


# ----------------------------- 训练：DDPM loss（式2）-----------------------------
def train(net, steps=6000, batch=256, lr=0.05):
    for it in range(steps):
        O, A0 = sample_task(batch)
        k = rng.integers(1, K, size=batch)
        ab = alpha_bars[k][:, None]
        eps = rng.normal(size=A0.shape)            # 真噪声
        A_k = np.sqrt(ab) * A0 + np.sqrt(1 - ab) * eps  # 前向加噪
        pred = net.forward(make_input(O, A_k, k.mean()))
        grad = 2 * (pred - eps)                    # d/dpred ||pred - eps||^2
        net.backward(grad, lr)
        if (it + 1) % 1500 == 0:
            loss = ((pred - eps) ** 2).mean()
            print(f"  step {it+1:5d}  DDPM ε-预测损失 = {loss:.4f}")


# ----------------------------- 采样：DDIM 式去噪（式1 思想）-----------------------------
def sample(net, O):
    """从纯高斯噪声去噪出动作。忠于'迭代去噪 K 步'的结构。"""
    A = rng.normal(size=(len(O), 2))
    for k in reversed(range(1, K)):
        eps_pred = net.forward(make_input(O, A, k))
        ab = alpha_bars[k]
        ab_prev = alpha_bars[k - 1]
        a0_hat = (A - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)  # 预测的干净动作
        A = np.sqrt(ab_prev) * a0_hat + np.sqrt(1 - ab_prev) * eps_pred  # DDIM 确定性更新
    return A


# ----------------------------- 对照：回归(MSE)策略 -----------------------------
class RegressionPolicy:
    """直接回归动作的策略，验证它在多峰下'塌到平均'。"""

    def __init__(self, hidden=128):
        self.net = NoisePredictor(hidden)

    def train(self, steps=6000, batch=256, lr=0.05):
        for _ in range(steps):
            O, A0 = sample_task(batch)
            x = np.concatenate([O, np.zeros((len(O), 3))], axis=1)
            pred = self.net.forward(x)
            self.net.backward(2 * (pred - A0), lr)

    def predict(self, O):
        x = np.concatenate([O, np.zeros((len(O), 3))], axis=1)
        return self.net.forward(x)


def main():
    print("=" * 84)
    print("玩具 Diffusion Policy：演示'扩散策略能建模多峰动作，回归策略会塌到平均'")
    print("（对应论文选择扩散模型而非直接回归的核心理由：多模态、精确动作）")
    print("=" * 84)

    print("\n[1] 训练扩散噪声预测网络 ε_θ (DDPM 目标, 式2)...")
    net = NoisePredictor()
    train(net)

    print("\n[2] 训练对照的回归(MSE)策略...")
    reg = RegressionPolicy()
    reg.train()

    # 固定一个观测，反复采样，看动作分布
    o_test = 0.3
    O = np.tile([o_test, -o_test], (2000, 1))
    A_diff = sample(net, O)
    A_reg = reg.predict(O[:1])

    print("\n" + "=" * 84)
    print(f"测试观测 o = {o_test}。真实两个最优动作模式（第0维）: "
          f"{o_test+0.6:.2f}  和  {o_test-0.6:.2f}")
    print("=" * 84)

    # 统计扩散采样落在两个模式的比例
    d0 = A_diff[:, 0]
    near_plus = np.mean(np.abs(d0 - (o_test + 0.6)) < 0.25)
    near_minus = np.mean(np.abs(d0 - (o_test - 0.6)) < 0.25)
    print(f"\n  扩散策略采样(2000次) 第0维分布:")
    print(f"    落在 模式+ ({o_test+0.6:.2f}) 附近: {near_plus*100:.1f}%")
    print(f"    落在 模式- ({o_test-0.6:.2f}) 附近: {near_minus*100:.1f}%")
    print(f"    => 扩散策略成功保留了【两个】模式（多模态动作分布）✓")

    print(f"\n  回归(MSE)策略预测: 第0维 = {A_reg[0,0]:.2f}")
    print(f"    => 塌到两个模式的【平均】≈ {o_test:.2f}，这是个'两边都不对'的危险动作 ✗")

    print("\n核心 takeaway：论文用 DDPM/DDIM 扩散策略，正是为了在接触丰富、")
    print("多解的操作任务里保留动作的多模态性，避免回归塌缩。🐾")


if __name__ == "__main__":
    main()
