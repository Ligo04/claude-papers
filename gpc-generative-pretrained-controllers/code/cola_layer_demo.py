"""
CoLA (Conditional Low-rank Adaptation) demo —— GPC Stage 3 的 PEFT
====================================================================

对照论文式 (7):
  Wx = W₀x + m · [ B(diag(γ(c))·Ax + β(c)) ] / ‖ B(diag(γ(c))·Ax + β(c)) ‖_F

CoLA = DoRA(幅度/方向解耦低秩更新) + FiLM(任务条件调制)。
本 demo:
  1. 实现 CoLA 前向, 逐项对应公式;
  2. 统计"新增参数 < 1%"(冻结主干 W₀, 只训 A,B,m,γ,β);
  3. 展示同一套 CoLA 参数在不同任务观测 c 下产生不同的等效权重(条件化)。

纯 NumPy, 直接运行:  python cola_layer_demo.py
"""

import numpy as np


class TinyMLP:
    """FiLM 用的轻量 MLP: 从任务观测 c 生成 γ(c) 或 β(c) ∈ R^r。"""
    def __init__(self, c_dim, r, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.5, size=(r, c_dim))
        self.b = np.zeros(r)

    def __call__(self, c):
        return np.tanh(self.W @ c + self.b)

    def n_params(self):
        return self.W.size + self.b.size


class CoLALayer:
    """条件低秩适配层, 包裹一个冻结的预训练权重 W0。"""

    def __init__(self, d_in, d_out, r, c_dim, seed=0):
        rng = np.random.default_rng(seed)
        # ---- 冻结的预训练主干(不训练) ----
        self.W0 = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in))
        self.W0.flags.writeable = False   # 强调冻结

        # ---- 可训练的 CoLA 参数(< 1%) ----
        self.A = rng.normal(0, 0.02, size=(r, d_in))      # 低秩下投影
        # 论文中 B 初始化为 0 ⇒ 训练起点严格等于 W0(不破坏自然行为);
        # 这里给一个小的非零初始化, 模拟"微调若干步后"的状态, 以便展示任务条件化效果。
        self.B = rng.normal(0, 0.02, size=(d_out, r))      # 低秩上投影
        self.m = np.linalg.norm(self.W0, axis=1)           # DoRA 幅度向量(每行范数初始化)
        self.gamma_mlp = TinyMLP(c_dim, r, seed + 1)       # FiLM 缩放
        self.beta_mlp = TinyMLP(c_dim, r, seed + 2)        # FiLM 偏移

        self.d_in, self.d_out, self.r = d_in, d_out, r

    def forward(self, x, c):
        """式 (7) 的逐项实现。x: (d_in,) 输入; c: (c_dim,) 任务观测。"""
        gamma = self.gamma_mlp(c)                  # γ(c) ∈ R^r
        beta = self.beta_mlp(c)                    # β(c) ∈ R^r

        base = self.W0 @ x                          # W₀x  (冻结主干)

        # 低秩 + FiLM 条件调制: B( diag(γ)·A x + β )
        low = self.B @ (gamma * (self.A @ x) + beta)      # (d_out,)

        # DoRA 幅度/方向解耦: 方向 = low / ‖low‖_F, 再乘幅度 m
        norm = np.linalg.norm(low) + 1e-8
        direction = low / norm
        delta = self.m * direction                  # m ⊙ direction

        return base + delta

    def param_report(self):
        trainable = (self.A.size + self.B.size + self.m.size
                     + self.gamma_mlp.n_params() + self.beta_mlp.n_params())
        frozen = self.W0.size
        return trainable, frozen


def main():
    # 取一个类似 transformer 隐藏层的规模(真实主干越大, PEFT 占比越低)
    d_in, d_out, r, c_dim = 2048, 2048, 4, 8
    layer = CoLALayer(d_in, d_out, r, c_dim, seed=0)

    print("=" * 60)
    print("CoLA 层配置")
    print(f"  主干 W₀ : {d_out}×{d_in} (冻结)")
    print(f"  低秩 r  : {r}   (r ≪ min(d_in,d_out))")
    print(f"  任务观测 c 维度: {c_dim}")
    print("=" * 60)

    trainable, frozen = layer.param_report()
    ratio = 100 * trainable / frozen
    print(f"\n[参数占比]  可训练(CoLA) = {trainable:,}")
    print(f"            冻结(主干)   = {frozen:,}")
    print(f"            新增占比     = {ratio:.2f}%   (论文: < 1%)")
    if ratio < 1.0:
        print("            ✓ 满足 '< 1% 额外参数'")

    # 同一套 CoLA 参数, 不同任务观测 c → 不同输出(条件化)
    x = np.random.default_rng(3).normal(0, 1, size=d_in)
    tasks = {
        "任务A: 转向": np.array([1., 0, 0, 0, 1, 0, 0, 0]),
        "任务B: 轨迹": np.array([0., 1, 0, 0, 0, 1, 0, 0]),
        "任务C: 障碍": np.array([0., 0, 1, 0, 0, 0, 1, 0]),
    }
    outputs = {name: layer.forward(x, c) for name, c in tasks.items()}
    base = layer.W0 @ x

    print("\n[条件化验证]  固定输入 x, 改变任务观测 c → 输出随任务而变。")
    print("  每个任务相对冻结主干的偏移范数(DoRA 幅度 m 使其量级相近):")
    for name, y in outputs.items():
        print(f"    {name}: ‖y - W₀x‖ = {np.linalg.norm(y - base):.3f}")
    print("  但*不同任务之间*的输出确实不同(FiLM 让方向随 c 变化):")
    names = list(outputs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = np.linalg.norm(outputs[names[i]] - outputs[names[j]])
            print(f"    ‖{names[i]} − {names[j]}‖ = {diff:.3f}")

    print("\n要点:")
    print("  - 论文中 B 初始化为 0 ⇒ 训练起点严格等于预训练主干(不破坏自然行为)。")
    print("  - γ(c),β(c) 让'同一低秩子空间'按任务上下文动态调制(FiLM)。")
    print("  - m(幅度) 与方向解耦(DoRA) ⇒ 优化更稳定。")
    print("  - 冻结 W₀ ⇒ 预训练技能分布被保住, 迁移新任务而'不忘本'。")


if __name__ == "__main__":
    main()
