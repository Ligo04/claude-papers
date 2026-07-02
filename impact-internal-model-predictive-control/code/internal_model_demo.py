#!/usr/bin/env python3
"""
internal_model_demo.py
======================
对 IMPACT 核心控制环的极简、自包含模拟：在线学习的内模型 + 惊讶门控
(surprise gate)，加上前馈补偿，复现论文图 7 的三阶段行为
(IMPACT, Gao et al., arXiv:2606.10818)。

我们把机器人简化为单自由度（1-DOF）的"举升"轴，让物理变得透明，
但 IMPACT 的每个机制都在场且按论文命名：

    * 外部力旋量            w_ext  = m * g           （负载重力）
    * 基于模型的残差        tau_res ~= w_ext         （"虚拟传感器"）
    * 慢/快低通             w_slow = LowPass(w_ext_hat)     （式 2）
    * 内模型               w_pred = M_phi(context)         （2.3 节）
    * 惊讶门控             仅当被惊讶时才更新 M_phi
    * 前馈补偿             w_ff = w_pred  抵消负载
    * 残余位姿误差         阻抗控制器还需对抗的部分

协议（对应论文的分析）：
    阶段 A: 举 2.5 kg，模型幼稚    -> 惊讶触发，模型学习
    阶段 B: 保持 2.5 kg            -> 门控安静，零延迟补偿
    阶段 C: 无预警换成 5.0 kg      -> 惊讶重新触发，模型重新适应

运行：
    python3 internal_model_demo.py            # 打印阶段日志 + ASCII 图
依赖：numpy（仅此）。

为何做这个 demo：它表明 IMPACT 的*有趣*行为根本不是深度学习——
而是一个故意做慢的估计、一个快速学习的预测器、以及一个决定何时
信任两者的门控之间的相互作用。
"""

import numpy as np

g = 9.81  # m/s^2


# ---------------------------------------------------------------------------
# "内模型" M_phi。
# 论文中它把一段关节/末端状态的历史 -> 预测的力旋量。
# 这里唯一改变所需力旋量的是（隐藏的）负载，机器人可以通过它看到的稳态
# 力矩残差来*识别*负载。所以我们用最简单的上下文：当前低通力旋量桶。
# M_phi 是一个用梯度下降在线学习的微型 表/线性映射 —— 正是 2.3 节的精神
# （小、在线、梯度下降）。
# ---------------------------------------------------------------------------
class InternalModel:
    """慢变外部力旋量的在线学习预测器。"""

    def __init__(self, lr=0.30):
        # 单个可学习参数：我们预测的补偿力旋量 (N)。
        # （真实 M_phi 是对状态历史的小型 NN；其学习*动力学*相同 ——
        #  朝监督信号 w_slow 做梯度步。）
        self.w_pred = 0.0
        self.lr = lr  # 内模型学习率（论文：0.01，此处放大）

    def predict(self):
        return self.w_pred

    def gradient_step(self, w_slow_target):
        # 最小化 0.5 * (w_pred - w_slow)^2  ->  梯度 = (w_pred - w_slow)
        grad = self.w_pred - w_slow_target
        self.w_pred -= self.lr * grad


def low_pass(prev, new, eta=0.08, clip=10.0):
    """一阶低通 + 逐步饱和（论文式 2）。
    eta 故意取小，使监督信号干净但滞后 —— 这正是为何更快的学习预测器
    值得拥有。"""
    delta = np.clip(new - prev, -clip, clip)
    return prev + eta * delta


# ---------------------------------------------------------------------------
# 单个 IMPACT 控制环的模拟。
# ---------------------------------------------------------------------------
def simulate():
    # --- 惊讶门控阈值（论文：surprise >= 5 N；rate <= 0.2）---
    SURPRISE_ON = 5.0   # N : 高于此值模型才适应的预测误差
    SETTLE_RATE = 0.2   # N/步 : 仅当慢力旋量在稳定下来时才学习

    # 举升轴上的阻抗刚度（未补偿残余 = err = F/K）
    K = 400.0  # N/m  （对应论文的平移刚度）

    # --- 任务协议：随时间的负载质量 (kg) ---
    # 阶段 A+B：2.5 kg 持续 220 步，阶段 C：之后 5.0 kg。
    steps = 360
    mass = np.where(np.arange(steps) < 220, 2.5, 5.0)

    model = InternalModel(lr=0.30)
    w_slow = 0.0

    log = []
    for t in range(steps):
        # ---- 真实外部力旋量（负载重力）----
        w_ext = mass[t] * g

        # ---- (2.2 节) 虚拟力传感器：残差 ~= w_ext + 噪声 ----
        w_ext_hat = w_ext + np.random.normal(0, 1.2)  # 模型/传感器噪声

        # ---- (式 2) 慢/快分解：监督信号 ----
        prev_slow = w_slow
        w_slow = low_pass(w_slow, w_ext_hat)
        slow_rate = abs(w_slow - prev_slow)

        # ---- (2.3 节) 内模型预测 ----
        w_pred = model.predict()

        # ---- 惊讶门控：预测相对测得力有多错？----
        surprise = abs(w_pred - w_ext_hat)
        gate_open = (surprise >= SURPRISE_ON) or (slow_rate > SETTLE_RATE
                                                  and surprise > 1.0)
        if gate_open:
            # 把内模型朝（干净的）慢力旋量目标适应
            model.gradient_step(w_slow)

        # ---- 前馈补偿：w_ff = w_pred ----
        w_ff = model.predict()

        # ---- 阻抗控制器还需对抗的部分 ----
        residual_force = w_ext - w_ff          # N : 未补偿的负载
        pose_error = residual_force / K         # m : 稳态偏移

        log.append(dict(t=t, mass=mass[t], w_ext=w_ext, w_slow=w_slow,
                        w_pred=w_pred, surprise=surprise,
                        gate=int(gate_open), pose_err=pose_error))
    return log


# ---------------------------------------------------------------------------
# 报告：阶段摘要 + |位姿误差| 随时间的微型 ASCII 图。
# ---------------------------------------------------------------------------
def report(log):
    def phase(name, lo, hi):
        seg = [r for r in log if lo <= r["t"] < hi]
        gate_fires = sum(r["gate"] for r in seg)
        mean_err = np.mean([abs(r["pose_err"]) for r in seg]) * 1000  # mm
        end_err = abs(seg[-1]["pose_err"]) * 1000
        print(f"  {name:8s} 质量={seg[0]['mass']:.1f}kg  "
              f"门控触发={gate_fires:3d}  "
              f"平均|误差|={mean_err:6.2f}mm  末端|误差|={end_err:6.2f}mm")

    print("\nIMPACT 内模型 demo —— 三阶段协议（对照图 7）\n")
    print("阶段 A = 首次举 2.5kg（学习）| B = 重复 2.5kg（冻结）| "
          "C = 换成 5kg（重新适应）\n")
    phase("阶段 A", 0, 90)
    phase("阶段 B", 90, 220)
    phase("阶段 C", 220, 360)

    print("\n|位姿误差| 随时间（每行约 9 步；'#'=大，'.'=极小）：")
    bucket = 9
    rows = []
    for i in range(0, len(log), bucket):
        seg = log[i:i + bucket]
        e = np.mean([abs(r["pose_err"]) for r in seg]) * 1000  # mm
        gate = any(r["gate"] for r in seg)
        bar = "#" * min(int(e * 1.2), 40)
        marker = "  <-- 门控激活（学习中）" if gate else ""
        rows.append(f"  t={i:3d} |{bar:<40s}| {e:5.2f}mm{marker}")
    print("\n".join(rows))

    print("\n要点：误差仅在负载变化后骤升（A 和 C），惊讶门控在那里触发，")
    print("一旦内模型学到负载就以 ~零稳态误差前馈出去（阶段 B）。")
    print("纯阻抗控制器在每个阶段都会显示阶段 A 的误差水平 ——")
    print("那就是论文图 9 的基线。\n")


if __name__ == "__main__":
    np.random.seed(0)  # 确定性（此处用伪随机没问题，这是脚本）
    report(simulate())
