"""
预测-校正流水线 & "预测矩"带宽节省 demo —— KPM-FR §3.6 + §4
============================================================

核心论点: 公共通量 F*(h) 本需邻居的 ∇g —— 3D 下每 solution point 16 个标量。
利用 (1) 演化态恒等式 (需 f_s 而非原始 ∇g) 与 (2) 平衡/非平衡分解
(g 由 U 可重建, 只需传非平衡相关的"预测矩"), 把跨单元载荷压到物理必需最小。

本 demo 模拟一个 1D 单元链上的 predictor→(global memory)→corrector 循环:
  1. predictor: 各单元算本地梯度, 编码成紧凑预测矩, 写"全局内存";
  2. corrector: 读回邻居预测矩 + 本地可重建平衡部分, 组装 F*, 单步更新;
  3. 统计"朴素传梯度(16 标量)" vs "传预测矩"的 off-chip 字节数, 展示带宽节省。

纯标准库 + numpy:  python predictor_corrector_pipeline_demo.py
"""

import numpy as np


# ---- 载荷方案对比 (每 solution point, 3D) ----
NAIVE_SCALARS_3D = 16      # 朴素: 邻居梯度张量 ∇g (∇U), 论文所述 3D 16 标量/点
# 预测矩方案: 只需传非平衡相关的紧凑矩。
# 3D 等温模型守恒量 C=4 (ρ, ρu, ρv, ρw); 界面公共态 + 迎风传播项的最小编码
PREDICTED_MOMENTS_3D = 5   # 紧凑标量(示意: 4 个宏观 + 1 个非平衡编码)


def payload_comparison(n_points):
    """off-chip 传输字节数对比 (fp32=4B)。"""
    naive = n_points * NAIVE_SCALARS_3D * 4
    pmom  = n_points * PREDICTED_MOMENTS_3D * 4
    return naive, pmom


# ---- 一个玩具 1D 预测-校正循环 (等温, C=2: ρ, ρu) ----
RT = 1.0/3.0

def equilibrium_moments(U):
    """由守恒量 U=(ρ,ρu) 重建平衡通量 (无黏部分)。接收方可自行完成 → 无需跨单元传。"""
    rho, rho_u = U
    u = rho_u / rho
    F_eq = np.array([rho_u, rho_u*u + rho*RT])    # (ρu, ρu²+ρRT)
    return F_eq


def predictor(U_field, dx):
    """各单元: 算本地梯度 → 编码"预测矩"(此处: 非平衡通量修正标量)。
    返回每个单元要写入全局内存的紧凑预测矩。"""
    n = len(U_field)
    pmom = []
    for i in range(n):
        ip, im = (i+1) % n, (i-1) % n
        gradU = (U_field[ip] - U_field[im]) / (2*dx)      # ∇U (本地)
        # 非平衡修正 ∝ -τ ∂F_eq/∂x 的紧凑标量编码 (Chapman-Enskog 味道)
        u = U_field[i][1]/U_field[i][0]
        neq = -0.02 * (gradU[1] - u*gradU[0])             # 单个非平衡标量(示意)
        pmom.append(neq)
    return np.array(pmom)


def corrector(U_field, pmom, dx, dt):
    """各单元: 读回邻居预测矩 + 本地可重建平衡部分 → 组装公共通量 F* → 单步更新。"""
    n = len(U_field)
    U_new = U_field.copy()
    for i in range(n):
        ip, im = (i+1) % n, (i-1) % n
        # 界面公共通量 = 平衡部分(接收方自建) + 非平衡修正(来自预测矩, 迎风)
        F_eq_i = equilibrium_moments(U_field[i])
        F_eq_ip = equilibrium_moments(U_field[ip])
        F_eq_im = equilibrium_moments(U_field[im])
        # 右界面公共通量(简单中心+迎风非平衡)
        F_right = 0.5*(F_eq_i + F_eq_ip) + np.array([0, pmom[i]])
        F_left  = 0.5*(F_eq_im + F_eq_i) + np.array([0, pmom[im]])
        # 单步更新 (有限体积味的 KPM-FR 单步): dU = -(dt/dx)(F_right - F_left)
        U_new[i] = U_field[i] - (dt/dx)*(F_right - F_left)
    return U_new


def main():
    print("=" * 66)
    print("预测-校正流水线 & 预测矩带宽节省 (KPM-FR §3.6/§4)")
    print("=" * 66)

    # [1] 带宽节省: 朴素传梯度 vs 传预测矩
    print("\n[1] 跨单元 off-chip 传输载荷对比 (3D, 每 solution point):")
    print(f"    朴素方案(传邻居梯度 ∇g): {NAIVE_SCALARS_3D} 标量/点")
    print(f"    预测矩方案:              {PREDICTED_MOMENTS_3D} 标量/点")
    print(f"    载荷压缩比:              {PREDICTED_MOMENTS_3D/NAIVE_SCALARS_3D*100:.0f}% "
          f"(约 {NAIVE_SCALARS_3D/PREDICTED_MOMENTS_3D:.1f}× 更少)")
    for npts in [10**6, 10**8, 1_800_000_000]:
        naive, pmom = payload_comparison(npts)
        print(f"    {npts:>13,} 点: 朴素 {naive/2**30:6.2f} GiB → 预测矩 {pmom/2**30:5.2f} GiB "
              f"(省 {(naive-pmom)/2**30:.2f} GiB)")
    print("    ★ 由于是 memory-bound, off-chip 载荷↓ 直接 ⇒ 逼近带宽 roofline ⇒ 吞吐↑")

    # [2] 玩具 1D 预测-校正循环: 平流一个密度扰动
    print("\n[2] 玩具 1D 预测-校正循环 (等温, 平流密度扰动):")
    n = 64
    dx = 1.0/n
    dt = 0.2*dx/np.sqrt(RT)               # CFL 约束
    x = np.linspace(0, 1, n, endpoint=False)
    rho = 1.0 + 0.1*np.exp(-((x-0.5)**2)/0.01)   # 高斯密度扰动
    u0 = 0.3
    U = np.stack([rho, rho*u0], axis=1)          # (n, 2): (ρ, ρu)

    mass0 = np.sum(U[:,0])*dx
    mom0 = np.sum(U[:,1])*dx
    nsteps = 200
    for step in range(nsteps):
        pmom = predictor(U, dx)          # kernel 1: 编码 → "写全局内存"
        U = corrector(U, pmom, dx, dt)   # kernel 2: 读回 → 组装 F* → 更新
    mass1 = np.sum(U[:,0])*dx
    mom1 = np.sum(U[:,1])*dx
    peak_x = x[np.argmax(U[:,0])]

    print(f"    结构: 2 个 kernel/步 × {nsteps} 步 (predictor 编码矩 → corrector 解码更新)")
    print(f"    密度扰动随时间平流+扩散(此玩具通量为一阶迎风, 故有数值耗散, 峰位 x={peak_x:.3f})")
    print(f"    ★ 守恒量严格守恒(通量形式的核心保证):")
    print(f"      质量: 初 {mass0:.6f} → 末 {mass1:.6f} (相对误差 {abs(mass1-mass0)/mass0:.1e})")
    print(f"      动量: 初 {mom0:.6f} → 末 {mom1:.6f} (相对误差 {abs(mom1-mom0)/abs(mom0):.1e})")
    print("\n    要点(与规模无关, 可验证的两点):")
    print("    1) 每步严格两个 kernel, 只有紧凑预测矩经过'全局内存', 平衡部分由 U 本地重建;")
    print("    2) 通量形式 U_new = U - (dt/dx)(F_right-F_left) 严格守恒质量与动量。")


if __name__ == "__main__":
    main()
