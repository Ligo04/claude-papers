#!/usr/bin/env python3
"""教学演示：耦合接触轨迹的伴随法，以及 affine 形状导数中易漏的项。

运行：python3 contact_trajectory_adjoint_demo.py --output demo_results.json
仅依赖 NumPy。数值场景由本演示构造，不是论文实验或 DiffGIPC 实现。
没有实现三维网格 CCD、摩擦、FEM 或 GPU 求解，不能用于评估论文性能。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def barrier(gap: float, activation: float) -> tuple[float, float, float]:
    """对正间隙使用 squared-distance IPC 型 barrier，并返回关于间隙的导数。"""
    if gap <= 0:
        raise ValueError("接触间隙必须为正")
    if gap >= activation:
        return 0.0, 0.0, 0.0
    distance, threshold = gap * gap, activation * activation
    delta = distance - threshold
    log_ratio = np.log(distance / threshold)
    value = -delta * delta * log_ratio
    first_d = -2 * delta * log_ratio - delta * delta / distance
    second_d = -2 * log_ratio - 4 * delta / distance + delta * delta / distance**2
    return (
        float(value),
        float(2 * gap * first_d),
        float(2 * first_d + 4 * gap * gap * second_d),
    )


MASS = np.diag([1.0, 1.3])
H = 0.05
STEPS = 40
CONTACT_WEIGHT = H * H * 1e5
SEPARATION = 0.3
ACTIVATION = 0.15
NORMAL = np.array([-1.0, 1.0])
INITIAL = np.array([-0.4, 0.4])


def step_terms(y: np.ndarray, predictor: np.ndarray):
    gap = float(NORMAL @ y - SEPARATION)
    energy, slope, curvature = barrier(gap, ACTIVATION)
    residual = y - predictor
    value = 0.5 * residual @ MASS @ residual + CONTACT_WEIGHT * energy
    gradient = MASS @ residual + CONTACT_WEIGHT * slope * NORMAL
    hessian = MASS + CONTACT_WEIGHT * curvature * np.outer(NORMAL, NORMAL)
    return float(value), gradient, hessian


def forward(velocity: float):
    """用收敛的隐式步方程定义轨迹；反向不展开 Newton 迭代。"""
    states = [INITIAL.copy()]
    hessians = []
    worst_residual = 0.0
    for t in range(1, STEPS + 1):
        predictor = (
            INITIAL + H * np.array([velocity, 0.0])
            if t == 1
            else 2 * states[-1] - states[-2]
        )
        y = states[-1].copy()
        for _ in range(150):
            value, gradient, hessian = step_terms(y, predictor)
            if np.linalg.norm(gradient, ord=np.inf) < 1e-12:
                break
            direction = np.linalg.solve(hessian, -gradient)
            gap = NORMAL @ y - SEPARATION
            change = NORMAL @ direction
            # 一维正间隙的可行步长可解析求出；这不是三维网格 CCD。
            alpha = min(1.0, 0.9 * gap / -change) if change < 0 else 1.0
            for _ in range(70):
                candidate = y + alpha * direction
                if NORMAL @ candidate > SEPARATION:
                    new_value = step_terms(candidate, predictor)[0]
                    if new_value <= value + 1e-4 * alpha * gradient @ direction + 1e-18:
                        break
                alpha *= 0.5
            else:
                raise RuntimeError("Newton 线搜索未找到可行下降步")
            y = candidate
        else:
            raise RuntimeError("Newton 未收敛")
        _, gradient, hessian = step_terms(y, predictor)
        worst_residual = max(worst_residual, float(np.max(np.abs(gradient))))
        states.append(y)
        hessians.append(hessian)
    states = np.array(states)
    loss = 0.5 * (states[-1, 1] - 0.9) ** 2
    return float(loss), states, hessians, worst_residual


def adjoint(velocity: float, *, keep_coupling=True, keep_history=True):
    loss, states, hessians, residual = forward(velocity)
    rhs = np.zeros((STEPS, 2))
    rhs[-1, 1] = states[-1, 1] - 0.9
    lambdas = np.zeros_like(rhs)
    for t in range(STEPS - 1, -1, -1):
        matrix = hessians[t]
        if not keep_coupling:
            matrix = np.diag(np.diag(matrix))
        lambdas[t] = np.linalg.solve(matrix.T, rhs[t])
        if keep_history:
            if t >= 1:
                rhs[t - 1] += 2 * MASS @ lambdas[t]
            if t >= 2:
                rhs[t - 2] -= MASS @ lambdas[t]
    # 第一帧 G_v = -h M e_1，因此 -G_v^T lambda_1 为下面这个正号。
    gradient = H * (MASS @ lambdas[0])[0]
    return loss, float(gradient), states, residual


def forward_sensitivity(velocity: float) -> float:
    """独立的正向切线递推，用来核对伴随法的符号和时间索引。"""
    _, states, hessians, _ = forward(velocity)
    sensitivities = [np.zeros(2)]
    for t, matrix in enumerate(hessians, start=1):
        rhs = (
            H * MASS @ np.array([1.0, 0.0])
            if t == 1
            else 2 * MASS @ sensitivities[-1] - MASS @ sensitivities[-2]
        )
        sensitivities.append(np.linalg.solve(matrix, rhs))
    return float((states[-1, 1] - 0.9) * sensitivities[-1][1])


def point_map(rest: np.ndarray) -> np.ndarray:
    jacobian = np.zeros((3, 12))
    jacobian[:, :3] = np.eye(3)
    for row in range(3):
        jacobian[row, 3 + 3 * row : 6 + 3 * row] = rest
    return jacobian


def map_change(rest_change: np.ndarray) -> np.ndarray:
    jacobian = point_map(rest_change)
    jacobian[:, :3] = 0
    return jacobian


def relative_error(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-15))


def shape_checks():
    """在固定 affine 状态下，核对 dJ、dM 和直接损失导数。"""
    transform = np.array([[1.0, 0.03, 0.0], [-0.02, 1.0, 0.01], [0.01, 0.02, 1.0]])
    q = np.r_[np.array([0.12, 0.18, 0.25]), transform.ravel()]
    rest = np.array([0.2, -0.1, 0.05])
    change = np.array([0.6, -0.2, 0.3])
    change /= np.linalg.norm(change)
    jacobian, dj = point_map(rest), map_change(change)
    normal = np.array([0.0, 0.0, 1.0])

    def force(r):
        j = point_map(r)
        gap = float(normal @ j @ q)
        _, slope, _ = barrier(gap, 0.5)
        return j.T @ (slope * normal)

    _, slope, curvature = barrier(float(normal @ jacobian @ q), 0.5)
    point_force = slope * normal
    point_hessian = curvature * np.outer(normal, normal)
    geometry_term = jacobian.T @ point_hessian @ (dj @ q)
    map_term = dj.T @ point_force
    exact_contact = geometry_term + map_term
    epsilon = 1e-6
    fd_contact = (force(rest + epsilon * change) - force(rest - epsilon * change)) / (2 * epsilon)

    # 固定的八个求积质量用于解释 M=sum(m J^T J)；未实现论文的表面体积积分。
    samples = np.array(list(itertools.product([-0.2, 0.2], repeat=3)))
    sample_changes = np.zeros_like(samples)
    sample_changes[0] = change

    def mass_matrix(points):
        return sum(point_map(r).T @ point_map(r) / len(points) for r in points)

    dm = sum(
        (map_change(dr).T @ point_map(r) + point_map(r).T @ map_change(dr)) / len(samples)
        for r, dr in zip(samples, sample_changes)
    )
    fd_mass = (mass_matrix(samples + epsilon * sample_changes) - mass_matrix(samples - epsilon * sample_changes)) / (2 * epsilon)
    target = np.array([0.1, 0.2, 0.4])
    direct_loss_derivative = float((jacobian @ q - target) @ (dj @ q))

    def surface_loss(r):
        offset = point_map(r) @ q - target
        return float(0.5 * offset @ offset)

    fd_direct = (surface_loss(rest + epsilon * change) - surface_loss(rest - epsilon * change)) / (2 * epsilon)
    return {
        "场景说明": "合成 affine 点与平面的接触，固定状态的形状方向导数",
        "完整接触导数相对误差": relative_error(exact_contact, fd_contact),
        "漏掉_dJ_项的相对误差": relative_error(geometry_term, fd_contact),
        "质量矩阵导数相对误差": relative_error(dm, fd_mass),
        "直接表面损失导数相对误差": relative_error(direct_loss_derivative, fd_direct),
        "质量模型限制": "求积点质量固定，只核对 J 的变化；不包含体积和密度变化",
    }


def run():
    velocity = 1.2
    loss, gradient, states, residual = adjoint(velocity)
    finite_differences = []
    for epsilon in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        difference = (forward(velocity + epsilon)[0] - forward(velocity - epsilon)[0]) / (2 * epsilon)
        finite_differences.append({"epsilon": epsilon, "中心差分": difference, "相对误差": relative_error(gradient, difference)})
    shape = shape_checks()
    tangent = forward_sensitivity(velocity)
    result = {
        "性质": "教学验证，非论文复现，所有场景参数均为合成数据",
        "轨迹": {
            "模拟步数": STEPS, "时间步长": H, "loss": loss,
            "伴随梯度": gradient, "正向切线梯度": tangent,
            "伴随与切线相对误差": relative_error(gradient, tangent),
            "删除接触交叉块的错误梯度": adjoint(velocity, keep_coupling=False)[1],
            "删除时间依赖的错误梯度": adjoint(velocity, keep_history=False)[1],
            "最小正间隙": float(np.min(states @ NORMAL - SEPARATION)),
            "最大前向残差": residual, "有限差分": finite_differences,
        },
        "形状": shape,
    }
    assert relative_error(gradient, tangent) < 1e-10
    assert min(row["相对误差"] for row in finite_differences) < 1e-5
    assert result["轨迹"]["最小正间隙"] > 0
    assert abs(gradient) > 1e-4
    assert shape["完整接触导数相对误差"] < 1e-7
    assert shape["质量矩阵导数相对误差"] < 1e-7
    assert shape["直接表面损失导数相对误差"] < 1e-7
    assert shape["漏掉_dJ_项的相对误差"] > 1e-3
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
