#!/usr/bin/env python3
"""教学演示：多模态轨迹的均值、rectified flow 采样和 DCC 的适用范围。

只依赖 NumPy。所有场景与数值为合成示例，未训练或运行 PhysLDM。
混合高斯的最优 flow 速度可解析计算，用它说明采样机制。
旋转解码器人为保证体积不变，这项保证不属于论文的神经解码器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rotation_z(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle)
    result = np.zeros(angle.shape + (3, 3))
    result[..., 0, 0] = result[..., 1, 1] = np.cos(angle)
    result[..., 0, 1] = -np.sin(angle)
    result[..., 1, 0] = np.sin(angle)
    result[..., 2, 2] = 1
    return result


def tetra_volume(vertices: np.ndarray) -> np.ndarray:
    edges = vertices[..., 1:, :] - vertices[..., :1, :]
    return np.abs(np.linalg.det(edges)) / 6


def analytic_velocity(x, tau, mode=0.9, sigma=0.07):
    """论文采用 x_tau=(1-tau)z+tau*epsilon，目标速度为 epsilon-z。

    对 z ~ 0.5*N(-mode,sigma²)+0.5*N(mode,sigma²)，求条件期望速度。
    在给定 x_tau 时，先计算两个分量的后验概率，再加权各自速度。
    """
    means = np.array([-mode, mode])
    a = 1 - tau
    variance = a * a * sigma * sigma + tau * tau
    residual = x[..., None] - a * means
    logits = -0.5 * residual * residual / variance
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    covariance = tau - a * sigma * sigma
    component_velocity = -means + covariance / variance * residual
    return np.sum(weights * component_velocity, axis=-1)


def sample_flow(initial_noise, steps, mode=0.9, sigma=0.07):
    x = np.array(initial_noise, dtype=float, copy=True)
    # 从噪声 tau=1 积分到数据 tau=0，因此步长为负。
    for step in range(steps):
        tau = 1 - step / steps
        x -= analytic_velocity(x, tau, mode, sigma) / steps
    return x


def normal_cdf(x):
    from math import erf

    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def mixture_cdf(x, mode=0.9, sigma=0.07):
    return 0.5 * (normal_cdf((x + mode) / sigma) + normal_cdf((x - mode) / sigma))


def ks_distance(samples):
    samples = np.sort(samples)
    n = len(samples)
    cdf = mixture_cdf(samples)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))


def kabsch(rest, observed):
    """返回列向量约定 x=Rr+c 的最优刚体拟合；允许观测包含形变。"""
    rest_mean, observed_mean = rest.mean(axis=0), observed.mean(axis=0)
    u, _, vt = np.linalg.svd((rest - rest_mean).T @ (observed - observed_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = observed_mean - rotation @ rest_mean
    residual = (observed - translation) @ rotation - rest
    return rotation, translation, residual


def deformation_change(field):
    return np.linalg.norm(np.diff(field, axis=0), axis=-1).mean(axis=-1)


def dcc(reference, predicted):
    a, b = deformation_change(reference), deformation_change(predicted)
    if min(a.std(), b.std()) < 1e-14:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run(seed=20261020):
    rng = np.random.default_rng(seed)
    rest = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / 4
    times = np.linspace(0, 1, 64)
    profile = 3 * times**2 - 2 * times**3
    base_volume = float(tetra_volume(rest))
    mode = 0.9
    sigma = 0.07

    # 两条合法的旋转轨迹，逐顶点平均会引入不属于任一分支的收缩。
    plus = np.einsum("tij,nj->tni", rotation_z(mode * profile), rest)
    minus = np.einsum("tij,nj->tni", rotation_z(-mode * profile), rest)
    mean = 0.5 * (plus + minus)
    mean_ratio = tetra_volume(mean) / base_volume
    expected_mse_mean = 0.5 * (np.mean((mean - plus)**2) + np.mean((mean - minus)**2))
    expected_mse_branch = 0.5 * np.mean((plus - minus)**2)
    assert expected_mse_mean < expected_mse_branch
    assert np.allclose(mean_ratio, np.cos(mode * profile)**2, atol=1e-12)
    assert np.allclose(tetra_volume(plus) / base_volume, 1)

    noise = rng.normal(size=8192)
    samples = sample_flow(noise, 50, mode, sigma)
    refined = sample_flow(noise, 200, mode, sigma)
    flow_stats = []
    for steps, values in [(50, samples), (200, refined)]:
        flow_stats.append({
            "Euler 步数": steps,
            "样本均值": float(values.mean()),
            "样本方差": float(values.var()),
            "目标方差": mode * mode + sigma * sigma,
            "正分支比例": float((values > 0).mean()),
            "到目标混合分布的 KS 距离": ks_distance(values),
        })
    # 检查采样分布接近目标。这里未借助真实轨迹挑选最好样本。
    assert abs((samples > 0).mean() - 0.5) < 0.03
    assert ks_distance(samples) < 0.05
    assert ks_distance(refined) < 0.04
    decoded = np.einsum("btij,nj->btni", rotation_z(samples[:32, None] * profile), rest)
    assert np.allclose(tetra_volume(decoded) / base_volume, 1, atol=1e-12)

    # 独立验证：纯刚体运动移除后，局部形变应接近零。
    rotation, translation, residual = kabsch(rest, plus[40] + np.array([0.4, 0.7, -0.2]))
    kabsch_residual = float(np.max(np.abs(residual)))
    assert kabsch_residual < 1e-12
    assert np.allclose(np.linalg.det(rotation), 1)

    # DCC 只看时间变化曲线的相关性，对统一幅度缩放不敏感。
    pulse = np.exp(-((times - 0.4) / 0.08)**2) + 0.25 * np.sin(7 * times)**2
    field = pulse[:, None, None] * rest[None, :, :] * 0.12
    damped = 0.2 * field
    held = field[(np.arange(64) // 8) * 8]
    assert abs(dcc(field, damped) - 1) < 1e-12

    return {
        "性质": "合成教学验证；不是 PhysLDM 复现，不是论文 benchmark",
        "随机种子": seed,
        "均值与合法分支": {
            "末帧均值形状体积比例": float(mean_ratio[-1]),
            "单个旋转分支体积比例": 1.0,
            "均值的期望坐标 MSE": float(expected_mse_mean),
            "固定分支的期望坐标 MSE": float(expected_mse_branch),
            "结论": "平方误差更低的平均几何仍可能违反形状约束",
        },
        "解析 rectified flow": flow_stats,
        "刚体分解": {"Kabsch 最大残差": kabsch_residual},
        "DCC": {
            "形变幅度压到20%的相关性": dcc(field, damped),
            "每8帧保持一次的相关性": dcc(field, held),
            "限制": "块保持只是 staircase 的示意，不是论文 standard VAE 的实现",
        },
        "压缩口径": {
            "N": 5000, "T": 64, "K": 512, "M": 8, "D": 48,
            "原始顶点帧token数": 64 * 5000,
            "潜空间token数": 8 * 513,
            "token压缩比": 64 * 5000 / (8 * 513),
            "按xyz与48通道计的标量数比例": (64 * 5000 * 3) / (8 * 513 * 48),
            "限制": "标量数比例是教学推导，不是测得的内存或文件压缩比",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
