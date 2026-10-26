#!/usr/bin/env python3
"""TapeSim 的 bond 能量与前沿规则教学，依赖 NumPy。

这不是作者求解器。四面体边长 2 mm、参考高 0.2 mm 和材料行图是
教学设置，不是论文测得的 bond 几何。κ、断裂阈值及前沿门槛取自
论文 IV-B。省略薄壳、IPC/CCD、Newton-PCG、真实卷绕、切割与机器人。

运行后把独立的行为检查和力扫描写入本脚本所在目录。
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VirtualBond:
    """复用 T0、T1、T2、P 四个顶点，不增加质量或未知数。"""

    inv_reference: np.ndarray
    volume: float
    reference_gap: float
    stiffness_pa: float = 1e6
    break_force_n: float = 0.5

    @staticmethod
    def edge_matrix(vertices: np.ndarray) -> np.ndarray:
        return (vertices[1:] - vertices[0]).T

    @classmethod
    def create(cls, vertices: np.ndarray) -> VirtualBond:
        db = cls.edge_matrix(vertices)
        # 共面/病态参考形状无法定义可靠的逆矩阵，不能静默继续。
        if not np.all(np.isfinite(db)) or np.linalg.cond(db) > 1e12:
            raise ValueError("bond 参考形状奇异或病态")
        gap = cls.point_plane_gap(vertices)
        return cls(np.linalg.inv(db), abs(float(np.linalg.det(db))) / 6, gap)

    @staticmethod
    def point_plane_gap(vertices: np.ndarray) -> float:
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        length = float(np.linalg.norm(normal))
        if length == 0:
            raise ValueError("当前三角形退化")
        # 教学形状的 P 投影在三角形内部，因此可用点到平面的距离。
        return abs(float(np.dot(vertices[3] - vertices[0], normal))) / length

    def energy_gradient(self, vertices: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
        f = self.edge_matrix(vertices) @ self.inv_reference
        a = f @ f.T - np.eye(3)
        scale = dt * dt * self.volume * self.stiffness_pa
        energy = scale * float(np.sum(a * a))
        d_ds = 4 * scale * a @ f @ self.inv_reference.T
        gradient = np.empty((4, 3))
        gradient[1:] = d_ds.T
        gradient[0] = -gradient[1:].sum(axis=0)
        return energy, gradient

    def physical_force_on_point(self, vertices: np.ndarray, dt: float) -> np.ndarray:
        _, gradient = self.energy_gradient(vertices, dt)
        # 增量势能含 Δt²。去掉它后，力才可以和 N 单位的阈值比较。
        return -gradient[3] / (dt * dt)

    def should_release(self, vertices: np.ndarray, dt: float) -> bool:
        opening = self.point_plane_gap(vertices) > self.reference_gap
        force = float(np.linalg.norm(self.physical_force_on_point(vertices, dt)))
        return bool(opening and force > self.break_force_n)


def numerical_gradient(bond: VirtualBond, vertices: np.ndarray, dt: float) -> np.ndarray:
    """独立中心差分，不复用解析导数。"""
    result = np.zeros_like(vertices)
    h = 1e-8
    for index in np.ndindex(vertices.shape):
        plus, minus = vertices.copy(), vertices.copy()
        plus[index] += h
        minus[index] -= h
        result[index] = (bond.energy_gradient(plus, dt)[0] - bond.energy_gradient(minus, dt)[0]) / (2 * h)
    return result


def project_rigid_motion(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray):
    """质量加权最小二乘刚体速度投影，验证形成 cluster 时的运动学。"""
    center = np.average(positions, axis=0, weights=masses)
    linear = np.average(velocities, axis=0, weights=masses)
    r = positions - center
    inertia = sum(m * (np.dot(x, x) * np.eye(3) - np.outer(x, x)) for x, m in zip(r, masses))
    angular_momentum = np.sum(masses[:, None] * np.cross(r, velocities - linear), axis=0)
    angular = np.linalg.solve(inertia, angular_momentum)
    return center, linear, angular


class MaterialFront:
    """材料行图上的前沿状态机，不求解胶带动力学。

    行图中的 collar 用顶点到自由区距离 ≤N 定义；真实论文按三角形
    任一角点到自由区距离 <N 恢复元素，不能把这个类作为网格实现。
    """

    def __init__(self, count: int = 40, collar_edges: int = 10):
        self.count = count
        self.collar_edges = collar_edges
        self.adjacency = {i: {j for j in (i - 1, i + 1) if 0 <= j < count} for i in range(count)}
        self.free = {0}
        self.cluster = set(range(collar_edges + 1, count))
        self.collar = set(range(1, collar_edges + 1))
        # 每行的 rest 长度 1 mm 是教学离散；材料密度/宽/厚来自表 I。
        self.masses = np.full(count, 1300 * 0.019 * 0.00018 * 0.001)
        self.rest_positions = np.column_stack((np.arange(count) * 0.001, np.zeros(count), np.zeros(count)))
        self.winding_beta = np.linspace(0.25, 0.95, count)
        self.restored_beta: dict[int, float] = {}

    def graph_distance(self, sources: set[int]) -> dict[int, int]:
        distance = {i: 0 for i in sources}
        queue = deque(sources)
        while queue:
            i = queue.popleft()
            for j in self.adjacency[i]:
                if j not in distance:
                    distance[j] = distance[i] + 1
                    queue.append(j)
        return distance

    def advance(self, displacement_m: dict[int, float], release_events: set[int], live_support: set[int],
                family: str = "Bond", delta_m: float = 0.0009):
        old_free = set(self.free)
        old_cluster = set(self.cluster)
        # 同一步的所有候选相对同一份旧状态判断，避免原地 BFS 递归释放。
        candidates = set().union(*(self.adjacency[i] for i in old_free)) & self.collar
        accepted = {
            i for i in candidates
            if displacement_m.get(i, 0) > delta_m
            and (family == "Coh" or (i in release_events and i not in live_support))
        }
        self.free = old_free | accepted
        distance = self.graph_distance(self.free)
        self.cluster = {i for i in old_cluster if distance[i] > self.collar_edges}
        self.collar = set(range(self.count)) - self.free - self.cluster
        thawed = old_cluster - self.cluster
        for i in thawed:
            self.restored_beta[i] = float(self.winding_beta[i])
        if not old_free <= self.free or not self.cluster <= old_cluster:
            raise AssertionError("材料分类单调性被破坏")
        if (self.free & self.cluster) or (self.free & self.collar) or (self.cluster & self.collar):
            raise AssertionError("材料重复归属")
        if self.free | self.collar | self.cluster != set(range(self.count)):
            raise AssertionError("材料丢失")
        return accepted, thawed


def run_demo() -> dict:
    reference = np.array([[0., 0., 0.], [0.002, 0., 0.], [0., 0.002, 0.], [0.0006, 0.0006, 0.0002]])
    bond = VirtualBond.create(reference)
    dt = 0.01
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    checks: list[dict] = []

    def check(name: str, condition, **details):
        if not bool(condition):
            raise AssertionError(f"检查失败：{name}；{details}")
        checks.append({"name": name, "passed": True, **details})

    energy_rigid, _ = bond.energy_gradient(reference @ rotation.T + np.array([0.01, -0.02, 0.03]), dt)
    check("共同刚体变换不产生 bond 应变", energy_rigid < 1e-25, energy=energy_rigid)

    deformed = reference.copy()
    deformed[3] += [0.00007, -0.00003, 0.00009]
    deformed[1, 1] += 0.00002
    _, analytic = bond.energy_gradient(deformed, dt)
    numerical = numerical_gradient(bond, deformed, dt)
    relative = float(np.max(np.abs(analytic - numerical)) / max(np.max(np.abs(analytic)), 1e-20))
    check("解析梯度与独立中心差分一致", relative < 1e-7, maximum_relative_error=relative)
    check("四顶点合力为零", np.linalg.norm(analytic.sum(axis=0)) < 1e-18)

    rotated_force = bond.physical_force_on_point(deformed @ rotation.T + [0.01, 0.02, 0.03], dt)
    force = bond.physical_force_on_point(deformed, dt)
    check("恢复力随整体旋转协变", np.allclose(rotated_force, rotation @ force, atol=1e-10))
    check("物理力不含额外时间步缩放", np.allclose(force, bond.physical_force_on_point(deformed, dt / 2)))

    compressed = reference.copy()
    compressed[3, 2] *= 0.5
    compression_force = float(np.linalg.norm(bond.physical_force_on_point(compressed, dt)))
    check("超过力阈值的压缩仍不触发拉伸断裂", compression_force > bond.break_force_n and not bond.should_release(compressed, dt), force_n=compression_force)
    opened = reference.copy()
    opened[3, 2] *= 1.5
    check("张开且超过阈值才断裂", bond.should_release(opened, dt), force_n=float(np.linalg.norm(bond.physical_force_on_point(opened, dt))))
    check("参考态不释放", not bond.should_release(reference, dt))

    reflected = reference @ np.diag([-1., 1., 1.])
    check("单独的正交势能也允许反射", bond.energy_gradient(reflected, dt)[0] < 1e-25,
          note="数学边界，不表示作者系统实际发生翻转")
    degenerate = reference.copy()
    degenerate[3, 2] = 0
    try:
        VirtualBond.create(degenerate)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("拒绝共面参考四面体", rejected)

    front = MaterialFront()
    displacement = {i: 0.0018 for i in range(front.count)}
    total_mass = float(front.masses.sum())
    old_cluster = set(front.cluster)
    accepted, thawed = front.advance(displacement, set(range(front.count)), set())
    check("一次更新只接受旧自由区的邻居", accepted == {1}, released=sorted(accepted))
    check("collar 推进会恢复下一批材料", thawed == {11} and front.cluster < old_cluster)
    check("恢复原黏附历史而非重置为一", front.restored_beta[11] == front.winding_beta[11] and front.restored_beta[11] != 1)

    rows = []
    for step in range(8):
        old_free = set(front.free)
        old_cluster = set(front.cluster)
        accepted, thawed = front.advance(displacement, set(range(front.count)), set())
        masses = [float(front.masses[list(s)].sum()) for s in (front.free, front.collar, front.cluster)]
        check(f"第 {step + 2} 次更新保持材料与质量", old_free <= front.free and front.cluster <= old_cluster and np.isclose(sum(masses), total_mass, rtol=1e-14))
        rows.append({"step": step + 2, "free": len(front.free), "collar": len(front.collar), "cluster": len(front.cluster), "released": sorted(accepted), "thawed": sorted(thawed)})
    frozen = set(front.free)
    front.advance({}, set(), set())
    check("没有新门槛事件时不回卷或继续前进", front.free == frozen)

    supported = MaterialFront()
    a, _ = supported.advance(displacement, {1}, {1})
    check("live support 阻止 Bond 候选释放", not a)
    no_event = MaterialFront()
    a, _ = no_event.advance(displacement, set(), set())
    check("Bond 需要明确 release event", not a)
    cohesive = MaterialFront()
    a, _ = cohesive.advance(displacement, set(), {1}, family="Coh")
    check("Coh 几何前沿不要求 beta 已为零", a == {1})

    positions = reference @ rotation.T + [0.01, -0.02, 0.03]
    masses = np.array([1., 2., 3., 4.]) * 1e-5
    center = np.average(positions, axis=0, weights=masses)
    linear = np.array([0.1, -0.03, 0.02])
    angular = np.array([0.4, -0.5, 2.])
    velocities = linear + np.cross(angular, positions - center)
    recovered_center, recovered_linear, recovered_angular = project_rigid_motion(positions, velocities, masses)
    inherited = recovered_linear + np.cross(recovered_angular, positions - recovered_center)
    velocity_error = float(np.max(np.abs(velocities - inherited)))
    check("刚体速度投影和恢复保持运动", np.allclose(recovered_angular, angular) and velocity_error < 1e-12, maximum_error_m_s=velocity_error)
    mass_g_m = 1300 * 0.019 * 0.00018 * 1000
    check("表 I 的单位长度质量可由独立单位计算恢复", abs(mass_g_m - 4.446) < 1e-12, grams_per_m=mass_g_m)

    out = Path(__file__).resolve().parent
    with (out / "teaching-bond-force.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["教学高度相对参考比", "物理恢复力模_N", "张开", "释放"])
        for ratio in np.linspace(0.3, 1.8, 31):
            current = reference.copy()
            current[3, 2] *= ratio
            writer.writerow([float(ratio), float(np.linalg.norm(bond.physical_force_on_point(current, dt))), bool(ratio > 1), bond.should_release(current, dt)])
    result = {
        "scope": "CPU 教学验证；不是 TapeSim 物理或性能复现",
        "paper_parameters": {"stiffness_pa": 1e6, "break_force_n": 0.5, "dt_s": dt, "front_delta_m": 0.0009, "collar_edges": 10},
        "synthetic_settings": {"triangle_edge_m": 0.002, "reference_height_m": 0.0002, "material_rows": 40, "row_length_m": 0.001},
        "checks": checks,
        "front_trace": rows,
        "limitations": ["没有薄壳、IPC/CCD 或隐式运动求解", "行图并非三角网格前沿实现", "没有执行真实放带、切割、GPU 计时或机器人策略"],
    }
    (out / "demo-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    report = run_demo()
    print(f"通过 {len(report['checks'])} 项教学行为检查。")
    print("验证了 virtual bond 梯度、释放条件、前沿快照、质量分类与刚体速度转移。")
    print("本结果不代表 TapeSim 仿真、性能或真机复现。")
