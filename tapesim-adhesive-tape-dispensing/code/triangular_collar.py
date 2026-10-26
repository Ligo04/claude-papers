#!/usr/bin/env python3
"""三角网格上的 collar 状态更新教学，依赖 NumPy。

演示元素级 <N 规则、旧自由区快照、元素质量归属、刚体速度继承和历史
恢复。输入位置由教学程序指定；没有薄壳/接触运动求解，也不是作者实现。
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class InterfaceHistory:
    beta: float
    reference_edges: np.ndarray


class TriangularCollar:
    def __init__(self, collar_edges=10, rows=31, across=3):
        if not 2 <= collar_edges < rows - 2 or across < 2:
            raise ValueError("教学带要求 2≤N<rows−2，宽度方向至少两个顶点")
        self.n = collar_edges
        self.rows = rows
        self.across = across
        s, w = np.meshgrid(np.arange(rows) * 0.002, np.linspace(-0.0095, 0.0095, across), indexing="ij")
        self.rest = np.column_stack((s.ravel(), w.ravel(), np.zeros(s.size)))
        # 初始携带形状与平坦 rest shape 分开保存，不能把两者混成一份。
        radius = 0.0211
        angle = s.ravel() / radius
        self.local = np.column_stack((radius * np.cos(angle), w.ravel(), radius * np.sin(angle)))
        self.rest.setflags(write=False)
        self.local.setflags(write=False)
        triangles = []
        for row in range(rows - 1):
            for col in range(across - 1):
                a = row * across + col
                b, c, d = a + 1, a + across, a + across + 1
                triangles.extend([(a, c, b), (b, c, d)])
        self.triangles = np.asarray(triangles)
        self.graph = {i: set() for i in range(s.size)}
        for triangle in self.triangles:
            for i in triangle:
                self.graph[int(i)].update(int(j) for j in triangle if i != j)
        e1 = self.rest[self.triangles[:, 1]] - self.rest[self.triangles[:, 0]]
        e2 = self.rest[self.triangles[:, 2]] - self.rest[self.triangles[:, 0]]
        area = np.linalg.norm(np.cross(e1, e2), axis=1) / 2
        self.element_mass = 1300 * 0.00018 * area
        self.free = set(range(across))
        distance = self.distances(self.free)
        self.cluster_elements = {e for e, t in enumerate(self.triangles) if min(distance[int(i)] for i in t) >= self.n}
        self.rotation = np.array([[np.cos(0.3), -np.sin(0.3), 0.], [np.sin(0.3), np.cos(0.3), 0.], [0., 0., 1.]])
        self.translation = np.array([0.01, -0.02, 0.03])
        self.positions = self.carried_positions()
        cluster_weights, _ = self.mass_partition()
        self.center = np.average(self.positions, axis=0, weights=cluster_weights)
        self.linear = np.array([0.05, -0.02, 0.03])
        self.angular = np.array([0.4, -0.5, 1.2])
        self.velocities = self.linear + np.cross(self.angular, self.positions - self.center)
        # 每元素关联一个合成接口记录，仅为验证恢复语义，不代表真实 pair 数。
        self.archive = {e: InterfaceHistory(float(0.2 + 0.6 * e / len(triangles)), np.diag([0.001, 0.001, 0.0002 + e * 1e-8])) for e in range(len(triangles))}
        for value in self.archive.values():
            value.reference_edges.setflags(write=False)
        self.active_history = {e: self.copy_history(e) for e in range(len(triangles)) if e not in self.cluster_elements}

    def copy_history(self, e):
        value = self.archive[e]
        return InterfaceHistory(value.beta, value.reference_edges.copy())

    def carried_positions(self):
        return self.local @ self.rotation.T + self.translation

    def rigid_vertices(self):
        return set(int(i) for e in self.cluster_elements for i in self.triangles[e])

    def independent_vertices(self):
        return set(range(len(self.rest))) - self.rigid_vertices()

    def distances(self, sources):
        distance = {i: 0 for i in sources}
        queue = deque(sources)
        while queue:
            i = queue.popleft()
            for j in self.graph[i]:
                if j not in distance:
                    distance[j] = distance[i] + 1
                    queue.append(j)
        return distance

    def mass_partition(self):
        cluster = np.zeros(len(self.rest))
        shell = np.zeros(len(self.rest))
        # 每个元素的 m/3 仅进入一个归属，边界顶点可同时收到两类贡献。
        for e, triangle in enumerate(self.triangles):
            target = cluster if e in self.cluster_elements else shell
            target[triangle] += self.element_mass[e] / 3
        return cluster, shell

    def totals(self):
        cluster_weights, shell_weights = self.mass_partition()
        weights = cluster_weights + shell_weights
        momentum = (weights[:, None] * self.velocities).sum(axis=0)
        angular = (weights[:, None] * np.cross(self.positions, self.velocities)).sum(axis=0)
        kinetic = float(0.5 * np.sum(weights[:, None] * self.velocities**2))
        return float(weights.sum()), momentum, angular, kinetic

    def prescribe_collar_displacement(self, displacement=0.0018):
        vertices = sorted(self.independent_vertices() - self.free)
        self.positions[vertices] = self.carried_positions()[vertices] + [0., 0., displacement]

    def advance(self, released_vertices, live_support, family="Bond", delta=0.0009):
        old_free = set(self.free)
        old_elements = set(self.cluster_elements)
        old_rigid = self.rigid_vertices()
        neighbors = set().union(*(self.graph[i] for i in old_free))
        candidates = (neighbors - old_free) & self.independent_vertices()
        carried = self.carried_positions()
        accepted = {i for i in candidates if np.linalg.norm(self.positions[i] - carried[i]) > delta
                    and (family == "Coh" or (i in released_vertices and i not in live_support))}
        self.free = old_free | accepted
        distance = self.distances(self.free)
        thawed = {e for e in old_elements if min(distance[int(i)] for i in self.triangles[e]) < self.n}
        self.cluster_elements = old_elements - thawed
        new_independent = old_rigid - self.rigid_vertices()
        for i in new_independent:
            self.positions[i] = carried[i]
            self.velocities[i] = self.linear + np.cross(self.angular, self.positions[i] - self.center)
        for e in thawed:
            self.active_history[e] = self.copy_history(e)
        # cluster 质量中心变化时，线速度必须随参考点变换，保持同一速度场。
        weights, _ = self.mass_partition()
        if weights.sum() > 0:
            new_center = np.average(self.positions, axis=0, weights=weights)
            self.linear = self.linear + np.cross(self.angular, new_center - self.center)
            self.center = new_center
        if not old_free <= self.free or not self.cluster_elements <= old_elements:
            raise AssertionError("材料分类不单调")
        return dict(candidates=sorted(candidates), accepted=sorted(accepted), thawed=sorted(thawed),
                    new_independent=sorted(new_independent))

    def snapshot(self, step, update=None):
        cluster_weights, shell_weights = self.mass_partition()
        rigid = self.rigid_vertices()
        return dict(step=step, n=self.n, free=sorted(self.free), rigid=sorted(rigid),
                    collar=sorted(self.independent_vertices() - self.free),
                    cluster_elements=sorted(self.cluster_elements),
                    cluster_mass_g=float(cluster_weights.sum() * 1000),
                    shell_mass_g=float(shell_weights.sum() * 1000),
                    independent_dof=3 * len(self.independent_vertices()) + (6 if self.cluster_elements else 0),
                    update=update)


def validate_mesh():
    checks = []

    def check(name, condition, **data):
        if not bool(condition):
            raise AssertionError(f"{name}: {data}")
        checks.append(dict(name=name, passed=True, **data))

    datasets = {}
    for n in (3, 10):
        mesh = TriangularCollar(n)
        total_mass = float(mesh.element_mass.sum())
        rest_before, local_before = mesh.rest.tobytes(), mesh.local.tobytes()
        archive_before = {e: (h.beta, h.reference_edges.tobytes()) for e, h in mesh.archive.items()}
        distances = mesh.distances(mesh.free)
        exact_boundary = {e for e, t in enumerate(mesh.triangles) if min(distances[int(i)] for i in t) == n}
        check(f"N={n}：距离恰为 N 的元素保持 cluster", bool(exact_boundary) and exact_boundary <= mesh.cluster_elements)
        c, s = mesh.mass_partition()
        boundary = mesh.rigid_vertices() & set(np.flatnonzero(s))
        check(f"N={n}：边界顶点确实收到两类元素质量贡献", bool(boundary) and all(c[i] > 0 and s[i] > 0 for i in boundary))
        check(f"N={n}：平坦 rest 与携带形状独立", np.all(mesh.rest[:, 2] == 0) and np.any(np.abs(mesh.local[:, 2]) > 0.001))
        states = [mesh.snapshot(0)]
        for step in range(1, 7):
            mesh.prescribe_collar_displacement()
            before = mesh.totals()
            old_free, old_elements = set(mesh.free), set(mesh.cluster_elements)
            old_positions = mesh.positions.copy()
            old_velocities = mesh.velocities.copy()
            update = mesh.advance(set(range(len(mesh.rest))), set())
            after = mesh.totals()
            errors = dict(mass_kg=abs(before[0] - after[0]), linear_momentum=float(np.max(np.abs(before[1] - after[1]))),
                          angular_momentum=float(np.max(np.abs(before[2] - after[2]))), kinetic_energy=abs(before[3] - after[3]))
            expected_row = set(range(step * mesh.across, (step + 1) * mesh.across))
            check(f"N={n}/步{step}：一轮只推进旧前沿的一层", set(update["accepted"]) == expected_row and old_free <= mesh.free and mesh.cluster_elements <= old_elements)
            check(f"N={n}/步{step}：完整元素行恢复为 shell", len(update["thawed"]) == 2 * (mesh.across - 1) and len(update["new_independent"]) == mesh.across)
            check(f"N={n}/步{step}：重分类保持位置和速度", np.allclose(mesh.positions, old_positions, atol=1e-14, rtol=0) and np.allclose(mesh.velocities, old_velocities, atol=1e-14, rtol=0))
            check(f"N={n}/步{step}：质量、动量、动能在重分类瞬间一致", abs(after[0] - total_mass) < 1e-15 and max(errors.values()) < 1e-14, errors=errors)
            check(f"N={n}/步{step}：恢复原始界面参考", all(mesh.active_history[e].beta == mesh.archive[e].beta and np.array_equal(mesh.active_history[e].reference_edges, mesh.archive[e].reference_edges) for e in update["thawed"]))
            states.append(mesh.snapshot(step, update))
        check(f"N={n}：rest 与初始携带坐标始终未改写", mesh.rest.tobytes() == rest_before and mesh.local.tobytes() == local_before)
        check(f"N={n}：归档历史未被活动记录覆盖", archive_before == {e: (h.beta, h.reference_edges.tobytes()) for e, h in mesh.archive.items()})
        c, s = mesh.mass_partition()
        weights = c + s
        rigid = sorted(mesh.rigid_vertices())
        velocities_from_body = mesh.linear + np.cross(mesh.angular, mesh.positions[rigid] - mesh.center)
        check(f"N={n}：更换 cluster 质量中心后保留刚体速度场", np.allclose(velocities_from_body, mesh.velocities[rigid], atol=1e-14))
        # 元素归属下的 cluster 质量与 shell 贡献分别累加，对照整网格 lumping。
        body_momentum = c.sum() * mesh.linear
        shell_momentum = (s[:, None] * mesh.velocities).sum(axis=0)
        whole_momentum = (weights[:, None] * mesh.velocities).sum(axis=0)
        check(f"N={n}：body 与 shell 分开装配的线动量一致", np.allclose(body_momentum + shell_momentum, whole_momentum, atol=1e-14))
        datasets[str(n)] = dict(rows=mesh.rows, across=mesh.across, rest_m=mesh.rest.tolist(), triangles=mesh.triangles.tolist(), states=states)

    supported = TriangularCollar(3)
    supported.prescribe_collar_displacement()
    update = supported.advance(set(range(len(supported.rest))), {3, 4, 5})
    check("live support 阻止整行 Bond 释放", not update["accepted"])
    missing_event = TriangularCollar(3)
    missing_event.prescribe_collar_displacement()
    check("缺失 release event 时不推进 Bond 前沿", not missing_event.advance(set(), set())["accepted"])
    no_displacement = TriangularCollar(3)
    check("只有断裂事件而没有变形门槛也不推进", not no_displacement.advance(set(range(len(no_displacement.rest))), set())["accepted"])
    cohesive = TriangularCollar(3)
    cohesive.prescribe_collar_displacement()
    check("Coh 前沿的几何门槛不等同于 beta 清零", set(cohesive.advance(set(), {3, 4, 5}, family="Coh")["accepted"]) == {3, 4, 5})
    existing = TriangularCollar(3)
    existing.active_history[0] = InterfaceHistory(0.17, existing.active_history[0].reference_edges.copy())
    existing.prescribe_collar_displacement()
    existing.advance(set(range(len(existing.rest))), set())
    check("已活动界面不被归档历史反复重置", existing.active_history[0].beta == 0.17)

    broken = TriangularCollar(3)
    broken.prescribe_collar_displacement()
    # 故意示范错误：原地扩大自由集会在同一更新中穿过全部旧 collar。
    propagated = set(broken.free)
    old_collar = broken.independent_vertices() - broken.free
    while True:
        next_set = propagated | (set().union(*(broken.graph[i] for i in propagated)) & old_collar)
        if next_set == propagated:
            break
        propagated = next_set
    correct = broken.advance(set(range(len(broken.rest))), set())
    check("原地递归错误会释放多层而正确版本只释放一层", len(propagated - set(range(broken.across))) > len(correct["accepted"]))
    report = dict(scope="三角网格状态重分类教学，位置由程序指定，不是动力学积分",
                  measured_parameters=dict(width_m=0.019, thickness_m=0.00018, density_kg_m3=1300),
                  teaching_settings=dict(rows=31, across=3, material_step_m=0.002, normal_displacement_m=0.0018,
                                         n_values=[3, 10], n_note="10 为论文设置；3 仅用于教学对照；初始曲线非真实卷绕稳态"),
                  checks=checks, passed=len(checks), datasets=datasets,
                  recursion_counterexample=dict(correct_released=len(correct["accepted"]), wrong_released=len(propagated - set(range(broken.across)))),
                  omissions=["薄壳/接触运动求解", "真实卷绕与黏附配对", "全局质量矩阵与预条件器", "GPU 数据布局和 CCD"])
    return report


if __name__ == "__main__":
    report = validate_mesh()
    Path(__file__).with_name("collar-mesh-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {report['passed']} 项三角网格、状态转移和守恒检查。")
    print("验证的是已指定状态的重分类瞬间，没有模拟真实放带动力学。")
