"""
Convex Primitive Decomposition - 核心算法简化实现

演示论文的核心思想：
1. 用 nn^T 外积作为线性算子
2. 特征分解确定基元方向
3. 基于超量体积的贪心合并

依赖: numpy (pip install numpy)
"""
import numpy as np
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import List, Tuple, Set


@dataclass
class OBB:
    """定向包围盒 - 最基础的基元"""
    center: np.ndarray
    axes: np.ndarray      # 3x3, 每行一个轴方向
    half_extents: np.ndarray  # 3个半径长

    @property
    def volume(self) -> float:
        return 8.0 * np.prod(self.half_extents)


@dataclass
class Primitive:
    """代表一个面集合及其对应的基元"""
    Q: np.ndarray           # 3x3 线性算子
    face_indices: Set[int]  # 包含的面索引
    points: np.ndarray      # 所有被包含的顶点
    obb: OBB = None

    def fit_obb(self):
        """从 Q 的特征分解拟合 OBB"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.Q)
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, idx].T  # 每行一个轴

        # 在局部坐标系下计算包围盒
        local_coords = self.points @ axes.T
        upper = np.max(local_coords, axis=0)
        lower = np.min(local_coords, axis=0)

        half_extents = np.maximum((upper - lower) / 2, 1e-3)
        center_local = (upper + lower) / 2
        center = center_local @ axes

        self.obb = OBB(center=center, axes=axes, half_extents=half_extents)

    @property
    def volume(self) -> float:
        if self.obb is None:
            self.fit_obb()
        return self.obb.volume


def compute_face_quadric(vertices: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """计算单个面的线性算子 Q = area * (nn^T + ε*tt^T)"""
    # 计算法线（未归一化，长度=2*面积）
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)

    area = np.linalg.norm(normal) / 2
    if area < 1e-10:
        return np.zeros((3, 3))

    n = normal / (2 * area)  # 单位法线

    # 切线方向（最长边方向）
    edges = [edge1, edge2, v2 - v1]
    lengths = [np.linalg.norm(e) for e in edges]
    tangent = edges[np.argmax(lengths)]
    tangent = tangent / np.linalg.norm(tangent)

    # Q = area * (nn^T + ε*tt^T)
    Q = area * (np.outer(n, n) + epsilon * np.outer(tangent, tangent))
    return Q


def merge_cost(prim_a: Primitive, prim_b: Primitive) -> Tuple[float, Primitive]:
    """计算合并两个基元的代价（超量体积）"""
    # 合并 Q 矩阵
    merged_Q = prim_a.Q + prim_b.Q

    # 合并点集
    merged_points = np.vstack([prim_a.points, prim_b.points])
    merged_faces = prim_a.face_indices | prim_b.face_indices

    # 创建合并后的基元
    merged = Primitive(Q=merged_Q, face_indices=merged_faces, points=merged_points)
    merged.fit_obb()

    # 超量体积 = V(merged) - V(a) - V(b)
    cost = merged.volume - prim_a.volume - prim_b.volume
    return cost, merged


def decompose(vertices: np.ndarray, faces: np.ndarray,
              target_n: int = 5, volume_threshold: float = float('inf')) -> List[Primitive]:
    """
    凸基元分解主算法

    Args:
        vertices: (V, 3) 顶点坐标
        faces: (F, 3) 面索引（三角形）
        target_n: 目标基元数
        volume_threshold: 超量体积阈值

    Returns:
        分解后的基元列表
    """
    num_faces = len(faces)
    print(f"输入: {len(vertices)} 顶点, {num_faces} 面")
    print(f"目标: {target_n} 个基元")

    # Step 1: 初始化每个面为一个基元
    primitives = []
    for i, face in enumerate(faces):
        face_verts = vertices[face]
        Q = compute_face_quadric(face_verts)
        prim = Primitive(Q=Q, face_indices={i}, points=face_verts.copy())
        prim.fit_obb()
        primitives.append(prim)

    # 面→基元的映射（用并查集思想）
    face_to_prim = list(range(num_faces))

    def find_prim(face_idx):
        while face_to_prim[face_idx] != face_idx:
            face_to_prim[face_idx] = face_to_prim[face_to_prim[face_idx]]
            face_idx = face_to_prim[face_idx]
        return face_idx

    # Step 2: 建立面邻接关系
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([face[j], face[(j+1) % 3]]))
            if edge in edge_to_faces:
                edge_to_faces[edge].append(fi)
            else:
                edge_to_faces[edge] = [fi]

    # Step 3: 初始化优先队列
    pq = []
    counter = 0
    num_primitives = num_faces

    for edge, adj_faces in edge_to_faces.items():
        if len(adj_faces) == 2:
            f0, f1 = adj_faces
            cost, merged = merge_cost(primitives[f0], primitives[f1])
            if cost <= volume_threshold:
                heappush(pq, (cost, counter, f0, f1))
                counter += 1

    # Step 4: 贪心合并
    while pq and num_primitives > target_n:
        cost, _, f0, f1 = heappop(pq)

        # 检查是否过期（面可能已被合并）
        p0 = find_prim(f0)
        p1 = find_prim(f1)
        if p0 == p1:
            continue

        # 重新计算代价（惰性更新）
        actual_cost, merged = merge_cost(primitives[p0], primitives[p1])

        # 执行合并
        primitives[p0] = merged
        face_to_prim[p1] = p0
        num_primitives -= 1

        # 更新邻居代价（简化版：重新加入所有相关边）
        for fi in merged.face_indices:
            face = faces[fi]
            for j in range(3):
                edge = tuple(sorted([face[j], face[(j+1) % 3]]))
                if edge in edge_to_faces and len(edge_to_faces[edge]) == 2:
                    nf0, nf1 = edge_to_faces[edge]
                    np0, np1 = find_prim(nf0), find_prim(nf1)
                    if np0 != np1:
                        new_cost, _ = merge_cost(primitives[np0], primitives[np1])
                        if new_cost <= volume_threshold:
                            heappush(pq, (new_cost, counter, np0, np1))
                            counter += 1

    # 收集唯一基元
    unique_prims = {}
    for fi in range(num_faces):
        pi = find_prim(fi)
        unique_prims[pi] = primitives[pi]

    result = list(unique_prims.values())
    print(f"输出: {len(result)} 个基元")
    total_volume = sum(p.volume for p in result)
    print(f"总体积: {total_volume:.4f}")

    return result


# === 演示 ===

def create_test_mesh():
    """创建一个简单的 L 形网格用于测试"""
    vertices = np.array([
        # 底部长方体
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
        [0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1],
        # 竖直部分
        [0, 1, 0], [0.8, 1, 0], [0.8, 2.5, 0], [0, 2.5, 0],
        [0, 1, 1], [0.8, 1, 1], [0.8, 2.5, 1], [0, 2.5, 1],
    ], dtype=float)

    # 三角化的面（手动定义一些面）
    faces = np.array([
        # 底部顶面
        [0, 1, 2], [0, 2, 3],
        # 底部底面
        [4, 6, 5], [4, 7, 6],
        # 底部前面
        [0, 5, 1], [0, 4, 5],
        # 底部后面
        [3, 2, 6], [3, 6, 7],
        # 底部左面
        [0, 3, 7], [0, 7, 4],
        # 底部右面
        [1, 5, 6], [1, 6, 2],
        # 竖直前面
        [8, 13, 9], [8, 12, 13],
        # 竖直后面
        [11, 10, 14], [11, 14, 15],
        # 竖直左面
        [8, 11, 15], [8, 15, 12],
        # 竖直右面
        [9, 13, 14], [9, 14, 10],
        # 竖直顶面
        [10, 11, 15], [10, 15, 14],
    ])

    return vertices, faces


if __name__ == "__main__":
    print("=" * 60)
    print("凸基元分解 (Convex Primitive Decomposition) - 简化演示")
    print("=" * 60)
    print()

    vertices, faces = create_test_mesh()
    print("测试网格: L 形物体")
    print()

    # 分解为 2 个基元（期望：底部长方体 + 竖直部分）
    primitives = decompose(vertices, faces, target_n=2)
    print()

    for i, prim in enumerate(primitives):
        print(f"基元 {i+1}:")
        print(f"  包含面数: {len(prim.face_indices)}")
        print(f"  OBB 中心: {prim.obb.center}")
        print(f"  OBB 半径长: {prim.obb.half_extents}")
        print(f"  体积: {prim.volume:.4f}")
        print()

    print("=" * 60)
    print("核心原理验证：特征分解给出方向")
    print("=" * 60)
    print()

    # 演示单个面的 Q
    face_verts = vertices[faces[0]]
    Q = compute_face_quadric(face_verts)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    idx = np.argsort(eigenvalues)[::-1]

    print(f"面 0 的顶点: {face_verts}")
    print(f"Q 矩阵:\n{Q}")
    print(f"特征值 (降序): {eigenvalues[idx]}")
    print(f"最大特征向量 (≈法线): {eigenvectors[:, idx[0]]}")

    actual_normal = np.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
    actual_normal = actual_normal / np.linalg.norm(actual_normal)
    print(f"实际法线: {actual_normal}")
