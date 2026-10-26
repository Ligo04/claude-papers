#!/usr/bin/env python3
"""从 TapeSim Eq.4 推导并检查精确 Hessian/HVP，依赖 NumPy。

Hessian 是本学习包的推导，不是作者公开实现。四点几何沿用原教学
demo。数值结果是代数/有限差分检查，不是胶带物理或性能复现。
"""

from pathlib import Path
import json

import numpy as np

from releasable_bond_and_front import VirtualBond


def hessian_vector(bond: VirtualBond, vertices: np.ndarray, direction: np.ndarray,
                   dt: float, gauss_newton: bool = False) -> np.ndarray:
    f = bond.edge_matrix(vertices) @ bond.inv_reference
    u = bond.edge_matrix(direction) @ bond.inv_reference
    a = f @ f.T - np.eye(3)
    scale = dt * dt * bond.volume * bond.stiffness_pa
    # 第一项是残差 Jacobian 贡献；第二项是非线性残差的曲率。
    derivative = 4 * scale * (u @ f.T + f @ u.T) @ f
    if not gauss_newton:
        derivative += 4 * scale * a @ u
    d_ds = derivative @ bond.inv_reference.T
    result = np.empty((4, 3))
    result[1:] = d_ds.T
    result[0] = -result[1:].sum(axis=0)
    return result


def dense_hessian(bond, vertices, dt, gauss_newton=False):
    basis = np.eye(12).reshape(12, 4, 3)
    return np.column_stack([hessian_vector(bond, vertices, u, dt, gauss_newton).ravel() for u in basis])


def check_math():
    reference = np.array([[0., 0., 0.], [0.002, 0., 0.], [0., 0.002, 0.], [0.0006, 0.0006, 0.0002]])
    bond = VirtualBond.create(reference)
    dt = 0.01
    checks = []

    def check(name, condition, **data):
        if not bool(condition):
            raise AssertionError(f"{name}: {data}")
        checks.append(dict(name=name, passed=True, **data))

    deformed = reference.copy()
    deformed[3] += [0.00007, -0.00003, 0.00009]
    deformed[1, 1] += 0.00002
    hessian = dense_hessian(bond, deformed, dt)
    finite = np.empty((12, 12))
    step = 1e-8
    for j in range(12):
        direction = np.eye(12)[j].reshape(4, 3)
        plus = bond.energy_gradient(deformed + step * direction, dt)[1]
        minus = bond.energy_gradient(deformed - step * direction, dt)[1]
        finite[:, j] = ((plus - minus) / (2 * step)).ravel()
    relative_error = float(np.max(np.abs(hessian - finite)) / np.max(np.abs(hessian)))
    check("精确 Hessian 与梯度中心差分一致", relative_error < 1e-7, normalized_max_error=relative_error)
    check("精确 Hessian 对称", np.allclose(hessian, hessian.T, rtol=1e-12, atol=1e-12))

    rng = np.random.default_rng(260928)
    u = rng.normal(size=(4, 3))
    u /= np.linalg.norm(u)
    energy_step = 1e-7
    e_plus = bond.energy_gradient(deformed + energy_step * u, dt)[0]
    e_mid = bond.energy_gradient(deformed, dt)[0]
    e_minus = bond.energy_gradient(deformed - energy_step * u, dt)[0]
    second_finite = (e_plus - 2 * e_mid + e_minus) / energy_step**2
    second_exact = float(u.ravel() @ hessian @ u.ravel())
    check("二阶方向导数与能量中心差分一致", abs(second_exact - second_finite) < 1e-6 * max(1., abs(second_exact)),
          exact=second_exact, finite_difference=second_finite)
    translations = np.tile([0.1, -0.2, 0.3], (4, 1))
    check("平移方向无曲率", np.linalg.norm(hessian @ translations.ravel()) < 1e-12)

    rest_hessian = dense_hessian(bond, reference, dt)
    eigenvalues = np.linalg.eigvalsh(rest_hessian)
    threshold = float(np.max(np.abs(eigenvalues))) * 1e-10
    check("参考态有六个刚体零模态", int(np.sum(np.abs(eigenvalues) < threshold)) == 6,
          eigenvalues=eigenvalues.tolist())
    check("参考态精确 Hessian 等于 Gauss–Newton", np.allclose(rest_hessian, dense_hessian(bond, reference, dt, True), atol=1e-12))
    omega = np.array([0.3, -0.2, 0.7])
    rotation_direction = np.cross(omega, reference - reference.mean(axis=0))
    check("参考态的无穷小旋转为零模态", np.linalg.norm(rest_hessian @ rotation_direction.ravel()) < 1e-12)

    compressed = reference.copy()
    compressed[3, 2] *= 0.5
    normal = np.zeros((4, 3))
    normal[3, 2] = 1
    exact_compressed = dense_hessian(bond, compressed, dt)
    gn_compressed = dense_hessian(bond, compressed, dt, True)
    normal_curvature = float(normal.ravel() @ exact_compressed @ normal.ravel())
    check("压缩态精确 Hessian 存在负曲率", normal_curvature < 0, normal_curvature=normal_curvature)
    check("压缩态 Gauss–Newton 为半正定", float(np.linalg.eigvalsh(gn_compressed).min()) > -1e-12)
    check("压缩态两种 Hessian 确有差别", not np.allclose(exact_compressed, gn_compressed))

    area = np.linalg.norm(np.cross(reference[1] - reference[0], reference[2] - reference[0])) / 2
    stretch_samples = []
    for stretch in [0.5, 1.0, 1.05, 1.1, 1.5]:
        x = reference.copy()
        x[3, 2] *= stretch
        closed_force_z = -4 * bond.stiffness_pa * area / 3 * stretch * (stretch**2 - 1)
        numeric_force_z = float(bond.physical_force_on_point(x, dt)[2])
        check(f"法向伸缩 λ={stretch} 的标量公式", abs(closed_force_z - numeric_force_z) < 1e-10)
        stretch_samples.append(dict(stretch=stretch, force_z_n=numeric_force_z, force_norm_n=abs(numeric_force_z), release=bond.should_release(x, dt)))

    # 正根上二分求阈值，避免把解析力阈值和位移阈值混淆。
    lo, hi = 1., 2.
    for _ in range(70):
        mid = (lo + hi) / 2
        force = 4 * bond.stiffness_pa * area / 3 * mid * (mid**2 - 1)
        if force > bond.break_force_n:
            hi = mid
        else:
            lo = mid
    critical = (lo + hi) / 2
    at_threshold_force = 4 * bond.stiffness_pa * area / 3 * critical * (critical**2 - 1)
    check("法向释放阈值求根", abs(at_threshold_force - bond.break_force_n) < 1e-12, critical_stretch=critical)
    check("压缩示例虽超过阈值也不释放", stretch_samples[0]["force_norm_n"] > bond.break_force_n and not stretch_samples[0]["release"])

    # 一个旋转不变的二次顶点势能，用来独立展示 reduced Hessian 的第二项。
    position = np.array([0.03, 0.02, 0.01])
    tangent = np.cross([0., 0., 1.], position)
    acceleration = np.cross([0., 0., 1.], tangent)
    first_term = float(np.dot(tangent, tangent))
    mapping_term = float(np.dot(position, acceleration))
    check("非线性旋转的链式曲率可精确抵消 JᵀHJ", first_term > 0 and abs(first_term + mapping_term) < 1e-15,
          jt_h_j=first_term, coordinate_curvature=mapping_term, total=first_term + mapping_term)

    report = dict(scope="本包推导的局部能量/Hessian 教学，非作者求解器",
                  reference_vertices_m=reference.tolist(), dt_s=dt, stiffness_pa=bond.stiffness_pa,
                  break_force_n=bond.break_force_n, reference_volume_m3=bond.volume,
                  triangle_area_m2=float(area), critical_normal_stretch=critical,
                  stretch_samples=stretch_samples, checks=checks, passed=len(checks),
                  omissions=["全局 Newton/PCG", "Hessian 正定化策略", "IPC/CCD", "真实胶带参数辨识"])
    return report


if __name__ == "__main__":
    report = check_math()
    target = Path(__file__).with_name("bond-math-validation.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {report['passed']} 项 bond 二阶导数和断裂条件检查。")
    print(f"教学几何下，法向临界伸长比约 {report['critical_normal_stretch']:.8f}。")
