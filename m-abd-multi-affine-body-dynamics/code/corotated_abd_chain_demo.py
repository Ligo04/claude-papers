#!/usr/bin/env python3
"""
Co-rotational Multi-Affine-Body Dynamics (M-ABD) Chain Demo
===========================================================

Implements the core ideas from the M-ABD paper:
  "M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics"
  He, Guo, Guo, Zhao, Matusik, Su, Jiang, Chen, Yang (2026)

Each body is parameterized by an affine transformation:
    x_i = A * x_bar_i + t
where x_bar_i are rest-space positions, A is a 3x3 affine matrix, and t is
a translation vector. The generalized coordinate is:
    q = [vec(A)^T, t^T]^T  in R^12

Key ideas demonstrated:
  1. ABD kinematics with 12-DOF affine bodies
  2. Co-rotational formulation: factor out rotation so the system matrix
     H_bar = (1/h^2) M_A + K_bar_A remains CONSTANT across all timesteps
  3. Ball-joint constraints between adjacent bodies (3 DOFs each)
  4. Dual-space KKT solver exploiting chain (block-tridiagonal) topology
     via the Thomas algorithm
  5. Back-substitution to recover primal (body) updates

We simulate a pendulum chain: N affine bodies hanging under gravity,
connected end-to-end by ball joints. The top body is pinned.

Requires: numpy only.
"""

import numpy as np
import time

# ---- Parameters ----
N_BODIES = 5
DT = 0.01           # time step (seconds) -- paper uses h=0.01
N_STEPS = 300
GRAVITY = np.array([0.0, -9.81, 0.0])
DENSITY = 1000.0
STIFFNESS_MU = 1e5  # Lame mu (shear)
STIFFNESS_LAM = 1e5 # Lame lambda (bulk)
BODY_SIZE = 0.1     # half-extent of cubic body


def polar_decomp(A):
    """Polar decomposition A = R @ S via SVD."""
    U, s, Vt = np.linalg.svd(A)
    det = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, np.sign(det)])
    R = U @ D @ Vt
    return R


def vec(A):
    """Columnwise vectorization of 3x3 matrix -> 9-vector."""
    return A.flatten(order='F')


def unvec(v):
    """Inverse of vec: 9-vector -> 3x3 matrix."""
    return v.reshape(3, 3, order='F')


# ---- Precompute rest-shape quantities for a cube ----
def precompute_body(half_ext):
    """
    Precompute constant matrices for an affine body (cube).
    Returns M_A (12x12), K_bar_A (12x12), H_bar (12x12), H_bar_inv.
    """
    s = half_ext
    vol = (2 * s) ** 3
    mass = DENSITY * vol

    # For a uniform cube, integral of x_bar x_bar^T dV = vol * s^2/3 * I
    Ixx = vol * s**2 / 3.0

    # Generalized mass M_A (12x12) = J^T M J
    M_A = np.zeros((12, 12))
    M_A[:9, :9] = Ixx * np.eye(9)
    M_A[9:, 9:] = mass * np.eye(3)

    # Co-rotational stiffness K_bar_A (12x12)
    # Linear elasticity Hessian w.r.t. vec(A), integrated over body at rest
    mu, lam = STIFFNESS_MU, STIFFNESS_LAM

    I9 = np.eye(9)
    # Permutation for transpose: vec(A^T) = P @ vec(A)
    P9 = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            P9[3*i+j, 3*j+i] = 1.0
    e_I = vec(np.eye(3))
    K_AA = vol * (mu * (I9 + P9) + lam * np.outer(e_I, e_I))

    K_bar_A = np.zeros((12, 12))
    K_bar_A[:9, :9] = K_AA

    # Pre-factorized system matrix (THE KEY: this is CONSTANT)
    H_bar = (1.0 / DT**2) * M_A + K_bar_A
    H_bar_inv = np.linalg.inv(H_bar)

    return mass, vol, M_A, K_bar_A, H_bar, H_bar_inv


class AffineBody:
    """A single affine body with q = [vec(A), t] in R^12."""

    def __init__(self, center, mass, vol, M_A, K_bar_A, H_bar, H_bar_inv):
        self.A = np.eye(3)
        self.t = np.array(center, dtype=float)
        self.v = np.zeros(12)  # velocity dq/dt
        self.mass = mass
        self.vol = vol
        self.M_A = M_A
        self.K_bar_A = K_bar_A
        self.H_bar = H_bar
        self.H_bar_inv = H_bar_inv
        self.pinned = False

    @property
    def q(self):
        return np.concatenate([vec(self.A), self.t])

    @q.setter
    def q(self, val):
        self.A = unvec(val[:9])
        self.t = val[9:]

    def tip_pos(self, local_pt):
        return self.A @ local_pt + self.t


class BallJoint:
    """
    Ball joint: A_a @ a_bar + t_a = A_b @ b_bar + t_b
    C(q_a, q_b) = J_a @ q_a + J_b @ q_b = 0  (3 equations)
    The Jacobians are CONSTANT -- a key advantage of ABD.
    """
    def __init__(self, idx_a, idx_b, a_bar, b_bar):
        self.a = idx_a
        self.b = idx_b
        # J_a: dC/dq_a = [a_bar^T kron I_3, I_3]
        self.J_a = np.zeros((3, 12))
        self.J_a[:, :9] = np.kron(a_bar.reshape(1, 3), np.eye(3))
        self.J_a[:, 9:] = np.eye(3)
        # J_b: dC/dq_b = -[b_bar^T kron I_3, I_3]
        self.J_b = np.zeros((3, 12))
        self.J_b[:, :9] = -np.kron(b_bar.reshape(1, 3), np.eye(3))
        self.J_b[:, 9:] = -np.eye(3)

    def violation(self, bodies):
        return self.J_a @ bodies[self.a].q + self.J_b @ bodies[self.b].q


def compute_rhs(body):
    """
    Build RHS for implicit Euler:
      H_bar @ delta_q = f_A

    f_A = (1/h) M_A @ v + f_ext + f_elastic
    delta_q = q_{n+1} - q_n
    """
    # Inertial momentum
    rhs = (1.0 / DT) * body.M_A @ body.v

    # Gravity
    rhs[9:] += body.mass * GRAVITY

    # Elastic restoring force (co-rotational)
    R = polar_decomp(body.A)
    S = R.T @ body.A
    strain = S - np.eye(3)
    dPsi_dS = body.vol * (STIFFNESS_MU * (strain + strain.T) +
                           STIFFNESS_LAM * np.trace(strain) * np.eye(3))
    dPsi_dA = R @ dPsi_dS
    rhs[:9] -= vec(dPsi_dA)

    return rhs


def solve_dual_chain(bodies, joints):
    """
    Dual-space KKT solver for chain topology.

    1. Compute unconstrained RHS for each body
    2. Build block-tridiagonal Schur complement: S = J H^{-1} J^T
    3. Solve with Thomas algorithm: O(K)
    4. Back-substitute: delta_q = H^{-1} (rhs - J^T lambda)
    """
    n_b = len(bodies)
    n_j = len(joints)

    # Step 1: unconstrained RHS
    rhs = []
    for b in bodies:
        rhs.append(np.zeros(12) if b.pinned else compute_rhs(b))

    # Step 2: Build dual system blocks
    D = []  # diagonal blocks (3x3)
    for jt in joints:
        Dk = (jt.J_a @ bodies[jt.a].H_bar_inv @ jt.J_a.T +
              jt.J_b @ bodies[jt.b].H_bar_inv @ jt.J_b.T)
        D.append(Dk)

    B = []  # off-diagonal blocks (3x3)
    for i in range(n_j - 1):
        shared = joints[i].b
        Bi = joints[i].J_b @ bodies[shared].H_bar_inv @ joints[i+1].J_a.T
        B.append(Bi)

    # Dual RHS
    g = []
    for jt in joints:
        gk = (jt.J_a @ bodies[jt.a].H_bar_inv @ rhs[jt.a] +
              jt.J_b @ bodies[jt.b].H_bar_inv @ rhs[jt.b])
        # Constraint stabilization: add C/h to correct drift
        gk += jt.violation(bodies) / DT
        g.append(gk)

    # Step 3: Block Thomas algorithm
    Dp = [None] * n_j
    gp = [None] * n_j
    Dp[0] = D[0].copy()
    gp[0] = g[0].copy()
    for i in range(1, n_j):
        L = B[i-1].T @ np.linalg.inv(Dp[i-1])
        Dp[i] = D[i] - L @ B[i-1]
        gp[i] = g[i] - L @ gp[i-1]

    lam = [None] * n_j
    lam[-1] = np.linalg.solve(Dp[-1], gp[-1])
    for i in range(n_j - 2, -1, -1):
        lam[i] = np.linalg.solve(Dp[i], gp[i] - B[i] @ lam[i+1])

    # Step 4: Primal back-substitution
    adj_rhs = [r.copy() for r in rhs]
    for i, jt in enumerate(joints):
        adj_rhs[jt.a] -= jt.J_a.T @ lam[i]
        adj_rhs[jt.b] -= jt.J_b.T @ lam[i]

    dq = []
    for k, b in enumerate(bodies):
        dq.append(np.zeros(12) if b.pinned else b.H_bar_inv @ adj_rhs[k])

    return dq


def build_chain(n, precomp):
    """Build a pendulum chain of n bodies."""
    mass, vol, M_A, K_bar_A, H_bar, H_bar_inv = precomp
    bodies = []
    joints = []
    s = BODY_SIZE

    for i in range(n):
        center = np.array([0.0, -(2*s)*i - s, 0.0])
        b = AffineBody(center, mass, vol, M_A, K_bar_A, H_bar, H_bar_inv)
        if i == 0:
            b.pinned = True
        bodies.append(b)

    # Give a lateral push to start swinging
    for b in bodies[1:]:
        b.v[9] = 1.0  # v_x = 1 m/s

    # Ball joints: bottom of body i to top of body i+1
    for i in range(n - 1):
        a_bar = np.array([0.0, -s, 0.0])
        b_bar = np.array([0.0,  s, 0.0])
        joints.append(BallJoint(i, i+1, a_bar, b_bar))

    return bodies, joints


def timestep(bodies, joints):
    """One implicit Euler step."""
    dq = solve_dual_chain(bodies, joints)
    for k, b in enumerate(bodies):
        if not b.pinned:
            b.v = dq[k] / DT
            b.q = b.q + dq[k]


def total_energy(bodies):
    KE, PE, SE = 0.0, 0.0, 0.0
    for b in bodies:
        KE += 0.5 * b.v @ b.M_A @ b.v
        PE -= b.mass * GRAVITY @ b.t
        R = polar_decomp(b.A)
        S = R.T @ b.A
        strain = S - np.eye(3)
        SE += 0.5 * b.vol * (STIFFNESS_MU * np.sum((strain + strain.T)**2) / 2 +
                              STIFFNESS_LAM * np.trace(strain)**2)
    return KE, PE, SE


def main():
    print("=" * 72)
    print("  Co-rotational Multi-Affine-Body Dynamics (M-ABD) Chain Demo")
    print("=" * 72)

    precomp = precompute_body(BODY_SIZE)
    bodies, joints = build_chain(N_BODIES, precomp)

    n_dof = 12 * N_BODIES
    n_con = 3 * (N_BODIES - 1)
    print(f"  Bodies: {N_BODIES}  |  dt: {DT}s  |  Steps: {N_STEPS}")
    print(f"  Primal DOFs: {n_dof}  |  Dual DOFs: {n_con}  |  Ratio: {n_dof/n_con:.1f}x")
    print(f"  Stiffness: mu={STIFFNESS_MU:.0e}, lam={STIFFNESS_LAM:.0e}")
    print(f"  H_bar pre-factorized once (12x12 per body) -- reused every step")
    print("-" * 72)
    hdr = f"  {'step':>5s} | {'KE':>8s} | {'PE':>8s} | {'SE':>8s} | {'Total':>9s} | {'|C|_max':>9s} | tip_x    tip_y"
    print(hdr)
    print("-" * 72)

    t0 = time.perf_counter()

    for step in range(N_STEPS):
        timestep(bodies, joints)

        if step % 30 == 0 or step == N_STEPS - 1:
            KE, PE, SE = total_energy(bodies)
            total = KE + PE + SE
            max_c = max(np.linalg.norm(jt.violation(bodies)) for jt in joints)
            tip = bodies[-1].tip_pos(np.array([0.0, -BODY_SIZE, 0.0]))
            print(f"  {step:5d} | {KE:8.3f} | {PE:8.3f} | {SE:8.5f} | "
                  f"{total:9.3f} | {max_c:9.2e} | {tip[0]:+7.4f}  {tip[1]:+7.4f}")

    elapsed = time.perf_counter() - t0

    print("-" * 72)
    print(f"  Wall time: {elapsed:.3f}s  ({N_STEPS/elapsed:.0f} steps/s)")
    print(f"  Avg time/step: {elapsed/N_STEPS*1000:.2f} ms")
    print()

    print("  Near-rigidity check: ||A_k - R_k||_F")
    for i, b in enumerate(bodies):
        R = polar_decomp(b.A)
        err = np.linalg.norm(b.A - R, 'fro')
        print(f"    Body {i}: {err:.2e}" + ("  (pinned)" if b.pinned else ""))

    print()
    print("  Joint constraint satisfaction: ||C_k||")
    for i, jt in enumerate(joints):
        c = np.linalg.norm(jt.violation(bodies))
        print(f"    Joint {i}: {c:.2e}")

    print()
    print("  KEY TAKEAWAYS:")
    print("    * H_bar = (1/h^2)M_A + K_bar_A was factored ONCE, never rebuilt")
    print("    * Block-tridiagonal Thomas algorithm solved dual system in O(K)")
    print("    * Ball joints are LINEAR constraints in ABD (constant Jacobians)")
    print("    * Near-rigidity (||A-R|| ~ 0) validates the co-rotational approx")
    print("=" * 72)


if __name__ == "__main__":
    main()
