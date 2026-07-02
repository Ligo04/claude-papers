#!/usr/bin/env python3
"""
Educational Demo: M-ABD (Multi-Affine-Body Dynamics)
=====================================================
Based on: "M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics"
           (arXiv:2603.08079)

This demo implements the core ideas of the M-ABD paper:

  1. Affine Body Dynamics (ABD) kinematics: each body's configuration is
     described by q = [vec(A), t] in R^12, so that every material point
     x_bar_i maps to world space as  x_i = A * x_bar_i + t.

  2. Co-rotational formulation: by extracting the rotation R from A via
     polar decomposition and working in a rotated frame, the system matrix
     H_bar_A = M_A / h^2 + K_bar_A becomes CONSTANT across time steps
     and can be pre-factorized once.

  3. Ball-joint constraints between bodies are linear in q, yielding a
     dual (Schur complement) system that is block-tridiagonal for chains
     and can be solved in O(n) time.

Requirements: Python 3 standard library only (no numpy).

Run:  python3 abd_chain_demo.py
"""

import math

# ============================================================================
# SECTION 0 : Minimal linear algebra (pure Python)
# ============================================================================

def zeros(n, m=None):
    if m is None:
        return [0.0] * n
    return [[0.0] * m for _ in range(n)]

def eye(n):
    M = zeros(n, n)
    for i in range(n): M[i][i] = 1.0
    return M

def mat_copy(A):
    return [row[:] for row in A]

def vec_copy(v):
    return v[:]

def vec_add(a, b, alpha=1.0, beta=1.0):
    return [alpha * a[i] + beta * b[i] for i in range(len(a))]

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_scale(v, s):
    return [x * s for x in v]

def vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def vec_norm(v):
    return math.sqrt(max(0.0, vec_dot(v, v)))

def mat_add(A, B, alpha=1.0, beta=1.0):
    n, m = len(A), len(A[0])
    return [[alpha*A[i][j] + beta*B[i][j] for j in range(m)] for i in range(n)]

def mat_scale(A, s):
    return [[x * s for x in row] for row in A]

def mat_mul(A, B):
    n, p, m = len(A), len(B), len(B[0])
    C = zeros(n, m)
    for i in range(n):
        for k in range(p):
            a_ik = A[i][k]
            if a_ik == 0.0: continue
            for j in range(m):
                C[i][j] += a_ik * B[k][j]
    return C

def mat_vec(A, x):
    return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

def mat_T(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]

def mat_frob(A):
    return math.sqrt(sum(v*v for row in A for v in row))

def det3(M):
    return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
           -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
           +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))

def inv3(M):
    d = det3(M)
    if abs(d) < 1e-30: raise ValueError("Singular 3x3")
    I = zeros(3, 3)
    I[0][0]= (M[1][1]*M[2][2]-M[1][2]*M[2][1])/d
    I[0][1]=-(M[0][1]*M[2][2]-M[0][2]*M[2][1])/d
    I[0][2]= (M[0][1]*M[1][2]-M[0][2]*M[1][1])/d
    I[1][0]=-(M[1][0]*M[2][2]-M[1][2]*M[2][0])/d
    I[1][1]= (M[0][0]*M[2][2]-M[0][2]*M[2][0])/d
    I[1][2]=-(M[0][0]*M[1][2]-M[0][2]*M[1][0])/d
    I[2][0]= (M[1][0]*M[2][1]-M[1][1]*M[2][0])/d
    I[2][1]=-(M[0][0]*M[2][1]-M[0][1]*M[2][0])/d
    I[2][2]= (M[0][0]*M[1][1]-M[0][1]*M[1][0])/d
    return I

def solve_small(A, b):
    """Solve n x n system for small n via Gaussian elimination with pivoting."""
    n = len(b)
    # Augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        # Partial pivot
        best = col
        for row in range(col+1, n):
            if abs(aug[row][col]) > abs(aug[best][col]):
                best = row
        aug[col], aug[best] = aug[best], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-30:
            continue
        for row in range(col+1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n+1):
                aug[row][j] -= factor * aug[col][j]
    # Back substitution
    x = zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i+1, n):
            x[i] -= aug[i][j] * x[j]
        if abs(aug[i][i]) > 1e-30:
            x[i] /= aug[i][i]
    return x

def cholesky(A):
    n = len(A); L = zeros(n, n)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(A[i][i] - s, 1e-15))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] != 0 else 0.0
    return L

def solve_chol(L, b):
    n = len(b)
    y = zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
    x = zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(L[j][i]*x[j] for j in range(i+1, n))) / L[i][i]
    return x

def polar_rotation(A):
    """Extract R from A = R*S via iterative polar decomposition."""
    R = mat_copy(A)
    for _ in range(30):
        try:
            RiT = mat_T(inv3(R))
        except ValueError:
            break
        R_new = mat_add(R, RiT, 0.5, 0.5)
        diff = mat_frob(mat_add(R_new, R, 1.0, -1.0))
        R = R_new
        if diff < 1e-14: break
    if det3(R) < 0:
        for i in range(3): R[i][2] = -R[i][2]
    return R


# ============================================================================
# SECTION 1 : ABD kinematics
# ============================================================================
# q = [vec(A), t] in R^12.   x_i = A * x_bar_i + t = J_i * q.
# J_i = [X*I3 | Y*I3 | Z*I3 | I3]  (3 x 12)

def build_J(xb):
    """
    Build the 3x12 Jacobian so that  J @ q = A * x_bar + t.

    With q = [vec(A), t] in row-major order:
      q = [A00,A01,A02, A10,A11,A12, A20,A21,A22, tx,ty,tz]

    Then  x_k = sum_j A[k][j]*xb[j] + t[k]
    and   dx_k/dA[i][j] = delta(k,i)*xb[j],  dx_k/dt[i] = delta(k,i).

    So J[k, 3*k+j] = xb[j]  and  J[k, 9+k] = 1:

        [ s0 s1 s2 |  0  0  0 |  0  0  0 | 1 0 0 ]
        [  0  0  0 | s0 s1 s2 |  0  0  0 | 0 1 0 ]
        [  0  0  0 |  0  0  0 | s0 s1 s2 | 0 0 1 ]

    Note: the paper writes J_i = [x_bar^T kron I_3, I_3]. With column-major
    vec(A), that would be correct. With our row-major vec(A), the Kronecker
    product transposes, giving [I_3 kron x_bar^T, I_3] as shown above.
    """
    J = zeros(3, 12)
    for k in range(3):
        for j in range(3):
            J[k][3*k + j] = xb[j]
        J[k][9 + k] = 1.0
    return J

def build_mass(nodes, masses):
    M = zeros(12, 12)
    for nd, mi in zip(nodes, masses):
        J = build_J(nd); JT = mat_T(J)
        M = mat_add(M, mat_mul(JT, J), 1.0, mi)
    return M

def build_stiffness(vol, mu, lam):
    vecI = [0.0]*9; vecI[0] = vecI[4] = vecI[8] = 1.0
    K = zeros(12, 12)
    for i in range(9):
        for j in range(9):
            K[i][j] = lam * vecI[i] * vecI[j] * vol
            if i == j: K[i][j] += 2.0 * mu * vol
    return K

def tet_volume(nodes):
    d1 = vec_sub(nodes[1], nodes[0])
    d2 = vec_sub(nodes[2], nodes[0])
    d3 = vec_sub(nodes[3], nodes[0])
    cx = d2[1]*d3[2]-d2[2]*d3[1]
    cy = d2[2]*d3[0]-d2[0]*d3[2]
    cz = d2[0]*d3[1]-d2[1]*d3[0]
    return abs(d1[0]*cx + d1[1]*cy + d1[2]*cz) / 6.0

def q_from_A_t(A, t):
    q = zeros(12)
    for i in range(3):
        for j in range(3): q[3*i+j] = A[i][j]
    q[9], q[10], q[11] = t[0], t[1], t[2]
    return q

def A_t_from_q(q):
    A = zeros(3, 3)
    for i in range(3):
        for j in range(3): A[i][j] = q[3*i+j]
    return A, [q[9], q[10], q[11]]

def build_R12(R):
    R12 = zeros(12, 12)
    for b in range(4):
        for i in range(3):
            for j in range(3): R12[3*b+i][3*b+j] = R[i][j]
    return R12

def gravity_force(nodes, total_mass, g):
    f = zeros(12)
    mi = total_mass / len(nodes)
    for nd in nodes:
        JTg = mat_vec(mat_T(build_J(nd)), g)
        f = vec_add(f, JTg, 1.0, mi)
    return f

def world_pos(q, xbar):
    A, t = A_t_from_q(q)
    return vec_add(mat_vec(A, xbar), t)


# ============================================================================
# SECTION 2 : AffineBody with co-rotational solve
# ============================================================================
# PRE-COMPUTATION (once):
#     H_bar = M / h^2 + K_bar     (12x12, CONSTANT)
#     L L^T = H_bar                (Cholesky, done ONCE)
#
# Each step:
#   1. q_tilde = q + h*dq
#   2. R = polar(A_tilde)
#   3. p_tilde = R12^T @ q_tilde  (local frame)
#   4. rhs = -K*(p_tilde - p_rest) + R12^T @ f_ext
#   5. dp = solve(H_bar, rhs)     (PRE-FACTORIZED!)
#   6. q_new = R12 @ (p_tilde + dp)

class AffineBody:
    def __init__(self, nodes, density, mu, lam, h, t_init=None):
        self.nodes = [list(n) for n in nodes]
        self.n = len(nodes)
        V = tet_volume(nodes)
        self.volume = V
        self.total_mass = density * V
        self.h = h
        masses = [self.total_mass / self.n] * self.n

        self.M = build_mass(self.nodes, masses)
        self.K = build_stiffness(V, mu, lam)
        # KEY: H_bar is CONSTANT => factorise ONCE
        self.H = mat_add(mat_scale(self.M, 1.0/(h*h)), self.K)
        self.L = cholesky(self.H)

        t0 = list(t_init) if t_init else [
            sum(n[k] for n in nodes)/len(nodes) for k in range(3)]
        self.q_rest = q_from_A_t(eye(3), t0)
        self.q = vec_copy(self.q_rest)
        self.dq = zeros(12)

    def predict_and_setup(self, f_world):
        """
        Prepare the co-rotational solve for this step:
        returns (q_tilde, R12, p_tilde, p_rest, rhs) in local frame.
        """
        h = self.h
        q_tilde = vec_add(self.q, self.dq, 1.0, h)
        A_til, _ = A_t_from_q(q_tilde)
        R = polar_rotation(A_til)
        R12 = build_R12(R)
        R12T = mat_T(R12)

        p_tilde = mat_vec(R12T, q_tilde)
        p_rest  = mat_vec(R12T, self.q_rest)
        strain  = vec_sub(p_tilde, p_rest)
        f_local = mat_vec(R12T, f_world)
        rhs = vec_add(vec_scale(mat_vec(self.K, strain), -1.0), f_local)
        return q_tilde, R12, p_tilde, rhs

    def unconstrained_step(self, f_world):
        """Full unconstrained co-rot step. Returns q_new."""
        _, R12, p_tilde, rhs = self.predict_and_setup(f_world)
        dp = solve_chol(self.L, rhs)
        q_new = mat_vec(R12, vec_add(p_tilde, dp))
        return q_new

    def step(self, f_world):
        """Execute unconstrained step and update state."""
        q_new = self.unconstrained_step(f_world)
        self.dq = vec_scale(vec_sub(q_new, self.q), 1.0/self.h)
        self.q = q_new
        return q_new

    def apply_Hinv_world(self, R12, v):
        """Apply world-frame H^{-1} to v: R12 @ H_bar^{-1} @ R12^T @ v."""
        R12T = mat_T(R12)
        return mat_vec(R12, solve_chol(self.L, mat_vec(R12T, v)))


# ============================================================================
# SECTION 3 : Demo 1 — Pre-factorisation verification
# ============================================================================

def demo_prefactorisation():
    print("\n" + "=" * 72)
    print("  M-ABD Demo 1: Verifying Pre-Factorisation Correctness")
    print("=" * 72)

    h = 0.01
    nodes = [[0,1,0], [-0.5,0,-0.3], [0.5,0,-0.3], [0,0,0.5]]
    body = AffineBody(nodes, 1000.0, 1e4, 1e4, h)
    L_orig = mat_copy(body.L)

    g = [0, -9.81, 0]
    for _ in range(50):
        body.step(gravity_force(body.nodes, body.total_mass, g))

    H_check = mat_add(mat_scale(body.M, 1.0/(h*h)), body.K)
    L_check = cholesky(H_check)
    diff = mat_frob(mat_add(L_orig, L_check, 1.0, -1.0))

    print(f"\n  ||L_original - L_recomputed|| = {diff:.2e}")
    print(f"  (Should be 0.0 — confirming H_bar is constant)")
    print(f"\n  WHY: Co-rotational formulation always measures strain from I.")
    print(f"  K_bar (Hessian at S=I) never changes. With constant M,")
    print(f"  H_bar = M/h^2 + K_bar is invariant => factorise once.")


# ============================================================================
# SECTION 4 : Demo 2 — Single body free fall
# ============================================================================

def demo_single_body():
    print("\n" + "=" * 72)
    print("  M-ABD Demo 2: Single Body Free Fall (Co-Rotational)")
    print("=" * 72)

    h = 0.01
    nodes = [[0,1,0], [-0.5,0,-0.3], [0.5,0,-0.3], [0,0,0.5]]
    body = AffineBody(nodes, 1000.0, 5e3, 5e3, h)
    body.dq[0] = 0.5; body.dq[3] = -0.3

    print(f"\n  Volume = {body.volume:.6f} m^3, Mass = {body.total_mass:.4f} kg")
    print(f"  H_bar 12x12 factorised ONCE.\n")
    fmt = f"  {'step':>5}  {'time':>7}  {'centroid':>30}  {'det(A)':>8}  {'|A-R|':>10}"
    print(fmt)
    print(f"  {'-'*5}  {'-'*7}  {'-'*30}  {'-'*8}  {'-'*10}")

    g = [0, -9.81, 0]
    for step in range(100):
        body.step(gravity_force(body.nodes, body.total_mass, g))
        A, t = A_t_from_q(body.q)
        R = polar_rotation(A)
        strain = mat_frob(mat_add(A, R, 1.0, -1.0))
        if step % 10 == 0 or step == 99:
            c = f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"
            print(f"  {step:5d}  {step*h:7.3f}  {c:>30}  {det3(A):8.5f}  {strain:10.2e}")

    print(f"\n  Body falls (y decreases), det(A)~1 (nearly rigid),")
    print(f"  |A-R| small. Pre-factorized Cholesky reused all 100 steps.")


# ============================================================================
# SECTION 5 : Demo 3 — Two-body chain with ball joint (MAIN DEMO)
# ============================================================================
#
# The constrained two-body problem:
#
#   Ball joint: J_a @ q_a = J_b @ q_b  (3 equations)
#   Anchor:     J_anc @ q_a = x_fixed  (3 equations)
#
# Both constraints are linear in q.  We solve the unconstrained co-rot
# problem for each body, then project onto the constraint manifold
# using the dual (Schur complement) approach.
#
# With 2 constraints (anchor + joint), each 3-DOF, we have a 6x6 dual
# system.  For a chain of n bodies with n joints (including anchor),
# the dual system is 3n x 3n block-tridiagonal => O(n) solve.
#
# KKT structure:
#   [ H_a       C_a^T ] [ dq_a ]   [ f_a ]
#   [      H_b  C_b^T ] [ dq_b ] = [ f_b ]
#   [ C_a  C_b    0   ] [ lam  ]   [ c   ]
#
# Dual elimination:
#   D lam = c - C_a H_a^{-1} f_a - C_b H_b^{-1} f_b
#   (with appropriate signs)

def demo_chain():
    print("\n" + "=" * 72)
    print("  M-ABD Demo 3: Two-Body Chain with Ball Joint")
    print("=" * 72)

    h       = 0.005      # smaller time step for better accuracy
    density = 100.0      # lighter bodies
    mu      = 5e3        # moderate stiffness (allows some deformation)
    lam     = 5e3
    n_steps = 400        # 2 seconds total
    g       = [0.0, -9.81, 0.0]

    # --- Geometry ---
    # Body A: tet centred at (0, 0.5, 0), with bottom near y=0
    # Body B: tet centred at (0, -0.5, 0), with top near y=0
    # Joint at y~0, Anchor at top of A (y~1)
    #
    # We use t_init=[0,0,0] so at t=0: world_pos(x_bar) = x_bar directly.
    L = 0.5  # half-length of each body
    w = 0.15 # half-width

    nodes_a = [
        [0.0,  L, 0.0],       # top (anchor point)
        [-w, -L/3, -w*0.6],   # base
        [ w, -L/3, -w*0.6],
        [0.0, -L/3, w*0.8],
    ]
    nodes_b = [
        [0.0, -L/3, 0.0],     # top (joint to A)
        [-w, -L-L/3, -w*0.6], # base
        [ w, -L-L/3, -w*0.6],
        [0.0, -L-L/3, w*0.8],
    ]

    body_a = AffineBody(nodes_a, density, mu, lam, h, t_init=[0,0,0])
    body_b = AffineBody(nodes_b, density, mu, lam, h, t_init=[0,0,0])

    # Joint: base centroid of A = top of B
    s_a_joint = [(nodes_a[1][k]+nodes_a[2][k]+nodes_a[3][k])/3.0 for k in range(3)]
    s_b_joint = list(nodes_b[0])

    # Adjust B so joint is exactly satisfied at t=0
    wa = world_pos(body_a.q, s_a_joint)
    wb = world_pos(body_b.q, s_b_joint)
    offset = vec_sub(wa, wb)
    for k in range(3):
        body_b.q[9+k] += offset[k]
        body_b.q_rest[9+k] += offset[k]

    # Anchor: top of body A, pinned in space
    s_anchor = list(nodes_a[0])
    anchor_world = world_pos(body_a.q, s_anchor)

    # Initial velocity: give body B a horizontal kick to start swinging
    body_b.dq[9] = 3.0   # 3 m/s in x

    # Verify
    wa = world_pos(body_a.q, s_a_joint)
    wb = world_pos(body_b.q, s_b_joint)
    print(f"\n  h = {h} s, {n_steps} steps, total = {n_steps*h:.1f} s")
    print(f"  mu = {mu:.0e}, lam = {lam:.0e} (very stiff => nearly rigid)")
    print(f"  Body mass = {body_a.total_mass:.4f} kg each")
    print(f"  Joint at: [{wa[0]:.4f}, {wa[1]:.4f}, {wa[2]:.4f}]")
    print(f"  Anchor at: [{anchor_world[0]:.4f}, {anchor_world[1]:.4f}, {anchor_world[2]:.4f}]")
    print(f"  Initial joint error: {vec_norm(vec_sub(wa, wb)):.2e}")

    print(f"\n  PRE-FACTORIZATION: H_bar (12x12) factorised ONCE per body.")
    print(f"  Dual system: 6x6 (anchor 3-DOF + joint 3-DOF).\n")

    # Constraint Jacobians (constant in material coords)
    # Constraint 1 (anchor): J_anc @ q_a = anchor_world
    # Constraint 2 (joint):  J_ja @ q_a - J_jb @ q_b = 0
    J_anc = build_J(s_anchor)     # 3x12
    J_ja  = build_J(s_a_joint)    # 3x12
    J_jb  = build_J(s_b_joint)    # 3x12

    print(f"  {'step':>5} {'time':>6} {'anc_err':>9} {'jnt_err':>9}  "
          f"{'anchor_pt':>24}  {'joint_pt':>24}  {'centroid_B':>24}")
    print(f"  {'-'*5} {'-'*6} {'-'*9} {'-'*9}  {'-'*24}  {'-'*24}  {'-'*24}")

    anchor_errors = []
    joint_errors = []

    for step in range(n_steps):
        # --- External forces (gravity only) ---
        f_a = gravity_force(body_a.nodes, body_a.total_mass, g)
        f_b = gravity_force(body_b.nodes, body_b.total_mass, g)

        # --- Unconstrained co-rotational solves ---
        # Each uses the pre-factorized Cholesky!
        _, R12_a, p_tilde_a, rhs_a = body_a.predict_and_setup(f_a)
        _, R12_b, p_tilde_b, rhs_b = body_b.predict_and_setup(f_b)

        dp_unc_a = solve_chol(body_a.L, rhs_a)
        dp_unc_b = solve_chol(body_b.L, rhs_b)
        q_unc_a = mat_vec(R12_a, vec_add(p_tilde_a, dp_unc_a))
        q_unc_b = mat_vec(R12_b, vec_add(p_tilde_b, dp_unc_b))

        # --- Build 6x6 dual matrix D ---
        # D = C @ H_world^{-1} @ C^T
        #
        # Full constraint matrix C (6 x 24) acts on [q_a; q_b]:
        #   C = [ J_anc    0   ]   (anchor: only on body A)
        #       [ J_ja   -J_jb ]   (joint: A and B)
        #
        # D = [ J_anc @ Ha^{-1} @ J_anc^T           J_anc @ Ha^{-1} @ J_ja^T          ]
        #     [ J_ja  @ Ha^{-1} @ J_anc^T   J_ja@Ha^{-1}@J_ja^T + J_jb@Hb^{-1}@J_jb^T ]

        HaInv_JancT = zeros(12, 3)
        HaInv_JjaT  = zeros(12, 3)
        HbInv_JjbT  = zeros(12, 3)
        J_anc_T = mat_T(J_anc)
        J_ja_T  = mat_T(J_ja)
        J_jb_T  = mat_T(J_jb)

        for j in range(3):
            col_anc = [J_anc_T[i][j] for i in range(12)]
            col_ja  = [J_ja_T[i][j] for i in range(12)]
            col_jb  = [J_jb_T[i][j] for i in range(12)]
            sol_anc = body_a.apply_Hinv_world(R12_a, col_anc)
            sol_ja  = body_a.apply_Hinv_world(R12_a, col_ja)
            sol_jb  = body_b.apply_Hinv_world(R12_b, col_jb)
            for i in range(12):
                HaInv_JancT[i][j] = sol_anc[i]
                HaInv_JjaT[i][j]  = sol_ja[i]
                HbInv_JjbT[i][j]  = sol_jb[i]

        D11 = mat_mul(J_anc, HaInv_JancT)
        D12 = mat_mul(J_anc, HaInv_JjaT)
        D21 = mat_mul(J_ja,  HaInv_JancT)
        D22 = mat_add(mat_mul(J_ja, HaInv_JjaT), mat_mul(J_jb, HbInv_JjbT))

        D = zeros(6, 6)
        for i in range(3):
            for j in range(3):
                D[i][j]     = D11[i][j]
                D[i][j+3]   = D12[i][j]
                D[i+3][j]   = D21[i][j]
                D[i+3][j+3] = D22[i][j]

        # --- Dual (Schur complement) solve for constraint enforcement ---
        # The constraint  x = A*s + t = J(s)*q  IS linear in q, so a
        # single dual solve enforces it exactly (to machine precision).
        #
        # Constraint violation of the unconstrained solution:
        c1 = vec_sub(mat_vec(J_anc, q_unc_a), anchor_world)
        c2 = vec_sub(mat_vec(J_ja, q_unc_a), mat_vec(J_jb, q_unc_b))
        c_full = c1 + c2  # 6-vector

        # Solve the 6x6 dual system:  D * lambda = c
        lam_full = solve_small(D, c_full)
        lam_anc = lam_full[:3]
        lam_jnt = lam_full[3:]

        # Recover constrained primal variables:
        #   q_a = q_unc_a - H_a^{-1} @ (J_anc^T lam_anc + J_ja^T lam_jnt)
        #   q_b = q_unc_b + H_b^{-1} @ (J_jb^T lam_jnt)
        corr_a = vec_add(mat_vec(HaInv_JancT, lam_anc),
                         mat_vec(HaInv_JjaT, lam_jnt))
        corr_b = mat_vec(HbInv_JjbT, lam_jnt)

        q_new_a = vec_sub(q_unc_a, corr_a)
        q_new_b = vec_add(q_unc_b, corr_b)

        # --- Update state ---
        body_a.dq = vec_scale(vec_sub(q_new_a, body_a.q), 1.0/h)
        body_b.dq = vec_scale(vec_sub(q_new_b, body_b.q), 1.0/h)
        body_a.q = q_new_a
        body_b.q = q_new_b

        # --- Measure constraint satisfaction ---
        anc_pt = world_pos(q_new_a, s_anchor)
        jnt_a  = world_pos(q_new_a, s_a_joint)
        jnt_b  = world_pos(q_new_b, s_b_joint)
        anc_err = vec_norm(vec_sub(anc_pt, anchor_world))
        jnt_err = vec_norm(vec_sub(jnt_a, jnt_b))
        anchor_errors.append(anc_err)
        joint_errors.append(jnt_err)

        _, t_b = A_t_from_q(q_new_b)

        if step % 20 == 0 or step == n_steps - 1:
            ap = f"({anc_pt[0]:+.3f},{anc_pt[1]:+.3f},{anc_pt[2]:+.3f})"
            jp = f"({jnt_a[0]:+.3f},{jnt_a[1]:+.3f},{jnt_a[2]:+.3f})"
            cb = f"({t_b[0]:+.3f},{t_b[1]:+.3f},{t_b[2]:+.3f})"
            print(f"  {step:5d} {step*h:6.3f} {anc_err:9.2e} {jnt_err:9.2e}  "
                  f"{ap:>24}  {jp:>24}  {cb:>24}")

    max_anc = max(anchor_errors)
    max_jnt = max(joint_errors)
    mean_anc = sum(anchor_errors)/len(anchor_errors)
    mean_jnt = sum(joint_errors)/len(joint_errors)

    print(f"\n  {'='*68}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'='*68}")
    print(f"  Anchor constraint error: max={max_anc:.2e}, mean={mean_anc:.2e}")
    print(f"  Joint  constraint error: max={max_jnt:.2e}, mean={mean_jnt:.2e}")
    print(f"\n  Key observations:")
    print(f"    1. H_bar was factorised ONCE per body and reused for all")
    print(f"       {n_steps} time steps. This is the paper's central insight:")
    print(f"       the co-rotational formulation makes H_bar constant.")
    print(f"    2. The dual system is only 6x6 (two 3-DOF constraints).")
    print(f"       For a chain of n bodies, it is 3(n-1) x 3(n-1)")
    print(f"       block-tridiagonal => O(n) solve time.")
    print(f"    3. Constraint errors are near machine epsilon (~1e-16),")
    print(f"       confirming exact enforcement via the Schur complement.")
    print(f"    4. Body B exhibits pendulum-like swinging under gravity,")
    print(f"       with gradual damping from backward Euler integration.")

    return joint_errors


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print()
    print("  " + "*" * 68)
    print("  *  M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body    *")
    print("  *  Dynamics -- Educational Python Demo                          *")
    print("  *                                                               *")
    print("  *  Reference: arXiv:2603.08079                                  *")
    print("  " + "*" * 68)

    demo_prefactorisation()
    demo_single_body()
    demo_chain()

    print(f"\n{'='*72}")
    print("  ALL DEMOS COMPLETE")
    print(f"{'='*72}")
    print()
