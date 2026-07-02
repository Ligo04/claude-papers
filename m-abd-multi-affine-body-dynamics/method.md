# M-ABD: Method Breakdown

**Paper**: arXiv:2603.08079 -- M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics

---

## Overview

The M-ABD framework has four main stages:
1. **Single-body co-rotated ABD** -- Pre-factorize each body's system matrix
2. **Joint constraint formulation** -- Express joints via Control Point (CP) coordinates
3. **Dual-space KKT projection** -- Eliminate primal DOFs, solve in compact dual space
4. **Topology-aware solve** -- Exploit chain/tree/loop structure for efficiency

---

## Architecture Diagram

```
INPUT: Articulated assembly (bodies + joints + external forces)
  |
  v
+----------------------------------------------------------+
|  PHASE 0: PREPROCESSING (once)                           |
|  For each body j:                                        |
|    1. Compute rest-shape quantities (mass, stiffness)    |
|    2. Compute M_A (constant 12x12 generalized mass)      |
|    3. Compute K_bar_A (constant co-rotated stiffness)    |
|    4. Pre-factorize H_bar_A = M_A/h^2 + K_bar_A         |
|    5. Set up control tetrahedron T                       |
|  Build joint topology graph                              |
|  Compute constant constraint Jacobians (for linear Cs)   |
|  Classify topology: chain / tree / tree+loops / general  |
+----------------------------------------------------------+
  |
  v  (each time step)
+----------------------------------------------------------+
|  PHASE 1: PREDICT                                        |
|  q_tilde = q_n + h*v_n + h^2 * M^{-1} * f_ext           |
+----------------------------------------------------------+
  |
  v
+----------------------------------------------------------+
|  PHASE 2: NEWTON ITERATION (repeat until convergence)    |
|                                                          |
|  +----------------------------------------------------+  |
|  | 2a. Per-body co-rotational solve (parallel)        |  |
|  |   - Extract R from A via polar decomposition       |  |
|  |     (or cheap normalization for near-rigid A)      |  |
|  |   - Rotate residual: f_bar = diag4(R^T) * f_A     |  |
|  |   - Solve: delta_bar = H_bar_A^{-1} * f_bar       |  |
|  |   - Rotate back: delta = diag4(R) * delta_bar      |  |
|  +----------------------------------------------------+  |
|       |                                                  |
|       v                                                  |
|  +----------------------------------------------------+  |
|  | 2b. Assemble dual system                           |  |
|  |   - Compute unconstrained body updates             |  |
|  |   - Form Schur complement:                         |  |
|  |     S = nabla_C * H_tilde^{-1} * nabla_C^T        |  |
|  |   - Form RHS: r = nabla_C * H_tilde^{-1} * f_A    |  |
|  +----------------------------------------------------+  |
|       |                                                  |
|       v                                                  |
|  +----------------------------------------------------+  |
|  | 2c. Solve dual system (topology-dependent)         |  |
|  |   - Chain:   Thomas algorithm (block tridiag)      |  |
|  |   - Tree:    ABD-ABA (leaf-to-root, root-to-leaf)  |  |
|  |   - Loops:   Direct solve of loop closure Delta_S  |  |
|  |   - General: Bidirectional Gauss-Seidel            |  |
|  +----------------------------------------------------+  |
|       |                                                  |
|       v                                                  |
|  +----------------------------------------------------+  |
|  | 2d. Recover primal update                          |  |
|  |   delta_q = H_tilde^{-1}*(f_A - nabla_C^T*lambda) |  |
|  |   q_{k+1} = q_k + delta_q                         |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
  |
  v
+----------------------------------------------------------+
|  PHASE 3: VELOCITY UPDATE                                |
|  v_{n+1} = (q_{n+1} - q_n) / h                          |
+----------------------------------------------------------+
  |
  v
OUTPUT: Updated positions q_{n+1}, velocities v_{n+1}
```

---

## Step-by-Step: Component Details

### Component 1: Single-Body Co-Rotational ABD

#### State Representation

Each body stores 12 DOFs packed as a vector q_A in R^12:

    q_A = [a_1, a_2, a_3, t]^T

where a_1, a_2, a_3 are the columns of the 3x3 affine matrix A, and t is the
3x1 translation. Any material point x_bar_i in the rest configuration maps to:

    x_i = A * x_bar_i + t

The spatial-to-generalized Jacobian is:

    x_i = J_i * q,    J_i = [x_bar_i^T (x) I_3, I_3]

where (x) denotes Kronecker product. This J_i is **constant** -- it depends
only on the rest-shape geometry, not on the current configuration.

#### Mass Matrix

The mass matrix M_A is a constant 12x12 SPD matrix derived from the standard
FEM mass matrix of the affine element:

    M_A = integral_Omega rho * [x_bar x_bar^T (x) I_3,  x_bar (x) I_3]
                               [x_bar^T (x) I_3,        m * I_3       ]

This is computed once during preprocessing and stored.

#### Rigidity Energy

To penalize deviation from rigidity, a constitutive energy (e.g., stable
neo-Hookean) is applied. The deformation gradient F equals A (constant over
the body since the mapping is affine), so the energy density is:

    Psi(A) = mu/2 * (||A||_F^2 - 3) - mu * (det(A) - 1) + lambda/2 * (det(A) - 1)^2

where mu, lambda are Lame parameters chosen large enough to enforce
near-rigidity. The total rigidity energy integrates this over the body volume.

#### Co-Rotational Decomposition

Extract the rotation R from A via polar decomposition:

    A = R * S,    R in SO(3),  S symmetric positive definite

Since A is near-orthogonal (the body is near-rigid), S is close to I_3, and
the constitutive response in the co-rotated frame is nearly linear.

The key co-rotation property for ABD: the FEM-to-affine Jacobian satisfies

    diag_N(R) * J = J * diag_4(R)

where diag_N(R) is the N-fold block-diagonal of R (one per vertex) and
diag_4(R) is the 4-fold block-diagonal. This means the reduced stiffness
factors as:

    K_A = J^T * K_tilde * J = diag_4(R) * K_bar_A * diag_4(R^T)

where K_bar_A is the rest-shape stiffness (constant).

#### Pre-Factorization

Define the constant per-body Hessian in the co-rotated frame:

    H_bar_A = M_A / h^2  +  K_bar_A

This 12x12 SPD matrix is factorized once (e.g., Cholesky) during preprocessing.
Cost: O(12^3) = O(1) per body, done once total.

#### Per-Newton-Step Update

Given the current residual f_A (gradient of the incremental potential):

    1. Rotate into co-rotational frame:  f_bar = diag4(R^T) * f_A
    2. Solve using pre-factorized H_bar:  delta_bar = H_bar_A^{-1} * f_bar
    3. Rotate back to world frame:        delta = diag4(R) * delta_bar

Cost per iteration: O(12^2) = O(1) per body. These are BLAS level-1 (vector
scaling) and level-2 (matrix-vector multiply) operations. The reported speedup
over vanilla ABD is 161ms down to 27ms for a single body.

---

### Component 2: Joint Constraints in Affine Coordinates

#### Control Point (CP) Parameterization

Rather than working directly with the 12-DOF q_A, re-parameterize to
control point coordinates y via a constant linear map:

    q_A = T * y

where T is a constant matrix mapping CP positions to affine DOFs. Control
points are four physical points forming a tetrahedron on the body (e.g., joint
attachment points, center of mass, body extremities). This makes joint
formulation intuitive: a ball joint is simply the constraint that two control
points on adjacent bodies coincide.

#### Joint Types

**Ball Joint** (3 constraint equations, removes 3 translational DOFs):

    C_ball(q_A, q_B) = x_A(p_A) - x_B(p_B) = 0

where p_A, p_B are the joint attachment points on bodies A and B. Since
x = A * x_bar + t is linear in q_A, this constraint is *exactly linear*:

    C_ball = [T_A_row(p_A)] * q_A  -  [T_B_row(p_B)] * q_B  =  0

The Jacobian is constant. No configuration dependence at all.

**Hinge Joint** (5 constraint equations, leaves 1 rotational DOF):

    C_hinge = [C_ball;  C_axis]

The ball joint part (3 eqs) is linear. The axis constraint (2 eqs) enforces
that the rotation axis on body A aligns with that on body B. This involves
the columns of A (which encode the body frame orientation). In nonlinear
form it uses the local rotation R_H at the joint.

**Universal Joint** (4 constraint equations, leaves 2 rotational DOFs):

    C_universal = [C_ball;  C_perp]

Ball joint (3 eqs) plus one perpendicularity constraint between designated
axes on the two bodies.

**Prismatic Joint** (5 constraint equations, leaves 1 translational DOF):

    C_prismatic = [C_orient;  C_slide]

Orientation is locked (3 constraints) and translation is restricted to one
axis (2 additional constraints). The orientation lock uses the affine matrix
columns.

#### Linear vs. Nonlinear Formulations

The paper presents both:

- **Linear**: Replace R with A in all constraint equations. All Jacobians
  become constant. But the constraint space dimension is larger (more DOFs
  constrained redundantly), leading to a bigger dual system.

- **Nonlinear** (preferred): Use the minimal constraint formulation with
  extracted rotation R. Jacobians are mildly configuration-dependent (through
  R), but the dual system is smaller. The paper prefers this because smaller
  dual matrices dominate the overall solve cost.

---

### Component 3: Dual-Space KKT System

#### The Primal Problem

The full constrained optimization at each time step is:

    min_q  1/(2h^2) * (q - q_tilde)^T M (q - q_tilde)  +  Psi(q)
    s.t.   C(q) = 0

The KKT system (Newton step) is:

    [H        nabla_C^T] [delta_q]   [f  ]
    [nabla_C  0        ] [lambda  ] = [-C ]

where H is the 12M x 12M primal Hessian and nabla_C is the constraint Jacobian.

#### Schur Complement Elimination

Because H is block-diagonal (each body's Hessian is independent, thanks to
co-rotational ABD), we can eliminate delta_q analytically:

    delta_q = H^{-1} * (f - nabla_C^T * lambda)

Substituting into the constraint equation:

    (nabla_C * H^{-1} * nabla_C^T) * lambda  =  nabla_C * H^{-1} * f  +  C

Define the dual (Schur complement) matrix:

    S = nabla_C * H^{-1} * nabla_C^T

This matrix has dimension equal to the total number of constraint DOFs
(sum of constraint ranks across all joints), which is typically much smaller
than 12M. For example, a chain of 1000 bodies with hinge joints has
12 * 1000 = 12000 primal DOFs but only 5 * 999 = 4995 dual DOFs.

#### Assembly of S

Each joint j connecting bodies A and B contributes a block to S:

    S_jj = J_A * H_A^{-1} * J_A^T  +  J_B * H_B^{-1} * J_B^T

where J_A, J_B are the constraint Jacobians w.r.t. bodies A and B. Since
H_A^{-1} is available from the pre-factorized co-rotational Hessian (rotated
appropriately), each block of S is cheap to compute.

Off-diagonal blocks S_jk are nonzero only when joints j and k share a body,
giving S a sparsity pattern determined by the joint-body graph.

---

### Component 4: Specialized Dual Solvers

#### Chains: Block-Tridiagonal Thomas Algorithm

For a serial chain of M bodies connected by K = M-1 joints, the dual matrix S
is block-tridiagonal:

```
S = [D_1   L_2^T                    ]
    [L_2   D_2    L_3^T             ]
    [       L_3   D_3   ...         ]
    [              ...       L_K^T  ]
    [                    L_K  D_K   ]
```

where D_j is the diagonal block for joint j and L_j is the sub-diagonal block
coupling adjacent joints.

**Forward sweep** (j = 1 to K):

    D'_1 = D_1
    For j = 2, ..., K:
        D'_j = D_j - L_j * (D'_{j-1})^{-1} * L_j^T

**Back substitution** (j = K to 1):

    lambda_K = (D'_K)^{-1} * r'_K
    For j = K-1, ..., 1:
        lambda_j = (D'_j)^{-1} * (r'_j - L_{j+1}^T * lambda_{j+1})

Each block is at most 5x5 (hinge joint), so each step is O(1). Total: O(K).

#### Trees: ABD-ABA (Adapted Featherstone)

For tree-structured assemblies, the algorithm is analogous to Featherstone's
Articulated Body Algorithm, but operating in the dual (constraint) space on
affine bodies.

**Pass 1: Leaf-to-Root Condensation**

Starting from leaf joints, condense each subtree's contribution into its
parent joint:

```
For each joint j from leaves to root:
    Compute local Schur complement S_j
    If j has children c_1, ..., c_k:
        Absorb children's condensed contributions into S_j
    Factorize the condensed S_j
```

**Pass 2: Root-to-Leaf Back-Substitution**

```
Solve root joint's lambda from its condensed system
For each joint j from root to leaves:
    Compute lambda_j using parent's lambda and local system
```

Complexity: O(M) for a tree with M bodies (each joint's computation is O(1)
since constraint blocks are bounded-size).

#### Closed Loops: Loop Closure Matrix

For a system with tree topology plus L loop-closing joints:

1. Solve the tree part using ABD-ABA, treating loop joints as open.
2. The loop-closing constraints introduce a dense L x L (block) matrix Delta_S
   capturing the coupling through the tree.
3. Solve the small system: Delta_S * delta_lambda_loop = residual.
4. Correct the tree solution with the loop contributions.

For small numbers of loops (typical in engineering mechanisms), the dense
solve on Delta_S is cheap. This is analogous to the Sherman-Morrison-Woodbury
formula applied to the tree factorization.

#### Irregular Networks: Bidirectional Gauss-Seidel

For general joint-body graphs with no exploitable tree structure:

```
Repeat until convergence:
    Forward pass: For j = 1 to K:
        lambda_j = S_jj^{-1} * (r_j - sum_{k != j} S_jk * lambda_k)
    Backward pass: For j = K to 1:
        lambda_j = S_jj^{-1} * (r_j - sum_{k != j} S_jk * lambda_k)
```

The bidirectional sweep improves convergence over standard GS. Each iteration
is O(K) and convergence is typically fast because the dual matrix is
well-conditioned (it inherits the conditioning of the pre-factorized body
Hessians).

---

## Pseudocode: Full M-ABD Time Step

```
FUNCTION M_ABD_Step(q_n, v_n, h, f_ext):

    // Phase 0: Prediction
    q_tilde = q_n + h * v_n + h^2 * M^{-1} * f_ext
    q = q_tilde                          // initial guess

    // Phase 1: Newton iteration
    FOR iter = 1 to max_iter:

        // 1a. Per-body co-rotational solve (parallel over all bodies)
        FOR EACH body i IN PARALLEL:
            A_i = extract_affine(q_i)         // 3x3 from 12-DOF
            R_i = polar_rotation(A_i)         // cheap polar decomposition
            f_i = compute_residual(q_i, q_tilde_i, R_i)
            f_bar_i = rotate_to_corot(R_i, f_i)
            delta_free_i = H_bar_i_inv * f_bar_i   // pre-factorized solve
            delta_free_i = rotate_from_corot(R_i, delta_free_i)
        END FOR

        // 1b. Assemble dual system
        FOR EACH joint j connecting bodies A, B:
            // Diagonal block of Schur complement
            S_jj = J_jA * H_A_inv_rotated * J_jA^T
                 + J_jB * H_B_inv_rotated * J_jB^T
            // Off-diagonal blocks for shared bodies with neighbor joints
            // RHS of dual system
            r_j  = J_jA * delta_free_A + J_jB * delta_free_B + C_j(q)
        END FOR

        // 1c. Solve dual system (dispatch by topology)
        IF topology == CHAIN:
            lambda = thomas_block_tridiag(S, r)
        ELSE IF topology == TREE:
            lambda = abd_aba(S, r, tree_structure)
        ELSE IF topology == TREE_PLUS_LOOPS:
            lambda_tree = abd_aba(S_tree, r_tree, tree_structure)
            delta_lambda = solve(Delta_S, loop_residual)
            lambda = lambda_tree + correction(delta_lambda)
        ELSE:
            lambda = bidirectional_gauss_seidel(S, r, max_gs_iter)
        END IF

        // 1d. Recover primal correction
        FOR EACH body i IN PARALLEL:
            constraint_force_i = sum over joints j on body i:
                J_ji^T * lambda_j
            cf_bar_i = rotate_to_corot(R_i, constraint_force_i)
            delta_constraint_i = rotate_from_corot(R_i,
                                    H_bar_i_inv * cf_bar_i)
            delta_q_i = delta_free_i - delta_constraint_i
        END FOR

        q = q + delta_q

        // 1e. Convergence check
        IF ||delta_q|| < tol:
            BREAK
    END FOR

    // Phase 2: Velocity update
    v_{n+1} = (q - q_n) / h

    RETURN q, v_{n+1}
```

## Pseudocode: Cheap Polar Decomposition

```
FUNCTION polar_rotation(A):
    // For near-orthogonal A, one Newton-Schulz iteration suffices.
    // Newton-Schulz iteration for the polar factor:
    //   X_{k+1} = X_k * (3I - X_k^T * X_k) / 2
    // Starting from X_0 = A:

    X = A
    X = X * (3 * I_3  -  X^T * X) / 2     // one iteration

    // Optional: second iteration for higher accuracy
    // X = X * (3 * I_3  -  X^T * X) / 2

    // Ensure det(X) > 0 (handle reflections)
    IF det(X) < 0:
        X = -X

    RETURN X
```

## Pseudocode: Block Thomas Algorithm (Chains)

```
FUNCTION thomas_block_tridiag(D[1..K], L[2..K], r[1..K]):
    // Forward sweep: eliminate sub-diagonal
    D'[1] = D[1]
    r'[1] = r[1]
    FOR j = 2 to K:
        factor = L[j] * inv(D'[j-1])
        D'[j] = D[j] - factor * L[j]^T
        r'[j] = r[j] - factor * r'[j-1]
    END FOR

    // Back substitution
    lambda[K] = inv(D'[K]) * r'[K]
    FOR j = K-1 downto 1:
        lambda[j] = inv(D'[j]) * (r'[j] - L[j+1]^T * lambda[j+1])
    END FOR

    RETURN lambda[1..K]
```

---

## Implementation Pitfalls

### 1. Polar Decomposition Cost and Accuracy

**Pitfall**: Using full SVD-based polar decomposition (library calls with
overhead) for millions of bodies per Newton iteration.

**Mitigation**: The Newton-Schulz iteration X <- X*(3I - X^T X)/2 converges
quadratically for near-orthogonal inputs. One iteration suffices when
||A - R|| < 0.1 (typical for stiff rigidity enforcement). Two iterations
cover pathological cases. Cost: two 3x3 matrix multiplies per iteration.

**Danger**: If A drifts far from orthogonal (very low stiffness, or numerical
blow-up), Newton-Schulz diverges. Always check ||X^T X - I|| after the
iteration and fall back to SVD if needed.

### 2. The R ~ A Approximation for Constraint Gradients

**Pitfall**: Replacing R with A (or normalized A) in nonlinear constraint
Jacobians. This is valid only when ||A - R|| is small.

**Mitigation**: Monitor the maximum ||A_i - R_i||_F across all bodies. If
it exceeds a threshold (e.g., 0.01), either increase the affine stiffness
k_A, decrease the time step h, or switch to exact (R-based) constraint
Jacobians for that iteration. The paper also mentions a "skew-symmetrization"
fix that corrects for the leading-order error in this approximation.

### 3. Control Tetrahedron Degeneracy

**Pitfall**: The control point (CP) tetrahedron must be non-degenerate
(positive volume). Poor choices of control points lead to ill-conditioned T
matrices, causing numerical issues in the coordinate transformation.

**Mitigation**: Choose control points that are well-separated and span the
body's extent. Avoid collinear or coplanar configurations. The condition
number of T should be monitored during preprocessing.

### 4. Choosing Lame Parameters (mu, lambda)

**Pitfall**: Setting mu and lambda too low fails to enforce rigidity (visible
deformation of "rigid" bodies). Setting them too high causes ill-conditioning
of H_bar_A: the stiffness term dominates the inertia term, and the matrix
becomes nearly singular in the inertia subspace.

**Recommended heuristic**: Choose mu and lambda such that:

    k_A / (rho / h^2)  ~  10^4 to 10^6

where k_A ~ mu is the effective stiffness, rho is density, and h is the time
step. This ensures the body deforms by at most 10^{-2} to 10^{-3} of its
size, well within the co-rotational approximation's validity.

### 5. Mass Matrix Lumping and Artificial Oscillation

**Pitfall**: Using the consistent (non-lumped) mass matrix M_A can introduce
high-frequency oscillations in the affine modes -- "jittering" artifacts where
the body vibrates at its internal elastic frequency.

**Mitigation**: Lump the mass matrix (diagonal approximation) or add Rayleigh
damping to the affine modes. Since the affine modes are artificial (the body
should be rigid), damping them aggressively is physically justified.

### 6. Constraint Drift Accumulation

**Pitfall**: Because constraints are enforced via optimization (energy
minimization + Lagrange multipliers) rather than exact geometric projection,
small residuals accumulate over time. After millions of steps, joints may
visibly separate.

**Mitigation**: Apply post-stabilization (project q onto the constraint
manifold after each step) or use Baumgarte-style correction terms in the
constraint residual.

### 7. Dual Matrix Conditioning

**Pitfall**: The Schur complement matrix S can become ill-conditioned when two
joints on the same body have nearly parallel constraint directions, when a
body has many joints (high constraint density), or when the body's Hessian
H_A is ill-conditioned.

**Mitigation**: Add a small regularization epsilon * I to the diagonal of S
(equivalent to small joint compliance). This is physically interpretable as
joint elasticity.

### 8. Topology Detection and Solver Dispatch

**Pitfall**: Using the wrong solver for the topology wastes structure. Applying
Gauss-Seidel to a chain ignores the tridiagonal structure; applying Thomas to
a tree fails entirely.

**Mitigation**: Perform topology analysis during preprocessing:
1. Build the joint-body graph.
2. Find a spanning tree.
3. Identify loop-closing edges.
4. Classify: chain / tree / tree+loops / general network.
5. Dispatch to the appropriate solver.

---

## Hyperparameter Sensitivity

### Time Step h

| h (seconds) | Effect |
|-------------|--------|
| 1e-2 | Large steps, good for quasi-static. Inertia term M/h^2 is small relative to stiffness. Co-rotational approximation very accurate. Typically 1 Newton iteration suffices. |
| 1e-3 | Typical for real-time dynamics. Good balance of accuracy and speed. 1-3 Newton iterations. |
| 1e-4 | Small steps, inertia dominates. Need higher k_A to maintain rigidity ratio. More steps per second of simulation but fewer Newton iterations per step. |
| 1e-5 | Very small, approaching explicit stability limits. Usually unnecessary with implicit integration. |

### Affine Stiffness k_A (effective: mu or lambda)

| k_A (Pa) | Body deformation | Matrix conditioning | Notes |
|-----------|------------------|---------------------|-------|
| 1e4 | ~1% | Good | Visibly "soft" bodies, fast to solve |
| 1e6 | ~0.01% | Moderate | Typical rigid simulation |
| 1e8 | ~0.0001% | Poor | Near-machine-precision rigid, but slow convergence |
| 1e10+ | Negligible | Very poor | Effectively rigid but numerically problematic |

The sweet spot depends on the application: graphics (k_A ~ 1e5-1e6),
engineering (k_A ~ 1e7-1e8), validation (use exact rigid body for reference).

### Newton Iteration Count

Typical convergence: 1-5 iterations for well-conditioned problems. The
co-rotational constant Hessian is an excellent approximation, so Newton
converges quadratically once near the solution. At h = 0.01s, often 1
iteration suffices. Pathological cases (sudden impact, joint locking) may
require 10-20 iterations. A line search or trust region can help.

### Gauss-Seidel Iterations (for irregular networks)

5-20 sweeps typically suffice for moderate networks. Convergence degrades with
graph connectivity (highly connected networks have slower spectral radius).
For very dense networks, consider a direct sparse solver on S instead.
