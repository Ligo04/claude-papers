# M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics -- Technical Summary

**Paper**: [arXiv:2603.08079](https://arxiv.org/abs/2603.08079)
**Authors**: Zhiyong He, Dewen Guo, Minghao Guo, Yili Zhao, Wojciech Matusik, Hao Su, Chenfanfu Jiang, Peter Yichen Chen, Yin Yang

---

## 1. Background Context

### 1.1 Rigid Body Dynamics (RBD) and Its Limitations

Classical rigid body dynamics represents each body with 6 degrees of freedom: 3 for translation, 3 for rotation (via quaternions, Euler angles, rotation matrices, etc.). While well-studied, RBD has several persistent challenges:

- **Rotation parameterization**: Quaternions require normalization constraints; Euler angles suffer from gimbal lock; rotation matrices carry redundant DOFs. All approaches introduce nonlinearity into the equations of motion that complicates implicit integration.
- **Nonlinear system matrices**: Even for a single rigid body, implicit integration produces system matrices that depend on the current state. These matrices cannot be pre-factorized and must be reassembled and refactored at every iteration or time step.
- **Constraint drift**: Common approaches to joint constraints (penalty methods, Baumgarte stabilization) suffer from drift over time or require careful parameter tuning.
- **Scalability**: Explicit integrators are conditionally stable and struggle with stiff contacts or large time steps. Implicit integrators are robust but expensive per step due to nonlinear solves.

### 1.2 Affine Body Dynamics (ABD) Basics

ABD, introduced by Lan et al. (2022), replaces the rigid body representation with an affine one. Each body's configuration is described by:

- **A**: a 3x3 affine transformation matrix (9 DOFs)
- **t**: a translation vector (3 DOFs)

This gives 12 DOFs per body instead of 6. The kinematic mapping is linear: any material point x_bar on the body maps to world position as x = A * x_bar + t. The body is not assumed rigid a priori; instead, a high-stiffness elastic energy penalizes deviation from rigidity. As stiffness increases, the affine body converges to rigid behavior.

Key advantages of ABD over RBD:
- The configuration space is a linear vector space (no manifold constraints on rotations).
- Mass and stiffness matrices have simpler structure.
- No singularities or parameterization issues.
- Constraint Jacobians for joint constraints become constant.

The trade-off is the doubled DOF count (12 vs. 6), but M-ABD shows this is more than offset by the ability to pre-factorize constant system matrices.

---

## 2. Problem Statement

How can we efficiently simulate large-scale articulated multibody systems (10^5 to 10^6 bodies connected by mechanical joints) with the following requirements:

1. **Efficiency**: Interactive or near-interactive rates for systems with hundreds of thousands to over a million bodies.
2. **Robustness**: Stable behavior at large time steps without requiring very small step sizes for stability.
3. **Accuracy**: Exact constraint enforcement (no drift in joint constraints over time).
4. **Generality**: Support for diverse joint types (ball/spherical, hinge/revolute, universal, prismatic) and diverse topologies (chains, trees, closed loops, general graphs).

Existing methods fail to achieve all four simultaneously:
- Explicit integration (e.g., standard RBD engines) requires tiny time steps and becomes unstable at scale.
- Penalty-based constraint methods (common in game engines) suffer from constraint drift.
- Implicit RBD solvers require expensive per-step matrix factorizations due to nonlinear rotation dynamics.

---

## 3. Main Contributions

### 3.1 Co-Rotational ABD Formulation

**The core theoretical contribution.** Standard ABD uses a hyperelastic energy to penalize non-rigid deformation. When stiffness is high, this energy landscape is extremely steep, and the body stays near-rigid. However, the energy's Hessian still depends on the current configuration, preventing pre-factorization.

The co-rotational formulation resolves this by decomposing the affine matrix via polar decomposition:

```
A = R * S
```

where **R** is a rotation (the geometric nonlinearity) and **S** is a symmetric stretch (the constitutive part). The key insight is:

- **High stiffness suppresses S toward identity.** When S is near-identity, the constitutive response (relating S to stress) is effectively linear.
- **Geometric nonlinearity is isolated in R.** The rotation R is extracted via polar decomposition at each step and used to rotate the reference frame.

This yields a system matrix H_bar = (1/h^2) * M + K_bar that is **constant**---it does not depend on the current configuration. This matrix can be **pre-factorized once** and reused across all time steps and Newton iterations. The per-step cost reduces to:

1. Polar decomposition to extract R (a small, well-optimized operation per body).
2. A back-substitution using the pre-factored system matrix.
3. Updating the affine state.

This is what enables a single Newton iteration per time step to suffice in practice: the co-rotational linearization is accurate enough that one iteration captures the physics.

### 3.2 Joint Constraints in Affine Coordinates

The paper formulates standard mechanical joints directly in the 12-DOF affine coordinate space. For each joint type, the constraint equations are:

- **Ball joint (spherical)**: Two bodies share a common point. 3 scalar constraints on the translations and affine matrices of the connected bodies.
- **Hinge joint (revolute)**: Two bodies share a common axis of rotation. 5 scalar constraints (3 for the shared point + 2 for the shared axis direction).
- **Universal joint**: Two bodies share a point, and two designated axes on each body remain perpendicular. 4 scalar constraints.
- **Prismatic joint**: Two bodies share a sliding axis with no relative rotation about other axes. 5 scalar constraints.

A critical property: in affine coordinates, these constraint Jacobians are **constant** (they do not depend on the current configuration). This is a direct benefit of using affine rather than rotational DOFs and is essential for the pre-factorization strategy.

### 3.3 Dual-Space KKT Formulation

With constant system matrices and constant constraint Jacobians, the constrained equations of motion take the KKT form:

```
| H   J^T | | dq     |   | f |
|         | |        | = |   |
| J   0   | | lambda |   | g |
```

where:
- **H** is the constant system matrix (mass + stiffness contributions)
- **J** is the constant constraint Jacobian
- **dq** is the configuration update (12 DOFs per body)
- **lambda** is the vector of Lagrange multipliers (constraint forces)
- **f** is the external force / residual
- **g** is the constraint residual

Using a **Schur complement**, the system is condensed onto the dual (constraint) space:

```
(J H^{-1} J^T) lambda = g - J H^{-1} f
```

Since H is pre-factorized (constant), H^{-1} applications are cheap back-substitutions. The dual system has dimension equal to the total number of scalar constraints (3--5 per joint), which is much smaller than the primal DOF count (12 per body). Once lambda is found, dq is recovered via:

```
dq = H^{-1} (f - J^T lambda)
```

This dual-space approach enforces constraints **exactly** (to numerical precision), with no drift and no penalty parameter tuning.

### 3.4 Topology-Specific Solvers

The dual-space KKT system has structure determined by the connectivity (topology) of the articulated system. M-ABD provides specialized solvers for each case:

#### Chains (Block-Tridiagonal Solver)
For serial chains (body_1 -- joint_1 -- body_2 -- joint_2 -- ... -- body_n), the Schur complement matrix J H^{-1} J^T is **block-tridiagonal**. This structure arises because each joint only connects two adjacent bodies. Block-tridiagonal systems can be solved in O(n) time via block LDL factorization (a block Thomas algorithm), where n is the number of bodies. This factorization is also constant and can be pre-computed.

#### Trees (ABD-ABA: Adapted Featherstone Algorithm)
For tree-structured articulations (branching without loops), M-ABD adapts the celebrated Featherstone Articulated Body Algorithm (ABA) to the affine body setting. The original ABA operates on 6-DOF rigid bodies with joint-space coordinates; ABD-ABA operates on 12-DOF affine bodies with the dual-space (constraint force) formulation. The recursive structure gives O(n) complexity for trees with n bodies.

#### Closed Loops and Irregular Networks (Bidirectional Gauss-Seidel)
When the topology contains loops or is an irregular graph, exact O(n) recursive methods are not directly applicable. M-ABD uses a bidirectional Gauss-Seidel iteration over the dual variables (Lagrange multipliers). Each Gauss-Seidel sweep updates constraint forces along the graph, and the bidirectional (forward-backward) pattern improves convergence. The per-iteration cost is O(n), and convergence is typically fast due to the well-conditioned nature of the dual system.

---

## 4. Key Results and Quantitative Metrics

### 4.1 Scale

| Metric | Value |
|--------|-------|
| Maximum bodies simulated | **1,076,748** |
| Time step | h = 0.01 s |
| Newton iterations per step | **1** |
| Wall time per step (single CPU thread) | **904 ms** |

This is the largest articulated multibody simulation reported. The single-Newton-iteration convergence is a direct consequence of the co-rotational formulation's accuracy.

### 4.2 Single-Body Performance

| Method | Time |
|--------|------|
| Co-rotated ABD (implicit) | **27 ms** |
| Explicit RBD | **32 ms** |
| Vanilla implicit ABD | **161 ms** |

A fully implicit ABD solver is faster than an explicit RBD solver for a single body. This counterintuitive result arises because ABD's constant system matrices allow pre-factorization, making each implicit step extremely cheap (just a back-substitution), while explicit RBD must still evaluate nonlinear rotation dynamics. The co-rotational formulation is approximately 6x faster than vanilla implicit ABD.

### 4.3 Benchmark Scenarios

| Scenario | Bodies | Time/Step | CPU Threads |
|----------|--------|-----------|-------------|
| Chain (128 hinge-linked cubes) | 128 | 0.7 ms | 1 |
| Kinematic tree (200K links) | 200,000 | 263 ms | 1 |
| Pulley system | 1,076,748 | 904 ms | 1 |
| Loop chain (4K bodies) | 4,096 | 14 ms | 1 |
| Irregular network (8K bodies) | 8,192 | 113 ms | 1 |

### 4.4 Constraint Accuracy

Joint constraints are satisfied to numerical precision at every time step. There is **no drift** over long simulations, in contrast to penalty-based or stabilization-based methods where constraint error accumulates or oscillates. Energy drift is less than 0.01% over long simulation horizons, matching ground-truth RBD solutions.

### 4.5 Robustness at Large Time Steps

The fully implicit formulation with co-rotational treatment remains stable at time steps that would cause explicit methods to diverge. Notably, for the chain benchmark with 128 hinge-linked cubes, M-ABD converges in 1 Newton iteration at h=0.01s, while MuJoCo's implicit Euler diverges at the same timestep.

### 4.6 Comparisons

Compared with MuJoCo, Brax, and Drake, M-ABD is reported to be 10--100x faster while maintaining better constraint satisfaction and stability.

---

## 5. Significance and Context

M-ABD bridges two communities:
- **Graphics / computational mechanics**, where ABD originated as a way to simulate near-rigid deformable bodies.
- **Robotics / multibody dynamics**, where Featherstone-style algorithms and joint constraint formulations are standard.

By combining ABD's linear configuration space with robotics-style articulated body algorithms, M-ABD achieves a combination of scale, robustness, and constraint accuracy that neither community's existing tools provide alone. The constant-matrix property---enabled by the co-rotational formulation in the affine setting---is the unifying insight that makes this synthesis possible.

---

## 6. Summary of Contributions

| Contribution | What It Enables |
|---|---|
| Co-rotational ABD | Constant system matrices; pre-factorization; single Newton iteration per step |
| Joint constraints in affine coords | Constant constraint Jacobians; exact enforcement via KKT |
| Dual-space Schur complement | Compact solve in constraint space; decoupled from primal DOF count |
| Block-tridiagonal solver (chains) | O(n) exact solve for serial chains |
| ABD-ABA (trees) | O(n) exact solve for branching structures |
| Bidirectional Gauss-Seidel (loops/networks) | Efficient iterative solve for general topologies |
