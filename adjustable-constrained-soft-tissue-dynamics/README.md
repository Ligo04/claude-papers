# Adjustable Constrained Soft-Tissue Dynamics

> Wang, Zheng, Barbic - University of Southern California - Pacific Graphics 2020
> Computer Graphics Forum, Vol. 39, No. 7

## What this paper is about

In character animation, soft-tissue jiggle is usually produced by attaching a constraint-based FEM simulation to a geometric proxy (skeleton, blendshape rig, etc.). The authors observe that under **correct physics**, the natural **vibration frequency** and the **inertial deformation amplitude** of such a system are mathematically **coupled** - you cannot make a character jiggle *less* without also making it jiggle *faster* (and vice versa) by tweaking material parameters alone. The paper proposes a minimal, drop-in modification of the equations of motion that **decouples** these two quantities, and additionally provides exact, independent control over **gravity sag** and **damping decay time**.

The trick: re-express the equations of motion in a **per-vertex non-inertial reference frame** (the locally rotating frame attached to the deformation gradient at that vertex), **scale the resulting fictitious forces** (linear-acceleration, angular-acceleration, centrifugal, Coriolis) by a single user knob `epsilon`, and transform back to the world frame. The result is a *non-physical but artist-controllable* version of standard constrained FEM, with only an additive change to the RHS.

## Difficulty

**Advanced.** Requires comfort with:

- Constraint-based FEM (mass / stiffness / damping matrices, Rayleigh damping)
- Linear modal analysis (generalized eigenvalue problem, natural frequency)
- Polar decomposition of the deformation gradient and its time derivatives
- Non-inertial reference frames and d'Alembert / fictitious forces
- Implicit time integration (backward Euler, Bathe)

## How to navigate the materials

| File | Read order | Purpose |
|---|---|---|
| `summary.md` | 1 | Paper at a glance - context, contributions, results |
| `mental-model.md` | 2 | Where this paper sits in the research landscape |
| `insights.md` | 3 | The single conceptual shift and *why* it works |
| `method.md` | 4 | Equation-by-equation walkthrough of the algorithm |
| `qa.md` | 5 | 15 self-test questions (basic - advanced) |
| `code/` | 6 | Runnable 1-D and N-D demos of the core idea |
| `index.html` | any | Interactive parameter explorer |
| `paper.pdf` | reference | Original paper |

## Key takeaways

1. **Coupling theorem (informal):** for a 1-DOF constrained spring-mass system kicked with impulse `v0`, the post-impact amplitude is `v0 / omega_d`. Tuning `m`, `k` to change `omega_d` automatically changes the amplitude - they cannot be independently controlled by material parameters.
2. **Fictitious-force scaling:** in a per-vertex co-rotating frame, the inertial coupling appears explicitly as `M (p_dotdot + Rdotdot Rt (u - p) + 2 Rdot Rdot_t (u - p) + 2 Rdot Rt (udot - pdot))`. Multiplying the *whole bracket* by `(1 - epsilon)` and adding to the RHS gives an `epsilon` knob that scales inertial amplitude independently.
3. **Closed-form parameter recipe:** given target ratios of frequency `nu_bar/nu`, sag `zeta_bar/zeta`, damping half-time `tD_bar/tD`, amplitude `A_bar/A`, the paper derives explicit formulas for the multipliers `chi, mu, c_k, epsilon`.
4. **Cost:** ~25% overhead vs. baseline FEM - one extra static solve per timestep plus polar-decomposition derivatives.

## Estimated study time

- Skim `summary.md` + Figures 1, 2: **20 min**
- Walk through `method.md` derivation: **60-90 min**
- Run the 1-D demo + nudge `epsilon`, `chi`: **30 min**
- Full mastery (re-derive Eq. 17 from scratch): **half a day**

## Folder structure

```
adjustable-constrained-soft-tissue-dynamics/
- README.md                   <- you are here
- paper.pdf                   <- original PDF
- meta.json                   <- machine-readable metadata
- summary.md
- insights.md
- mental-model.md
- method.md
- qa.md
- reflection.md
- index.html                  <- interactive explorer
- images/                     <- extracted figures
- code/
  - README.md
  - one_d_demo.py             <- 1-DOF coupling + decoupling
  - inertial_scaling_3d.py    <- N-DOF co-rotational demo
```
