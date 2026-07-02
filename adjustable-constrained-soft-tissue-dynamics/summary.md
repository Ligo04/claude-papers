# Summary

## Background context

Modern character pipelines are layered:

1. **Modeling** - artist sculpts triangle mesh.
2. **Rigging** - skeleton, IK handles, blendshape sliders, skinning weights.
3. **Geometric animation** - keyframe joints/sliders -> static per-frame mesh shapes (no time history, no physics).
4. **Soft-tissue dynamics** - attach FEM tetrahedral simulation to geometrically-animated proxy via *constraints* (`C u = S x(t)`) so bellies, cheeks, ears, leaves, jellyfish tentacles jiggle realistically in response to proxy motion.

This last layer is focus. Widely used in research and industry (Pixar, Ziva, Disney; refs [CBC*05, HMT*12, HTC*13, XB16, LXB17, KDGI19, Tis13]).

## Problem statement

Animators direct *visual* dynamics of soft tissue using four primary knobs:

- vibration **frequency** `nu`
- gravity **sag** `zeta`
- inertial **amplitude** `A`
- damping **half-time** `t_D`

Conventional practice: tweak material parameters (mass `M`, stiffness `K`, gravity `g`, Rayleigh coefficients `d_m, d_k`).

Authors prove this is fundamentally insufficient. For 1-D mass-on-a-spring kicked with impulse to velocity `v0`:

```
z(t) = (v0 / omega_d) * exp(-c/(2m) * t) * sin(omega_d * t)
```

Amplitude is `v0 / omega_d`. Tuning `m, k, c` to change `omega_d` *automatically* rescales amplitude by `1 / omega_d`. **Frequency and inertial amplitude coupled** under correct physics. By linear superposition, generalizes to 3-D constrained FEM with arbitrary `x(t)`.

Consequence: animator cannot say "make jiggling slower but keep current amplitude" by adjusting Young's modulus alone.

## Main contributions

1. **Identification of coupling pathology** in standard constraint-based simulation - apparently not previously documented.
2. **Decoupling method** based on transforming EoM to per-vertex non-inertial frame, scaling fictitious forces by `epsilon`, transforming back. Modifies Eq.1 only by additive RHS terms. Compatible with any standard integrator and contact solver.
3. **Closed-form parameter recipe** - given desired ratios `nu_bar/nu, zeta_bar/zeta, A_bar/A, tD_bar/tD`, derive explicit `chi, mu, c_k, epsilon` so linear physics matches all four targets exactly. Stiffness-only Rayleigh damping (`d_m = 0`).
4. **Two-stage amplitude refinement** for nonlinear/contact cases - solve linearized formula, simulate, measure actual `A_actual`, rescale `epsilon' = epsilon * A_bar / A_actual`. Bisection if needed.
5. **Demonstrations** on four examples: linear & StVK sunflower, neo-Hookean jellyfish, StVK sumo, stable neo-Hookean jellyphant (Ziva data).
6. **Comparison vs. [LXB17]** - their method only modifies inertial RHS, so standard penalty contact still works correctly; [LXB17]'s rotation-aware blending creates spurious gaps at contact sites (Fig. 7).

## Key results / metrics

Performance table (Table 1 in paper):

| Model | frames | substeps | static [min] | dynamic [min] | adjusted [min] |
|---|---:|---:|---:|---:|---:|
| Sunflower (Linear FEM) | 300 | 1 | 0.17 | 1.3 | 0.0053 |
| Sunflower (StVK) | 300 | 1 | 0.2 | 4.0 | 0.0052 |
| Jellyfish (StVK) | 313 | 50 | - | 6.5 | 0.011 |
| Sumo (StVK) | 400 | 4 | 6.3 | 25.3 | 0.026 |
| Jellyphant (stable neo-Hookean) | 1200 | 50 | 107 | 401 | 0.36 |

- "Adjusted" column = cost of evaluating new RHS terms (`p, R, Rdot, Rdotdot` and products). 1000x smaller than dynamic simulation - negligible.
- **Static solve overhead** = ~25% of dynamic time when present. Dominant cost of method.
- **Memory overhead** = negligible (static solve reuses `f_int` / tangent stiffness datastructures).

## Quantitative validation

Figures 8, 10, 11 (linear sunflower):

- **Fig. 8**: hold any two of {frequency, damping, amplitude} fixed, sweep third -> RMS displacement plots verify orthogonality.
- **Fig. 10**: sag scaled 1x / 2x / 4x / 8x at each of four frequencies -> ratio plot shows sag actually multiplies as requested across frequency range.
- **Fig. 11**: amplitude held at 1x while frequency varies -> bottom-row curves ~flat (small ripples = damping causing mode-phase drift, not coupling failure).
- **Fig. 9** (nonlinear jellyfish "user story"): (a) only `epsilon` changed - amplitude scales, frequency unchanged. (b) only frequency changed - amplitude inflates as predicted. (c) `epsilon` recomputed - amplitude returned to original level while frequency stays halved.

## Limitations explicitly acknowledged

- Linear-physics formulas approximate under large deformations. Nonlinear case requires iterative two-stage `epsilon` refinement.
- Damping changes natural frequency too; small effect ignored.
- Only under-damped vibration considered ("jiggling" regime).
- One-way coupling only (proxy not influenced by tissue inertia/contact). Two-way slower and harder to control.
- Method is *non-physical* by construction. Animators want non-physical control. For correct physics, do not use this.

## Conflict of interest disclosure (verbatim from paper)

Jellyphant mesh and proxy motion provided by Ziva Dynamics. Barbic is shareholder, CTO, board member. Ziva not involved in research.
