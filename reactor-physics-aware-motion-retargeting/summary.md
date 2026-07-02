# ReActor: Summary

## Background

Motion retargeting adapts motion capture data from one character to another with different body proportions. Traditional kinematic retargeting fails for physics-based control due to the **embodiment gap**: retargeted motions may violate physical constraints, causing foot sliding, penetration, and infeasible dynamics.

## Problem Statement

**Goal**: Given source mocap and a target character, produce:
1. Physically plausible adapted reference motion
2. Robust RL tracking policy

**Challenge**: Reference quality affects policy learning, and policy performance depends on reference feasibility (interdependent).

## Main Contributions

### 1. Bilevel Optimization Framework

Jointly optimizes reference motion adaptation (upper level) and policy training (lower level):
- Upper: min_θ L_motion(θ, π*(θ))
- Lower: π*(θ) = argmin_π L_tracking(π, θ)

### 2. Simplified Gradient Estimate

Avoids expensive inverse Hessian computation:
```
∇_θ L ≈ ∇_θ L_motion + λ · E[∇_θ log π(a|s) · A(s,a)]
```

Uses policy gradient with advantage values instead of second-order derivatives.

### 3. Sparse Retargeting Parameterization

- Nominal retargeting (fixed) + learnable offsets (optimized)
- Constant per-joint offsets (not per-frame)
- Reduces parameters 10×, prevents overfitting, generalizes across motions

### 4. Physics-in-the-Loop Training

Optimization runs inside physics simulator:
- Handles contact dynamics automatically
- Prevents self-collision
- Respects joint/actuator limits
- 4096 parallel environments (Isaac Gym)

## Key Results

**Humanoid locomotion** (vs baselines):
- Foot sliding: 89% reduction
- Ground penetration: 94% reduction  
- Self-penetration: 97% reduction
- Motion quality: 92% preserved

**Quadruped locomotion**:
- Foot sliding: 85% reduction
- Tracking error: 23% lower
- Success rate: 96% vs 78%

**Generalization**:
- Single parameterization works across multiple clips
- 3.2× faster convergence than per-motion training

## Technical Highlights

**Reward function** (Table 1): pose tracking, velocity matching, end effector tracking, contact consistency, regularization

**Training**: PPO, 4096 envs, ~50M steps, ~6 hours on RTX 3090

## Limitations

1. Constant offsets limit expressiveness
2. External forces need penalty term
3. Extreme embodiment gaps may need more parameters
4. Hyperparameter λ may need per-motion tuning
