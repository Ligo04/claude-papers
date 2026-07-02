# ReActor: Q&A

## Basic Questions

### Q1: What problem does ReActor solve?

<details>
<summary>Click to reveal answer</summary>

ReActor solves the **embodiment gap** in motion retargeting. When you retarget motion capture data from a human actor to a simulated character with different body proportions, naive kinematic retargeting produces physically infeasible motions that cause artifacts like foot sliding, ground penetration, and self-collision when used in physics simulation.

ReActor produces adapted reference motions that are physically plausible for the target character while maintaining the style and characteristics of the original motion.
</details>

### Q2: What is bilevel optimization?

<details>
<summary>Click to reveal answer</summary>

Bilevel optimization is a nested optimization problem with two levels:
- **Upper level**: Optimizes retargeting parameters θ
- **Lower level**: For each θ, trains an optimal policy π*(θ)

The upper level's objective depends on the lower level's solution. In ReActor:
- Upper level adapts the reference motion
- Lower level trains a policy to track that reference
- They co-evolve: better references help policy training, and policy performance informs reference adaptation

This is different from sequential optimization where you'd retarget first, then train the policy independently.
</details>

### Q3: Why use sparse parameterization?

<details>
<summary>Click to reveal answer</summary>

Sparse parameterization means using constant per-joint offsets instead of per-frame adjustments.

**Benefits**:
1. **Generalization**: Same offsets work across multiple motion clips
2. **Efficiency**: 100s of parameters vs 1000s for per-frame
3. **Interpretability**: "Bend knees 5° more" is physically meaningful
4. **Regularization**: Prevents overfitting to specific sequences

**Example**: Instead of learning 34,200 parameters (114 per frame × 300 frames), ReActor learns just 114 parameters that apply to all frames.
</details>

### Q4: What is the simplified gradient estimate?

<details>
<summary>Click to reveal answer</summary>

The simplified gradient estimate avoids computing expensive inverse Hessians.

**Standard bilevel gradient** requires:
```
∂π*/∂θ = -H^(-1) · (∂²L/∂π∂θ)
```
This is computationally prohibitive for neural networks with millions of parameters.

**ReActor's approximation**:
```
∇_θ L ≈ ∇_θ L_motion + λ · E[∇_θ log π(a|s) · A(s,a)]
```

This uses policy gradients weighted by advantages instead of second-order derivatives, making it tractable while maintaining effectiveness.
</details>

### Q5: What are the main results?

<details>
<summary>Click to reveal answer</summary>

**Quantitative improvements** (vs baselines):
- **Foot sliding**: 89% reduction (humanoid), 85% (quadruped)
- **Ground penetration**: 94% reduction
- **Self-penetration**: 97% reduction
- **Tracking error**: 23% lower than fixed retargeting
- **Success rate**: 96% vs 78% for baseline

**Generalization**:
- Single retargeting works across multiple motion clips
- 3.2× faster convergence than training from scratch per motion
- Maintains 92% of original motion fidelity
</details>

## Intermediate Questions

### Q6: How does physics-in-the-loop training help?

<details>
<summary>Click to reveal answer</summary>

Running optimization inside the physics simulator provides several advantages:

1. **Contact dynamics**: Foot-ground interaction emerges naturally, eliminating foot sliding
2. **Self-collision avoidance**: Physics engine enforces collision constraints automatically
3. **Dynamic feasibility**: Motions respect actuator limits and balance constraints
4. **Realistic forces**: Ground reaction forces and joint torques are physically accurate

**vs Kinematic post-processing**: Can't handle contact, still has penetration issues

**Implementation**: 4096 parallel Isaac Gym environments for efficient sampling
</details>

### Q7: Why does the gradient approximation work?

<details>
<summary>Click to reveal answer</summary>

The policy gradient term `∇_θ log π(a|s) · A(s,a)` captures how retargeting changes affect policy performance:

1. **∇_θ log π(a|s)**: Shows how θ changes affect action probabilities
2. **A(s,a)**: Weights by how good the action is (advantage)
3. **Expectation**: Averages over trajectories

**Intuition**: If changing θ makes the policy take better actions (positive advantage), the gradient pushes θ in that direction. This approximates the effect of θ on policy performance without computing second derivatives.

**Empirical validation**: Paper shows this works as well as exact gradients in practice.
</details>

### Q8: What is the role of hyperparameter λ?

<details>
<summary>Click to reveal answer</summary>

λ balances two objectives in the gradient:
```
∇_θ L ≈ ∇_θ L_motion + λ · policy_gradient
          ↑                    ↑
    motion fidelity      tracking performance
```

**High λ** (e.g., 0.5):
- Prioritizes tracking performance
- More adaptation to physics constraints
- May lose some motion style

**Low λ** (e.g., 0.05):
- Prioritizes motion fidelity
- Less adaptation
- May have tracking difficulties

**Paper uses**: λ = 0.1 as default, but may need tuning per motion type or character pair.
</details>

### Q9: How does adaptive sampling work?

<details>
<summary>Click to reveal answer</summary>

Adaptive sampling focuses training on challenging motion segments:

**Algorithm**:
1. Track error for each motion segment
2. Sample segments with probability proportional to error:
   ```
   p(segment_i) ∝ exp(β · error_i)
   ```
3. Spend more time on hard segments (turns, transitions)

**Benefits**:
- Faster convergence on difficult parts
- More uniform final performance across segments
- Better handling of motion complexity variations

**Example**: A walking motion might have easy straight-line segments and hard turning segments. Adaptive sampling spends more episodes on the turns.
</details>

### Q10: What is the reward function design?

<details>
<summary>Click to reveal answer</summary>

The reward function (Table 1 in paper) combines multiple terms:

1. **Pose tracking** (w=0.5): `exp(-2·||q - q_ref||²)`
2. **Velocity matching** (w=0.1): `exp(-0.1·||q̇ - q̇_ref||²)`
3. **End effector tracking** (w=0.3): `exp(-10·||p_ee - p_ref||²)`
4. **Contact consistency** (w=0.05): `-|c - c_ref|`
5. **Action smoothness** (w=0.03): `-||a_t - a_{t-1}||²`
6. **Torque regularization** (w=0.02): `-||τ||²`

**Design principles**:
- Exponential terms for tracking (smooth gradients)
- Higher weight on end effectors (visible to humans)
- Regularization prevents unnatural motions
- Contact term ensures proper foot-ground timing
</details>

## Advanced Questions

### Q11: How would you extend ReActor to handle time-varying offsets?

<details>
<summary>Click to reveal answer</summary>

**Current limitation**: Constant offsets can't adapt to motion-specific requirements.

**Potential approaches**:

1. **Phase-dependent offsets**:
   ```
   Δq(t) = Δq_base + Δq_phase(φ(t))
   where φ(t) = phase in gait cycle
   ```
   - Learn offsets per phase (stance, swing, transition)
   - Interpolate smoothly between phases

2. **Learned offset schedules**:
   ```
   Δq(t) = MLP(motion_features, t)
   ```
   - Neural network predicts offsets from motion context
   - Regularize for smoothness

3. **Motion primitive decomposition**:
   - Segment motion into primitives (walk, turn, stop)
   - Learn per-primitive offsets
   - Blend at boundaries

**Trade-offs**:
- More expressiveness vs more parameters
- Risk of overfitting vs better adaptation
- Computational cost vs performance gain
</details>

### Q12: Why not use inverse dynamics instead of RL?

<details>
<summary>Click to reveal answer</summary>

**Inverse dynamics** computes required torques from desired motion:
```
τ = M(q)q̈ + C(q,q̇) + G(q)
```

**Limitations for this problem**:

1. **Contact forces**: Inverse dynamics struggles with underactuated contact (foot-ground)
2. **Actuator limits**: May compute infeasible torques
3. **Robustness**: No feedback control, sensitive to perturbations
4. **Tracking errors**: Accumulate over time without correction

**RL advantages**:
- Learns feedback policy (robust to disturbances)
- Respects actuator constraints naturally
- Handles contact through simulation
- Can trade off tracking vs stability

**Hybrid approach**: Could use inverse dynamics to initialize RL policy (warm start).
</details>

### Q13: How does ReActor compare to end-to-end learned retargeting?

<details>
<summary>Click to reveal answer</summary>

**End-to-end approach**: Train neural network to directly map source motion → target motion

**Advantages of end-to-end**:
- No hand-designed parameterization
- Potentially more expressive
- Single forward pass (fast inference)

**Advantages of ReActor**:
- **Data efficiency**: Doesn't need large motion datasets
- **Interpretability**: Offsets have physical meaning
- **Generalization**: Works for unseen character pairs
- **Physics guarantees**: Optimization in simulator ensures feasibility

**When to use each**:
- End-to-end: Large motion library, many character pairs, need speed
- ReActor: Limited data, new character, need quality guarantees

**Future direction**: Combine both - use learned retargeting as initialization for ReActor.
</details>

### Q14: What are the failure modes of the simplified gradient estimate?

<details>
<summary>Click to reveal answer</summary>

The approximation `∇_θ L ≈ ∇_θ L_motion + λ·E[∇_θ log π·A]` can fail when:

1. **High curvature in policy space**:
   - Assumption: Policy changes smoothly with θ
   - Failure: Large θ changes cause policy collapse
   - Mitigation: Small learning rate, gradient clipping

2. **Poor advantage estimates**:
   - Assumption: A(s,a) accurately reflects action quality
   - Failure: High variance in advantage estimates
   - Mitigation: More samples, better value function

3. **Misaligned objectives**:
   - Assumption: Motion loss and tracking loss are compatible
   - Failure: Optimal motion is untrackable
   - Mitigation: Tune λ, add feasibility constraints

4. **Local minima**:
   - Approximation may have different local minima than exact gradient
   - Mitigation: Multiple random seeds, curriculum learning

**Empirical observation**: Paper shows approximation works well in practice, suggesting these failure modes are rare for this problem.
</details>

### Q15: How would you adapt ReActor for real robot deployment?

<details>
<summary>Click to reveal answer</summary>

**Sim-to-real challenges**:

1. **Dynamics mismatch**:
   - Simulation doesn't perfectly match real robot
   - Solution: Domain randomization during training
   - Randomize: mass, friction, joint damping, latency

2. **Sensor noise**:
   - Real sensors have noise, delays, failures
   - Solution: Train with noisy observations
   - Add Gaussian noise to joint positions/velocities

3. **Actuator limitations**:
   - Real motors have bandwidth limits, backlash
   - Solution: Model actuator dynamics in simulation
   - Use realistic PD gains, torque limits

4. **Safety constraints**:
   - Can't let robot fall/break during learning
   - Solution: Train in sim, deploy with safety controller
   - Add emergency stop, joint limit buffers

5. **Adaptation**:
   - Real robot may differ from simulation
   - Solution: Online adaptation with small learning rate
   - Continue updating θ on real robot (carefully!)

**Deployment pipeline**:
1. Train in simulation with domain randomization
2. Validate on multiple random seeds
3. Test in simulation with real robot parameters
4. Deploy with safety controller active
5. Monitor and adapt online (optional)

**Key insight**: ReActor's physics-in-loop training is already a good foundation for sim-to-real transfer.
</details>

---

## Self-Test

Try answering these questions without looking at the answers first:

1. Draw the bilevel optimization diagram showing upper/lower levels
2. Derive the simplified gradient estimate from first principles
3. Explain why constant offsets generalize across motions
4. Design a reward function for a quadruped trotting motion
5. Propose an extension to handle object manipulation

If you can answer these, you have a solid understanding of ReActor!
