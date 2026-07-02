# ReActor: Mental Model and Research Positioning

## Problem Type

**Domain**: Character animation + Reinforcement learning + Physics simulation

**Problem class**: Motion retargeting with embodiment mismatch

**Core challenge**: Bridging the gap between kinematic motion capture data and physics-based character control when body proportions differ.

## Research Landscape

### The Motion Retargeting Problem

**Goal**: Transfer motion from source character to target character with different morphology.

**Approaches**:
1. **Kinematic retargeting**: Geometric mapping (fast, but ignores physics)
2. **Physics-based retargeting**: Optimization in simulation (slow, high quality)
3. **Learning-based retargeting**: Neural networks (data-hungry, generalizes well)

**ReActor's position**: Hybrid approach combining optimization and learning.

## Related Work and Positioning

### DeepMimic (2018)

**Approach**: RL for physics-based character control from motion capture.

**Key idea**: Train policy to imitate reference motion using reward shaping.

**Limitation**: Assumes reference motion is already physically plausible for target character.

**ReActor improvement**: Adapts reference motion jointly with policy training.

### PHC (Perpetual Humanoid Control, 2023)

**Approach**: Large-scale RL for humanoid control with motion tracking.

**Key idea**: Train on massive motion datasets for generalization.

**Limitation**: Still requires pre-retargeted motions; doesn't handle embodiment gap.

**ReActor improvement**: Learns retargeting parameters, not just tracking policy.

### DOC (Dynamics-Aware Optimal Control, 2022)

**Approach**: Trajectory optimization for motion retargeting.

**Key idea**: Optimize reference trajectory to be dynamically feasible.

**Limitation**: Computationally expensive; doesn't generalize across motions.

**ReActor improvement**: Sparse parameterization generalizes; RL is more efficient.

### GMR (Generative Motion Retargeting, 2024)

**Approach**: Generative model for motion retargeting.

**Key idea**: Learn distribution of valid retargeted motions from data.

**Limitation**: Requires large paired motion datasets; no physics guarantees.

**ReActor improvement**: Physics-in-loop ensures feasibility; works with single motions.

### OmniRetarget (2025)

**Approach**: Universal retargeting network trained on diverse character pairs.

**Key idea**: Single model handles arbitrary morphology differences.

**Limitation**: Still kinematic; produces foot sliding and penetration in simulation.

**ReActor improvement**: Physics-based optimization eliminates artifacts.

## When to Use ReActor vs Alternatives

### Use ReActor when:

1. **Physics simulation is required**: Need motions that work in physics engine
2. **Quality over speed**: Can afford 6 hours training per character-motion pair
3. **Limited data**: Don't have large motion datasets
4. **Novel characters**: Target character not in training set
5. **Artifact-free output**: Zero tolerance for foot sliding, penetration

### Use alternatives when:

**DeepMimic** if:
- Reference motion already physically plausible
- Just need tracking policy, not retargeting

**PHC** if:
- Have large motion library
- Target character is standard humanoid
- Need fast inference across many motions

**GMR/OmniRetarget** if:
- Speed is critical (real-time retargeting)
- Physics simulation not required
- Have training data available
- Can tolerate minor artifacts

**DOC** if:
- Need trajectory optimization (not just tracking)
- Single motion, willing to wait hours
- Want interpretable optimization

## Conceptual Frameworks

### Framework 1: Co-Evolution

**Analogy**: Teacher and student learning together.

- **Teacher (upper level)**: Adjusts lesson difficulty based on student performance
- **Student (lower level)**: Learns from lessons, performance informs teaching

**In ReActor**:
- Reference motion adapts based on policy tracking ability
- Policy training informs what references are trackable

### Framework 2: Representation Learning

**Perspective**: Learning the right representation for the task.

**Traditional**: Hand-designed retargeting (fixed representation)

**ReActor**: Learned retargeting parameters (adaptive representation)

**Key insight**: Representation and task should be learned jointly, not sequentially.

### Framework 3: Physics as Constraint

**View**: Physics simulation as a differentiable constraint layer.

**Kinematic methods**: Ignore constraints → infeasible outputs

**ReActor**: Optimize within constraints → feasible by construction

**Benefit**: Guarantees rather than post-hoc fixes.

## Conceptual Analogies

### Analogy 1: Language Translation

**Naive approach**: Word-by-word translation (kinematic retargeting)
- Fast but loses meaning, grammatically incorrect

**ReActor approach**: Adapt translation to target language grammar (physics constraints)
- Slower but preserves meaning and correctness

### Analogy 2: Recipe Adaptation

**Problem**: Adapt recipe from professional kitchen to home kitchen (different equipment)

**Naive**: Use exact same steps (kinematic retargeting)
- Fails because home oven different from professional

**ReActor**: Adjust temperatures, times based on equipment (learned offsets)
- Same dish, adapted to constraints

### Analogy 3: Music Transcription

**Problem**: Transcribe piano piece for guitar (different instrument)

**Naive**: Play same notes (kinematic retargeting)
- Some notes impossible on guitar

**ReActor**: Transpose to playable range, adjust fingerings (physics-aware adaptation)
- Preserves musical intent, respects instrument constraints

## Key Insights for Practitioners

### Insight 1: Joint Optimization Beats Sequential

**Lesson**: When two components interact, optimize them together.

**Applications beyond ReActor**:
- Curriculum learning: Task difficulty + policy
- Sim-to-real: Simulation parameters + control policy
- Multi-task learning: Task representation + shared policy

### Insight 2: Sparse Parameterization for Generalization

**Lesson**: Fewer parameters can generalize better than many.

**Trade-off**: Expressiveness vs generalization

**When to use sparse**:
- Limited data
- Want interpretability
- Need to generalize across instances

### Insight 3: Gradient Approximations Can Work

**Lesson**: Exact gradients not always necessary.

**ReActor example**: Policy gradient approximates bilevel gradient

**When to approximate**:
- Exact computation intractable
- Approximation has similar optima
- Can validate empirically

### Insight 4: Physics-in-Loop for Guarantees

**Lesson**: Optimize in the deployment environment.

**Benefit**: Guarantees by construction, not post-hoc validation

**Cost**: Slower optimization, requires simulator

## Research Trajectory

### Past: Sequential Pipelines

1. Retarget motion (kinematic)
2. Train policy (RL)
3. Post-process artifacts (heuristics)

**Problem**: Each stage doesn't know about others.

### Present: Joint Optimization (ReActor)

1. Retarget + train policy simultaneously
2. Co-evolution eliminates artifacts

**Advance**: Stages inform each other.

### Future: End-to-End Learning?

**Potential**: Neural network directly maps source → target motion

**Challenges**:
- Data requirements
- Physics guarantees
- Interpretability

**Likely outcome**: Hybrid approaches combining learning and optimization.

## Mental Model Summary

**Think of ReActor as**:
- A bilevel optimization framework
- Where motion adaptation and policy training co-evolve
- Using physics simulation as a constraint layer
- With sparse parameterization for generalization
- And gradient approximations for tractability

**Core innovation**: Joint refinement beats sequential pipeline.

**Key trade-off**: Training time vs output quality.

**Best use case**: High-quality physics-based animation with novel characters.
