# ReActor: Technical Method

## Bilevel Formulation

### Problem Setup

**Upper level** (motion adaptation):
```
min_θ L_motion(θ, π*(θ))
```

**Lower level** (policy training):
```
π*(θ) = argmin_π L_tracking(π, θ)
```

Where:
- θ: retargeting parameters
- π: RL tracking policy
- π*(θ): optimal policy for given θ

### Loss Functions

**Motion loss** L_motion:
```
L_motion = w_pose · ||q_adapted - q_original||² 
         + w_vel · ||q̇_adapted - q̇_original||²
         + w_contact · L_contact
```

**Tracking loss** L_tracking (RL reward):
```
L_tracking = -R(s, a) where R includes:
- Pose error
- Velocity matching
- End effector accuracy
- Contact consistency
```

## Gradient Derivation

### Simplified Gradient Estimate (Eq. 4-7)

Full gradient (intractable):
```
∇_θ L = ∇_θ L_motion + (∂π*/∂θ) · (∂L_motion/∂π)
```

Requires computing ∂π*/∂θ via implicit function theorem:
```
∂π*/∂θ = -H^(-1) · (∂²L_tracking/∂π∂θ)
where H = ∂²L_tracking/∂π²
```

**ReActor approximation** (Eq. 7):
```
∇_θ L ≈ ∇_θ L_motion + λ · E_τ[∇_θ log π(a|s) · A(s,a)]
```

Components:
- **Direct motion gradient**: ∇_θ L_motion (backprop through retargeting)
- **Policy gradient term**: ∇_θ log π(a|s) (sensitivity of policy to θ)
- **Advantage weighting**: A(s,a) (how good action a is in state s)
- **Balance parameter**: λ (trade-off between fidelity and tracking)

### Implementation

1. Sample trajectories τ using current policy π
2. Compute advantages A(s,a) using GAE
3. Compute motion gradient: ∇_θ L_motion
4. Compute policy gradient: E[∇_θ log π(a|s) · A(s,a)]
5. Update θ: θ ← θ - α(∇_θ L_motion + λ · policy_grad)

## Retargeting Parameterization

### Nominal Retargeting (Fixed)

**Position matching** (Eq. 8-9):
```
p_target = scale · p_source + offset
where scale = height_target / height_source
```

**Orientation transfer** (Eq. 10):
```
R_target = R_source (copy rotations directly)
```

**Inverse kinematics** (Eq. 11):
```
q_nominal = IK(p_target, R_target)
```

### Learnable Offsets (Optimized)

**Per-joint rotation offsets** (Eq. 12):
```
q_adapted[i] = q_nominal[i] + Δq[i]
where Δq[i] ∈ ℝ³ (axis-angle representation)
```

**Per-joint position offsets** for end effectors (Eq. 13):
```
p_adapted[j] = p_nominal[j] + Δp[j]
where Δp[j] ∈ ℝ³ (Cartesian offset)
```

**Constant over time** (Eq. 14):
```
Δq[i](t) = Δq[i] ∀t (same offset for all frames)
```

### Parameter Count

**Humanoid example** (34 DOF):
- Rotation offsets: 34 joints × 3 = 102 parameters
- Position offsets: 4 end effectors × 3 = 12 parameters
- **Total**: 114 parameters

**vs per-frame**: 114 × num_frames (e.g., 114 × 300 = 34,200 parameters)

## RL Setup

### State Space (Observation)

**Proprioceptive** (Table 1):
- Joint positions: q ∈ ℝ^n_joints
- Joint velocities: q̇ ∈ ℝ^n_joints
- Root orientation: R_root ∈ SO(3)
- Root linear velocity: v_root ∈ ℝ³
- Root angular velocity: ω_root ∈ ℝ³

**Reference motion**:
- Target joint positions: q_ref ∈ ℝ^n_joints
- Target joint velocities: q̇_ref ∈ ℝ^n_joints
- Target end effector positions: p_ref ∈ ℝ^(3×n_ee)

**Contact information**:
- Foot contact binary flags: c ∈ {0,1}^n_feet

**Total dimension**: ~200 for humanoid

### Action Space

**Joint torques**: τ ∈ ℝ^n_actuated

**Parameterization**:
- PD controller targets: q_target, q̇_target
- Network outputs target positions/velocities
- PD controller computes torques: τ = K_p(q_target - q) + K_d(q̇_target - q̇)

### Reward Function (Table 1)

**Pose tracking** (w=0.5):
```
r_pose = exp(-2 · ||q - q_ref||²)
```

**Velocity matching** (w=0.1):
```
r_vel = exp(-0.1 · ||q̇ - q̇_ref||²)
```

**End effector tracking** (w=0.3):
```
r_ee = exp(-10 · ||p_ee - p_ref||²)
```

**Contact consistency** (w=0.05):
```
r_contact = -|c - c_ref| (penalize contact mismatches)
```

**Action smoothness** (w=0.03):
```
r_smooth = -||a_t - a_{t-1}||²
```

**Torque regularization** (w=0.02):
```
r_torque = -||τ||²
```

**Total reward**:
```
R = Σ w_i · r_i
```

### Training Details

**Algorithm**: PPO (Proximal Policy Optimization)

**Hyperparameters**:
- Learning rate: 3e-4 (policy), 1e-3 (retargeting)
- Discount factor: γ = 0.99
- GAE lambda: λ_GAE = 0.95
- PPO clip: ε = 0.2
- Entropy coefficient: 0.01
- Batch size: 32768 transitions
- Mini-batch size: 4096
- PPO epochs: 5

**Environment**:
- Simulator: Isaac Gym (GPU-accelerated)
- Parallel environments: 4096
- Timestep: 0.0167s (60 Hz)
- Episode length: 300 steps (5 seconds)

**Update schedule**:
- 10 policy updates per retargeting update
- Retargeting updated every 10M environment steps
- Total training: ~50M steps (~6 hours on RTX 3090)

### Adaptive Sampling

**Problem**: Some motion segments are harder to track than others.

**Solution**: Sample episodes weighted by tracking error:
```
p(segment_i) ∝ exp(β · error_i)
```

**Effect**: Spend more time on challenging segments (turns, transitions).

## Implementation Details

### Network Architecture

**Policy network**:
- Input: state (dim ~200)
- Hidden: [512, 256, 128]
- Output: action mean and log std (dim ~34)
- Activation: ELU

**Value network**:
- Input: state (dim ~200)
- Hidden: [512, 256, 128]
- Output: value estimate (dim 1)
- Activation: ELU

### Retargeting Optimization

**Optimizer**: Adam
- Learning rate: 1e-3
- Betas: (0.9, 0.999)
- Weight decay: 1e-4

**Gradient clipping**: max norm = 1.0

**Initialization**: Δq = 0, Δp = 0 (start from nominal retargeting)

### Termination Conditions

Episode terminates if:
- Root height < 0.5m (fallen)
- Root orientation > 60° from upright
- Joint limit violation
- Maximum episode length reached

### Evaluation Metrics

**Foot sliding** (mm/frame):
```
slide = ||v_foot|| when foot should be in contact
```

**Ground penetration** (mm):
```
penetration = max(0, -foot_height)
```

**Self-penetration** (mm):
```
self_pen = max(0, overlap between body parts)
```

**Tracking error** (degrees):
```
error = ||q - q_ref||
```

**Success rate** (%):
```
success = episodes without termination / total episodes
```

## Computational Requirements

**Training**:
- GPU: NVIDIA RTX 3090 (24GB)
- Time: ~6 hours per character-motion pair
- Memory: ~20GB GPU, ~32GB RAM

**Inference**:
- Real-time (60 FPS) on CPU
- Policy network forward pass: ~1ms
- Physics simulation: ~15ms per step

**Scalability**:
- Scales to 34 DOF humanoids
- 4096 parallel environments
- Batch processing for multiple motions
