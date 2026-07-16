"""
Teaching demo for NeuralActuator's core idea (arXiv 2607.11734):
a torque-surrogate head can be trained WITHOUT any torque labels, purely by
back-propagating a pose-reconstruction loss through a differentiable simulator
(BPTT), because the real actuator (cheap geared servo) has nonlinear friction,
backlash and current saturation that break the linear tau = Kt * I model.

Toy system: 1-DoF "cheap servo + link" pendulum.
  - TrueSystem  = ground-truth plant with Stribeck friction + backlash + current
                  saturation (stands in for the real, unmodeled robot).
  - DiffSim     = the simulator NeuralActuator trains against: it only knows
                  rigid-body inertia + gravity, NOT friction/backlash/saturation
                  (mirrors the paper's "unloaded model, contact disabled" setup).
  - Surrogate   = small hand-written MLP, input = recent window of recorded
                  telemetry (current, position, velocity, tracking error).
                  Trained by rolling DiffSim forward under the surrogate's
                  (clipped) output and penalizing only POSITION error (Huber),
                  never given the true torque. Gradients are hand-derived
                  reverse-mode BPTT through the semi-implicit Euler step
                  (mirrors Eq. 10-12 of the paper).

After training we compare:
  (a) trajectory reconstruction error: linear Kt*I baseline vs learned surrogate
  (b) external-torque recovery: tau_ID - tau_pred, learned surrogate vs linear baseline
      (mirrors Eq. 4's tau_ext = tau_ID - tau_act, with tau_act replaced by the surrogate)

Pure numpy, no autodiff framework, no GPU required.
"""

import numpy as np

RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Physical constants (toy 1-DoF servo-driven link)
# ----------------------------------------------------------------------------
DT = 1.0 / 60.0          # 60 Hz control rate, matching the paper's control rate
J = 0.02                 # link inertia (kg*m^2)
MGL = 0.5                # gravity torque amplitude m*g*Lc (N*m)
KT_TRUE = 0.9             # true (unknown to the model) motor torque constant (N*m/A)
I_MAX = 1.0               # current saturation (A) -- low enough to be hit often, like a cheap servo
COULOMB = 0.3             # Coulomb (dry, sign-dependent) friction (N*m)
STRIBECK_EXTRA = 0.5       # extra static friction bump (N*m)
STRIBECK_VS = 0.1          # Stribeck velocity scale (rad/s)
VISCOUS = 0.05             # viscous friction coeff (N*m*s/rad)
BACKLASH_GAP = 0.15        # half-width of the gear backlash "slack" state (large -> visible lost motion)
BACKLASH_KI = 2.0          # motor-side gap-filling gain (rad/s per Amp, lumped unit)
KP_PD = 6.0                # current-space PD gains of the low-level servo controller
KD_PD = 0.3
DAMP_SIM = 0.15            # KNOWN linear damping modeled by DiffSim's C(q,qdot) term -- kept large enough
                           # to keep the rigid-body integrator numerically stable over long rollouts; the
                           # sign-dependent Coulomb/Stribeck bump, backlash lost-motion, and saturation
                           # above remain entirely unmodeled and must be absorbed by the learned surrogate

WINDOW = 7                 # L tokens per input (scaled-down analog of the paper's 9)
N_FEAT = 5                 # [q_cmd, theta, theta_dot, I_sat, e]
HIDDEN = 32
TAU_MIN, TAU_MAX = -1.6, 1.6   # clip bounds for the surrogate's generalized-effort output


def stribeck_friction(qdot: np.ndarray) -> np.ndarray:
    """Nonlinear friction: Coulomb + Stribeck bump + viscous. True torque != Kt*I because of this term."""
    sign = np.tanh(qdot / 1e-3)  # smoothed sign to keep the toy plant differentiable-free but well-behaved
    return sign * (COULOMB + STRIBECK_EXTRA * np.exp(-(qdot / STRIBECK_VS) ** 2)) + VISCOUS * qdot


class TrueSystem:
    """Ground-truth plant: Stribeck friction + backlash (gear-mesh dead zone) + current saturation."""

    def __init__(self):
        self.theta = 0.0
        self.theta_dot = 0.0
        self.gap = 0.0  # backlash state: motor-side mesh position within the slack, in [-BACKLASH_GAP, BACKLASH_GAP]

    def step(self, q_cmd: float, qdot_cmd: float, tau_ext: float = 0.0):
        e = q_cmd - self.theta
        edot = qdot_cmd - self.theta_dot
        i_cmd = KP_PD * e + KD_PD * edot
        i_sat = np.clip(i_cmd, -I_MAX, I_MAX)

        # backlash: torque only transmits once the gear mesh has taken up the slack
        self.gap = np.clip(self.gap + BACKLASH_KI * i_sat * DT, -BACKLASH_GAP, BACKLASH_GAP)
        engagement = abs(self.gap) / BACKLASH_GAP
        tau_transmitted = KT_TRUE * i_sat * engagement

        tau_true = tau_transmitted - stribeck_friction(self.theta_dot)
        theta_ddot = (tau_true - MGL * np.sin(self.theta) + tau_ext) / J

        self.theta_dot = self.theta_dot + theta_ddot * DT
        self.theta = self.theta + self.theta_dot * DT
        return i_sat, theta_ddot


def ref_trajectory(t: np.ndarray, freqs, amps, phases):
    """Smooth sum-of-sines reference trajectory: q_cmd(t) and qdot_cmd(t)."""
    q = np.zeros_like(t)
    qd = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phases):
        q += a * np.sin(2 * np.pi * f * t + p)
        qd += a * 2 * np.pi * f * np.cos(2 * np.pi * f * t + p)
    return q, qd


def make_trajectory(n_steps: int, seed: int, tau_ext_fn=None):
    """Roll out TrueSystem under a random smooth reference, recording telemetry + true dynamics."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.2, 0.8, size=3)
    amps = rng.uniform(0.2, 0.5, size=3)
    phases = rng.uniform(0, 2 * np.pi, size=3)
    t = np.arange(n_steps) * DT
    q_cmd, qdot_cmd = ref_trajectory(t, freqs, amps, phases)

    sys_ = TrueSystem()
    theta_log = np.zeros(n_steps)
    thetadot_log = np.zeros(n_steps)
    i_log = np.zeros(n_steps)
    thetaddot_log = np.zeros(n_steps)
    for k in range(n_steps):
        tau_ext = tau_ext_fn(t[k]) if tau_ext_fn is not None else 0.0
        i_sat, theta_ddot = sys_.step(q_cmd[k], qdot_cmd[k], tau_ext=tau_ext)
        theta_log[k] = sys_.theta
        thetadot_log[k] = sys_.theta_dot
        i_log[k] = i_sat
        thetaddot_log[k] = theta_ddot
    e_log = q_cmd - theta_log
    return dict(t=t, q_cmd=q_cmd, theta=theta_log, theta_dot=thetadot_log,
                i_sat=i_log, e=e_log, theta_ddot=thetaddot_log)


# ----------------------------------------------------------------------------
# Differentiable simulator: rigid-body-only, NO friction/backlash/saturation.
# The surrogate must absorb those unmodeled effects to reproduce the real motion.
# ----------------------------------------------------------------------------
def diffsim_step(theta, theta_dot, tau_bar):
    """Semi-implicit Euler step + analytic partials, needed for hand-written BPTT (mirrors Eq. 10).
    DiffSim models inertia + gravity + a known linear damping (part of the analytic C(q,qdot) term);
    it does NOT know the nonlinear Stribeck friction, backlash or current saturation of TrueSystem —
    the surrogate must absorb those unmodeled effects to reproduce the real motion."""
    theta_ddot = (tau_bar - MGL * np.sin(theta) - DAMP_SIM * theta_dot) / J
    theta_dot_next = theta_dot + theta_ddot * DT
    theta_next = theta + theta_dot_next * DT

    # partial derivatives for reverse-mode BPTT
    d_ddot_dtheta = -MGL * np.cos(theta) / J
    d_ddot_dthetadot = -DAMP_SIM / J
    d_ddot_dtau = 1.0 / J
    d_thetadotnext_dthetadot = 1.0 + DT * d_ddot_dthetadot
    d_thetadotnext_dtheta = DT * d_ddot_dtheta
    d_thetadotnext_dtau = DT * d_ddot_dtau
    d_thetanext_dtheta = 1.0 + DT * d_thetadotnext_dtheta
    d_thetanext_dthetadot = DT * d_thetadotnext_dthetadot  # theta_next = theta + theta_dot_next * dt
    d_thetanext_dtau = DT * d_thetadotnext_dtau

    partials = dict(
        d_theta_next_d_theta=d_thetanext_dtheta,
        d_theta_next_d_thetadot=d_thetanext_dthetadot,
        d_theta_next_d_tau=d_thetanext_dtau,
        d_thetadot_next_d_theta=d_thetadotnext_dtheta,
        d_thetadot_next_d_thetadot=d_thetadotnext_dthetadot,
        d_thetadot_next_d_tau=d_thetadotnext_dtau,
    )
    return theta_next, theta_dot_next, partials


# ----------------------------------------------------------------------------
# Hand-written MLP: torque-surrogate head. tanh hidden layer + linear output.
# ----------------------------------------------------------------------------
class TorqueSurrogateMLP:
    def __init__(self, in_dim, hidden, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale2, size=(hidden,))
        self.b2 = 0.0
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t_step = 0

    def _params(self):
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=np.array(self.b2))

    def forward(self, x):
        """x: (in_dim,) -> tau_pred scalar, cache for backward."""
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        out = a1 @ self.W2 + self.b2
        cache = (x, z1, a1)
        return out, cache

    def backward(self, cache, d_out):
        """d_out: upstream gradient dL/d(tau_pred). Returns grads dict (accumulate across time steps)."""
        x, z1, a1 = cache
        d_W2 = a1 * d_out
        d_b2 = d_out
        d_a1 = self.W2 * d_out
        d_z1 = d_a1 * (1 - a1 ** 2)
        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1
        return dict(W1=d_W1, b1=d_b1, W2=d_W2, b2=np.array(d_b2))

    def adamw_step(self, grads, lr=3e-3, wd=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t_step += 1
        params = self._params()
        for k in params:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g * g)
            mhat = self.m[k] / (1 - beta1 ** self.t_step)
            vhat = self.v[k] / (1 - beta2 ** self.t_step)
            update = lr * mhat / (np.sqrt(vhat) + eps) + lr * wd * params[k]
            if k == "W1":
                self.W1 -= update
            elif k == "b1":
                self.b1 -= update
            elif k == "W2":
                self.W2 -= update
            elif k == "b2":
                self.b2 -= float(update)


def moving_average(x, window):
    """Short boxcar smoothing: raw per-step tau_ID - tau_act residuals are high-frequency and noisy
    (finite-difference acceleration + near-discontinuous friction/backlash terms), same reason real
    force-estimation pipelines low-pass filter residuals and why NeuralActuator's own architecture
    aggregates a whole window via temporal mean pooling rather than reading a single instant."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def huber(x, beta=1.0):
    ax = abs(x)
    if ax <= beta:
        return 0.5 * x * x / beta, x / beta
    return beta * (ax - 0.5 * beta), np.sign(x)


def build_window_features(traj, center_idx, window):
    """Build the flattened [q_cmd, theta, theta_dot, i_sat, e] window ending at center_idx (inclusive)."""
    start = center_idx - window + 1
    feats = []
    for k in range(start, center_idx + 1):
        kk = max(k, 0)  # pad by repeating the first frame, like a cold-start history buffer
        feats.extend([traj["q_cmd"][kk], traj["theta"][kk], traj["theta_dot"][kk],
                      traj["i_sat"][kk], traj["e"][kk]])
    return np.array(feats)


def bptt_rollout_and_train(mlp, traj, horizon, start_idx, huber_beta=1.0):
    """One BPTT rollout+train window: forward through DiffSim driven by the surrogate, then reverse-mode backward."""
    theta0 = traj["theta"][start_idx]
    theta_dot0 = traj["theta_dot"][start_idx]

    thetas = [theta0]
    theta_dots = [theta_dot0]
    caches = []
    tau_preds = []
    clip_masks = []
    partials_list = []

    theta, theta_dot = theta0, theta_dot0
    for step in range(horizon):
        idx = start_idx + step
        x = build_window_features(traj, idx, WINDOW)
        tau_pred, cache = mlp.forward(x)
        tau_bar = np.clip(tau_pred, TAU_MIN, TAU_MAX)
        clip_mask = 1.0 if TAU_MIN < tau_pred < TAU_MAX else 0.0  # zero gradient once saturated (Pitfall #1)

        theta_next, theta_dot_next, partials = diffsim_step(theta, theta_dot, tau_bar)

        caches.append(cache)
        tau_preds.append(tau_pred)
        clip_masks.append(clip_mask)
        partials_list.append(partials)
        thetas.append(theta_next)
        theta_dots.append(theta_dot_next)
        theta, theta_dot = theta_next, theta_dot_next

    # position Huber loss at every rolled step, vs REAL recorded theta at the same absolute index
    total_loss = 0.0
    a_theta = 0.0
    a_thetadot = 0.0
    grads_accum = {k: np.zeros_like(v) for k, v in mlp._params().items()}

    loss_grads = []
    for step in range(horizon):
        idx = start_idx + step + 1
        real_theta = traj["theta"][idx]
        err = thetas[step + 1] - real_theta
        l, dl = huber(err, beta=huber_beta)
        total_loss += l
        loss_grads.append(dl)  # dL_t / d theta_{t+1}

    for step in reversed(range(horizon)):
        a_theta += loss_grads[step]  # inject local loss gradient into the adjoint at this state
        p = partials_list[step]

        d_tau = a_theta * p["d_theta_next_d_tau"] + a_thetadot * p["d_thetadot_next_d_tau"]
        d_tau_pred = d_tau * clip_masks[step]  # backprop through clip
        g = mlp.backward(caches[step], d_tau_pred)
        for k in grads_accum:
            grads_accum[k] += g[k]

        new_a_theta = a_theta * p["d_theta_next_d_theta"] + a_thetadot * p["d_thetadot_next_d_theta"]
        new_a_thetadot = a_theta * p["d_theta_next_d_thetadot"] + a_thetadot * p["d_thetadot_next_d_thetadot"]
        a_theta, a_thetadot = new_a_theta, new_a_thetadot

    return total_loss / horizon, grads_accum


def surrogate_rollout_predict(mlp, traj, horizon, start_idx):
    """Inference-time rollout: drive DiffSim with the trained surrogate, recorded-telemetry-conditioned."""
    theta, theta_dot = traj["theta"][start_idx], traj["theta_dot"][start_idx]
    out_theta = np.zeros(horizon)
    out_tau = np.zeros(horizon)
    for step in range(horizon):
        idx = start_idx + step
        x = build_window_features(traj, idx, WINDOW)
        tau_pred, _ = mlp.forward(x)
        tau_bar = np.clip(tau_pred, TAU_MIN, TAU_MAX)
        theta, theta_dot, _ = diffsim_step(theta, theta_dot, tau_bar)
        out_theta[step] = theta
        out_tau[step] = tau_pred
    return out_theta, out_tau


def linear_kt_baseline_fit(train_trajs):
    """Classic ID-Linear baseline: least-squares fit of tau_needed = tau_ID (Eq. 4 rigid-body side) against I_sat."""
    all_i = []
    all_tau_needed = []
    for traj in train_trajs:
        tau_needed = J * traj["theta_ddot"] + MGL * np.sin(traj["theta"]) + DAMP_SIM * traj["theta_dot"]
        all_i.append(traj["i_sat"])
        all_tau_needed.append(tau_needed)
    i_cat = np.concatenate(all_i)
    tau_cat = np.concatenate(all_tau_needed)
    kt_fit = float(np.dot(i_cat, tau_cat) / np.dot(i_cat, i_cat))
    return kt_fit


def baseline_rollout_predict(kt_fit, traj, horizon, start_idx):
    theta, theta_dot = traj["theta"][start_idx], traj["theta_dot"][start_idx]
    out_theta = np.zeros(horizon)
    out_tau = np.zeros(horizon)
    for step in range(horizon):
        idx = start_idx + step
        tau_pred = kt_fit * traj["i_sat"][idx]
        tau_bar = np.clip(tau_pred, TAU_MIN, TAU_MAX)
        theta, theta_dot, _ = diffsim_step(theta, theta_dot, tau_bar)
        out_theta[step] = theta
        out_tau[step] = tau_pred
    return out_theta, out_tau


def main():
    n_train_steps = 600
    n_epochs = 900
    horizon = 40
    n_train_trajs = 6

    train_trajs = [make_trajectory(n_train_steps, seed=100 + i) for i in range(n_train_trajs)]
    kt_fit = linear_kt_baseline_fit(train_trajs)
    print(f"[baseline] least-squares linear Kt fit: {kt_fit:.4f} N*m/A  (true Kt = {KT_TRUE:.4f}, "
          f"note the fit absorbs friction/backlash bias since it never sees them)")

    in_dim = WINDOW * N_FEAT
    mlp = TorqueSurrogateMLP(in_dim, HIDDEN, seed=1)

    print(f"\n[train] BPTT over {n_epochs} epochs, horizon={horizon} steps, "
          f"{n_train_trajs} trajectories, window={WINDOW} frames, NO torque labels used.")
    for epoch in range(n_epochs):
        traj = train_trajs[epoch % n_train_trajs]
        max_start = n_train_steps - horizon - 1
        start_idx = RNG.integers(WINDOW, max_start)
        loss, grads = bptt_rollout_and_train(mlp, traj, horizon, start_idx)
        mlp.adamw_step(grads, lr=3e-3, wd=1e-4)
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  rollout Huber pose loss = {loss:.6f}")

    # --- (a) trajectory reconstruction comparison on a held-out test trajectory ---
    test_steps = 400
    test_traj = make_trajectory(test_steps, seed=999)
    eval_horizon = 300
    start_idx = WINDOW

    surrogate_theta, surrogate_tau = surrogate_rollout_predict(mlp, test_traj, eval_horizon, start_idx)
    baseline_theta, baseline_tau = baseline_rollout_predict(kt_fit, test_traj, eval_horizon, start_idx)
    real_theta = test_traj["theta"][start_idx + 1: start_idx + 1 + eval_horizon]

    surrogate_mae = np.mean(np.abs(surrogate_theta - real_theta))
    baseline_mae = np.mean(np.abs(baseline_theta - real_theta))
    print("\n[eval-a] trajectory reconstruction on held-out test trajectory "
          f"({eval_horizon} steps, recorded-telemetry-conditioned rollout):")
    print(f"  linear Kt*I baseline   position MAE = {baseline_mae:.5f} rad ({np.degrees(baseline_mae):.3f} deg)")
    print(f"  learned surrogate      position MAE = {surrogate_mae:.5f} rad ({np.degrees(surrogate_mae):.3f} deg)")
    print(f"  improvement factor     = {baseline_mae / max(surrogate_mae, 1e-9):.2f}x")

    # --- (b) external-torque recovery: tau_ID - tau_surrogate vs tau_ID - Kt_fit*I ---
    def disturbance(t):
        return 0.25 if 2.0 <= t <= 3.5 else 0.0  # a constant "push" torque bump, unknown to the model

    dist_traj = make_trajectory(test_steps, seed=1234, tau_ext_fn=disturbance)
    dist_horizon = eval_horizon
    dist_start = WINDOW

    # tau_ID computed from the (privileged, offline) recorded true acceleration, mirrors Eq. 4
    tau_id = J * dist_traj["theta_ddot"] + MGL * np.sin(dist_traj["theta"]) + DAMP_SIM * dist_traj["theta_dot"]
    tau_id_eval = tau_id[dist_start: dist_start + dist_horizon]
    t_eval = dist_traj["t"][dist_start: dist_start + dist_horizon]
    true_ext = np.array([disturbance(tt) for tt in t_eval])

    # surrogate's own per-step prediction (not the rolled sim state) on the SAME recorded telemetry window
    surrogate_tau_pred = np.zeros(dist_horizon)
    for step in range(dist_horizon):
        idx = dist_start + step
        x = build_window_features(dist_traj, idx, WINDOW)
        tau_pred, _ = mlp.forward(x)
        surrogate_tau_pred[step] = tau_pred
    baseline_tau_pred = kt_fit * dist_traj["i_sat"][dist_start: dist_start + dist_horizon]

    f_ext_surrogate_raw = tau_id_eval - surrogate_tau_pred
    f_ext_baseline_raw = tau_id_eval - baseline_tau_pred
    smooth_win = 9  # ~ WINDOW length; both residuals are smoothed identically before comparison
    f_ext_surrogate = moving_average(f_ext_surrogate_raw, smooth_win)
    f_ext_baseline = moving_average(f_ext_baseline_raw, smooth_win)

    mae_surrogate_force = np.mean(np.abs(f_ext_surrogate - true_ext))
    mae_baseline_force = np.mean(np.abs(f_ext_baseline - true_ext))
    print("\n[eval-b] external-torque recovery (tau_ID - tau_act), 0.25 N*m bump applied over t in [2.0, 3.5] s:")
    print(f"  (raw per-step residual is high-frequency/noisy; both methods smoothed with a "
          f"{smooth_win}-step moving average before scoring, same idea as the paper's temporal mean pooling)")
    print(f"  linear Kt*I baseline   recovery MAE = {mae_baseline_force:.5f} N*m")
    print(f"  learned surrogate      recovery MAE = {mae_surrogate_force:.5f} N*m")
    print(f"  improvement factor     = {mae_baseline_force / max(mae_surrogate_force, 1e-9):.2f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        axes[0].plot(real_theta, label="real theta (ground truth)", color="black", linewidth=1.5)
        axes[0].plot(baseline_theta, label=f"linear Kt*I baseline (MAE={baseline_mae:.4f})", linestyle="--")
        axes[0].plot(surrogate_theta, label=f"learned surrogate (MAE={surrogate_mae:.4f})")
        axes[0].set_title("(a) Trajectory reconstruction: recorded-telemetry-conditioned rollout")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("theta (rad)")
        axes[0].legend()

        axes[1].plot(t_eval, true_ext, label="true external torque", color="black", linewidth=1.5)
        axes[1].plot(t_eval, f_ext_baseline, label=f"linear baseline recovery (MAE={mae_baseline_force:.4f})",
                     linestyle="--")
        axes[1].plot(t_eval, f_ext_surrogate, label=f"surrogate recovery (MAE={mae_surrogate_force:.4f})")
        axes[1].set_title("(b) External-torque recovery: tau_ID - tau_act")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("torque (N*m)")
        axes[1].legend()

        fig.tight_layout()
        out_path = __file__.replace("torque_surrogate_pose_supervision_demo.py", "torque_surrogate_demo.png")
        fig.savefig(out_path, dpi=130)
        print(f"\n[plot] saved figure to {out_path}")
    except ImportError:
        print("\n[plot] matplotlib not available, skipping figure (numeric results above stand on their own)")


if __name__ == "__main__":
    main()
