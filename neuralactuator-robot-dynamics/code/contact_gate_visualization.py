"""Contact-gate demo for NeuralActuator (arXiv 2607.11734).

Illustrates the paper's two-stage force output f_hat = g * f_raw (Sec. III-D):
a raw force regressor is inevitably noisy in no-contact segments, and a
contact-probability gate g in [0,1] suppresses those false positives.
The paper reports classical baselines with 33-59% false-positive contact
rates, while the gated NeuralActuator output stays near zero (0.00-0.02 N).

Self-contained: numpy only; matplotlib (Agg) is optional for the figure.
"""

import numpy as np

rng = np.random.default_rng(0)

# --- synthetic ground truth: two contact episodes on a 20 s timeline ---
HZ = 60
T = 20.0
t = np.arange(0, T, 1.0 / HZ)
f_gt = np.zeros_like(t)
f_gt[(t > 4.0) & (t < 7.0)] = 2.0 * np.sin(np.pi * (t[(t > 4.0) & (t < 7.0)] - 4.0) / 3.0) ** 2
f_gt[(t > 12.0) & (t < 15.5)] = 1.2

# --- stage 1: raw force head = ground truth + regression noise + slow drift ---
# The drift mimics unmodeled friction/thermal effects leaking into the residual.
f_raw = f_gt + 0.12 * rng.standard_normal(t.size) + 0.15 * np.sin(0.3 * t)

# --- stage 2: contact gate ---
# Train-time label rule from the paper: g_gt = 1[||f_gt|| > eps], eps = 0.01 N.
EPS = 0.01
g_gt = (np.abs(f_gt) > EPS).astype(float)
# A trained gate outputs a sharp but not perfect probability; simulate it with
# a logistic response on a noisy contact score plus a little label lag.
score = np.convolve(g_gt, np.ones(5) / 5, mode="same") + 0.05 * rng.standard_normal(t.size)
g = 1.0 / (1.0 + np.exp(-18.0 * (score - 0.5)))

f_gated = g * f_raw  # final output f_hat = g * f_raw

# --- metrics: false-positive force in no-contact segments ---
no_contact = g_gt < 0.5
mae_raw = np.abs(f_raw[no_contact]).mean()
mae_gated = np.abs(f_gated[no_contact]).mean()
fp_rate_raw = (np.abs(f_raw[no_contact]) > 0.1).mean() * 100
fp_rate_gated = (np.abs(f_gated[no_contact]) > 0.1).mean() * 100
mae_contact_raw = np.abs(f_raw[~no_contact] - f_gt[~no_contact]).mean()
mae_contact_gated = np.abs(f_gated[~no_contact] - f_gt[~no_contact]).mean()

print("[contact-gate demo] two-stage output f_hat = g * f_raw   (eps = %.2f N)" % EPS)
print("  no-contact segments (%d%% of timeline):" % (100 * no_contact.mean()))
print("    raw head   |f|  MAE = %.3f N   false-positive rate (>0.1 N) = %4.1f%%" % (mae_raw, fp_rate_raw))
print("    gated out  |f|  MAE = %.3f N   false-positive rate (>0.1 N) = %4.1f%%" % (mae_gated, fp_rate_gated))
print("  contact segments:")
print("    raw head   force MAE = %.3f N" % mae_contact_raw)
print("    gated out  force MAE = %.3f N  (gate barely hurts in-contact accuracy)" % mae_contact_gated)
print("  paper analogue: classical baselines 33-59%% false positives vs near-zero for the gated model")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, f_gt, "k-", lw=1.5, label="ground truth $f_{gt}$")
    axes[0].plot(t, f_raw, color="tab:red", alpha=0.6, lw=0.8, label="raw head $\\hat f_{raw}$")
    axes[0].set_ylabel("force (N)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Stage 1: raw force regression drifts in no-contact segments")

    axes[1].plot(t, g_gt, "k--", lw=1, label="gate label $g_{gt}$")
    axes[1].plot(t, g, color="tab:blue", lw=1.2, label="predicted gate $g$")
    axes[1].set_ylabel("contact prob.")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Stage 2: contact-probability gate")

    axes[2].plot(t, f_gt, "k-", lw=1.5, label="ground truth")
    axes[2].plot(t, f_gated, color="tab:green", lw=1.0, label="gated output $\\hat f = g\\,\\hat f_{raw}$")
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Gated output: false positives suppressed, contact force preserved")

    fig.tight_layout()
    out = __file__.replace("contact_gate_visualization.py", "contact_gate_demo.png")
    fig.savefig(out, dpi=110)
    print("[plot] saved figure to %s" % out)
except ImportError:
    print("[plot] matplotlib unavailable, skipped figure")
