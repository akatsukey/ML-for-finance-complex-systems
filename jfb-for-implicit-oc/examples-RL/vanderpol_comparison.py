"""
examples-RL.vanderpol_comparison
----------------------------------
Head-to-head comparison of two gradient-estimation strategies on the
**Van der Pol oscillator stabilisation** problem:

    Method A — JFB-RL (ours)
        Implicit Hamiltonian policy (ImplicitNetOC_RL) trained via the
        JFB-with-estimates surrogate.  The environment is treated as a
        **black box**: dynamics are never differentiated; local Jacobians
        are estimated online by Recursive Least Squares (RLS).

    Method B — Autodiff-BPTT
        Explicit MLP policy trained by backpropagating *directly through
        the differentiable Euler rollout* (standard BPTT).  This requires
        access to the analytical f, which is unavailable in real RL
        settings.

Both methods solve the same problem, use the same optimizer (Adam), and
are trained for the same number of epochs.  The comparison tests whether
JFB-RL, which asks strictly less of the environment, can match the
performance of the privileged Autodiff baseline.

Run from the repo root:
    python jfb-for-implicit-oc/examples-RL/vanderpol_comparison.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "core_RL"),
    os.path.join(_ROOT, "models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.ImplicitNets import Phi
from core_RL.Environment import AnalyticalEnvironment
from core_RL.ImplicitNets_RL import ImplicitNetOC_RL
from core_RL.JacobianEstimator import RLSJacobianEstimator
from models.VanDerPolOC_RL import VanDerPolOC_RL

# ===========================================================================
# Config
# ===========================================================================
DEVICE = "cpu"
SEED   = 42

BATCH      = 64      # training batch size
N_EPOCHS   = 300     # epochs per method
LR         = 3e-3    # Adam learning rate for both methods
GRAD_CLIP  = 1.0     # gradient norm clip (applied to both)

# JFB-RL specific
EXPLORE_STD   = 0.3   # initial exploration noise standard deviation
EXPLORE_DECAY = 0.99  # multiplicative decay per epoch
FP_ALPHA      = 0.5   # fixed-point step size  (< 2/α_L = 2 for this problem)
FP_MAX_ITERS  = 30    # maximum FP iterations
FP_TOL        = 1e-4  # FP convergence tolerance
U_MIN, U_MAX  = -3.0, 3.0

# Evaluation
N_EVAL        = 256   # fresh test ICs (never seen during training)
N_WARMUP      = 8     # jac_est warm-up rollouts for JFB-RL evaluation
WARMUP_STD    = 0.4
WARMUP_DECAY  = 0.6

# Smoothing for loss curves in the plot
SMOOTH_WIN = 15       # moving-average window (epochs)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ===========================================================================
# Problem
# ===========================================================================
prob = VanDerPolOC_RL(
    x10_min=1.5, x10_max=2.5,
    x20_min=-0.5, x20_max=0.5,
    batch_size=BATCH,
    t_initial=0.0, t_final=3.0, nt=60,
    alphaL=1.0, alphaG=5.0,
    device=DEVICE,
)
print(f"Problem  : {prob.oc_problem_name}")
print(f"State    : {prob.state_dim}-D  |  Control: {prob.control_dim}-D")
print(f"Horizon  : T={prob.t_final}, nt={prob.nt}, h={prob.h:.3f}")
print(f"ICs      : x1~U[{prob.x10_min},{prob.x10_max}], x2~U[{prob.x20_min},{prob.x20_max}]")
print(f"Batch    : {BATCH}  |  Epochs: {N_EPOCHS}  |  LR: {LR}")
print("-" * 60)

# ===========================================================================
# A. JFB-RL setup
# ===========================================================================
phi = Phi(3, 50, prob.state_dim, dev=DEVICE)

inn = ImplicitNetOC_RL(
    prob.state_dim, prob.control_dim,
    alpha=FP_ALPHA, max_iters=FP_MAX_ITERS, tol=FP_TOL,
    p_net=phi, oc_problem=prob,
    u_min=U_MIN, u_max=U_MAX, use_control_limits=True,
    dev=DEVICE,
).to(DEVICE)

jac_est = RLSJacobianEstimator(
    nt=prob.nt,
    state_dim=prob.state_dim,
    control_dim=prob.control_dim,
    dt=prob.h,
    alpha_rls=0.9,
    q0=1.0,
    device=DEVICE,
)

env = AnalyticalEnvironment(
    state_dim=prob.state_dim,
    control_dim=prob.control_dim,
    t_initial=prob.t_initial,
    t_final=prob.t_final,
    nt=prob.nt,
    f_callable=prob.compute_f,
    device=DEVICE,
)

opt_jfb = torch.optim.Adam(inn.parameters(), lr=LR)

# ===========================================================================
# B. Autodiff-BPTT setup — explicit MLP policy trained with BPTT
#
#    The MLP maps (z, t) -> u directly.  The training loop keeps the full
#    computation graph through the Euler rollout, so PyTorch autograd
#    propagates ∂J/∂θ through all nt dynamics steps.  This is only possible
#    because the dynamics are differentiable (we call prob.compute_f with
#    grad enabled).  In a real RL setting with a simulator black box, BPTT
#    would not be available.
# ===========================================================================

class MLPPolicy(nn.Module):
    """Explicit policy  π_θ(z, t) → u.  Input: (x1, x2, t), output: scalar u."""

    def __init__(self, state_dim: int, control_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),        nn.Tanh(),
            nn.Linear(hidden, control_dim),
        )
        self.u_min = U_MIN
        self.u_max = U_MAX

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        t_feat = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype) * t
        x = torch.cat([z, t_feat], dim=-1)
        return self.net(x).clamp(self.u_min, self.u_max)


mlp = MLPPolicy(prob.state_dim, prob.control_dim, hidden=64).to(DEVICE)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=LR)

# ===========================================================================
# Training helper — Autodiff-BPTT step
# ===========================================================================

def autodiff_step(policy: MLPPolicy, prob: VanDerPolOC_RL, opt) -> dict:
    """One BPTT training step — differentiable Euler rollout."""
    policy.train()
    opt.zero_grad()

    z0 = prob.sample_initial_condition()
    z = z0.clone()           # starts with no grad; graph builds from first u_k
    running = torch.zeros(prob.batch_size, device=DEVICE)

    for k in range(prob.nt):
        t_k = prob.t_initial + k * prob.h
        u_k = policy(z, t_k)                          # (B, 1)
        running = running + prob.h * prob.compute_lagrangian(t_k, z, u_k)
        dz = prob.compute_f(t_k, z, u_k)             # differentiable!
        z = z + prob.h * dz                           # graph grows nt steps deep

    terminal = prob.compute_G(z)
    loss = (prob.alphaL * running + prob.alphaG * terminal).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    opt.step()

    return {
        "loss":          loss.item(),
        "running_cost":  (prob.alphaL * running.mean()).item(),
        "terminal_cost": (prob.alphaG * terminal.mean()).item(),
    }


# ===========================================================================
# Train A: JFB-RL
# ===========================================================================
print("Training JFB-RL …")
hist_jfb = {"loss": [], "running_cost": [], "terminal_cost": []}
explore_std = EXPLORE_STD
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    z0 = prob.sample_initial_condition()
    out = prob.compute_loss_RL(
        policy=inn,
        env=env,
        jac_est=jac_est,
        z0=z0,
        exploration_std=explore_std,
    )
    opt_jfb.zero_grad()
    out["surrogate"].backward()
    torch.nn.utils.clip_grad_norm_(inn.parameters(), GRAD_CLIP)
    opt_jfb.step()
    explore_std *= EXPLORE_DECAY

    hist_jfb["loss"].append(out["total_cost"])
    hist_jfb["running_cost"].append(out["running_cost"])
    hist_jfb["terminal_cost"].append(out["terminal_cost"])

    if epoch % 50 == 0:
        print(f"  [{epoch:4d}/{N_EPOCHS}]  loss={out['total_cost']:.4f}  "
              f"L={out['running_cost']:.3f}  G={out['terminal_cost']:.3f}  "
              f"explore_std={explore_std:.3f}")

print(f"JFB-RL training done in {time.time()-t0:.1f}s")
print("-" * 60)

# ===========================================================================
# Train B: Autodiff-BPTT
# ===========================================================================
print("Training Autodiff-BPTT …")
hist_mlp = {"loss": [], "running_cost": [], "terminal_cost": []}
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    out = autodiff_step(mlp, prob, opt_mlp)

    hist_mlp["loss"].append(out["loss"])
    hist_mlp["running_cost"].append(out["running_cost"])
    hist_mlp["terminal_cost"].append(out["terminal_cost"])

    if epoch % 50 == 0:
        print(f"  [{epoch:4d}/{N_EPOCHS}]  loss={out['loss']:.4f}  "
              f"L={out['running_cost']:.3f}  G={out['terminal_cost']:.3f}")

print(f"Autodiff-BPTT training done in {time.time()-t0:.1f}s")
print("-" * 60)

# ===========================================================================
# Save checkpoints
# ===========================================================================
out_dir = os.path.join(_ROOT, "results", "VanDerPolOC_RL")
ckpt_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(inn.state_dict(),  os.path.join(ckpt_dir, "jfb_rl.pth"))
torch.save(mlp.state_dict(),  os.path.join(ckpt_dir, "autodiff.pth"))
print(f"Checkpoints saved to {ckpt_dir}")

# ===========================================================================
# Evaluation — fresh ICs never seen during training
# ===========================================================================
print(f"\nEvaluating on N_EVAL={N_EVAL} fresh ICs …")

# Sample fixed test ICs for a fair comparison.
torch.manual_seed(SEED + 1)
prob.batch_size = N_EVAL
z0_eval = prob.sample_initial_condition()
prob.batch_size = BATCH   # restore

# ------ JFB-RL evaluation ---------------------------------------------------
# Build a fresh jac_est for the test set and warm it up with decaying
# exploration noise (same technique as evaluate_portfolio_rl.py).
jac_est_eval = RLSJacobianEstimator(
    nt=prob.nt, state_dim=prob.state_dim, control_dim=prob.control_dim,
    dt=prob.h, alpha_rls=0.9, q0=1.0, device=DEVICE,
)
inn.eval()
print(f"  Warming up jac_est_eval ({N_WARMUP} rollouts, std {WARMUP_STD}→0) …", end=" ")
with torch.no_grad():
    std = WARMUP_STD
    for _ in range(N_WARMUP):
        z = z0_eval.clone()
        t = prob.t_initial
        for k in range(prob.nt):
            _, b_k = jac_est_eval.AB(k)
            inn.set_step_jacobian(b_k)
            u_k = inn(z, t).view(N_EVAL, prob.control_dim)
            u_exc = (u_k + std * torch.randn_like(u_k)).clamp(U_MIN, U_MAX)
            z_next = env.step(z, u_exc, t)
            jac_est_eval.update(k, z, u_exc, z_next)
            z = z_next
            t += prob.h
        std *= WARMUP_DECAY
print("done.")

with torch.no_grad():
    def _setter_eval(k):
        _, b_k = jac_est_eval.AB(k)
        inn.set_step_jacobian(b_k)

    # Reuse AnalyticalEnvironment with N_EVAL batch — just roll out.
    z_jfb = z0_eval.clone()
    t_jfb = prob.t_initial
    z_traj_jfb = torch.zeros(N_EVAL, prob.state_dim, prob.nt + 1, device=DEVICE)
    u_traj_jfb = torch.zeros(N_EVAL, prob.control_dim, prob.nt, device=DEVICE)
    z_traj_jfb[:, :, 0] = z_jfb
    for k in range(prob.nt):
        _setter_eval(k)
        u_k = inn(z_jfb, t_jfb).view(N_EVAL, prob.control_dim)
        z_next = env.step(z_jfb, u_k, t_jfb)
        u_traj_jfb[:, :, k] = u_k
        z_traj_jfb[:, :, k + 1] = z_next
        z_jfb = z_next
        t_jfb += prob.h

# ------ Autodiff-BPTT evaluation --------------------------------------------
mlp.eval()
with torch.no_grad():
    z_ad = z0_eval.clone()
    t_ad = prob.t_initial
    z_traj_ad = torch.zeros(N_EVAL, prob.state_dim, prob.nt + 1, device=DEVICE)
    u_traj_ad = torch.zeros(N_EVAL, prob.control_dim, prob.nt, device=DEVICE)
    z_traj_ad[:, :, 0] = z_ad
    for k in range(prob.nt):
        u_k = mlp(z_ad, t_ad).view(N_EVAL, prob.control_dim)
        z_next = env.step(z_ad, u_k, t_ad)
        u_traj_ad[:, :, k] = u_k
        z_traj_ad[:, :, k + 1] = z_next
        z_ad = z_next
        t_ad += prob.h

# ------ Cost computation ----------------------------------------------------
def total_cost_traj(z_traj, u_traj):
    B = z_traj.shape[0]
    running = torch.zeros(B, device=DEVICE)
    with torch.no_grad():
        for k in range(prob.nt):
            t_k = prob.t_initial + k * prob.h
            running += prob.h * prob.compute_lagrangian(t_k, z_traj[:, :, k], u_traj[:, :, k])
    terminal = prob.compute_G(z_traj[:, :, -1])
    total = prob.alphaL * running + prob.alphaG * terminal
    return total.cpu().numpy(), running.cpu().numpy(), terminal.cpu().numpy()


cost_jfb, run_jfb, term_jfb = total_cost_traj(z_traj_jfb, u_traj_jfb)
cost_ad,  run_ad,  term_ad  = total_cost_traj(z_traj_ad,  u_traj_ad)

print(f"\n{'':>22} {'JFB-RL':>12} {'Autodiff':>12}  {'gap':>8}")
print("-" * 60)
for label, a, b in [
    ("Mean total cost",  cost_jfb.mean(), cost_ad.mean()),
    ("  Mean running",   run_jfb.mean(),  run_ad.mean()),
    ("  Mean terminal",  term_jfb.mean(), term_ad.mean()),
    ("Std  total cost",  cost_jfb.std(),  cost_ad.std()),
]:
    print(f"{label:>22} {a:>12.4f} {b:>12.4f}  {a-b:>+8.4f}")

gap_pct = (cost_jfb.mean() - cost_ad.mean()) / abs(cost_ad.mean()) * 100
print(f"\nJFB-RL vs Autodiff gap: {cost_jfb.mean()-cost_ad.mean():.4f}  ({gap_pct:+.2f}%)")

# ===========================================================================
# Plots
# ===========================================================================
t_grid = np.linspace(prob.t_initial, prob.t_final, prob.nt + 1)
t_ctrl = t_grid[:-1]

x1_jfb = z_traj_jfb[:, 0, :].cpu().numpy()   # (N_EVAL, nt+1)
x2_jfb = z_traj_jfb[:, 1, :].cpu().numpy()
u_jfb  = u_traj_jfb[:, 0, :].cpu().numpy()   # (N_EVAL, nt)

x1_ad  = z_traj_ad[:, 0, :].cpu().numpy()
x2_ad  = z_traj_ad[:, 1, :].cpu().numpy()
u_ad   = u_traj_ad[:, 0, :].cpu().numpy()

COL_JFB = "#d6604d"   # warm red — JFB-RL
COL_AD  = "#4393c3"   # blue    — Autodiff-BPTT
ALF     = 0.20


def smooth(arr, w):
    """Simple moving-average smoothing."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "JFB-RL (black-box env)  vs  Autodiff-BPTT (differentiable env)\n"
    f"Van der Pol stabilisation — N = {N_EVAL} fresh ICs\n"
    f"Mean cost:  JFB-RL = {cost_jfb.mean():.3f}  |  "
    f"Autodiff = {cost_ad.mean():.3f}  |  "
    f"Gap = {cost_jfb.mean()-cost_ad.mean():+.3f}  ({gap_pct:+.1f}%)",
    fontsize=11,
)

# ── Panel 1: Training loss curves ───────────────────────────────────────────
ax = axes[0, 0]
losses_jfb_smooth = smooth(hist_jfb["loss"], SMOOTH_WIN)
losses_ad_smooth  = smooth(hist_mlp["loss"], SMOOTH_WIN)
ep_sm = np.arange(SMOOTH_WIN, N_EPOCHS + 1)

ax.plot(hist_jfb["loss"],   color=COL_JFB, alpha=0.25, lw=0.8)
ax.plot(hist_mlp["loss"],   color=COL_AD,  alpha=0.25, lw=0.8)
ax.plot(ep_sm, losses_jfb_smooth, color=COL_JFB, lw=2,
        label=f"JFB-RL (final={hist_jfb['loss'][-1]:.3f})")
ax.plot(ep_sm, losses_ad_smooth,  color=COL_AD,  lw=2, ls="--",
        label=f"Autodiff-BPTT (final={hist_mlp['loss'][-1]:.3f})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total cost J")
ax.set_title("Training convergence")
ax.legend(fontsize=9)
ax.grid(True, ls="--", alpha=0.4)

# ── Panel 2: Phase portrait (x₁ vs x₂, 20 random trajectories each) ────────
ax = axes[0, 1]
n_show = min(20, N_EVAL)
rng = np.random.default_rng(0)
idx = rng.choice(N_EVAL, n_show, replace=False)

for i in idx:
    ax.plot(x1_jfb[i], x2_jfb[i], color=COL_JFB, alpha=0.4, lw=0.8)
    ax.plot(x1_ad[i],  x2_ad[i],  color=COL_AD,  alpha=0.4, lw=0.8, ls="--")

# Draw one thick representative trajectory for each method.
rep = idx[0]
ax.plot(x1_jfb[rep], x2_jfb[rep], color=COL_JFB, lw=2, label="JFB-RL")
ax.plot(x1_ad[rep],  x2_ad[rep],  color=COL_AD,  lw=2, ls="--", label="Autodiff-BPTT")
ax.plot(0, 0, "k*", ms=10, label="target")
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.set_title("Phase portrait (20 test trajectories)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

# ── Panel 3: Control u(t) (mean ± p10/p90) ──────────────────────────────────
ax = axes[1, 0]
for u_arr, col, label in [(u_jfb, COL_JFB, "JFB-RL"), (u_ad, COL_AD, "Autodiff-BPTT")]:
    lo = np.percentile(u_arr, 10, axis=0)
    hi = np.percentile(u_arr, 90, axis=0)
    ax.fill_between(t_ctrl, lo, hi, color=col, alpha=ALF)
    ls = "-" if col == COL_JFB else "--"
    ax.plot(t_ctrl, u_arr.mean(axis=0), color=col, lw=2, ls=ls,
            label=f"{label} (mean ± p10/p90)")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("t"); ax.set_ylabel("u(t)")
ax.set_title("Control signal u(t)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

# ── Panel 4: Total cost distribution ────────────────────────────────────────
ax = axes[1, 1]
all_cost = np.concatenate([cost_jfb, cost_ad])
bins = np.linspace(all_cost.min() * 0.97, all_cost.max() * 1.03, 40)
for cost, col, label in [(cost_jfb, COL_JFB, "JFB-RL"),
                          (cost_ad,  COL_AD,  "Autodiff-BPTT")]:
    ax.hist(cost, bins=bins, color=col, alpha=0.5, density=True,
            label=f"{label}  (μ={cost.mean():.3f})")
    ax.axvline(cost.mean(), color=col, lw=1.5, ls="--")
ax.set_xlabel("Total cost J"); ax.set_ylabel("density")
ax.set_title("Cost distribution on test set")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

fig_path = os.path.join(out_dir, "comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {fig_path}")
plt.show()
