"""
core-RL.ImplicitOC_RL
---------------------
Abstract base class for optimal-control problems in the **RL setting** —
i.e. when the system dynamics ``f`` are unknown to the agent.

Side-by-side comparison with ``core/ImplicitOC.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==========================================  =================================================
``core/ImplicitOC.py`` (known dynamics)     ``core-RL/ImplicitOC_RL.py`` (unknown dynamics)
==========================================  =================================================
abstract  ``compute_f``                     **REMOVED** — agent never sees f
abstract  ``compute_grad_f_u``              **REMOVED** — agent never sees ∂f/∂u
abstract  ``compute_grad_f_z``              **REMOVED** — agent never sees ∂f/∂z
abstract  ``compute_lagrangian``            kept (designer choice)
abstract  ``compute_grad_lagrangian``       kept (designer choice; w.r.t. u)
—                                            **NEW**: ``compute_grad_lagrangian_z``
                                                 (w.r.t. z, needed for backward adjoint;
                                                  default autograd-based)
abstract  ``compute_G``                     kept
abstract  ``compute_grad_G_z``              kept
abstract  ``sample_initial_condition``      kept
``compute_grad_H_u``                        replaced by **``compute_grad_H_u_estimated``**
                                                 which takes ``b_k`` instead of querying
                                                 ``compute_grad_f_u``
``compute_loss``                            replaced by **``compute_loss_RL``** which
                                                 routes the rollout through ``env.step``,
                                                 updates the Jacobian estimator, runs the
                                                 data-driven backward adjoint, and returns
                                                 a JFB-with-estimates surrogate scalar
                                                 with the property that
                                                 ``surrogate.backward()`` produces the
                                                 RL-flavoured JFB gradient
``alphaHJB`` / ``alphaadj``                 **REMOVED** — those penalties depend on f;
                                                 they re-appear, if at all, only after we
                                                 add a learned model in Step 3.
==========================================  =================================================

Sign / shape conventions
~~~~~~~~~~~~~~~~~~~~~~~~
We follow the existing ``core/`` repo:

* Hamiltonian (minimisation form):  ``H = L + ⟨p, f⟩``.
* ``∇_u H = ∇_u L + b_k @ p`` where ``b_k`` has shape ``(B, m, n)`` with
  ``b_k[:, i, j] = ∂f_j / ∂u_i``  (so ``b_k @ p`` has shape ``(B, m)``).
* Fixed-point step: ``T̂_k(u; z) = u - α ∇_u H``.

With this convention, the JFB-with-estimates gradient is

    dĴ/dθ ≈ Σ_k (∂T̂_k / ∂θ)ᵀ · [ Δt · ∇_u L + b_k @ p_{k+1} · Δt ]
           = Σ_k (∂T̂_k / ∂θ)ᵀ · Δt · [ ∇_u L + b_k @ p_{k+1} ].

The data-driven discrete adjoint, with continuous-time ``a_k = ∂f/∂z``, is

    p_k = p_{k+1} + Δt · ( a_kᵀ @ p_{k+1} + ∇_z L ),    p_N = ∇G(z_N).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from Environment import Environment
from JacobianEstimator import JacobianEstimator


TimeLike = float | torch.Tensor


class ImplicitOC_RL(ABC):
    """Abstract optimal-control problem with **unknown** dynamics.

    Concrete subclasses must implement only the *designer-side* pieces:
    running cost, terminal cost, and an initial-condition sampler. The
    dynamics ``f`` belong to the :class:`Environment`, not to this class.

    Parameters
    ----------
    state_dim, control_dim
    batch_size            initial-condition batch size (forwarded to
                          ``sample_initial_condition``).
    t_initial, t_final, nt
    alphaL, alphaG        weights on running and terminal cost (kept from
                          the original ``ImplicitOC`` so the loss stays in
                          comparable units).
    device
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        batch_size: int,
        t_initial: float,
        t_final: float,
        nt: int,
        alphaL: float = 1.0,
        alphaG: float = 1.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.batch_size = batch_size
        self.t_initial = t_initial
        self.t_final = t_final
        self.nt = nt
        self.h = (t_final - t_initial) / nt
        self.alphaL = alphaL
        self.alphaG = alphaG
        self.device = device

        self.oc_problem_name = "Generic ImplicitOC_RL"

    # ================================================================== #
    # Abstract: designer-side knowns                                     #
    # ================================================================== #
    @abstractmethod
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Running cost ``L(t, z, u)``. Shape ``(B,)``."""

    @abstractmethod
    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∇_u L``. Shape ``(B, control_dim)``."""

    def compute_grad_lagrangian_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∇_z L``, used in the backward adjoint pass.

        **Default**: autograd-based. Concrete problems whose ``L`` is
        independent of ``z`` (e.g. our Merton example, where the running
        cost depends only on the portfolio weight ``π``) can leave this
        unchanged — autograd will return zeros automatically. Problems
        with a non-trivial ``z``-dependence (e.g. consumption-savings
        with an inventory-risk term ``½ σ² q²``) should override this with
        the analytical formula for speed and numerical stability.
        """
        z_req = z.detach().requires_grad_(True)
        L = self.compute_lagrangian(t, z_req, u).sum()
        (grad,) = torch.autograd.grad(L, z_req, create_graph=False)
        return grad.detach()

    @abstractmethod
    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """Terminal cost ``G(z(T))``. Shape ``(B,)``."""

    @abstractmethod
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """``∇G``. Shape ``(B, state_dim)``."""

    @abstractmethod
    def sample_initial_condition(self) -> torch.Tensor:
        """Draw a batch of initial states ``z_0 ∈ R^{B × state_dim}``."""

    # ================================================================== #
    # The Hamiltonian gradient — using estimated b_k                     #
    # ================================================================== #
    def compute_grad_H_u_estimated(
        self,
        t: TimeLike,
        z: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        b_k: torch.Tensor,
    ) -> torch.Tensor:
        """``∇_u H`` using estimated control Jacobian ``b_k`` in place of
        the true ``∂f/∂u``.

        Parameters
        ----------
        t : scalar or 0-d tensor
        z : ``(B, state_dim)``
        u : ``(B, control_dim)``
        p : ``(B, state_dim)``  — costate (typically ``∇_z φ_θ(t, z)``)
        b_k : ``(B, m, n)`` *or* ``(1, m, n)`` (broadcastable from a
              shared-across-batch estimator like RLS).

        Returns
        -------
        ``(B, control_dim)``.

        This is the RL counterpart of
        ``ImplicitOC.compute_grad_H_u`` — same shape, same sign — but with
        ``b_k`` injected from the outside instead of ``self.compute_grad_f_u(...)``.
        """
        B = z.shape[0]
        # ∇_u L
        grad_L = self.compute_grad_lagrangian(t, z, u)         # (B, m)
        # b_k @ p — broadcast b_k if it's (1, m, n)
        if b_k.shape[0] == 1 and B > 1:
            b_k = b_k.expand(B, -1, -1)
        grad_pf = torch.bmm(b_k, p.unsqueeze(-1)).view(B, self.control_dim)  # (B, m)
        return grad_L + grad_pf

    # ================================================================== #
    # The RL training-loss routine                                       #
    # ================================================================== #
    def compute_loss_RL(
        self,
        policy,                       # ImplicitNetOC_RL
        env: Environment,
        jac_est: JacobianEstimator,
        z0: torch.Tensor,
    ) -> dict:
        """End-to-end RL JFB loss computation.

        Pipeline (matches Algorithm 1 of the Pontryagin-RL notes):

        1. **Forward rollout in env**, collecting ``z̄_{0:N}``, ``ū_{0:N-1}``.
           Inside the loop we (a) push the current ``b_k`` estimate into
           the policy via ``policy.set_step_jacobian(b_k)``, (b) call the
           policy to obtain ``ū_k``, (c) step the env, (d) update the
           Jacobian estimator with the freshly observed transition. All
           tensors collected here are detached — no autograd graph runs
           through the env.

        2. **Backward adjoint pass** (no grad):
               p_N = ∇G(z_N)
               p_k = p_{k+1} + Δt (a_kᵀ p_{k+1} + ∇_z L_k)

        3. **JFB surrogate construction**: a scalar ``S(θ)`` such that
           ``S(θ).backward()`` populates ``param.grad`` with the
           JFB-with-estimates gradient. Specifically,

               S(θ) = Σ_k ⟨ T̂_k(ū_k; z̄_k),  Δt · (∇_u L + b_k @ p̂_{k+1}) ⟩

           where ``ū_k``, the bracket, ``b_k``, ``p̂_{k+1}`` are all
           detached, and ``T̂_k`` carries θ-dependence only through the
           value-function gradient ``∇_z φ_θ`` it consumes inside ``∇_u H``.
           Then  ``∂S/∂θ = Σ_k (∂T̂_k/∂θ)ᵀ · bracket_k``  by linearity,
           which is exactly eqn. (23). This trick avoids ever having to
           compute or store ``∂T̂/∂θ`` as a Jacobian.

        4. **Loss reporting**: separately compute the actual scalar
           cost ``J = α_L Σ L Δt + α_G G(z_N)`` (no grad) for monitoring.
           This is what we log; the surrogate is what we backward.

        Returns
        -------
        dict with keys
          ``surrogate``       — scalar tensor with autograd graph; trainer
                                calls ``.backward()`` on this.
          ``total_cost``      — float, value of ``J`` (running + terminal).
          ``running_cost``    — float, ``Σ L Δt``.
          ``terminal_cost``   — float, ``G(z_N)``.
          ``z_traj``          — ``(B, n, nt+1)``, detached.
          ``u_traj``          — ``(B, m, nt)``, detached.
          ``p_traj``          — ``(B, n, nt+1)``, detached costate sequence.
          ``lin_residual``    — float, mean linear-model residual averaged
                                over time steps (estimator diagnostic).
        """
        device = z0.device
        B = z0.shape[0]
        n, m = self.state_dim, self.control_dim
        N = self.nt
        dt = self.h

        # -------- 1. Forward rollout (no grad) ------------------------- #
        z_traj = torch.zeros(B, n, N + 1, device=device, dtype=z0.dtype)
        u_traj = torch.zeros(B, m, N, device=device, dtype=z0.dtype)
        z_traj[:, :, 0] = z0
        z = z0.detach()
        t = self.t_initial

        running_cost_acc = torch.zeros(B, device=device, dtype=z0.dtype)
        lin_residual_acc = 0.0

        with torch.no_grad():
            for k in range(N):
                # Push the *current* b_k into the policy before its FP iteration.
                _, b_k = jac_est.AB(k)
                policy.set_step_jacobian(b_k)

                u_k = policy(z, t).view(B, m)
                z_next = env.step(z, u_k, t)

                # Update the estimator with this freshly observed transition.
                jac_est.update(k, z, u_k, z_next)
                lin_residual_acc += jac_est.linear_model_residual(k, z, u_k, z_next).item()

                # Running cost (analytical L; the agent knows L by assumption).
                running_cost_acc = running_cost_acc + dt * self.compute_lagrangian(t, z, u_k)

                u_traj[:, :, k] = u_k
                z_traj[:, :, k + 1] = z_next
                z = z_next
                t = t + dt

            terminal_cost_per_sample = self.compute_G(z)  # (B,)

        # -------- 2. Backward adjoint pass (no grad) ------------------- #
        p_traj = torch.zeros(B, n, N + 1, device=device, dtype=z0.dtype)
        with torch.no_grad():
            p_kp1 = self.compute_grad_G_z(z_traj[:, :, N])    # (B, n)
            p_traj[:, :, N] = p_kp1
            t_back = self.t_initial + (N - 1) * dt
            for k in range(N - 1, -1, -1):
                z_k = z_traj[:, :, k]
                u_k = u_traj[:, :, k]
                a_k, _ = jac_est.AB(k)
                if a_k.shape[0] == 1 and B > 1:
                    a_k = a_k.expand(B, -1, -1)
                # a_kᵀ @ p_{k+1}: (B, n, n)ᵀ @ (B, n, 1) -> (B, n, 1) -> (B, n)
                aT_p = torch.bmm(a_k.transpose(1, 2), p_kp1.unsqueeze(-1)).squeeze(-1)
                grad_z_L = self.compute_grad_lagrangian_z(t_back, z_k, u_k)
                p_k = p_kp1 + dt * (aT_p + grad_z_L)
                p_traj[:, :, k] = p_k
                p_kp1 = p_k
                t_back = t_back - dt

        # -------- 3. JFB surrogate (autograd through phi only) --------- #
        # Important: the policy was used WITHOUT grad in step 1, so its
        # internal FP iteration cached fixed points without a graph. To
        # build the JFB surrogate we re-evaluate T̂_k once per step using
        # the *detached* converged controls — only the value-function
        # gradient that appears inside T̂ carries θ-dependence, which is
        # exactly the JFB approximation.
        surrogate = torch.zeros((), device=device, dtype=z0.dtype)
        for k in range(N):
            t_k = self.t_initial + k * dt
            z_k = z_traj[:, :, k].detach()
            u_k = u_traj[:, :, k].detach()
            _, b_k = jac_est.AB(k)
            b_k_det = b_k.detach()
            p_kp1_det = p_traj[:, :, k + 1].detach()

            # ∇_u L at (t, z̄, ū) — no θ-dependence; safe to leave attached
            # but we detach for clarity.
            grad_uL = self.compute_grad_lagrangian(t_k, z_k, u_k).detach()

            # The θ-dependent piece: ∇_z φ_θ(t, z̄_k). policy.p_net is the
            # value-function network; passing requires_grad=False z_k still
            # lets the gradient flow through θ.
            p_phi = policy.p_net(t_k, z_k)                # (B, n)

            # ∇_u H at (z̄, ū, ∇_z φ_θ, b_k) — only p_phi is θ-dependent.
            if b_k_det.shape[0] == 1 and B > 1:
                b_k_b = b_k_det.expand(B, -1, -1)
            else:
                b_k_b = b_k_det
            grad_uH = grad_uL + torch.bmm(b_k_b, p_phi.unsqueeze(-1)).view(B, m)

            # T̂_k(ū_k; z̄_k) — with the existing repo's sign convention.
            T_k = u_k - policy.alpha * grad_uH            # (B, m)

            # Bracket — fully detached.
            bracket = dt * (grad_uL + torch.bmm(b_k_b.detach(), p_kp1_det.unsqueeze(-1)).view(B, m))
            bracket = bracket.detach()

            # Inner product, mean over batch (matches the existing
            # mean-over-batch convention in compute_loss).
            surrogate = surrogate + (T_k * bracket).sum(dim=1).mean()

        # Add the parts of J that depend on θ ONLY through the rollout —
        # for the JFB approximation those don't contribute to the gradient
        # (the rollout is detached), so we add them to ``surrogate`` as
        # detached scalars *only* for logging convenience. The trainer
        # backwards ``surrogate`` for the gradient and reports
        # ``total_cost`` for the loss curve.
        running_cost_mean = running_cost_acc.mean().item()
        terminal_cost_mean = terminal_cost_per_sample.mean().item()
        total_cost = self.alphaL * running_cost_mean + self.alphaG * terminal_cost_mean

        return {
            "surrogate": surrogate,
            "total_cost": total_cost,
            "running_cost": running_cost_mean,
            "terminal_cost": terminal_cost_mean,
            "z_traj": z_traj.detach(),
            "u_traj": u_traj.detach(),
            "p_traj": p_traj.detach(),
            "lin_residual": lin_residual_acc / N,
        }

    # ================================================================== #
    # Convenience: deterministic rollout for plotting                    #
    # ================================================================== #
    def generate_trajectory(
        self,
        policy,
        env: Environment,
        z0: torch.Tensor,
        return_full_trajectory: bool = True,
    ) -> torch.Tensor:
        """Stand-in for the original ``generate_trajectory`` that uses
        ``env.step`` instead of ``compute_f``. The trainer's plotting
        dispatch calls this instead of going through the model class's
        analytical rollout.
        """
        z_traj, _ = env.rollout(policy, z0, return_full_trajectory=return_full_trajectory)
        return z_traj