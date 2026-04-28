"""
core-RL.ImplicitNets_RL
-----------------------
RL-aware implicit policy network.

What changes vs ``core/ImplicitNets.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original :class:`ImplicitNetOC` builds its fixed-point operator ``T`` by
calling ``self.oc_problem.compute_grad_H_u(...)``, which in turn queries
``compute_grad_f_u`` — the **true** Jacobian of the dynamics w.r.t. the
control. In the RL setting that Jacobian is unknown and replaced by the
estimate ``b_k`` produced by a :class:`JacobianEstimator`.

We subclass :class:`ImplicitNetOC` and override exactly two things:

* ``T`` — the fixed-point step. Uses ``oc_problem.compute_grad_H_u_estimated``
  with ``self._current_b_k`` instead of ``compute_grad_H_u``.
* ``set_step_jacobian`` — a setter the trainer calls *before* each forward
  pass to tell the policy which ``b_k`` to use for the next FP iteration.

Every other piece of the original ``ImplicitNetOC`` — the FP loop with
optional Anderson acceleration, control limits, convergence tracking, and
the ``tracked_iters`` differentiable tail used by JFB — is reused
unchanged. That tail is what implements the "JFB inside the policy" half
of our pipeline; the ``compute_loss_RL`` surrogate in
:mod:`ImplicitOC_RL` implements the "JFB across the trajectory" half.

Stateful contract
~~~~~~~~~~~~~~~~~
Because we don't want to modify the parent class's ``forward`` signature
(it's referenced internally by Anderson, the FP loop, the convergence
tracker, ...), we pass ``b_k`` to ``T`` via instance state. The trainer
**must** call ``policy.set_step_jacobian(b_k)`` before each
``policy(z, t)`` invocation. Failing to do so will raise — there is no
safe default.
"""

from __future__ import annotations

import torch

# core/ is on sys.path; flat import works.
from core.ImplicitNets import ImplicitNetOC


class ImplicitNetOC_RL(ImplicitNetOC):
    """Implicit policy that uses an externally supplied ``b_k`` inside its
    fixed-point operator. Drop-in replacement for :class:`ImplicitNetOC`
    in the RL training pipeline.

    Parameters
    ----------
    Same as :class:`ImplicitNetOC`, with the added expectation that
    ``oc_problem`` is a :class:`ImplicitOC_RL` instance (i.e. it provides
    ``compute_grad_H_u_estimated`` rather than ``compute_grad_H_u``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sentinel value: ``None`` until the trainer pushes a real estimate.
        # We reach for a deliberately unhelpful error if T is called before
        # ``set_step_jacobian`` so misuse fails fast.
        self._current_b_k: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Setter the trainer / loss-routine calls before each step           #
    # ------------------------------------------------------------------ #
    def set_step_jacobian(self, b_k: torch.Tensor) -> None:
        """Set the local control Jacobian to be used by the next FP run.

        Expected shape: ``(B, control_dim, state_dim)`` *or*
        ``(1, control_dim, state_dim)`` (broadcastable from a
        shared-across-batch estimator).
        """
        self._current_b_k = b_k

    def clear_step_jacobian(self) -> None:
        """Reset to ``None`` — useful between epochs to surface bugs early."""
        self._current_b_k = None

    # ------------------------------------------------------------------ #
    # Override only T                                                    #
    # ------------------------------------------------------------------ #
    def T(self, u: torch.Tensor, x: torch.Tensor, t) -> torch.Tensor:
        """One gradient-ascent step on the **estimated** Hamiltonian.

        ``T̂_k(u; z) = u - α · ∇_u Ĥ(t, z, u, ∇_z φ_θ(t, z), b_k)``

        Same sign convention as the parent class.
        """
        if self._current_b_k is None:
            raise RuntimeError(
                "ImplicitNetOC_RL.T() called without a current b_k. "
                "The training loop must call policy.set_step_jacobian(b_k) "
                "before each policy(z, t) invocation."
            )

        batch_size = x.shape[0]
        t_scalar = torch.ones(1, device=x.device, dtype=x.dtype) * t
        assert x.shape == (batch_size, self.state_dim)

        # Costate: ∇_z φ_θ(t, z). The parent's ``Phi`` returns this
        # directly when called as ``Phi(t, z)`` (vs ``Phi.getPhi(t, z)``
        # which returns the scalar value).
        p = self.p_net(t, x)

        grad_H_u = self.oc_problem.compute_grad_H_u_estimated(
            t_scalar, x, u, p, self._current_b_k
        )
        assert grad_H_u.shape == u.shape, (
            f"compute_grad_H_u_estimated returned shape {tuple(grad_H_u.shape)}, "
            f"expected {tuple(u.shape)}"
        )

        # Sign convention identical to the parent: T(u) = u - α ∇_u H.
        return u - self.alpha * grad_H_u