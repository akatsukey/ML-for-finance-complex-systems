from __future__ import annotations

from typing import List

import numpy as np
import torch

from ImplicitOC import ImplicitOC, TimeLike
from utils import GradientTester
from benchmarking import Trajectory
from benchmarking.plotter import Panel, liquidation_panels

class LiquidationPortfolioOC(ImplicitOC):
    """
    Single-asset liquidation as a finite-horizon optimal control problem.

    State ``z = [q, S, X]``: remaining inventory, impacted execution price, and
    accumulated cash. Control ``u`` is the scalar trading (selling) rate.

    Continuous-time dynamics (right-hand side of the controlled ODE):

        dq/dt = -u
        dS/dt = -kappa * u
        dX/dt = S*u - eta * (u^2 + epsilon)^(gamma/2)

    Running cost (Lagrangian): ``0.5 * sigma^2 * q^2`` — inventory-risk penalty
    while liquidating. Terminal cost: ``G = -X(T) + alpha * q(T)^2`` — reward
    terminal cash via ``-X``, penalize leftover inventory via ``alpha * q^2``.
    Minimizing the objective therefore discourages carrying risk during the
    trade, pushes toward higher terminal proceeds, and discourages unfinished
    liquidation (large ``q(T)``).
    """

    def __init__(
        self,
        batch_size=64,
        t_initial=0.0,
        t_final=2.0,
        nt=100,
        n_assets=2,
        sigma=(0.02, 0.02),
        kappa=(1.0e-4, 1.0e-4),
        eta=(0.1, 0.1),
        gamma=2.0,
        epsilon=1.0e-2,
        alpha=30,
        q0_min=(0.5, 0.5),
        q0_max=(1.5, 1.5),
        S0=(1.0, 1.0),
        X0=0.0,
        device="cpu",
        alphaHJB=(0.0, 0.0),
        alphaadj=(0.0, 0.0),
    ):
        # State layout: q = remaining inventory, S = impacted price, X = accumulated cash.
        # Control: u = selling rate (aligned with dq/dt = -u).
        state_dim = 2 * n_assets + 1  # (q1, q2,..., S1, S2,..., X) 
        control_dim = n_assets  # (u1, u2, ...)
        # Time discretization and batch: horizon [t_initial, t_final], nt steps, parallel trajectories.
        super().__init__(
            state_dim,
            control_dim,
            batch_size,
            t_initial,
            t_final,
            nt,
            alphaL=1.0,
            alphaG=1.0,
            alphaHJB=list(alphaHJB),
            alphaadj=list(alphaadj),
            device=device,
        )
        self.oc_problem_name = "Liquidation Portfolio"

        self.n_assets = n_assets

        # Terminal-impact smoothing in (u^2 + epsilon)^(gamma/2); also used in dX/dt.
        self.epsilon = epsilon

        

        # function to convert scalar or vector parameters into asset-aligned vectors of shape (n_assets,) on the correct device
        def _to_asset_vector(x, n_assets, device, name):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            if x_t.ndim == 0:
                x_t = x_t.repeat(n_assets)
            elif x_t.ndim == 1 and x_t.numel() == n_assets:
                pass
            else:
                raise ValueError(f"{name} must be a scalar or a vector of length {n_assets}")
            return x_t

        
        
        # # Market model: inventory risk scale sigma; linear price impact kappa; nonlinear cash friction eta, gamma.
        # self.sigma = sigma
        # self.kappa = kappa
        # self.eta = eta
        self.gamma = gamma
        self.sigma = _to_asset_vector(sigma, n_assets, device, "sigma")
        self.kappa = _to_asset_vector(kappa, n_assets, device, "kappa")
        self.eta = _to_asset_vector(eta, n_assets, device, "eta")


        # Terminal penalty weight on leftover inventory (with -X term in G).
        self.alpha = alpha



        # Initial-condition distribution / levels for sampling z0 at episode start.
        # self.q0_min = q0_min
        # self.q0_max = q0_max
        # self.S0 = S0
        self.q0_min = _to_asset_vector(q0_min, n_assets, device, "q0_min")
        self.q0_max = _to_asset_vector(q0_max, n_assets, device, "q0_max")
        self.S0 = _to_asset_vector(S0, n_assets, device, "S0")
        self.X0 = X0


    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Running inventory-risk penalty (Lagrangian).

        ``L = 0.5 * sigma^2 * q^2`` depends only on inventory ``q``, not directly on
        the control ``u``. Returns one scalar per batch element, shape ``(batch,)``,
        or a scalar when a single unbatched state/control pair is passed (see branch
        below) for ``vmap``/Jacobian checks.

        Args:
            t: Current time (unused in this running cost).
            z: State, shape ``(batch, state_dim)`` or ``(state_dim,)`` for single-sample calls.
            u: Control, shape ``(batch, control_dim)`` or ``(control_dim,)`` (unused in L).

        Returns:
            Tensor of shape ``(batch,)`` or 0-dim when the unbatched branch strips the batch dim.
        """

        # Single-sample path: normalize one (state_dim,) / (control_dim,) pair to batch shape (1, ...)
        # so downstream indexing z[:, 0] stays valid; autograd + vmap use this layout.
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q = z[:, :self.n_assets]  # inventory, shape (batch, n_assets)
        # Carrying inventory is risky; larger sigma amplifies the quadratic penalty on q.
        lag = 0.5 * torch.sum((self.sigma**2) * (q**2), dim=1)  # shape (batch,) — identity covariance: ½ qᵀ diag(σ²) q
        return lag[0] if squeeze else lag

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Partial derivative of the running cost with respect to control: ``dL/du``.

        Here ``L`` does not depend on ``u``, so ``dL/du`` is identically zero. Along
        trajectories, the Hamiltonian control signal still comes from the dynamics
        term ``p^T f`` via ``compute_grad_H_u`` in the trainer / implicit layer, not
        from ``dL/du``.

        Args:
            t: Current time.
            z: State, batched or unbatched like ``compute_lagrangian``.
            u: Control; same shape as the policy output, ``(batch, control_dim)``.

        Returns:
            Zero tensor with the **same shape as ``u``** (required by OC interface).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Match control shape (batch, control_dim) exactly for adjoint / consistency checks.
        grad = torch.zeros_like(u)

        return grad[0] if squeeze else grad

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Right-hand side ``f(t, z, u) = dz/dt`` of the controlled ODE.

        Returns the time derivative of the state with shape ``(batch, state_dim)``.
        This tensor is what explicit Euler multiplies by ``dt`` when rolling trajectories
        forward in ``generate_trajectory``.

        Args:
            t: Current time.
            z: State ``[q, S, X]``, shape ``(batch, 3)`` or ``(3,)``.
            u: Selling rate, shape ``(batch, 1)`` or ``(1,)``.

        Returns:
            ``dz/dt``, shape ``(batch, state_dim)`` (or unbatched equivalent when the squeeze branch applies).
        """

        # Same single-sample → batch normalization as in compute_lagrangian (vmap / jacrev callers).
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Column slices (batch, 1): componentwise dynamics, concatenated into (batch, 3) below.
        q = z[:, :self.n_assets]  # inventory (batch, n_assets)
        S = z[:, self.n_assets:2*self.n_assets]  # impacted price (batch, n_assets)
        X = z[:, 2*self.n_assets:2*self.n_assets+1]  # accumulated cash (batch, 1)

        dq = -u  # selling reduces remaining inventory q
        dS = -self.kappa * u  # linear permanent impact: selling depresses the mid / impacted price S
        # dX = S * u - self.eta * torch.abs(u).pow(self.gamma)
        # Cash: revenue S*u minus smoothed nonlinear impact/friction in the selling rate u.
        trading_cashflow = S * u
        impact_cost = self.eta * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0)
        dX = torch.sum(trading_cashflow - impact_cost, dim=1, keepdim=True)  # (batch, 1)

        result = torch.cat((dq, dS, dX), dim=1)  # stack dq/dt, dS/dt, dX/dt into dz/dt
        return result[0] if squeeze else result

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Jacobian ``∂f/∂u`` of the dynamics with respect to the control.

        Tensor layout: shape ``(batch, control_dim, state_dim)``. With a single scalar
        control, row ``[:, 0, :]`` is the gradient of each state equation
        ``(dq/dt, dS/dt, dX/dt)`` with respect to the trading rate ``u``.

        Args:
            t: Current time.
            z: State, shape ``(batch, state_dim)``.
            u: Control, shape ``(batch, control_dim)``.

        Returns:
            ``(batch, control_dim, state_dim)`` Jacobian; here ``(batch, 1, 3)``.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        # One Jacobian "row" per control component × one column per state equation (q, S, X).
        grad = torch.zeros(batch, self.control_dim,self.state_dim, device=z.device)

        S = z[:,  self.n_assets:2*self.n_assets]

        for i in range(self.n_assets):
            grad[:, i, i] = -1.0                                                # ∂(dq_i/dt)/∂u_i = -1
            grad[:, i, self.n_assets + i] = -self.kappa[i]                      # ∂(dS_i/dt)/∂u_i = -kappa_i


        impact_grad = (self.eta * self.gamma* u* (u.pow(2) + self.epsilon).pow(self.gamma / 2.0 - 1.0))  # derivative of eta*(u^2+eps)^(gamma/2) w.r.t. u (smooth impact term in dX/dt)
        grad[:, :, 2 * self.n_assets] = S - impact_grad # ∂(dX/dt)/∂u: marginal cash change per unit increase in selling rate

        return grad[0] if squeeze else grad

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Jacobian ``∂f/∂z`` of the dynamics with respect to the state.

        Shape ``(batch, state_dim, state_dim)``.  **Layout convention** matches
        :meth:`compute_grad_f_u` and what :meth:`ImplicitOC.compute_grad_H_z`
        consumes via ``bmm(grad, p)``: the **first** index is the state-derivative
        axis (``∂/∂z_i``), the **second** index is the dynamics-component axis
        (``f_s``).  In other words ``grad[b, i, s] == ∂f_s/∂z_i`` — the
        *transpose* of the natural mathematical Jacobian, exactly as
        ``torch.bmm(grad, p)`` requires to produce ``∂(p^T f)/∂z``.

        Most entries vanish: only ``dX/dt`` depends on ``S`` (through
        ``Σⱼ S_j u_j``), so ``∂f_X/∂z_{S_j} = u_j`` lives at slot
        ``[i = n_assets + j, s = 2 * n_assets]`` (for n_assets=1 this
        reduces to ``[i=S=1, s=X=2]``).

        Args:
            t: Current time.
            z: State, shape ``(batch, state_dim)``.
            u: Control, shape ``(batch, control_dim)``.

        Returns:
            ``(batch, state_dim, state_dim)`` Jacobian in transposed convention.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        # Square Jacobian (batch, 3, 3): columns index ∂/∂q, ∂/∂S, ∂/∂X; rows index state equations.
        grad = torch.zeros(batch, self.state_dim, self.state_dim, device=z.device)

        # ∂f_X/∂z_{S_j} = u_j: only dX/dt depends on each S_j (through S_j*u_j),
        # so we write u into the column s=X for the rows i=S_1,...,S_n.
        grad[:, self.n_assets:2 * self.n_assets, 2 * self.n_assets] = u

        return grad[0] if squeeze else grad

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Terminal cost ``G(z(T))`` (discounting handled in the base OC if applicable).

        ``-X`` rewards terminal cash (larger X lowers G). ``alpha * q^2`` penalizes
        leftover inventory at the horizon. Minimizing G pushes the policy toward
        higher terminal proceeds while completing liquidation (small ``q(T)``).

        Args:
            z: Terminal state, shape ``(batch, state_dim)`` or ``(state_dim,)``.

        Returns:
            Scalar terminal cost per trajectory, shape ``(batch,)`` or 0-dim when unbatched.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q = z[:, :self.n_assets]  # inventory (batch, n_assets)
        X = z[:, 2*self.n_assets]  # accumulated cash (batch,)

        G = -X + self.alpha * torch.sum(q ** 2, dim=1)  # cash reward vs. unfinished liquidation penalty
        return G[0] if squeeze else G

    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient ``∂G/∂z`` of the terminal cost with respect to the terminal state.

        Supplies the terminal condition for the adjoint / costate equation in
        Pontryagin-type training. Shape ``(batch, state_dim)``.

        Args:
            z: Terminal state, shape ``(batch, state_dim)`` or ``(state_dim,)``.

        Returns:
            ``(batch, state_dim)`` gradient (or unbatched 1D slice when squeeze applies).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, self.state_dim, device=z.device)

        q = z[:, :self.n_assets]
        grad[:, :self.n_assets] = 2.0 * self.alpha * q  # marginal cost of carrying one more unit of q at T
        grad[:, 2 * self.n_assets] = -1.0  # marginal value of terminal cash: +1 to X decreases G by 1

        return grad[0] if squeeze else grad

    def sample_initial_condition(self):
        """
        Sample initial states for training rollouts.

        Initial inventory ``q0`` is uniform on ``[q0_min, q0_max]``; ``S0`` and ``X0``
        are fixed constants across the batch. The policy is therefore trained against
        a **family** of initial inventories, not a single deterministic ``z0``.
        """
        q0 = self.q0_min.unsqueeze(0) + (self.q0_max - self.q0_min).unsqueeze(0) * torch.rand(self.batch_size, self.n_assets, device=self.device)

        S0 = self.S0.unsqueeze(0).expand(self.batch_size, -1)
        X0 = torch.full((self.batch_size, 1), self.X0, device=self.device)
        # Concatenate along feature dim → shape (batch, 3) == (batch, state_dim).
        return torch.cat((q0, S0, X0), dim=1).to(self.device)

    # ------------------------------------------------------------------
    # Legacy plotting (kept for reference, superseded by panels/to_trajectory)
    # ------------------------------------------------------------------
    # def plot_position_trajectories(
    #     self,
    #     z_traj: torch.Tensor,
    #     policy=None,
    #     save_path: str | None = None,
    #     n_show: int = 5,
    #     title_str: str = "Liquidation policy rollout",
    # ):
    #     """
    #     Plot ``q(t), u(t), S(t), X(t)`` for the first ``n_show`` trajectories of
    #     ``z_traj`` (shape ``(batch, state_dim, nt+1)``).
    #
    #     This method matched the calling convention of
    #     :meth:`OptimalControlTrainer.train`, which invoked::
    #
    #         self.oc_problem.plot_position_trajectories(z_traj.detach(), self.policy)
    #
    #     every ``plot_frequency`` epochs. ``policy`` is optional: when supplied the
    #     trading rate ``u(t)`` is reconstructed by evaluating ``policy(z, t)`` along
    #     the rolled-out state trajectory (exactly what ``generate_trajectory`` did
    #     during the Euler march), otherwise the control panel is left empty.
    #
    #     ``save_path`` defaults to an auto-numbered PNG under
    #     ``results_<class_name>/standard_mode/plots/`` so mid-training snapshots
    #     accumulate rather than overwrite each other. The figure is always closed
    #     after writing, so the call never blocks training.
    #     """
    #     import os
    #     import matplotlib.pyplot as plt
    #
    #     z_traj = z_traj.detach()
    #     batch, _, nt1 = z_traj.shape
    #     nt = nt1 - 1
    #     n_show = max(1, min(n_show, batch))
    #
    #     t = torch.linspace(self.t_initial, self.t_final, nt1).cpu().numpy()
    #
    #     q = z_traj[:n_show, 0, :].cpu().numpy()
    #     S = z_traj[:n_show, 1, :].cpu().numpy()
    #     X = z_traj[:n_show, 2, :].cpu().numpy()
    #
    #     u_arr = None
    #     if policy is not None:
    #         dt = (self.t_final - self.t_initial) / nt
    #         u_buf = torch.zeros(n_show, self.control_dim, nt, device=z_traj.device)
    #         with torch.no_grad():
    #             for i in range(nt):
    #                 z_i = z_traj[:n_show, :, i]
    #                 t_i = self.t_initial + i * dt
    #                 u_buf[:, :, i] = policy(z_i, t_i).view(n_show, self.control_dim)
    #         u_arr = u_buf[:, 0, :].cpu().numpy()
    #
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    #     for b in range(n_show):
    #         axes[0, 0].plot(t, q[b], alpha=0.75, label=f"traj {b}" if n_show > 1 else None)
    #         axes[1, 0].plot(t, S[b], alpha=0.75)
    #         axes[1, 1].plot(t, X[b], alpha=0.75)
    #         if u_arr is not None:
    #             axes[0, 1].plot(t[:-1], u_arr[b], alpha=0.75)
    #
    #     axes[0, 0].set_title("Inventory  q(t)")
    #     axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("q"); axes[0, 0].grid(True)
    #     axes[0, 1].set_title("Trading rate  u(t)")
    #     axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel("u"); axes[0, 1].grid(True)
    #     axes[1, 0].set_title("Impacted price  S(t)")
    #     axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel("S"); axes[1, 0].grid(True)
    #     axes[1, 0].ticklabel_format(style="plain", axis="y", useOffset=False)
    #     axes[1, 1].set_title("Cash  X(t)")
    #     axes[1, 1].set_xlabel("t"); axes[1, 1].set_ylabel("X"); axes[1, 1].grid(True)
    #
    #     if n_show > 1:
    #         axes[0, 0].legend(fontsize=8, loc="best")
    #
    #     fig.suptitle(title_str)
    #     fig.tight_layout(rect=[0, 0, 1, 0.96])
    #
    #     if save_path is None:
    #         if not hasattr(self, "_plot_counter"):
    #             self._plot_counter = 0
    #         self._plot_counter += 1
    #         from core.paths import results_dir
    #         plot_dir = results_dir(type(self).__name__, "training", "training-plots")
    #         save_path = os.path.join(plot_dir, f"rollout_{self._plot_counter:04d}.png")
    #
    #     fig.savefig(save_path, dpi=150, bbox_inches="tight")
    #     plt.close(fig)
    #     print(f"    -> saved rollout figure to {os.path.abspath(save_path)}")

    # ------------------------------------------------------------------
    # BenchmarkPlotter integration
    # ------------------------------------------------------------------
    def panels(self) -> List[Panel]:
        """Multi-asset Almgren-Chriss panel layout (``q_i, u*_i, S_i`` per
        asset, plus a shared ``X(t)``).

        Consumed by :class:`benchmarking.BenchmarkPlotter` (see the trainer's
        plotting dispatch). For ``n_assets == 1`` this collapses to the
        legacy four-panel ``q, u*, S, X`` layout (same indices, just with no
        subscript in the titles).
        """
        return liquidation_panels(self.n_assets)

    def to_trajectory(
        self,
        z_traj: torch.Tensor,
        policy=None,
        path_index: int = 0,
        label: str = "JFB",
    ) -> Trajectory:
        """Pack a rolled-out tensor into a :class:`benchmarking.Trajectory`.

        Parameters
        ----------
        z_traj
            Output of :meth:`generate_trajectory(..., return_full_trajectory=True)`,
            shape ``(batch, state_dim, nt+1)``.
        policy
            Optional callable ``policy(z, t) -> u``. If supplied, the control
            ``u(t)`` is reconstructed by evaluating the policy along the
            selected trajectory exactly as :meth:`generate_trajectory` does
            during the forward Euler march.
        path_index
            Which sample of the batch to package (default ``0``). The returned
            trajectory is *deterministic* in the benchmarking sense: shape
            ``(N, state_dim)`` for ``z`` and ``(N-1, control_dim)`` for ``u``.
        label
            Legend label propagated into the resulting figure.
        """
        z_traj = z_traj.detach()
        batch, state_dim, nt1 = z_traj.shape
        if not 0 <= path_index < batch:
            raise IndexError(f"path_index={path_index} out of range for batch={batch}")
        nt = nt1 - 1

        t_np = np.linspace(self.t_initial, self.t_final, nt1)
        z_np = z_traj[path_index].transpose(0, 1).cpu().numpy()  # (nt+1, state_dim)

        u_np = None
        if policy is not None:
            dt = (self.t_final - self.t_initial) / nt
            u_buf = torch.zeros(self.control_dim, nt, device=z_traj.device)
            with torch.no_grad():
                z_path = z_traj[path_index : path_index + 1]  # keep batch axis = 1
                for i in range(nt):
                    t_i = self.t_initial + i * dt
                    u_i = policy(z_path[:, :, i], t_i).view(1, self.control_dim)
                    u_buf[:, i] = u_i[0]
            u_np = u_buf.transpose(0, 1).cpu().numpy()  # (nt, control_dim)

        return Trajectory(
            t=t_np,
            z=z_np,
            u=u_np,
            label=label,
            style={"color": "#d6604d", "lw": 2.0},
        )

    def generate_trajectory(self, u, z0, nt, return_full_trajectory=False):
        """
        Forward-simulate the controlled ODE with **explicit Euler** time stepping.

        **State layout.** ``z0`` has shape ``(batch, state_dim)`` with ``state_dim = 3``.
        The trajectory buffer ``traj`` has shape ``(batch, state_dim, nt + 1)``:
        ``traj[:, :, 0]`` is the initial condition; index ``i`` holds the state after
        ``i`` Euler steps (consistent with control index ``i`` used below).

        **Control input.** ``u`` may be either:

        1. A tensor of shape ``(batch, control_dim, nt)`` giving the open-loop control
           at each discrete time index ``i = 0, ..., nt-1``, or
        2. A callable ``u(z, t)`` implementing a feedback policy: must return controls
           of shape ``(batch, control_dim)`` for the current state ``z`` and time ``t``.

        **Update.** With uniform step ``dt = (t_final - t_initial) / nt``,

            z_{i+1} = z_i + dt * f(t_i, z_i, u_i).

        **Return value.** If ``return_full_trajectory`` is False, returns only the
        terminal state ``traj[:, :, -1]`` of shape ``(batch, state_dim)``; otherwise
        returns the full path ``traj``.
        """
        batch = z0.shape[0]
        D = self.state_dim

        # Preallocate full path so time index i is always traj[:, :, i] (and controls align with i).
        traj = torch.zeros(batch, D, nt + 1, device=z0.device)
        traj[:, :, 0] = z0  # initial condition at discrete time index 0
        dt = (self.t_final - self.t_initial) / nt  # uniform Euler step over the horizon
        t = self.t_initial

        # March forward: one explicit Euler step per loop iteration.
        for i in range(nt):
            if torch.is_tensor(u):
                curr = u[:, :, i]  # open-loop: use control already stored for this time index
            else:
                curr = u(traj[:, :, i], t)  # closed-loop: policy evaluated at current state and time
            traj[:, :, i + 1] = traj[:, :, i] + dt * self.compute_f(
                t, traj[:, :, i], curr
            )  # Euler discretization of dz/dt = f(t, z, u)
            t += dt
        # Either the entire state trajectory or only the terminal layer z(T).
        return traj if return_full_trajectory else traj[:, :, -1]


# Example usage
if __name__ == "__main__":

    # Local smoke test: derivative consistency (analytical vs autograd references).
    device = "cpu"
    batch_size = 10
    nt = 100

    prob = LiquidationPortfolioOC(
        batch_size=batch_size,
        t_initial=0.0,
        t_final=2.0,
        nt=nt,
        n_assets=2,
        sigma=(0.02, 0.03),
        kappa=(1.0e-4, 2.0e-4),
        eta=(0.1, 0.15),
        gamma=2.0,
        epsilon=1.0e-2,
        alpha=30,
        q0_min=(0.5, 0.5),
        q0_max=(1.5, 1.5),
        S0=(1.0, 1.05),
        X0=0.0,
        device=device,
    )

    # Example open-loop control trajectory for quick sanity checks (not used in GradientTester below).
    u_rand = torch.randn(batch_size, prob.control_dim, nt, device=device)

    # Gradient tests: small hand-picked (z, u) pairs in the multi-asset
    # state layout [q_1,...,q_n, S_1,...,S_n, X]. Two distinct seed states
    # are tiled across the batch so the analytical/autograd Jacobians are
    # exercised on more than one operating point.
    n = prob.n_assets
    q_seed = torch.tensor([[1.0] * n, [0.8] * n], dtype=torch.float32)
    S_seed = torch.tensor([[1.0] * n, [1.1] * n], dtype=torch.float32)
    X_seed = torch.tensor([[0.0], [0.1]], dtype=torch.float32)
    test_z = torch.cat([q_seed, S_seed, X_seed], dim=1)
    test_u = torch.tensor([[0.1] * n, [0.2] * n], dtype=torch.float32)

    # ...tiled to match prob.batch_size so Jacobian checks run in batch layout.
    test_z = test_z.repeat(batch_size // 2, 1).to(device)
    test_u = test_u.repeat(batch_size // 2, 1).to(device)

    print("Running gradient tests...")
    # Compares hand-coded dynamics / cost derivatives against autograd-based references.
    GradientTester.run_all_tests(prob, test_z, test_u)
