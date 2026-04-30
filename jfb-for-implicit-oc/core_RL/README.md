# `core-RL`: RL extension of the implicit-Hamiltonian pipeline

This directory mirrors `core/`, but for the **unknown-dynamics** setting.
The agent never queries `compute_f` or its Jacobians; it interacts with
an `Environment` and uses local Jacobian estimates produced by a
`JacobianEstimator`.

## File map


| `core-RL/` file               | Mirrors                                  | Status   |
| ----------------------------- | ---------------------------------------- | -------- |
| `Environment.py`              | — (new)                                  | new      |
| `JacobianEstimator.py`        | — (new)                                  | new      |
| `ImplicitOC_RL.py`            | `core/ImplicitOC.py`                     | rewrite  |
| `ImplicitNets_RL.py`          | `core/ImplicitNets.py` (`ImplicitNetOC`) | subclass |
| `OptimalControlTrainer_RL.py` | `core/OptimalControlTrainer.py`          | subclass |


The following `core/` files are **reused unchanged** by the RL pipeline:
`paths.py`, `run_io.py`, `log_format.py`, `utils.py`, and the entire
`benchmarking/` subpackage. The artifact bundle and plotting service
stay identical.

`core/CVXPolicy.py` and `core/DirectControlNets.py` (the JBB / direct-
transcription baselines) have **no RL counterpart**: a CVXPyLayers policy
relies on differentiable convex optimisation that, in our setting,
requires the analytical Hamiltonian — which we don't have. If we want
non-implicit baselines for the RL paper, we would write a separate
`models-RL` direct-transcription baseline that also uses estimated
Jacobians; that is a future-work item.

## What changes, mathematically

In one line: the gradient $dJ/d\theta$ that used to come for free from
`total_cost.backward()` is now built explicitly. The pipeline becomes:

```
forward rollout (env.step, no autograd)
    -> Jacobian estimator update (RLS)
        -> backward adjoint pass (data-driven, no autograd)
            -> JFB surrogate (autograd through phi only)
                -> surrogate.backward()
```

The full derivation lives in §3 of the Pontryagin-RL notes; the
relevant identities, in the existing repo's sign / shape convention, are:

- **Fixed-point operator.** $\hat T_k(u; z) = u - \alpha (\nabla_u L + b_kp)$,
where $b_k$ has shape $(B, m, n)$ with $b_k[\cdot, i, j] = \partial f_j / \partial u_i$.
This matches `core/ImplicitOC.compute_grad_H_u`'s shape contract.
- **Backward adjoint.** $p_k = p_{k+1} + \Delta t (a_k^\top p_{k+1} + \nabla_z L)$,
with $a_k$ the standard Jacobian (shape $(B, n, n)$, $a_k[i,j] = \partial f_i / \partial z_j$).
Terminal condition $p_N = \nabla G(z_N)$.
- **JFB surrogate.**
  $$S(\theta) = \sum_k \big\langle \hat T_k(\bar u_k; \bar z_k),\ \Delta t \cdot (\nabla_u L + b_k\,p_{k+1})\big\rangle$$
  with $\bar u_k$, $b_k$, $p_{k+1}$ all detached. Then $\nabla_\theta S$ equals
  the JFB-with-estimates gradient of the actual objective.

## Stateful contract (read this once)

`ImplicitNetOC_RL` does **not** discover $b_k$ on its own. The trainer or
loss-routine **must** call `policy.set_step_jacobian(b_k)` immediately
before each `policy(z_k, t_k)` invocation. This applies inside
`compute_loss_RL`, inside `env.rollout` when called for plotting, and
inside any custom diagnostic that re-evaluates the policy. If you forget,
`T()` will raise.

## Sanity-check workflow

1. Pick a problem where `compute_f` is known (e.g. an existing model in
  `models/`).
2. Wrap it in `AnalyticalEnvironment(f_callable=prob.compute_f, ...)`.
3. Train twice on the same seed:
  - **Run A**: original `OptimalControlTrainer` with known dynamics.
  - **Run B**: `OptimalControlTrainer_RL` with `OracleJacobianEstimator`
  pointing at `prob.compute_grad_f_z` and `prob.compute_grad_f_u`.
4. The two loss curves should track each other up to numerical noise. If
  not, the bug is in the new surrogate-construction code (since the
   oracle estimator removes the only other source of error). This test
   isolates the gradient construction from the Jacobian-estimation
   quality and is the single most useful diagnostic to set up early.
5. Once Run A ≈ Run B, swap `OracleJacobianEstimator` for
  `RLSJacobianEstimator`. The gap between the resulting loss curve and
   Run A is then exactly the cost of estimating Jacobians from data.

## What is NOT yet implemented

- **HJB / adjoint consistency penalties** (`alphaHJB`, `alphaadj`): require
$f$, dropped here. They could be reintroduced after Step 3 (when we have
a learned model $\hat f_\eta$).
- **Stochastic dynamics** (Step 3 of the project): the `Environment.step`
contract is deterministic by design. Stochastic transitions are a
later refactor to a probabilistic step + a backward SDE in place of the
deterministic adjoint.
- **CVXPyLayers / direct-control baselines**: see file map above.

