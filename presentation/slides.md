---
title: "Implicit Hamiltonian Learning for Optimal Control: Replication and Stochastic Extension"
theme: default
class: text-left
highlighter: shiki
transition: fade
mdc: true
---

# Implicit Hamiltonian learning for high dimensional optimal control

### RL and Stochastic control Extension

<em class="paper-ref">End-to-End Training of High-Dimensional Optimal Control with Implicit<br />Hamiltonians via Jacobian-Free Backpropagation</em>

<!--
Title and motivation together: many OC problems admit a Hamiltonian whose maximizer has no closed form, so standard “differentiate through an explicit policy map” pipelines break. We still want a neural surrogate phi_theta for the value, recover feedback implicitly from H_u=0, roll trajectories z_x(t) forward from sampled initial conditions x, and train against the objective. Roadmap: HJB/PMP recap, implicit vs explicit Hamiltonians, JFB, Almgren–Chriss reproduction, why full state (q,S,X) failed, reduced (q,S) results, stochastic outlook, conclusions.
-->

---

## 1.1 - Classical optimal control - PMP perspective


$$
\min_{u \in U} J(u; x) = \int_0^T L\bigl(t, z_x(t), u(t)\bigr)\,dt + G\bigl(z_x(T)\bigr)
$$

$$
\dot z_x(t) = f\bigl(t, z_x(t), u(t)\bigr), \qquad z_x(0) = x
$$

**Generalized Hamiltonian**

$$
\mathcal{H}(t, z_x, p_x, u) = -\,p_x^\top f(t, z_x, u) - L(t, z_x, u)
$$

**State and adjoint** along an optimal control $u^\star$

$$
\dot z_x = -\nabla_{p_x} \mathcal{H}(t, z_x, p_x, u^\star), \qquad z_x(0) = x
$$

$$
\dot p_x = \nabla_{z_x} \mathcal{H}(t, z_x, p_x, u^\star), \qquad p_x(T) = \nabla G\bigl(z_x(T)\bigr)
$$
**Optimality**
$$
u^\star(t) \in \arg\max_{u} \mathcal{H}\bigl(t, z_x(t), p_x(t), u\bigr)
\quad \Longrightarrow \quad
\nabla_u \mathcal{H}\bigl(t, z_x(t), p_x(t), u^\star(t)\bigr) = 0
$$


---

## 1.2 - Classical optimal control — HJB formulation


<br>

**Value surrogate** $\phi_\theta(t,z)$

$$
\phi_\theta(t, z) = \inf_{u} \left[ \int_t^T L\bigl(s, z(s), u(s)\bigr)\,ds + G\bigl(z(T)\bigr) \right]
$$

**HJB**

$$
\partial_t \phi_\theta(t, z) + \sup_{u \in U} \mathcal{H}(t, z, p, u) = 0
$$

**Bridge with original formulation**

$$
p_x(t) = \nabla_z \phi_\theta\bigl(t, z_x(t)\bigr)
$$



---

## 1.3. Paper's contribution

- **Problem:** **no closed form** for $u$ in $\nabla_u \mathcal{H} = 0$

<PipelineBox title="End to end training pipeline">

$$
\phi_\theta(t,z) \;\longrightarrow\; p_\theta = \nabla_z \phi_\theta(t,z) \;\longrightarrow\; \text{INN } u_\theta^*(t,z, p_\theta) \;\longrightarrow\; z_x^\theta(t) \;\longrightarrow\; J(\theta)
$$

</PipelineBox>

<br>

<div class="grid grid-cols-2 gap-10 items-start">

<div>

**Fixed-point characterization**

$$
u_\theta^* = T_\theta(u_\theta^*;\, t, z)
$$
$$
T_\theta(u;\, t, z) = u + \alpha \,\nabla_u \mathcal{H}(t, z, \nabla_z \phi_\theta, u)
$$

</div>

<div>

**JFB approximation**



$$
\frac{\partial u_\theta^*}{\partial \theta}
=
\underbrace{\left(I - \tfrac{\partial T_\theta}{\partial u}\right)^{-1}}_{\mathcal{O}(m^3)\text{ solve}}
\frac{\partial T_\theta}{\partial \theta}
\;\approx\;
\frac{\partial T_\theta}{\partial \theta}
$$



</div>


</div>

<div class="text-sm mt-4">

| | Exact IFT | AD | JFB |
|:--|:--|:--|:--|
| **Compute** | $\mathcal{O}(m^3P)$ | $\mathcal{O}(K m^2)$ | $\mathcal{O}(mP)$ |
| **Memory** | $\mathcal{O}(K)$ | $\mathcal{O}(K)$ | $\mathcal{O}(1)$ |

</div>

*AD — autodiff / BPTT through $K$ unrolled inner iterations on tape.*


---

## 1.3.2 - Convergence condition

**Contractivity condition** 

<PipelineBox title="End to end training pipeline">

$$
\phi_\theta(t,z) \;\longrightarrow\; p_\theta = \nabla_z \phi_\theta(t,z) \;\longrightarrow\; \text{INN } u_\theta^*(t,z, p_\theta) \;\longrightarrow\; z_x^\theta(t) \;\longrightarrow\; J(\theta)
$$

</PipelineBox>


<br>
<br>

$$
\Gamma_c := \left\|\frac{\partial T_\theta}{\partial u}\right\| = \left\|I + \alpha\,\nabla^2_{uu}\mathcal{H}\right\| < 1
$$

<br>

$$
\Longleftrightarrow \quad \nabla^2_{uu}\mathcal{H} \prec 0 \quad (\mathcal{H} \text{ strictly concave in } u)
$$

---

## 1.4. Training algorithm - (extra details)

<PipelineBox title="End to end training pipeline">

$$
\phi_\theta(t,z) \;\longrightarrow\; p_\theta = \nabla_z \phi_\theta(t,z) \;\longrightarrow\; \text{INN } u_\theta^*(t,z, p_\theta) \;\longrightarrow\; z_x^\theta(t) \;\longrightarrow\; J(\theta)
$$

</PipelineBox>

<pre v-pre class="algo-box"><code><span class="algo-line-muted"><span class="algo-ln"> 1:</span>  <span class="algo-kw">Initialize</span> networks with parameters <span class="algo-math">θ</span></span>
<span class="algo-line-muted"><span class="algo-ln"> 2:</span>  <span class="algo-kw">for</span> iteration = 1, 2, … <span class="algo-kw">do</span></span>
<span class="algo-line-muted"><span class="algo-ln"> 3:</span>      Sample a batch of initial states <span class="algo-math">{x_i} ∼ ρ</span></span>
<span class="algo-line-muted"><span class="algo-ln"> 4:</span>      <span class="algo-kw">for</span> each trajectory <span class="algo-kw">do</span></span>
<span class="algo-ln"> 5:</span>         <span class="algo-kw">for</span> <span class="algo-math">k</span> = 0, …, <span class="algo-math">N_t − 1</span> <span class="algo-kw">do</span>
<span class="algo-ln"> 6:</span>              Compute grad of the value function <span class="algo-math">p ← ∇_z φ_θ (t_k, z)</span> <span class="algo-cm"># discrete adjoint</span>
<span class="algo-ln"> 7:</span>              INN solves fixed point eq for optimal <span class="algo-math">u</span>  <span class="algo-cm"># K steps detached, then K′ steps on-graph</span>
<span class="algo-ln"> 8:</span>              Increase running loss and evole state <span class="algo-math">z</span>
<span class="algo-ln"> 9:</span>         <span class="algo-kw">end for</span>
<span class="algo-line-muted"><span class="algo-ln">10:</span>         Mix running loss with terminal <span class="algo-math">G(z)</span></span>
<span class="algo-line-muted"><span class="algo-ln">11:</span>      <span class="algo-kw">end for</span></span>
<span class="algo-line-muted"><span class="algo-ln">12:</span>      Average batch objectives for the final loss</span>
<span class="algo-line-muted"><span class="algo-ln">13:</span>      Backprop on <span class="algo-math">θ</span></span>
<span class="algo-line-muted"><span class="algo-ln">14:</span>  <span class="algo-kw">end for</span></span>
</code></pre>




- Gradient flows **only** through the **tracked tail** of length $K'$ — not through the $K$-step convergence loop or the state transitions


<!--
Speaker note: Inner loop on **u**: many Hamiltonian-ascent steps with no graph tape, then a short **tracked** tail (length K′) so only that tail contributes to gradients in **θ** (Term I / JFB). Dynamics in **z** are stepped without differentiating through the transition. Costate **p** is whatever **∇_z φ_θ** implementation you couple into **H** — the narrative here is Pontryagin/HJB bookkeeping, not a claim about computing that gradient by reverse-mode autodiff.
-->

---

## 1.5. Almgren–Chriss reproduction example

**State**

$$
z = (q, S, X) \qquad
\dot z = f(z,u), \qquad
f(z,u) = \begin{pmatrix} -u \\ -\kappa u \\ Su - \eta\,|u|^\gamma \end{pmatrix}
$$

$q$ is inventory, $S$ is the mid-price, $X$ generated cash and $u$ the **control** liquidation rate.

**Objective**

<div class="objective-math">

$$
J(u)
=
\overbrace{\int_0^T \frac{\sigma^2}{2}\, q(t)^2 \, dt}^{\,L}
+
\overbrace{\alpha\, q(T)^2}^{\,G(T)}
-
\bigl(X(T)-X_0\bigr).
$$

</div>

Inventory-risk running cost $\frac{\sigma^2}{2} q^2$, terminal liquidation penalty $\alpha q(T)^2$ and **cash bookkeeping** via $\mathrm{d}X$.

<div class="equiv-box">

<div class="equiv-box__title">Equivalently</div>

$$
\mathrm{d}q = -u\,\mathrm{d}t,\qquad
\mathrm{d}S = -\kappa u\,\mathrm{d}t,\qquad
\mathrm{d}X = \bigl(Su - \eta\,|u|^\gamma\bigr)\,\mathrm{d}t
$$

</div>

<!--
This is the canonical optimal execution model with temporary price impact and liquidation costs. The benchmark is attractive because a deterministic boundary-value or PDE solution can be compared against the learned feedback without ambiguity about the “truth.”
-->

---

## 1.6. Contractivity condition violated - $\nabla_{uu}^2\mathcal H$ picks up $p$

<br>

Recall the contractivity condition
<br>

$$
\Gamma_c := \left\|\frac{\partial T_\theta}{\partial u}\right\| = \left\|I + \alpha\,\nabla^2_{uu}\mathcal{H}\right\| < 1
$$


but here we have 

$$
\nabla_{uu}^2\mathcal{H}= - \nabla_{uu}^2 L - \nabla_{uu}^2 f = - p_X \eta \gamma (\gamma - 1) u^{\gamma - 2}
$$

If the whole thing is positive and $p_X = -1$, we cannot have convergence.



---

## 1.7. Reduced formulation and empirical outcome

**OC state**

$$
z=(q,S)
$$

$$
\begin{aligned}
J(u)
&= \int_0^T \frac{\sigma^2}{2}\, q(t)^2 \, dt + \alpha\, q(T)^2 - \bigl(X(T)-X_0\bigr) \\
&= \int_0^T \frac{\sigma^2}{2}\, q(t)^2 \, dt + \alpha\, q(T)^2 - \int_0^T \dot X(t)\, dt,
   && \text{since } X(T)-X_0 = \int_0^T \dot X \, dt \\
&= \int_0^T \Bigl( \tfrac{\sigma^2}{2}\, q(t)^2 - S(t)\, u(t) + \eta\,|u(t)|^\gamma \Bigr)\, dt + \alpha\, q(T)^2,
   && \text{(\S\,1.5)} \quad \dot X = Su - \eta\,|u|^\gamma\text{.}
\end{aligned}
$$

Running cost $=$ reduced $L'$; no $X$ in the OC state.

$$
\phi_\theta(t,q,S,X)=X+\widetilde{\phi}_\theta(t,q,S),\qquad \partial_X\phi_\theta=1.
$$

$\gamma$, $\nabla_{uu}^2\mathcal{H}$ (smooth $\psi_\gamma(u)=(u^2+\varepsilon)^{\gamma/2}$)

<!-- $$
\text{Augmented (§\,1.6)}\quad
\bigl[\nabla_{uu}^2\mathcal{H}\bigr]_{ij}=\delta_{ij}\,\bigl(-p_X\,\eta_i\,\psi_\gamma''(u_i)\bigr),
\qquad
\text{Reduced §\,1.7, }\gamma=2\text{}\quad \mathrm{diag}(2\eta_i).
$$ -->

<!--
Speaker: $X(T)-X_0=\int \dot X$ absorbs cash into $L'$; decomposition $\phi=X+\tilde\phi$ fixes $\partial_X\phi=1$. Hessian recap §1.6 vs §1.7.
-->

---

## 1.9. Stochastic extension (outlook)

$$
dZ_t = f(t, Z_t, u_t)\,dt + \sigma(t, Z_t, u_t)\,dW_t.
$$

$$
\partial_t \phi_\theta + \max_u \left[
L + \nabla_z \phi_\theta^\top f + \tfrac{1}{2}\operatorname{Tr}\bigl(\sigma \sigma^\top \nabla_{zz}^2 \phi_\theta\bigr)
\right] = 0.
$$

- If $\sigma$ **does not** depend on $u$: fixed-point **form** for $u^*$ often **unchanged**, but $\phi_\theta$ must capture **diffusion**
- If $\sigma$ **depends** on $u$: the **algebraic** fixed-point for $u^*$ **changes**
- **Open angles:** Hessian / trace estimation, MC rollouts, contractivity of the stochastic Hamiltonian map

<!--
This is the forward-looking slide: stochastic HJB adds a trace term involving the Hessian of $\phi_\theta$. For neural $\phi_\theta$, that raises questions of variance and stability. Control-dependent diffusion couples into the implicit first-order condition for u, altering T_theta. Research questions include efficient JVP/HVP schemes, sample-based pathwise losses, and whether the inner map remains well-posed / contractive after discretization.
-->

---

## 1.10. Takeaways

1. **JFB** trains with **implicit** Hamiltonians **without** unrolling the inner fixed-point solver in the autodiff tape
2. **Almgren–Chriss** required exploiting **$\phi_\theta = X + \widetilde{\phi}_\theta$** so $X$ is not learned as a redundant coordinate
3. **Stochastic** models add **diffusion**, **Monte Carlo** rollouts, and—when $\sigma$ depends on $u$—**new** implicit structure in the control equation

<!--
Close by tying back to the opening pipeline: implicit u_theta^* from grad_u H = 0, differentiated via JFB, integrated along z_x(t). The benchmark story shows that physics/bookkeeping structure matters as much as the implicit-differentiation trick. The stochastic slide frames honest next steps rather than finished work. Section 2 (merged `slides-part2-rl.md`) covers the RL / stochastic-control extension in more detail.
-->

---
src: ./slides-part2-rl.md
---
