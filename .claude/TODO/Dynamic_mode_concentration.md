# Dynamic Mode Concentration in Adjoint Matching

## Overview

The static theory (separate document) establishes that Adjoint Matching has a structural bias: the regression weight for each mode is proportional to the policy’s current allocation $\alpha_k$, not the target weight $w_k$. This document asks the dynamical question: does this bias cause mode weights to diverge from the target across training epochs?

The main difficulty is that the mode weight dynamics are a projection of infinite-dimensional controller dynamics and do not form a closed system in general. Mode separation provides the reduction: when modes are well-separated relative to the noise scale, the intra-mode structure approximately decouples from the inter-mode balance, and the relevant dynamics are well-approximated by a finite-dimensional system on the simplex with an explicitly computable Jacobian.

## Prerequisites

All definitions and results from the static theory are assumed:

- **Definitions 1–6**: Boltzmann distribution, mode basins, mode decomposition, mode weights $\alpha_k$, target weights $w_k$, oracle regression coefficients $\eta_k^*$, Gram matrix of mode adjoints.
- **Proposition 1**: $u^*(x,t) = -g^2 \sum_k \eta_k(x,t),\bar{Y}_k(x,t)$ with $\eta_k \propto \alpha_k$.

-----

## Definitions

**Definition D1 (VE-SDE).**
The controlled VE-SDE on $[0,1]$ is

$$dX_t = g(t)^2, u(X_t, t),dt + g(t),dW_t, \qquad X_0 \sim \mu$$

where $g : [0,1] \to (0, \infty)$ is the diffusion schedule. The uncontrolled SDE ($u = 0$) has law $Q^0$. The cumulative noise scale is $\sigma(1) = \left(\int_0^1 g(s)^2,ds\right)^{1/2}$.

**Definition D2 (Novikov condition and Girsanov weight).**
A controller $u$ satisfies the Novikov condition if

$$\mathbb{E}_{Q^0}!\left[\exp!\left(\tfrac{1}{2}\int_0^1 g(t)^2 |u(X_t,t)|^2,dt\right)\right] < \infty.$$

This holds for any uniformly bounded controller (e.g., any neural network with bounded outputs). Under Novikov, Girsanov’s theorem defines the change of measure:

$$R^u := \frac{dQ^u}{dQ^0} = \exp!\left(\int_0^1 g(t), u(X_t, t) \cdot dW_t^{Q^0} - \frac{1}{2}\int_0^1 g(t)^2 |u(X_t,t)|^2,dt\right)$$

and for any measurable event $A$: $P_{Q^u}(A) = \mathbb{E}_{Q^0}[\mathbf{1}_A \cdot R^u]$.

**Definition D3 (AM epoch operator).**
Under exact AM regression and exact corrector (CM converged), the AM epoch operator $T$ maps a controller $u$ to a new controller $T(u)$ defined by:

$$T(u)(x,t) := -g(t)^2,\mathbb{E}_{Q^u}[Y_1 \mid X_t = x]$$

where $Y_1 = -\nabla\Phi_0(X_1)$ is the adjoint terminal value and $Q^u$ is the law of the SDE with controller $u$. The mode weights induced by $T(u)$ are:

$$F_k(u) := P_{Q^{T(u)}}(X_1 \in \mathcal{M}*k) = \mathbb{E}*{Q^0}!\left[\mathbf{1}_{X_1 \in \mathcal{M}_k} \cdot R^{T(u)}\right].$$

**Definition D4 (Mode separation parameter).**
Let $d_{\min} := \min_{j \neq k} |\mu_j - \mu_k|$ be the minimum inter-mode distance, where $\mu_k$ are the mode centers (local minima of $E$). The mode separation ratio is

$$\rho_{\mathrm{sep}} := \frac{d_{\min}}{\sigma(1)}.$$

We say the modes are *well-separated* when $\rho_{\mathrm{sep}} \gg 1$.

-----

## Lemma D1 (Balanced Fixed Point)

The SOC-optimal controller $u^*$ is a fixed point of the epoch operator: $T(u^*) = u^*$. Consequently, $F(u^*) = w$ (the target mode weights).

**Proof.** The SOC-optimal controller satisfies $u^*(x,t) = -g(t)^2,\nabla V(x,t)$, where $V$ is the value function solving the Hamilton-Jacobi-Bellman equation.

For the VE-SDE ($f = 0$), the adjoint BSDE has zero drift: $dY_t = Z_t,dW_t$, so $Y_t$ is a martingale and $Y_t = \mathbb{E}_{Q^*}[Y_1 \mid \mathcal{F}_t]$. By the Markov property of the optimal SDE, this conditional expectation depends on $\mathcal{F}_t$ only through $X_t$:

$$Y_t = \mathbb{E}_{Q^*}[Y_1 \mid \mathcal{F}*t] = \mathbb{E}*{Q^*}[Y_1 \mid X_t].$$

By the stochastic Pontryagin maximum principle (the verification theorem for stochastic optimal control), $Y_t = \nabla V(X_t, t)$ along optimal paths. Since $\nabla V(x,t)$ is a deterministic function of $(x,t)$:

$$\mathbb{E}_{Q^*}[Y_1 \mid X_t = x] = \nabla V(x, t).$$

Therefore:

$$T(u^*)(x,t) = -g(t)^2,\mathbb{E}_{Q^*}[Y_1 \mid X_t = x] = -g(t)^2,\nabla V(x,t) = u^*(x,t).$$

Since $u^*$ is the SOC-optimal controller, $Q^{u^*}_{\mathrm{terminal}} = p$ (the target distribution), so $F_k(u^*) = P_p(X \in \mathcal{M}_k) = w_k$ for all $k$. $\square$

-----

## Proposition D1 (Girsanov Sensitivity of Mode Weights)

For any controller perturbation $\delta u$ around $u^*$ satisfying standard regularity, the first-order change in mode weights under the perturbed SDE is:

$$\delta F_k = \mathbb{E}*{Q^*}!\left[\mathbf{1}*{X_1 \in \mathcal{M}_k} \cdot \int_0^1 g(t),\delta u(X_t, t) \cdot dW_t\right]$$

where the expectation and Brownian motion are under $Q^*$.

**Proof.**

From Definition D2, $F_k(u^* + \delta u) = \mathbb{E}*{Q^0}[\mathbf{1}*{\mathcal{M}_k} \cdot R^{u^*+\delta u}]$.

Write $L[v] = \int_0^1 g, v \cdot dW^{Q^0} - \frac{1}{2}\int_0^1 g^2 |v|^2,dt$, so $R^v = \exp(L[v])$.

Linearize:

$$L[u^* + \delta u] = L[u^*] + \int_0^1 g,\delta u \cdot dW^{Q^0} - \int_0^1 g^2, u^* \cdot \delta u,dt + O(|\delta u|^2).$$

Therefore:

$$R^{u^*+\delta u} \approx R^{u^*}\left(1 + \int_0^1 g,\delta u \cdot dW^{Q^0} - \int_0^1 g^2, u^* \cdot \delta u,dt\right).$$

Applying $\mathbb{E}*{Q^0}[\cdot, R^{u^*}] = \mathbb{E}*{Q^*}[\cdot]$:

$$\delta F_k = \mathbb{E}*{Q^*}!\left[\mathbf{1}*{\mathcal{M}_k}\left(\int_0^1 g,\delta u \cdot dW^{Q^0} - \int_0^1 g^2, u^* \cdot \delta u,dt\right)\right].$$

Under $Q^*$, the Brownian motions are related by $dW^{Q^0}_t = dW^{Q^*}_t + g(t),u^*(X_t,t),dt$. Substituting:

$$\int_0^1 g,\delta u \cdot dW^{Q^0} = \int_0^1 g,\delta u \cdot dW^{Q^*} + \int_0^1 g^2, u^* \cdot \delta u,dt.$$

The two $\int g^2 u^* \cdot \delta u,dt$ terms cancel exactly, leaving:

$$\delta F_k = \mathbb{E}*{Q^*}!\left[\mathbf{1}*{X_1 \in \mathcal{M}_k} \cdot \int_0^1 g(t),\delta u(X_t, t) \cdot dW_t\right]. \quad\square$$

*Note.* The cancellation is the key simplification: the first-order mode weight response depends only on the martingale part $\int g,\delta u \cdot dW^{Q^*}$, not on any drift interaction with $u^*$.

-----

## Proposition D2 (Controller Sensitivity Decomposition)

At the fixed point $u^*$, the change in the AM regression output when the data-generating controller is perturbed by $\delta u_{\mathrm{old}}$ decomposes as:

$$\delta T(u^*)(x,t) = \underbrace{-g(t)^2 \sum_{k=1}^K \delta\eta_k(x,t) \cdot \bar{Y}*k(x,t)}*{\text{$\eta$-response}} + \underbrace{\left(-g(t)^2 \sum_{k=1}^K \eta_k^*(x,t) \cdot \delta\bar{Y}*k(x,t)\right)}*{\text{bridge response}}$$

where:

- $\delta\eta_k(x,t) = \sum_j \frac{\eta_k^*(\delta_{kj} - \eta_j^*)}{w_j},\delta\alpha_j$ is the change in regression coefficient, determined algebraically by the induced mode weight change $\delta\alpha_j = \delta F_j(\delta u_{\mathrm{old}})$ through Proposition 1 of the static theory.
- $\delta\bar{Y}*k(x,t)$ is the change in the conditional mode adjoint due to the change in the bridge distribution $Q^u(X_1 \mid X_t, M=k)$. This depends on the full controller perturbation $\delta u*{\mathrm{old}}$, not only on the induced mode weight change $\delta\alpha$.

**Proof.** From Proposition 1 (static theory), $T(u)(x,t) = -g^2 \sum_k \eta_k(x,t; Q^u),\bar{Y}_k(x,t; Q^u)$. Both $\eta_k$ and $\bar{Y}_k$ depend on $Q^u$. Differentiating by the product rule:

$$\delta T = -g^2 \sum_k [\delta\eta_k,\bar{Y}_k + \eta_k,\delta\bar{Y}_k].$$

At the fixed point, $\eta_k = \eta_k^*$. The formula for $\delta\eta_k$ follows from the quotient rule applied to $\eta_k = \alpha_k q_t^{(k)} / \sum_l \alpha_l q_t^{(l)}$ (the partial derivatives are algebraic identities from the static theory). $\square$

*Remark.* The $\eta$-response depends on $\delta u_{\mathrm{old}}$ only through the induced $\delta\alpha$ (a $K$-dimensional projection). The bridge response depends on the full infinite-dimensional perturbation. This asymmetry is why the mode weight dynamics do not form a closed system on the simplex in general.

-----

## Theorem D1 (Mode Weight Dynamics: Simplex-Closed Part and Residual)

Under exact AM, exact corrector, and Novikov, the first-order mode weight dynamics at the balanced fixed point decompose as:

$$\delta\alpha_k’ = \sum_j J_{kj}^\eta,\delta\alpha_j + R_k(\delta u_{\mathrm{old}})$$

where:

**$\eta$-Jacobian (explicit, depends only on $\delta\alpha$):**

$$J_{kj}^\eta = -\frac{1}{w_j},\mathbb{E}*{Q^*}!\left[\mathbf{1}*{X_1 \in \mathcal{M}_k} \cdot \int_0^1 g(t)^3,\eta_j^*(X_t,t)\left[\bar{Y}_j(X_t,t) - \bar{Y}(X_t,t)\right] \cdot dW_t\right]$$

with $\bar{Y}(x,t) = \sum_l \eta_l^*(x,t),\bar{Y}_l(x,t)$.

**Residual (depends on the full controller perturbation):**

$$R_k(\delta u_{\mathrm{old}}) = \mathbb{E}*{Q^*}!\left[\mathbf{1}*{X_1 \in \mathcal{M}_k} \cdot \int_0^1 g(t) \left(-g(t)^2\sum_l \eta_l^*(X_t,t),\delta\bar{Y}_l(X_t,t)\right) \cdot dW_t\right]$$

where $\delta\bar{Y}_l$ is the bridge response from Proposition D2.

These satisfy the constraints:

1. $\sum_k J_{kj}^\eta = 0$ for all $j$ (so $J^\eta$ maps $T\Delta$ to $T\Delta$).
1. $\sum_k R_k(\delta u_{\mathrm{old}}) = 0$ for all $\delta u_{\mathrm{old}}$ (mode weights sum to 1).

**Proof.**

Combine Propositions D1 and D2. The total new controller perturbation at the fixed point is:

$$\delta v(x,t) = -g^2 \sum_l [\delta\eta_l(x,t),\bar{Y}_l(x,t) + \eta_l^*(x,t),\delta\bar{Y}_l(x,t)].$$

Substituting into the Girsanov sensitivity (Proposition D1):

$$\delta\alpha_k’ = \mathbb{E}*{Q^*}!\left[\mathbf{1}*{\mathcal{M}_k} \int_0^1 g(t),\delta v(X_t, t) \cdot dW_t\right]$$

which splits linearly into the $\delta\eta$ and $\delta\bar{Y}$ contributions.

**$\eta$-term.** Substitute $\delta\eta_l = \sum_j \frac{\eta_l^*(\delta_{lj} - \eta_j^*)}{w_j}\delta\alpha_j$ and collect terms in $\delta\alpha_j$:

$$\sum_l \delta\eta_l,\bar{Y}*l = \sum_j \left(\sum_l \frac{\eta_l^*(\delta*{lj} - \eta_j^*)}{w_j}\bar{Y}_l\right)\delta\alpha_j = \sum_j \frac{\eta_j^*(\bar{Y}_j - \bar{Y})}{w_j},\delta\alpha_j$$

where the simplification uses:

$$\sum_l \eta_l^*(\delta_{lj} - \eta_j^*)\bar{Y}_l = \eta_j^*\bar{Y}_j - \eta_j^*\sum_l \eta_l^*\bar{Y}_l = \eta_j^*(\bar{Y}_j - \bar{Y}).$$

Substituting into the Girsanov sensitivity gives $\sum_j J_{kj}^\eta,\delta\alpha_j$ with $J_{kj}^\eta$ as stated.

**$\bar{Y}$-term.** The $\delta\bar{Y}*l$ contribution gives $R_k(\delta u*{\mathrm{old}})$ directly.

**Constraint 1.** Since $\sum_k \mathbf{1}_{X_1 \in \mathcal{M}_k} = 1$ almost surely:

$$\sum_k J_{kj}^\eta = -\frac{1}{w_j},\mathbb{E}_{Q^*}!\left[\int_0^1 g^3,\eta_j^*(\bar{Y}_j - \bar{Y}) \cdot dW_t\right] = 0$$

by the martingale property (the Ito integral of an adapted, square-integrable process has zero expectation).

**Constraint 2.** Similarly:

$$\sum_k R_k = \mathbb{E}_{Q^*}!\left[\int_0^1 g\left(-g^2\sum_l \eta_l^*,\delta\bar{Y}_l\right) \cdot dW_t\right] = 0$$

by the same martingale property. $\square$

-----

## Lemma D2 (Residual Bound Under Mode Separation)

Under mode separation ($\rho_{\mathrm{sep}} = d_{\min}/\sigma(1) \gg 1$) and standard regularity (the energy $E$ has bounded curvature $|\nabla^2 E(\mu_k)| \leq \kappa$ near each mode center, and the optimal controller satisfies $|u^*|*\infty \leq B$), the residual satisfies: for any bounded controller perturbation $|\delta u*{\mathrm{old}}|_\infty \leq M$,

$$|R_k(\delta u_{\mathrm{old}})| \leq C(d, K, \kappa, B)\cdot M \cdot \exp!\left(-c,\rho_{\mathrm{sep}}^2\right)$$

for all $k$, where $c > 0$ is a universal constant.

In particular, for a perturbation $\delta u_{\mathrm{old}}$ that induces mode weight change $|\delta\alpha|_2 \leq \epsilon$:

$$\frac{|R_k(\delta u_{\mathrm{old}})|}{|\delta\alpha|*2} \leq C’ \cdot \exp!\left(-c,\rho*{\mathrm{sep}}^2\right)$$

where $C’$ depends on $C$ and the Girsanov sensitivity (Proposition D1). This ratio is bounded because the Girsanov map from controller perturbations to mode weight perturbations is a bounded linear operator (by the Ito isometry), so $|\delta u_{\mathrm{old}}|$ and $|\delta\alpha|$ are comparable up to problem-dependent constants.

**Proof sketch.**

The residual is controlled by $|\delta\bar{Y}_k|$, the sensitivity of the conditional mode adjoint to the controller perturbation.

*Step 1 (Bridge localization).* Under the uncontrolled VE-SDE, $X_t \mid X_0, X_1$ is Gaussian with variance at most $\sigma(1)^2$. The conditional $P_{Q^0}(X_1 \mid X_t = x,, X_1 \in \mathcal{M}_k)$ concentrates on $\mathcal{M}_k$ with Gaussian tails. Under $Q^*$, the Girsanov weight modifies this by at most $\exp(B^2\sigma(1)^2)$, a bounded factor that preserves exponential localization.

*Step 2 (Cross-mode insensitivity).* A controller perturbation $\delta u_{\mathrm{old}}$ with $|\delta u_{\mathrm{old}}|*\infty \leq M$ changes the SDE paths. For a point $x$ within distance $O(\sigma(1))$ of mode $k$, the change in the bridge $P(X_1 \mid X_t = x, M = k)$ requires the perturbation to propagate across the inter-mode gap $d*{\min}$. By the Gaussian transition kernel of the VE-SDE, this cross-mode coupling is suppressed by $\exp(-d_{\min}^2 / (C’’\sigma(1)^2)) = \exp(-C’’\rho_{\mathrm{sep}}^2)$. Therefore $|\delta\bar{Y}*k(x,t)| \leq C_1 M \exp(-c,\rho*{\mathrm{sep}}^2)$ for $x$ near mode $k$.

*Step 3 (Integration).* The inter-mode overlap region — the set of $(x,t)$ where the bridge is not exponentially localized — has $Q^*$-probability $O(\exp(-c,\rho_{\mathrm{sep}}^2))$. On this set, $|\delta\bar{Y}*k|$ is bounded by $C_2 M$ (polynomial in problem parameters). The contribution to $R_k$ from this set is therefore $O(C_2 M \exp(-c,\rho*{\mathrm{sep}}^2))$.

Combining both regions: $|R_k| \leq C M \exp(-c,\rho_{\mathrm{sep}}^2)$. $\square$

*Remark.* Making the constants explicit requires tracking dimension-dependent Gaussian tail factors for the specific $g(t)$. For any concrete problem, this is mechanical.

-----

## Theorem D2 (Instability Characterization)

Let $\lambda_{\max}^\eta$ denote the largest eigenvalue of $J^\eta$ restricted to the simplex tangent space $T\Delta = {v : \sum_k v_k = 0}$, with corresponding unit eigenvector $v^*$. Let $|J^\eta|*{\mathrm{op}}$ denote the operator norm of $J^\eta|*{T\Delta}$ (i.e., the largest singular value).

Under mode separation ($\rho_{\mathrm{sep}} \gg 1$), the balanced fixed point $\alpha = w$ is:

- **Linearly unstable** if $\lambda_{\max}^\eta > 1 + \sqrt{K},C’\exp(-c,\rho_{\mathrm{sep}}^2)$
- **Linearly stable** if $|J^\eta|*{\mathrm{op}} < 1 - \sqrt{K},C’\exp(-c,\rho*{\mathrm{sep}}^2)$

where $C’, c$ are from Lemma D2.

*Note on the two conditions.* The instability condition uses the spectral radius $\lambda_{\max}^\eta$, while the stability condition uses the operator norm $|J^\eta|*{\mathrm{op}}$. For non-symmetric $J^\eta$, the operator norm can exceed the spectral radius, leaving a gap where neither condition applies. For symmetric $J^\eta$ (which holds for $K = 2$ symmetric modes, see Corollary D2), the spectral radius equals the operator norm and the conditions are complementary. In general, for sufficiently large $\rho*{\mathrm{sep}}$, the stability of the balanced fixed point is determined by the spectral properties of $J^\eta$.

**Proof.**

**Instability.** Consider a perturbation $\delta\alpha = \epsilon,v^*$ along the eigenvector $v^*$ of $J^\eta|_{T\Delta}$ (with $|v^*|*2 = 1$), induced by some bounded controller perturbation $\delta u*{\mathrm{old}}$.

By Theorem D1:

$$\delta\alpha_k’ = \sum_j J_{kj}^\eta,(\epsilon v_j^*) + R_k(\delta u_{\mathrm{old}}) = \epsilon,\lambda_{\max}^\eta,v_k^* + R_k(\delta u_{\mathrm{old}}).$$

By Lemma D2, $|R_k| \leq C’ |\epsilon| \exp(-c,\rho_{\mathrm{sep}}^2)$ (using the bound relative to $|\delta\alpha|$). Therefore:

$$|\delta\alpha’|*2 \geq |\epsilon|,\lambda*{\max}^\eta,|v^*|*2 - |R|*2 \geq |\epsilon|\left(\lambda*{\max}^\eta - \sqrt{K},C’\exp(-c,\rho*{\mathrm{sep}}^2)\right).$$

If $\lambda_{\max}^\eta > 1 + \sqrt{K},C’\exp(-c,\rho_{\mathrm{sep}}^2)$, then $|\delta\alpha’|_2 > |\epsilon| = |\delta\alpha|_2$: the perturbation grows, and the balanced fixed point is linearly unstable.

**Stability.** For any perturbation $\delta\alpha = \epsilon,v$ with $v \in T\Delta$, $|v|_2 = 1$:

$$|\delta\alpha’|*2 \leq |J^\eta|*{T\Delta},(\epsilon v)|*2 + |R|*2 \leq |\epsilon|\left(|J^\eta|*{\mathrm{op}} + \sqrt{K},C’\exp(-c,\rho*{\mathrm{sep}}^2)\right).$$

The first inequality is the triangle inequality applied to Theorem D1. The second uses $|J^\eta v|*2 \leq |J^\eta|*{\mathrm{op}}|v|_2$ (definition of operator norm) and the Lemma D2 bound.

If $|J^\eta|*{\mathrm{op}} < 1 - \sqrt{K},C’\exp(-c,\rho*{\mathrm{sep}}^2)$, then $|\delta\alpha’|_2 < |\epsilon| = |\delta\alpha|_2$ for all perturbation directions: all perturbations decay. $\square$

-----

## Corollary D1 (Reduction to a Computable Criterion)

For a specified VE-SDE schedule $g(t)$ and energy function $E$ with well-separated modes ($\rho_{\mathrm{sep}} \gg 1$), the instability question reduces to evaluating a single spectral condition:

$$\boxed{\lambda_{\max}^\eta > 1}$$

where $\lambda_{\max}^\eta$ is the largest eigenvalue of the matrix $J^\eta|_{T\Delta}$ with entries

$$J_{kj}^\eta = -\frac{1}{w_j},\mathbb{E}*{Q^*}!\left[\mathbf{1}*{X_1 \in \mathcal{M}_k} \cdot \int_0^1 g(t)^3,\eta_j^*(X_t,t)\left[\bar{Y}_j(X_t,t) - \bar{Y}(X_t,t)\right] \cdot dW_t\right].$$

All quantities in this formula are determined by the optimal policy $Q^*$, which depends only on $g(t)$ and $E$:

1. $g(t)$ is the noise schedule (a design choice).
1. $\eta_j^*$, $\bar{Y}_j$, and $\bar{Y}$ are computable from the energy function and the mode decomposition.
1. The path-space expectation can be estimated by Monte Carlo over trajectories of the optimal SDE.

The exponential decay of the residual (Lemma D2) ensures that for $\rho_{\mathrm{sep}}$ sufficiently large, $\lambda_{\max}^\eta > 1$ is both necessary and sufficient for instability.

-----

## Corollary D2 (Two-Mode Specialization)

For $K = 2$ modes with target weights $w_1, w_2$, the Jacobian $J^\eta$ restricted to the one-dimensional tangent space $T\Delta = \mathrm{span}{(1, -1)}$ has a single eigenvalue:

$$\lambda = J_{11}^\eta - J_{12}^\eta$$

and the instability criterion reduces to $J_{11}^\eta - J_{12}^\eta > 1$.

For **symmetric** modes ($w_1 = w_2 = \tfrac{1}{2}$), the column-sum constraint and permutation symmetry give $J_{12}^\eta = -J_{11}^\eta$, so the criterion simplifies to:

$$J_{11}^\eta > \frac{1}{2}.$$

**Proof.** The tangent space of $\Delta^1$ is spanned by $v = (1, -1)$. The eigenvalue is:

$$\lambda = \frac{v^\top J^\eta v}{v^\top v} = \frac{J_{11}^\eta - J_{12}^\eta - J_{21}^\eta + J_{22}^\eta}{2}.$$

Column sums: $J_{11}^\eta + J_{21}^\eta = 0$ and $J_{12}^\eta + J_{22}^\eta = 0$, so $J_{21}^\eta = -J_{11}^\eta$ and $J_{22}^\eta = -J_{12}^\eta$. Substituting:

$$\lambda = \frac{2J_{11}^\eta - 2J_{12}^\eta}{2} = J_{11}^\eta - J_{12}^\eta.$$

For symmetric modes, the epoch map is equivariant under mode permutation: $F_1(\alpha_1, \alpha_2) = F_2(\alpha_2, \alpha_1)$. Differentiating: $J_{11}^\eta = J_{22}^\eta$ and $J_{12}^\eta = J_{21}^\eta$. Combined with $J_{21}^\eta = -J_{11}^\eta$: $J_{12}^\eta = -J_{11}^\eta$. Therefore $\lambda = 2J_{11}^\eta$, and instability holds iff $J_{11}^\eta > 1/2$. $\square$

-----

## Summary

|#|Result    |Statement                                                                                                             |Assumptions beyond static theory       |
|-|----------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------|
|1|Lemma D1  |$u^*$ is a fixed point; $w$ is the balanced mode weight                                                               |Exact AM, exact corrector              |
|2|Prop D1   |Mode weight response to controller perturbation via Girsanov                                                          |Novikov (holds for bounded controllers)|
|3|Prop D2   |Controller sensitivity = $\eta$-response + bridge response                                                            |None (product rule)                    |
|4|**Thm D1**|**$\delta\alpha’ = J^\eta\delta\alpha + R(\delta u_{\mathrm{old}})$ with $J^\eta$ explicit**                          |**None beyond 1–3**                    |
|5|Lemma D2  |$                                                                                                                     |R_k                                    |
|6|**Thm D2**|**Unstable if $\lambda_{\max}^\eta > 1$; stable if $|J^\eta|_{\mathrm{op}} < 1$ (up to exponentially small residual)**|**Mode separation**                    |
|7|Cor D1    |Instability iff $\lambda_{\max}^\eta > 1$ (computable for given $g(t)$, $E$)                                          |Mode separation + known $g(t)$         |
|8|Cor D2    |For $K=2$ symmetric: instability iff $J_{11}^\eta > 1/2$                                                              |Symmetric modes                        |

**The logical chain:**

1. The AM epoch operator has a balanced fixed point at the SOC-optimal controller (Lemma D1).
1. The mode weight response to perturbations decomposes into an explicit simplex-closed part $J^\eta\delta\alpha$ and a residual $R$ that depends on the full infinite-dimensional controller perturbation (Theorem D1). The decomposition is exact, with no assumptions beyond regularity.
1. Under mode separation — the regime where mode collapse is an actual problem — the residual is exponentially small relative to the simplex-closed part (Lemma D2). This is the finite-dimensional reduction.
1. Stability is therefore determined by the computable spectral condition $\lambda_{\max}^\eta > 1$ (Theorem D2).
1. For any concrete energy function and noise schedule, this condition can be evaluated by Monte Carlo over optimal-policy trajectories, giving a definitive answer to whether mode concentration occurs (Corollary D1).