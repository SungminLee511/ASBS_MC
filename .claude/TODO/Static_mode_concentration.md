# Mode Concentration in Adjoint Matching: A Rigorous Theory

## Notation

- $E : \mathbb{R}^d \to \mathbb{R}$: energy function
- $p(x) = Z^{-1}\exp(-E(x))$: target Boltzmann distribution
- $Q_\theta$: law of the controlled SDE with controller $u_\theta$
- $g(t)$: diffusion coefficient of the VE-SDE
- $Y_1 = -\nabla \Phi_0(X_1)$: adjoint terminal value
- $\mathcal{L}_{\mathrm{AM}}(\theta)$: Adjoint Matching loss
- For a vector-valued random variable $Y$, we write $\mathrm{Var}[Y] := \mathbb{E}[|Y - \mathbb{E}[Y]|^2] = \mathrm{tr}(\mathrm{Cov}[Y])$.

-----

## Definitions

**Definition 1 (Boltzmann distribution).**
Let $E : \mathbb{R}^d \to \mathbb{R}$ be an energy function with $Z = \int \exp(-E(x)),dx < \infty$. The target distribution is $p(x) = Z^{-1}\exp(-E(x))$.

**Definition 2 (Mode basin).**
Let $\mu_1, \ldots, \mu_K$ be the local minima of $E$. Assume $E$ is a Morse function (all critical points are non-degenerate). The basin of mode $k$ is

$$\mathcal{M}_k := {x \in \mathbb{R}^d : \text{the gradient flow } \dot{x} = -\nabla E(x) \text{ starting at } x \text{ converges to } \mu_k}.$$

Under the Morse assumption, the basins partition $\mathbb{R}^d$ almost everywhere: $\mathbb{R}^d = \bigcup_k \mathcal{M}_k$ with $\mathcal{M}_j \cap \mathcal{M}_k = \emptyset$ for $j \neq k$, up to a measure-zero separating boundary.

**Definition 3 (Mode decomposition).**

$$p(x) = \sum_{k=1}^K w_k, p_k(x), \qquad w_k = \int_{\mathcal{M}*k} p(x),dx, \qquad p_k(x) = \frac{p(x),\mathbf{1}*{\mathcal{M}_k}(x)}{w_k}.$$

By construction, $w_k > 0$, $\sum_k w_k = 1$, and each $p_k$ is a normalized density supported on $\mathcal{M}_k$.

**Definition 4 (Mode weights under the policy).**
At training epoch $n$, the policy’s mode weights are

$$\alpha_k^{(n)} := P_{Q_\theta^{(n)}}(X_1 \in \mathcal{M}*k), \qquad \sum*{k=1}^K \alpha_k^{(n)} = 1.$$

Mode $k$ is *over-represented* if $\alpha_k^{(n)} > w_k$ and *under-represented* if $\alpha_k^{(n)} < w_k$.

-----

## Lemma 1 (AM Regression Minimizer)

The global minimizer of $\mathcal{L}_{\mathrm{AM}}$ at a fixed $(x, t)$ is

$$u_\theta^*(x, t) = -g(t)^2, \mathbb{E}*{Q*\theta}[Y_1 \mid X_t = x].$$

*Proof.* The conditional expectation is the unique $L^2$-optimal predictor: for any random variable $Z$ and sigma-algebra $\mathcal{F}$,

$$\arg\min_v \mathbb{E}[|v - Z|^2 \mid \mathcal{F}] = \mathbb{E}[Z \mid \mathcal{F}].$$

Apply with $Z = -g(t)^2 Y_1$ and $\mathcal{F} = \sigma(X_t)$. $\square$

-----

## Proposition 1 (Policy-Biased Regression Target)

The global minimizer of $\mathcal{L}_{\mathrm{AM}}$ decomposes as

$$u_\theta^*(x, t) = -g(t)^2 \sum_{k=1}^K \eta_k(x, t), \bar{Y}_k(x, t)$$

where $\bar{Y}*k(x,t) = \mathbb{E}*{Q_\theta}[Y_1 \mid X_t = x,, X_1 \in \mathcal{M}_k]$ is the conditional adjoint for mode $k$, and

$$\eta_k(x, t) = \frac{\alpha_k, q_t^{(k)}(x)}{\sum_l \alpha_l, q_t^{(l)}(x)}$$

with $q_t^{(k)}(x) = p_{Q_\theta}(X_t = x \mid X_1 \in \mathcal{M}_k)$.

The regression coefficient $\eta_k$ is proportional to the policy’s current mode weight $\alpha_k$, not the target weight $w_k$.

*Proof.* From Lemma 1, $u_\theta^* = -g^2, \mathbb{E}[Y_1 \mid X_t = x]$. Apply the law of total expectation, partitioned over the mode label $M \in {1,\ldots,K}$ where $X_1 \in \mathcal{M}_M$:

$$\mathbb{E}[Y_1 \mid X_t = x] = \sum_{k=1}^K P(M = k \mid X_t = x)\cdot \mathbb{E}[Y_1 \mid X_t = x,, M = k].$$

Apply Bayes’ rule to the posterior over modes:

$$P(M = k \mid X_t = x) = \frac{P(X_t = x \mid M = k), P(M = k)}{P(X_t = x)} = \frac{\alpha_k, q_t^{(k)}(x)}{\sum_l \alpha_l, q_t^{(l)}(x)} = \eta_k(x,t). \quad \square$$

-----

## Lemma 2 (Imbalance Arises Generically)

For any policy $Q_\theta$ over $K \geq 2$ modes, either $\alpha_k = w_k$ for all $k$, or there exist modes $j, m$ with $\alpha_j > w_j$ and $\alpha_m < w_m$.

*Proof.* Since $\sum_k \alpha_k = \sum_k w_k = 1$, we have $\sum_k (\alpha_k - w_k) = 0$. If every term is $\leq 0$, the sum is zero only if every term equals zero. Symmetrically for $\geq 0$. So if any $\alpha_k \neq w_k$, there must exist both a strictly positive and a strictly negative deviation. $\square$

*Remark.* The event $\alpha_k = w_k$ for all $k$ simultaneously is non-generic: it is a codimension-$(K{-}1)$ condition in the space of policy parameters. For any random initialization with a continuous parameter distribution, perfect balance has probability zero.

-----

## Lemma 3 (Integrated Regression Weight Equals Mode Weight)

For any policy $Q_\theta$ with mode weights $\alpha$,

$$\int \eta_k(x, t), p_{Q_\theta}(X_t \in dx) = \alpha_k \quad \text{for all } t \in [0,1].$$

*Proof.* By Proposition 1, $\eta_k(x,t) = P_{Q_\theta}(X_1 \in \mathcal{M}_k \mid X_t = x)$. Integrating over the marginal of $X_t$:

$$\int \eta_k(x,t), p_{Q_\theta}(X_t \in dx) = \int P(X_1 \in \mathcal{M}*k \mid X_t = x), p*{Q_\theta}(X_t \in dx) = P_{Q_\theta}(X_1 \in \mathcal{M}_k) = \alpha_k$$

by the law of iterated expectation. $\square$

*Interpretation.* The total regression signal that mode $k$ receives, integrated across all spatial locations, is exactly $\alpha_k$. A mode with half the target weight receives half the target regression signal. This is an identity, not an approximation.

-----

## Definition 5 (Oracle Regression Coefficient)

The oracle regression coefficient is the value $\eta_k$ would take if the policy matched the target mode weights $\alpha = w$:

$$\eta_k^*(x, t) := \frac{w_k, q_t^{(k)}(x)}{\sum_l w_l, q_t^{(l)}(x)}.$$

-----

## Definition 6 (Gram Matrix of Mode Adjoints)

At a fixed $(x, t)$, the Gram matrix of mode adjoints is

$$G(x,t) \in \mathbb{R}^{K \times K}, \qquad G_{kl}(x,t) = \bar{Y}_k(x,t)^\top \bar{Y}_l(x,t).$$

-----

## Theorem 1 (Irreducible Loss Decomposition)

The minimum achievable AM loss at a fixed $(x, t)$ decomposes as

$$\min_v \mathbb{E}*{Q*\theta}!\left[|v + g^2 Y_1|^2 \mid X_t = x\right] = g(t)^4 \left[V_{\mathrm{intra}}(x,t) + V_{\mathrm{inter}}(x,t)\right]$$

where

$$V_{\mathrm{intra}}(x,t) = \sum_{k=1}^K \eta_k(x,t), \mathbb{E}*{Q*\theta}!\left[|Y_1 - \bar{Y}_k(x,t)|^2 ,\big|, X_t = x,, X_1 \in \mathcal{M}_k\right]$$

$$V_{\mathrm{inter}}(x,t) = \sum_{k=1}^K \eta_k(x,t), \left|\bar{Y}_k(x,t) - \bar{Y}(x,t)\right|^2$$

and $\bar{Y}(x,t) = \sum_k \eta_k(x,t),\bar{Y}_k(x,t)$ is the overall conditional mean.

**Proof.**

By Lemma 1, the minimum is achieved at $v^* = -g^2 \mathbb{E}[Y_1 \mid X_t = x]$, and the residual is

$$g(t)^4,\mathbb{E}!\left[|Y_1 - \mathbb{E}[Y_1 \mid X_t = x]|^2 ,\big|, X_t = x\right] = g(t)^4,\mathrm{Var}[Y_1 \mid X_t = x].$$

We decompose $\mathrm{Var}[Y_1 \mid X_t = x]$ by conditioning on the mode label $M$. Write

$$Y_1 - \bar{Y} = (Y_1 - \bar{Y}_M) + (\bar{Y}_M - \bar{Y})$$

and expand the squared norm:

$$|Y_1 - \bar{Y}|^2 = |Y_1 - \bar{Y}_M|^2 + 2(Y_1 - \bar{Y}_M)^\top(\bar{Y}_M - \bar{Y}) + |\bar{Y}_M - \bar{Y}|^2.$$

Take $\mathbb{E}[\cdot \mid X_t = x,, M = k]$. The cross term vanishes:

$$\mathbb{E}!\left[(Y_1 - \bar{Y}_k)^\top(\bar{Y}_k - \bar{Y}) ,\big|, X_t = x,, M = k\right] = \left(\mathbb{E}[Y_1 \mid X_t = x, M = k] - \bar{Y}_k\right)^\top(\bar{Y}_k - \bar{Y}) = 0$$

since $\mathbb{E}[Y_1 \mid X_t = x, M = k] = \bar{Y}_k$ by definition.

Taking $\mathbb{E}[\cdot \mid X_t = x]$ over the mode label:

$$\mathrm{Var}[Y_1 \mid X_t = x] = \sum_k \eta_k, \mathbb{E}!\left[|Y_1 - \bar{Y}_k|^2 ,\big|, X_t = x,, M = k\right] + \sum_k \eta_k |\bar{Y}_k - \bar{Y}|^2. \quad \square$$

-----

## Corollary 1 (Collapse Reduces the Regression Loss)

The inter-mode variance satisfies:

1. $V_{\mathrm{inter}}(x,t) \geq 0$ always (it is a sum of non-negative terms).
1. $V_{\mathrm{inter}}(x,t) = 0$ if and only if either:
- (a) $\eta_k(x,t) = 1$ for some $k$ (all regression weight on a single mode at this $(x,t)$), or
- (b) all modes with $\eta_k(x,t) > 0$ share the same conditional adjoint: $\bar{Y}_k(x,t) = \bar{Y}_l(x,t)$ whenever $\eta_k, \eta_l > 0$.

Condition (b) generically fails for distinct modes with different energy landscape geometry.

Consequently, any policy that places positive regression weight on multiple modes with distinct adjoints faces a strictly larger irreducible AM loss than a collapsed policy concentrating on a single mode. The AM regression problem is structurally easier for collapsed solutions.

**Proof.** Each term $\eta_k |\bar{Y}_k - \bar{Y}|^2 \geq 0$, so the sum is non-negative. The sum equals zero if and only if every term equals zero, i.e., for each $k$, either $\eta_k = 0$ or $\bar{Y}_k = \bar{Y}$. If $\eta_k = 1$ for some $k$, then $\bar{Y} = \bar{Y}_k$ and all terms vanish, giving (a). Otherwise, if multiple modes have $\eta_k > 0$, then $\bar{Y}_k = \bar{Y}$ for all such $k$, which means they share the same conditional adjoint, giving (b). $\square$

-----

## Proposition 2 (Controller Deficiency Identity)

Let $u^{\mathrm{AM}}$ be the AM-optimal controller with policy weights $\alpha$, and $u^{\mathrm{oracle}}$ the controller that would result from AM with target weights $w$. Their difference is

$$\Delta u(x,t) := u^{\mathrm{AM}}(x,t) - u^{\mathrm{oracle}}(x,t) = -g(t)^2 \sum_{k=1}^K \bigl(\eta_k(x,t) - \eta_k^*(x,t)\bigr), \bar{Y}_k(x,t)$$

and the squared deficiency satisfies:

$$|\Delta u(x,t)|^2 = g(t)^4, \delta(x,t)^\top, G(x,t), \delta(x,t)$$

where $\delta_k(x,t) = \eta_k(x,t) - \eta_k^*(x,t)$ and $G$ is the Gram matrix (Definition 6).

If $\alpha \neq w$ and the conditional mode adjoints ${\bar{Y}*k(x,t)}*{k=1}^K$ are linearly independent at some $(x,t)$ with all conditional densities positive ($q_t^{(k)}(x) > 0$ for all $k$), then $\Delta u(x,t) \neq 0$ at that point.

**Proof.**

*Identity.* Apply Proposition 1 twice: once with the policy weights $\alpha$ (yielding regression coefficients $\eta_k$) and once with the target weights $w$ (yielding $\eta_k^*$). Subtract:

$$\Delta u = -g^2 \sum_k \eta_k \bar{Y}_k - \left(-g^2 \sum_k \eta_k^* \bar{Y}_k\right) = -g^2 \sum_k (\eta_k - \eta_k^*)\bar{Y}_k.$$

The squared norm expands as:

$$|\Delta u|^2 = g^4 \left|\sum_k \delta_k \bar{Y}*k\right|^2 = g^4 \sum*{k,l} \delta_k, \delta_l, \bar{Y}_k^\top \bar{Y}_l = g^4, \delta^\top G, \delta.$$

*Nonvanishing.* Suppose $\alpha \neq w$ and all $q_t^{(k)}(x) > 0$ at some $(x,t)$. If $\delta(x,t) = 0$, then $\eta_k(x,t) = \eta_k^*(x,t)$ for all $k$, i.e.,

$$\frac{\alpha_k, q_t^{(k)}(x)}{\sum_l \alpha_l, q_t^{(l)}(x)} = \frac{w_k, q_t^{(k)}(x)}{\sum_l w_l, q_t^{(l)}(x)} \quad \text{for all } k.$$

Since $q_t^{(k)}(x) > 0$, cancel it:

$$\frac{\alpha_k}{A(x,t)} = \frac{w_k}{W(x,t)} \quad \text{for all } k$$

where $A = \sum_l \alpha_l q_t^{(l)}$ and $W = \sum_l w_l q_t^{(l)}$. This gives $\alpha_k / w_k = A/W$ for all $k$. Summing over $k$: $1 = (A/W) \cdot 1$, so $A = W$, hence $\alpha_k = w_k$ for all $k$. Contradiction.

Therefore $\delta(x,t) \neq 0$. If the mode adjoints ${\bar{Y}_k}$ are linearly independent at this $(x,t)$, then $G(x,t)$ is positive definite, so $\delta^\top G, \delta > 0$. $\square$

*Remark (regularity).* The condition $q_t^{(k)}(x) > 0$ for all $k$ holds at every $(x,t)$ with $t < 1$ under standard SDE regularity: if the diffusion coefficient $g(t) > 0$ and the controller $u_\theta$ is locally bounded, the transition kernel of the controlled SDE has full support. The condition $K \leq d$ (necessary for linear independence of $K$ vectors in $\mathbb{R}^d$) holds in all practical settings where the number of modes is much smaller than the ambient dimension.

*Interpretation.* Whenever the policy’s mode weights differ from the target’s, the AM-optimal controller is provably wrong — not because of finite capacity or optimization error, but because the regression problem itself has the wrong weighting. The deficiency is structural and persists even with a universal function approximator trained to global optimality.

-----

## Theorem 2 (Distributional Error Lower Bound)

For any policy $Q_\theta$ with terminal mode weights $\alpha$,

$$\mathrm{KL}!\left(Q_\theta^{\mathrm{terminal}} ,|, p\right) \geq \mathrm{KL}(\alpha ,|, w) = \sum_{k=1}^K \alpha_k \log \frac{\alpha_k}{w_k}$$

and

$$\mathrm{TV}!\left(Q_\theta^{\mathrm{terminal}},, p\right) \geq \frac{1}{2}\sum_{k=1}^K |\alpha_k - w_k|.$$

**Proof.**

Define the mode-labeling map $\phi : \mathbb{R}^d \to {1,\ldots,K}$ by $\phi(x) = k$ if $x \in \mathcal{M}*k$ (defined $p$-almost everywhere and $Q*\theta$-almost everywhere by the Morse assumption in Definition 2). The pushforward of $Q_\theta^{\mathrm{terminal}}$ through $\phi$ is the categorical distribution $\mathrm{Cat}(\alpha)$, and the pushforward of $p$ is $\mathrm{Cat}(w)$.

**KL bound.** By the data processing inequality for KL divergence — for any measurable function $\phi$, $\mathrm{KL}(Q | P) \geq \mathrm{KL}(\phi_# Q | \phi_# P)$ — we have:

$$\mathrm{KL}(Q_\theta^{\mathrm{terminal}} | p) \geq \mathrm{KL}(\mathrm{Cat}(\alpha) | \mathrm{Cat}(w)) = \sum_k \alpha_k \log\frac{\alpha_k}{w_k}.$$

**TV bound.** By the data processing inequality for total variation — $\mathrm{TV}(Q, P) \geq \mathrm{TV}(\phi_# Q, \phi_# P)$ — we have:

$$\mathrm{TV}(Q_\theta^{\mathrm{terminal}}, p) \geq \mathrm{TV}(\mathrm{Cat}(\alpha), \mathrm{Cat}(w)) = \frac{1}{2}\sum_k |\alpha_k - w_k|. \quad \square$$

*Interpretation.* No matter how accurately the controller matches the intra-mode structure of the target (i.e., even if $Q_\theta(\cdot \mid X_1 \in \mathcal{M}_k) = p_k(\cdot)$ for each $k$), the wrong mode allocation imposes an irreducible distributional cost of at least $\mathrm{KL}(\alpha | w)$.

-----

## Summary of the Logical Chain

|#|Result       |Statement                                                                                         |Assumptions beyond Defs 1–4                                                |
|-|-------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
|1|Lemma 1      |AM minimizer is the conditional expectation of the adjoint                                        |None ($L^2$ optimality)                                                    |
|2|Proposition 1|AM regression weight $\eta_k \propto \alpha_k$, not $w_k$                                         |None (total expectation + Bayes)                                           |
|3|Lemma 2      |Any non-trivial policy has both over- and under-represented modes                                 |$K \geq 2$                                                                 |
|4|Lemma 3      |Total regression signal for mode $k$ equals $\alpha_k$                                            |None (iterated expectation)                                                |
|5|Theorem 1    |Irreducible AM loss decomposes; inter-mode term vanishes under collapse                           |None (total variance)                                                      |
|—|Corollary 1  |Collapsed policies face strictly easier AM regression                                             |Distinct mode adjoints (generic)                                           |
|6|Proposition 2|Controller deficiency is exactly $g^4 \delta^\top G \delta$; nonzero when $\alpha \neq w$         |Linear independence of mode adjoints + SDE regularity (generic, verifiable)|
|7|Theorem 2    |$\mathrm{KL}(Q | p) \geq \mathrm{KL}(\alpha | w)$ and $\mathrm{TV} \geq \frac{1}{2}|\alpha - w|_1$|None (data processing inequality)                                          |

**The complete chain:**

1. **Proposition 1** establishes that AM’s regression weight for each mode is proportional to the policy’s current allocation $\alpha_k$, not the target weight $w_k$.
1. **Lemma 2** establishes that mode imbalance ($\alpha \neq w$) arises generically.
1. **Lemma 3** quantifies the imbalance: the total regression signal for mode $k$ is exactly $\alpha_k$.
1. **Theorem 1** establishes that AM’s loss landscape is strictly easier for collapsed policies than balanced ones.
1. **Proposition 2** establishes that the resulting controller is provably and structurally deficient whenever $\alpha \neq w$.
1. **Theorem 2** establishes that this structural deficiency has an unavoidable distributional cost.