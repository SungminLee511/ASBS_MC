# Experimental and Evaluation Guide v2: Diagnosing AM Mode Concentration

## Goal

The v1 experiments confirmed collapse on heterogeneous targets (B5, B7, Muller-Brown) but found a puzzle: B1 symmetric shows $\hat{J}_{11}^\eta \approx 0.8$–$1.0 > 1/2$ yet exhibits 0% collapse across all separations. The theorem predicts instability; reality shows stability. We need experiments that diagnose the mechanism of this discrepancy — NOT just more confirmations of collapse.

The experiments below are organized into four diagnostic axes:

1. **Symmetry-breaking probes** — does shielding break when symmetry is broken externally?
1. **Quantitative rate verification** — do theoretical predictions match empirical rates, not just thresholds?
1. **Idealization gap measurements** — how far is real AM from exact AM, and does the gap correlate with shielding?
1. **Mechanism isolation** — which specific ingredient (architecture, initialization, SGD, batch size) provides the shielding?

Every experiment below should produce a clear yes/no verdict about a specific hypothesis.

-----

## Axis 1: Symmetry-Breaking Probes

### A1.1: Controller Bias Injection on Symmetric B1

**Hypothesis:** Symmetric B1 is stable only because nothing breaks the symmetry. Injecting an arbitrarily small asymmetric bias should trigger collapse that matches the theorem’s predicted growth rate.

**Protocol:**

- B1 symmetric, $w_1 = w_2 = 0.5$, $d = 6$.
- At epoch 0, before any training, directly modify the controller’s final-layer bias to produce initial $\alpha_1 \in {0.50, 0.51, 0.52, 0.55, 0.60, 0.70}$. Achieve this by generating samples with a small constant drift toward mode 1 during the first forward rollout.
- Train for 1000 epochs, 10 seeds per initial $\alpha_1$.

**Measure:**

- $\alpha_1^{(n)}$ vs epoch for each initial condition.
- Empirical growth rate of $|\alpha_1^{(n)} - 0.5|$ in the early linear regime.
- Compare growth rate to $\lambda_{\max}^\eta$ measured at the fixed point.

**Verdict criteria:**

- If initial $\alpha_1 = 0.51$ diverges to one-sided collapse → theorem holds, symmetric B1 was shielded by exact symmetry.
- If initial $\alpha_1 = 0.51$ returns to 0.5 → real ASBS has a restoring force not in the theorem. Proceed to A1.2.

-----

### A1.2: Architecture Symmetry Breaking

**Hypothesis:** The controller network has implicit symmetric inductive bias that restores balance. Breaking architectural symmetry should eliminate the shielding.

**Protocol:**

- B1 symmetric, fixed $d = 6$.
- Train ASBS with three architecture variants:
  - **Symmetric:** standard MLP controller.
  - **Frozen-asymmetric:** add a fixed random bias vector to the controller output at every step. The bias is drawn once and never updated. Use 5 different bias scales: ${0, 0.01, 0.05, 0.1, 0.5}$.
  - **Mode-aware:** duplicate the network into two copies, one for each “side” of the x-axis, trained independently on the respective samples. This removes parameter sharing across the symmetry axis.
- 10 seeds per variant.

**Measure:** Final $\mathrm{KL}(\alpha | w)$, collapse rate (binary: $|\alpha_1 - 0.5| > 0.3$ by epoch 500).

**Verdict criteria:**

- If frozen-asymmetric collapse increases monotonically with bias scale → architecture symmetry IS the shielding mechanism.
- If mode-aware variant collapses catastrophically → parameter sharing across modes is the shielding mechanism.
- If all variants behave the same → shielding is not architectural; proceed to A1.3.

-----

### A1.3: Initialization Symmetry Breaking

**Hypothesis:** Network initialization is symmetric in expectation, which biases SGD toward symmetric solutions. Non-symmetric initialization should break shielding.

**Protocol:**

- B1 symmetric, fixed $d = 6$.
- Train with three initialization schemes:
  - **Standard:** PyTorch default initialization.
  - **Biased initialization:** after standard init, add a small constant $\epsilon \in {0.01, 0.05, 0.1, 0.3}$ to all weights in the final layer.
  - **Pretrained asymmetric:** pretrain the controller for 50 epochs on a 80/20 asymmetric version of B1, then fine-tune on symmetric B1 for 1000 epochs.
- 10 seeds each.

**Measure:** $\alpha_1^{(n)}$ trajectory, final KL.

**Verdict criteria:**

- If biased init collapses (collapse rate increasing with $\epsilon$) → initialization symmetry IS the shielding.
- If pretrained asymmetric collapses faster / deeper than symmetric init → initialization determines the basin of attraction.
- If all collapse or all stay balanced → the shielding is not at initialization.

-----

## Axis 2: Quantitative Rate Verification

These experiments test whether the theorem’s *predictions* match reality, not just its thresholds. A threshold check (”$\lambda > 1/2$ should cause collapse”) is binary and weak. A rate check (“collapse speed should be $\log\lambda$”) is quantitative and strong.

### A2.1: Growth Rate Matching on B5 and B7

**Hypothesis:** In the regime where collapse occurs, the rate of $|\alpha - w|$ growth matches the theorem’s predicted rate from $\lambda_{\max}^\eta$.

**Protocol:**

- B5 at center_scale $\in {4, 5, 7}$ (where collapse is 80-100%).
- B7 at $\kappa_3 = 20$ (where collapse is 40%).
- For each configuration, identify the “collapsing” seeds and extract the time window where collapse is happening (typically epochs 20-200 before saturation).
- Plot $\log|\alpha^{(n)} - w|$ vs epoch. Fit a linear slope over the linear-growth region.
- Separately, estimate $\lambda_{\max}^\eta$ by the perturbation method (small $\delta\alpha$, measure $\delta\alpha’$, ratio).

**Measure:** Empirical slope vs $\log\lambda_{\max}^\eta$.

**Verdict criteria:**

- Slope $\approx \log\lambda_{\max}^\eta$ within a factor of 2 → quantitative confirmation of the theorem in the regime where it applies.
- Slope systematically different → theorem predicts qualitatively but not quantitatively; residual from bridge response or SGD noise is non-negligible.

-----

### A2.2: Multi-Epoch Jacobian Consistency

**Hypothesis:** The linearization $\delta\alpha^{(n+1)} = J^\eta \delta\alpha^{(n)}$ is accurate for several epochs, not just one.

**Protocol:**

- B1 with $w = (0.6, 0.4)$ (mild asymmetry, in the regime where theorem should apply).
- Train to approximate fixed point $\alpha \approx (0.6, 0.4)$.
- Apply a known perturbation $\delta\alpha^{(0)} = (0.02, -0.02)$.
- Run 1 AM epoch, measure $\delta\alpha^{(1)}$. Compute empirical $J^{(1)} = \delta\alpha^{(1)} / \delta\alpha^{(0)}$.
- Continue: measure $\delta\alpha^{(2)}, \delta\alpha^{(3)}, \ldots$
- Predict: under linear theory, $\delta\alpha^{(n)} = (J^\eta)^n \delta\alpha^{(0)}$.

**Measure:** Ratio of predicted to actual $\delta\alpha^{(n)}$ for $n = 1, 2, 5, 10$.

**Verdict criteria:**

- Ratio $\approx 1$ for $n \leq 3$, drifting for larger $n$ → linearization valid near fixed point, breaks down at larger perturbations (expected).
- Ratio far from 1 even at $n = 1$ → the Jacobian estimate itself is wrong; investigate why.

-----

### A2.3: Asymmetry-Resolved Phase Diagram

**Hypothesis:** Collapse severity scales continuously with asymmetry, not as a binary symmetric/asymmetric cliff.

**Protocol:**

- B1 with $w_1 \in {0.50, 0.51, 0.52, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90}$.
- Fixed $d = 6$. 10 seeds each.

**Measure:** Final $\mathrm{KL}(\alpha | w)$, final $|\alpha_1 - w_1| / w_1$, collapse rate.

**Plot:** Collapse severity vs $|w_1 - 0.5|$ on log-log axes. If theorem is right, expect monotonic growth. If there’s a sharp threshold, expect a cliff. The answer tells you whether shielding is continuous or binary.

**Verdict criteria:**

- Monotonic curve → shielding is continuous; symmetry is one point on a smooth axis.
- Sharp threshold at some $w_1^* > 0.5$ → there’s a critical asymmetry; shielding has a width.
- Constant KL up to $w_1 \approx 0.8$ then jump → shielding is very strong for modest asymmetry; something qualitative happens only at extreme weights.

-----

## Axis 3: Idealization Gap Measurements

These measure how far real AM deviates from exact AM regression, testing whether the deviation correlates with shielding.

### A3.1: AM Regression Residual Decomposition

**Hypothesis:** “Exact AM” assumes the controller reaches the regression’s global optimum each epoch. Real SGD leaves a residual. The residual’s structure (mode-specific or mode-agnostic) determines whether shielding happens.

**Protocol:**

- On B1 symmetric and B1 asymmetric (80/20), at each epoch after training:
- Decompose the AM regression residual $|u_\theta - u^{\text{AM-opt}}|^2$ where $u^{\text{AM-opt}}$ is the ideal AM target.
- Split the residual by mode: $R_k = \mathbb{E}[|u_\theta - u^{\text{AM-opt}}|^2 \mid X_1 \in \mathcal{M}_k]$.

**Measure:** $R_1 / R_2$ ratio across epochs, for both symmetric and asymmetric B1.

**Verdict criteria:**

- Symmetric B1 has $R_1 \approx R_2$ (symmetric residual) → SGD preserves symmetry in residuals, which restores $\alpha$.
- Asymmetric B1 has $R_1 \neq R_2$ systematically → SGD’s residual reflects the asymmetric structure and doesn’t restore.
- Both have similar structure → the residual is not the mechanism.

-----

### A3.2: Network Capacity Ablation

**Hypothesis:** Larger networks approach “exact AM” better, so they should collapse more (closer to the theorem’s idealization). Smaller networks are further from exact, so they’re more shielded.

**Protocol:**

- B1 symmetric and B5 at center_scale=4 (near phase transition).
- Network widths: ${32, 64, 128, 256, 512, 1024}$ hidden units (keep depth fixed at 3 layers).
- 5 seeds each.

**Measure:** AM regression residual (the idealization gap), final $\mathrm{KL}(\alpha | w)$, collapse rate.

**Plot:** Collapse rate vs network width, for both benchmarks.

**Verdict criteria:**

- On B1 symmetric: collapse increases with width → larger networks ARE approaching the theorem’s predictions; small networks are shielded by capacity limits. This would be a clean confirmation of the idealization-gap hypothesis.
- On B1 symmetric: collapse stays at 0 regardless → capacity is not the mechanism; something deeper (symmetry, not approximation) is shielding.
- On B5: collapse rate changes with width → the boundary between stable/unstable regimes shifts with capacity.

-----

### A3.3: Batch Size and Learning Rate Ablation

**Hypothesis:** SGD noise provides a restoring force. Smaller batches → more noise → more shielding. Larger batches → closer to gradient flow → more collapse.

**Protocol:**

- B1 symmetric, fixed network and data.
- Batch sizes: ${32, 64, 128, 512, 2048}$ (keep total compute constant by adjusting steps per epoch).
- Learning rates: ${1e-5, 1e-4, 1e-3, 1e-2}$.
- 5 seeds each configuration.

**Measure:** Collapse rate, final KL, $\alpha_1$ variance across seeds.

**Verdict criteria:**

- Collapse appears at large batch / low LR → SGD noise IS the shielding mechanism.
- Variance grows with small batches but mean stays at 0.5 → SGD noise adds exploration but doesn’t break symmetry.
- No effect → shielding is not from SGD noise.

-----

## Axis 4: Mechanism Isolation

These experiments isolate individual components to identify which piece of real AM differs from the idealization.

### A4.1: Exact vs Approximate AM Comparison

**Hypothesis:** Replacing SGD with “as-exact-as-possible” AM should increase collapse.

**Protocol:**

- On B1 symmetric and a small asymmetric variant (w = 0.55/0.45):
- Standard ASBS: SGD on the neural network.
- Over-trained ASBS: do 10x more SGD steps per AM epoch, until regression loss plateaus.
- Kernel-based AM: replace the neural network with a kernel regression estimator over a fine spatial grid. This is computationally expensive but can be made arbitrarily close to exact on 2D problems.

**Measure:** Final $\mathrm{KL}(\alpha | w)$, time to collapse (if any).

**Verdict criteria:**

- Kernel-based AM collapses on symmetric B1 where standard ASBS doesn’t → the theorem is exactly right about exact AM; SGD is the shielding.
- All three behave similarly → SGD approximation is not the shielding mechanism.

-----

### A4.2: Controller Parameter Trajectory Analysis

**Hypothesis:** The controller parameters evolve along a low-dimensional symmetric manifold on symmetric targets. On asymmetric targets, they explore a higher-dimensional space.

**Protocol:**

- Record controller parameter vector $\theta^{(n)}$ every 10 epochs during B1 symmetric and B1 asymmetric training.
- Run PCA on the trajectories ${\theta^{(n)}}_n$ for each seed.

**Measure:** Effective dimensionality (number of PCs explaining 95% variance); symmetry of the principal directions under the mode-swap operation.

**Verdict criteria:**

- Symmetric B1 trajectories have lower effective dim, and PCs are symmetric under mode-swap → SGD is constrained to the symmetric subspace.
- Asymmetric B1 trajectories have higher dim → SGD explores asymmetric directions.

-----

### A4.3: Seed Variance Analysis

**Hypothesis:** If SGD truly explores the full parameter space, different seeds should find different asymmetric basins. If SGD is constrained to symmetric solutions, all seeds converge to the same balanced attractor.

**Protocol:**

- B1 symmetric, $d \in {3, 6, 10}$, 50 seeds each.
- For each seed, at convergence, record: $\alpha_1$, controller output at 10 fixed $(x, t)$ evaluation points.

**Measure:**

- Distribution of $\alpha_1$ across 50 seeds (histogram).
- Clustering of controller outputs (are there multiple attractors?).

**Verdict criteria:**

- Single-peaked $\alpha_1 \approx 0.5$ distribution, controllers cluster into one group → single symmetric attractor.
- Bimodal $\alpha_1$ distribution (peaks at 0.3 and 0.7), controllers in two clusters → symmetry-broken attractors exist but the 50-seed sample hasn’t hit them.
- Broad $\alpha_1$ distribution → continuous family of solutions.

-----

## Axis 5: Detailed Evaluation on Existing Runs

These don’t require new training — just new analyses on the existing v1 experiment outputs.

### A5.1: Full Jacobian Eigenvalue Analysis

**For all existing B5, B7, Muller-Brown runs:**

- At each checkpoint, estimate the full $J^\eta$ matrix (not just $J_{11}$).
- Compute all eigenvalues of $J^\eta |_{T\Delta}$.
- Plot the spectrum vs epoch.

**Goal:** See whether collapse is driven by a single dominant eigenvalue (one-dimensional instability) or by multiple simultaneously-unstable directions. This distinguishes “one mode dies” from “many modes die in parallel.”

-----

### A5.2: $\eta_k$ Field Temporal Evolution

**For B5, B7, Muller-Brown:**

- At epochs ${10, 50, 100, 200, 500, 1000}$, evaluate $\eta_k(x, t)$ on a dense grid.
- Animate the evolution as a video or grid of heatmaps.

**Goal:** Observe the dying mode’s $\eta_k$ field shrinking over time. This is direct visualization of Proposition 1’s mechanism — the regression weight for the dying mode contracts spatially, accelerating its death.

-----

### A5.3: Adjoint Target $\bar{Y}_k$ Drift

**Hypothesis:** As modes collapse, the conditional adjoint $\bar{Y}_k$ for the dying mode drifts (because the bridge distribution changes). This drift is the “bridge response” $\delta\bar{Y}_k$ that Lemma D2 claims is exponentially small under mode separation.

**Protocol:** For existing B5, B7 runs, at each checkpoint:

- Estimate $\bar{Y}_k(x, t)$ for each surviving mode by averaging adjoint values conditioned on terminal mode.
- Compute $|\bar{Y}_k^{(n)} - \bar{Y}_k^{(0)}|$ over time.

**Verdict criteria:**

- If the drift is small for most modes but large for the dying mode → the bridge response isn’t negligible for collapsing modes, and Lemma D2’s “uniform bound” is conservative.
- If drift is uniformly small → Lemma D2 is tight.

-----

### A5.4: Symmetry Diagnostic on B1 and B5

**Test:** For B1 symmetric, check whether the learned controller $u_\theta(x, t)$ satisfies the reflection symmetry $u_\theta(-x, t) = -u_\theta(x, t)$ (up to coordinate swap). For B5, check if any residual symmetries are preserved.

**Goal:** Quantify how symmetric the learned solution actually is. If B1 symmetric produces nearly perfectly symmetric controllers, that’s strong evidence that symmetry is preserved throughout training. If B5 produces broken symmetries (even when collapse hasn’t happened yet), that’s evidence that symmetry-breaking precedes collapse.

**Measure:** Symmetry residual $|u_\theta(x,t) + u_\theta(R x, t)|$ where $R$ is the relevant symmetry transformation.

-----

## Priority Ordering

**Tier 1 (critical for theory correction, run these first):**

1. A1.1 (controller bias injection) — directly tests whether symmetric B1 is shielded by symmetry alone.
1. A2.3 (asymmetry-resolved phase diagram) — connects B1 (null) to 80/20 (collapsing).
1. A3.2 (network capacity ablation on B1 symmetric) — tests if larger networks approach the theorem.
1. A5.4 (symmetry diagnostic) — no new training, just analysis.

**Tier 2 (mechanism identification):**
5. A4.1 (exact vs approximate AM) — isolates SGD as the shielding mechanism.
6. A1.3 (initialization symmetry breaking) — identifies whether init matters.
7. A3.3 (batch size / LR ablation) — tests SGD noise hypothesis.
8. A2.1 (growth rate matching on B5, B7) — quantitative theorem verification in collapsing regime.

**Tier 3 (deeper characterization):**
9. A1.2 (architecture ablation).
10. A4.2 (parameter trajectory PCA).
11. A4.3 (seed variance).
12. A5.1, A5.2, A5.3 (detailed evaluations on existing runs).
13. A2.2 (multi-epoch Jacobian consistency).

-----

## Expected Outcomes and What Each Tells You

|Result Pattern                                                                      |Interpretation                                             |Theory Implication                                                                                        |
|------------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
|A1.1 injection → collapse, A3.2 larger net → more collapse, A4.1 exact AM → collapse|Theorem is exactly right, SGD is the shielding             |Restrict theorem’s conclusion to exact AM; characterize SGD bias separately                               |
|A1.1 injection → no collapse, A3.2 no effect, A4.1 no effect                        |Shielding is fundamental, not about approximation          |Theorem is wrong or incomplete; needs reformulation on symmetry grounds                                   |
|A2.3 continuous curve, A1.1 weak injection → slow divergence                        |Symmetry is a measure-zero edge case of a continuous theory|Theorem is correct generically; symmetric case is a non-generic fixed point with extra structure          |
|Mixed results across A1/A3/A4                                                       |Multiple shielding mechanisms                              |Theorem applies in the generic heterogeneous case; multiple factors conspire to preserve symmetric targets|

The most likely outcome, based on the v1 data, is the third row: the theorem is correct in the generic case, and symmetric targets are a non-generic fixed point that requires separate treatment. This is the cleanest scientific story and also the most publishable.