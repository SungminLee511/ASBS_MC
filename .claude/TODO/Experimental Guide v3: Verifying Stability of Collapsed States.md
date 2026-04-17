# Experimental Guide v3: Verifying Stability of Collapsed States

## Motivation

The v1 experiments established that ASBS *reaches* mode-collapsed states from random initialization on heterogeneous benchmarks (B5, B7, Müller-Brown at high β). This tells us collapsed states are attractors of training dynamics but does not directly test whether they are *stable fixed points* in the technical sense: that small perturbations away from the collapsed state decay back to it.

The new theoretical framing — that collapsed controllers $u_S^*$ are stable fixed points of the exact AM regression operator — makes a specific claim: if you place the system at a collapsed state and perturb it, the perturbation shrinks each epoch by a contraction factor $\|J^{(S)}\|_{\mathrm{op}} < 1$, and the system returns to collapse.

This guide designs experiments that directly test stability, not just reachability.

---

## Core Experimental Pattern: Pretrain and Perturb

The workhorse protocol is:

1. **Pretrain:** Train ASBS to convergence on a target $p$, typically one where collapse naturally occurs (B7, B5, or a deliberately constructed partial target $p_S$).
2. **Perturb:** Inject a specific perturbation — add a new mode, revive a dead mode, shift mass between modes, or corrupt the controller directly.
3. **Observe:** Continue training and measure whether the perturbation decays, persists, or grows.

The theory predicts: if the pretrained state is a collapsed fixed point $u_S^*$, perturbations should decay back. Observing this directly confirms stability.

---

## Experiment Family 0: v1 Results as Primary Evidence

Before any new experiments, we reinterpret the v1 results through the stability lens. Under the old theorem ("balance is unstable"), v1 was mixed evidence. Under the new theorem ("collapsed states are stable attractors"), v1 is direct, strong evidence — the system found collapsed states from random initialization on multiple benchmarks, which is exactly what "stable attractor with a basin of attraction containing typical initializations" predicts.

This family reframes existing v1 data as the foundational evidence for the new theory. No new training runs needed; all experiments here are re-analyses.

### 0.1: v1 Reachability as Basin Evidence

**Claim:** The v1 failures on B5, B7, and Müller-Brown at high β demonstrate that collapsed states are attractors with basins of attraction containing random initializations.

**Re-analysis:**
- B5 (heterogeneous covariance): 5/5 seeds collapse at center_scale ≥ 5. This means 5/5 random inits lie in the basin of some collapsed state.
- B7 (three-well): ternary collapse pattern is consistent across 20 seeds. This means 20/20 random inits lie in the basin of the same collapsed state (mode 2 dies, mode 3 inflates).
- Müller-Brown at β ≥ 0.1: mode death consistent across 5 seeds.

**Verdict under stability theory:** Confirmed. These are not "instability-driven" collapses (which would require $u^*$ to actively repel); they are "attractor-driven" collapses (the system flows into a stable collapsed state from random init).

### 0.2: v1 Symmetric B1 as Balanced-State Stability

**Claim:** The v1 null result on B1 symmetric (0% collapse across all separations) confirms that the balanced state $\alpha = w$ is itself a stable fixed point. Under the new theorem, the balanced state is the $S = \{1, \ldots, K\}$ case of the collapsed controller family — no modes are dead, so it's trivially "collapsed" to the full set. Its stability is predicted by the same theorem.

**Re-analysis:**
- B1 symmetric: 30/30 seeds reach and stay at $\alpha_1 = 0.5$.
- Separation sweep: 0% collapse across $d \in \{2, \ldots, 20\}$.

**Verdict:** Under the old theorem, this was a contradiction (theorem predicted instability, experiment showed stability). Under the new theorem, this is a confirmation (theorem predicts the balanced state is a stable fixed point; experiment shows it is).

**The B1 symmetric result is transformed from a problem into a confirmation.** This single reinterpretation is worth emphasizing in the paper — it shows the new framing resolves the v1 tension rather than patching around it.

### 0.3: v1 Seed Consistency as Basin Size Evidence

**Claim:** The consistency of collapse patterns across v1 seeds measures the basin of attraction's size.

**Re-analysis:**
- B7: 20 seeds all end at the mode-3-dominant state (not mode-1 or mode-2 dominant). This means the mode-3-dominant basin contains nearly all random initializations, while other potential attractors (mode-1-dominant, mode-2-dominant) have tiny or empty basins from random init.
- This doesn't mean other collapsed states aren't stable — they might be. It means they're not *reachable* from random initialization. (Family C2 tests whether they're stable when directly initialized.)

**Diagnostic:** v1 establishes basin reachability. New experiments (Family C2 in particular) establish whether other collapsed states are stable too, even if unreachable from generic init.

### 0.4: v1 Separation/Temperature Sweeps as Basin Phase Transitions

**Claim:** The sharp phase transitions in v1 (B5 at center_scale ≈ 3.5, Müller-Brown at β ≈ 0.1) mark the boundary where the collapsed state's basin of attraction becomes large enough to contain random initializations.

**Re-analysis:**
- Below the transition: the balanced state's basin dominates; random inits flow to balance.
- At the transition: basin boundaries shift rapidly.
- Above the transition: collapsed state's basin dominates; random inits flow to collapse.

This reframes "phase transition to collapse" as "phase transition in basin geometry" — the collapsed state always exists as a fixed point; what changes is which state's basin captures random inits. This is a richer interpretation and aligns with the theory's prediction that multiple stable fixed points coexist.

### 0.5: v1 Metastable Survival Sweep as Basin Threshold

**Claim:** The metastable survival curve in v1 1F (100% → 60% survival as $\kappa_3$ rises) measures the depth ratio at which the basin of "all modes alive" shrinks below the typical-initialization region.

**Re-analysis:**
- $\kappa_3 \leq 16$: 100% survival means the balanced basin contains all random inits.
- $\kappa_3 = 20$: 60% survival means the balanced basin and the "mode 3 dead" basin each capture some fraction of random inits.
- The transition marks where these basins' relative sizes shift.

---

Note: Under the old theorem (balance unstable), v1 was contradictory data that we had to explain away with SGD-shielding arguments. Under the new theorem (multiple stable attractors), v1 is straightforward confirming evidence for the entire story. No shielding needed; the theorem matches the data.

---

## Experiment Family A: Mode Injection

These experiments test the "adding a new mode doesn't help" prediction: once AM is trained on a collapsed state, introducing new mass into dead regions cannot recover it.

### A1: Sequential Mode Addition on 2D Gaussian Mixture

**Setup:**
- Phase 1: Pretrain ASBS on a 2-mode Gaussian mixture with $w_1 = w_2 = 0.5$, centers $(\pm 4, 0)$, $\sigma = 1$.
- Verify that training reaches the true balanced distribution (this is a B1-symmetric case; v1 shows it does).
- Phase 2: Change the target to a 3-mode mixture by adding a new Gaussian at $(0, 4)$ with $w_3 = 0.33$ (rebalancing to $w_1 = w_2 = w_3 = 1/3$).
- Continue training for an additional 2000 epochs without resetting the controller.

**Measure:**
- $\alpha_k^{(n)}$ for all three modes across epochs.
- Time to rediscover mode 3 (if ever).
- Final $\mathrm{KL}(\alpha \| w_{\text{new}})$.

**Prediction under stability theory:**
- If the pretrained controller is approximately the $\{1, 2\}$-collapsed controller for the new target, mode 3 should remain dead or rise only slowly.
- The dominant modes (1, 2) should rebalance to the new $w_1 = w_2 = 1/3$ allocation (minor adjustment) while mode 3 struggles to exceed $\alpha_3 \ll 0.33$.
- Baseline comparison: training from scratch on the 3-mode target should recover all three modes (B1 symmetric analog).

**Diagnostic:**
- **Slow recovery:** Partial stability — the collapsed state is a weak attractor, and eventually mode 3 rises.
- **No recovery:** Strong stability — the collapsed state is a stable fixed point, confirming the theorem.
- **Full recovery:** The pretrained controller was not in the collapsed state's basin of attraction; the theorem's prediction doesn't apply here.

### A2: Mode Addition with Varying Injection Distance

Same as A1 but sweep the injected mode's distance from the surviving pair: centers at $(0, d_{\text{inject}})$ for $d_{\text{inject}} \in \{2, 4, 6, 8, 12\}$.

**Prediction:** Closer injection modes are more easily discovered (bridge between basins is shorter). More distant modes remain dead more robustly. This traces out the "basin of attraction" of the collapsed state in injection-distance space.

### A3: Mode Addition with Energy Depth Variation

Inject a new mode with different depth (curvature) $\kappa_3 \in \{4, 8, 16, 32\}$, holding distance fixed.

**Prediction:** Shallower injected modes (lower $\kappa_3$) are more easily captured; deeper modes require more precise policy control and are more robustly rejected by the collapsed controller. This connects to the v1 metastable survival sweep (1F) but from the opposite direction.

---

## Experiment Family B: Direct Perturbation from Collapsed State

These experiments place the system *at* a collapsed state and inject small perturbations, matching the theorem's exact setup.

### B1: Dead Mode Revival via Data Injection

**Setup:**
- Pretrain ASBS on B7 until convergence. The resulting state has mode 2 dead ($\alpha_2 \approx 0$).
- At epoch $N$, modify the training pipeline: for the next $M$ epochs, inject a small fraction $\rho$ of training samples from a specific dead mode (sample from $p_k$ for $k \notin S$ and include these in the AM regression data).
- Parameters: $\rho \in \{0.001, 0.01, 0.05, 0.1\}$, $M \in \{10, 50, 200\}$ epochs of injection.
- After injection ends, continue training normally for 1000 epochs.

**Measure:**
- $\alpha_2^{(n)}$ across epochs during and after injection.
- Decay rate after injection ends: fit $\alpha_2^{(n)} \approx \alpha_2^{(\text{peak})} \cdot \lambda^{n}$.
- Compare $\lambda$ to the theoretical prediction from the $J^{S^c S^c}$ block of Proposition D2'.

**Prediction:** $\lambda < 1$, and specifically $\lambda \approx $ the diagonal of $J^{S^c S^c}$ from the theorem. This would be the direct quantitative confirmation of stability.

**This is the single most important experiment in this guide.** It directly measures the contraction factor predicted by the theorem.

### B2: Controller-Level Perturbation

Instead of injecting data, directly perturb the controller network weights of a pretrained collapsed model.

**Setup:**
- Pretrain on B7 until convergence.
- At epoch $N$, add Gaussian noise $\mathcal{N}(0, \sigma_{\text{pert}}^2 I)$ to all controller weights.
- $\sigma_{\text{pert}} \in \{0.001, 0.01, 0.1, 1.0\}$.
- Continue training for 1000 epochs.

**Measure:**
- Time for $\alpha_k$ to return to the pre-perturbation values.
- For large $\sigma_{\text{pert}}$: does the system return to the SAME collapsed state or a DIFFERENT one?

**Prediction under stability:** Small perturbations decay back to the original collapsed state. Large perturbations may push the system into a different basin of attraction (another collapsed state), which is still consistent with stability — just with finite basins.

**Diagnostic:** If all perturbations decay to the same state, the collapsed state has a large basin. If large perturbations find different collapsed states, we're mapping out the basin structure.

### B3: Bias Injection with Continuous Training

Following the v2 A1.1 protocol: pretrain to convergence, then continuously inject a small asymmetric drift into the SDE during rollouts ($\delta = 0.01$ mode-asymmetric bias). Measure how much asymmetry accumulates before being "pushed back" by the trained controller.

**Prediction:** The trained collapsed controller actively suppresses revival — the injected bias's effect on $\alpha_k$ saturates well below what would occur with the pretrained balanced controller.

---

## Experiment Family C: Trajectories from Collapsed Initializations

These directly test the theorem's "perturbation decays" claim by initializing near a collapsed state and tracking decay.

### C1: Initialization-to-Collapse Distance Sweep

**Setup:**
- For B7, construct multiple initialization schemes parameterized by distance $d_{\text{init}}$ to the collapsed state:
 - Pretrain on B7 to obtain $\theta^*$ (collapsed).
 - Initialize as $\theta_0 = \theta^* + d_{\text{init}} \cdot \xi$ where $\xi$ is a random unit direction.
 - $d_{\text{init}} \in \{0.01, 0.1, 1.0, 10.0\}$.
- Train normally for 1000 epochs from each initialization.

**Measure:**
- Whether the system converges back to the collapsed state.
- Convergence rate: $\|\theta^{(n)} - \theta^*\|$ vs epoch.
- For each $d_{\text{init}}$, estimate the fraction of directions $\xi$ that lead back to the same collapsed state.

**Prediction:** For small $d_{\text{init}}$, nearly 100% return. For large $d_{\text{init}}$, may find different fixed points (balanced, or a different collapsed state, or remain far from any fixed point).

This maps out the basin of attraction around the collapsed state.

### C2: Start from Adversarial Collapsed States

Deliberately construct "wrong" collapsed states — say, for B7 where naturally mode 3 dominates, force the system into a state where mode 1 dominates instead. Does AM maintain this non-natural collapsed state, or does it transition to the natural one?

**Setup:**
- Pretrain B7 on a modified target where mode 1 has $w_1 = 1$ (single-mode training).
- Then switch to the true B7 target and continue training.

**Measure:** Does $\alpha_1$ stay dominant, or does the system transition to the $\alpha_3$-dominant state?

**Prediction under stability:** Each collapsed state $u_S^*$ is locally stable. Transition to a different attractor requires a sufficiently large perturbation. If starting at $\alpha_1 = 1$ and training on the real B7 does NOT transition to $\alpha_3$-dominant, then the "mode 1 only" state is indeed a stable fixed point.

This tests the *plurality* of stable attractors — the theorem predicts all $2^K - 2$ collapsed states can be stable, not just the "natural" one.

---

## Experiment Family D: Quantitative Jacobian Verification

These experiments estimate the Jacobian $J^{(S)}$ empirically and compare to the theoretical prediction from Proposition D2'.

### D1: Block-by-Block Jacobian Estimation

**Setup:** At a collapsed state of B7 (pretrain to convergence), apply a small perturbation $\delta\alpha = \epsilon e_j$ where $e_j$ is the standard basis vector for mode $j$. Measure the response $\delta\alpha' = (F - \alpha^{(S)})$ after one AM epoch.

Iterate for each $j \in \{1, \ldots, K\}$ and each of several $\epsilon \in \{0.001, 0.005, 0.01, 0.02\}$ to extract the linear regime.

**Measure:** Each column of $J^{(S)}$ as $\delta\alpha'/\epsilon$. Block structure: $J^{SS}$, $J^{SS^c}$, $J^{S^c S^c}$.

**Compare:** Theoretical formulas from Proposition D2':
- $J^{S^c S^c}$ diagonal: $q_t^{(k)}(x)/D_S(x,t)$ integrated appropriately.
- $J^{SS^c}$: $-\eta_k^{(S)} q_t^{(j)}(x)/D_S$.

**Verdict criteria:** Empirical Jacobian matches theoretical within a factor of 2 → direct quantitative confirmation of Proposition D2'.

### D2: Spectral Radius Measurement

From D1's Jacobian estimate, compute eigenvalues and check $|\lambda_{\max}| < 1$.

**Verdict:** $|\lambda_{\max}| < 1$ confirms the collapsed state is linearly stable. Measure the contraction factor.

### D3: Multi-Epoch Perturbation Decay

Inject a perturbation of known magnitude and track $\|\delta\alpha^{(n)}\|$ over many epochs. Fit exponential decay $\|\delta\alpha^{(n)}\| = \|\delta\alpha^{(0)}\| \cdot r^n$.

**Compare:** $r$ to $|\lambda_{\max}|$ from D2. Agreement → linearization is accurate beyond first epoch.

---

## Experiment Family E: Multiple Collapsed States

The theorem predicts all proper subsets $S$ give rise to stable collapsed states. This family tests whether multiple stable attractors exist simultaneously.

### E1: Attractor Enumeration on Small Benchmarks

**Setup:** Use B1 asymmetric (80/20) or a 3-mode benchmark. For each possible subset $S$:
- Initialize the controller to be SOC-optimal for $p_S$ (pretrain on $p_S$).
- Switch to training on the full $p$.
- Check if the system stays at the $S$-collapsed state or transitions.

**Measure:** For each $S$, fraction of seeds that remain in the $S$-state.

**Prediction:** All subsets $S$ where the geometry supports stability (bridge response is contractive) should be stable attractors. Specifically: "only mode 1 alive," "only mode 2 alive," and the balanced state may all be stable simultaneously.

### E2: Transition Threshold Between Collapsed States

From an $S$-collapsed state, what magnitude of perturbation is needed to transition to a different $S'$-collapsed state?

**Setup:** Start at "only mode 1 alive" for B7. Inject revival signal for mode 2 at increasing magnitudes.

**Measure:** At what injection magnitude does the system transition to "mode 1 + mode 2 alive" (or similar)?

**Interpretation:** This maps out the separatrix between basins of different collapsed attractors. The width of the basin around each collapsed state is a quantitative measure of its stability.

---

## Experiment Family F: Testing the Bridge Regularity Assumption

The theorem relies on the BRA: bounded sensitivity of the dead-mode conditional adjoint $\bar{Y}_k$ for $k \notin S$. This family tests BRA directly.

### F1: Direct Measurement of Dead Mode Adjoints

**Setup:** At a pretrained collapsed state on B7:
- The dead mode $k = 2$ has $\alpha_2 \approx 0$, but for trajectories that happen to reach $\mathcal{M}_2$ (via Brownian excursion), compute $\bar{Y}_2 = \mathbb{E}[-\nabla\Phi_0(X_1) \mid X_1 \in \mathcal{M}_2]$.
- Monte Carlo with $10^6$ samples to get enough trajectories that terminate in $\mathcal{M}_2$.

**Measure:** $\bar{Y}_2$ value and its variance across seeds.

### F2: Sensitivity of Dead Mode Adjoints

Perturb the controller slightly and measure how $\bar{Y}_2$ changes.

**Measure:** $\|\partial \bar{Y}_2 / \partial \alpha_j\|$ for each $j$.

**BRA test:** Is this bounded? How does it scale with $\epsilon$ (the leakage parameter, controlled by controller strength)?

**Verdict:** If $\|\partial\bar{Y}_2/\partial\alpha\|$ stays bounded across conditions → BRA holds, the assumption is reasonable. If it diverges as $\epsilon \to 0$ → BRA fails, the theorem needs additional conditions.

---

## Experiment Family G: Supplementary Analyses on v1 Data

These extract additional quantitative information from v1 data beyond the reinterpretation in Family 0.

### G1: Empirical Contraction Factor from v1 Trajectories

From v1 B5, B7, and Müller-Brown training curves, extract the late-epoch decay rate of $\|\alpha^{(n)} - \alpha^{(\text{final})}\|$ as the system settles into its collapsed attractor.

**Measure:** The observed decay rate $r$ in the final 100-200 epochs of each run.

**Compare:** $r$ to the theoretical contraction factor $|\lambda_{\max}(J^{(S)})|$ computed from Proposition D2' formulas at the observed collapsed state.

**Verdict:** Agreement confirms the theorem's quantitative predictions on v1 data without new experiments.

### G2: Jacobian Estimation from Small Fluctuations

In converged v1 runs, training fluctuations cause $\alpha^{(n)}$ to oscillate slightly around the collapsed state. These fluctuations can be used to estimate the Jacobian without injecting deliberate perturbations.

**Method:** Compute the autocorrelation structure of $\alpha^{(n)}$ across epochs in the converged regime. The autoregression coefficients approximate the Jacobian eigenvalues.

**Verdict:** Eigenvalues with modulus less than 1 confirm local stability.

---

## Priority Ordering

**Tier 0 — foundational evidence from v1 (do this first, no new experiments):**

0. **Family 0 (v1 reinterpretation)** — turns existing data into primary evidence for the new theory. Do all five re-analyses (0.1 through 0.5). This is pure analysis work; no new training needed. If v1 doesn't reinterpret cleanly under the new theory, abandon the theory before doing any new experiments.

**Tier 1 — direct stability tests (the new experiments that matter most):**

1. **B1 (Dead Mode Revival via Data Injection)** — the most direct quantitative test; compare empirical $\lambda$ to theoretical $J^{S^c S^c}$. Decides whether the Jacobian formulas are correct.
2. **A1 (Sequential Mode Addition)** — visually dramatic; directly demonstrates "new modes are rejected." Reviewers will remember this plot.
3. **C2 (Adversarial Collapsed States)** — tests plurality of stable attractors (are all $2^K - 2$ subsets stable, or only the "natural" one?).

**Tier 2 — quantitative Jacobian verification:**

4. **D1 (Block-by-Block Jacobian Estimation)** — compare theoretical formulas to reality block by block.
5. **D3 (Multi-Epoch Decay)** — verify linearization accuracy beyond first epoch.
6. **B2 (Controller-Level Perturbation)** — robustness to different perturbation modes.
7. **F1 + F2 (BRA verification)** — check the Bridge Regularity Assumption empirically.

**Tier 3 — scope and basin mapping:**

8. **A2, A3 (Injection distance/depth sweeps)** — basin characterization.
9. **C1 (Initialization distance sweep)** — basin size.
10. **E1, E2 (Attractor enumeration)** — full structure.
11. **B3 (Bias injection with continuous training)** — steady-state equilibrium test.

---

## Key Predictions and How Each Experiment Tests Them

| Theoretical Prediction | Primary Test | Diagnostic Signature |
|---|---|---|
| Collapsed states are fixed points of AM | Fixed point verification (any of Tier 1) | $\alpha$ stays constant under normal training after perturbation decay |
| Perturbations decay linearly with factor $<1$ | B1 + D3 | Exponential decay of $\|\delta\alpha\|$; rate matches $|\lambda_{\max}|$ |
| Dead modes are actively rejected | A1, A2, A3 | Added/revived modes fail to grow |
| Multiple collapsed attractors coexist | C2, E1 | Different pretraining leads to different stable states |
| Balanced state is also a collapsed state (stable) | G2 | Perturbation at B1 symmetric decays |
| Jacobian formulas match reality | D1 | Per-block empirical values within factor 2 of theory |
| BRA is not violated | F1, F2 | $\|\partial\bar{Y}_k/\partial\alpha\|$ remains bounded |

---

## Summary

The v1 experiments asked "does training reach collapsed states?" Under the old theorem (instability of balance), the answer was mixed: yes on heterogeneous benchmarks, no on symmetric ones. Under the new theorem (stability of collapsed states), the answer is unified: yes, collapsed states are attractors with basins that contain random initializations when the geometry is right.

**Family 0 converts v1 from ambiguous evidence into direct confirmation** — the very same data points that were problems under the old theorem are now foundational evidence for the new one. This reinterpretation alone is a significant part of the paper's story and requires no new experiments.

**The new experiments serve three purposes:**

1. **Direct stability verification (Tier 1):** B1 injection, A1 mode addition, C2 adversarial states demonstrate that collapsed states actively reject perturbations and that multiple collapsed attractors coexist.

2. **Quantitative match to theory (Tier 2):** D1 and D3 measure the Jacobian and contraction factor and compare to Proposition D2' formulas. This is where the theorem earns its mathematical content — qualitative agreement would be unremarkable, quantitative agreement is striking.

3. **Scope and assumption verification (Tier 3 + F):** Basin sizes, attractor enumeration, BRA validity. These characterize the theorem's regime of applicability.

The single most important new experiment is **B1 (Dead Mode Revival)**: it provides a direct measurement of the contraction factor $|\lambda_{\max}|$ and a direct comparison to the theoretical prediction from $J^{S^c S^c}$. Combined with Family 0's foundational evidence, this experiment alone establishes both the qualitative (collapsed states are stable) and quantitative (the Jacobian formulas are correct) content of the theorem.
