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

## Experiment Family G: Connection to v1 Results

Reinterpret v1 data through the stability lens rather than the instability lens.

### G1: Post-Hoc Perturbation on Collapsed v1 Runs

Using existing converged B5, B7, Müller-Brown checkpoints from v1:
- Load the final state (collapsed).
- Apply small $\delta\alpha$ perturbations (data injection or controller noise).
- Run 200 more epochs.
- Measure decay.

This repurposes v1 data to test stability directly. No new pretraining needed.

### G2: B1 Symmetric Revisited as Balanced Collapsed State

The v1 result that B1 symmetric stays balanced is now reinterpreted: the balanced state $\alpha = w$ is *also* a collapsed state (with $S = $ full set, no modes dead). Under the new theorem, it should be a stable fixed point like any other collapsed state.

- Perturb: inject small asymmetric bias at a converged B1 symmetric run.
- Measure: does the bias decay back to balance?

**Prediction:** Yes, because balance is a collapsed state (with $S =$ full set) and therefore stable. This reinterprets the v1 B1 symmetric result as *confirming* (not contradicting) the new theorem.

### G3: Comparing Collapse Rates Across v1 Benchmarks

Extract the empirical contraction factor from v1 trajectory data on B5, B7, Müller-Brown. Does the decay rate near the collapsed state match the theoretical prediction?

This uses existing v1 training curves without new experiments, extracting additional quantitative verification from data you already have.

---

## Priority Ordering

**Tier 1 — direct stability tests (run these first):**

1. **B1 (Dead Mode Revival via Data Injection)** — the most direct test; compare empirical $\lambda$ to theoretical $J^{S^c S^c}$.
2. **A1 (Sequential Mode Addition)** — visually dramatic and directly tests "new modes are rejected."
3. **G2 (B1 Symmetric Revisited)** — turns a v1 null result into a positive confirmation; requires only one experiment.
4. **C2 (Adversarial Collapsed States)** — tests plurality of stable attractors.

**Tier 2 — quantitative Jacobian verification:**

5. **D1 (Block-by-Block Jacobian Estimation)** — compare formulas to reality.
6. **D3 (Multi-Epoch Decay)** — verify linearization accuracy.
7. **B2 (Controller-Level Perturbation)** — robustness test.
8. **F1 + F2 (BRA verification)** — check the assumption empirically.

**Tier 3 — scope and basin mapping:**

9. **A2, A3 (Injection distance/depth sweeps)** — basin characterization.
10. **C1 (Initialization distance sweep)** — basin size.
11. **E1, E2 (Attractor enumeration)** — full structure.
12. **G1, G3 (v1 reinterpretation)** — reuse existing data.

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

The v1 experiments asked "does training reach collapsed states?" — the wrong question for the new theory. This v3 guide asks "are collapsed states stable fixed points?", which is the direct question the theorem answers.

The single most important experiment is **B1 (Dead Mode Revival via Data Injection)**: it provides a direct quantitative measurement of the contraction factor predicted by $J^{S^c S^c}$. If this experiment confirms $\lambda < 1$ and matches the theoretical value within a factor of 2, the theorem is empirically verified.

Tier 1 experiments (B1, A1, G2, C2) collectively establish whether the stability theory is correct. Tier 2 and 3 provide quantitative verification and scope characterization. All together, they replace the v1 "does it collapse?" question with a much richer "how does the collapsed state respond to perturbations?" analysis, which matches the theory's content.
