# Experimental Verification of Mode Concentration Theory

## Design Principles

Every experiment targets a specific theoretical result. Every benchmark is hard enough that collapse is a genuine obstacle, not a toy demonstration. We use targets where modes have different depths, shapes, scales, or connectivity — the conditions under which real SOC samplers fail.

-----

## Benchmarks

### B1: Asymmetric Two-Mode Gaussian (2D) — Calibration Baseline

$$p(x) \propto 0.8,\mathcal{N}(x; \mu_1, \sigma^2 I) + 0.2,\mathcal{N}(x; \mu_2, \sigma^2 I)$$

with $d = |\mu_1 - \mu_2| = 8$, $\sigma = 1$. Also run at $(0.9, 0.1)$ split.

**Why it’s here:** The simplest possible case that still has a definite “victim” mode. Everything is analytically tractable. If the theory fails here, something is fundamentally wrong. If it works here but not on harder benchmarks, the theory’s scope is limited.

**What makes it non-trivial:** The 80/20 split means the minority mode matters. A sampler that misses 20% of the mass has a KL cost of $0.8\log(0.8/1) + 0.2\log(0.2/0) = \infty$ if it fully collapses, or $\mathrm{KL}(\alpha | w) \approx 0.05$ even for moderate $\alpha_2$ suppression. The theory must quantitatively predict the collapse trajectory, not just that collapse happens.

-----

### B2: Müller-Brown Potential (2D)

$$E(x) = \sum_{i=1}^4 A_i \exp!\left[a_i(x_1 - \bar{x}_1^i)^2 + b_i(x_1 - \bar{x}_1^i)(x_2 - \bar{x}_2^i) + c_i(x_2 - \bar{x}_2^i)^2\right]$$

with standard Müller-Brown parameters ($A = [-200, -100, -170, 15]$, etc.).

Three local minima at approximately $(-0.558, 1.442)$, $(0.623, 0.028)$, $(-0.050, 0.467)$, connected by two saddle points with different barrier heights.

**Why it’s hard:**

- Mode weights $w_k$ are determined by the energy landscape, not chosen by us — you have to integrate $\exp(-\beta E)$ over each basin to know the ground truth, and the weights depend on the inverse temperature $\beta$.
- The basins are non-Gaussian: elongated, curved, and asymmetric.
- The two saddle barriers have different heights, so the “path of least resistance” between modes is asymmetric.
- At low temperature ($\beta$ large), the deepest well dominates and the shallowest is a tiny minority — exactly the metastable state that SOC samplers miss.

**Run at:** $\beta \in {0.5, 1.0, 2.0, 5.0}$. Low $\beta$ makes modes overlap (easy). High $\beta$ makes modes sharp and well-separated (hard, dramatic collapse).

-----

### B3: Warped Double Well (2D)

$$E(x, y) = (x^2 - 1)^2 + \gamma(y - x^2)^2$$

with $\gamma \in {1, 5, 10}$.

Two modes at approximately $(\pm 1, 1)$, banana-shaped. The basins from gradient flow are curved and non-convex.

**Why it’s hard:**

- The mode basins are banana-shaped, not elliptical. The gradient flow partition is genuinely irregular.
- The conditional adjoint $\bar{Y}_k(x,t)$ varies significantly across each basin (unlike Gaussian modes where it’s nearly linear).
- The “oracle” controller for this target requires the controller to learn non-trivial curved steering, not just point-toward-center.
- At large $\gamma$, the bananas are narrow and hard to hit.

**What it tests specifically:** Whether the controller deficiency (Proposition 2) is detectable for non-Gaussian basins, and whether the loss decomposition (Theorem 1) holds when the intra-mode variance is itself large and spatially varying.

-----

### B4: Neal’s Funnel (2D projection)

$$v \sim \mathcal{N}(0, 9), \qquad x \sim \mathcal{N}(0, e^v)$$

The joint distribution has a wide, shallow region (large $v$) and a narrow, deep spike (small $v$). The effective scale varies by orders of magnitude: $\text{std}(x) = e^{v/2}$ ranges from $\sim 0.05$ to $\sim 20$.

**Why it’s hard:**

- This is not a discrete multi-mode target in the classical sense — it’s a continuous family of scales. But the SOC sampler must cover both the wide region and the narrow spike, and mode concentration theory applies: the “spike” is a low-probability region that the controller systematically underweights.
- Particles that end up in the wide region provide almost no regression signal about the spike’s structure (they’re in a different scale regime).
- The controller needs to steer some particles into a region 100× smaller than the typical scale. The regression weight $\eta$ for the spike region is proportional to how many particles currently reach it — the exact feedback loop the theory describes.

**What it tests specifically:** Whether the theory extends beyond discrete well-separated modes to continuous multi-scale targets. The “mode basins” here are defined by scale, not by spatial location.

-----

### B5: Heterogeneous Covariance Mixture (2D)

$$p(x) \propto \sum_{k=1}^4 \frac{1}{4},\mathcal{N}(x; \mu_k, \Sigma_k)$$

with centers at $(\pm 5, \pm 5)$ and covariances:

- Mode 1: $\Sigma_1 = \mathrm{diag}(0.1, 0.1)$ — tight spike
- Mode 2: $\Sigma_2 = \mathrm{diag}(4.0, 4.0)$ — broad cloud
- Mode 3: $\Sigma_3 = \mathrm{diag}(0.5, 2.0)$ — elongated ellipse
- Mode 4: $\Sigma_4 = \mathrm{diag}(1.0, 1.0)$ — unit Gaussian

Equal weights $w_k = 1/4$.

**Why it’s hard:**

- The tight spike (mode 1) is the hardest to hit: it occupies a tiny volume in $\mathbb{R}^2$, so random exploration finds it rarely. Once the controller underweights it, the feedback loop kills it.
- The broad cloud (mode 2) dominates early training simply because its basin captures more random trajectories.
- The controller must learn qualitatively different strategies for different modes: gentle drift toward the cloud, precise targeting of the spike.
- The theory predicts mode 1 dies first (smallest effective volume → fewest initial particles → weakest regression signal).

**What it tests specifically:** That $\eta_k$ suppression (Proposition 1) hits small-volume modes hardest, even when all target weights are equal. Volume and weight are different, but the feedback loop amplifies both.

-----

### B6: 25-Mode Grid with Power-Law Weights (2D)

$$p(x) \propto \sum_{k=1}^{25} w_k,\mathcal{N}(x; \mu_k, \sigma^2 I)$$

with $\mu_k$ on a $5 \times 5$ grid with spacing $d = 6$, $\sigma = 0.8$, and weights $w_k \propto k^{-1.5}$ (Zipf-like). The largest mode has ~14% of the mass; the smallest has ~0.8%.

**Why it’s hard:**

- 25 modes is enough that the controller cannot “memorize” each one — it must generalize.
- The power-law weights mean many modes have very small $w_k$. The theory predicts these die first and die fast ($\eta_k \propto \alpha_k$ means the small modes get exponentially less training signal).
- The number of “surviving” modes at convergence is a quantitative prediction of the theory.
- With 25 modes, you can plot a histogram of $\alpha_k / w_k$ and show systematic suppression of the tail.

**What it tests specifically:** The theory at scale. Does the mode concentration cascade — large modes absorbing small ones sequentially — match the theory’s prediction that $\mathrm{KL}(\alpha | w)$ grows monotonically? Does the number of surviving modes decrease with $\rho_{\mathrm{sep}}$?

-----

### B7: Three-Well with Metastable State (2D)

$$E(x, y) = -\log!\left[e^{-20((x-1)^2 + y^2)} + e^{-20((x+1)^2 + y^2)} + e^{-8((x-0.3)^2 + (y-1.5)^2)}\right]$$

Three modes: two deep wells at $(\pm 1, 0)$ with curvature 20, and one shallow metastable state at $(0.3, 1.5)$ with curvature 8. The metastable state has lower depth and broader basin.

**Why it’s hard:**

- The metastable state is the scientifically interesting one (in molecular dynamics, metastable conformations are what you’re trying to find).
- Its weight $w_3$ is smaller than $w_1, w_2$ (shallower well), so the theory predicts it dies.
- This is exactly the failure mode that matters in practice: the sampler finds the obvious deep wells but misses the rare, physically important state.
- Tuning the curvature ratio and depth ratio lets you control how fast the metastable state collapses.

**Run at:** varying depth ratios (curvature of mode 3 from 4 to 20) to trace the transition from “mode 3 always dies” to “mode 3 survives.”

-----

### B8: LJ3 (Lennard-Jones 3 Particles, 2D)

Three particles in 2D interacting via the Lennard-Jones potential:

$$E(\mathbf{r}*1, \mathbf{r}*2, \mathbf{r}*3) = \sum*{i < j} 4\varepsilon\left[\left(\frac{\sigma*{\mathrm{LJ}}}{r*{ij}}\right)^{12} - \left(\frac{\sigma_{\mathrm{LJ}}}{r_{ij}}\right)^6\right]$$

After removing translation and rotation (3 DoF removed), the effective configuration space is 3D. The potential has two distinct types of minima: an equilateral triangle (global minimum, highly symmetric) and permutational variants.

**Why it’s hard:**

- This is a real molecular system, not a synthetic benchmark.
- The energy landscape has permutational symmetry ($S_3$): the 6 permutations of 3 particles give 6 equivalent but spatially distinct configurations. The sampler must find all of them.
- At low temperature, the basins are narrow and separated by high barriers.
- The effective dimensionality (3D after symmetry reduction) is just visualizable.

**What it tests specifically:** The theory on a physically meaningful target. The permutational symmetry means all modes have equal weight ($w_k = 1/6$), so any collapse is purely due to the AM feedback loop, not weight asymmetry. You can count how many of the 6 permutational copies the sampler finds.

**Visualization:** Project onto the 3 inter-particle distances $(r_{12}, r_{13}, r_{23})$ and show which permutational basins are populated.

-----

## Experiments

### E1: Mode Weight Tracking

**Targets:** All benchmarks.

**Protocol:** Every 10 epochs, forward-simulate $N = 10000$ particles. Assign each $X_1^i$ to its nearest mode center (for Gaussian benchmarks) or to its gradient-flow basin (for potential benchmarks — run $\dot{x} = -\nabla E(x)$ from $X_1^i$ until convergence).

**Plots:**

- $\alpha_k^{(n)}$ vs. epoch for all modes, dashed lines at $w_k$.
- For B1: 20 seeds overlaid, showing bifurcation.
- For B6 (25 modes): sort modes by target weight, plot $\alpha_k / w_k$ as a heatmap (modes × epochs). The theory predicts a “death wave” sweeping from smallest to largest modes.
- For B8 (LJ3): count distinct permutational basins found vs. epoch.

**Verifies:** Theorem D2 (instability), Lemma D1 (initial balance), the entire dynamic theory narrative.

-----

### E2: Regression Weight Heatmaps

**Targets:** B1, B3, B7.

**Protocol:** At epochs where $\alpha \neq w$, estimate $\eta_k(x, t_{\mathrm{fix}})$ on a dense 2D grid for $t_{\mathrm{fix}} \in {0.3, 0.5, 0.7}$. For each grid point $x$, simulate 500 particles starting from $X_{t_{\mathrm{fix}}} = x$ and count which mode they reach.

**Plots:**

- Three columns (one per $t$), two rows: top = learned $\eta_k$, bottom = oracle $\eta_k^*$.
- Difference map $\eta_k - \eta_k^*$: red where over-represented, blue where under-represented.
- For B3 (warped well): the basin boundary is curved, so the $\eta$ distortion follows the banana shape — much more visually interesting than a straight Voronoi boundary.

**Verifies:** Proposition 1 (η_k ∝ α_k). The distorted heatmap is the theory’s core mechanism made visible.

-----

### E3: AM Loss Decomposition

**Targets:** B1, B2, B5, B7.

**Protocol:** At each epoch, on the training batch:

1. Compute total AM residual $\frac{1}{N}\sum_i |u_\theta(X_t^i, t) + g^2 Y_1^i|^2$.
1. Assign each sample to its mode. Compute $\bar{Y}_k$ and $\bar{Y} = \sum_k \hat{\eta}_k \bar{Y}_k$.
1. Inter-mode: $V_{\mathrm{inter}} = \sum_k \hat{\eta}_k |\bar{Y}_k - \bar{Y}|^2$.
1. Intra-mode: $V_{\mathrm{intra}} = \sum_k \hat{\eta}_k \cdot \mathrm{Var}[Y_1 \mid M = k]$.

**Plots:**

- Stacked area chart: $g^4 V_{\mathrm{intra}}$ (bottom, blue) + $g^4 V_{\mathrm{inter}}$ (top, red) = total irreducible loss, vs. epoch.
- Normalize by total: fraction of loss that is inter-mode, vs. epoch.
- For B5 (heterogeneous): the inter-mode term should dominate early (4 very different modes) and collapse as the tight spike dies.

**Verifies:** Theorem 1 (loss decomposition) and Corollary 1 (collapse reduces loss). The inter-mode fraction dropping toward zero is the smoking gun.

-----

### E4: Controller Field Comparison

**Targets:** B1, B3, B7.

**Protocol:** At a fixed $t$ (e.g., $t = 0.5$), evaluate $u_\theta(x, t)$ on a 2D grid. Compute the oracle controller by running AM regression on importance-reweighted data (reweight terminal samples by $w_k / \alpha_k$ for each mode).

**Plots:**

- Three-panel figure: (left) learned controller quiver plot, (center) oracle controller quiver plot, (right) deficiency $\Delta u = u_\theta - u^{\mathrm{oracle}}$ with magnitude as color.
- Overlay energy contours on all three.
- For B3 (warped well): the deficiency field follows the banana curvature, showing the controller systematically steers away from the dying basin along the curved valley.

**Verifies:** Proposition 2 (controller deficiency). The deficiency field pointing away from the under-represented mode is the structural error made visible.

-----

### E5: Terminal Density Visualization

**Targets:** All benchmarks.

**Protocol:** Forward-simulate $N = 50000$ particles. Compute KDE of terminal positions.

**Plots:**

- For each benchmark, four panels at epochs ${50, 200, 500, 1000}$: KDE contours of learned (filled) vs. target density contours (black lines).
- For B2 (Müller-Brown): overlay on the energy landscape with basin boundaries. Show temperature sweep ($\beta$) as rows: at low $\beta$ the sampler covers all modes; at high $\beta$ it misses the shallowest.
- For B6 (25 modes): bird’s-eye scatter plots where dot size is proportional to $\alpha_k / w_k$. Dying modes shrink, dominant modes grow.

**Verifies:** Theorem 2 (distributional error lower bound). The visual gap between learned and target densities IS the KL cost.

-----

### E6: Particle Trajectory Visualization

**Targets:** B1, B3, B7.

**Protocol:** Forward-simulate $N = 200$ particles, saving trajectories at 50 time steps. Color each trajectory by terminal mode assignment.

**Plots:**

- Three snapshots: early training (epoch 20), mid training (epoch 200), late training (epoch 1000).
- Each snapshot: spaghetti plot of all 200 trajectories, colored by terminal mode, with mode centers marked.
- For B7 (three-well with metastable state): watch the metastable state’s trajectories disappear. Early: some green trajectories reach the shallow well. Late: all green trajectories get redirected to the deep wells.
- Animated GIF across training epochs.

**Verifies:** The mechanism of mode concentration in action. Not tied to a single theorem — it’s the entire theory as a movie.

-----

### E7: Phase Portrait of Mode Weights

**Targets:** B1 (K=2), B7 (K=3).

**Protocol:**

- B1: Track $\alpha_1^{(n)}$ across epochs for 30 seeds.
- B7: Track $(\alpha_1, \alpha_2, \alpha_3)$ across epochs and project onto the 2-simplex (ternary plot).

**Plots:**

- **B1 pitchfork:** 30 traces of $\alpha_1$ vs. epoch on one plot. All start near 0.5 (or 0.8 for asymmetric). Half diverge up, half down. The balanced point is visibly a repeller.
- **B7 ternary plot:** Each seed is a trajectory on the simplex, starting near the target $(w_1, w_2, w_3)$ and flowing toward a vertex (single-mode collapse) or an edge (two surviving modes). Multiple seeds show the flow field of the epoch map — the target point is an unstable node with trajectories radiating outward.

**Verifies:** Theorem D2 (instability of balanced fixed point). The ternary plot for K=3 is especially striking — it shows the simplex flow field directly.

-----

### E8: $J_{11}^\eta$ Estimation and Threshold Verification

**Targets:** Symmetric 2-mode Gaussian with varying separation (sweep benchmark).

**Setup:** $w_1 = w_2 = 0.5$, $\sigma = 1$, $d \in {3, 5, 7, 9, 12, 15, 20}$.

**Protocol:** For each $d$:

1. Train ASBS to near-convergence (if it converges to balanced solution, great; if not, use early training when $\alpha \approx w$).
1. Empirically perturb: add a small bias $\epsilon = 0.02$ to $\alpha_1$ (by reweighting the replay buffer), run one AM epoch, measure new $\alpha_1’$.
1. Estimate $J_{11}^\eta \approx (\alpha_1’ - 0.5) / \epsilon$.
1. Repeat 20 times, report mean and stderr.
1. Separately, for each $d$, run 10 full training runs and record whether collapse occurs (binary: $|\alpha_1 - 0.5| > 0.3$ by epoch 500).

**Plots:**

- $\hat{J}*{11}^\eta$ vs. $\rho*{\mathrm{sep}} = d/\sigma(1)$, with error bars. Horizontal line at 0.5 (instability threshold from Corollary D2).
- On secondary y-axis (or separate panel): collapse probability vs. $\rho_{\mathrm{sep}}$.
- The threshold crossing in $J_{11}^\eta$ should align with the collapse probability sigmoid.

**Verifies:** Corollary D2 ($J_{11}^\eta > 1/2 \Leftrightarrow$ instability). This is the theory’s most concrete quantitative prediction.

-----

### E9: KL Decomposition Tracking

**Targets:** B1, B2, B5, B6.

**Protocol:** At each epoch:

- $\mathrm{KL}(\alpha | w) = \sum_k \alpha_k \log(\alpha_k / w_k)$ from mode counting.
- $\mathrm{KL}(Q_\theta | p)$ estimated via Girsanov log-weights along sampled trajectories: $\log(dQ_\theta / dp)$ integrated over the path.
- $\mathrm{TV}(\alpha, w) = \frac{1}{2}\sum_k |\alpha_k - w_k|$ from mode counting.

**Plots:**

- $\mathrm{KL}(Q | p)$ and $\mathrm{KL}(\alpha | w)$ vs. epoch on same axes. The gap between them is the intra-mode error; the lower curve is the mode-weight error alone.
- For B6 (25 modes): also plot the number of “effectively dead” modes ($\alpha_k < 0.01 w_k$) vs. epoch on secondary axis.

**Verifies:** Theorem 2 (KL lower bound). The theory says $\mathrm{KL}(Q | p) \geq \mathrm{KL}(\alpha | w)$ always — this should hold exactly in every single data point, with no exceptions. Violations would falsify the theory.

-----

### E10: Separation Sweep

**Targets:** Symmetric 2-mode Gaussian, Heterogeneous 4-mode (B5 with varying spacing).

**Protocol:** For symmetric 2-mode: $d \in {2, 3, 4, 5, 6, 8, 10, 12, 15, 20}$, 10 seeds each. For B5: scale all centers by a factor $s \in {0.5, 1.0, 1.5, 2.0, 3.0}$, 10 seeds each.

**Measure:**

- Collapse indicator (binary: any mode drops below 10% of its target weight by epoch 500).
- Time to collapse (epoch where $\max_k |\alpha_k - w_k| / w_k > 0.5$).
- Final $\mathrm{KL}(\alpha | w)$.
- Which mode dies (for B5: record the identity of the first mode to collapse).

**Plots:**

- **Collapse probability** vs. $\rho_{\mathrm{sep}}$: sigmoid curve. Mark the $\rho_{\mathrm{sep}}^*$ where $J_{11}^\eta = 1/2$ (from E8) and verify it aligns with the 50% collapse probability.
- **Time to collapse** vs. $\rho_{\mathrm{sep}}$: decreasing curve (more separated → faster collapse).
- **For B5:** bar chart showing “which mode dies first” across seeds and separations. The theory predicts the tight spike (mode 1, smallest volume) dies first regardless of separation.

**Verifies:** Lemma D2 (residual bound depends on $\rho_{\mathrm{sep}}$), the entire ρ_sep narrative.

-----

### E11: Temperature Sweep on Müller-Brown

**Targets:** B2 (Müller-Brown) at $\beta \in {0.25, 0.5, 1.0, 2.0, 5.0, 10.0}$.

**Protocol:** At each temperature, compute ground-truth weights $w_k(\beta)$ by numerical integration over each basin. Train ASBS, track $\alpha_k$.

**Plots:**

- Three-row panel: each row is a temperature. Left column: energy landscape with basin boundaries. Right column: $\alpha_k$ vs. epoch (3 curves for 3 modes).
- At low $\beta$: modes overlap, no collapse, $\alpha \approx w$ throughout.
- At high $\beta$: modes sharpen, the shallowest well collapses first, then the medium well.
- The transition temperature where collapse begins is a prediction of the theory (it’s where $\rho_{\mathrm{sep}}(\beta)$ crosses the instability threshold).

**Verifies:** The full theory on a realistic, non-synthetic benchmark. The temperature sweep continuously interpolates between the “easy” (overlapping) and “hard” (separated) regimes.

-----

### E12: Mode Death Cascade on 25-Mode Grid

**Targets:** B6.

**Protocol:** Track all 25 mode weights across 2000 epochs. Record the epoch at which each mode “dies” ($\alpha_k < 0.01 w_k$ for 50 consecutive epochs).

**Plots:**

- **Survival curve:** number of surviving modes vs. epoch. A decaying staircase.
- **Death order vs. target weight:** scatter plot of (target weight $w_k$, death epoch). The theory predicts a strong negative correlation — small modes die first.
- **$5 \times 5$ grid snapshots:** at epochs ${50, 200, 500, 1000, 2000}$, show the grid with each mode colored by $\alpha_k / w_k$ (green = healthy, yellow = suppressed, red = dead). The death wave propagating from the smallest modes is visually striking.
- **Redistribution matrix:** where does the mass go when a mode dies? For each dead mode $k$, plot the change in $\alpha_l$ for all surviving modes $l$. The theory predicts mass flows to the nearest surviving mode (spatially local redistribution via the controller field).

**Verifies:** Proposition 1 (η suppression hits small modes), Theorem 1 (loss drops as modes die), Theorem 2 (KL grows as modes die). This is the most comprehensive single experiment.

-----

### E13: Metastable State Survival (Critical Experiment)

**Targets:** B7 (three-well with metastable state).

**Protocol:** Sweep the depth ratio of the metastable well: curvature parameter from 4 (very shallow) to 20 (equal depth to deep wells), in steps of 2. For each, run 20 seeds.

**Measure:**

- Metastable mode weight $\alpha_3$ at convergence (epoch 1000).
- Whether the metastable state survives ($\alpha_3 > 0.5 w_3$).
- $J_{11}^\eta$ estimated at each depth ratio.

**Plots:**

- **Survival probability** of the metastable state vs. depth ratio: sigmoid from 0 to 1 as the metastable well deepens.
- **$\alpha_3 / w_3$** vs. depth ratio: should transition from $\approx 0$ (killed) to $\approx 1$ (healthy) with the theory predicting the transition point.
- **Phase diagram:** 2D plot with x-axis = depth ratio, y-axis = separation, colored by survival/death. The boundary is the instability frontier.

**Verifies:** The theory’s most important practical prediction: SOC samplers miss metastable states, and the theory quantitatively predicts which states survive and which die.

-----

## Summary: Theory-to-Experiment Map

|Theory Result                      |Experiments                       |Key Benchmarks |
|-----------------------------------|----------------------------------|---------------|
|Prop 1 (η_k ∝ α_k)                 |E2 (heatmaps)                     |B1, B3, B7     |
|Theorem 1 (loss decomposition)     |E3 (stacked area)                 |B1, B2, B5, B7 |
|Corollary 1 (collapse reduces loss)|E3 (inter-mode fraction drops)    |B5, B6         |
|Prop 2 (controller deficiency)     |E4 (quiver plots)                 |B1, B3, B7     |
|Theorem 2 (KL lower bound)         |E9 (dual KL tracking)             |B1, B2, B5, B6 |
|Lemma 3 (integrated η = α)         |E2 (as sanity check)              |B1             |
|Lemma D1 (balanced fixed point)    |E7 (trajectories start near w)    |B1, B7         |
|Theorem D2 (instability)           |E1, E7 (bifurcation, ternary plot)|All            |
|Corollary D2 (J_11^η > 1/2)        |E8 (threshold estimation)         |Sweep benchmark|
|Lemma D2 (ρ_sep dependence)        |E10 (separation sweep sigmoid)    |B1, B5         |

-----

## Priority Ordering

**Tier 1 — do these first (they make the paper):**

1. **E7 on B1** (pitchfork, 30 seeds) + **E7 on B7** (ternary simplex flow): proves instability visually.
1. **E1 on B6** (25-mode death cascade heatmap): proves the theory at scale.
1. **E8** (J_11^η threshold): the theory’s most concrete quantitative prediction.
1. **E11** (Müller-Brown temperature sweep): proves the theory on a real chemistry benchmark.
1. **E13** (metastable survival phase diagram): the theory’s most practically important prediction.

**Tier 2 — strengthens the story:**

1. E6 on B7 (trajectory animation showing metastable state death)
1. E4 on B3 (controller field deficiency along the banana)
1. E3 on B5 (loss decomposition with heterogeneous modes)
1. E10 (separation sweep sigmoid + threshold alignment with E8)
1. E9 on B6 (KL bound tracking for 25 modes)

**Tier 3 — completeness:**

1. E2 on B3 (regression weight heatmaps on warped basins)
1. E5 on B2 (terminal density vs. target on Müller-Brown)
1. E1 on B8 (LJ3 permutation counting)
1. E12 (full death cascade analysis on B6)