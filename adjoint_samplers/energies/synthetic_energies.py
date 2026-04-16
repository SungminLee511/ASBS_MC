"""
Synthetic 2D benchmark energies for mode concentration experiments.

Benchmarks B1-B7 from the experimental plan:
  B1: AsymmetricTwoModeGaussian
  B2: MullerBrownEnergy
  B3: WarpedDoubleWellEnergy
  B4: NealsFunnelEnergy
  B5: HeterogeneousCovarianceMixture
  B6: PowerLawGridMixture
  B7: ThreeWellMetastableEnergy
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


def _estimate_mode_weights_grid(energy_fn, mode_centers, grid_range=(-5, 5), grid_size=200):
    """Estimate mode weights by numerical integration on a 2D grid.

    Assigns each grid point to the nearest mode center and integrates
    exp(-E(x)) over each basin.
    """
    device = mode_centers.device
    lin = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (G², 2)

    with torch.no_grad():
        E = energy_fn(grid)  # (G²,)
        log_p = -E
        # Shift for numerical stability
        log_p = log_p - log_p.max()
        p = torch.exp(log_p)

        # Assign to modes
        dists = torch.cdist(grid.unsqueeze(0), mode_centers.unsqueeze(0)).squeeze(0)
        assignments = dists.argmin(dim=-1)

        K = mode_centers.shape[0]
        weights = torch.zeros(K, device=device)
        for k in range(K):
            weights[k] = p[assignments == k].sum()
        weights = weights / weights.sum()

    return weights


class AsymmetricTwoModeGaussian(BaseEnergy):
    """B1: Asymmetric two-mode Gaussian mixture in 2D.

    p(x) ∝ w1 * N(x; mu1, σ²I) + w2 * N(x; mu2, σ²I)

    Default: w = [0.8, 0.2], centers at (-4, 0) and (4, 0), σ = 1.
    """

    def __init__(
        self,
        dim=2,
        w1=0.8,
        mu1=None,
        mu2=None,
        sigma=1.0,
        device="cpu",
    ):
        super().__init__("asymmetric_two_mode_gaussian", dim)
        assert dim == 2, "AsymmetricTwoModeGaussian is a 2D benchmark"
        self.device = device

        self.w1 = w1
        self.w2 = 1.0 - w1

        if mu1 is None:
            mu1 = [-4.0, 0.0]
        if mu2 is None:
            mu2 = [4.0, 0.0]

        self.register_params(mu1, mu2, sigma, device)

    def register_params(self, mu1, mu2, sigma, device):
        self.mu1 = torch.tensor(mu1, dtype=torch.float32, device=device)
        self.mu2 = torch.tensor(mu2, dtype=torch.float32, device=device)
        self.sigma = sigma
        self.var = sigma**2

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """E(x) = -log p(x) up to a constant."""
        d1 = ((x - self.mu1) ** 2).sum(dim=-1) / (2 * self.var)
        d2 = ((x - self.mu2) ** 2).sum(dim=-1) / (2 * self.var)

        log_p1 = -d1 + np.log(self.w1)
        log_p2 = -d2 + np.log(self.w2)

        log_p = torch.logsumexp(torch.stack([log_p1, log_p2], dim=-1), dim=-1)
        return -log_p

    @property
    def mode_centers(self):
        return torch.stack([self.mu1, self.mu2])

    @property
    def mode_weights(self):
        return torch.tensor([self.w1, self.w2])


class MullerBrownEnergy(BaseEnergy):
    """B2: Müller-Brown potential in 2D.

    E(x) = Σ_{i=1}^{4} A_i exp[a_i(x1-x1_bar_i)^2
                                + b_i(x1-x1_bar_i)(x2-x2_bar_i)
                                + c_i(x2-x2_bar_i)^2]

    Three local minima connected by two saddle points with different
    barrier heights. Temperature β controls sharpness.
    """

    # Standard Müller-Brown parameters
    _A = [-200.0, -100.0, -170.0, 15.0]
    _a = [-1.0, -1.0, -6.5, 0.7]
    _b = [0.0, 0.0, 11.0, 0.6]
    _c = [-10.0, -10.0, -6.5, 0.7]
    _x1_bar = [1.0, 0.0, -0.5, -1.0]
    _x2_bar = [0.0, 0.5, 1.5, 1.0]

    def __init__(self, dim=2, beta=1.0, device="cpu"):
        super().__init__("muller_brown", dim)
        assert dim == 2, "MullerBrownEnergy is a 2D benchmark"
        self.device = device
        self.beta = beta

        self._register_params(device)

    def _register_params(self, device):
        self.A = torch.tensor(self._A, dtype=torch.float32, device=device)
        self.a_coeff = torch.tensor(self._a, dtype=torch.float32, device=device)
        self.b_coeff = torch.tensor(self._b, dtype=torch.float32, device=device)
        self.c_coeff = torch.tensor(self._c, dtype=torch.float32, device=device)
        self.x1_bar = torch.tensor(self._x1_bar, dtype=torch.float32, device=device)
        self.x2_bar = torch.tensor(self._x2_bar, dtype=torch.float32, device=device)

    def _raw_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw Müller-Brown energy (before temperature scaling)."""
        x1 = x[..., 0:1]  # (B, 1)
        x2 = x[..., 1:2]  # (B, 1)

        dx1 = x1 - self.x1_bar  # (B, 4)
        dx2 = x2 - self.x2_bar  # (B, 4)

        exponent = (
            self.a_coeff * dx1**2
            + self.b_coeff * dx1 * dx2
            + self.c_coeff * dx2**2
        )
        terms = self.A * torch.exp(exponent)
        return terms.sum(dim=-1)  # (B,)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """E(x) = β * V_MB(x), where V_MB is the raw Müller-Brown potential."""
        return self.beta * self._raw_energy(x)

    @property
    def mode_centers(self):
        """Approximate mode centers (local minima)."""
        return torch.tensor(
            [[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]],
            device=self.device,
        )

    @property
    def mode_weights(self):
        """Estimate mode weights by numerical integration.

        Uses a grid tailored to the Müller-Brown landscape, which
        lies roughly in [-1.5, 1.5] × [-0.5, 2.0].
        """
        if not hasattr(self, "_cached_mode_weights"):
            device = self.device
            gx = torch.linspace(-1.5, 1.5, 400, device=device)
            gy = torch.linspace(-0.5, 2.0, 400, device=device)
            grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
            grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

            with torch.no_grad():
                E = self.eval(grid)
                log_p = -E
                log_p = log_p - log_p.max()
                p = torch.exp(log_p)

                centers = self.mode_centers
                dists = torch.cdist(grid.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
                assignments = dists.argmin(dim=-1)

                K = centers.shape[0]
                weights = torch.zeros(K, device=device)
                for k in range(K):
                    weights[k] = p[assignments == k].sum()
                weights = weights / weights.sum()

            self._cached_mode_weights = weights
        return self._cached_mode_weights


class WarpedDoubleWellEnergy(BaseEnergy):
    """B3: Warped (banana-shaped) double well in 2D.

    E(x, y) = (x² - 1)² + γ(y - x²)²

    Two modes at approximately (±1, 1), banana-shaped basins.
    """

    def __init__(self, dim=2, gamma=5.0, device="cpu"):
        super().__init__("warped_double_well", dim)
        assert dim == 2, "WarpedDoubleWellEnergy is a 2D benchmark"
        self.device = device
        self.gamma = gamma

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        return (x1**2 - 1) ** 2 + self.gamma * (x2 - x1**2) ** 2

    @property
    def mode_centers(self):
        return torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=self.device)

    @property
    def mode_weights(self):
        return torch.tensor([0.5, 0.5])


class NealsFunnelEnergy(BaseEnergy):
    """B4: Neal's Funnel in 2D.

    v ~ N(0, 9),  x | v ~ N(0, exp(v))

    Joint: log p(v, x) = -v²/18 - log(exp(v/2)) - x²/(2 exp(v))
                        = -v²/18 - v/2 - x²/(2 exp(v))
    E(v, x) = v²/18 + v/2 + x²/(2 exp(v))
    """

    def __init__(self, dim=2, device="cpu"):
        super().__init__("neals_funnel", dim)
        assert dim == 2, "NealsFunnelEnergy is a 2D benchmark"
        self.device = device

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        v = x[..., 0]
        x_val = x[..., 1]
        return v**2 / 18.0 + v / 2.0 + x_val**2 / (2.0 * torch.exp(v))


class HeterogeneousCovarianceMixture(BaseEnergy):
    """B5: 4-mode Gaussian mixture with different covariances in 2D.

    p(x) ∝ Σ_{k=1}^{4} (1/4) N(x; μ_k, Σ_k)

    Mode 1: tight spike (Σ = 0.1I)
    Mode 2: broad cloud (Σ = 4.0I)
    Mode 3: elongated ellipse (Σ = diag(0.5, 2.0))
    Mode 4: unit Gaussian (Σ = 1.0I)

    Centers at (±5, ±5). Equal weights.
    """

    def __init__(self, dim=2, center_scale=5.0, device="cpu"):
        super().__init__("heterogeneous_covariance_mixture", dim)
        assert dim == 2, "HeterogeneousCovarianceMixture is a 2D benchmark"
        self.device = device
        self.K = 4
        self.center_scale = center_scale

        self._register_params(device)

    def _register_params(self, device):
        s = self.center_scale
        self.means = torch.tensor(
            [[s, s], [-s, s], [-s, -s], [s, -s]],
            dtype=torch.float32,
            device=device,
        )
        # Diagonal covariances
        self.vars = torch.tensor(
            [[0.1, 0.1], [4.0, 4.0], [0.5, 2.0], [1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        self.log_det_covs = self.vars.log().sum(dim=-1)  # (K,)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2), means: (K, 2), vars: (K, 2)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (B, K, 2)
        mahal = (diff**2 / self.vars.unsqueeze(0)).sum(dim=-1)  # (B, K)

        # log N(x; μ_k, Σ_k) = -0.5 * mahal - 0.5 * log det Σ_k - log(2π)
        log_components = -0.5 * mahal - 0.5 * self.log_det_covs.unsqueeze(0)
        # Equal weights: log(1/K)
        log_mix = log_components + np.log(1.0 / self.K)
        log_p = torch.logsumexp(log_mix, dim=-1)
        return -log_p

    @property
    def mode_centers(self):
        return self.means

    @property
    def mode_weights(self):
        return torch.tensor([0.25, 0.25, 0.25, 0.25])


class PowerLawGridMixture(BaseEnergy):
    """B6: 25-mode Gaussian grid with power-law (Zipf) weights in 2D.

    p(x) ∝ Σ_{k=1}^{25} w_k N(x; μ_k, σ²I)

    Centers on a 5×5 grid with spacing d, σ = 0.8, w_k ∝ k^{-1.5}.
    """

    def __init__(self, dim=2, spacing=6.0, sigma=0.8, alpha=1.5, device="cpu"):
        super().__init__("power_law_grid_mixture", dim)
        assert dim == 2, "PowerLawGridMixture is a 2D benchmark"
        self.device = device
        self.sigma = sigma
        self.var = sigma**2
        self.K = 25

        self._register_params(spacing, alpha, device)

    def _register_params(self, spacing, alpha, device):
        # 5x5 grid centers
        grid = torch.arange(5, dtype=torch.float32)
        grid = grid - grid.mean()  # center at origin
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        self.means = (
            torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(device)
            * spacing
        )

        # Power-law weights: w_k ∝ k^{-alpha}
        ks = torch.arange(1, self.K + 1, dtype=torch.float32, device=device)
        raw_weights = ks ** (-alpha)
        self.weights = raw_weights / raw_weights.sum()
        self.log_weights = self.weights.log()

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (B, K, 2)
        sq_dist = (diff**2).sum(dim=-1)  # (B, K)

        log_components = -sq_dist / (2 * self.var) + self.log_weights.unsqueeze(0)
        log_p = torch.logsumexp(log_components, dim=-1)
        return -log_p

    @property
    def mode_centers(self):
        return self.means

    @property
    def mode_weights(self):
        return self.weights


class ThreeWellMetastableEnergy(BaseEnergy):
    """B7: Three-well potential with a metastable state in 2D.

    E(x, y) = -log[ exp(-κ1*((x-1)² + y²))
                   + exp(-κ1*((x+1)² + y²))
                   + exp(-κ3*((x-0.3)² + (y-1.5)²)) ]

    Two deep wells at (±1, 0) with curvature κ1 = 20, and one shallow
    metastable state at (0.3, 1.5) with tunable curvature κ3.
    """

    def __init__(self, dim=2, kappa1=20.0, kappa3=8.0, device="cpu"):
        super().__init__("three_well_metastable", dim)
        assert dim == 2, "ThreeWellMetastableEnergy is a 2D benchmark"
        self.device = device
        self.kappa1 = kappa1
        self.kappa3 = kappa3

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]

        log_w1 = -self.kappa1 * ((x1 - 1.0) ** 2 + x2**2)
        log_w2 = -self.kappa1 * ((x1 + 1.0) ** 2 + x2**2)
        log_w3 = -self.kappa3 * ((x1 - 0.3) ** 2 + (x2 - 1.5) ** 2)

        log_sum = torch.logsumexp(
            torch.stack([log_w1, log_w2, log_w3], dim=-1), dim=-1
        )
        return -log_sum

    @property
    def mode_centers(self):
        return torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.3, 1.5]], device=self.device
        )

    @property
    def mode_weights(self):
        """Estimate mode weights by numerical integration."""
        if not hasattr(self, "_cached_mode_weights"):
            self._cached_mode_weights = _estimate_mode_weights_grid(
                self.eval, self.mode_centers, grid_range=(-3, 3), grid_size=300
            )
        return self._cached_mode_weights
