import dataclasses
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal, Normal

from .models import MLP, MLPConfig, ConvMLP, ConvMLPConfig


class ProposalNetwork(Protocol):
    """A learnable negative sampler p_xi(y | x)."""

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw `num_samples` proposal samples per row of `x`.
        Returns a tensor of shape (x.size(0), num_samples, target_dim).
        """
        pass


@dataclasses.dataclass(frozen=True)
class GaussianProposalConfig:
    """Config for the Gaussian proposal q_xi(y | x)."""

    obs_dim: int
    act_dim: int
    hidden_dim: int = 256
    hidden_depth: int = 2
    log_std_init: float = 0.0
    cov_rank: int = 0
    """Rank of a low-rank covariance term. 0 (default) = diagonal covariance.

    With action chunking the action is a whole trajectory, whose dimensions are
    strongly correlated: for Push-T at prediction_horizon=16 (a 32-D action) the
    mean |off-diagonal correlation| is 0.49 and 90%/99% of the variance lives in
    just 2/6 of the 32 dimensions. A diagonal Gaussian assumes independence, so
    it over-covers that manifold by ~85 nats of entropy and draws negatives that
    are trivially far from the data -- the contrastive loss then saturates (the
    chunked IBC/R-NCE train loss collapses toward 0) and stops shaping the
    energy where it matters. A rank-k term Sigma = diag + U U^T captures the
    correlated directions (fitting real chunks ~60 nats/sample better at k=4),
    so negatives stay on-manifold and hard. This is the cheap closed-form
    stand-in for the normalizing-flow proposal Singh et al. use.
    """


class GaussianProposal(nn.Module):
    """A learned Gaussian proposal q_xi(y | x): state-conditioned mean mu(x) from
    an MLP, plus a single GLOBAL learnable covariance shared across all x. This
    is the stable-baselines3 ``DiagGaussianDistribution`` + actor pattern.

    With ``cov_rank=0`` the covariance is diagonal (log_std only). With
    ``cov_rank=k>0`` it is diag(std^2) + U U^T for a learnable (act_dim, k)
    factor U, which lets the proposal model correlations between action
    dimensions -- see GaussianProposalConfig.cov_rank.

    Satisfies the ``ProposalNetwork`` Protocol (``sample``) and additionally
    exposes ``log_prob`` so the R-NCE trainer can score candidates.
    """

    def __init__(self, config: GaussianProposalConfig) -> None:
        super().__init__()
        self.act_dim = config.act_dim
        self.cov_rank = config.cov_rank
        self.mlp = MLP(MLPConfig(config.obs_dim, config.hidden_dim, config.act_dim, config.hidden_depth))

        self.log_std = nn.Parameter(torch.ones(self.act_dim) * config.log_std_init)
        if self.cov_rank > 0:
            # Small init so the proposal starts near the diagonal solution and
            # grows the correlated directions only as the MLE step wants them.
            self.cov_factor = nn.Parameter(
                torch.randn(self.act_dim, self.cov_rank) * 0.01
            )

    @property
    def mean_net(self) -> nn.Module:
        """The mean network mu(x). The L2 penalty regularizes THIS, not log_std
        (an L2 prior on log_std would pull the proposal variance toward 1.0, i.e.
        the whole action range, quietly widening the negative sampler)."""
        return self.mlp

    def _distribution(self, mean: torch.Tensor):
        """Gaussian over the action, batched over `mean`'s leading dims."""
        std = self.log_std.exp()
        if self.cov_rank > 0:
            return LowRankMultivariateNormal(
                loc=mean,
                cov_factor=self.cov_factor.expand(*mean.shape[:-1], self.act_dim, self.cov_rank),
                cov_diag=(std ** 2).expand_as(mean),
            )
        return Normal(mean, std)

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw `num_samples` proposal samples per row of x.
        Returns (B, num_samples, act_dim)  <-- MUST match _sample_uniform's shape.
        """
        mean = self.mlp(x)
        samples = self._distribution(mean).rsample(sample_shape=(num_samples,))
        return samples.permute(1, 0, 2).detach()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of candidates y under q_xi(.|x).
        y:  (B, N, act_dim)   ->  returns (B, N).
        """
        mean = self.mlp(x).unsqueeze(1)  # add in dimension for broadcasting
        if self.cov_rank > 0:
            # LowRankMultivariateNormal is multivariate: it already sums over the
            # event (action) dim, so no trailing .sum(-1) here.
            return self._distribution(mean.expand_as(y)).log_prob(y)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(y)
        return log_prob.sum(-1)

@dataclasses.dataclass(frozen=True)
class CNNGaussianProposalConfig:
    """Config for the image-observation diagonal-Gaussian proposal q_xi(y | x).

    Wraps a ConvMLPConfig whose mlp_config.output_dim MUST equal act_dim (the mean
    network outputs the action mean). act_dim is stored separately for log_std.
    """

    conv_mlp_config: ConvMLPConfig
    act_dim: int
    log_std_init: float = 0.0


class CNNGaussianProposal(nn.Module):
    """A learned diagonal-Gaussian proposal q_xi(y | x) for IMAGE observations:
    identical to GaussianProposal, but the mean network mu(x) is a ConvMLP (CNN
    backbone -> MLP head) instead of a plain MLP, so x can be an image (B, C, H, W).

    Sibling of GaussianProposal (cf. EBMMLP / EBMConvMLP in models.py). sample and
    log_prob are the SAME logic -- only the mean network differs.
    """

    def __init__(self, config: CNNGaussianProposalConfig) -> None:
        super().__init__()
        self.act_dim = config.act_dim
        # ConvMLP maps an image to config.conv_mlp_config.mlp_config.output_dim,
        # which must be act_dim.
        self.conv_mlp = ConvMLP(config.conv_mlp_config)
        self.log_std = nn.Parameter(torch.ones(self.act_dim) * config.log_std_init)

    @property
    def mean_net(self) -> nn.Module:
        """The mean network mu(x); the L2 penalty regularizes this, not log_std.
        See GaussianProposal.mean_net."""
        return self.conv_mlp

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw num_samples per row. x is an image (B, C, H, W).
        Returns (B, num_samples, act_dim). SAME as GaussianProposal.sample but the
        mean comes from self.conv_mlp(x) instead of self.mlp(x).
        """
        mean = self.conv_mlp(x)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        samples = dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1, 0, 2).detach()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of candidates y:(B, N, act_dim) under q_xi(.|x). Returns (B, N).
        SAME as GaussianProposal.log_prob but the mean comes from self.conv_mlp(x).
        """
        mean = self.conv_mlp(x).unsqueeze(1) # add in dimension for broadcasting
        std = self.log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(y)
        return log_prob.sum(-1)


@dataclasses.dataclass
class UniformProposal:
    """Fixed (non-learnable) uniform proposal over the action bounds.

    The IBC negative sampler: draws uniformly within the per-dimension `bounds`.
    Satisfies the same `sample(x, num_samples) -> (B, num_samples, act_dim)`
    interface as GaussianProposal, so trainers can draw training negatives from a
    proposal regardless of method. Has no `log_prob` because InfoNCE does not use
    proposal log-probs (R-NCE, which does, uses GaussianProposal instead).
    """

    device: torch.device
    bounds: np.ndarray
    """Per-dimension [low, high] action bounds, shape (2, act_dim)."""

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        bounds = torch.as_tensor(self.bounds, dtype=torch.float32, device=x.device)
        low, high = bounds[0], bounds[1]
        samples = low + torch.rand(
            x.size(0) * num_samples, low.shape[0], device=x.device
        ) * (high - low)
        return samples.reshape(x.size(0), num_samples, -1)
