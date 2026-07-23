import dataclasses
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

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
    """Config for the diagonal-Gaussian proposal q_xi(y | x)."""

    obs_dim: int
    act_dim: int
    hidden_dim: int = 256
    hidden_depth: int = 2
    log_std_init: float = 0.0


class GaussianProposal(nn.Module):
    """A learned diagonal-Gaussian proposal q_xi(y | x): state-conditioned mean
    mu(x) from an MLP, plus a single GLOBAL learnable log_std (one per action
    dim), shared across all x. This is the stable-baselines3
    ``DiagGaussianDistribution`` + actor pattern.

    Satisfies the ``ProposalNetwork`` Protocol (``sample``) and additionally
    exposes ``log_prob`` so the R-NCE trainer can score candidates.
    """

    def __init__(self, config: GaussianProposalConfig) -> None:
        super().__init__()
        self.act_dim = config.act_dim
        self.mlp = MLP(MLPConfig(config.obs_dim, config.hidden_dim, config.act_dim, config.hidden_depth))

        self.log_std = nn.Parameter(torch.ones(self.act_dim) * config.log_std_init)

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw `num_samples` proposal samples per row of x.
        Returns (B, num_samples, act_dim)  <-- MUST match _sample_uniform's shape.
        """
        mean = self.mlp(x)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        samples = dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1, 0, 2).detach()
    
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of candidates y under q_xi(.|x).
        y:  (B, N, act_dim)   ->  returns (B, N).
        """
        mean = self.mlp(x).unsqueeze(1) # add in dimension for broadcasting
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
