from __future__ import annotations

import dataclasses
from typing import Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks.proposal_network import ProposalNetwork

@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    lr_scheduler_step: int = 100
    lr_scheduler_gamma: float = 0.99

@dataclasses.dataclass
class StochasticOptimizerConfig:
    bounds: np.ndarray
    """Bounds on the samples, min/max for each dimension."""

    iters: int
    """The total number of inference iters."""

    inference_samples: int
    """The number of candidates to sample per iter during inference."""


class StochasticOptimizer(Protocol):
    """Inference-time action search over a trained EBM.

    Training negatives are drawn directly from a proposal (see
    ebp.networks.proposal_network), NOT from the optimizer -- the optimizer's sole
    job is to find the best action given a trained EBM.
    """

    device: torch.device

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""


@dataclasses.dataclass
class DerivativeFreeConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.33
    noise_shrink: float = 0.5
    iters: int = 3
    inference_samples: int = 2 ** 14




@dataclasses.dataclass
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer for inference. Great for up to 5 dimensions."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    inference_samples: int
    bounds: np.ndarray

    @staticmethod
    def initialize(
        config: DerivativeFreeConfig,
        device_type: str,
        proposal: Optional[ProposalNetwork] = None,
    ) -> DerivativeFreeOptimizer:
        del proposal 
        return DerivativeFreeOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            noise_shrink=config.noise_shrink,
            iters=config.iters,
            inference_samples=config.inference_samples,
            bounds=config.bounds,
        )

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Draw uniform samples within the per-dimension bounds, on-device."""
        bounds = torch.as_tensor(self.bounds, dtype=torch.float32, device=self.device)
        low, high = bounds[0], bounds[1]
        return low + torch.rand(num_samples, low.shape[0], device=self.device) * (high - low)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        bounds = torch.as_tensor(self.bounds).to(self.device)

        samples = self._sample(x.size(0) * self.inference_samples)
        samples = samples.reshape(x.size(0), self.inference_samples, -1)

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]


@dataclasses.dataclass
class LangevinDynamicsConfig(StochasticOptimizerConfig):
    step_size: float = 1e-3
    noise_scale: float = 0.1
    iters: int = 5
    inference_samples: int = 2 ** 8


@dataclasses.dataclass
class LangevinOptimizer:
    """Unadjusted Langevin Algorithm (ULA) sampler for inference (paper Algorithm 2):
    warm-start y_0 from the proposal (or uniform), then run Langevin steps on E_theta.
    """

    device: torch.device
    bounds: np.ndarray
    step_size: float
    noise_scale: float
    iters: int
    inference_samples: int
    proposal: Optional[ProposalNetwork] = None

    @staticmethod
    def initialize(
        config: LangevinDynamicsConfig,
        device_type: str,
        proposal: Optional[ProposalNetwork] = None,
    ) -> LangevinOptimizer:
        return LangevinOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            bounds=config.bounds,
            step_size=config.step_size,
            noise_scale=config.noise_scale,
            iters=config.iters,
            inference_samples=config.inference_samples,
            proposal=proposal,
        )

    def _sample_uniform(self, num_samples: int) -> torch.Tensor:
        """Draw uniform samples within the per-dimension bounds, on-device."""
        bounds = torch.as_tensor(self.bounds, dtype=torch.float32, device=self.device)
        low, high = bounds[0], bounds[1]
        return low + torch.rand(num_samples, low.shape[0], device=self.device) * (high - low)

    def _warm_start(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw y_0 from the proposal distribution p_xi(y | x), or uniform if
        no proposal network is given."""
        if self.proposal is not None:
            return self.proposal.sample(x, num_samples).to(self.device)
        samples = self._sample_uniform(x.size(0) * num_samples)
        return samples.reshape(x.size(0), num_samples, -1)

    def _langevin_step(
        self, x: torch.Tensor, samples: torch.Tensor, ebm: nn.Module, bounds: torch.Tensor
    ) -> torch.Tensor:
        """ULA update step"""
        with torch.enable_grad():
            samples = samples.detach().requires_grad_(True)
            energies = ebm(x, samples)
            grad = torch.autograd.grad(energies.sum(), samples)[0]

        noise = torch.randn_like(samples) * np.sqrt(2.0 * self.step_size) * self.noise_scale
        samples = samples.detach() - self.step_size * grad + noise
        return samples.clamp(min=bounds[0, :], max=bounds[1, :])

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        bounds = torch.as_tensor(self.bounds, dtype=torch.float32).to(self.device)

        samples = self._warm_start(x, self.inference_samples)
        for _ in range(self.iters):
            samples = self._langevin_step(x, samples, ebm, bounds)

        with torch.no_grad():
            energies = ebm(x, samples)
            best_idxs = energies.argmin(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]
