import torch
import torch.nn as nn

from ebp.tasks import CoordinateRegression, DatasetConfig
from ebp.optimizers import (
    DerivativeFreeConfig,
    DerivativeFreeOptimizer,
    LangevinDynamicsConfig,
    LangevinOptimizer,
)


class _FakeEBM(nn.Module):
    """Minimal EBM stand-in: quadratic energy, so gradients are well-defined."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del x
        return (y ** 2).sum(dim=-1)


class _FakeProposal:
    """Minimal proposal network stand-in: standard normal, clipped by the caller."""

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        return torch.randn(x.size(0), num_samples, 2)


def test_derivative_free_optimizer():
    '''Very simple test for output shape of (similar to cross entropy optimizer)
    '''
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = DerivativeFreeConfig(bounds=bounds, train_samples=256)
    so = DerivativeFreeOptimizer.initialize(config, "gpu")

    x = torch.zeros(64, 1)
    negatives = so.sample(x, nn.Identity())
    assert negatives.shape == (64, config.train_samples, bounds.shape[1])


def test_langevin_mcmc_optimizer_uniform_proposal():
    '''Langevin sampler with no proposal network should warm-start uniformly.'''
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = LangevinDynamicsConfig(bounds=bounds, iters=2, train_samples=32)
    so = LangevinOptimizer.initialize(config, "gpu")

    x = torch.zeros(4, 1)
    ebm = _FakeEBM()

    negatives = so.sample(x, ebm)
    assert negatives.shape == (4, config.train_samples, bounds.shape[1])
    assert torch.all(negatives >= -1.0) and torch.all(negatives <= 1.0)

    best = so.infer(x, ebm)
    assert best.shape == (4, bounds.shape[1])


def test_langevin_mcmc_optimizer_learned_proposal():
    '''Langevin sampler should warm-start from the proposal network when given one.'''
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = LangevinDynamicsConfig(bounds=bounds, iters=2, train_samples=32)
    so = LangevinOptimizer.initialize(config, "gpu", proposal=_FakeProposal())

    x = torch.zeros(4, 1)
    ebm = _FakeEBM()

    negatives = so.sample(x, ebm)
    assert negatives.shape == (4, config.train_samples, bounds.shape[1])
    assert torch.all(negatives >= -1.0) and torch.all(negatives <= 1.0)