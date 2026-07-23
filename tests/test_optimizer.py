import torch
import torch.nn as nn

from ebp.tasks import CoordinateRegression, DatasetConfig
from ebp.optimizers import (
    DerivativeFreeConfig,
    DerivativeFreeOptimizer,
    LangevinDynamicsConfig,
    LangevinOptimizer,
)
from ebp.networks.proposal_network import UniformProposal


class _FakeEBM(nn.Module):
    """Minimal EBM stand-in: quadratic energy, so gradients are well-defined."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del x
        return (y ** 2).sum(dim=-1)


class _FakeProposal:
    """Minimal proposal network stand-in: standard normal."""

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        return torch.randn(x.size(0), num_samples, 2)


def test_uniform_proposal_shape_and_bounds():
    """Training negatives now come from a proposal, not the optimizer."""
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    proposal = UniformProposal(device=torch.device("cpu"), bounds=bounds)
    x = torch.zeros(64, 1)
    negatives = proposal.sample(x, 256)
    assert negatives.shape == (64, 256, bounds.shape[1])
    assert torch.all(negatives >= -1.0) and torch.all(negatives <= 1.0)


def test_derivative_free_optimizer_infer():
    """DFO is inference-only now: infer returns one action per row."""
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = DerivativeFreeConfig(bounds=bounds, iters=2, inference_samples=128)
    so = DerivativeFreeOptimizer.initialize(config, "cuda")

    x = torch.zeros(4, 1)
    best = so.infer(x, _FakeEBM())
    assert best.shape == (4, bounds.shape[1])


def test_langevin_infer_uniform_warmstart():
    """Langevin infer with no proposal warm-starts uniformly (Alg. 2)."""
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = LangevinDynamicsConfig(bounds=bounds, iters=2, inference_samples=32)
    so = LangevinOptimizer.initialize(config, "cuda")

    best = so.infer(torch.zeros(4, 1), _FakeEBM())
    assert best.shape == (4, bounds.shape[1])


def test_langevin_infer_learned_warmstart():
    """Langevin infer warm-starts from the proposal when given one (Alg. 2)."""
    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = LangevinDynamicsConfig(bounds=bounds, iters=2, inference_samples=32)
    so = LangevinOptimizer.initialize(config, "cuda", proposal=_FakeProposal())

    best = so.infer(torch.zeros(4, 1), _FakeEBM())
    assert best.shape == (4, bounds.shape[1])
