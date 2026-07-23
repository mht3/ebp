import numpy as np
import torch
from torch.utils.data import TensorDataset

from ebp.networks.models import EBMMLP, MLPConfig
from ebp.networks.proposal_network import GaussianProposal, GaussianProposalConfig
from ebp.optimizers import LangevinDynamicsConfig, LangevinOptimizer, OptimizerConfig
from ebp.trainer import RNCETrainer


def _build(obs_dim=1, act_dim=1, K=8):
    ebm = EBMMLP(MLPConfig(obs_dim + act_dim, 64, 1, 2))
    proposal = GaussianProposal(GaussianProposalConfig(obs_dim, act_dim))
    bounds = np.stack([-np.ones(act_dim), np.ones(act_dim)]).astype(np.float32)  # (2, act_dim)
    ds = TensorDataset(torch.zeros(16, obs_dim), torch.zeros(16, act_dim))
    inference = LangevinOptimizer.initialize(
        LangevinDynamicsConfig(bounds=bounds, iters=2, inference_samples=32),
        "cpu",
        proposal=proposal,
    )
    trainer = RNCETrainer(
        ebm, ds, optimizer_config=OptimizerConfig(), device_type="cpu", batch_size=8,
        proposal=proposal, num_counterexamples=K, stochastic_optimizer=inference,
    )
    return trainer, ebm, proposal


def test_proposal_shapes():
    proposal = GaussianProposal(GaussianProposalConfig(obs_dim=1, act_dim=2))
    x = torch.zeros(4, 1)
    assert proposal.sample(x, 32).shape == (4, 32, 2)
    assert proposal.log_prob(x, torch.zeros(4, 33, 2)).shape == (4, 33)


def test_proposal_step_updates_proposal_only():
    trainer, ebm, proposal = _build()
    x, y = torch.zeros(4, 1), torch.zeros(4, 1)

    prop_before = proposal.log_std.detach().clone()
    ebm_before = ebm.mlp.net[0].weight.detach().clone()

    loss = trainer._proposal_step(x, y)          # steps the proposal optimizer
    assert isinstance(loss, float)
    assert not torch.equal(proposal.log_std.detach(), prop_before), "proposal did not update"
    assert torch.equal(ebm.mlp.net[0].weight.detach(), ebm_before), "EBM moved in proposal phase"


def test_ebm_step_updates_ebm_and_isolates_proposal_grads():
    trainer, ebm, proposal = _build()
    x, y = torch.zeros(4, 1), torch.zeros(4, 1)

    ebm_before = ebm.mlp.net[0].weight.detach().clone()
    loss = trainer._ebm_step(x, y)               # owns its own zero_grad/backward/step
    assert isinstance(loss, float) and np.isfinite(loss)

    # INVARIANT: the EBM step must NOT deposit gradient on the proposal (detach works).
    assert proposal.log_std.grad is None or torch.count_nonzero(proposal.log_std.grad) == 0
    # And it MUST move the EBM params.
    assert not torch.equal(ebm.mlp.net[0].weight.detach(), ebm_before), "EBM did not update"


def test_one_epoch_runs():
    trainer, _, _ = _build()
    trainer.train(epochs=1)   # both phases over the loader, logging, no crash


def test_l2_penalty():
    """l2_penalty is 0 when weight=0, positive & grad-carrying when weight>0."""
    from ebp.loss import l2_penalty

    module = torch.nn.Linear(3, 2)
    zero = l2_penalty(module, 0.0)
    assert float(zero) == 0.0

    pen = l2_penalty(module, 1.0)
    assert pen.item() > 0.0
    # scales linearly with weight
    assert torch.isclose(l2_penalty(module, 2.0), 2.0 * pen)
    # carries grad back into the module's params
    pen.backward()
    assert module.weight.grad is not None and torch.count_nonzero(module.weight.grad) > 0


def test_cnn_gaussian_proposal_shapes():
    """CNN-mean Gaussian proposal: image x -> sample (B,K,act_dim), log_prob (B,N)."""
    from ebp.networks.models import CNNConfig, ConvMLPConfig, MLPConfig, SpatialReduction
    from ebp.networks.proposal_network import (
        CNNGaussianProposal,
        CNNGaussianProposalConfig,
    )

    act_dim = 2
    conv_mlp_config = ConvMLPConfig(
        cnn_config=CNNConfig(in_channels=3),
        mlp_config=MLPConfig(16 * 2, 256, act_dim, 1),
        spatial_reduction=SpatialReduction.SPATIAL_SOFTMAX,
    )
    proposal = CNNGaussianProposal(CNNGaussianProposalConfig(conv_mlp_config, act_dim=act_dim))

    x = torch.zeros(4, 3, 96, 96)  # (B, C, H, W) image observation
    assert proposal.sample(x, 32).shape == (4, 32, act_dim)
    assert proposal.log_prob(x, torch.zeros(4, 33, act_dim)).shape == (4, 33)


if __name__ == "__main__":
    test_proposal_shapes()
    test_proposal_step_updates_proposal_only()
    test_ebm_step_updates_ebm_and_isolates_proposal_grads()
    test_one_epoch_runs()
    test_l2_penalty()
    test_cnn_gaussian_proposal_shapes()
    print("all R-NCE checks passed")
