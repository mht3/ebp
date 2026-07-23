'''
Parent class for behavior cloning trainers.

Trainers are model agnostic: MSETrainer expects model(x) -> y_hat, while
IBCTrainer expects an EBM model(x, y_candidates) -> (B, N) energies.
'''

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .loss import InfoNCE, RNCE, l2_penalty
from .networks.proposal_network import GaussianProposal
from .optimizers import OptimizerConfig, StochasticOptimizer
from .utils import WandbLogger


class BaseTrainer:
    """Shared training loop, evaluation, and wandb logging."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        device_type: str = 'cuda',
        batch_size: int = 8,
        project_name: str = 'ebp',
        l2_weight: float = 0.0,
    ) -> None:
        self.device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer_config = optimizer_config
        # Explicit L2 (weight-penalty) loss term, default 0.0 = off. Distinct from
        # Adam's weight_decay. Used by MSETrainer (over model) and RNCETrainer (over
        # the proposal, in its MLE step).
        self.l2_weight = l2_weight
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=optimizer_config.lr_scheduler_step,
            gamma=optimizer_config.lr_scheduler_gamma,
        )

        self.logger = WandbLogger(project_name=project_name)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the training loss for a single batch."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an action for each observation in the batch."""
        raise NotImplementedError

    def train(self, epochs: int, eval_every: int = 10) -> None:
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.train_step(x, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= len(self.train_loader)

            metrics = {'train/loss': train_loss, 'lr': self.scheduler.get_last_lr()[0]}
            if self.test_loader is not None and (epoch + 1) % eval_every == 0:
                metrics['test/mse'] = self.evaluate()
            self.logger.update(metrics, step=epoch)
            pbar.set_postfix(loss=train_loss)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean squared error of predictions over the test set."""
        self.model.eval()
        mse = 0.0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            mse += F.mse_loss(self.predict(x), y).item()
        return mse / len(self.test_loader)

    def save(self, path: str) -> None:
        """Save a checkpoint dict. Subclasses may add more entries (e.g. the
        R-NCE proposal). Loaders read ckpt['model'] for the EBM/MSE weights."""
        torch.save({"model": self.model.state_dict()}, path)


class MSETrainer(BaseTrainer):
    """Vanilla behavior cloning with an MSE loss."""

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.model(x), y) + l2_penalty(self.model, self.l2_weight)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class IBCTrainer(BaseTrainer):
    """Implicit behavior cloning: an EBM trained with InfoNCE.

    Training negatives are drawn from `proposal` (a UniformProposal for standard
    IBC). `stochastic_optimizer` is used only at inference time (predict).
    """

    def __init__(
        self,
        *args,
        stochastic_optimizer: StochasticOptimizer,
        proposal,
        num_counterexamples: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stochastic_optimizer = stochastic_optimizer  # inference only
        self.proposal = proposal
        self.num_counterexamples = num_counterexamples
        self.loss_fn = InfoNCE()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        negatives = self.proposal.sample(x, self.num_counterexamples)
        # Hide the positive among sampled negatives at a random index.
        targets = torch.cat([y.unsqueeze(1), negatives], dim=1)
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        labels = (permutation == 0).nonzero()[:, 1].to(self.device)

        energies = self.model(x, targets)
        return self.loss_fn(energies, labels)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.stochastic_optimizer.infer(x, self.model)


class RNCETrainer(BaseTrainer):
    """Ranking Noise Contrastive Estimation (paper Algorithm 1).

    Full-epoch phases (overrides BaseTrainer.train): each epoch trains the proposal
    q_xi over the whole loader by maximum likelihood (minimize -log q(y|x)), THEN
    trains the EBM theta over the whole loader by the R-NCE ranking loss over
    {y} + K negatives. Each phase owns its own optimizer step (proposal_optimizer /
    self.optimizer). The negatives and their log q must NOT carry theta-gradients
    (paper Remark 3.2), so the log q fed to RNCE is detached.
    """

    def __init__(
        self,
        *args,
        proposal: GaussianProposal,
        num_counterexamples: int,
        stochastic_optimizer: StochasticOptimizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if proposal is None:
            raise ValueError("Proposal model must not be none for R-NCE.")
        # Training negatives are drawn directly from the proposal (Alg. 1, line 11):
        # K = num_counterexamples draws per observation.
        self.num_counterexamples = num_counterexamples
        # Inference-time action search (Alg. 2): Langevin warm-started from the
        # proposal. Used only in predict().
        self.stochastic_optimizer = stochastic_optimizer
        # The proposal is trained by its OWN optimizer (separate from the EBM's
        # self.optimizer built in BaseTrainer). Move it onto the device first.
        self.proposal = proposal.to(self.device)
        self.loss_fn = RNCE()
        # Reuse the EBM optimizer's learning rate (BaseTrainer already resolved it
        # from optimizer_config), so we don't re-thread the config through **kwargs.
        self.proposal_optimizer = torch.optim.Adam(
            self.proposal.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )

    def train(self, epochs: int, eval_every: int = 10) -> None:
        """Algorithm 1: full-epoch phases. Each epoch trains the proposal over the
        whole loader (MLE), THEN the EBM over the whole loader (R-NCE). The two
        phases never touch per batch. This overrides BaseTrainer.train, so this
        method also owns the scheduler step, logging, and periodic evaluation.
        """
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train()

            proposal_loss = 0.0
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                mle_loss = self._proposal_step(x, y)
                proposal_loss += mle_loss

            proposal_loss = proposal_loss / len(self.train_loader)

            ebm_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self._ebm_step(x, y)
                ebm_loss += loss
            self.scheduler.step() 

            ebm_loss = ebm_loss / len(self.train_loader)
            # Logging + periodic eval (mirror BaseTrainer.train, trainer.py:80-84).
            metrics = {"train/loss": ebm_loss, "lr": self.scheduler.get_last_lr()[0], "train/proposal_loss": proposal_loss}
            # Track the proposal's Gaussian std over training (mean + per action dim).
            with torch.no_grad():
                std = self.proposal.log_std.detach().exp()
            metrics["proposal/std"] = std.mean().item()
            for i, s in enumerate(std):
                metrics[f"proposal/std_{i}"] = s.item()
            if self.test_loader is not None and (epoch + 1) % eval_every == 0:
                metrics["test/mse"] = self.evaluate()
            self.logger.update(metrics, step=epoch)
            pbar.set_postfix(loss=ebm_loss)

    def _proposal_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Proposal MLE step (xi): minimize -log q(y | x). Steps proposal_optimizer.
        y is (B, act_dim); log_prob wants (B, N, act_dim), so score y as N=1 candidates.
        """
        self.proposal_optimizer.zero_grad()
        logq_pos = self.proposal.log_prob(x, y.unsqueeze(1)).squeeze(1)
        # L2 penalty regularizes the proposal (xi) only, added to its MLE loss.
        proposal_loss = -logq_pos.mean() + l2_penalty(self.proposal, self.l2_weight)
        proposal_loss.backward()
        self.proposal_optimizer.step()
        return proposal_loss.item()

    def _ebm_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """EBM R-NCE step (theta): rank y against K negatives. Steps self.optimizer.
        IBCTrainer.train_step (trainer.py:118-126) PLUS a (detached) log q term, and
        -- because we no longer use BaseTrainer's loop -- this method now owns the
        EBM optimizer step itself.
        """
        negatives = self.proposal.sample(x, self.num_counterexamples)
        # Hide the positive among sampled negatives at a random index.
        targets = torch.cat([y.unsqueeze(1), negatives], dim=1)
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        labels = (permutation == 0).nonzero()[:, 1].to(self.device)
        self.optimizer.zero_grad()
        energies = self.model(x, targets)
        # Detach: negatives / their log q must not carry theta-gradients (Remark 3.2).
        logq = self.proposal.log_prob(x, targets).detach()
        loss = self.loss_fn(energies, logq, labels)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # start from proposal distribution. Use energy model to sample low energy actions.
        return self.stochastic_optimizer.infer(x, self.model)

    def save(self, path: str) -> None:
        """Checkpoint the EBM and the learned proposal together (SB3 style), so
        inference can warm-start from the trained proposal (Algorithm 2)."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "proposal": self.proposal.state_dict(),
            },
            path,
        )
