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

from .loss import InfoNCE
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
    ) -> None:
        self.device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

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
        torch.save(self.model.state_dict(), path)


class MSETrainer(BaseTrainer):
    """Vanilla behavior cloning with an MSE loss."""

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.model(x), y)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class IBCTrainer(BaseTrainer):
    """Implicit behavior cloning: an EBM trained with InfoNCE."""

    def __init__(self, *args, stochastic_optimizer: StochasticOptimizer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stochastic_optimizer = stochastic_optimizer
        self.loss_fn = InfoNCE()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Hide the positive among sampled negatives at a random index.
        negatives = self.stochastic_optimizer.sample(x, self.model)
        targets = torch.cat([y.unsqueeze(1), negatives], dim=1)
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        labels = (permutation == 0).nonzero()[:, 1].to(self.device)

        energies = self.model(x, targets)
        return self.loss_fn(energies, labels)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.stochastic_optimizer.infer(x, self.model)
