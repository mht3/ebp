'''
Loss function classes.

IBC (InfoNCE Loss)
R-NCE Loss
L2 (weight-penalty) regularization
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_penalty(module: nn.Module, weight: float) -> torch.Tensor:
    """weight * sum of squared parameters of `module`.

    An explicit L2 loss term (distinct from Adam's weight_decay). Used by
    MSETrainer (over the model) and RNCETrainer (over the proposal's MLE loss).
    """
    l2_loss = 0.0
    for p in module.parameters():
        l2_loss += (p ** 2).sum()
    return weight * l2_loss


class InfoNCE(nn.Module):
    """InfoNCE loss over EBM energies (IBC).
    """

    def forward(self, energies: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """energies: (B, N + 1) energies for 1 positive and N negatives.
        labels: (B,) index of the positive within each row.
        """
        logits = -1.0 * energies
        return F.cross_entropy(logits, labels)


class RNCE(nn.Module):
    """Ranking Noise Contrastive Estimation loss."""

    def forward(self, energies: torch.Tensor, proposal_log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = -1.0 * (energies) - proposal_log_probs
        return F.cross_entropy(logits, labels)
