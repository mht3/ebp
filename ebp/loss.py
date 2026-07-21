'''
Loss function classes.

IBC (InfoNCE Loss)
R-NCE Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """InfoNCE loss over EBM energies (IBC).

    The positive target is hidden among N negatives; treating negated energies
    as logits, the loss is the cross entropy of picking the positive.
    """

    def forward(self, energies: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """energies: (B, N + 1) energies for 1 positive and N negatives.
        labels: (B,) index of the positive within each row."""
        logits = -1.0 * energies
        return F.cross_entropy(logits, labels)


class RNCE(nn.Module):
    """Ranking Noise Contrastive Estimation loss (arXiv 2309.05803)."""

    def forward(self, energies: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
