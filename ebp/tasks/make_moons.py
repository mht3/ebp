'''
Make-moons task in PyTorch.

sklearn's two-moons dataset framed as a conditional distribution problem:
state = x1 (first coordinate), action = x2 (second coordinate). Where the two
half-moons overlap in x1, a single x1 maps to two valid x2 values -- the
multimodality an MSE regressor averages over but an energy-based model (IBC,
R-NCE) can represent.

The held-out strip x1 in [strip_min, strip_max] spans both moons, so it is a
genuinely bimodal test region. Train sets exclude it; test sets keep only it.
'''

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import Dataset


_CENTER = np.array([0.5, 0.25])
_SCALE = 1.8 / np.array([1.5, 0.75])


def _rescale(x: np.ndarray) -> np.ndarray:
    return (x - _CENTER) * _SCALE


@dataclasses.dataclass
class MakeMoonsConfig:
    dataset_size: int = 1000
    """Number of (state, action) pairs to keep after strip filtering."""

    seed: Optional[int] = None
    """Seed passed to sklearn.make_moons. Disabled if None."""

    noise: float = 0.1
    """Standard deviation of Gaussian noise added to the moons."""

    held_out: bool = False
    """If True, keep only the strip (test set); otherwise exclude it (train)."""

    strip_min: float = 0.3
    """Lower x1 bound of the held-out multimodal strip (rescaled [-2, 2] axis)."""

    strip_max: float = 0.5
    """Upper x1 bound of the held-out multimodal strip (rescaled [-2, 2] axis)."""


class MakeMoons(Dataset):
    """Predict x2 from x1 on sklearn's two-moons dataset (rescaled to [-2, 2])."""

    def __init__(self, config: MakeMoonsConfig) -> None:
        self.config = config


        oversample = 20 if config.held_out else 2
        x, _ = make_moons(
            n_samples=config.dataset_size * oversample,
            noise=config.noise,
            random_state=config.seed,
        )
        # Rescale to [-2, 2] first so the strip bounds refer to the same x1
        # axis the plots show.
        x = _rescale(x)

        in_strip = (x[:, 0] >= config.strip_min) & (x[:, 0] <= config.strip_max)
        x = x[in_strip] if config.held_out else x[~in_strip]
        x = x[: config.dataset_size]

        self._states = x[:, 0:1].astype(np.float32)
        self._actions = x[:, 1:2].astype(np.float32)

    def get_target_bounds(self) -> np.ndarray:
        """Return per-dimension action (x2) min/max, shape (2, 1).

        Computed from a fixed, unfiltered noiseless draw so train and test
        share identical bounds regardless of seed or strip.
        """
        x = _rescale(make_moons(n_samples=2000, noise=0.0, random_state=0)[0])
        low, high = x[:, 1].min(), x[:, 1].max()
        pad = 0.5
        return np.array([[low - pad], [high + pad]], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._states[index]),
            torch.from_numpy(self._actions[index]),
        )
