'''
Base class for MLP, CNN
'''

import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CoordConv, GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax


class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Sequence[int] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    ResidualBlock(depth_out, config.activation_fn),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)


@dataclasses.dataclass(frozen=True)
class ResNetMLPConfig:
    input_dim: int
    output_dim: int
    width: int = 128
    depth: int = 16
    """Number of hidden dense layers; must be even (two per residual block)."""
    dropout_prob: float = 0.0
    activation_fn: ActivationType = ActivationType.RELU


class ResNetPreActivationBlock(nn.Module):
    """Pre-activation residual block from IBC's ResNetPreActivationLayer:
    x + dense(drop(act(dense(drop(act(x)))))), no normalization."""

    def __init__(self, config: ResNetMLPConfig) -> None:
        super().__init__()

        self.activation = config.activation_fn.value()
        self.dropout = nn.Dropout(config.dropout_prob)
        self.dense1 = nn.Linear(config.width, config.width)
        self.dense2 = nn.Linear(config.width, config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense1(self.dropout(self.activation(x)))
        out = self.dense2(self.dropout(self.activation(out)))
        return x + out


class ResNetMLP(nn.Module):
    """IBC's MLP for state-based tasks (networks/mlp_ebm.py with
    layers='ResNetPreActivation'): a linear projection to `width`, depth/2
    pre-activation residual blocks, and a linear head."""

    def __init__(self, config: ResNetMLPConfig) -> None:
        super().__init__()

        if config.depth % 2:
            raise ValueError("'depth' must be even.")
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.width),
            *[ResNetPreActivationBlock(config) for _ in range(config.depth // 2)],
            nn.Linear(config.width, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EBMResNetMLP(nn.Module):
    """ResNetMLP as an EBM: input is [state, candidate action], output a scalar
    energy per candidate."""

    def __init__(self, config: ResNetMLPConfig) -> None:
        super().__init__()

        self.mlp = ResNetMLP(config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)


class EBMMLP(nn.Module):
    """MLP as an EBM: input is [state, candidate action], output a scalar
    energy per candidate."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)