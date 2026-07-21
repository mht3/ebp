'''
Training code. Specify a BC method (MSE, IBC, RNCE), dataset, train data, and optionally test dataset.
Training metrics automatically upload to wandb.

python train.py --method mse --train_dataset coordinate_regression_n_30_seed_0.npz \
    --test_dataset coordinate_regression_n_30_seed_1_test.npz

The trained model is saved to models/<method>_<train_dataset stem>.pt.
'''

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ebp import models
from ebp.optimizers import (
    DerivativeFreeConfig,
    DerivativeFreeOptimizer,
    LangevinDynamicsConfig,
    LangevinOptimizer,
    OptimizerConfig,
    StochasticOptimizer,
)
from ebp.trainer import IBCTrainer, MSETrainer

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load_dataset(dataset_path: str) -> Tuple[TensorDataset, np.ndarray]:
    """Load a datasets/*.npz file into a (state, action) TensorDataset.

    Also returns the per-dimension action bounds used by the stochastic optimizers.
    """
    data = np.load(dataset_path)
    dataset = TensorDataset(
        torch.from_numpy(data["states"]), torch.from_numpy(data["actions"])
    )
    # float32 so the stochastic optimizers' clamp doesn't promote samples to double.
    return dataset, data["target_bounds"].astype(np.float32)


def load_model(task: str, method: str) -> nn.Module:
    """Build the model for a given task and BC method.

    The trainers are model agnostic, so this factory is the only place
    architectures are chosen; new tasks (e.g. with vector states and an MLP)
    just need a new branch here.
    """
    if task == "coordinate_regression":
        # Kevin Zakka's IBC architecture: CNN backbone -> 1x1 conv (16 channels)
        # -> spatial soft argmax (16 * 2 = 32-dim feature) -> MLP. The MSE and
        # EBM models are identical except for the MLP head: the EBM's input also
        # includes the 2D action and its output is a scalar energy.
        feature_dim, action_dim = 16 * 2, 2
        if method == "mse":
            mlp_config = models.MLPConfig(feature_dim, 256, action_dim, 1)
        else:
            mlp_config = models.MLPConfig(feature_dim + action_dim, 256, 1, 1)

        config = models.ConvMLPConfig(
            cnn_config=models.CNNConfig(3),
            mlp_config=mlp_config,
            spatial_reduction=models.SpatialReduction.SPATIAL_SOFTMAX,
            coord_conv=False,
        )
        if method == "mse":
            return models.ConvMLP(config)
        return models.EBMConvMLP(config)
    raise ValueError(f"Unknown task '{task}'.")


def load_stochastic_optimizer(
    name: str, bounds: np.ndarray, device_type: str
) -> StochasticOptimizer:
    if name == "derivative_free":
        return DerivativeFreeOptimizer.initialize(
            DerivativeFreeConfig(bounds=bounds), device_type
        )
    return LangevinOptimizer.initialize(
        LangevinDynamicsConfig(bounds=bounds), device_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["mse", "ibc"])
    parser.add_argument("--task", default="coordinate_regression")
    parser.add_argument("--train_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument("--test_dataset", default=None, help="Filename in datasets/.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--stochastic_optimizer",
        default="derivative_free",
        choices=["derivative_free", "langevin"],
        help="Negative sampler and inference optimizer for the EBM methods.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, target_bounds = load_dataset(
        os.path.join(DATASETS_DIR, args.train_dataset)
    )
    test_dataset = None
    if args.test_dataset is not None:
        test_dataset, _ = load_dataset(os.path.join(DATASETS_DIR, args.test_dataset))

    model = load_model(args.task, args.method)

    if args.method == "mse":
        trainer = MSETrainer(
            model,
            train_dataset,
            test_dataset,
            optimizer_config=OptimizerConfig(),
            device_type=args.device,
            batch_size=args.batch_size,
        )
    else:
        stochastic_optimizer = load_stochastic_optimizer(
            args.stochastic_optimizer, target_bounds, args.device
        )
        trainer = IBCTrainer(
            model,
            train_dataset,
            test_dataset,
            optimizer_config=OptimizerConfig(),
            device_type=args.device,
            batch_size=args.batch_size,
            stochastic_optimizer=stochastic_optimizer,
        )

    trainer.train(args.epochs, eval_every=args.eval_every)

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = f"{args.method}_{os.path.splitext(args.train_dataset)[0]}.pt"
    trainer.save(os.path.join(MODELS_DIR, checkpoint))
    print(f"Saved model to models/{checkpoint}")
