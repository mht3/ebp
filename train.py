'''
Training code. Specify a BC method (MSE, IBC, RNCE), dataset, train data, and optionally test dataset.
Training metrics automatically upload to wandb.

python train.py --method mse --train_dataset coordinate_regression_n_30_seed_0.npz \
    --test_dataset coordinate_regression_n_30_seed_1_test.npz

python train.py --method ibc --task push_t --train_dataset push_t_train.npz \
    --test_dataset push_t_test.npz --stochastic_optimizer langevin \
    --inference_samples 1024

R-NCE (all three tasks; --l2_weight optional, defaults to 0.0):

python train.py --method rnce --task make_moons \
    --train_dataset make_moons_n_1000_seed_0.npz --test_dataset make_moons_test.npz

python train.py --method rnce --task push_t --train_dataset push_t_train.npz \
    --test_dataset push_t_test.npz

python train.py --method rnce --task coordinate_regression \
    --train_dataset coordinate_regression_n_30_seed_0.npz \
    --test_dataset coordinate_regression_n_30_test.npz --l2_weight 1e-4

The trained model is saved to models/<method>_<train_dataset stem>.pt.
'''

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ebp.networks import models
from ebp.optimizers import (
    DerivativeFreeConfig,
    DerivativeFreeOptimizer,
    LangevinDynamicsConfig,
    LangevinOptimizer,
    OptimizerConfig,
    StochasticOptimizer,
)
from ebp.networks.proposal_network import (
    CNNGaussianProposal,
    CNNGaussianProposalConfig,
    GaussianProposal,
    GaussianProposalConfig,
    UniformProposal,
)
from ebp.trainer import IBCTrainer, MSETrainer, RNCETrainer

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def window_episodes(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack a same-trajectory observation history for each action (many-to-one).

    Step t becomes ([obs_{t-s+1} ... obs_t] flattened oldest-first, a_t), as in
    IBC's sequence_length windowing. The first s-1 steps of each episode lack a
    full history and are dropped; windows never cross episode boundaries.
    """
    xs, ys = [], []
    start = 0
    for end in episode_ends:
        for t in range(start + sequence_length - 1, end):
            xs.append(states[t - sequence_length + 1 : t + 1].reshape(-1))
            ys.append(actions[t])
        start = end
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def load_dataset(
    dataset_path: str, sequence_length: int = 2
) -> Tuple[TensorDataset, np.ndarray]:
    """Load a datasets/*.npz file into a (state, action) TensorDataset.

    Episodic datasets (those with an "episode_ends" key) store per-step states
    and get windowed into observation histories of `sequence_length`.

    Also returns the per-dimension action bounds used by the stochastic optimizers.
    """
    data = np.load(dataset_path)
    states, actions = data["states"], data["actions"]
    if "episode_ends" in data:
        states, actions = window_episodes(
            states, actions, data["episode_ends"], sequence_length
        )
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
    # float32 so the stochastic optimizers' clamp doesn't promote samples to double.
    return dataset, data["target_bounds"].astype(np.float32)


def load_model(
    task: str,
    method: str,
    coord_conv: bool = False,
    sequence_length: int = 2,
    arch: str = "mlp",
) -> nn.Module:
    """Build the model for a given task and BC method.
    """
    if task == "coordinate_regression":
        feature_dim, action_dim = 16 * 2, 2
        if method == "mse":
            mlp_config = models.MLPConfig(feature_dim, 256, action_dim, 1)
        else:
            mlp_config = models.MLPConfig(feature_dim + action_dim, 256, 1, 1)

        # CoordConv prepends 2 normalized (x, y) coordinate channels, so the CNN
        # sees 3 (RGB) + 2 = 5 input channels when it is enabled.
        in_channels = 5 if coord_conv else 3
        config = models.ConvMLPConfig(
            cnn_config=models.CNNConfig(in_channels),
            mlp_config=mlp_config,
            spatial_reduction=models.SpatialReduction.SPATIAL_SOFTMAX,
            coord_conv=coord_conv,
        )
        if method == "mse":
            return models.ConvMLP(config)
        return models.EBMConvMLP(config)
    if task == "push_t":
        # 9 T-block keypoints (18) + agent xy (2) per observation step, from
        # Diffusion Policy's PushTKeypointsEnv. The EBM also takes the 2D
        # action (absolute agent target, normalized) and outputs a scalar
        # energy.
        obs_dim, action_dim = 20 * sequence_length, 2
        if arch == "mlp":
            # Diffusion Policy's IBC baseline net: 4 hidden layers x 1024,
            # ReLU, dropout 0.1.
            if method == "mse":
                return models.MLP(
                    models.MLPConfig(obs_dim, 1024, action_dim, 4, dropout_prob=0.1)
                )
            return models.EBMMLP(
                models.MLPConfig(obs_dim + action_dim, 1024, 1, 4, dropout_prob=0.1)
            )
        # IBC's pushing_states architecture: ResNetPreActivation MLP, width
        # 128, depth 16 (dropout per their gin configs: MSE 0.1, EBM 0).
        if method == "mse":
            return models.ResNetMLP(
                models.ResNetMLPConfig(obs_dim, action_dim, dropout_prob=0.1)
            )
        return models.EBMResNetMLP(models.ResNetMLPConfig(obs_dim + action_dim, 1))
    if task == "make_moons":
        # state = x1, action = x2 (both scalar). The EBM also takes the
        # candidate action and outputs a scalar energy.
        obs_dim, action_dim = 1, 1
        if method == "mse":
            return models.MLP(models.MLPConfig(obs_dim, 256, action_dim, 2))
        return models.EBMMLP(models.MLPConfig(obs_dim + action_dim, 256, 1, 2))
    raise ValueError(f"Unknown task '{task}'.")


def load_proposal(task: str, sequence_length: int = 2, coord_conv: bool = False):
    """Build the diagonal-Gaussian proposal q_xi(y | x) for R-NCE, sized per task.

    Vector-observation tasks (push_t, make_moons) get an MLP-mean GaussianProposal.
    The image task (coordinate_regression) gets a CNN-mean CNNGaussianProposal, sized
    exactly like load_model's coordinate_regression EBM branch.
    """
    if task == "coordinate_regression":
        feature_dim, action_dim = 16 * 2, 2
        in_channels = 5 if coord_conv else 3
        conv_mlp_config = models.ConvMLPConfig(
            cnn_config=models.CNNConfig(in_channels),
            mlp_config=models.MLPConfig(feature_dim, 256, action_dim, 1),
            spatial_reduction=models.SpatialReduction.SPATIAL_SOFTMAX,
            coord_conv=coord_conv,
        )
        return CNNGaussianProposal(
            CNNGaussianProposalConfig(conv_mlp_config, act_dim=action_dim))

    if task == "push_t":
        obs_dim, action_dim = 20 * sequence_length, 2
    elif task == "make_moons":
        obs_dim, action_dim = 1, 1
    else:
        raise ValueError("Unsupported task: `{}`".format(task))

    config = GaussianProposalConfig(obs_dim=obs_dim, act_dim=action_dim,
                                    hidden_dim=256, hidden_depth=2, log_std_init=0.0)
    return GaussianProposal(config)


def load_stochastic_optimizer(
    name: str,
    bounds: np.ndarray,
    device_type: str,
    inference_samples: Optional[int] = None,
    iters: Optional[int] = None,
    proposal: Optional[GaussianProposal] = None,
) -> StochasticOptimizer:
    """Build the inference-time action-search optimizer. Optionally warm-started
    from `proposal` (Algorithm 2). Training negatives are NOT drawn here -- the
    trainers draw those straight from a proposal."""
    kwargs = {}
    if inference_samples is not None:
        kwargs["inference_samples"] = inference_samples
    if iters is not None:
        kwargs["iters"] = iters
    if name == "derivative_free":
        return DerivativeFreeOptimizer.initialize(
            DerivativeFreeConfig(bounds=bounds, **kwargs), device_type, proposal=proposal
        )
    return LangevinOptimizer.initialize(
        LangevinDynamicsConfig(bounds=bounds, **kwargs), device_type, proposal=proposal
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["mse", "ibc", "rnce"])
    parser.add_argument("--task", default="coordinate_regression")
    parser.add_argument("--train_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument("--test_dataset", default=None, help="Filename in datasets/.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument(
        "--stochastic_optimizer",
        default="derivative_free",
        choices=["derivative_free", "langevin"],
        help="Negative sampler and inference optimizer for the EBM methods.",
    )
    parser.add_argument(
        "--coord_conv",
        action="store_true",
        help="Prepend normalized (x, y) coordinate channels to the CNN input.",
    )
    parser.add_argument(
        "--arch",
        default="mlp",
        choices=["mlp", "resnet"],
        help="Network for the task.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=2,
        help="Observation history length. ONly used for Push T task.",
    )
    parser.add_argument(
        "--inference_samples",
        type=int,
        default=None,
        help="Stochastic optimizer action samples at inference.",
    )
    parser.add_argument(
        "--num_counterexamples",
        type=int,
        default=256,
        help="Negative (counter-example) actions sampled per observation for the "
        "EBM's contrastive loss (InfoNCE/R-NCE) each training step. This is K: the "
        "network scores num_counterexamples + 1 candidates per observation, so cost "
        "scales linearly with it.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Stochastic optimizer inference iterations (defaults to the "
        "optimizer config's own default when unset).",
    )
    parser.add_argument(
        "--l2_weight",
        type=float,
        default=0.0,
        help="Weight of the explicit L2 (sum-of-squared-parameters) penalty added "
        "to the loss. Applies to the MSE model (mse) and the learned proposal's MLE "
        "loss (rnce). Default 0.0 = off. Distinct from Adam weight_decay.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, target_bounds = load_dataset(
        os.path.join(DATASETS_DIR, args.train_dataset), args.sequence_length
    )
    test_dataset = None
    if args.test_dataset is not None:
        test_dataset, _ = load_dataset(
            os.path.join(DATASETS_DIR, args.test_dataset), args.sequence_length
        )

    model = load_model(
        args.task,
        args.method,
        coord_conv=args.coord_conv,
        sequence_length=args.sequence_length,
        arch=args.arch,
    )

    if args.method == "mse":
        trainer = MSETrainer(
            model,
            train_dataset,
            test_dataset,
            optimizer_config=OptimizerConfig(),
            device_type=args.device,
            batch_size=args.batch_size,
            l2_weight=args.l2_weight,
        )
    elif args.method == "ibc":
        # IBC training negatives: uniform draws. Inference: the chosen optimizer.
        proposal = UniformProposal(device=torch.device(args.device), bounds=target_bounds)
        stochastic_optimizer = load_stochastic_optimizer(
            args.stochastic_optimizer,
            target_bounds,
            args.device,
            inference_samples=args.inference_samples,
            iters=args.iters,
        )
        trainer = IBCTrainer(
            model,
            train_dataset,
            test_dataset,
            optimizer_config=OptimizerConfig(),
            device_type=args.device,
            batch_size=args.batch_size,
            stochastic_optimizer=stochastic_optimizer,
            proposal=proposal,
            num_counterexamples=args.num_counterexamples,
        )
    else:  # rnce
        # One learnable proposal, three roles: trained by RNCETrainer (MLE), drawn
        # from for training negatives, and warm-starting Langevin inference (Alg. 2).
        proposal = load_proposal(
            args.task, sequence_length=args.sequence_length, coord_conv=args.coord_conv
        )
        # Inference is always Langevin for R-NCE (Alg. 2), regardless of the CLI flag.
        stochastic_optimizer = load_stochastic_optimizer(
            "langevin",
            target_bounds,
            args.device,
            inference_samples=args.inference_samples,
            iters=args.iters,
            proposal=proposal,
        )
        trainer = RNCETrainer(
            model,
            train_dataset,
            test_dataset,
            optimizer_config=OptimizerConfig(),
            device_type=args.device,
            batch_size=args.batch_size,
            proposal=proposal,
            num_counterexamples=args.num_counterexamples,
            stochastic_optimizer=stochastic_optimizer,
            l2_weight=args.l2_weight,
        )

    trainer.train(args.epochs, eval_every=args.eval_every)

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = f"{args.method}_{os.path.splitext(args.train_dataset)[0]}.pt"
    trainer.save(os.path.join(MODELS_DIR, checkpoint))
    print(f"Saved model to models/{checkpoint}")
