'''
Plot a trained make_moons policy: predicted actions over the held-out strip and
(for energy-based methods) the learned energy landscape.

python plot_make_moons.py --method ibc \
    --checkpoint models/ibc_make_moons_n_1000_seed_0.pt \
    --train_dataset make_moons_n_1000_seed_0.npz \
    --test_dataset make_moons_test.npz

The predictions figure scatters the training data in grey and, for each test x1
in the held-out strip, the model's predicted x2 sampled --num_samples times. An
MSE policy collapses to one branch of the moons; an energy-based policy (IBC,
R-NCE) spreads across both. The energy figure (energy-based methods only) shows
the learned energy over the (x1, x2) plane with the training data overlaid.

Figures are saved to images/{method}_make_moons_predictions.png and
images/{method}_make_moons_energy.png.
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import (
    DATASETS_DIR,
    load_dataset,
    load_model,
    load_proposal,
    load_stochastic_optimizer,
)

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def load_arrays(filename):
    data = np.load(os.path.join(DATASETS_DIR, filename))
    return data["states"], data["actions"]


@torch.no_grad()
def sample_actions(model, states, method, stochastic_optimizer, device,
                   num_samples):
    """Return (num_samples, N) predicted x2 for each state.

    MSE is deterministic (its rows are identical); the stochastic optimizer
    resamples candidates each call, so IBC/R-NCE spread over the modes.
    """
    model.eval()
    states = torch.from_numpy(states).to(device)
    samples = []
    for _ in range(num_samples):
        if method == "mse":
            pred = model(states)
        else:
            pred = stochastic_optimizer.infer(states, model)
        samples.append(pred.cpu().numpy().reshape(-1))
    return np.array(samples)


def plot_predictions(train_states, train_actions, test_states, predictions,
                     strip, method, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(train_states, train_actions, s=8, c="tab:blue",
               label="train", zorder=1)
    x1 = np.repeat(test_states.reshape(-1)[None, :], len(predictions), axis=0)
    ax.scatter(x1, predictions, s=6, c="tab:green", alpha=0.5,
               label="predicted", zorder=2)
    # strip bounds are already in the plotted (rescaled) x1 axis units.
    for edge in strip:
        ax.axvline(edge, color="tab:red", ls="--", lw=1, zorder=0)
    ax.set_xlabel("$x_1$ (state)")
    ax.set_ylabel("$x_2$ (action)")
    ax.set_title(f"{method}: predicted actions on held-out strip")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)


@torch.no_grad()
def plot_energy(model, train_states, train_actions, bounds, device, method, path):
    model.eval()
    x1 = np.linspace(train_states.min() - 0.3, train_states.max() + 0.3, 200)
    x2 = np.linspace(float(bounds[0, 0]), float(bounds[1, 0]), 200)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)

    states = torch.from_numpy(x1.astype(np.float32)).reshape(-1, 1).to(device)
    candidates = torch.from_numpy(x2.astype(np.float32)).reshape(1, -1, 1)
    candidates = candidates.expand(len(x1), -1, -1).to(device)
    energies = model(states, candidates).cpu().numpy().T  # (len(x2), len(x1))

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(grid_x1, grid_x2, energies, cmap="viridis",
                         shading="auto")
    fig.colorbar(mesh, ax=ax, label="energy")
    ax.scatter(train_states, train_actions, s=6, c="white", alpha=0.5,
               edgecolors="none", zorder=2)
    # Clip the axes to the mesh so the energy landscape fills the plot (train
    # points outside the grid range are simply cropped).
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())
    ax.set_xlabel("$x_1$ (state)")
    ax.set_ylabel("$x_2$ (action)")
    ax.set_title(f"{method}: energy landscape")
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["mse", "ibc", "rnce"])
    parser.add_argument("--checkpoint", required=True, help="Path to a models/*.pt file.")
    parser.add_argument("--train_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument("--test_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument(
        "--stochastic_optimizer",
        default="derivative_free",
        choices=["derivative_free", "langevin"],
    )
    parser.add_argument("--inference_samples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Predicted actions sampled per test state.")
    parser.add_argument("--strip", type=float, nargs=2, default=[0.3, 0.5],
                        help="Held-out x1 strip, for the guide lines.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, target_bounds = load_dataset(os.path.join(DATASETS_DIR, args.train_dataset))
    train_states, train_actions = load_arrays(args.train_dataset)
    test_states, _ = load_arrays(args.test_dataset)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = load_model("make_moons", args.method).to(device)
    model.load_state_dict(checkpoint["model"])

    # R-NCE inference (Alg. 2) warm-starts Langevin from the trained proposal, which
    # is checkpointed alongside the EBM.
    proposal = None
    optimizer_name = args.stochastic_optimizer
    if args.method == "rnce":
        proposal = load_proposal("make_moons").to(device)
        proposal.load_state_dict(checkpoint["proposal"])
        optimizer_name = "langevin"
    stochastic_optimizer = load_stochastic_optimizer(
        optimizer_name, target_bounds, args.device,
        inference_samples=args.inference_samples, proposal=proposal,
    )

    os.makedirs(IMAGES_DIR, exist_ok=True)

    predictions = sample_actions(
        model, test_states, args.method, stochastic_optimizer, device,
        args.num_samples,
    )
    pred_path = os.path.join(IMAGES_DIR, f"{args.method}_make_moons_predictions.png")
    plot_predictions(train_states, train_actions, test_states, predictions,
                     args.strip, args.method, pred_path)
    print(f"Saved plot to images/{os.path.basename(pred_path)}")

    # The energy landscape only exists for the energy-based methods.
    if args.method != "mse":
        energy_path = os.path.join(IMAGES_DIR, f"{args.method}_make_moons_energy.png")
        plot_energy(model, train_states, train_actions, target_bounds, device,
                    args.method, energy_path)
        print(f"Saved plot to images/{os.path.basename(energy_path)}")
