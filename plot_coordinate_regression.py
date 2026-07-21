'''
Plot coordinate regression predictions for a trained model, like
https://github.com/kevinzakka/ibc/blob/main/plot.py.

python plot_coordinate_regression.py --method mse \
    --checkpoint models/mse_coordinate_regression_n_30_seed_0.pt \
    --train_dataset coordinate_regression_n_30_seed_0.npz \
    --test_dataset coordinate_regression_n_30_seed_1_test.npz

Plots the train coordinates (black x) with their convex hull, and the ground
truth test coordinates colored by the model's pixel error at that location
(blue circles when the prediction is < 1 pixel off). Saves the figure to
assets/<checkpoint stem>.png.
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull

from train import DATASETS_DIR, load_dataset, load_model, load_stochastic_optimizer

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

RESOLUTION = (96, 96)
ERROR_CAP = 30.0
"""Pixel error above which the colormap saturates."""


def unscale(coords: np.ndarray) -> np.ndarray:
    """Inverse of CoordinateRegression._scale_coordinates: [-1, 1] -> pixels."""
    return (coords + 1) / 2 * (np.array(RESOLUTION) - 1)


@torch.no_grad()
def predict(
    model, states, method, stochastic_optimizer, device, batch_size: int = 32
) -> np.ndarray:
    """Predict in batches; EBM inference draws thousands of candidates per state."""
    model.eval()
    predictions = []
    for batch in torch.split(states, batch_size):
        batch = batch.to(device)
        if method == "mse":
            predictions.append(model(batch).cpu())
        else:
            predictions.append(stochastic_optimizer.infer(batch, model).cpu())
    return torch.cat(predictions).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["mse", "ibc"])
    parser.add_argument("--task", default="coordinate_regression")
    parser.add_argument("--checkpoint", required=True, help="Path to a models/*.pt file.")
    parser.add_argument("--train_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument("--test_dataset", required=True, help="Filename in datasets/.")
    parser.add_argument(
        "--stochastic_optimizer",
        default="derivative_free",
        choices=["derivative_free", "langevin"],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, target_bounds = load_dataset(os.path.join(DATASETS_DIR, args.train_dataset))
    train_coords = np.load(os.path.join(DATASETS_DIR, args.train_dataset))["coordinates"]
    test_data = np.load(os.path.join(DATASETS_DIR, args.test_dataset))

    model = load_model(args.task, args.method).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    stochastic_optimizer = load_stochastic_optimizer(
        args.stochastic_optimizer, target_bounds, args.device
    )

    predictions = predict(
        model,
        torch.from_numpy(test_data["states"]),
        args.method,
        stochastic_optimizer,
        device,
    )
    predictions = unscale(predictions)
    test_coords = test_data["coordinates"].astype(np.float32)
    errors = np.linalg.norm(predictions - test_coords, axis=1)
    print(f"Mean test pixel error: {errors.mean():.3f} (max {errors.max():.3f})")

    # Train coordinates and their convex hull.
    plt.scatter(
        train_coords[:, 0], train_coords[:, 1], marker="x", c="black", zorder=2, alpha=0.5
    )
    for simplex in ConvexHull(train_coords).simplices:
        plt.plot(
            train_coords[simplex, 0],
            train_coords[simplex, 1],
            "--",
            zorder=2,
            alpha=0.5,
            c="black",
        )

    # Ground truth test coordinates colored by the prediction's pixel error;
    # blue if the prediction is less than 1 pixel off.
    plt.scatter(
        test_coords[:, 0],
        test_coords[:, 1],
        c=np.clip(errors, 0.0, ERROR_CAP),
        cmap="Reds",
        vmin=0.0,
        vmax=ERROR_CAP,
        zorder=1,
    )
    plt.colorbar(label="pixel error")
    accurate = errors < 1.0
    plt.scatter(test_coords[accurate, 0], test_coords[accurate, 1], c="blue", zorder=1)

    plt.xlim(-2, RESOLUTION[0] + 2)
    plt.ylim(-2, RESOLUTION[1] + 2)
    plt.title(f"{args.method} ({accurate.sum()}/{len(errors)} within 1 pixel)")

    os.makedirs(ASSETS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    plot_path = os.path.join(ASSETS_DIR, f"{stem}.png")
    plt.savefig(plot_path, format="png", dpi=200)
    print(f"Saved plot to assets/{stem}.png")
