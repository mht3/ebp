'''
Generates an NPZ file of (state, action) pairs for a task defined in ebp/tasks/.

Usage:
    python generate_data.py --task coordinate_regression --samples 30 --seed 0
    python generate_data.py --task coordinate_regression --samples 30 --seed 0 \
        --filename coordinate_regression_n_30_seed_0.npz

    task:     the task located in ebp/tasks/ (currently: coordinate_regression)
    samples:  number of (state, action) pairs to generate
    seed:     numpy seed for reproducibility
    filename: name of the .npz file, saved into the datasets/ folder.
              Defaults to <task>_n_<samples>_seed_<seed>.npz.

Test sets:
    A test set is a second dataset generated with a different seed. Can optionally pass --exclude \
    to make sure there are no repeats in the training data.

    python generate_data.py --task coordinate_regression --samples 30 --seed 1 \
        --filename coordinate_regression_n_30_seed_1_test.npz \
        --exclude coordinate_regression_n_30_seed_0.npz
'''

import argparse
import os

import numpy as np

from ebp.tasks import CoordinateRegression, DatasetConfig

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

TASKS = {
    "coordinate_regression": (CoordinateRegression, DatasetConfig),
}


def build_dataset(task: str, samples: int, seed: int):
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Available: {list(TASKS)}")
    dataset_cls, config_cls = TASKS[task]
    return dataset_cls(config_cls(dataset_size=samples, seed=seed))


def dump_npz(dataset, path: str) -> None:
    """Stack every (state, action) pair and save them as a single .npz.
    """
    states = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
    actions = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])

    arrays = {"states": states, "actions": actions}
    if hasattr(dataset, "coordinates"):
        arrays["coordinates"] = dataset.coordinates
    if hasattr(dataset, "get_target_bounds"):
        arrays["target_bounds"] = dataset.get_target_bounds()

    np.savez(path, **arrays)
    print(
        f"Saved {len(dataset)} pairs to {path} "
        f"(states {states.shape} {states.dtype}, actions {actions.shape} {actions.dtype})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="coordinate_regression", choices=list(TASKS))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filename", default=None)
    parser.add_argument(
        "--exclude",
        default=None,
        help="An existing datasets/*.npz whose coordinates the new set must avoid "
        "(used to build a leakage-free test set).",
    )
    args = parser.parse_args()

    filename = args.filename or f"{args.task}_n_{args.samples}_seed_{args.seed}.npz"

    os.makedirs(DATASETS_DIR, exist_ok=True)
    dataset = build_dataset(args.task, args.samples, args.seed)

    if args.exclude is not None:
        # --exclude is only meaningful for coordinate regression, whose npz
        # files store raw integer coordinates to dedupe train/test against.
        exclude_path = os.path.join(DATASETS_DIR, args.exclude)
        train_coords = np.load(exclude_path)["coordinates"]
        dataset.exclude(train_coords)

    dump_npz(dataset, os.path.join(DATASETS_DIR, filename))


