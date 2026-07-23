'''
Converts Diffusion Policy's official Push-T demonstration zarr into train/test
npz files for train.py.

The zarr (206 human teleop episodes, 25650 steps) is downloaded by
scripts/generate_push_t_dataset.sh from
https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip.

States are the 20-D lowdim observation (9 T-block keypoints + agent xy) and
actions the 2-D absolute agent target, both in [0, 512] pixel coordinates.
Everything is normalized to [-1, 1] with the fixed screen bounds (x/256 - 1)
rather than Diffusion Policy's per-dimension data-limits normalizer: the fixed
scale is deterministic, identical for train/test, and stays valid when new
teleop episodes are appended.

python convert_pusht_dataset.py --zarr datasets/pusht/pusht_cchi_v7_replay.zarr
'''

import argparse
import os
from typing import Tuple

import numpy as np

from ebp.tasks.pusht import TARGET_BOUNDS, normalize

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def take_episodes(
    arrays: Tuple[np.ndarray, ...], episode_ends: np.ndarray, indices: np.ndarray
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Re-concatenate the given episodes and recompute exclusive episode_ends."""
    starts = np.insert(episode_ends[:-1], 0, 0)
    slices = [slice(starts[i], episode_ends[i]) for i in indices]
    taken = tuple(np.concatenate([arr[s] for s in slices]) for arr in arrays)
    new_ends = np.cumsum([s.stop - s.start for s in slices]).astype(np.int64)
    return taken, new_ends


def dump_npz(path: str, states, actions, episode_ends) -> None:
    np.savez(
        path,
        states=states,
        actions=actions,
        episode_ends=episode_ends,
        target_bounds=TARGET_BOUNDS,
    )
    print(
        f"Saved {len(episode_ends)} episodes ({len(states)} steps) to {path} "
        f"(states {states.shape}, actions {actions.shape})."
    )


if __name__ == "__main__":
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr", default=os.path.join(DATASETS_DIR, "pusht/pusht_cchi_v7_replay.zarr")
    )
    parser.add_argument("--test_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_filename", default="push_t_train.npz")
    parser.add_argument("--test_filename", default="push_t_test.npz")
    args = parser.parse_args()

    root = zarr.open(args.zarr, mode="r")
    keypoint = root["data/keypoint"][:]  # (T, 9, 2)
    state = root["data/state"][:]  # (T, 5): agent xy, block xy, block angle
    action = root["data/action"][:]  # (T, 2): absolute agent target
    episode_ends = root["meta/episode_ends"][:]
    assert np.all(np.diff(episode_ends) > 0) and episode_ends[-1] == len(action)

    states = np.concatenate(
        [keypoint.reshape(len(keypoint), -1), state[:, :2]], axis=1
    )
    states = normalize(states).astype(np.float32)
    actions = normalize(action).astype(np.float32)

    # Seeded episode-level split; the last test_episodes of the permutation
    # form the test set.
    permutation = np.random.default_rng(args.seed).permutation(len(episode_ends))
    train_eps = np.sort(permutation[: len(permutation) - args.test_episodes])
    test_eps = np.sort(permutation[len(permutation) - args.test_episodes :])

    os.makedirs(DATASETS_DIR, exist_ok=True)
    (train_states, train_actions), train_ends = take_episodes(
        (states, actions), episode_ends, train_eps
    )
    dump_npz(
        os.path.join(DATASETS_DIR, args.train_filename),
        train_states, train_actions, train_ends,
    )
    (test_states, test_actions), test_ends = take_episodes(
        (states, actions), episode_ends, test_eps
    )
    dump_npz(
        os.path.join(DATASETS_DIR, args.test_filename),
        test_states, test_actions, test_ends,
    )
