'''
Push-T environment vendored from Diffusion Policy (real-stanford/diffusion_policy).

The env keeps its original Gym 0.21 calling conventions — obs-only reset() and a
4-tuple step() — and must be constructed directly (never via gym.make or
gymnasium wrappers). Call env.seed(seed) before env.reset(); the seed fully
determines the initial condition.
'''

import numpy as np

from ebp.tasks.pusht.pusht_env import PushTEnv
from ebp.tasks.pusht.pusht_keypoints_env import PushTKeypointsEnv

# Fixed [0, 512] screen bounds -> [-1, 1], used for both the 20-D keypoint
# states and the 2-D actions. Deterministic (unlike Diffusion Policy's
# data-limits normalizer), so train/test/teleop data all share the same scale.
NORM_SCALE = 256.0

TARGET_BOUNDS = np.array([[-1.0, -1.0], [1.0, 1.0]])


def normalize(x: np.ndarray) -> np.ndarray:
    return x / NORM_SCALE - 1.0


def denormalize(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) * NORM_SCALE


__all__ = [
    "PushTEnv",
    "PushTKeypointsEnv",
    "NORM_SCALE",
    "TARGET_BOUNDS",
    "normalize",
    "denormalize",
]
