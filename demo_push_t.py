'''
Collect Push-T demonstrations with the mouse, ported from Diffusion Policy's
demo_pusht.py but saving straight to a train.py-compatible .npz instead of a
zarr replay buffer.

python demo_push_t.py --output push_t_teleop.npz

Hover the mouse close to the blue circle to start. Push the T block into the
green area; the episode terminates on success (95% coverage).
Press "Q" to exit, "R" to retry the episode, hold "Space" to pause.

Episodes are appended to the output file (saved atomically after each episode),
seeded in recording order so retries reuse the same initial condition. States
(20-D keypoints + agent xy) and actions (2-D agent target) are normalized to
[-1, 1] exactly like convert_pusht_dataset.py.
'''

import argparse
import os

import numpy as np
import pygame

from convert_pusht_dataset import DATASETS_DIR, TARGET_BOUNDS, normalize
from ebp.tasks.pusht import PushTKeypointsEnv


def load_existing(path):
    if not os.path.exists(path):
        return [], [], []
    data = np.load(path)
    return [data["states"]], [data["actions"]], list(data["episode_ends"])


def save_atomic(path, states, actions, episode_ends):
    tmp_path = path[: -len(".npz")] + ".tmp.npz"
    np.savez(
        tmp_path,
        states=np.concatenate(states),
        actions=np.concatenate(actions),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        target_bounds=TARGET_BOUNDS,
    )
    os.replace(tmp_path, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="push_t_teleop.npz",
                        help="Filename in datasets/.")
    parser.add_argument("--render_size", type=int, default=96)
    parser.add_argument("--control_hz", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(DATASETS_DIR, exist_ok=True)
    output_path = os.path.join(DATASETS_DIR, args.output)
    all_states, all_actions, episode_ends = load_existing(output_path)

    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(
        render_size=args.render_size, render_action=False, **kp_kwargs
    )
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    while True:
        episode_states, episode_actions = [], []
        # Record in seed order, starting with 0; retries reuse the seed.
        seed = len(episode_ends)
        print(f"starting seed {seed}")
        env.seed(seed)
        obs = env.reset()
        img = env.render(mode="human")

        retry = False
        pause = False
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = True
                    elif event.key == pygame.K_r:
                        retry = True
                    elif event.key == pygame.K_q:
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            if retry:
                break
            if pause:
                continue

            # Action from the mouse; None until it comes close to the agent.
            act = agent.act(obs)
            if act is not None:
                # The observation is captured before stepping, so
                # actions[t] is the action taken at states[t].
                episode_states.append(normalize(obs[:20]).astype(np.float32))
                episode_actions.append(normalize(np.array(act)).astype(np.float32))

            obs, reward, done, info = env.step(act)
            img = env.render(mode="human")
            clock.tick(args.control_hz)

        if not retry and episode_states:
            all_states.append(np.stack(episode_states))
            all_actions.append(np.stack(episode_actions))
            prev_end = episode_ends[-1] if episode_ends else 0
            episode_ends.append(prev_end + len(episode_states))
            save_atomic(output_path, all_states, all_actions, episode_ends)
            print(f"saved seed {seed} ({len(episode_states)} steps)")
        else:
            print(f"retry seed {seed}")
