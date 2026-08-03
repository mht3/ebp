'''
Roll out a trained Push-T policy in a few initial conditions and visualize the
trajectories as one row of panels, saved to images/push_t_<method>.png.
'''

import argparse
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from ebp.tasks.pusht import PushTKeypointsEnv, denormalize, normalize
from train import (
    DATASETS_DIR,
    load_dataset,
    load_model,
    load_proposal,
    load_stochastic_optimizer,
)

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


TEE_BAR = np.array([(-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)])
TEE_STEM = np.array([(-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)])
AGENT_RADIUS = 15.0

GOAL_COLOR = "#7fc97f"

TIME_CMAP = LinearSegmentedColormap.from_list(
    "time", ["#eef3f7", "#aec6dd", "#3f6ea5", "#0c1c4d"]
)
PUSHER_CMAP = LinearSegmentedColormap.from_list(
    "pusher", ["#f9e04c", "#f6a020", "#e0201b", "#4a0d67"]
)


def draw_tee(ax, pose, **kwargs) -> None:
    """Draw the T block at pose (x, y, theta) as two matplotlib polygons."""
    x, y, theta = pose
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    for vertices in (TEE_BAR, TEE_STEM):
        ax.fill(*(vertices @ rotation.T + [x, y]).T, **kwargs)


def multimodal_initial_state(
    goal_pose: np.ndarray,
    push_distance: float = 40.0,
    agent_offset: float = 30.0,
) -> np.ndarray:

    x, y, theta = goal_pose
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    block = np.array([x, y]) + rotation @ [0.0, push_distance]
    # Above the top of the bar (bar spans local y in [0, 30]).
    agent = block + rotation @ [0.0, -(30.0 + agent_offset)]
    return np.concatenate([agent, block, [theta]])


@torch.no_grad()
def rollout(env, model, method, stochastic_optimizer, sequence_length, seed,
            max_steps, device, reset_to_state=None, prediction_horizon=1,
            action_horizon=1):
    """Closed-loop rollout with receding-horizon action chunking; returns agent
    positions, block poses, and score. Each prediction is a chunk of
    prediction_horizon actions, the first action_horizon of which are executed
    before re-planning (Diffusion Policy Tp / Ta).

    The initial condition comes from `seed` unless an explicit 5-D
    `reset_to_state` ([agent_x, agent_y, block_x, block_y, block_angle]) is
    given.
    """
    model.eval()
    env.reset_to_state = reset_to_state
    if reset_to_state is None:
        env.seed(seed)
    obs = env.reset()
    info = env._get_info()
    history = collections.deque(
        [normalize(obs[:20]).astype(np.float32)] * sequence_length,
        maxlen=sequence_length,
    )
    agent_positions = [info["pos_agent"]]
    block_poses = [info["block_pose"]]
    success_step = None
    # Episode score: max over time of s = min(coverage / 0.95, 1), which the
    # env returns as its per-step reward (matches eval_push_t.py).
    score = 0.0
    step = 0
    while step < max_steps:
        x = torch.from_numpy(np.concatenate(history))[None].to(device)
        pred = model(x) if method == "mse" else stochastic_optimizer.infer(x, model)
        chunk = pred[0].cpu().numpy().astype(np.float64).reshape(prediction_horizon, -1)
        done = False
        for i in range(action_horizon):
            obs, reward, done, info = env.step(denormalize(chunk[i]))
            history.append(normalize(obs[:20]).astype(np.float32))
            agent_positions.append(info["pos_agent"])
            block_poses.append(info["block_pose"])
            score = max(score, float(reward))
            step += 1
            if done or step >= max_steps:
                break
        if done:
            success_step = step
            break

    return np.array(agent_positions), np.array(block_poses), score, success_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["mse", "ibc", "rnce"])
    parser.add_argument("--checkpoint", required=True, help="Path to a models/*.pt file.")
    parser.add_argument("--train_dataset", default="push_t_train.npz",
                        help="Filename in datasets/ (provides the action bounds).")
    parser.add_argument("--arch", default="mlp", choices=["mlp", "resnet"],
                        help="Must match the arch the checkpoint was trained with.")
    parser.add_argument(
        "--stochastic_optimizer",
        default="derivative_free",
        choices=["derivative_free", "langevin"],
    )
    parser.add_argument("--inference_samples", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None,
                        help="Langevin inference iterations. Must match training "
                        "to reproduce the wandb test/mse; defaults to the config.")
    parser.add_argument("--step_size", type=float, default=None,
                        help="Langevin step size override (R-NCE defaults to a "
                        "small support-preserving step; see load_stochastic_optimizer).")
    parser.add_argument("--noise_scale", type=float, default=None,
                        help="Langevin noise scale override (R-NCE default 1.0).")
    parser.add_argument("--sequence_length", type=int, default=2)
    parser.add_argument("--prediction_horizon", type=int, default=1,
                        help="Action-chunk length the checkpoint was trained with (Tp).")
    parser.add_argument("--action_horizon", type=int, default=1,
                        help="Actions executed per predicted chunk before re-planning (Ta).")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[100000, 100001, 100002, 100003],
                        help="Initial conditions (Diffusion Policy's test seeds).")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Overlay many short rollouts per initial condition (Diffusion "
        "Policy Fig. 3) instead of one full rollout per panel.",
    )
    parser.add_argument("--num_rollouts", type=int, default=20,
                        help="Rollouts per panel in --multimodal mode.")
    parser.add_argument("--multimodal_steps", type=int, default=40,
                        help="Steps per rollout in --multimodal mode.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, target_bounds = load_dataset(
        os.path.join(DATASETS_DIR, args.train_dataset),
        args.sequence_length, args.prediction_horizon,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = load_model(
        "push_t", args.method, sequence_length=args.sequence_length, arch=args.arch,
        prediction_horizon=args.prediction_horizon,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    # R-NCE inference warm-starts Langevin from the checkpointed proposal (Alg. 2).
    proposal = None
    optimizer_name = args.stochastic_optimizer
    if args.method == "rnce":
        proposal = load_proposal(
            "push_t", sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
        ).to(device)
        proposal.load_state_dict(checkpoint["proposal"])
        optimizer_name = "langevin"
    stochastic_optimizer = load_stochastic_optimizer(
        optimizer_name, target_bounds, args.device,
        inference_samples=args.inference_samples, iters=args.iters,
        proposal=proposal, step_size=args.step_size, noise_scale=args.noise_scale,
    )

    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_action=False, **kp_kwargs)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    if args.multimodal:
        # A single symmetric initial condition; many short rollouts overlaid,
        # block poses as time-shaded fills, agent trajectories as time-shaded
        # lines.
        fig, axes = plt.subplots(1, 1, figsize=(3, 3), squeeze=False)
        ax = axes[0][0]
        env.seed(0)
        env.reset()  # populates env.goal_pose for the crafted state below
        initial_state = multimodal_initial_state(env.goal_pose)
        for _ in range(args.num_rollouts):
            agent_positions, block_poses, _, _ = rollout(
                env, model, args.method, stochastic_optimizer,
                args.sequence_length, None, args.multimodal_steps, device,
                reset_to_state=initial_state,
                prediction_horizon=args.prediction_horizon,
                action_horizon=args.action_horizon,
            )
            steps = len(block_poses)
            for t in range(steps):
                draw_tee(ax, block_poses[t],
                         color=TIME_CMAP(t / max(steps - 1, 1)), zorder=1)
            points = agent_positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            ax.add_collection(LineCollection(
                segments, colors=PUSHER_CMAP(np.linspace(0, 1, len(segments))),
                linewidth=1.5, capstyle="round", zorder=3,
            ))
        # The (shared) initial agent position, as in Diffusion Policy Fig. 3.
        ax.add_patch(plt.Circle(initial_state[:2], AGENT_RADIUS,
                                color="royalblue", zorder=4))
        print(f"{args.num_rollouts} rollouts of {args.multimodal_steps} steps "
              "from the symmetric initial condition")
    else:
        fig, axes = plt.subplots(
            1, len(args.seeds), figsize=(3 * len(args.seeds), 3), squeeze=False
        )
        for ax, seed in zip(axes[0], args.seeds):
            agent_positions, block_poses, score, success_step = rollout(
                env, model, args.method, stochastic_optimizer,
                args.sequence_length, seed, args.max_steps, device,
                prediction_horizon=args.prediction_horizon,
                action_horizon=args.action_horizon,
            )

            # Paint the block at every timestep in chronological order, so
            # later (darker) poses cover earlier (lighter) ones.
            steps = len(agent_positions)
            for t in range(steps):
                draw_tee(ax, block_poses[t],
                         color=TIME_CMAP(t / max(steps - 1, 1)), zorder=1)
            draw_tee(ax, block_poses[-1], facecolor="none",
                     edgecolor="black", linewidth=0.8, zorder=2)
            # The pusher path as a yellow-orange-red-purple heatmap line, and
            # the starting position marked with a blue dot.
            points = agent_positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            ax.add_collection(LineCollection(
                segments, colors=PUSHER_CMAP(np.linspace(0, 1, len(segments))),
                linewidth=1.5, capstyle="round", zorder=3,
            ))
            ax.add_patch(plt.Circle(agent_positions[0], AGENT_RADIUS,
                                    color="royalblue", zorder=4))

            # Score above each panel (Singh et al. Table 5 style).
            ax.set_title("r={:.3f}".format(score))
            outcome = (
                f"success at step {success_step}" if success_step is not None
                else f"{steps - 1} steps"
            )
            print(f"seed {seed}: r={score:.3f} ({outcome})")

    for ax in axes[0]:
        # goal_pose exists only after the env's first reset; a high zorder
        # draws the goal on top of the moving block (as in Singh et al. Table 5).
        draw_tee(ax, env.goal_pose, color=GOAL_COLOR, zorder=5)

        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)  # y-axis down, matching the env's screen coordinates
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    suffix = "_multimodal" if args.multimodal else ""
    # Include the action horizon so figures for different Tp/Ta don't overwrite
    # each other (Tp=1 keeps the original unsuffixed filenames).
    horizon = "" if args.prediction_horizon == 1 else f"_p{args.prediction_horizon}a{args.action_horizon}"
    plot_path = os.path.join(IMAGES_DIR, f"push_t_{args.method}{horizon}{suffix}.png")
    fig.savefig(plot_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to images/{os.path.basename(plot_path)}")
