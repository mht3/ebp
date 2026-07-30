'''
Score a trained Push-T policy with the evaluation protocol from Singh et al.
(arXiv:2309.05803).

python eval_push_t.py --method ibc --checkpoint models/ibc_push_t_train.pt \
    --stochastic_optimizer langevin --inference_samples 1024

At each timestep the score of the current configuration is s = min(r / 0.95, 1)
where r is the ratio of the block-goal intersection area to the block area (the
env returns exactly this as its per-step reward). An episode ends when s = 1 or
after --max_steps steps, and its score is the maximum s over the episode. The
reported number is the mean episode score over --num_seeds random initial
conditions, each rolled out --num_rollouts times, as mean +/- a 95% confidence
interval (normal approximation, 1.96 * std / sqrt(n)) -- the same format as Singh
et al. Table 6 (the paper uses 256 seeds x 32 rollouts; the default here is
smaller so it runs in reasonable time). MSE is deterministic, so its rollouts
within a seed are identical and --num_rollouts only adds variance for IBC.
'''

import argparse
import collections
import os

import numpy as np
import torch

from ebp.tasks.pusht import PushTKeypointsEnv, denormalize, normalize
from train import (
    DATASETS_DIR,
    load_dataset,
    load_model,
    load_proposal,
    load_stochastic_optimizer,
)


@torch.no_grad()
def score_rollout(env, model, method, stochastic_optimizer, sequence_length,
                  seed, max_steps, device, prediction_horizon=1, action_horizon=1):
    """Closed-loop rollout with receding-horizon action chunking; returns the
    episode's max score (max reward). Each prediction is a chunk of
    prediction_horizon actions, of which the first action_horizon are executed
    before re-planning (Diffusion Policy Tp / Ta)."""
    env.seed(seed)
    obs = env.reset()
    history = collections.deque(
        [normalize(obs[:20]).astype(np.float32)] * sequence_length,
        maxlen=sequence_length,
    )
    best_score = 0.0
    step = 0
    while step < max_steps:
        x = torch.from_numpy(np.concatenate(history))[None].to(device)
        pred = model(x) if method == "mse" else stochastic_optimizer.infer(x, model)
        chunk = pred[0].cpu().numpy().astype(np.float64).reshape(prediction_horizon, -1)
        for i in range(action_horizon):
            obs, reward, done, _ = env.step(denormalize(chunk[i]))
            history.append(normalize(obs[:20]).astype(np.float32))
            best_score = max(best_score, float(reward))
            step += 1
            if done or step >= max_steps:
                break
        if done:
            break
    return best_score


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
                        help="Langevin inference iterations. Match training to "
                        "reproduce the wandb test/mse; defaults to the config.")
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
    parser.add_argument("--num_seeds", type=int, default=20,
                        help="Random initial conditions (paper uses 256).")
    parser.add_argument("--num_rollouts", type=int, default=32,
                        help="Rollouts per initial condition (paper uses 32). "
                        "MSE is deterministic, so this only adds variance for IBC.")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Episode step cap (paper uses 200).")
    parser.add_argument("--seed_start", type=int, default=100000,
                        help="First initial-condition seed (Diffusion Policy's "
                        "test seeds start here).")
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
    model.eval()

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

    scores = []
    seed_means = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        seed_scores = [
            score_rollout(env, model, args.method, stochastic_optimizer,
                          args.sequence_length, seed, args.max_steps, device,
                          args.prediction_horizon, args.action_horizon)
            for _ in range(args.num_rollouts)
        ]
        scores.extend(seed_scores)
        seed_means.append(np.mean(seed_scores))
        print(f"seed {seed}: mean score {seed_means[-1]:.3f}")

    scores = np.array(scores)
    seed_means = np.array(seed_means)
    mean = scores.mean()
    # 95% CI of the mean (normal approximation). MSE is deterministic, so the
    # num_rollouts rollouts per seed are identical and the independent units are
    # the seeds -- use n = num_seeds over the per-seed scores (n = seeds*rollouts
    # would overstate precision). The stochastic methods (IBC/R-NCE) do vary per
    # rollout, so they keep the full n = seeds*rollouts.
    if args.method == "mse":
        n = len(seed_means)
        std = seed_means.std(ddof=1) if n > 1 else 0.0
        basis = f"{args.num_seeds} seeds"
    else:
        n = len(scores)
        std = scores.std(ddof=1) if n > 1 else 0.0
        basis = f"{args.num_seeds} seeds x {args.num_rollouts} rollouts"
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    print(f"{args.method}: mean score {mean:.3f} +/- {ci:.3f} (95% CI, n={n} = {basis})")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_t_scores.txt")
    with open(results_path, "a") as f:
        f.write(f"{args.method}\tTp={args.prediction_horizon}\tTa={args.action_horizon}\t"
                f"score {mean:.3f} +/- {ci:.3f}\t(95% CI, n={n} = {basis})\n")
