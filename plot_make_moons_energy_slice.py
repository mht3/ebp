import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from train import DATASETS_DIR, MODELS_DIR, load_dataset, load_model, load_proposal

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


@torch.no_grad()
def energy_slice(model, obs, x2_grid, device):
    """Energy E(obs, x2) over the action grid for a fixed scalar state `obs`."""
    states = torch.tensor([[obs]], dtype=torch.float32, device=device)
    candidates = torch.from_numpy(x2_grid.astype(np.float32)).reshape(1, -1, 1).to(device)
    return model(states, candidates).cpu().numpy().reshape(-1)


def langevin_descent(model, obs, particles, iters, step, noise, decay, bounds, device, seed=0):
    """Run Langevin descent of `particles` (1D actions) on E(obs, .), recording
    every iteration. Returns a list of length iters+1 of position arrays.

    The noise scale is annealed to 0 as noise * (1 - t/iters)**decay: a high
    initial scale kicks samples out of shallow local minima, and the steeper
    (decay > 1) cooldown then settles them crisply into the wells so they don't
    jitter once converged. Samples are clamped to the action `bounds` (low, high)
    each step, like the real inference optimizer."""
    torch.manual_seed(seed)
    low, high = bounds
    state = torch.tensor([[obs]], dtype=torch.float32, device=device)
    p = torch.as_tensor(particles, dtype=torch.float32, device=device).reshape(1, -1, 1)
    traj = [p.detach().cpu().numpy().reshape(-1).copy()]
    for t in range(iters):
        p = p.detach().requires_grad_(True)
        grad = torch.autograd.grad(model(state, p).sum(), p)[0]
        noise_t = noise * (1 - t / iters) ** decay
        p = p.detach() - step * grad + noise_t * np.sqrt(2 * step) * torch.randn_like(p)
        p = p.clamp(low, high)
        traj.append(p.detach().cpu().numpy().reshape(-1).copy())
    return traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dataset", default="make_moons_n_1000_seed_0.npz",
                        help="Filename in datasets/ (provides the action bounds and "
                        "the models/ checkpoint stem).")
    parser.add_argument("--obs", type=float, default=0.4,
                        help="State x1 to slice the energy at (default inside the strip).")
    parser.add_argument("--n_particles", type=int, default=15,
                        help="Uniformly-spaced action samples animated in the IBC GIF.")
    parser.add_argument("--iters", type=int, default=90,
                        help="Langevin iterations for the IBC descent animation "
                        "(the R-NCE animation uses a few more).")
    parser.add_argument("--step", type=float, default=0.014, help="IBC Langevin step size.")
    parser.add_argument("--noise", type=float, default=1.0,
                        help="Initial Langevin noise scale (annealed to 0). High enough "
                        "to kick samples out of shallow local minima.")
    parser.add_argument("--noise_decay", type=float, default=1.5,
                        help="Noise anneal exponent: noise * (1 - t/iters)**decay. "
                        ">1 cools faster late so converged samples stop jittering.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--hold", type=int, default=15,
                        help="Extra frames holding the converged state before the loop restarts.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, bounds = load_dataset(os.path.join(DATASETS_DIR, args.train_dataset))
    low, high = float(bounds[0, 0]), float(bounds[1, 0])
    x2_grid = np.linspace(low, high, 400)
    stem = os.path.splitext(args.train_dataset)[0]
    os.makedirs(IMAGES_DIR, exist_ok=True)

    def normalized_energy(model):
        e = energy_slice(model, args.obs, x2_grid, device)
        return (e - e.min()) / (np.ptp(e) + 1e-8)

    train = np.load(os.path.join(DATASETS_DIR, args.train_dataset))
    tr_s, tr_a = train["states"].reshape(-1), train["actions"].reshape(-1)

    def make_gif(label, color, model, energy_norm, particles0, gif_name,
                 step, noise, iters, proposal_ci=None):
        """Two-panel descent animation: (state, action) plane on the left, energy
        slice on the right. Samples descend E(obs, .) from `particles0` in tandem.
        If `proposal_ci` = (mu, ci) is given (R-NCE), the proposal mean and 95% CI
        that warm-start the chain are drawn on the energy panel."""
        traj = langevin_descent(model, args.obs, particles0, iters, step,
                                noise, args.noise_decay, (low, high), device)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Left: (state, action) plane. Training data gray; at the fixed state the
        # samples move in ACTION and land on the two moons (the two valid actions).
        axL.scatter(tr_s, tr_a, s=8, c="0.7", alpha=0.6, label="train", zorder=1)
        axL.axvline(args.obs, color="0.4", ls="--", lw=1.2, zorder=1)
        scatL = axL.scatter([], [], s=45, c="tab:blue", edgecolors="black", zorder=3,
                            label="samples")
        axL.grid(True, alpha=0.3)
        axL.set_xlim(tr_s.min() - 0.2, tr_s.max() + 0.2)
        axL.set_ylim(low, high)
        axL.set_xlabel("$x_1$ (state)")
        axL.set_ylabel("$x_2$ (action)")
        axL.legend(loc="upper right")

        # Right: energy slice at s = obs; the same samples descend the landscape.
        axR.plot(x2_grid, energy_norm, color=color, lw=2, label=f"{label} energy", zorder=2)
        if proposal_ci is not None:
            mu, ci = proposal_ci
            axR.axvspan(mu - ci, mu + ci, color="tab:purple", alpha=0.15,
                        label="proposal 95% CI")
            axR.axvline(mu, color="tab:purple", ls="--", lw=1.5,
                        label=r"proposal mean $\mu(s)$")
        scatR = axR.scatter([], [], s=45, c="tab:blue", edgecolors="black", zorder=3,
                            label="action samples")
        axR.grid(True, alpha=0.3)
        axR.set_xlim(low, high)
        axR.set_ylim(-0.05, 1.05)
        axR.set_xlabel("$x_2$ (action)")
        axR.set_ylabel("normalized energy")
        axR.legend(loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        def update(k):
            k = min(k, len(traj) - 1)  # extra frames hold the converged state
            xk = traj[k]
            scatL.set_offsets(np.column_stack([np.full_like(xk, args.obs), xk]))
            scatR.set_offsets(np.column_stack([xk, np.interp(xk, x2_grid, energy_norm)]))
            fig.suptitle(f"{label} inference at $s = {args.obs:g}$: iter {k}/{iters}")
            return scatL, scatR

        anim = FuncAnimation(fig, update, frames=len(traj) + args.hold, blit=False)
        gif_path = os.path.join(IMAGES_DIR, gif_name)
        anim.save(gif_path, writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"Saved plot to images/{gif_name}")

    # IBC: uniform init -- the global (uniform-negative) energy lets a uniform
    # sample flow to the minima.
    ibc = load_model("make_moons", "ibc").to(device)
    ibc.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"ibc_{stem}.pt"), map_location=device)["model"]
    )
    ibc.eval()
    ibc_init = np.linspace(low + 0.1, high - 0.1, args.n_particles)
    make_gif("IBC", "tab:orange", ibc, normalized_energy(ibc), ibc_init,
             "make_moons_energy_slice.gif", step=args.step, noise=args.noise,
             iters=args.iters)

    # R-NCE: warm-start from the learned proposal q(a | s) (Alg. 2) instead of
    # uniform -- samples are drawn from N(mu, sigma) and refined on the energy.
    rnce = load_model("make_moons", "rnce").to(device)
    rnce.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"rnce_{stem}.pt"), map_location=device)["model"]
    )
    rnce.eval()
    proposal = load_proposal("make_moons").to(device)
    proposal.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"rnce_{stem}.pt"), map_location=device)["proposal"]
    )
    proposal.eval()
    with torch.no_grad():
        state = torch.tensor([[args.obs]], dtype=torch.float32, device=device)
        mu = proposal.mean_net(state).item()
        sigma = proposal.log_std.exp().item()
    rnce_init = np.clip(
        mu + sigma * np.random.default_rng(0).standard_normal(args.n_particles), low, high
    )
    # R-NCE's energy trough is shallow, so it needs a hotter chain (larger step
    # and noise) to reach the wells than IBC's deep-welled landscape.
    make_gif("R-NCE", "tab:green", rnce, normalized_energy(rnce), rnce_init,
             "make_moons_rnce_energy_slice.gif", step=0.02, noise=1.3, iters=100,
             proposal_ci=(mu, 1.96 * sigma))
    print(f"(proposal mu={mu:.3f}, sigma={sigma:.3f})")
