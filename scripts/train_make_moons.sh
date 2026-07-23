#!/usr/bin/env bash
# Train MSE and IBC on the make_moons task, then plot each model's predicted
# actions over the held-out strip (and the IBC energy landscape) to images/.
#
# IBC uses the derivative-free optimizer (train.py's default), which is well
# suited to the 1D action here.
set -euo pipefail

cd "$(dirname "$0")/.."

train="make_moons_n_1000_seed_0.npz"
test="make_moons_test.npz"

for method in mse ibc rnce; do
    # extra_args are shared with the plot call; train_args are training-only.
    extra_args=()
    train_args=()
    if [ "$method" = "ibc" ]; then
        extra_args=(--stochastic_optimizer langevin)
    elif [ "$method" = "rnce" ]; then
        # R-NCE forces langevin inference in train.py; the plotter still needs it told.
        extra_args=(--stochastic_optimizer langevin)
        # Explicit R-NCE knobs: K negatives from the learnable Gaussian proposal,
        train_args=(--num_counterexamples 64 --l2_weight 0.0)
    fi
    python train.py \
        --method "$method" \
        --task make_moons \
        --train_dataset "$train" \
        --test_dataset "$test" \
        --epochs 2000 \
        --batch_size 64 \
        --eval_every 200 \
        --iters 10 \
        "${extra_args[@]}" \
        "${train_args[@]}"

    python plot_make_moons.py \
        --method "$method" \
        --checkpoint "models/${method}_make_moons_n_1000_seed_0.pt" \
        --train_dataset "$train" \
        --test_dataset "$test" \
        "${extra_args[@]}"
done
