#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

train="make_moons_n_1000_seed_0.npz"
test="make_moons_test.npz"

for method in mse ibc rnce; do
    extra_args=()
    train_args=()
    if [ "$method" = "ibc" ]; then
        extra_args=(--stochastic_optimizer langevin)
    elif [ "$method" = "rnce" ]; then
        extra_args=(--stochastic_optimizer langevin)
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
