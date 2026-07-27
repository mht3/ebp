#!/usr/bin/env bash
# Train MSE and IBC on the coordinate regression task for n=10 and n=30,
set -euo pipefail

cd "$(dirname "$0")/.."

for n in 10 30; do
    train="coordinate_regression_n_${n}_seed_0.npz"
    test="coordinate_regression_n_${n}_test.npz"

    for method in mse ibc rnce; do
        # extra_args are shared with the plot call; train_args are training-only.
        extra_args=()
        train_args=()
        if [ "$method" = "mse" ]; then
            train_args=(--l2_weight 0.0001)
        elif [ "$method" = "ibc" ]; then
            extra_args=(--stochastic_optimizer derivative_free)
            train_args=(--num_counterexamples 64)
        elif [ "$method" = "rnce" ]; then
            extra_args=(--stochastic_optimizer langevin --step_size 5e-5 --iters 20)
            # note for n=10, some slight l2 regularization helps and is what is in the readme results (5e-4). for n=30, l2 reg inflates the std and underfits.
            train_args=(--num_counterexamples 64 --l2_weight 0.0)
        fi

        python train.py \
            --method "$method" \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --epochs 2000 \
            --batch_size 8 \
            --eval_every 200 \
            --coord_conv \
            --iters 20 \
            "${extra_args[@]}" \
            "${train_args[@]}"

        python plot_coordinate_regression.py \
            --method "$method" \
            --checkpoint "models/${method}_coordinate_regression_n_${n}_seed_0.pt" \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --coord_conv \
            "${extra_args[@]}"
    done
done
