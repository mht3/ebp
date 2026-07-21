#!/usr/bin/env bash
# Train MSE and IBC on the coordinate regression task for n=10 and n=30, then
# plot each model's test predictions to assets/.
set -euo pipefail

cd "$(dirname "$0")/.."

for n in 10 30; do
    train="coordinate_regression_n_${n}_seed_0.npz"
    test="coordinate_regression_n_${n}_test.npz"

    for method in mse ibc; do
        python train.py \
            --method "$method" \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --eval_every 50

        python plot_coordinate_regression.py \
            --method "$method" \
            --checkpoint "models/${method}_coordinate_regression_n_${n}_seed_0.pt" \
            --train_dataset "$train" \
            --test_dataset "$test"
    done
done
