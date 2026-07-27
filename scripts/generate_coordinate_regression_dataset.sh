#!/usr/bin/env bash
# Generate train and test sets for the coordinate regression task at n=10 and n=30.
set -euo pipefail

cd "$(dirname "$0")/.."

for n in 10 30; do
    # Train set (seed 0).
    python generate_data.py \
        --task coordinate_regression \
        --samples "$n" \
        --seed 0 \
        --filename "coordinate_regression_n_${n}_seed_0.npz"

    # 500-sample test set (seed 1), excluding any train coordinates.
    python generate_data.py \
        --task coordinate_regression \
        --samples 500 \
        --seed 1 \
        --filename "coordinate_regression_n_${n}_test.npz" \
        --exclude "coordinate_regression_n_${n}_seed_0.npz"
done
