#!/usr/bin/env bash
# Generate train and test sets for the coordinate regression task at n=10 and n=30.
#
# Train sets use seed 0. Following Kevin Zakka's IBC, each test set has 500
# samples (seed 1) and passes --exclude <train file> so any coordinate colliding
# with the train set is resampled, giving a leakage-free measure of
# generalization to unseen locations.
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
