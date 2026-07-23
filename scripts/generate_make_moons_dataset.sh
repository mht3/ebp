#!/usr/bin/env bash
# Generate train and test sets for the make_moons task.
# The train set (seed 0) excludes the multimodal strip x1 in [0.3, 0.5]; the

set -euo pipefail

cd "$(dirname "$0")/.."

python generate_data.py \
    --task make_moons \
    --samples 1000 \
    --seed 0 \
    --filename make_moons_n_1000_seed_0.npz

python generate_data.py \
    --task make_moons \
    --samples 200 \
    --seed 1 \
    --make_moons_test \
    --filename make_moons_test.npz
