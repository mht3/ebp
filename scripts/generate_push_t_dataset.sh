#!/usr/bin/env bash
# Download Diffusion Policy's official Push-T demonstration data (206 human
# mouse-teleop episodes, 25650 steps, ~100MB) and convert it to
# datasets/push_t_train.npz + datasets/push_t_test.npz.
#
# Extra demonstrations can be recorded with demo_push_t.py.
set -euo pipefail

cd "$(dirname "$0")/.."

zarr_path="datasets/pusht/pusht_cchi_v7_replay.zarr"
if [ ! -d "$zarr_path" ]; then
    mkdir -p datasets
    wget -O datasets/pusht.zip https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    unzip -o datasets/pusht.zip -d datasets/
    rm datasets/pusht.zip
fi

python convert_pusht_dataset.py --zarr "$zarr_path" --test_episodes 20 --seed 0
