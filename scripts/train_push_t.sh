#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

train="push_t_train.npz"
test="push_t_test.npz"

for method in mse ibc rnce; do
    extra_args=()
    train_args=()
    if [ "$method" = "ibc" ]; then
        extra_args=(--stochastic_optimizer langevin --inference_samples 1024)
        train_args=(--num_counterexamples 64 --iters 20)
    elif [ "$method" = "rnce" ]; then
        extra_args=(--stochastic_optimizer langevin --inference_samples 1024 --iters 20)
        train_args=(--num_counterexamples 64 --l2_weight 0.01)
    fi

    python train.py \
        --method "$method" \
        --task push_t \
        --train_dataset "$train" \
        --test_dataset "$test" \
        --sequence_length 2 \
        --epochs 2000 \
        --batch_size 256 \
        --eval_every 1000 \
        "${extra_args[@]}" \
        "${train_args[@]}"

    python plot_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        "${extra_args[@]}"

    python plot_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        --multimodal \
        "${extra_args[@]}"

    python eval_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        --num_seeds 20 \
        --num_rollouts 32 \
        --max_steps 200 \
        "${extra_args[@]}"
done
