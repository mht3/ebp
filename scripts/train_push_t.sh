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
        # Training-only flags: extra_args is shared with the plot/eval calls
        # below, which don't accept these. 64 negatives keeps IBC training
        # tractable; 20 Langevin iters at training time.
        train_args=(--num_counterexamples 64 --iters 20)
    elif [ "$method" = "rnce" ]; then
        # R-NCE forces langevin inference in train.py; plot/eval still need it told.
        extra_args=(--stochastic_optimizer langevin --inference_samples 1024)
        # 64 negatives from the learnable Gaussian proposal; 20 Langevin iters at
        # inference; L2 on the proposal MLE off by default.
        train_args=(--num_counterexamples 64 --l2_weight 0.0 --iters 20)
    fi

    python train.py \
        --method "$method" \
        --task push_t \
        --train_dataset "$train" \
        --test_dataset "$test" \
        --sequence_length 2 \
        --epochs 2000 \
        --batch_size 256 \
        --eval_every 200 \
        "${extra_args[@]}" \
        "${train_args[@]}"

    python plot_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        "${extra_args[@]}"

    # Multimodal behavior figure (Diffusion Policy Fig. 3): many short
    # rollouts overlaid per initial condition.
    python plot_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        --multimodal \
        "${extra_args[@]}"

    # Score:mean max-coverage score over random initial conditions x rollouts).
    python eval_push_t.py \
        --method "$method" \
        --checkpoint "models/${method}_push_t_train.pt" \
        --train_dataset "$train" \
        --num_seeds 20 \
        --num_rollouts 32 \
        --max_steps 200 \
        "${extra_args[@]}"
done
