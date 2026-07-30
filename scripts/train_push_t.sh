#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

train="push_t_train.npz"
test="push_t_test.npz"

for horizon in "1 1" "16 8"; do
    read -r tp ta <<< "$horizon"
    suffix=""
    [ "$tp" -gt 1 ] && suffix="_p${tp}"
    horizon_args=(--prediction_horizon "$tp" --action_horizon "$ta")

    for method in mse ibc rnce; do
        ckpt="models/${method}_push_t_train${suffix}.pt"

        if [ "$tp" -eq 1 ]; then
            epochs=2000; batch_size=256; iters=20
            lr_args=(); step_args=(); noise_args=()
            if [ "$method" = "ibc" ]; then
                K=64; l2_args=()
            else
                K=32; l2_args=(--l2_weight 0.01)
            fi
        else
            epochs=2000; batch_size=256; iters=20
            lr_args=()
            if [ "$method" = "ibc" ]; then
                K=64; l2_args=()
                step_args=(--step_size 0.03); noise_args=(--noise_scale 0.3)
            else
                K=32; l2_args=(--l2_weight 0.01)
                step_args=(--step_size 2e-4); noise_args=(--noise_scale 0.5)
            fi
        fi

        extra_args=(--stochastic_optimizer langevin --inference_samples 1024 \
                    --iters "$iters" "${step_args[@]}" "${noise_args[@]}")
        train_args=(--num_counterexamples "$K" "${l2_args[@]}")

        python train.py \
            --method "$method" \
            --task push_t \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --sequence_length 2 \
            --epochs "$epochs" \
            --batch_size "$batch_size" \
            --eval_every 1000 \
            "${horizon_args[@]}" \
            "${lr_args[@]}" \
            "${extra_args[@]}" \
            "${train_args[@]}"

        if [ "$tp" -eq 1 ]; then
            python plot_push_t.py \
                --method "$method" \
                --checkpoint "$ckpt" \
                --train_dataset "$train" \
                "${horizon_args[@]}" \
                "${extra_args[@]}"

            python plot_push_t.py \
                --method "$method" \
                --checkpoint "$ckpt" \
                --train_dataset "$train" \
                --multimodal \
                "${horizon_args[@]}" \
                "${extra_args[@]}"
        fi

        echo "=== eval: ${method}  Tp=${tp} Ta=${ta} ==="
        python eval_push_t.py \
            --method "$method" \
            --checkpoint "$ckpt" \
            --train_dataset "$train" \
            --num_seeds 256 \
            --num_rollouts 5 \
            --max_steps 200 \
            "${horizon_args[@]}" \
            "${extra_args[@]}"
    done
done
