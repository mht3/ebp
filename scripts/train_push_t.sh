#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

train="push_t_train.npz"
test="push_t_test.npz"

# trying out different action horizons to see what's best. first is planned/predicted action, second is executed action.
for horizon in "1 1" "2 2" "4 4" "16 8"; do
    read -r tp ta <<< "$horizon"
    suffix=""
    [ "$tp" -gt 1 ] && suffix="_p${tp}"
    horizon_args=(--prediction_horizon "$tp" --action_horizon "$ta")

    for method in mse ibc rnce; do
        ckpt="models/${method}_push_t_train${suffix}.pt"

        # Params fpr eeacj action horizon setting. Different action dimensions require different optimization parameters.
        samples=1024
        case "${method}_${tp}" in
            rnce_1)  st=3.5e-5; ns=0.25 ;;
            rnce_2)  st=5e-5;   ns=0.5  ;;
            rnce_4)  st=2e-5;   ns=1.0  ;;
            rnce_16) st=1e-4;   ns=0.05 ;;
            ibc_1)   st=2e-3;   ns=0.25 ;;
            ibc_2)   st=2e-5;   ns=0.1;  samples=4096 ;;
            ibc_4)   st=1e-2;   ns=0.5  ;;
            ibc_16)  st=0.1;    ns=0.6  ;;
            *)       st="";     ns=""   ;;   # mse has no inference parameters
        esac

        extra_args=(--stochastic_optimizer langevin --inference_samples "$samples" --iters 20)
        [ -n "$st" ] && extra_args+=(--step_size "$st" --noise_scale "$ns")

        if [ "$method" = "mse" ]; then
            train_args=(--l2_weight 0.0)
        elif [ "$method" = "ibc" ]; then
            train_args=(--num_counterexamples 64)
        else
            # l2_weight acts on the proposal's mean network; 0.001 was optimal at
            # every horizon tested (more L2 widens the proposal and costs score).
            train_args=(--num_counterexamples 32 --l2_weight 0.001)
        fi

        python train.py \
            --method "$method" \
            --task push_t \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --sequence_length 2 \
            --epochs 2000 \
            --batch_size 256 \
            --eval_every 2000 \
            "${horizon_args[@]}" \
            "${extra_args[@]}" \
            "${train_args[@]}"

        # Rollout figure and multimodal figure. Filenames carry the horizon, so
        # each configuration keeps its own images.
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

        echo "=== eval: ${method}  Tp=${tp} Ta=${ta} ==="
        python eval_push_t.py \
            --method "$method" \
            --checkpoint "$ckpt" \
            --train_dataset "$train" \
            --num_seeds 256 \
            --num_rollouts 1 \
            --max_steps 200 \
            "${horizon_args[@]}" \
            "${extra_args[@]}"
    done
done
