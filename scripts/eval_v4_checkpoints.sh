#!/bin/bash
# Eval script for v4 checkpoints (beta=0.001, lr=5e-5)
# Run this on the second node (node2) which has free GPUs

set -e
cd /efs/rlvr-experiments

CHECKPOINT_DIR="/efs/rlvr-experiments/checkpoints"
RESULTS_DIR="/efs/rlvr-experiments/experiments/math_only_minerva_v4/evals"
mkdir -p "$RESULTS_DIR"

# Watch for new checkpoints and run evals
while true; do
    # Find v4 checkpoints that haven't been evaluated yet
    for ckpt in $(ls "$CHECKPOINT_DIR" | grep "math_only_minerva_v4_step" | sort -V); do
        step=$(echo "$ckpt" | grep -oP 'step\K\d+')

        # Skip if already evaluated
        if [ -f "$RESULTS_DIR/minerva_math_step${step}.json" ]; then
            continue
        fi

        echo "=== Evaluating checkpoint: $ckpt (step $step) ==="
        ckpt_path="$CHECKPOINT_DIR/$ckpt"

        # Run all 5 evals
        echo "[$(date)] Running minerva_math..."
        uv run lm_eval --model vllm --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks minerva_math --num_fewshot 4 --batch_size auto \
            --output_path "$RESULTS_DIR/minerva_math_step${step}.json" 2>&1 | tee -a "$RESULTS_DIR/minerva_math_step${step}.log"

        echo "[$(date)] Running gsm8k..."
        uv run lm_eval --model vllm --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks gsm8k --num_fewshot 8 --batch_size auto \
            --output_path "$RESULTS_DIR/gsm8k_step${step}.json" 2>&1 | tee -a "$RESULTS_DIR/gsm8k_step${step}.log"

        echo "[$(date)] Running gsm8k_cot..."
        uv run lm_eval --model vllm --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks gsm8k_cot --num_fewshot 8 --batch_size auto \
            --output_path "$RESULTS_DIR/gsm8k_cot_step${step}.json" 2>&1 | tee -a "$RESULTS_DIR/gsm8k_cot_step${step}.log"

        echo "[$(date)] Running mbpp..."
        uv run lm_eval --model vllm --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks mbpp --num_fewshot 3 --batch_size auto \
            --output_path "$RESULTS_DIR/mbpp_step${step}.json" 2>&1 | tee -a "$RESULTS_DIR/mbpp_step${step}.log"

        echo "[$(date)] Running ifeval..."
        uv run lm_eval --model vllm --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks ifeval --num_fewshot 0 --batch_size auto \
            --output_path "$RESULTS_DIR/ifeval_step${step}.json" 2>&1 | tee -a "$RESULTS_DIR/ifeval_step${step}.log"

        echo "[$(date)] Completed all evals for step $step"

        # Extract and print results
        echo "=== Results for step $step ==="
        for task in minerva_math gsm8k gsm8k_cot mbpp ifeval; do
            if [ -f "$RESULTS_DIR/${task}_step${step}.json" ]; then
                result=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/${task}_step${step}.json')); print(f'{task}: {list(d[\"results\"].values())[0].get(\"acc,none\", list(d[\"results\"].values())[0].get(\"exact_match,none\", list(d[\"results\"].values())[0].get(\"prompt_level_strict_acc,none\", \"?\")))*100:.2f}%')" 2>/dev/null || echo "$task: parse error"
            fi
        done
        echo ""
    done

    echo "[$(date)] Waiting 60s for new checkpoints..."
    sleep 60
done
