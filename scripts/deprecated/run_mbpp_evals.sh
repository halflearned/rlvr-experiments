#!/bin/bash
# Run MBPP evals for all checkpoints
# Usage: ./run_mbpp_evals.sh <gpu_pair> <start_idx> <end_idx>

set -e

GPU_PAIR=${1:-"0,1"}
START_IDX=${2:-0}
END_IDX=${3:-999}

export HF_ALLOW_CODE_EVAL=1

EVAL_BIN="/efs/rlvr-experiments/.venv/bin/lm_eval"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"

# List of checkpoints to evaluate
CHECKPOINTS=(
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-025110/mixed_lr1e5_random_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step600"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step600"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step600"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step600"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step300"
)

TOTAL=${#CHECKPOINTS[@]}
echo "MBPP eval worker on GPUs $GPU_PAIR, processing checkpoints $START_IDX to $END_IDX (total: $TOTAL)"

for i in $(seq $START_IDX $END_IDX); do
    if [ $i -ge $TOTAL ]; then
        echo "No more checkpoints"
        break
    fi

    CKPT_PATH="${CHECKPOINTS[$i]}"
    CKPT_NAME=$(basename "$CKPT_PATH")
    OUTPUT_NAME="${CKPT_NAME}_mbpp_3shot"

    # Skip if already done
    if [ -d "$OUTPUT_DIR/$OUTPUT_NAME" ] && find "$OUTPUT_DIR/$OUTPUT_NAME" -name "results*.json" | grep -q .; then
        echo "[$i] SKIP $OUTPUT_NAME (already done)"
        continue
    fi

    echo "[$i] Running MBPP eval for $CKPT_NAME on GPUs $GPU_PAIR"

    CUDA_VISIBLE_DEVICES=$GPU_PAIR $EVAL_BIN --model vllm \
        --model_args "pretrained=$CKPT_PATH,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
        --tasks mbpp \
        --num_fewshot 3 \
        --batch_size auto \
        --seed 42 \
        --gen_kwargs temperature=0 \
        --confirm_run_unsafe_code \
        --output_path "$OUTPUT_DIR/$OUTPUT_NAME" \
        2>&1 | tee "$LOG_DIR/${OUTPUT_NAME}.log"

    echo "[$i] Done $OUTPUT_NAME"
done

echo "MBPP worker finished"
