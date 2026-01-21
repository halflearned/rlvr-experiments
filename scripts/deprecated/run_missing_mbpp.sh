#!/bin/bash
# Run missing MBPP evals
# Usage: ./run_missing_mbpp.sh <gpu> <start_idx> <end_idx>

set -e

GPU=${1:-"0"}
START_IDX=${2:-0}
END_IDX=${3:-999}

export HF_ALLOW_CODE_EVAL=1

EVAL_BIN="/efs/rlvr-experiments/.venv/bin/lm_eval"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"

# Missing MBPP evals: (checkpoint_path, output_name)
EVALS=(
    # annotations-adhoc-20260117-064406 (mixed_lr1e6_random_s1)
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step400 mixed_lr1e6_random_s1_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step500 mixed_lr1e6_random_s1_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step600 mixed_lr1e6_random_s1_step600"
    # annotations-adhoc-20260117-064436 (mixed_lr5e6_random_s1)
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step100 mixed_lr5e6_random_s1_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step200 mixed_lr5e6_random_s1_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step300 mixed_lr5e6_random_s1_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step400 mixed_lr5e6_random_s1_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step500 mixed_lr5e6_random_s1_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step600 mixed_lr5e6_random_s1_step600"
    # annotations-adhoc-20260117-061547 (mixed_lr5e6_random)
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step100 mixed_lr5e6_random_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step200 mixed_lr5e6_random_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step300 mixed_lr5e6_random_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step400 mixed_lr5e6_random_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step500 mixed_lr5e6_random_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step600 mixed_lr5e6_random_step600"
    # annotations-adhoc-20260116-021228 (qwen3_1_7b_mixed)
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step100 qwen3_1_7b_mixed_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step200 qwen3_1_7b_mixed_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step300 qwen3_1_7b_mixed_step300"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step400 qwen3_1_7b_mixed_step400"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step500 qwen3_1_7b_mixed_step500"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step600 qwen3_1_7b_mixed_step600"
    # annotations-adhoc-20260116-021345 (qwen3_1_7b_sequential)
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step100 qwen3_1_7b_sequential_step100"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step200 qwen3_1_7b_sequential_step200"
    "/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step300 qwen3_1_7b_sequential_step300"
)

TOTAL=${#EVALS[@]}
echo "MBPP worker on GPU $GPU, processing evals $START_IDX to $END_IDX (total: $TOTAL)"

for i in $(seq $START_IDX $END_IDX); do
    if [ $i -ge $TOTAL ]; then
        echo "No more evals"
        break
    fi

    eval_info="${EVALS[$i]}"
    CKPT_PATH=$(echo "$eval_info" | awk '{print $1}')
    CKPT_NAME=$(echo "$eval_info" | awk '{print $2}')
    OUTPUT_NAME="${CKPT_NAME}_mbpp_3shot"

    # Skip if already done
    if [ -d "$OUTPUT_DIR/$OUTPUT_NAME" ] && find "$OUTPUT_DIR/$OUTPUT_NAME" -name "results*.json" 2>/dev/null | grep -q .; then
        echo "[$i] SKIP $OUTPUT_NAME (already done)"
        continue
    fi

    echo "[$i] Running $OUTPUT_NAME on GPU $GPU"

    CUDA_VISIBLE_DEVICES=$GPU $EVAL_BIN --model vllm \
        --model_args "pretrained=$CKPT_PATH,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
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
