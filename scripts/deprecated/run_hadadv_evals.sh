#!/bin/bash
# Run evals for hadadv-* jobs using job name as output prefix
# Usage: ./run_hadadv_evals.sh <gpu_pair> <start_idx> <end_idx>

set -e

GPU=${1:-"0"}
START_IDX=${2:-0}
END_IDX=${3:-999}

export HF_ALLOW_CODE_EVAL=1

EVAL_BIN="/efs/rlvr-experiments/.venv/bin/lm_eval"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"

# hadadv-adhoc-20260116-011803: lr=5e-6, Mixed, staleness=0
# hadadv-adhoc-20260116-011826: lr=5e-6, Sequential, staleness=0

# Build eval list: (job_dir, checkpoint_name, task, fewshot)
EVALS=(
    # hadadv-adhoc-20260116-011803 Mixed lr=5e-6 (6 checkpoints × 5 tasks = 30 evals)
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step100 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step100 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step100 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step100 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step100 mbpp 3"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step200 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step200 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step200 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step200 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step200 mbpp 3"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step300 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step300 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step300 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step300 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step300 mbpp 3"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step400 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step400 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step400 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step400 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step400 mbpp 3"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step500 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step500 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step500 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step500 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step500 mbpp 3"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step600 gsm8k 8"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step600 gsm8k 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step600 hendrycks_math 4"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step600 ifeval 0"
    "hadadv-adhoc-20260116-011803 qwen3_1_7b_mixed_step600 mbpp 3"
    # hadadv-adhoc-20260116-011826 Sequential lr=5e-6 (3 checkpoints × 5 tasks = 15 evals)
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step100 gsm8k 8"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step100 gsm8k 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step100 hendrycks_math 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step100 ifeval 0"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step100 mbpp 3"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step200 gsm8k 8"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step200 gsm8k 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step200 hendrycks_math 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step200 ifeval 0"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step200 mbpp 3"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step300 gsm8k 8"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step300 gsm8k 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step300 hendrycks_math 4"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step300 ifeval 0"
    "hadadv-adhoc-20260116-011826 qwen3_1_7b_sequential_step300 mbpp 3"
)

TOTAL=${#EVALS[@]}
echo "Hadadv eval worker on GPU $GPU, processing evals $START_IDX to $END_IDX (total: $TOTAL)"

for i in $(seq $START_IDX $END_IDX); do
    if [ $i -ge $TOTAL ]; then
        echo "No more evals"
        break
    fi

    eval_info="${EVALS[$i]}"
    JOB=$(echo "$eval_info" | awk '{print $1}')
    CKPT=$(echo "$eval_info" | awk '{print $2}')
    TASK=$(echo "$eval_info" | awk '{print $3}')
    FEWSHOT=$(echo "$eval_info" | awk '{print $4}')

    CKPT_PATH="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/${JOB}/${CKPT}"
    OUTPUT_NAME="${JOB}_${CKPT}_${TASK}_${FEWSHOT}shot"

    # Skip if already done
    if [ -d "$OUTPUT_DIR/$OUTPUT_NAME" ] && find "$OUTPUT_DIR/$OUTPUT_NAME" -name "results*.json" 2>/dev/null | grep -q .; then
        echo "[$i] SKIP $OUTPUT_NAME (already done)"
        continue
    fi

    echo "[$i] Running $OUTPUT_NAME on GPU $GPU"

    # Add --confirm_run_unsafe_code for mbpp
    EXTRA_ARGS=""
    if [ "$TASK" = "mbpp" ]; then
        EXTRA_ARGS="--confirm_run_unsafe_code"
    fi

    CUDA_VISIBLE_DEVICES=$GPU $EVAL_BIN --model vllm \
        --model_args "pretrained=$CKPT_PATH,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
        --tasks "$TASK" \
        --num_fewshot "$FEWSHOT" \
        --batch_size auto \
        --seed 42 \
        --gen_kwargs temperature=0 \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR/$OUTPUT_NAME" \
        2>&1 | tee "$LOG_DIR/${OUTPUT_NAME}.log"

    echo "[$i] Done $OUTPUT_NAME"
done

echo "Hadadv worker finished"
