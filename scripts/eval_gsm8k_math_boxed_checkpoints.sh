#!/bin/bash
# Evaluate all checkpoints from gsm8k_math_boxed_curriculum run
# Tasks: gsm8k (8-shot), gsm8k_cot (8-shot), hendrycks_math (4-shot), minerva_math (4-shot), math_qwen (4-shot)

set -e

CHECKPOINTS_DIR="/efs/rlvr-experiments/checkpoints"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"
EVAL_SCRIPT="/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Find all checkpoints from this run
CHECKPOINTS=$(ls -d ${CHECKPOINTS_DIR}/gsm8k_math_boxed_curriculum_step* 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found matching gsm8k_math_boxed_curriculum_step*"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Tasks to run (task:fewshot format)
TASKS="gsm8k:8 gsm8k_cot:8 hendrycks_math:4 minerva_math:4 math_qwen:4"

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Available GPUs: $NUM_GPUS"

# Build job list
declare -a JOBS
for CKPT in $CHECKPOINTS; do
    CKPT_NAME=$(basename "$CKPT")
    for TASK_SPEC in $TASKS; do
        TASK="${TASK_SPEC%:*}"
        FEWSHOT="${TASK_SPEC#*:}"
        OUTPUT_NAME="${CKPT_NAME}_${TASK}_${FEWSHOT}shot"

        # Skip if already completed
        if [ "$TASK" = "math_qwen" ]; then
            if [ -f "$OUTPUT_DIR/${OUTPUT_NAME}.json" ]; then
                echo "SKIP: $OUTPUT_NAME (completed)"
                continue
            fi
        else
            if [ -d "$OUTPUT_DIR/$OUTPUT_NAME" ] && find "$OUTPUT_DIR/$OUTPUT_NAME" -name "results*.json" 2>/dev/null | grep -q .; then
                echo "SKIP: $OUTPUT_NAME (completed)"
                continue
            fi
        fi

        JOBS+=("$CKPT|$CKPT_NAME|$TASK|$FEWSHOT")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo ""
echo "Total jobs to run: $TOTAL_JOBS"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "All evaluations already completed!"
    exit 0
fi

# Run jobs, cycling through GPUs
JOB_IDX=0
for JOB in "${JOBS[@]}"; do
    CKPT="${JOB%%|*}"
    REST="${JOB#*|}"
    CKPT_NAME="${REST%%|*}"
    REST="${REST#*|}"
    TASK="${REST%%|*}"
    FEWSHOT="${REST#*|}"

    GPU=$((JOB_IDX % NUM_GPUS))
    OUTPUT_NAME="${CKPT_NAME}_${TASK}_${FEWSHOT}shot"

    echo ""
    echo "=== Job $((JOB_IDX+1))/$TOTAL_JOBS: $OUTPUT_NAME on GPU $GPU ==="

    # Run eval (blocking - one at a time for simplicity)
    $EVAL_SCRIPT "$CKPT" "$CKPT_NAME" "$TASK" "$FEWSHOT" "$GPU" 2>&1 | tee "$LOG_DIR/${OUTPUT_NAME}.log"

    JOB_IDX=$((JOB_IDX + 1))
done

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $OUTPUT_DIR"
