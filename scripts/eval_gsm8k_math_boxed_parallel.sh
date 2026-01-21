#!/bin/bash
# Evaluate all checkpoints from gsm8k_math_boxed_curriculum run IN PARALLEL
# Uses all 8 GPUs simultaneously

set -e

CHECKPOINTS_DIR="/efs/rlvr-experiments/checkpoints"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"
EVAL_SCRIPT="/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Find all checkpoints from this run
CHECKPOINTS=$(ls -d ${CHECKPOINTS_DIR}/gsm8k_math_boxed_curriculum_step* 2>/dev/null | sort -V)

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Tasks to run (task:fewshot format) - skip minerva_math for now (has issues)
TASKS="gsm8k:8 gsm8k_cot:8 hendrycks_math:4 math_qwen:4"

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

# Run jobs in parallel batches of 8 (one per GPU)
NUM_GPUS=8
JOB_IDX=0

while [ $JOB_IDX -lt $TOTAL_JOBS ]; do
    echo ""
    echo "=== Starting batch at job $JOB_IDX ==="

    # Launch up to 8 jobs in parallel
    PIDS=()
    for GPU in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $JOB_IDX -ge $TOTAL_JOBS ]; then
            break
        fi

        JOB="${JOBS[$JOB_IDX]}"
        CKPT="${JOB%%|*}"
        REST="${JOB#*|}"
        CKPT_NAME="${REST%%|*}"
        REST="${REST#*|}"
        TASK="${REST%%|*}"
        FEWSHOT="${REST#*|}"
        OUTPUT_NAME="${CKPT_NAME}_${TASK}_${FEWSHOT}shot"

        echo "GPU $GPU: $OUTPUT_NAME"

        # Run in background
        $EVAL_SCRIPT "$CKPT" "$CKPT_NAME" "$TASK" "$FEWSHOT" "$GPU" > "$LOG_DIR/${OUTPUT_NAME}.log" 2>&1 &
        PIDS+=($!)

        JOB_IDX=$((JOB_IDX + 1))
    done

    # Wait for all jobs in this batch to complete
    echo "Waiting for ${#PIDS[@]} jobs to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    echo "Batch complete!"
done

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $OUTPUT_DIR"
