#!/bin/bash
# Run all missing MBPP + HumanEval evals sequentially
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_all_evals.sh <job_list_file>
# Each line in job_list_file: <checkpoint_path> <output_dir> <benchmark>

set -e
source /efs/rlvr-experiments/.venv/bin/activate

JOB_FILE="$1"
if [ -z "$JOB_FILE" ]; then
    echo "Usage: $0 <job_list_file>"
    exit 1
fi

TOTAL=$(wc -l < "$JOB_FILE")
CURRENT=0

while IFS=' ' read -r CKPT OUTDIR BENCH; do
    CURRENT=$((CURRENT + 1))
    if [ -d "$OUTDIR" ] && [ -f "$OUTDIR/summary.json" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (already done): $OUTDIR"
        continue
    fi
    echo "[$CURRENT/$TOTAL] Running: $BENCH on $(basename $CKPT) -> $OUTDIR"
    python -u /efs/rlvr-experiments/scripts/eval_pass_at_k.py "$BENCH" \
        --model-path "$CKPT" \
        --output-dir "$OUTDIR" \
        --n 1 --temperature 0 --max-model-len 4096 \
        ${EVAL_GPUS:+--gpus $EVAL_GPUS} \
        2>&1 | tail -5
    echo ""
done < "$JOB_FILE"

echo "ALL DONE"
