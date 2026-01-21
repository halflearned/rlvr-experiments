#!/bin/bash
# Batch evaluation runner - runs multiple checkpoints through all standard evals
#
# Usage:
#   ./run_batch_evals.sh <jobs_file> [gpu] [start_idx] [end_idx]
#
# Jobs file format (JSON array):
# [
#   {"checkpoint_path": "/path/to/ckpt", "output_name": "my_ckpt_step100"},
#   ...
# ]
#
# Each checkpoint will be evaluated on all 5 standard tasks:
#   - gsm8k 4-shot (uses `Question: X\nAnswer:` prompt, strict expects `#### N`)
#   - gsm8k_cot 4-shot (uses `Q: X\nA:` prompt, strict expects `The answer is N.`)
#   - math_qwen 4-shot (Qwen-style CoT, NOT hendrycks_math)
#   - ifeval 0-shot
#   - mbpp 3-shot
#
# NOTE: Our training uses gsm8k_cot format (`Q: X\nA:` + `The answer is X.`)
#       so gsm8k_cot-strict is the PRIMARY metric for GSM8K.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOBS_FILE=$1
GPU=${2:-0}
START_IDX=${3:-0}
END_IDX=${4:-999999}

if [ -z "$JOBS_FILE" ]; then
    echo "Usage: $0 <jobs_file.json> [gpu] [start_idx] [end_idx]"
    echo ""
    echo "Jobs file format:"
    echo '[{"checkpoint_path": "/path/to/ckpt", "output_name": "name"}, ...]'
    exit 1
fi

if [ ! -f "$JOBS_FILE" ]; then
    echo "ERROR: Jobs file not found: $JOBS_FILE"
    exit 1
fi

# Standard evaluation tasks (6 tasks total)
# NOTE: gsm8k uses `Question: X\nAnswer:` prompt, strict expects `#### N`
# NOTE: gsm8k_cot uses `Q: X\nA:` prompt, strict expects `The answer is N.`
# NOTE: minerva_math uses \boxed{} extraction - key metric for MATH benchmark
# NOTE: math_qwen uses Qwen-style CoT prompting (for comparison)
TASKS=(
    "gsm8k:4"
    "gsm8k_cot:4"
    "minerva_math:4"
    "ifeval:0"
    "mbpp:3"
)

# Read jobs
JOBS=$(cat "$JOBS_FILE")
TOTAL=$(echo "$JOBS" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")

echo "Batch eval worker on GPU $GPU"
echo "Processing checkpoints $START_IDX to $END_IDX (total available: $TOTAL)"
echo "Tasks per checkpoint: ${#TASKS[@]}"

for i in $(seq $START_IDX $END_IDX); do
    if [ $i -ge $TOTAL ]; then
        echo "No more checkpoints"
        break
    fi

    # Extract job info
    JOB=$(echo "$JOBS" | python3 -c "import json,sys; j=json.load(sys.stdin)[$i]; print(json.dumps(j))")
    CKPT_PATH=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['checkpoint_path'])")
    OUTPUT_NAME=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_name'])")

    echo ""
    echo "[$i] Checkpoint: $OUTPUT_NAME"

    # Run each task
    for task_spec in "${TASKS[@]}"; do
        TASK="${task_spec%%:*}"
        FEWSHOT="${task_spec##*:}"

        "$SCRIPT_DIR/run_lm_eval.sh" "$CKPT_PATH" "$OUTPUT_NAME" "$TASK" "$FEWSHOT" "$GPU"
    done
done

echo ""
echo "Batch worker finished"
