#!/bin/bash
# Evaluate all checkpoints for a given run on a given benchmark.
# Usage: ./scripts/eval_all_checkpoints.sh <run_name> <benchmark> <gpu>
#
# Example:
#   ./scripts/eval_all_checkpoints.sh qwen3-1.7B-gsm8k-dapo-lr5e6_20260129-213056 gsm8k 0
#
# Outputs summary.jsonl in results/<run_name>/evals/<benchmark>/summary.jsonl

set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name> <benchmark> <gpu>}"
BENCHMARK="${2:?Usage: $0 <run_name> <benchmark> <gpu>}"
GPU="${3:?Usage: $0 <run_name> <benchmark> <gpu>}"

RESULTS_DIR="/efs/rlvr-experiments/results/${RUN_NAME}"
CHECKPOINTS_DIR="${RESULTS_DIR}/checkpoints"
EVALS_DIR="${RESULTS_DIR}/evals/${BENCHMARK}"
SUMMARY_JSONL="${EVALS_DIR}/summary.jsonl"

mkdir -p "${EVALS_DIR}"

# Get sorted list of step numbers
STEPS=$(ls "${CHECKPOINTS_DIR}" | grep -oP 'step\K\d+' | sort -n)

echo "[eval_all] Run: ${RUN_NAME}"
echo "[eval_all] Benchmark: ${BENCHMARK}"
echo "[eval_all] GPU: ${GPU}"
echo "[eval_all] Steps: ${STEPS}"
echo "[eval_all] Output: ${SUMMARY_JSONL}"

for STEP in ${STEPS}; do
    CHECKPOINT="${CHECKPOINTS_DIR}/${RUN_NAME}_step${STEP}"
    OUTPUT_DIR="${EVALS_DIR}/step${STEP}"

    # Skip if already evaluated (check if summary.json exists)
    if [ -f "${OUTPUT_DIR}/summary.json" ]; then
        echo "[eval_all] step${STEP}: already evaluated, skipping"
        continue
    fi

    echo "[eval_all] step${STEP}: evaluating..."
    /efs/rlvr-experiments/.venv/bin/python /efs/rlvr-experiments/scripts/eval_checkpoint.py \
        "${CHECKPOINT}" "${OUTPUT_DIR}" \
        --benchmark "${BENCHMARK}" --gpu "${GPU}"

    # Extract key fields and append to summary.jsonl
    if [ -f "${OUTPUT_DIR}/summary.json" ]; then
        if [ "${BENCHMARK}" = "gsm8k" ] || [ "${BENCHMARK}" = "math" ]; then
            /efs/rlvr-experiments/.venv/bin/python3 -c "
import json
with open('${OUTPUT_DIR}/summary.json') as f:
    d = json.load(f)
out = {'step': ${STEP}, 'accuracy': round(d['accuracy'], 4), 'n_correct': d['n_correct'], 'n_examples': d['n_examples']}
if 'level_accuracy' in d:
    out['level_accuracy'] = d['level_accuracy']
print(json.dumps(out))
" >> "${SUMMARY_JSONL}"
        elif [ "${BENCHMARK}" = "ifeval" ] || [ "${BENCHMARK}" = "ifbench" ]; then
            /efs/rlvr-experiments/.venv/bin/python3 -c "
import json
with open('${OUTPUT_DIR}/summary.json') as f:
    d = json.load(f)
out = {'step': ${STEP}, 'prompt_level_strict_acc': round(d['prompt_level_strict_acc'], 4), 'inst_level_acc': round(d['inst_level_acc'], 4), 'prompt_pass': d['prompt_pass'], 'n_prompts': d['n_prompts']}
print(json.dumps(out))
" >> "${SUMMARY_JSONL}"
        elif [ "${BENCHMARK}" = "aime" ] || [ "${BENCHMARK}" = "beyondaime" ]; then
            /efs/rlvr-experiments/.venv/bin/python3 -c "
import json
with open('${OUTPUT_DIR}/summary.json') as f:
    d = json.load(f)
out = {'step': ${STEP}, 'accuracy': round(d['accuracy'], 4), 'n_correct': d['n_correct'], 'n_examples': d['n_examples']}
print(json.dumps(out))
" >> "${SUMMARY_JSONL}"
        fi
        echo "[eval_all] step${STEP}: done ($(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/summary.json')).get('accuracy', 'N/A'))"))"
    else
        echo "[eval_all] step${STEP}: WARNING - no summary.json produced"
    fi
done

echo "[eval_all] All done! Results in ${SUMMARY_JSONL}"
