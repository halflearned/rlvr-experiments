#!/bin/bash
# Launch batched IFEval generation across 8 GPUs
#
# Usage:
#   ./scripts/ifeval_batched_launcher.sh [seed_start] [num_seeds] [n_per_batch] [num_prompts]
#
# Examples:
#   # Run seeds 0-7 (one per GPU), 16 completions each, 50 prompts
#   ./scripts/ifeval_batched_launcher.sh 0 8 16 50
#
#   # Run seeds 8-15
#   ./scripts/ifeval_batched_launcher.sh 8 8 16 50
#
# To get 512 completions per prompt with n=16:
#   Run 4 batches: seeds 0-7, 8-15, 16-23, 24-31 (32 seeds total)

set -e

SEED_START=${1:-0}
NUM_SEEDS=${2:-8}
N_PER_BATCH=${3:-16}
NUM_PROMPTS=${4:-50}

OUTPUT_DIR="experiments/ifeval-curriculum"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "IFEval Batched Generation Launcher"
echo "=============================================="
echo "Seeds: ${SEED_START} to $((SEED_START + NUM_SEEDS - 1))"
echo "Completions per batch: ${N_PER_BATCH}"
echo "Number of prompts: ${NUM_PROMPTS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Launch jobs
PIDS=()
for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((SEED_START + i))
    GPU=$((i % 8))

    LOG_FILE="${LOG_DIR}/seed${SEED}_${TIMESTAMP}.log"

    echo "Launching seed ${SEED} on GPU ${GPU}..."

    .venv/bin/python scripts/ifeval_batched_gen.py \
        --seed "$SEED" \
        --n "$N_PER_BATCH" \
        --num-prompts "$NUM_PROMPTS" \
        --gpu "$GPU" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs. PIDs: ${PIDS[*]}"
echo "Logs in: ${LOG_DIR}"
echo ""
echo "Waiting for completion..."

# Wait for all jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    SEED=$((SEED_START + i))

    if wait "$PID"; then
        echo "  Seed ${SEED}: SUCCESS"
    else
        echo "  Seed ${SEED}: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "WARNING: ${FAILED} jobs failed. Check logs in ${LOG_DIR}"
fi

# Show file sizes
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}"/seed*.jsonl 2>/dev/null || echo "No output files found"
