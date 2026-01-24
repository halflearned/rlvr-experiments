#!/bin/bash
# Launch IF_multi_constraints_upto5 pass@k evaluation across 8 GPUs
# Each GPU processes ~12-13 samples (100 total / 8 GPUs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/efs/rlvr-experiments/.venv/bin/activate"
SCRIPT="/efs/rlvr-experiments/scripts/adhoc/if_multi_constraints_passk_eval.py"
OUTPUT_DIR="/efs/rlvr-experiments/experiments/if_multi_constraints_passk"
LOG_DIR="/tmp/if_multi_constraints_eval"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Starting IF_multi_constraints evaluation on 8 GPUs..."
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"

source "$VENV"

# 100 samples / 8 GPUs = 12.5 samples per GPU
# GPU 0-3: 13 samples each (0-13, 13-26, 26-39, 39-52)
# GPU 4-7: 12 samples each (52-64, 64-76, 76-88, 88-100)

python "$SCRIPT" --gpu 0 --start-idx 0  --end-idx 13  2>&1 | tee "$LOG_DIR/gpu0.log" &
python "$SCRIPT" --gpu 1 --start-idx 13 --end-idx 26  2>&1 | tee "$LOG_DIR/gpu1.log" &
python "$SCRIPT" --gpu 2 --start-idx 26 --end-idx 39  2>&1 | tee "$LOG_DIR/gpu2.log" &
python "$SCRIPT" --gpu 3 --start-idx 39 --end-idx 52  2>&1 | tee "$LOG_DIR/gpu3.log" &
python "$SCRIPT" --gpu 4 --start-idx 52 --end-idx 64  2>&1 | tee "$LOG_DIR/gpu4.log" &
python "$SCRIPT" --gpu 5 --start-idx 64 --end-idx 76  2>&1 | tee "$LOG_DIR/gpu5.log" &
python "$SCRIPT" --gpu 6 --start-idx 76 --end-idx 88  2>&1 | tee "$LOG_DIR/gpu6.log" &
python "$SCRIPT" --gpu 7 --start-idx 88 --end-idx 100 2>&1 | tee "$LOG_DIR/gpu7.log" &

echo "Waiting for all workers to complete..."
wait

echo ""
echo "All workers completed! Computing pass@k..."
python /efs/rlvr-experiments/scripts/adhoc/compute_if_multi_constraints_passk.py --results-dir "$OUTPUT_DIR"
