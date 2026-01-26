#!/bin/bash
# Run all pass@k evaluations with n=512 for Qwen3-1.7B-Base on TRAINING sets

set -e

# Activate virtual environment
cd /efs/rlvr-experiments
source .venv/bin/activate

OUTPUT_DIR="experiments/qwen3-1.7b-base-pass-rate"
MODEL_PATH="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
SPLIT="train"  # Use training sets

echo "=============================================="
echo "Starting all pass@k evaluations with n=512"
echo "Model: $MODEL_PATH"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# 1. GSM8k: 7473 prompts, n=1024, max_tokens=512
echo ""
echo "[1/4] GSM8k evaluation (7473 train prompts × 512 completions)"
echo "Start time: $(date)"
python3 scripts/eval_pass_rate.py \
    --dataset gsm8k \
    --split "$SPLIT" \
    --num-prompts 0 \
    --n 512 \
    --max-tokens 512 \
    --output-dir "$OUTPUT_DIR"
echo "GSM8k completed at: $(date)"

# 2. MATH: 7500 prompts, n=1024, max_tokens=1024
echo ""
echo "[2/4] MATH evaluation (7500 train prompts × 512 completions)"
echo "Start time: $(date)"
python3 scripts/eval_pass_rate.py \
    --dataset math \
    --split "$SPLIT" \
    --num-prompts 0 \
    --n 512 \
    --max-tokens 1024 \
    --max-model-len 2048 \
    --output-dir "$OUTPUT_DIR"
echo "MATH completed at: $(date)"

# 3. MBPP: 374 prompts, n=1024, max_tokens=512
echo ""
echo "[3/4] MBPP evaluation (374 train prompts × 512 completions)"
echo "Start time: $(date)"
python3 scripts/eval_pass_rate.py \
    --dataset mbpp \
    --split "$SPLIT" \
    --num-prompts 0 \
    --n 512 \
    --max-tokens 512 \
    --output-dir "$OUTPUT_DIR"
echo "MBPP completed at: $(date)"

# 4. APPS (if available)
echo ""
echo "[4/4] APPS evaluation (if implemented)"
echo "Start time: $(date)"
if python3 -c "from scripts.eval_pass_rate import APPSLoader" 2>/dev/null; then
    python3 scripts/eval_pass_rate.py \
        --dataset apps \
        --split "$SPLIT" \
        --num-prompts 0 \
        --n 512 \
        --max-tokens 1024 \
        --output-dir "$OUTPUT_DIR"
    echo "APPS completed at: $(date)"
else
    echo "APPS not yet implemented, skipping..."
fi

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
