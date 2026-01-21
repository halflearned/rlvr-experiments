#!/bin/bash
# =============================================================================
# Universal lm_eval runner for all checkpoint evaluations
# =============================================================================
#
# CRITICAL SETTINGS (DO NOT CHANGE):
#   - tensor_parallel_size=1  (TP>1 causes non-determinism and wrong results)
#   - seed=42                 (in model_args AND --seed flag)
#   - temperature=0           (greedy decoding)
#   - max_model_len=4096      (prevent OOM)
#
# IMPORTANT: DO NOT CREATE NEW SCRIPTS FOR CORNER CASES.
# If something doesn't work, FIX THIS SCRIPT instead.
# This is the ONE script for running lm_eval. Keep it that way.
#
# Usage:
#   ./run_lm_eval.sh <checkpoint_path> <output_name> <task> <num_fewshot> [gpu]
#
# Examples:
#   ./run_lm_eval.sh /path/to/checkpoint my_checkpoint_step100 gsm8k 8 0
#   ./run_lm_eval.sh /path/to/checkpoint my_checkpoint_step100 mbpp 3 1
#   ./run_lm_eval.sh /path/to/checkpoint my_checkpoint_step100 ifeval 0 2
#
# Tasks supported:
#   - gsm8k (4-shot) - uses `Question: X\nAnswer:` prompt, strict expects `#### N`
#   - gsm8k_cot (4-shot) - uses `Q: X\nA:` prompt, strict expects `The answer is N.`
#                         OUR TRAINING USES THIS FORMAT - use gsm8k_cot-strict as primary metric
#   - hendrycks_math (4-shot) - lm_eval version (~17% on base model) - DON'T USE
#   - math_qwen (4-shot) - Qwen-style CoT prompting (~43% on base model)
#   - ifeval (0-shot)
#   - mbpp (3-shot) - requires code execution
#   - humaneval (0-shot) - requires code execution
#
# NOTE: math_qwen uses scripts/adhoc/eval_math_qwen_style.py (not lm_eval)
#       See scripts/adhoc/NOTES_math_eval.md for why this matters.
#
# Output: Results saved to $OUTPUT_DIR/<output_name>_<task>_<fewshot>shot/

set -e

# Configuration
EVAL_BIN="/efs/rlvr-experiments/.venv/bin/lm_eval"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
LOG_DIR="/efs/rlvr-experiments/eval_logs"

# Required for MBPP/HumanEval code execution
export HF_ALLOW_CODE_EVAL=1

# Parse arguments
CKPT_PATH=$1
OUTPUT_NAME=$2
TASK=$3
FEWSHOT=$4
GPU=${5:-0}

if [ -z "$CKPT_PATH" ] || [ -z "$OUTPUT_NAME" ] || [ -z "$TASK" ] || [ -z "$FEWSHOT" ]; then
    echo "Usage: $0 <checkpoint_path> <output_name> <task> <num_fewshot> [gpu]"
    echo ""
    echo "Tasks: gsm8k, hendrycks_math, math_qwen, ifeval, mbpp, humaneval"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/checkpoint mixed_lr1e6_step100 gsm8k 8 0"
    echo "  $0 /path/to/checkpoint mixed_lr1e6_step100 mbpp 3 1"
    exit 1
fi

# Validate checkpoint exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CKPT_PATH"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Build output name with task suffix
FULL_OUTPUT_NAME="${OUTPUT_NAME}_${TASK}_${FEWSHOT}shot"

# Skip if already completed
# - For lm_eval tasks: check for results*.json in output directory
# - For math_qwen: check for output JSON file directly
if [ "$TASK" = "math_qwen" ]; then
    if [ -f "$OUTPUT_DIR/${FULL_OUTPUT_NAME}.json" ]; then
        echo "SKIP: $FULL_OUTPUT_NAME (already completed)"
        exit 0
    fi
else
    if [ -d "$OUTPUT_DIR/$FULL_OUTPUT_NAME" ] && find "$OUTPUT_DIR/$FULL_OUTPUT_NAME" -name "results*.json" 2>/dev/null | grep -q .; then
        echo "SKIP: $FULL_OUTPUT_NAME (already completed)"
        exit 0
    fi
fi

# Fix safetensors filename if needed (some checkpoints have wrong naming)
# Use flock to prevent race conditions when multiple evals run in parallel
LOCK_FILE="$CKPT_PATH/.safetensors_rename.lock"
(
    flock -x 200
    if [ -f "$CKPT_PATH/model.safetensors" ] && [ ! -f "$CKPT_PATH/model-00001-of-00001.safetensors" ]; then
        echo "Fixing: Renaming model.safetensors -> model-00001-of-00001.safetensors"
        mv "$CKPT_PATH/model.safetensors" "$CKPT_PATH/model-00001-of-00001.safetensors"
    fi
) 200>"$LOCK_FILE"

# Build extra args for code execution tasks
EXTRA_ARGS=""
if [ "$TASK" = "mbpp" ] || [ "$TASK" = "humaneval" ]; then
    EXTRA_ARGS="--confirm_run_unsafe_code"
fi

echo "=========================================="
echo "Running evaluation"
echo "  Checkpoint: $CKPT_PATH"
echo "  Output: $FULL_OUTPUT_NAME"
echo "  Task: $TASK ($FEWSHOT-shot)"
echo "  GPU: $GPU"
echo "=========================================="

source /efs/rlvr-experiments/.venv/bin/activate

# Handle math_qwen separately (uses custom script, not lm_eval)
if [ "$TASK" = "math_qwen" ]; then
    MATH_SCRIPT="/efs/rlvr-experiments/scripts/adhoc/eval_math_qwen_style.py"
    OUTPUT_FILE="${OUTPUT_DIR}/${FULL_OUTPUT_NAME}.json"

    CUDA_VISIBLE_DEVICES=$GPU python "$MATH_SCRIPT" \
        --model "$CKPT_PATH" \
        --num-shots "$FEWSHOT" \
        --tensor-parallel-size 1 \
        --output "$OUTPUT_FILE" \
        2>&1 | tee "$LOG_DIR/${FULL_OUTPUT_NAME}.log"
else
    # Standard lm_eval tasks
    CUDA_VISIBLE_DEVICES=$GPU $EVAL_BIN --model vllm \
        --model_args "pretrained=$CKPT_PATH,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
        --tasks "$TASK" \
        --num_fewshot "$FEWSHOT" \
        --batch_size auto \
        --seed 42 \
        --gen_kwargs temperature=0 \
        --log_samples \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR/$FULL_OUTPUT_NAME" \
        2>&1 | tee "$LOG_DIR/${FULL_OUTPUT_NAME}.log"
fi

echo "Done: $FULL_OUTPUT_NAME"
