#!/bin/bash
# Run all benchmark evaluations

set -e
source /efs/rlvr-experiments/.venv/bin/activate

RESULTS_DIR="/efs/rlvr-experiments/eval_results_sagemaker"
mkdir -p "$RESULTS_DIR"

BASE_MODEL="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
CHECKPOINTS_BASE="/efs/rlvr-experiments/checkpoints/sagemaker_downloads"

# Define checkpoints to evaluate
declare -A CHECKPOINTS
CHECKPOINTS["base"]="$BASE_MODEL"
CHECKPOINTS["mixed_lr5e6_step600"]="$CHECKPOINTS_BASE/hadadv-adhoc-20260116-011803/qwen3_1_7b_mixed_step600"
CHECKPOINTS["mixed_lr5e6_final"]="$CHECKPOINTS_BASE/hadadv-adhoc-20260116-011803/qwen3_1_7b_mixed_final"
CHECKPOINTS["seq_lr5e6_step100"]="$CHECKPOINTS_BASE/hadadv-adhoc-20260116-011826/qwen3_1_7b_sequential_step100"
CHECKPOINTS["seq_lr5e6_step300"]="$CHECKPOINTS_BASE/hadadv-adhoc-20260116-011826/qwen3_1_7b_sequential_step300"
CHECKPOINTS["mixed_lr1e6_step100"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step100"
CHECKPOINTS["mixed_lr1e6_step200"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step200"
CHECKPOINTS["mixed_lr1e6_step300"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step300"
CHECKPOINTS["mixed_lr1e5_step100"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021258/qwen3_1_7b_mixed_step100"
CHECKPOINTS["seq_lr1e6_step100"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step100"
CHECKPOINTS["seq_lr1e6_step200"]="$CHECKPOINTS_BASE/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step200"
CHECKPOINTS["mixed_lr5e6_random_step200"]="$CHECKPOINTS_BASE/annotations-adhoc-20260117-061547/mixed_lr5e6_random_step200"
CHECKPOINTS["mixed_lr1e6_random_s1_step200"]="$CHECKPOINTS_BASE/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step200"
CHECKPOINTS["mixed_lr1e6_random_s1_step250"]="$CHECKPOINTS_BASE/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step250"
CHECKPOINTS["mixed_lr5e6_random_s1_step150"]="$CHECKPOINTS_BASE/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step150"

run_eval() {
    local name=$1
    local model_path=$2
    local task=$3
    local num_fewshot=$4
    local output_name="${name}_${task}_${num_fewshot}shot"

    if [ -f "$RESULTS_DIR/${output_name}/results.json" ]; then
        echo "Already exists: $output_name"
        return
    fi

    echo "Running: $output_name"
    lm_eval --model vllm \
        --model_args "pretrained=$model_path,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=4096" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        --seed 42 \
        --gen_kwargs temperature=0 \
        --output_path "$RESULTS_DIR/$output_name" \
        2>&1 | tee "$RESULTS_DIR/${output_name}.log"
}

# Run evaluations
for name in "${!CHECKPOINTS[@]}"; do
    model_path="${CHECKPOINTS[$name]}"
    if [ -d "$model_path" ]; then
        echo "Evaluating: $name ($model_path)"

        # GSM8K 8-shot
        run_eval "$name" "$model_path" "gsm8k" 8

        # hendrycks_math 4-shot
        run_eval "$name" "$model_path" "hendrycks_math" 4

        # IFEval 0-shot
        run_eval "$name" "$model_path" "ifeval" 0

        # MBPP 3-shot (requires --confirm_run_unsafe_code)
        if [ ! -f "$RESULTS_DIR/${name}_mbpp_3shot/results.json" ]; then
            echo "Running: ${name}_mbpp_3shot"
            lm_eval --model vllm \
                --model_args "pretrained=$model_path,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=4096" \
                --tasks mbpp \
                --num_fewshot 3 \
                --batch_size auto \
                --seed 42 \
                --gen_kwargs temperature=0 \
                --confirm_run_unsafe_code \
                --output_path "$RESULTS_DIR/${name}_mbpp_3shot" \
                2>&1 | tee "$RESULTS_DIR/${name}_mbpp_3shot.log"
        fi
    else
        echo "Skipping $name: path not found"
    fi
done

echo "All evaluations complete!"
