#!/bin/bash
# Run all benchmark evaluations on step300/step600/final checkpoints

set -e
source /efs/rlvr-experiments/.venv/bin/activate

RESULTS_DIR="/efs/rlvr-experiments/eval_results_sagemaker"
mkdir -p "$RESULTS_DIR"

# Define checkpoints to evaluate (step300/step600/final only)
declare -A CHECKPOINTS
CHECKPOINTS["base"]="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
CHECKPOINTS["mixed_lr5e6_step600"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/hadadv-adhoc-20260116-011803/qwen3_1_7b_mixed_step600"
CHECKPOINTS["mixed_lr5e6_final"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/hadadv-adhoc-20260116-011803/qwen3_1_7b_mixed_final"
CHECKPOINTS["seq_lr5e6_step300"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/hadadv-adhoc-20260116-011826/qwen3_1_7b_sequential_step300"
CHECKPOINTS["mixed_lr1e6_step300"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021228/qwen3_1_7b_mixed_step300"
CHECKPOINTS["seq_lr1e6_step300"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260116-021345/qwen3_1_7b_sequential_step300"
CHECKPOINTS["mixed_lr1e6_random_s1_step600"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064406/mixed_lr1e6_random_s1_step600"
CHECKPOINTS["mixed_lr5e6_random_s1_final"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_final"

run_eval() {
    local name=$1
    local model_path=$2
    local task=$3
    local num_fewshot=$4
    local extra_args=$5
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
        $extra_args \
        --output_path "$RESULTS_DIR/$output_name" \
        2>&1 | tee "$RESULTS_DIR/${output_name}.log"
}

# Run evaluations on each checkpoint
for name in "${!CHECKPOINTS[@]}"; do
    model_path="${CHECKPOINTS[$name]}"
    if [ -d "$model_path" ]; then
        echo ""
        echo "========================================="
        echo "Evaluating: $name"
        echo "Path: $model_path"
        echo "========================================="

        # GSM8K 8-shot
        run_eval "$name" "$model_path" "gsm8k" 8 ""

        # hendrycks_math 4-shot
        run_eval "$name" "$model_path" "hendrycks_math" 4 ""

        # IFEval 0-shot
        run_eval "$name" "$model_path" "ifeval" 0 ""

        # MBPP 3-shot
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

echo ""
echo "All evaluations complete!"
