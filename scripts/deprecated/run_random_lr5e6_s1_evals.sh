#!/bin/bash
# Run evaluations on Random_lr5e6_s1 checkpoints (missing from all_results.md)

set -e
source /efs/rlvr-experiments/.venv/bin/activate

RESULTS_DIR="/efs/rlvr-experiments/eval_results_sagemaker"
mkdir -p "$RESULTS_DIR"

# Define checkpoints to evaluate
declare -A CHECKPOINTS
CHECKPOINTS["random_lr5e6_s1_step300"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step300"
CHECKPOINTS["random_lr5e6_s1_step600"]="/efs/rlvr-experiments/checkpoints/sagemaker_downloads/annotations-adhoc-20260117-064436/mixed_lr5e6_random_s1_step600"

run_eval() {
    local name=$1
    local model_path=$2
    local task=$3
    local num_fewshot=$4
    local output_name="${name}_${task}_${num_fewshot}shot"

    if [ -f "$RESULTS_DIR/${output_name}/results.json" ] || find "$RESULTS_DIR/${output_name}" -name "*.json" 2>/dev/null | grep -q .; then
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

# Run evaluations on each checkpoint
for name in "${!CHECKPOINTS[@]}"; do
    model_path="${CHECKPOINTS[$name]}"
    if [ -d "$model_path" ]; then
        echo ""
        echo "========================================="
        echo "Evaluating: $name"
        echo "Path: $model_path"
        echo "========================================="

        # GSM8K 4-shot
        run_eval "$name" "$model_path" "gsm8k" 4 ""

        # GSM8K 8-shot
        run_eval "$name" "$model_path" "gsm8k" 8 ""

        # hendrycks_math 4-shot
        run_eval "$name" "$model_path" "hendrycks_math" 4 ""

        # IFEval 0-shot
        run_eval "$name" "$model_path" "ifeval" 0 ""

        # MBPP 3-shot (requires --confirm_run_unsafe_code)
        local mbpp_output_name="${name}_mbpp_3shot"
        if [ ! -f "$RESULTS_DIR/${mbpp_output_name}/results.json" ] && ! find "$RESULTS_DIR/${mbpp_output_name}" -name "*.json" 2>/dev/null | grep -q .; then
            echo "Running: ${mbpp_output_name}"
            HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm \
                --model_args "pretrained=$model_path,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=4096" \
                --tasks mbpp \
                --num_fewshot 3 \
                --batch_size auto \
                --seed 42 \
                --gen_kwargs temperature=0 \
                --confirm_run_unsafe_code \
                --output_path "$RESULTS_DIR/$mbpp_output_name" \
                2>&1 | tee "$RESULTS_DIR/${mbpp_output_name}.log"
        else
            echo "Already exists: ${mbpp_output_name}"
        fi
    else
        echo "Skipping $name: path not found ($model_path)"
    fi
done

echo ""
echo "All evaluations complete!"
