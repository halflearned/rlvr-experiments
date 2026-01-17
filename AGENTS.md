# AGENTS.md

Notes for AI agents working on this codebase.

## General Guidelines

### Long-Running Commands
When running commands that take minutes (evals, training, etc.), always:
1. **Log to a temp file** - Use `tee /tmp/<descriptive_name>.log` to capture stdout/stderr
2. **Let the user monitor progress** - They can `tail -f` the log file
3. **Avoid losing output** - Don't rely on small `tail -N` values that might miss results

Example:
```bash
lm_eval --model vllm ... 2>&1 | tee /tmp/eval_gsm8k_base.log
```

### Package Management
- Use `uv add <package>` to add dependencies (updates pyproject.toml)
- Do NOT use `uv pip install` for persistent dependencies

## SageMaker Job Submission

### Required IAM Permissions

The EC2 instance role needs the following permissions to submit SageMaker training jobs:

#### ECR Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:CreateRepository",
                "ecr:DescribeRepositories",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage"
            ],
            "Resource": "arn:aws:ecr:us-west-2:503561457547:repository/rlvr-experiments"
        }
    ]
}
```

#### SageMaker Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:ListTrainingJobs"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::503561457547:role/*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        }
    ]
}
```

### SageMaker Job Name Constraints

SageMaker training job names must match the regex pattern: `[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}`

- Only alphanumeric characters and hyphens allowed
- No periods, underscores, or other special characters
- Max 63 characters

**Job naming convention:** `annotations-adhoc-{YYYYMMDD-HHMMSS}`

Use non-descript names to avoid leaking experiment details. Log the mapping in `docs/job_log.md`.

### Submitting Jobs

Use `src/rlvr_experiments/submit.py` to submit jobs:

```bash
# Activate venv first
source .venv/bin/activate

# Submit a job (auto-generates annotations-adhoc-{timestamp} name)
python src/rlvr_experiments/submit.py configs/qwen3-1.7B-mixed.yaml

# Build and push Docker image, then submit
python src/rlvr_experiments/submit.py configs/my-config.yaml --build --push

# Run locally with specific GPUs (bypasses SageMaker)
python src/rlvr_experiments/submit.py configs/my-config.yaml --local --gpus "0,1,2,3"

# Specify instance type (default: ml.p4de.24xlarge)
python src/rlvr_experiments/submit.py configs/my-config.yaml --instance-type ml.p4d.24xlarge
```

After submitting, add an entry to `docs/job_log.md` with the job name, config, and description.

### Job Log

Keep a log of submitted jobs in `docs/job_log.md`:

```markdown
| Job Name | Config | Description | Submitted |
|----------|--------|-------------|-----------|
| annotations-adhoc-20260116-011803 | configs/qwen3-1.7B-mixed.yaml | Mixed dataset, 624 steps | 2026-01-16 01:18 |
```

### Model Weights

SageMaker instances don't have access to `/efs`. Options for model weights:

**Option 1: HuggingFace Hub (recommended for public models)**

Use HF Hub paths directly in configs. The model will be downloaded at job start:
```yaml
model:
  path: "Qwen/Qwen3-1.7B-Base"  # Downloaded from HF Hub
```

**Option 2: S3 (for private/fine-tuned models)**

Upload to S3, then the `sagemaker_launcher.py` downloads at job start:
```bash
# Upload model to S3
aws s3 sync /efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base \
    s3://sagemaker-us-west-2-503561457547/rlvr-experiments/models/Qwen3-1.7B-Base/
```

The launcher maps HF-style paths to S3:
- Config: `Qwen/Qwen3-1.7B-Base` → S3: `s3://.../models/Qwen3-1.7B-Base/`

Note: The current `sagemaker_launcher.py` always tries S3 first for paths with `/`. To use HF Hub directly, either use the full HF path or modify the launcher.

### Checkpoints and Traces

On SageMaker, checkpoints and trace files are uploaded directly to S3 as training progresses (not saved to MODEL_DIR). This avoids waiting for job completion to access results.

**S3 checkpoint location:**
```
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints/{job_name}/{run_name}_step{N}/
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints/{job_name}/{run_name}_final/
```

**S3 traces location:**
```
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints/{job_name}/traces/
```

**To download a checkpoint:**
```bash
# List available checkpoints for a job
aws s3 ls s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints/annotations-adhoc-20260117-041250/

# Download a specific checkpoint
aws s3 sync s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints/annotations-adhoc-20260117-041250/mixed_lr1e5_curriculum_final/ \
    /efs/rlvr-experiments/checkpoints/mixed_lr1e5_curriculum_final/
```

### Datasets

SageMaker instances in the VPC cannot access HuggingFace Hub. Datasets are pre-cached to S3 and the data loaders (`src/rlvr_experiments/data.py`) automatically load from S3 first, falling back to HuggingFace Hub if S3 is unavailable.

**S3 dataset locations:**
```
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/gsm8k_train/
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/math_train/
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/mbpp_train/
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/ifeval_train/
```

**To add a new dataset to S3:**
```python
from datasets import load_dataset

# Download from HuggingFace and save as Arrow format
ds = load_dataset("org/dataset-name", split="train")
ds.save_to_disk("/tmp/dataset_train_cache")
```

```bash
# Upload to S3
aws s3 sync /tmp/dataset_train_cache s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/dataset_train/
```

Then update the loader in `data.py` to check S3 first (see existing loaders for pattern).

### Curriculum / Order Files

For curriculum learning, you can specify an `order` file in the data config that lists prompt IDs in the desired order. These files must be accessible from SageMaker, so use S3 paths:

**Config example:**
```yaml
data:
  dataset: mixed
  datasets:
    - name: gsm8k
      weight: 0.25
      order: "s3://sagemaker-us-west-2-503561457547/rlvr-experiments/curricula/gsm8k_curriculum.txt"
```

**S3 curriculum location:**
```
s3://sagemaker-us-west-2-503561457547/rlvr-experiments/curricula/
```

**To upload curriculum files:**
```bash
# Upload from local experiments directory
aws s3 sync /efs/rlvr-experiments/experiments/curricula/ \
    s3://sagemaker-us-west-2-503561457547/rlvr-experiments/curricula/
```

The `load_mixed` function in `data.py` handles S3 paths automatically by downloading to a temp file before reading.

### Common Issues

1. **ECR Repository Not Found**: The ECR repository `rlvr-experiments` must exist before pushing. The script attempts to create it automatically, but requires `ecr:CreateRepository` permission.

2. **Docker Image Build**: Use `--build` flag to rebuild the image. Use `--push` to push to ECR.

3. **Local Testing**: Use `--local --gpus "0,1,2,3"` to run in Docker locally before submitting to SageMaker.

4. **Model Not Found on SageMaker**: Ensure model weights are uploaded to S3 before submitting the job (see "Model Weights in S3" above).

5. **Dataset Download Timeout**: If HuggingFace dataset downloads fail with connection timeouts, the dataset needs to be cached to S3 (see "Datasets" above).

## Local Training

For quick local experiments (not SageMaker), run training directly with Ray.

### Starting Ray

```bash
source .venv/bin/activate

# Start Ray with NVMe temp directory (required on this machine)
ray start --head --num-gpus=8 --temp-dir=/opt/dlami/nvme/ray_tmp

# Check Ray status
ray status

# Stop Ray when done
ray stop
```

### Running Training

```bash
# Run training script directly
python entrypoints/train_grpo.py configs/my-config.yaml
```

### Notes

- Always use `--temp-dir=/opt/dlami/nvme/ray_tmp` to avoid filling up the root filesystem
- The config should use local `/efs/` paths for model and tokenizer
- Training logs are printed to stdout

## Model Evaluation with lm_eval

Use `lm_eval` with the **vLLM backend** for fast evaluation. The vLLM backend is ~50-100x faster than the HuggingFace backend.

### Basic Usage (vLLM backend, recommended)

```bash
source .venv/bin/activate

# gsm8k_cot evaluation with all 8 GPUs
lm_eval --model vllm \
  --model_args pretrained=/path/to/model,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,max_model_len=4096 \
  --tasks gsm8k_cot \
  --num_fewshot 8 \
  --batch_size auto \
  --gen_kwargs temperature=0 \
  --output_path ./eval_results/my_eval
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--model vllm` | Use vLLM backend (fast). Alternative: `hf` (slow) |
| `tensor_parallel_size=8` | Use all 8 GPUs for tensor parallelism |
| `gpu_memory_utilization=0.8` | Use 80% of GPU memory |
| `max_model_len=4096` | Limit context length (faster, sufficient for most evals) |
| `--num_fewshot N` | Number of few-shot examples (0, 4, 8 typical) |
| `--gen_kwargs temperature=0` | Greedy decoding |
| `--batch_size auto` | Auto-detect optimal batch size |

### Common Tasks

```bash
# GSM8K (math reasoning) - use gsm8k for base models, gsm8k_cot for instruct models
--tasks gsm8k --num_fewshot 8        # Base models: ~65% at 8-shot
--tasks gsm8k_cot --num_fewshot 8    # Instruct models with CoT

# MATH (harder math)
--tasks minerva_math --num_fewshot 4

# MBPP (code generation)
--tasks mbpp --num_fewshot 3
```

**Note:** For reproducibility, add `--seed 42` to the command.

### Example: Evaluate a Checkpoint

```bash
# Evaluate a trained checkpoint (base model)
lm_eval --model vllm \
  --model_args pretrained=checkpoints/my_checkpoint,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,max_model_len=512,seed=42 \
  --tasks gsm8k \
  --num_fewshot 8 \
  --batch_size auto \
  --gen_kwargs temperature=0 \
  --seed 42 \
  --output_path ./eval_results/my_checkpoint_gsm8k_8shot
```

### Gotchas

1. **Don't use `max_tokens` in gen_kwargs with vLLM** - it conflicts with vLLM's internal handling. Let lm_eval use its defaults.

2. **Use `tensor_parallel_size=8` for 8 GPUs** - this distributes the model across all GPUs for faster inference.

3. **Set `max_model_len`** - vLLM defaults to the model's full context (32K for Qwen3), which wastes memory. 4096 is sufficient for most evals.

4. **The `hf` backend is ~50-100x slower** - don't use it.

5. **Checkpoint safetensors naming** - Our training code saves weights as `model.safetensors`, but the index file (`model.safetensors.index.json`) expects `model-00001-of-00001.safetensors`. Before loading a checkpoint with vLLM, rename the file:
   ```bash
   cd checkpoints/my_checkpoint/
   mv model.safetensors model-00001-of-00001.safetensors
   ```

## Qwen3-1.7B-Base Evaluation Baselines

Reference scores for comparing trained checkpoints. All evals: seed=42, temperature=0, greedy decoding, vLLM backend.

**Commands used to generate these baselines:**
```bash
# GSM8K 0-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks gsm8k --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 0

# GSM8K 4-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks gsm8k --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 4

# GSM8K 8-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks gsm8k --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 8

# hendrycks_math 4-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks hendrycks_math --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 4

# IFEval 0-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks ifeval --batch_size auto --seed 42 --gen_kwargs temperature=0

# MBPP 3-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks mbpp --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 3 --confirm_run_unsafe_code

# HumanEval 0-shot
lm_eval --model vllm --model_args pretrained=/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks humaneval --batch_size auto --seed 42 --gen_kwargs temperature=0 --num_fewshot 0 --confirm_run_unsafe_code
```

### GSM8K (Math Reasoning)

| Setting | flexible-extract | strict-match |
|---------|------------------|--------------|
| 0-shot  | 14.71%           | 0.00%        |
| 4-shot  | 67.48%           | 59.14%       |
| 8-shot  | 69.60%           | 68.92%       |

### MATH (hendrycks_math, 4-shot)

| Metric | Value |
|--------|-------|
| Overall exact_match | **17.68%** |

### IFEval (Instruction Following, 0-shot)

| Metric | Value |
|--------|-------|
| prompt_level_strict_acc | 22.00% |
| inst_level_strict_acc | 33.81% |

### Code Generation

| Benchmark | Setting | pass@1 |
|-----------|---------|--------|
| MBPP      | 3-shot  | 55.8%  |
| HumanEval | 0-shot  | 48.78% |

## MBPP Training Notes

### Prompt Format
Uses lm_eval-compatible format:
```
You are an expert Python programmer, and here is your task: {problem_description} Your code should pass these tests:

{test_case_1}
{test_case_2}
{test_case_3}
[BEGIN]
```

Model generates code, stops at `[DONE]` token.

### Code Extraction
The verifier (`extract_code_from_markdown()`) handles:
- Markdown code blocks (```python ... ```)
- Trailing ``` from incomplete blocks
- [DONE] markers

### Verifier
- Runs generated code + test assertions in isolated subprocess
- 5 second timeout per verification
- Binary reward: 1.0 if all tests pass, 0.0 otherwise
