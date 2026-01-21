# AGENTS.md

Notes for AI agents working on this codebase.

## Compute Resources

### Available Nodes

There are **two compute nodes** available for evaluation and training:

| Node | Private IP | GPUs | Notes |
|------|------------|------|-------|
| Primary (this node) | (current) | 8x A100-80GB | Main development node |
| Secondary | 172.31.17.116 | 8x A100-80GB | Same VPC, SSH accessible |

**SSH to secondary node:**
```bash
ssh ubuntu@172.31.17.116
```

Both nodes mount the same EFS filesystem at `/efs/rlvr-experiments/`, so scripts and checkpoints are shared.

**IMPORTANT: When running commands on the secondary node via SSH, always use absolute paths.**

Due to shell initialization quirks, `cd /path && source .venv/bin/activate` often fails. Instead:
- Use `/efs/rlvr-experiments/.venv/bin/activate` (absolute path)
- Use `/efs/rlvr-experiments/scripts/...` for scripts
- Reference checkpoints with full paths `/efs/rlvr-experiments/checkpoints/...`

Example:
```bash
# WRONG - often fails
ssh ubuntu@172.31.17.116 "cd /efs/rlvr-experiments && source .venv/bin/activate && python script.py"

# CORRECT - use absolute paths
ssh ubuntu@172.31.17.116 "source /efs/rlvr-experiments/.venv/bin/activate && python /efs/rlvr-experiments/script.py"
```

**Running parallel evals on both nodes:**
```bash
# On primary node - run jobs 0-19 on GPUs 0-7
./scripts/evaluation/launch_math_qwen_workers.sh 0 8 0

# On secondary node - run jobs 20-38 on GPUs 0-7
ssh ubuntu@172.31.17.116 "cd /efs/rlvr-experiments && ./scripts/evaluation/launch_math_qwen_workers.sh 20 8 0"
```

## General Guidelines

### GPU Utilization - MAXIMIZE PARALLELISM

**CRITICAL: Always use ALL available GPUs when running parallel jobs.**

**BEFORE LAUNCHING ANY GPU JOB:**
1. **CHECK if the GPU is free first**: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
2. If memory > 0 but utilization is 0%, it's likely a zombie process - **KILL IT** and use that GPU
3. If memory > 0 and utilization > 0%, that GPU is actively in use - pick a different one
4. Only launch on GPUs showing 0 MiB memory usage (or after killing zombies)

**Killing zombie GPU processes:**
```bash
# Check full nvidia-smi to see PIDs
nvidia-smi
# Kill specific zombie process
kill -9 <PID>
```

When launching batch evaluations or any parallel workload:
1. **Count the total jobs** (e.g., 6 checkpoints × 5 tasks = 30 jobs)
2. **Check which GPUs are free** before launching anything
3. **Use ALL available FREE GPUs** (typically 8 per node, 16 across both nodes)
4. **Launch jobs in parallel** - don't serialize when you can parallelize
5. **Monitor GPU utilization** - if GPUs are idle, you're wasting expensive compute

**Example - WRONG (wastes resources):**
```bash
# Only using 3 GPUs when 8 are available
for ckpt in step20 step40 step60; do
    run_eval.sh $ckpt minerva_math 4 $gpu &
    gpu=$((gpu + 1))
done
```

**Example - CORRECT (maximizes utilization):**
```bash
# Using all 8 GPUs across both nodes
# 6 checkpoints = 6 GPUs
for i in {0..5}; do
    run_eval.sh ${CKPTS[$i]} minerva_math 4 $i &
done
```

**Multi-node parallelism:**
- Primary node: GPUs 0-7
- Secondary node (172.31.17.116): GPUs 0-7
- Launch jobs on BOTH nodes simultaneously for maximum throughput

This is expensive cloud compute. Every idle GPU is wasted money.

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

### Tulu Thinker GRPO (AllenAI Mixed) - Local + Parallel Eval

Local training on GPUs 0-5 with OLMES eval on GPUs 6-7:

```bash
source .venv/bin/activate

# Start Ray with only training GPUs visible
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ray start --head --num-gpus=6 --temp-dir=/opt/dlami/nvme/ray_tmp

# Train (leave GPUs 6-7 free)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python entrypoints/train_grpo.py \
  configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e3-thinker.yaml \
  2>&1 | tee /tmp/grpo_thinker_train.log

# In another shell, run OLMES eval watch on GPUs 6-7.
# NOTE: OLMES runs in its own venv in the Docker image (/opt/olmes-venv).
# For local runs, DO NOT install OLMES into .venv (it will clash with torch).
# Instead, use /opt/olmes-venv/bin/python if present, or create a separate venv
# (e.g., /efs/rlvr-experiments/.olmes-venv) and install requirements-olmes.txt there.
# Set watch.local_dir in configs/eval/olmes-tulu-watch.yaml to /efs/rlvr-experiments/checkpoints
CUDA_VISIBLE_DEVICES=6,7 /efs/rlvr-experiments/.venv/bin/python \
  entrypoints/eval_benchmarks.py configs/eval/olmes-tulu-watch.yaml \
  2>&1 | tee /tmp/olmes_eval_watch.log
```

If running locally, consider changing `output_dir` in `configs/eval/olmes-tulu-watch.yaml` to a local path
like `/efs/rlvr-experiments/eval_results/olmes_tulu_watch`.

### Tulu Thinker GRPO (AllenAI Mixed) - SageMaker

Single-job training + eval (trainer on GPUs 0-5, eval on GPUs 6-7 by default):

```bash
source .venv/bin/activate

python src/rlvr_experiments/submit.py \
  configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e3-thinker.yaml \
  --eval-config configs/eval/olmes-tulu-watch.yaml \
  --build --push
```

If you need an explicit GPU split, add `--train-gpus "0,1,2,3,4,5" --eval-gpus "6,7"`.

Notes:
- On SageMaker, the launcher uses `/opt/olmes-venv/bin/python` for OLMES evals by default.
- Locally, keep OLMES in a separate venv (do not install into `.venv`); use `/opt/olmes-venv/bin/python`
  or create `/efs/rlvr-experiments/.olmes-venv` and install `requirements-olmes.txt` there.

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
# GSM8K (math reasoning) - use gsm8k_cot for CoT prompting (matches Qwen3 eval)
# Note: Qwen3 technical report uses "gsm8k (4-shot, cot)" which maps to gsm8k_cot
--tasks gsm8k_cot --num_fewshot 4    # Chain-of-thought prompting (primary metric: strict-match)
--tasks gsm8k --num_fewshot 8        # Alternative: non-CoT format

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

6. **MBPP/HumanEval require unsafe code execution** - These benchmarks run generated code. You MUST enable code execution with BOTH:
   - Environment variable: `export HF_ALLOW_CODE_EVAL=1`
   - CLI flag: `--confirm_run_unsafe_code` (for lm_eval)

   Example for MBPP:
   ```bash
   export HF_ALLOW_CODE_EVAL=1
   lm_eval --model vllm \
     --model_args pretrained=/path/to/model,dtype=bfloat16,tensor_parallel_size=2 \
     --tasks mbpp --num_fewshot 3 --batch_size auto --seed 42 \
     --gen_kwargs temperature=0 \
     --confirm_run_unsafe_code \
     --output_path ./eval_results/mbpp_3shot
   ```

   For batch evaluation scripts, add to the worker script:
   ```bash
   # Enable code evaluation for MBPP
   export HF_ALLOW_CODE_EVAL=1
   ```

   Without these flags, MBPP evaluations will fail with:
   ```
   ValueError: The "code_eval" metric executes untrusted model-generated code...
   set the environment variable HF_ALLOW_CODE_EVAL="1"
   ```

## Qwen3-1.7B-Base Evaluation Baselines

Reference scores for comparing trained checkpoints.

### CRITICAL: Evaluation Requirements

**NEVER use tensor_parallel_size > 1 for evaluation.** TP introduces non-determinism and gives wildly wrong results (e.g., 10% instead of 28% on minerva_math). These are small models - parallelize TASKS across GPUs, not the model itself.

**Required settings for ALL evaluations:**
```
tensor_parallel_size=1    # MANDATORY - never use TP > 1
seed=42                   # In model_args AND --seed flag
temperature=0             # --gen_kwargs temperature=0
max_model_len=4096        # Prevent OOM on long sequences
gpu_memory_utilization=0.8
```

**Standard command template:**
```bash
CUDA_VISIBLE_DEVICES=$GPU lm_eval --model vllm \
  --model_args "pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
  --tasks $TASK --num_fewshot $FEWSHOT \
  --batch_size auto --seed 42 --gen_kwargs temperature=0 \
  --output_path $OUTPUT_DIR
```

**Parallel evaluation across GPUs:**
```bash
# Run 7 different benchmarks in parallel on GPUs 0-6
CUDA_VISIBLE_DEVICES=0 lm_eval ... --tasks gsm8k --num_fewshot 4 &
CUDA_VISIBLE_DEVICES=1 lm_eval ... --tasks gsm8k --num_fewshot 8 &
CUDA_VISIBLE_DEVICES=2 lm_eval ... --tasks gsm8k_cot --num_fewshot 4 &
# ... etc
```

### Baselines (2026-01-20, TP=1, seed=42, temperature=0)

#### GSM8K (Math Reasoning)

| Task | Setting | flexible-extract | strict-match |
|------|---------|------------------|--------------|
| gsm8k | 4-shot | 68.61% | 59.44% |
| gsm8k | 8-shot | 70.13% | 69.60% |
| gsm8k_cot | 4-shot | 69.83% | 51.63% |
| gsm8k_cot | 8-shot | 73.24% | 67.17% |

**Note**: `gsm8k` uses `#### N` answer format. `gsm8k_cot` uses `The answer is N.` format.

#### MATH

| Benchmark | Setting | Metric | Value |
|-----------|---------|--------|-------|
| hendrycks_math | 4-shot | exact_match | **17.78%** |
| minerva_math | 4-shot | exact_match | **28.20%** |
| minerva_math | 4-shot | math_verify | **39.08%** |

**minerva_math breakdown by subject:**

| Subject | exact_match | math_verify |
|---------|-------------|-------------|
| prealgebra | 47.30% | 60.85% |
| algebra | 38.53% | 56.08% |
| num_theory | 23.33% | 27.96% |
| geometry | 21.71% | 31.94% |
| counting_and_prob | 17.30% | 32.49% |
| precalc | 12.82% | 20.70% |
| intermediate_algebra | 12.18% | 19.49% |

#### IFEval (Instruction Following, 0-shot)

| Metric | Value |
|--------|-------|
| prompt_level_strict_acc | 21.26% |
| inst_level_strict_acc | 32.85% |

#### Code Generation

| Benchmark | Setting | pass@1 |
|-----------|---------|--------|
| MBPP      | 3-shot  | 55.8%  |
| HumanEval | 0-shot  | 48.78% |

## Batch Checkpoint Evaluation

### CRITICAL: Use Existing Scripts

**DO NOT CREATE NEW EVALUATION SCRIPTS.** Use the existing scripts in `scripts/evaluation/`. If something doesn't work, FIX THE EXISTING SCRIPTS.

The scripts already enforce the correct settings (TP=1, seed=42, temperature=0). Creating ad-hoc scripts leads to bugs like using TP=8 which gives completely wrong results.

### Scripts Location

| Script | Purpose |
|--------|---------|
| `run_lm_eval.sh` | Run single eval: `./run_lm_eval.sh <ckpt_path> <output_name> <task> <fewshot> [gpu]` |
| `run_batch_evals.sh` | Run all standard tasks on multiple checkpoints from a JSON file |
| `extract_results.py` | Extract lm_eval JSON results to CSV/Markdown |

### Running Single Evaluations

```bash
# GSM8K 8-shot on GPU 0
./scripts/evaluation/run_lm_eval.sh /path/to/checkpoint my_ckpt_step100 gsm8k 8 0

# MBPP 3-shot on GPU 1 (code execution handled automatically)
./scripts/evaluation/run_lm_eval.sh /path/to/checkpoint my_ckpt_step100 mbpp 3 1

# IFEval 0-shot on GPU 2
./scripts/evaluation/run_lm_eval.sh /path/to/checkpoint my_ckpt_step100 ifeval 0 2
```

### Running Parallel Baseline Evals

To evaluate a model on all standard benchmarks in parallel (one per GPU):

```bash
MODEL="/path/to/model"
OUTPUT_DIR="/path/to/output"
COMMON="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42"

# Launch all in parallel on GPUs 0-6
CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm --model_args "$COMMON" --tasks gsm8k --num_fewshot 4 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/gsm8k_4shot" &
CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm --model_args "$COMMON" --tasks gsm8k --num_fewshot 8 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/gsm8k_8shot" &
CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm --model_args "$COMMON" --tasks gsm8k_cot --num_fewshot 4 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/gsm8k_cot_4shot" &
CUDA_VISIBLE_DEVICES=3 lm_eval --model vllm --model_args "$COMMON" --tasks gsm8k_cot --num_fewshot 8 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/gsm8k_cot_8shot" &
CUDA_VISIBLE_DEVICES=4 lm_eval --model vllm --model_args "$COMMON" --tasks hendrycks_math --num_fewshot 4 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/hendrycks_math_4shot" &
CUDA_VISIBLE_DEVICES=5 lm_eval --model vllm --model_args "$COMMON" --tasks minerva_math --num_fewshot 4 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/minerva_math_4shot" &
CUDA_VISIBLE_DEVICES=6 lm_eval --model vllm --model_args "$COMMON" --tasks ifeval --num_fewshot 0 --batch_size auto --seed 42 --gen_kwargs temperature=0 --output_path "$OUTPUT_DIR/ifeval_0shot" &
wait
```

### Running Batch Evaluations

Create a JSON file with checkpoints:
```json
[
  {"checkpoint_path": "/path/to/ckpt1", "output_name": "exp1_step100"},
  {"checkpoint_path": "/path/to/ckpt2", "output_name": "exp1_step200"}
]
```

Run all 5 tasks (gsm8k 8-shot, gsm8k 4-shot, hendrycks_math, ifeval, mbpp):
```bash
./scripts/evaluation/run_batch_evals.sh jobs.json 0  # Run on GPU 0
```

### Extracting Results

```bash
# Generate CSV and Markdown from eval_results_batch/
python scripts/evaluation/extract_results.py

# Custom paths
python scripts/evaluation/extract_results.py --results-dir /custom/results --csv output.csv
```

### Configuration Details

All evals use these standard settings:
- vLLM backend with bfloat16
- tensor_parallel_size=1 (single GPU per eval)
- seed=42, temperature=0 (greedy decoding)
- gpu_memory_utilization=0.8, max_model_len=4096

**Code execution tasks (MBPP, HumanEval):**
- `HF_ALLOW_CODE_EVAL=1` environment variable (set automatically by scripts)
- `--confirm_run_unsafe_code` flag (added automatically for mbpp/humaneval tasks)

### Metric Keys in Results JSON

- GSM8K: `results.gsm8k.exact_match,flexible-extract` and `results.gsm8k.exact_match,strict-match`
- hendrycks_math: `results.hendrycks_math.exact_match,none`
- IFEval: `results.ifeval.prompt_level_strict_acc,none` and `results.ifeval.inst_level_strict_acc,none`
- MBPP: `results.mbpp.pass_at_1,none` (note: underscore, not @)

### Adding New Job Types

When you have a new training job with different naming conventions:
1. **DO NOT create a new script**
2. Edit `scripts/evaluation/extract_results.py` and add the job to `JOB_METADATA` or `HADADV_JOBS`
3. The run_lm_eval.sh script is generic and works with any checkpoint

## MATH Benchmark Evaluation

**CRITICAL: ALWAYS use `math_qwen` task, NEVER use `hendrycks_math`.**

The `lm_eval` `hendrycks_math` task gives ~17% accuracy on Qwen3-1.7B-Base, but Qwen reports 43.5%. This is NOT a model issue - it's an evaluation methodology issue. The `hendrycks_math` results are **misleading and should not be used**.

### Why `hendrycks_math` is Wrong

| Aspect | lm_eval `hendrycks_math` (DON'T USE) | `math_qwen` (USE THIS) |
|--------|--------------------------------------|------------------------|
| Prompt template | `Problem: {X}\nAnswer:` | `Question: {X}\nAnswer:` |
| Few-shot answers | Just the boxed value | Full CoT with reasoning + `\boxed{}` |
| Answer extraction | String matching | `\boxed{}` + "The answer is X" patterns |
| Verification | String normalization | Sympy symbolic equivalence |
| **Base model accuracy** | **17.68% (WRONG)** | **42.9% (CORRECT)** |

Base models need CoT exemplars showing HOW to reason and format answers.

### Running MATH Evaluation

Use the standard `run_lm_eval.sh` script with `math_qwen` task:

```bash
# Single checkpoint
./scripts/evaluation/run_lm_eval.sh /path/to/checkpoint my_ckpt_step100 math_qwen 4 0

# Batch evaluation (uses math_qwen automatically for MATH)
./scripts/evaluation/run_batch_evals.sh jobs.json 0
```

The `math_qwen` task internally calls `scripts/adhoc/eval_math_qwen_style.py`.

See `scripts/adhoc/NOTES_math_eval.md` for technical details.

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
