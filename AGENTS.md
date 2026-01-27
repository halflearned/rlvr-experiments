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
# On primary node - evaluate checkpoint on gsm8k using GPU 0
python scripts/eval_checkpoint.py /path/to/checkpoint results/evals/gsm8k --benchmark gsm8k --gpu 0 &

# On secondary node - evaluate same checkpoint on ifeval using GPU 0
ssh ubuntu@172.31.17.116 "source /efs/rlvr-experiments/.venv/bin/activate && python /efs/rlvr-experiments/scripts/eval_checkpoint.py /path/to/checkpoint /efs/rlvr-experiments/results/evals/ifeval --benchmark ifeval --gpu 0"
```

## General Guidelines

### Training Config Variables - DO NOT SUGGEST CHANGES

**NEVER suggest changing `max_staleness`.** This is a key experimental variable being ablated. If a run fails due to sync issues, the problem is the sync mechanism itself - not the staleness setting. Diagnose the actual root cause (e.g., vLLM workers dying, sync timing issues) rather than suggesting config workarounds.

### User Communication - TRUST THE USER

When the user says something is stuck, BELIEVE THEM. They have been staring at it for several minutes before telling you. Take IMMEDIATE action - kill the process, check logs, investigate. Do not waste time with "let me check if it's growing" - if the user says it's stuck, it's stuck.

### Don't Kill Runs During Active Conversation

**CRITICAL**: When actively conversing with the user:
1. **INVESTIGATE THOROUGHLY FIRST** - Read logs, understand the pattern, form a hypothesis
2. **ASK THE USER** before killing - "Can I kill this run to try X?"
3. **NEVER overwrite logs** by restarting without understanding the previous failure

When working autonomously (user stepped away, gave blanket permission), take action as needed.

Restarting blindly destroys evidence. Logs are precious. Investigate BEFORE killing.

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

### Run Artifacts (train_grpo)
Training runs create a self-contained run folder at startup:

- **Location**:
  - SageMaker: `$SM_MODEL_DIR/<config_stem>_<YYYYMMDD-HHMMSS>/`
  - Local: `results/<config_stem>_<YYYYMMDD-HHMMSS>/`
- **Structure**:
  - `config.yaml` (exact config used)
  - `RESULTS.md` (notes scratchpad)
  - `run.json` (metadata: host, timestamps, git info, etc.)
  - `patches/` (`git_status.txt`, `git_diff.patch`, `git_diff_cached.patch`, `git_untracked.txt`, optional `untracked.tar.gz`)
  - `traces/` (`trace.jsonl`, `samples.jsonl`, `rollouts.jsonl`)
  - `checkpoints/` (all checkpoints for this run)

Notes:
- Trace/rollout files no longer include timestamps in their filenames; the run folder itself is unique.
- `RLVR_RUN_DIR` is set internally so checkpoints/traces route into this folder.
- Existing S3 uploads for traces/rollouts continue unchanged.

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

**To download completed job results:**

Use `scripts/download_job_results.sh` to download and unpack the full job output (config, traces, checkpoints):

```bash
./scripts/download_job_results.sh annotations-adhoc-20260125-192905
```

This downloads `model.tar.gz` from S3, unpacks it, and flattens the directory structure so that `config.yaml`, `traces/`, and `checkpoints/` are directly under `results/<job_name>/`.

**To sync traces from running jobs:**

Use `scripts/sync_traces.sh` to poll S3 and download `trace.jsonl` and `config.yaml` for running jobs:

```bash
# One-time sync for a specific job
./scripts/sync_traces.sh annotations-adhoc-20260125-192901

# Poll every 60s (auto-exits after 1h with no changes)
./scripts/sync_traces.sh annotations-adhoc-20260125-192901 60

# Sync all jobs matching a pattern
./scripts/sync_traces.sh 20260125 60

# Run in background
nohup ./scripts/sync_traces.sh annotations-adhoc-20260125-192901 60 > /tmp/sync_job.log 2>&1 &
```

Files are saved to `results/<job_name>/traces/trace.jsonl` and `results/<job_name>/config.yaml`.

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

### Local lm_eval Watcher (Checkpoint Polling)

Use `entrypoints/eval_benchmarks.py` in watch mode to poll a checkpoint folder and run lm_eval
tasks sequentially per GPU (TP=1). Static settings live in `configs/eval/lm_eval_watch.yaml`;
dynamic checkpoint glob is passed via `--watch-path`.

```bash
# Check GPUs 6-7 are free
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Run watcher on GPUs 6-7 (task groups use visible GPU indices 0 and 1)
CUDA_VISIBLE_DEVICES=6,7 /efs/rlvr-experiments/.venv/bin/python \
  entrypoints/eval_benchmarks.py configs/eval/lm_eval_watch.yaml \
  --watch-path /efs/rlvr-experiments/checkpoints/allenai_full_lr5e6_beta1e3_repro_step* \
  2>&1 | tee /tmp/lm_eval_watch.log
```

Notes:
- `configs/eval/lm_eval_watch.yaml` sets `gpu_mode: visible`, so task groups use indices
  within `CUDA_VISIBLE_DEVICES`. If you want to use absolute GPU IDs in YAML, set
  `gpu_mode: absolute`.
- Results are written under `/efs/rlvr-experiments/eval_results/repro_watch/<checkpoint_name>/`.
- Stop the watcher with: `pkill -f "entrypoints/eval_benchmarks.py configs/eval/lm_eval_watch.yaml"`.

## Checkpoint Evaluation

Use `scripts/eval_checkpoint.py` to evaluate checkpoints on supported benchmarks. This is the canonical evaluation script - generates completions with vLLM (TP=1, greedy) and verifies using our verifiers.

### Supported Benchmarks

| Benchmark | Dataset | max_tokens | Verifier |
|-----------|---------|------------|----------|
| `gsm8k` | GSM8K test (1319 examples) | 1024 | `MathVerifier` |
| `ifeval` | Google IFEval (541 examples) | 2048 | `IFMultiConstraintsVerifier` |
| `ifbench` | AllenAI IFBench test | 2048 | `IFBenchVerifier` |

### Usage

```bash
python scripts/eval_checkpoint.py <checkpoint_path> <output_dir> --benchmark <benchmark> [--gpu GPU]
```

### Examples

```bash
# Evaluate on GSM8K
python scripts/eval_checkpoint.py \
    results/qwen3-1.7B-gsm8k-grpo-stale0_20260125-033959/checkpoints/step100 \
    results/qwen3-1.7B-gsm8k-grpo-stale0_20260125-033959/evals/gsm8k \
    --benchmark gsm8k --gpu 0

# Evaluate on IFEval
python scripts/eval_checkpoint.py \
    results/my_run/checkpoints/step100 \
    results/my_run/evals/ifeval \
    --benchmark ifeval --gpu 0

# Evaluate on IFBench
python scripts/eval_checkpoint.py \
    results/my_run/checkpoints/step100 \
    results/my_run/evals/ifbench \
    --benchmark ifbench --gpu 0
```

### Output Files

The script creates three files in `<output_dir>/`:

| File | Contents |
|------|----------|
| `completions.jsonl` | Raw completions with prompts |
| `results.jsonl` | Completions with verification results |
| `summary.json` | Aggregate metrics |

### Parallel Evaluation

Run multiple benchmarks in parallel on different GPUs:

```bash
# Evaluate same checkpoint on all benchmarks in parallel
python scripts/eval_checkpoint.py ckpt evals/gsm8k --benchmark gsm8k --gpu 0 &
python scripts/eval_checkpoint.py ckpt evals/ifeval --benchmark ifeval --gpu 1 &
python scripts/eval_checkpoint.py ckpt evals/ifbench --benchmark ifbench --gpu 2 &
wait
```

### Evaluation Settings

All evaluations use:
- vLLM with `tensor_parallel_size=1` (single GPU)
- Greedy decoding (`temperature=0`)
- `dtype=bfloat16`, `gpu_memory_utilization=0.9`

### Metrics

**GSM8K:**
- `accuracy`: Fraction of correct answers (verified by `MathVerifier`)

**IFEval / IFBench:**
- `prompt_level_strict_acc`: % of prompts where ALL constraints pass
- `inst_level_acc`: % of individual instructions that pass

### Legacy Scripts

Older evaluation scripts have been moved to `scripts/legacy/`. These include pass rate evaluation, lm_eval wrappers, and other superseded approaches. Use `scripts/eval_checkpoint.py` for new evaluations.

## Pass@k Evaluation

For computing pass@k metrics (probability of getting at least one correct answer in k attempts), use `scripts/eval_pass_at_k.py`. This generates N completions per prompt with temperature sampling and verifies each one.

### Basic Usage

```bash
# Single GPU evaluation
python scripts/eval_pass_at_k.py <dataset> \
  --model-path /path/to/model \
  --output-dir /path/to/output \
  --n 128 \
  --batch-size 16 \
  --gpus 0

# Multi-GPU parallel evaluation (recommended)
./scripts/launch_pass_at_k.sh <dataset> <output_dir> <gpu_list> <max_model_len> [extra_args]
```

### Supported Datasets

The script uses a registry mapping dataset names to loaders and verifiers:

| Dataset | Loader | Verifier |
|---------|--------|----------|
| `gsm8k` | `load_gsm8k` | `MathVerifier` |
| `math` | `load_math` | `MathVerifier` |
| `if_multi_constraints` | `load_if_multi_constraints` | `IFMultiConstraintsVerifier` |
| `ifbench` | `load_ifbench` | `IFBenchVerifier` |
| `mbpp` | `load_mbpp` | `MBPPVerifier` |

### Example: Full Pass@k Run on Secondary Node

```bash
# GSM8K with 128 completions per prompt, all 8 GPUs
ssh ubuntu@172.31.17.116 "source /efs/rlvr-experiments/.venv/bin/activate && \
  cd /efs/rlvr-experiments && \
  ./scripts/launch_pass_at_k.sh gsm8k \
    /efs/rlvr-experiments/results/qwen3-1.7B-base/evals/gsm8k/pass-at-k \
    0,1,2,3,4,5,6,7 \
    4096 \
    '--shuffle --verifier-workers 8'"

# IF Multi Constraints (longer prompts, need filtering)
ssh ubuntu@172.31.17.116 "source /efs/rlvr-experiments/.venv/bin/activate && \
  cd /efs/rlvr-experiments && \
  ./scripts/launch_pass_at_k.sh if_multi_constraints \
    /efs/rlvr-experiments/results/qwen3-1.7B-base/evals/if_multi_constraints/pass-at-k \
    0,1,2,3,4,5,6,7 \
    4096 \
    '--shuffle --verifier-workers 8'"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n` | 128 | Number of completions per prompt |
| `--batch-size` | 16 | Prompts per batch |
| `--max-tokens` | 1024 | Max tokens per completion |
| `--temperature` | 1.0 | Sampling temperature |
| `--max-model-len` | 4096 | vLLM max sequence length |
| `--max-prompt-len` | auto | Filter prompts exceeding this token count |
| `--shuffle` | False | Shuffle dataset before sharding |
| `--verifier-workers` | 4 | Parallel verification workers |

### Output Files

Each shard creates:
- `verification_results.jsonl`: Per-prompt results with all completions and scores
- `pass_at_k.json`: Pass@k metrics for the shard

After all shards complete, `launch_pass_at_k.sh` merges results into:
- `merged_results.jsonl`: Combined results from all shards
- `pass_at_k.json`: Overall pass@k metrics

### Resume Support

The script automatically resumes from where it left off by checking existing results. If a run crashes, just relaunch with the same command.

### Gotchas

1. **Long prompts**: Some datasets (e.g., IF Multi Constraints) have prompts that exceed `max_model_len`. The script filters these automatically based on `max_prompt_len` (defaults to `max_model_len - max_tokens`).

2. **GPU memory**: Each shard runs independently with TP=1. If you get OOM errors, reduce `max_model_len` or `batch_size`.

3. **Orphaned processes**: If vLLM crashes, GPU memory may not be released. Check `nvidia-smi` for orphaned processes and kill them manually.

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
