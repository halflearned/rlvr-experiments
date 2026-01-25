#!/usr/bin/env python3
"""Submit GRPO training jobs to SageMaker.

Uses boto3 directly to avoid sagemaker SDK's omegaconf dependency which
conflicts with antlr4 (required by lm-eval).
"""

import argparse
import boto3
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import pathspec

# AWS config
ACCOUNT = "503561457547"
REGION = "us-west-2"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/SageMaker"
ECR_PREFIX = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
BUCKET = f"sagemaker-{REGION}-{ACCOUNT}"

# VPC config (EFA-enabled)
SUBNETS = ["subnet-08b78e8d13faebd65"]
SECURITY_GROUP_IDS = ["sg-0448b04b00c40d716"]

# Defaults
DEFAULT_INSTANCE_TYPE = "ml.p4de.24xlarge"
DEFAULT_IMAGE_NAME = "rlvr-experiments"

PROJECT_DIR = Path(__file__).resolve().parents[2]


def run_cmd(cmd: str, check: bool = True):
    """Run a shell command."""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, cwd=PROJECT_DIR)


def stage_source_dir(root: Path) -> str:
    """Copy source, excluding gitignored files."""
    gitignore = root / ".gitignore"
    if gitignore.exists():
        spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore.open())
    else:
        spec = pathspec.PathSpec([])

    tmp = Path(tempfile.mkdtemp(prefix="sm_src_"))

    for src in root.rglob("*"):
        rel = src.relative_to(root)
        # Skip gitignored, hidden dirs, and large dirs
        if spec.match_file(str(rel)) or any(p.startswith(".") for p in rel.parts):
            continue
        if src.is_dir():
            (tmp / rel).mkdir(parents=True, exist_ok=True)
        else:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / rel)

    return str(tmp)


def upload_source_to_s3(source_dir: str, job_name: str) -> str:
    """Create tarball of source and upload to S3."""
    s3_client = boto3.client("s3", region_name=REGION)
    s3_prefix = f"rlvr-experiments/source/{job_name}"

    # Create tarball
    tarball_path = f"/tmp/{job_name}-source.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(source_dir, arcname="")

    s3_key = f"{s3_prefix}/sourcedir.tar.gz"
    print(f"Uploading source to s3://{BUCKET}/{s3_key}")
    s3_client.upload_file(tarball_path, BUCKET, s3_key)

    return f"s3://{BUCKET}/{s3_key}"


def get_image_uri(tag: str | None = None, build: bool = False, push: bool = False) -> str:
    """Get or build Docker image URI."""
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d")

    local_uri = f"{DEFAULT_IMAGE_NAME}:{tag}"
    remote_uri = f"{ECR_PREFIX}/{DEFAULT_IMAGE_NAME}:{tag}"

    if build:
        # Login to AWS DL container ECR (for base image)
        run_cmd(f"aws ecr get-login-password --region {REGION} | "
                f"docker login --username AWS --password-stdin 763104351884.dkr.ecr.{REGION}.amazonaws.com")
        run_cmd(f"docker build --no-cache -f Dockerfile -t {local_uri} .")

    if push:
        # Create repo if needed
        ecr = boto3.client("ecr", region_name=REGION)
        try:
            ecr.create_repository(repositoryName=DEFAULT_IMAGE_NAME)
        except ecr.exceptions.RepositoryAlreadyExistsException:
            pass

        # Login and push
        run_cmd(f"aws ecr get-login-password --region {REGION} | "
                f"docker login --username AWS --password-stdin {ECR_PREFIX}")
        run_cmd(f"docker tag {local_uri} {remote_uri}")
        run_cmd(f"docker push {remote_uri}")

    return remote_uri


def submit_job(
    config: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = 1,
    image_tag: str | None = None,
    build: bool = False,
    push: bool = False,
    job_name: str | None = None,
    wait: bool = False,
    local: bool = False,
    gpus: str | None = None,
    eval_config: str | None = None,
    train_gpus: str = "0,1,2,3,4,5",
    eval_gpus: str = "6,7",
    entrypoint: str | None = None,
):
    """Submit a training job to SageMaker.

    Args:
        config: Training config path
        eval_config: Optional eval config path. If provided, uses combined launcher
                    that runs training on train_gpus and eval watcher on eval_gpus
        train_gpus: GPUs for training (default: "0,1,2,3,4,5")
        eval_gpus: GPUs for evaluation (default: "6,7")
    """
    # Convert config paths to relative paths (SageMaker extracts source to /opt/ml/code/)
    def make_relative(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            try:
                return str(p.relative_to(PROJECT_DIR))
            except ValueError:
                # Path is not under PROJECT_DIR, use as-is
                return path
        return path

    config = make_relative(config)
    eval_config = make_relative(eval_config)

    local_uri = f"{DEFAULT_IMAGE_NAME}:{image_tag or datetime.now().strftime('%Y%m%d')}"
    remote_uri = get_image_uri(tag=image_tag, build=build, push=push)

    # Determine which launcher to use
    use_combined = eval_config is not None

    # For local mode with specific GPUs, run docker directly (bypass SageMaker)
    if local and gpus:
        print(f"\nRunning locally with GPUs: {gpus}")
        print(f"  Image: {local_uri}")
        print(f"  Config: {config}")

        num_gpus = len(gpus.split(","))
        # cuDNN fix: PyTorch bundles the correct cuDNN in nvidia/cudnn
        nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

        if use_combined:
            train_num_gpus = len(train_gpus.split(","))
            entrypoint = (
                f"python entrypoints/sagemaker_combined_launcher.py "
                f"--config {config} "
                f"--eval-config {eval_config} "
                f"--train-gpus {train_gpus} "
                f"--eval-gpus {eval_gpus}"
            )
            ray_cmd = f"ray start --head --num-gpus={train_num_gpus}"
        else:
            entrypoint = f"python entrypoints/train_grpo.py {config}"
            ray_cmd = f"ray start --head --num-gpus={num_gpus}"

        cmd = (
            f"docker run --rm "
            f'--gpus \'"device={gpus}"\' '
            f"--ipc=host "
            f"-v /dev/shm:/dev/shm "
            f"-v {PROJECT_DIR}:/opt/ml/code "
            f"-v {PROJECT_DIR}:{PROJECT_DIR} "
            f"-v {PROJECT_DIR}/training_output:/opt/ml/model "
            f"-e PYTORCH_ALLOC_CONF=expandable_segments:True "
            f"-e PYTHONPATH=/opt/ml/code/src "
            f"-e LD_LIBRARY_PATH={nvidia_libs} "
            f"-w /opt/ml/code "
            f"{local_uri} "
            f"bash -c '{ray_cmd} && {entrypoint}'"
        )
        print(f"$ {cmd}")
        run_cmd(cmd)
        return None

    # For local mode without GPU selection, run directly (no docker)
    if local and not gpus:
        print(f"\nRunning locally (no docker)")
        print(f"  Config: {config}")
        cmd = f"python entrypoints/train_grpo.py {config}"
        run_cmd(cmd)
        return None

    # Job name
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"annotations-adhoc-{timestamp}"

    # Stage and upload source
    source_dir = stage_source_dir(PROJECT_DIR)
    print(f"Staged source to: {source_dir}")
    s3_source_uri = upload_source_to_s3(source_dir, job_name)

    # Create training job via boto3
    sm_client = boto3.client("sagemaker", region_name=REGION)

    # Build hyperparameters based on launcher type
    if entrypoint:
        # Custom entrypoint script (e.g., for testing)
        hyperparams = {
            "sagemaker_program": entrypoint,
            "sagemaker_submit_directory": s3_source_uri,
        }
    elif use_combined:
        hyperparams = {
            "config": config,
            "eval-config": eval_config,
            "train-gpus": train_gpus,
            "eval-gpus": eval_gpus,
            "sagemaker_program": "entrypoints/sagemaker_combined_launcher.py",
            "sagemaker_submit_directory": s3_source_uri,
        }
    else:
        hyperparams = {
            "config": config,
            "sagemaker_program": "entrypoints/sagemaker_launcher.py",
            "sagemaker_submit_directory": s3_source_uri,
        }

    training_job_config = {
        "TrainingJobName": job_name,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": remote_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": hyperparams,
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 5 * 24 * 3600,  # 5 days
        },
        # VpcConfig removed to allow direct internet/S3 access
        "Environment": {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{BUCKET}/rlvr-experiments/output/",
        },
    }

    print(f"\nSubmitting job: {job_name}")
    print(f"  Image: {remote_uri}")
    print(f"  Instance: {instance_type} x {instance_count}")
    print(f"  Config: {config}")
    if use_combined:
        print(f"  Eval config: {eval_config}")
        print(f"  Train GPUs: {train_gpus}")
        print(f"  Eval GPUs: {eval_gpus}")

    sm_client.create_training_job(**training_job_config)

    print(f"\nJob submitted: {job_name}")
    print(f"View at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")

    if wait:
        print("\nWaiting for job to complete...")
        waiter = sm_client.get_waiter("training_job_completed_or_stopped")
        waiter.wait(TrainingJobName=job_name)
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        print(f"Job status: {response['TrainingJobStatus']}")

    return job_name


def main():
    parser = argparse.ArgumentParser(description="Submit GRPO training to SageMaker")
    parser.add_argument("config", nargs="?", default=None,
                        help="Training config file (e.g., configs/qwen3-17B-base-gsm8k.yaml)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--image-tag", default=None, help="Docker image tag (default: YYYYMMDD)")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--push", action="store_true", help="Push Docker image to ECR")
    parser.add_argument("--job-name", default=None, help="Job name (default: auto-generated)")
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--local", action="store_true", help="Run locally using Docker")
    parser.add_argument("--gpus", default=None, help="GPU IDs for local mode (e.g., '5,6,7')")
    parser.add_argument("--eval-config", default=None,
                        help="Eval config for combined training+eval mode")
    parser.add_argument("--train-gpus", default="0,1,2,3,4,5",
                        help="GPUs for training (default: 0,1,2,3,4,5)")
    parser.add_argument("--eval-gpus", default="6,7",
                        help="GPUs for evaluation (default: 6,7)")
    parser.add_argument("--entrypoint", default=None,
                        help="Custom entrypoint script (overrides config-based launcher)")

    args = parser.parse_args()

    submit_job(
        config=args.config,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        image_tag=args.image_tag,
        build=args.build,
        push=args.push,
        job_name=args.job_name,
        wait=args.wait,
        local=args.local,
        gpus=args.gpus,
        eval_config=args.eval_config,
        train_gpus=args.train_gpus,
        eval_gpus=args.eval_gpus,
        entrypoint=args.entrypoint,
    )


if __name__ == "__main__":
    main()
