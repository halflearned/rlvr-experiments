#!/usr/bin/env python3
"""Submit evaluation jobs to SageMaker.

Uses boto3 directly to avoid sagemaker SDK's omegaconf dependency which
conflicts with antlr4 (required by lm-eval).

Example:
    # Submit to SageMaker (8x L40S on g6e.48xlarge)
    python -m rlvr_experiments.submit_eval configs/eval/hadadv-adhoc-mixed.yaml

    # Run locally with specific GPUs
    python -m rlvr_experiments.submit_eval configs/eval/hadadv-adhoc-mixed.yaml --local --gpus 0,1,2,3,4,5,6,7

    # Use smaller instance (4x L40S on g6e.24xlarge)
    python -m rlvr_experiments.submit_eval configs/eval/example.yaml --instance-type ml.g6e.24xlarge
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

# VPC config
SUBNETS = ["subnet-08b78e8d13faebd65"]
SECURITY_GROUP_IDS = ["sg-0448b04b00c40d716"]

# Defaults
DEFAULT_INSTANCE_TYPE = "ml.g6e.48xlarge"  # 8x L40S (48GB each)
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


def get_image_uri(tag: str | None = None) -> str:
    """Get Docker image URI."""
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d")
    return f"{ECR_PREFIX}/{DEFAULT_IMAGE_NAME}:{tag}"


def submit_eval(
    config: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    image_tag: str | None = None,
    job_name: str | None = None,
    wait: bool = False,
    local: bool = False,
    gpus: str | None = None,
):
    """Submit an evaluation job to SageMaker."""
    local_uri = f"{DEFAULT_IMAGE_NAME}:{image_tag or datetime.now().strftime('%Y%m%d')}"
    remote_uri = get_image_uri(tag=image_tag)

    # For local mode with specific GPUs, run docker directly
    if local and gpus:
        print(f"\nRunning locally with GPUs: {gpus}")
        print(f"  Image: {local_uri}")
        print(f"  Config: {config}")

        nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
        cmd = (
            f"docker run --rm "
            f'--gpus \'"device={gpus}"\' '
            f"--ipc=host "
            f"-v /dev/shm:/dev/shm "
            f"-v {PROJECT_DIR}:/opt/ml/code "
            f"-v {PROJECT_DIR}:{PROJECT_DIR} "
            f"-v {PROJECT_DIR}/eval_output:/opt/ml/model "
            f"-e PYTHONPATH=/opt/ml/code/src "
            f"-e LD_LIBRARY_PATH={nvidia_libs} "
            f"-e HF_ALLOW_CODE_EVAL=1 "
            f"-w /opt/ml/code "
            f"{local_uri} "
            f"python entrypoints/eval_benchmarks.py {config}"
        )
        print(f"$ {cmd}")
        run_cmd(cmd)
        return None

    # For local mode without GPU selection, just run directly (no docker)
    if local and not gpus:
        print(f"\nRunning locally (no docker)")
        print(f"  Config: {config}")
        cmd = f"python entrypoints/eval_benchmarks.py {config}"
        run_cmd(cmd)
        return None

    # Job name
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Extract config name for job name
        config_name = Path(config).stem.replace("_", "-")
        job_name = f"eval-{config_name}-{timestamp}"

    # Stage and upload source
    source_dir = stage_source_dir(PROJECT_DIR)
    print(f"Staged source to: {source_dir}")
    s3_source_uri = upload_source_to_s3(source_dir, job_name)

    # Create training job via boto3
    sm_client = boto3.client("sagemaker", region_name=REGION)

    training_job_config = {
        "TrainingJobName": job_name,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": remote_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "config": config,
            "sagemaker_program": "entrypoints/sagemaker_eval_launcher.py",
            "sagemaker_submit_directory": s3_source_uri,
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 24 * 3600,  # 24 hours max
        },
        "VpcConfig": {
            "Subnets": SUBNETS,
            "SecurityGroupIds": SECURITY_GROUP_IDS,
        },
        "Environment": {
            "HF_ALLOW_CODE_EVAL": "1",
        },
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{BUCKET}/rlvr-experiments/eval_output/",
        },
    }

    print(f"\nSubmitting eval job: {job_name}")
    print(f"  Image: {remote_uri}")
    print(f"  Instance: {instance_type}")
    print(f"  Config: {config}")

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
    parser = argparse.ArgumentParser(description="Submit evaluation job to SageMaker")
    parser.add_argument("config", help="Eval config file (e.g., configs/eval/example.yaml)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"Instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--image-tag", default=None, help="Docker image tag (default: YYYYMMDD)")
    parser.add_argument("--job-name", default=None, help="Job name (default: auto-generated)")
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--local", action="store_true", help="Run locally")
    parser.add_argument("--gpus", default=None, help="GPU IDs for local mode (e.g., '0,1,2,3')")

    args = parser.parse_args()

    submit_eval(
        config=args.config,
        instance_type=args.instance_type,
        image_tag=args.image_tag,
        job_name=args.job_name,
        wait=args.wait,
        local=args.local,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
