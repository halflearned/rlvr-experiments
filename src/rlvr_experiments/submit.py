#!/usr/bin/env python3
"""Submit GRPO training jobs to SageMaker."""

import argparse
import boto3
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pathspec
from sagemaker import Session
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch

# AWS config
ACCOUNT = "503561457547"
REGION = "us-west-2"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/SageMaker"
ECR_PREFIX = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"

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
):
    """Submit a training job to SageMaker."""
    local_uri = f"{DEFAULT_IMAGE_NAME}:{image_tag or datetime.now().strftime('%Y%m%d')}"
    remote_uri = get_image_uri(tag=image_tag, build=build, push=push)

    # For local mode with specific GPUs, run docker directly (bypass SageMaker local mode)
    if local and gpus:
        print(f"\nRunning locally with GPUs: {gpus}")
        print(f"  Image: {local_uri}")
        print(f"  Config: {config}")

        # Run docker directly with GPU selection
        # Note: CUDA_VISIBLE_DEVICES inside container sees GPUs as 0,1,2... based on --gpus selection
        # Start Ray head inside container before running training
        # Use --ipc=host for shared memory, but NOT --network=host to avoid Ray port conflicts
        num_gpus = len(gpus.split(","))
        # cuDNN fix: the base image has cuDNN 9.1, but PyTorch 2.9+cu128 needs 9.8+
        # PyTorch bundles the correct cuDNN in nvidia/cudnn, so we prepend it to LD_LIBRARY_PATH
        nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
        cmd = (
            f"docker run --rm "
            f'--gpus \'"device={gpus}"\' '
            f"--ipc=host "
            f"-v /dev/shm:/dev/shm "
            f"-v {PROJECT_DIR}:/opt/ml/code "
            f"-v {PROJECT_DIR}:{PROJECT_DIR} "  # Also mount at original path for config compatibility
            f"-v {PROJECT_DIR}/training_output:/opt/ml/model "
            f"-e PYTORCH_ALLOC_CONF=expandable_segments:True "
            f"-e PYTHONPATH=/opt/ml/code/src "
            f"-e LD_LIBRARY_PATH={nvidia_libs} "
            f"-w /opt/ml/code "
            f"{local_uri} "
            f"bash -c 'ray start --head --num-gpus={num_gpus} && python entrypoints/train_grpo.py {config}'"
        )
        print(f"$ {cmd}")
        run_cmd(cmd)
        return None

    # Job name - sanitize to match SageMaker pattern: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
    if job_name is None:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        job_name = f"adhoc-{timestamp}"

    # Stage source
    source_dir = stage_source_dir(PROJECT_DIR)
    print(f"Staged source to: {source_dir}")

    # Create session
    if local:
        sm_session = LocalSession()
        image_uri = local_uri
        instance_type = "local_gpu"
        instance_count = 1
        output_path = f"file://{PROJECT_DIR}/training_output"
    else:
        boto_session = boto3.session.Session(region_name=REGION)
        sm_session = Session(boto_session=boto_session)
        image_uri = remote_uri
        output_path = None  # let SageMaker decide

    # Use launcher for multi-node, direct entrypoint for single-node
    if instance_count > 1:
        entry_point = "entrypoints/sagemaker_launcher.py"
    else:
        entry_point = "entrypoints/sagemaker_launcher.py"  # Use launcher for both (handles Ray init)

    # Create estimator
    estimator_kwargs = dict(
        image_uri=image_uri,
        role=ROLE,
        source_dir=source_dir,
        entry_point=entry_point,
        instance_type=instance_type,
        instance_count=instance_count,
        sagemaker_session=sm_session,
        output_path=output_path,
        max_run=5 * 24 * 3600,  # 5 days
        environment={
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        hyperparameters={
            "config": config,
        },
    )

    # Add VPC config for EFA networking (required for multi-node, optional for single)
    if not local:
        estimator_kwargs["subnets"] = SUBNETS
        estimator_kwargs["security_group_ids"] = SECURITY_GROUP_IDS

    estimator = PyTorch(**estimator_kwargs)

    # Submit
    print(f"\nSubmitting job: {job_name}")
    print(f"  Image: {image_uri}")
    print(f"  Instance: {instance_type} x {instance_count}")
    print(f"  Config: {config}")
    print(f"  Local: {local}")

    estimator.fit(job_name=job_name, wait=wait or local)

    if not local:
        print(f"\nJob submitted: {job_name}")
        print(f"View at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")

    return job_name


def main():
    parser = argparse.ArgumentParser(description="Submit GRPO training to SageMaker")
    parser.add_argument("config", help="Training config file (e.g., configs/qwen3-17B-base-gsm8k.yaml)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--image-tag", default=None, help="Docker image tag (default: YYYYMMDD)")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--push", action="store_true", help="Push Docker image to ECR")
    parser.add_argument("--job-name", default=None, help="Job name (default: auto-generated)")
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--local", action="store_true", help="Run locally using Docker")
    parser.add_argument("--gpus", default=None, help="GPU IDs for local mode (e.g., '5,6,7')")

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
    )


if __name__ == "__main__":
    main()
