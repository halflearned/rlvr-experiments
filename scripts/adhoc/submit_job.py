#!/usr/bin/env python3
"""Minimal SageMaker job submission script that avoids omegaconf import issue."""

import boto3
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pathspec

# AWS config
ACCOUNT = "503561457547"
REGION = "us-west-2"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/SageMaker"
ECR_PREFIX = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"

# VPC config (EFA-enabled)
SUBNETS = ["subnet-08b78e8d13faebd65"]
SECURITY_GROUP_IDS = ["sg-0448b04b00c40d716"]

PROJECT_DIR = Path(__file__).resolve().parents[2]


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--instance-type", default="ml.p4de.24xlarge")
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--job-name", default=None)
    args = parser.parse_args()

    image_tag = args.image_tag or datetime.now().strftime("%Y%m%d")
    image_uri = f"{ECR_PREFIX}/rlvr-experiments:{image_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"annotations-adhoc-{timestamp}"

    # Stage source
    source_dir = stage_source_dir(PROJECT_DIR)
    print(f"Staged source to: {source_dir}")

    # Upload source to S3
    s3_client = boto3.client("s3", region_name=REGION)
    sm_client = boto3.client("sagemaker", region_name=REGION)

    bucket = f"sagemaker-{REGION}-{ACCOUNT}"
    s3_prefix = f"rlvr-experiments/source/{job_name}"

    # Create tarball and upload
    import tarfile
    tarball_path = f"/tmp/{job_name}-source.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(source_dir, arcname="")

    s3_key = f"{s3_prefix}/sourcedir.tar.gz"
    print(f"Uploading source to s3://{bucket}/{s3_key}")
    s3_client.upload_file(tarball_path, bucket, s3_key)

    # Create training job
    training_job_config = {
        "TrainingJobName": job_name,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "config": args.config,
            "sagemaker_program": "entrypoints/sagemaker_launcher.py",
            "sagemaker_submit_directory": f"s3://{bucket}/{s3_key}",
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 5 * 24 * 3600,  # 5 days
        },
        "VpcConfig": {
            "Subnets": SUBNETS,
            "SecurityGroupIds": SECURITY_GROUP_IDS,
        },
        "Environment": {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/rlvr-experiments/output/",
        },
    }

    print(f"\nSubmitting job: {job_name}")
    print(f"  Image: {image_uri}")
    print(f"  Instance: {args.instance_type}")
    print(f"  Config: {args.config}")

    sm_client.create_training_job(**training_job_config)

    print(f"\nJob submitted: {job_name}")
    print(f"View at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")


if __name__ == "__main__":
    main()
