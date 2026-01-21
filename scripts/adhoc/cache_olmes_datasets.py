#!/usr/bin/env python3
"""Cache OLMES datasets locally and optionally upload to S3.

Includes IFBench_test for ifeval_ood tasks.

Usage:
  HF_HOME=/tmp/hf_cache_olmes python scripts/adhoc/cache_olmes_datasets.py \
    --s3-uri s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/hf_cache/olmes/
"""

from __future__ import annotations

import argparse
import os
import subprocess

from datasets import load_dataset


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def cache_gsm8k() -> None:
    load_dataset("gsm8k", "main", split="train")
    load_dataset("gsm8k", "main", split="test")


def cache_minerva_math() -> None:
    for subject in MATH_SUBJECTS:
        load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        load_dataset("EleutherAI/hendrycks_math", subject, split="test")


def cache_ifeval() -> None:
    load_dataset("HuggingFaceH4/ifeval", split="train")


def cache_ifeval_ood() -> None:
    load_dataset("allenai/IFBench_test", split="train")


def sync_to_s3(local_dir: str, s3_uri: str) -> None:
    print(f"Syncing {local_dir} -> {s3_uri}")
    subprocess.run(["aws", "s3", "sync", local_dir, s3_uri, "--quiet"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache OLMES datasets to HF_HOME")
    parser.add_argument("--s3-uri", default=None, help="Optional S3 prefix to sync cache")
    args = parser.parse_args()

    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        raise SystemExit("HF_HOME must be set to a writable cache directory")

    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))

    print(f"HF_HOME={cache_dir}")
    cache_gsm8k()
    cache_minerva_math()
    cache_ifeval()
    cache_ifeval_ood()

    if args.s3_uri:
        sync_to_s3(cache_dir, args.s3_uri)


if __name__ == "__main__":
    main()
