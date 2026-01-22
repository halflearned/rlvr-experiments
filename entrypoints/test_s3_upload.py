#!/usr/bin/env python3
"""Minimal S3 upload test script for SageMaker debugging.

This script:
1. Creates a test file
2. Tries to upload it to S3 using boto3
3. Reports success or failure

Run this on SageMaker to verify S3 connectivity without VPC restrictions.
"""

import os
import sys
import time
import tempfile

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("S3 UPLOAD TEST SCRIPT")
    log("=" * 60)

    # Print environment info
    log(f"Python: {sys.version}")
    log(f"CWD: {os.getcwd()}")
    log(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', 'NOT SET')}")
    log(f"SM_TRAINING_ENV: {os.environ.get('SM_TRAINING_ENV', 'NOT SET')[:200]}...")

    # Import boto3
    log("Importing boto3...")
    try:
        import boto3
        from botocore.config import Config
        log(f"  boto3 version: {boto3.__version__}")
    except Exception as e:
        log(f"  FAILED to import boto3: {e}")
        return 1

    # Create test file
    log("Creating test file...")
    test_content = f"Test file created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    test_content += f"Host: {os.uname().nodename}\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    log(f"  Created: {test_file} ({len(test_content)} bytes)")

    # S3 config
    bucket = "sagemaker-us-west-2-503561457547"
    s3_key = f"rlvr-experiments/test-uploads/test_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # Try to upload
    log(f"Uploading to s3://{bucket}/{s3_key}...")
    try:
        config = Config(connect_timeout=30, read_timeout=60, retries={'max_attempts': 3})
        s3 = boto3.client('s3', config=config)

        t0 = time.time()
        s3.upload_file(test_file, bucket, s3_key)
        elapsed = time.time() - t0

        log(f"  SUCCESS! Upload completed in {elapsed:.2f}s")
        log(f"  Uploaded to: s3://{bucket}/{s3_key}")
    except Exception as e:
        log(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        os.unlink(test_file)

    # Try to list the uploaded file to verify
    log("Verifying upload by listing...")
    try:
        response = s3.head_object(Bucket=bucket, Key=s3_key)
        log(f"  Verified: {response['ContentLength']} bytes, ETag={response['ETag']}")
    except Exception as e:
        log(f"  Verification failed: {e}")

    log("=" * 60)
    log("TEST COMPLETE - SUCCESS")
    log("=" * 60)

    # Keep running for a bit so we can see logs
    log("Sleeping 30s before exit...")
    time.sleep(30)

    return 0

if __name__ == "__main__":
    sys.exit(main())
