#!/bin/bash
# Download and unpack SageMaker job results
#
# Usage: ./scripts/download_job_results.sh <job_name>
#
# Downloads model.tar.gz from S3, unpacks it, and flattens the directory
# structure so that config.yaml, traces/, and checkpoints/ are directly
# under results/<job_name>/

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <job_name>"
    echo "Example: $0 annotations-adhoc-20260125-192905"
    exit 1
fi

JOB_NAME="$1"
S3_PATH="s3://sagemaker-us-west-2-503561457547/rlvr-experiments/output/${JOB_NAME}/output/model.tar.gz"
RESULTS_DIR="results/${JOB_NAME}"
TMP_FILE="/tmp/model_${JOB_NAME}.tar.gz"

echo "Downloading ${S3_PATH}..."
aws s3 cp "$S3_PATH" "$TMP_FILE"

echo "Removing existing ${RESULTS_DIR} if present..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "Extracting to ${RESULTS_DIR}..."
tar -xzf "$TMP_FILE" -C "$RESULTS_DIR" 2>/dev/null || tar -xf "$TMP_FILE" -C "$RESULTS_DIR"

# Find and flatten the run subfolder (e.g., config_rewritten_20260125-193842/)
# Look for a directory containing config.yaml
RUN_DIR=$(find "$RESULTS_DIR" -maxdepth 2 -name "config.yaml" -type f | head -1 | xargs dirname)

if [ -n "$RUN_DIR" ] && [ "$RUN_DIR" != "$RESULTS_DIR" ]; then
    echo "Flattening ${RUN_DIR} -> ${RESULTS_DIR}..."
    mv "$RUN_DIR"/* "$RESULTS_DIR"/
    rmdir "$RUN_DIR" 2>/dev/null || true
fi

# Clean up temp file
rm -f "$TMP_FILE"

echo ""
echo "Done. Contents of ${RESULTS_DIR}:"
ls -la "$RESULTS_DIR"
