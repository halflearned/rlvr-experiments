# AGENTS.md

Notes for AI agents working on this codebase.

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

The `submit.py` script uses `adhoc-{timestamp}` format (e.g., `adhoc-0111-0644`) to avoid issues with config filenames containing periods (like `qwen3-1.7B-gsm8k.yaml`).

### Model Weights in S3

SageMaker instances don't have access to `/efs`. Model weights must be uploaded to S3 first:

```bash
# Upload model to S3
aws s3 sync /efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base \
    s3://sagemaker-us-west-2-503561457547/rlvr-experiments/models/Qwen3-1.7B-Base/
```

The `sagemaker_launcher.py` automatically downloads models from S3 at job start. It looks for HF-style paths in the config (e.g., `Qwen/Qwen3-1.7B-Base`) and maps them to S3:

- Config path: `Qwen/Qwen3-1.7B-Base`
- S3 path: `s3://sagemaker-us-west-2-503561457547/rlvr-experiments/models/Qwen3-1.7B-Base/`
- Local path (on SageMaker): `/opt/ml/model/model_cache/Qwen3-1.7B-Base`

### Common Issues

1. **ECR Repository Not Found**: The ECR repository `rlvr-experiments` must exist before pushing. The script attempts to create it automatically, but requires `ecr:CreateRepository` permission.

2. **Docker Image Build**: Use `--build` flag to rebuild the image. Use `--push` to push to ECR.

3. **Local Testing**: Use `--local --gpus "0,1,2,3"` to run in Docker locally before submitting to SageMaker.

4. **Model Not Found on SageMaker**: Ensure model weights are uploaded to S3 before submitting the job (see "Model Weights in S3" above).

### Example Commands

```bash
# Build and push image, then submit to SageMaker
python src/rlvr_experiments/submit.py configs/my-config.yaml --build --push

# Just submit (image already in ECR)
python src/rlvr_experiments/submit.py configs/my-config.yaml

# Test locally first
python src/rlvr_experiments/submit.py configs/my-config.yaml --local --gpus "0,1,2,3"
```
