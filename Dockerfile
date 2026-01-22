# Base: AWS Deep Learning Container with CUDA 12.4
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2

# Install uv for fast dependency resolution
RUN pip install uv

# Upgrade PyTorch to 2.9 (uses CUDA 12.8, but is backward compatible with 12.4 drivers)
# Also install matching cudnn/cublas/cusparse/cufft/curand/cusolver/nccl for CUDA 12.8
RUN uv pip install --system torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --system nvidia-cudnn-cu12==9.5.1.17 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12

# Install torchtitan (needed before other deps to avoid conflicts)
RUN uv pip install --system torchtitan==0.2.0

# Pin huggingface-hub and transformers first to avoid version conflicts with vllm
RUN uv pip install --system "huggingface-hub==0.36.0"
RUN uv pip install --system "transformers==4.57.3"
RUN uv pip install --system "vllm==0.11.2"

# Reinstall sagemaker-training toolkit (may have been removed by dependency upgrades)
RUN uv pip install --system sagemaker-training

# Copy and install from pyproject.toml (skipping torch/vllm/torchtitan already installed above)
COPY pyproject.toml README.md /tmp/pkg/
COPY src /tmp/pkg/src
WORKDIR /tmp/pkg
RUN uv pip install --system .

# SageMaker will copy source to /opt/ml/code and set PYTHONPATH
WORKDIR /opt/ml/code
