# Base: AWS Deep Learning Container with CUDA 12.4
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2

# Install uv for fast dependency resolution
RUN pip install uv

# Upgrade PyTorch to 2.9 (uses CUDA 12.8, but is backward compatible with 12.4 drivers)
RUN uv pip install --system torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install torchtitan 0.2.0 (latest stable, compatible with torch 2.9)
RUN uv pip install --system torchtitan==0.2.0

# Install main dependencies with specific versions matching our working venv
# vllm 0.11.2 needs transformers>=4.50, and transformers 4.57.3 works with huggingface_hub<1.0
RUN uv pip install --system "ray[default]>=2.51.2" \
    "pyyaml>=6.0.3" "fastapi" "uvicorn" "tomli-w>=1.2.0" "lm-eval>=0.4.9.2"

# Pin huggingface-hub first, then install transformers and vllm
RUN uv pip install --system "huggingface-hub==0.36.0"
RUN uv pip install --system "transformers==4.57.3"
RUN uv pip install --system "vllm==0.11.2"

# SageMaker will copy source to /opt/ml/code and set PYTHONPATH
WORKDIR /opt/ml/code
