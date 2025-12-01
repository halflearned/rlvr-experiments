FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# add symlink
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install rust (needed for torchtitan -> tiktoken)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv (needed for this package)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies from pyproject.toml
RUN uv pip install --system --no-cache -r pyproject.toml

# Build flash-attn from source
ENV MAX_JOBS=32
RUN uv pip install --system --no-cache --no-build-isolation flash-attn

# Run the launch script
CMD ["sleep", "infinity"]