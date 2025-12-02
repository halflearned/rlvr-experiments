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

# install uv (needed for this package)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy project files
COPY pyproject.toml uv.lock ./

# install dependencies from pyproject.toml
RUN uv pip install --system --no-cache -r pyproject.toml

# dummy command, this will keep the container running. docker compose overrides this.
CMD ["sleep", "infinity"]