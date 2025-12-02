FROM vllm/vllm-openai:v0.11.2

# install uv (needed for this package)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy project files
COPY pyproject.toml uv.lock ./

# install dependencies from pyproject.toml
RUN uv pip install --system --no-cache ".[inference]"


CMD ["sleep", "infinity"]

