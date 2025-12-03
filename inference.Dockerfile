FROM vllm/vllm-openai:v0.11.2

# install uv (needed for this package)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy project files
# not copying uv.lock as we want to make sure not to overwrite the torch version
COPY pyproject.toml ./

# install dependencies from pyproject.toml
RUN uv pip install --system --no-cache ".[inference]"

RUN uv pip install --system trl

CMD ["sleep", "infinity"]

