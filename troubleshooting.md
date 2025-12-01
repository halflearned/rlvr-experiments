## Troubleshooting

Logging installation issues, etc.


### tiktoken

Package `vllm` will require `tiktoken`, which in turn requires a rust compiler for install. 
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# follow the prompts, usually just press Enter

# then in the same shell:
source $HOME/.cargo/env
```


### xformers

Make sure gcc is compatible with our CUDA version (currently: 12.4).

```bash
# Install GCC 11 or 12 from conda-forge (these work with CUDA 12.4)
conda install -c conda-forge gcc_linux-64=11.4.0 gxx_linux-64=11.4.0

# Then set these environment variables before running uv
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="8.6"

# Now run uv
uv sync
```