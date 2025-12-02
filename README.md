# RVLR experiments

## Overview (2025-12-01)

_This documentation will be updated more or less daily!_

Assumptions: we have `docker compose` (v2) and access to a machine with 4+ GPUs.

To quickly see the code in action, build the docker image and run it:
```bash
docker build . -t rlvr-experiments:latest
docker compose up
```

This will run two separate processes: 

* On gpus 0,1: A vLLM inference engine that loads Qwen3-0.6B and is ready to run inference. This will be used to produce rollouts.
* On gpus 2,3: A torchtitan-based training script that loads Qwen3-0.6B and runs a quick forward-backward smoke test (right now, just tensor parallelism is enabled, with tp=2)

Next up:

* Integrate the two processes, of course. The trainer periodically saves checkpoints that are hot-loaded by the inference engine.
* Add dataloaders, using gsm8k as the first test dataset
* Add reward checkers
* Add the actual policy learning code, likely GRPO to start
* Add compute performance metrics


## Notes

* See [troubleshooting.md](./troubleshooting.md) for common install issues.


## TODOs

* vLLM inference engine seems to segfault unless we set `enforce_eager = true` and `tensor_parallel = 1`