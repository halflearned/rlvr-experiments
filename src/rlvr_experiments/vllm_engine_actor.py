import asyncio
import uuid
import torch
import ray

from typing import Any, AsyncIterator, Callable, Sequence
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.gpu_worker import Worker

from .weight_sync import WeightSyncManager
from .tracer import traced, trace_span

WORKER_CLS = "rlvr_experiments.vllm_engine_actor.WeightSyncVLLMWorker"


@ray.remote(num_gpus=0)
class VLLMEngineRank:
    def __init__(self, engine_kwargs: dict[str, Any]) -> None:
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs.pop("data_parallel_size", None)
        engine_args = AsyncEngineArgs(
            worker_cls=WORKER_CLS, distributed_executor_backend="ray", **engine_kwargs
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def ready(self) -> bool:
        return True

    async def add_sync_channel(self, *, channel_name: str, host: str, port: int, world_size: int, rank: int) -> None:
        await self.engine.collective_rpc(
            "add_sync_channel",
            kwargs={"channel_name": channel_name, "host": host, "port": port, "world_size": world_size, "rank": rank},
        )

    async def generate(self, prompts: Sequence[str], **sampling_params):
        sp = SamplingParams(**sampling_params)
        if sp.output_kind is None:
            sp.output_kind = RequestOutputKind.FINAL_ONLY

        async def gen_single(prompt):
            final = None
            async for out in self.engine.generate(prompt, sp.clone(), str(uuid.uuid4())):
                final = out if final is None else final.add(out, aggregate=False) or final
            return final

        return await asyncio.gather(*[gen_single(p) for p in prompts])

    async def recv_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int):
        # channel is unused - vLLM only has one sync channel
        await self.engine.collective_rpc(
            "recv_chunk", kwargs={"chunk": chunk, "dtype_str": dtype_str, "src_rank": src_rank}
        )


class VLLMHandle:
    """Handle for one or more vLLM instances (data parallel replicas).

    Manages GPU generation and producer lifecycle for rollout collection.

    Usage:
        async for samples in rollout.run_epoch(producer_fn, buffer, batch_size=8, ...):
            train_on(samples)
    """

    def __init__(self, actors: list, name: str = "vllm"):
        self._actors = actors
        self.name = name
        self._stop_event = asyncio.Event()
        self._current_replica = 0
        self._producer_tasks: list[asyncio.Task] = []

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @traced("vllm.generate")
    async def generate(self, prompts, **sampling_params):
        actor = self._actors[self._current_replica % len(self._actors)]
        self._current_replica += 1
        return await actor.generate.remote(prompts, **sampling_params)

    # ---- Producer lifecycle ----

    def _producers_done(self) -> bool:
        return len(self._producer_tasks) > 0 and all(t.done() for t in self._producer_tasks)

    def _check_producer_errors(self) -> None:
        for i, t in enumerate(self._producer_tasks):
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    raise RuntimeError(f"Producer {i} failed") from exc

    async def run_epoch(
        self,
        data_iter,
        buffer,
        *,
        reward: Callable,
        pad_token_id: int,
        batch_size: int,
        sampling_params: dict | None = None,
        epoch: int = 0,
    ):
        """Run generation/verification and yield training batches.

        Users provide one callback:
        - reward(problem, completions) -> list[float]: compute rewards

        Example:
            async for step, batch in rollout.run_epoch(
                data_iter, buffer,
                reward=verifier.verify_completions,
                pad_token_id=pad_token_id,
                batch_size=8,
                sampling_params=plan.sampling,
                epoch=epoch,
            ):
                ref_logprobs = await reference.compute_logprobs(batch.input_ids, batch.completion_ids)
                loss = await trainer.forward_backward(...)

        Args:
            data_iter: DataIterator with .next_batch() returning {"templates": [...], "problems": [...]}
            buffer: RolloutBuffer for producer/consumer coordination
            reward: async (problem, completions) -> list[float]
            pad_token_id: For creating RolloutSample tensors
            batch_size: Number of samples per training batch
            sampling_params: vLLM sampling parameters (temperature, top_p, n, etc.)
            epoch: Current epoch (for buffer versioning)

        Yields:
            (step, batch) tuples where step is 1-indexed and batch has input_ids,
            completion_ids, logprobs, advantages, mask, avg_reward.
            Zero-variance batches are filtered internally.
        """
        from .batch import RolloutSample, make_batch
        from .tracer import get_tracer

        tracer = get_tracer()

        # Always request logprobs (required for GRPO)
        sp = {**(sampling_params or {}), "logprobs": 0}

        async def producer(rollout, buffer):
            """Internal producer - handles TaskGroup and buffer.put."""
            async def process_one(response, problem):
                completions = [out.text for out in response.outputs]
                rewards = await reward(problem, completions)
                sample = RolloutSample.from_vllm(response, pad_token_id, rewards)
                await buffer.put(sample, epoch)

            async with asyncio.TaskGroup() as tasks:
                while not rollout.is_stopped():
                    data_batch = await data_iter.next_batch()
                    if data_batch is None:
                        break
                    responses = await rollout.generate(data_batch["templates"], **sp)
                    for response, problem in zip(responses, data_batch["problems"]):
                        tasks.create_task(process_one(response, problem))

        # Run the internal producer and yield batches
        self._stop_event.clear()
        self._producer_tasks = [
            asyncio.create_task(producer(self, buffer))
            for _ in self._actors
        ]

        step = 0
        try:
            while True:
                with trace_span("wait_for_batch"):
                    self._check_producer_errors()

                    if self._producers_done() and buffer.size() == 0:
                        return

                    samples = []
                    while len(samples) < batch_size:
                        self._check_producer_errors()
                        if self._producers_done() and buffer.size() == 0:
                            break
                        timeout = 10.0 if buffer.size() == 0 else 5.0
                        popped = await buffer.pop_batch(batch_size - len(samples), min_version=epoch, timeout=timeout)
                        samples.extend(popped)

                if not samples:
                    return

                # Log reward stats
                all_rewards = [r for s in samples for r in s.rewards]
                tracer.counter("rewards", {
                    "mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
                    "max": max(all_rewards) if all_rewards else 0,
                    "num_positive": sum(1 for r in all_rewards if r > 0),
                })

                with trace_span("make_batch"):
                    batch = make_batch(samples, pad_token_id)

                if batch is None:
                    tracer.counter("skipped", {"zero_variance_batches": 1})
                    continue

                step += 1
                yield step, batch
        finally:
            self._stop_event.set()
            for t in self._producer_tasks:
                t.cancel()
            await asyncio.gather(*self._producer_tasks, return_exceptions=True)
            self._producer_tasks = []



class WeightSyncVLLMWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_sync: WeightSyncManager | None = None

    def add_sync_channel(self, channel_name: str, host: str, port: int, world_size: int, rank: int):
        sync_rank = rank + self.rank
        self.weight_sync = WeightSyncManager(
            host=host, port=port, world_size=world_size, rank=sync_rank, device=self.device
        )

    def recv_chunk(self, chunk: dict, dtype_str: str, src_rank: int):
        """Receive a chunk via NCCL broadcast and load weights into vLLM model."""
        dtype = getattr(torch, dtype_str)
        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=self.device)
        self.weight_sync.communicator.broadcast(flat, src=src_rank)

        offset = 0
        for p in chunk["params"]:
            weight = flat[offset : offset + p["numel"]].view(p["shape"])
            offset += p["numel"]
            self.model_runner.model.load_weights(weights=[(p["name"], weight)])
