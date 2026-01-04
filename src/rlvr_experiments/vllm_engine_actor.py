import asyncio
import logging
import time
import uuid
import torch
import ray

from typing import Any, Sequence
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.gpu_worker import Worker

from .syncing import WeightSyncManager
from .tracer import trace_span, get_tracer

logger = logging.getLogger(__name__)

# Metrics tracking for vLLM generation
_VLLM_METRICS_ENABLED = True

WORKER_CLS = "rlvr_experiments.vllm_engine_actor.WeightSyncVLLMWorker"


@ray.remote(num_gpus=0)
class VLLMEngineRank:
    def __init__(self, engine_kwargs: dict[str, Any], replica_id: int = 0) -> None:
        self.replica_id = replica_id
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs.pop("data_parallel_size", None)
        engine_args = AsyncEngineArgs(
            worker_cls=WORKER_CLS, distributed_executor_backend="ray", **engine_kwargs
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._active_requests: set[str] = set()
        # Accumulated metrics for periodic collection
        self._metrics_accum = {"output_tokens": 0, "prompt_tokens": 0, "elapsed_s": 0.0, "calls": 0}

    def ready(self) -> bool:
        return True

    async def add_sync_channel(self, *, channel_name: str, host: str, port: int, world_size: int, rank: int) -> None:
        await self.engine.collective_rpc(
            "add_sync_channel",
            kwargs={"channel_name": channel_name, "host": host, "port": port, "world_size": world_size, "rank": rank},
        )

    async def abort_all(self) -> int:
        """Abort all in-flight requests. Returns number of requests aborted."""
        to_abort = list(self._active_requests)
        for req_id in to_abort:
            await self.engine.abort(req_id)
        self._active_requests.clear()
        if to_abort:
            logger.info(f"[replica={self.replica_id}] aborted {len(to_abort)} requests")
        return len(to_abort)

    async def generate(self, prompts: Sequence[str], **sampling_params):
        t0 = time.perf_counter()

        sp = SamplingParams(**sampling_params)
        if sp.output_kind is None:
            sp.output_kind = RequestOutputKind.FINAL_ONLY

        async def gen_single(prompt):
            req_id = str(uuid.uuid4())
            self._active_requests.add(req_id)
            try:
                final = None
                async for out in self.engine.generate(prompt, sp.clone(), req_id):
                    final = out if final is None else final.add(out, aggregate=False) or final
                return final
            finally:
                self._active_requests.discard(req_id)

        results = await asyncio.gather(*[gen_single(p) for p in prompts])

        if _VLLM_METRICS_ENABLED:
            elapsed = time.perf_counter() - t0
            # Count total output tokens across all completions
            total_output_tokens = sum(
                len(out.token_ids)
                for r in results
                for out in r.outputs
            )
            # Count prompt tokens
            total_prompt_tokens = sum(len(r.prompt_token_ids) for r in results)
            n_completions = sum(len(r.outputs) for r in results)

            gen_tps = total_output_tokens / elapsed if elapsed > 0 else 0
            print(
                f"[vLLM METRICS replica={self.replica_id}] "
                f"prompts={len(prompts)}  "
                f"completions={n_completions}  "
                f"prompt_tokens={total_prompt_tokens}  "
                f"output_tokens={total_output_tokens}  "
                f"time={elapsed:.2f}s  "
                f"gen_tps={gen_tps:.0f}",
                flush=True
            )

            # Accumulate for periodic collection
            self._metrics_accum["output_tokens"] += total_output_tokens
            self._metrics_accum["prompt_tokens"] += total_prompt_tokens
            self._metrics_accum["elapsed_s"] += elapsed
            self._metrics_accum["calls"] += 1

        return results

    async def recv_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int):
        # channel is unused - vLLM only has one sync channel
        await self.engine.collective_rpc(
            "recv_chunk", kwargs={"chunk": chunk, "dtype_str": dtype_str, "src_rank": src_rank}
        )

    async def log_stats(self):
        """Trigger vLLM's built-in stats logging."""
        await self.engine.do_log_stats()

    def get_metrics(self) -> dict:
        """Get accumulated metrics and reset counters."""
        metrics = dict(self._metrics_accum)
        self._metrics_accum = {"output_tokens": 0, "prompt_tokens": 0, "elapsed_s": 0.0, "calls": 0}
        # Compute gen_tps from accumulated data
        if metrics["elapsed_s"] > 0:
            metrics["gen_tps"] = metrics["output_tokens"] / metrics["elapsed_s"]
        else:
            metrics["gen_tps"] = 0
        return metrics


class VLLMHandle:
    """Handle for one or more vLLM instances (data parallel replicas)."""

    def __init__(self, actors: list, name: str = "vllm"):
        self._actors = actors
        self._current_replica = 0
        self._in_flight = 0
        self._in_flight_zero = asyncio.Event()
        self._in_flight_zero.set()  # Initially no in-flight requests
        self._paused = asyncio.Event()
        self._paused.set()  # Initially not paused (gate is open)
        self.name = name
        self._model_version = 0  # Increments on each weight sync

    @property
    def model_version(self) -> int:
        """Current model version. Increments after each weight sync."""
        return self._model_version

    def increment_version(self) -> int:
        """Increment model version after weight sync. Returns new version."""
        self._model_version += 1
        return self._model_version

    @property
    def num_replicas(self) -> int:
        return len(self._actors)

    async def abort_all(self) -> int:
        """Abort all in-flight requests on all replicas. Returns total aborted."""
        results = await asyncio.gather(*[a.abort_all.remote() for a in self._actors])
        return sum(results)

    async def stop(self, abort: bool = False) -> None:
        """Stop accepting new generation requests.

        Args:
            abort: If True, abort in-flight requests instead of waiting for them.
        """
        self._paused.clear()  # Close the gate for new requests
        if abort:
            await self.abort_all()
        await self._in_flight_zero.wait()  # Wait for in-flight requests to finish

    def resume(self) -> None:
        """Resume accepting generation requests after sync."""
        self._paused.set()  # Open the gate

    async def generate(self, prompts, **sampling_params):
        # Wait if paused (will block until resume() is called)
        if not self._paused.is_set():
            await self._paused.wait()

        # Track in-flight requests
        self._in_flight += 1
        if self._in_flight == 1:
            self._in_flight_zero.clear()

        try:
            replica_idx = self._current_replica % len(self._actors)
            actor = self._actors[replica_idx]
            self._current_replica += 1

            with trace_span("vllm.generate", args={"num_prompts": len(prompts), "replica": replica_idx}):
                return await actor.generate.remote(prompts, **sampling_params)
        finally:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._in_flight_zero.set()

    async def log_stats(self):
        """Trigger vLLM's built-in stats logging on all replicas."""
        await asyncio.gather(*[a.log_stats.remote() for a in self._actors])

    async def get_metrics(self) -> dict:
        """Get aggregated metrics from all replicas and reset their counters."""
        results = await asyncio.gather(*[a.get_metrics.remote() for a in self._actors])
        # Aggregate across replicas
        total = {"output_tokens": 0, "prompt_tokens": 0, "elapsed_s": 0.0, "calls": 0, "gen_tps": 0}
        for m in results:
            total["output_tokens"] += m["output_tokens"]
            total["prompt_tokens"] += m["prompt_tokens"]
            total["elapsed_s"] += m["elapsed_s"]
            total["calls"] += m["calls"]
            total["gen_tps"] += m["gen_tps"]
        # Average gen_tps across replicas that had activity
        active_replicas = sum(1 for m in results if m["calls"] > 0)
        if active_replicas > 0:
            total["gen_tps"] = total["gen_tps"] / active_replicas
        return total


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
