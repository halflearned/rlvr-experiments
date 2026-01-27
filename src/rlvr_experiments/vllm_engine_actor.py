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


class AbortedError(Exception):
    """Raised when a generation request was aborted due to weight sync."""
    pass


# Metrics tracking for vLLM generation
_VLLM_METRICS_ENABLED = True

WORKER_CLS = "rlvr_experiments.vllm_engine_actor.WeightSyncVLLMWorker"


@ray.remote(num_gpus=0, max_concurrency=100)
class VLLMEngineRank:
    def __init__(self, engine_kwargs: dict[str, Any], replica_id: int = 0) -> None:
        self.replica_id = replica_id
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs.pop("data_parallel_size", None)
        engine_kwargs.pop("max_concurrent_per_replica", None)
        engine_args = AsyncEngineArgs(
            worker_cls=WORKER_CLS, distributed_executor_backend="ray", **engine_kwargs
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._active_requests: set[str] = set()
        # Accumulated metrics for periodic collection
        self._metrics_accum = {"output_tokens": 0, "prompt_tokens": 0, "calls": 0}
        # Rolling window for tok/s calculation (last 30s)
        self._token_history: list[tuple[float, int]] = []  # (timestamp, output_tokens)
        self._rolling_window_s = 30.0

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
        t1 = time.perf_counter()

        if _VLLM_METRICS_ENABLED:
            elapsed = t1 - t0
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
            self._metrics_accum["calls"] += 1
            # Add to rolling window
            self._token_history.append((t1, total_output_tokens))

        return results

    async def get_logprobs(
        self,
        token_ids_list: list[list[int]],
        prompt_lens: list[int],
        temperature: float = 1.0,
    ) -> list[list[float]]:
        """Get logprobs for pre-tokenized sequences (for reference model use).

        This is used for computing reference model logprobs efficiently. Instead of
        running a full forward pass through a Titan model, we leverage vLLM's
        prompt_logprobs feature with max_tokens=0.

        Args:
            token_ids_list: List of token ID sequences (prompt + completion).
            prompt_lens: Length of the prompt portion for each sequence.
                        Logprobs are returned only for tokens after this position.

        Returns:
            List of logprob lists, one per sequence. Each inner list contains
            the logprob of each completion token (length = seq_len - prompt_len).
        """
        t0 = time.perf_counter()

        # Use prompt_logprobs to get logprobs for input tokens, max_tokens=0 for no generation
        sp = SamplingParams(
            max_tokens=1,  # vLLM requires at least 1, but we ignore the output
            prompt_logprobs=1,  # Get logprob of each prompt token
            temperature=temperature,
        )
        sp.output_kind = RequestOutputKind.FINAL_ONLY

        async def process_single(token_ids: list[int], prompt_len: int) -> list[float]:
            req_id = str(uuid.uuid4())
            self._active_requests.add(req_id)
            try:
                # Convert token_ids to a TokensPrompt for vLLM
                from vllm.inputs import TokensPrompt
                prompt = TokensPrompt(prompt_token_ids=token_ids)

                final = None
                async for out in self.engine.generate(prompt, sp.clone(), req_id):
                    final = out if final is None else final.add(out, aggregate=False) or final

                # Extract logprobs for completion tokens (after prompt_len)
                # prompt_logprobs is a list of dicts, one per token position
                # Each dict maps token_id -> Logprob object
                if final is None or final.prompt_logprobs is None:
                    # Fallback: return zeros if no logprobs available
                    return [0.0] * (len(token_ids) - prompt_len)

                completion_logprobs = []
                # prompt_logprobs[i] contains logprob info for token at position i
                # We want logprobs for positions [prompt_len, len(token_ids))
                # The logprob at position i is P(token[i] | token[0:i])
                for i in range(prompt_len, len(token_ids)):
                    if i < len(final.prompt_logprobs) and final.prompt_logprobs[i] is not None:
                        token_id = token_ids[i]
                        lp_dict = final.prompt_logprobs[i]
                        if token_id in lp_dict:
                            completion_logprobs.append(lp_dict[token_id].logprob)
                        else:
                            # Token not in top-k, use a small default
                            completion_logprobs.append(-100.0)
                    else:
                        completion_logprobs.append(-100.0)

                return completion_logprobs
            finally:
                self._active_requests.discard(req_id)

        results = await asyncio.gather(*[
            process_single(tids, plen)
            for tids, plen in zip(token_ids_list, prompt_lens)
        ])

        t1 = time.perf_counter()
        total_tokens = sum(len(tids) for tids in token_ids_list)
        print(
            f"[vLLM LOGPROBS replica={self.replica_id}] "
            f"seqs={len(token_ids_list)}  "
            f"total_tokens={total_tokens}  "
            f"time={t1-t0:.2f}s  "
            f"tps={total_tokens/(t1-t0):.0f}",
            flush=True
        )

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
        now = time.perf_counter()
        cutoff = now - self._rolling_window_s

        # Prune old entries and compute rolling average
        self._token_history = [(ts, toks) for ts, toks in self._token_history if ts > cutoff]

        if self._token_history:
            total_tokens = sum(toks for _, toks in self._token_history)
            oldest_ts = self._token_history[0][0]
            window_duration = now - oldest_ts
            gen_tps = total_tokens / window_duration if window_duration > 0 else 0
        else:
            gen_tps = 0

        metrics = dict(self._metrics_accum)
        metrics["gen_tps"] = gen_tps

        # Reset accumulators (but keep rolling window)
        self._metrics_accum = {"output_tokens": 0, "prompt_tokens": 0, "calls": 0}

        return metrics


class LoadAwareRouter:
    """Route prompts to replicas based on current load."""

    def __init__(self, num_replicas: int, max_concurrent_per_replica: int | None = None):
        self._in_flight = [0] * num_replicas
        self._capacity = max_concurrent_per_replica  # None = unlimited
        self._not_full = asyncio.Condition()
        # Track which slots are occupied per replica (for visualization)
        self._occupied_slots: list[set[int]] = [set() for _ in range(num_replicas)]

    async def acquire_slot(self) -> tuple[int, int]:
        """Get the index of the least-loaded replica and acquire a slot.

        Returns:
            Tuple of (replica_idx, slot_idx) where slot_idx is a stable index
            for visualization.
        """
        async with self._not_full:
            # Wait until at least one replica has capacity (if capacity is limited)
            if self._capacity is not None:
                while all(c >= self._capacity for c in self._in_flight):
                    await self._not_full.wait()

            # Find replica with lowest in-flight count
            best_idx = min(range(len(self._in_flight)), key=lambda i: self._in_flight[i])
            self._in_flight[best_idx] += 1

            # Find first available slot index for this replica
            occupied = self._occupied_slots[best_idx]
            slot_idx = 0
            while slot_idx in occupied:
                slot_idx += 1
            occupied.add(slot_idx)

            return best_idx, slot_idx

    async def release_slot(self, replica_idx: int, slot_idx: int) -> None:
        """Release a slot on the given replica."""
        async with self._not_full:
            self._in_flight[replica_idx] -= 1
            self._occupied_slots[replica_idx].discard(slot_idx)
            # Use notify(1) instead of notify_all() to avoid thundering herd
            # Only one slot was released, so only one waiter needs to wake
            self._not_full.notify(1)

    def get_load(self) -> list[int]:
        """Get current in-flight counts (for metrics)."""
        return list(self._in_flight)

    async def reset(self) -> None:
        """Clear router state after a pause/sync."""
        async with self._not_full:
            num_replicas = len(self._in_flight)
            self._in_flight = [0] * num_replicas
            self._occupied_slots = [set() for _ in range(num_replicas)]
            # Wake any waiters so they can re-acquire with fresh state.
            self._not_full.notify_all()


class VLLMHandle:
    """Handle for one or more vLLM instances (data parallel replicas)."""

    def __init__(self, actors: list, name: str = "vllm", max_concurrent_per_replica: int = 8):
        self._actors = actors
        self._router = LoadAwareRouter(len(actors), max_concurrent_per_replica)
        self._in_flight = 0
        self._in_flight_zero = asyncio.Event()
        self._in_flight_zero.set()  # Initially no in-flight requests
        self._paused = asyncio.Event()
        self._paused.set()  # Initially not paused (gate is open)
        self._pause_generation = 0
        self.name = name
        self._trainer_version = 0  # Trainer version when weights were last synced

    @property
    def trainer_version(self) -> int:
        """Trainer version when weights were last synced to this vLLM."""
        return self._trainer_version

    def set_trainer_version(self, version: int) -> None:
        """Update trainer version after weight sync."""
        self._trainer_version = version

    @property
    def num_replicas(self) -> int:
        return len(self._actors)

    async def abort_all(self) -> int:
        """Abort all in-flight requests on all replicas. Returns total aborted."""
        results = await asyncio.gather(*[a.abort_all.remote() for a in self._actors])
        return sum(results)

    async def stop(self, abort: bool = False, reset_router: bool = False) -> None:
        """Stop accepting new generation requests.

        Args:
            abort: If True, abort in-flight requests instead of waiting for them.
            reset_router: If True, clear router state after all in-flight requests finish.
        """
        self._pause_generation += 1
        self._paused.clear()  # Close the gate for new requests
        if abort:
            await self.abort_all()
        await self._in_flight_zero.wait()  # Wait for in-flight requests to finish
        loads = self._router.get_load()
        if any(loads):
            logger.warning(f"[vllm.stop] router loads non-zero after drain: {loads}")
        else:
            logger.info(f"[vllm.stop] router loads after drain: {loads}")
        if reset_router:
            await self._router.reset()
            logger.info(f"[vllm.stop] router reset complete: {self._router.get_load()}")

    def resume(self) -> None:
        """Resume accepting generation requests after sync."""
        self._pause_generation += 1
        self._paused.set()  # Open the gate

    async def generate_single(self, prompt: str, **sampling_params):
        """Generate for a single prompt with load-aware routing.

        Routes the prompt to the replica with the fewest in-flight requests,
        enabling better GPU utilization when completion times vary.

        If the request is aborted due to weight sync, automatically retries
        after sync completes. Callers don't need to handle AbortedError.
        """
        t_enter = time.perf_counter()
        empty_failures = 0
        max_empty_retries = 2

        def is_cancellation_error(e: BaseException) -> bool:
            msg = str(e).lower()
            return ("cancel" in msg) or ("abort" in msg)

        while True:
            # Wait if paused
            t_pause_start = time.perf_counter()
            if not self._paused.is_set():
                await self._paused.wait()
            t_pause_end = time.perf_counter()

            # Acquire slot first (this is where backpressure happens)
            t_slot_start = time.perf_counter()
            replica_idx, slot_idx = await self._router.acquire_slot()
            t_slot_end = time.perf_counter()

            # Log timing if there was significant wait
            pause_wait = t_pause_end - t_pause_start
            slot_wait = t_slot_end - t_slot_start
            # if pause_wait > 0.1 or slot_wait > 0.1:
            #     print(f"[TIMING generate_single] pause_wait={pause_wait:.2f}s slot_wait={slot_wait:.2f}s", flush=True)

            # Now track in-flight (only after we have a slot)
            self._in_flight += 1
            if self._in_flight == 1:
                self._in_flight_zero.clear()

            try:
                # Check if we got paused while waiting for slot
                if not self._paused.is_set():
                    # Release and wait for resume, then retry
                    continue

                actor = self._actors[replica_idx]
                pause_generation_before_call = self._pause_generation
                try:
                    t_gen_start = time.perf_counter()
                    with trace_span("vllm.generate_single", args={"replica": replica_idx, "slot": slot_idx}):
                        result = await actor.generate.remote([prompt], **sampling_params)
                    t_gen_end = time.perf_counter()
                    gen_time = t_gen_end - t_gen_start
                    total_time = t_gen_end - t_enter
                    # if total_time > 5.0:
                    #     print(f"[TIMING generate_single DONE] gen={gen_time:.2f}s total={total_time:.2f}s overhead={total_time-gen_time:.2f}s", flush=True)
                    # Check if result was aborted (empty outputs)
                    if result and result[0].outputs and len(result[0].outputs) > 0:
                        # Check if any output has tokens (not all aborted)
                        if any(len(out.token_ids) > 0 for out in result[0].outputs):
                            return result[0]
                    # Result was aborted or empty - retry iff a weight sync happened while in-flight.
                    if pause_generation_before_call != self._pause_generation:
                        empty_failures = 0
                        continue

                    empty_failures += 1
                    if empty_failures >= max_empty_retries:
                        raise RuntimeError(
                            "vLLM returned empty outputs repeatedly (not during a weight sync). "
                            "This is often caused by prompts exceeding max_model_len."
                        )
                    logger.info(f"[generate_single] got empty result, retrying ({empty_failures}/{max_empty_retries})")
                    continue
                except ray.exceptions.RayTaskError as e:
                    # During weight sync we abort in-flight requests; Ray can surface that as task errors.
                    # Avoid infinite retry loops on real errors (e.g. prompt too long).
                    if pause_generation_before_call != self._pause_generation or is_cancellation_error(e):
                        empty_failures = 0
                        continue
                    raise
            finally:
                await self._router.release_slot(replica_idx, slot_idx)
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._in_flight_zero.set()

    async def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs for reference model use (Titan-compatible interface).

        This method provides the same interface as DistributedModelHandle.compute_logprobs
        so vLLM can be used as a drop-in replacement for the reference model.

        Args:
            input_ids: [B, seq_len] - full sequence (prompt + completion + padding)
            completion_ids: [B, completion_len] - just the completion tokens (for shape reference)
            prompt_lens: [B] - length of prompt for each sample

        Returns:
            [B, completion_len] tensor of logprobs for completion tokens.
        """
        batch_size = input_ids.shape[0]
        completion_len = completion_ids.shape[1]

        # Convert tensors to lists for vLLM
        # Strip padding from input_ids based on actual sequence lengths
        token_ids_list = []
        prompt_lens_list = prompt_lens.tolist()

        for i in range(batch_size):
            # Get actual sequence length (up to first pad token or end)
            seq = input_ids[i].tolist()
            plen = prompt_lens_list[i]
            # Keep full padded completion length; padding is masked later.
            # Avoid trimming based on a hardcoded pad_token_id (0 is wrong for Qwen).
            actual_seq_len = plen + completion_len
            token_ids_list.append(seq[:actual_seq_len])

        # Get logprobs from vLLM
        logprobs_lists = await self.get_logprobs_single(
            token_ids_list,
            prompt_lens_list,
            temperature=temperature,
        )

        # Convert back to padded tensor [B, completion_len]
        result = torch.zeros(batch_size, completion_len, dtype=torch.float32)
        for i, lps in enumerate(logprobs_lists):
            actual_len = min(len(lps), completion_len)
            result[i, :actual_len] = torch.tensor(lps[:actual_len], dtype=torch.float32)

        return result

    async def get_logprobs_single(
        self,
        token_ids_list: list[list[int]],
        prompt_lens: list[int],
        temperature: float = 1.0,
    ) -> list[list[float]]:
        """Get logprobs for pre-tokenized sequences with load-aware routing.

        Used for reference model logprob computation. Routes to the replica
        with fewest in-flight requests.

        Args:
            token_ids_list: List of token ID sequences (prompt + completion).
            prompt_lens: Length of prompt portion for each sequence.

        Returns:
            List of logprob lists for completion tokens.
        """
        while True:
            # Wait if paused
            if not self._paused.is_set():
                await self._paused.wait()

            # Acquire slot first (this is where backpressure happens)
            replica_idx, slot_idx = await self._router.acquire_slot()

            # Track in-flight
            self._in_flight += 1
            if self._in_flight == 1:
                self._in_flight_zero.clear()

            try:
                # Check if we got paused while waiting for slot
                if not self._paused.is_set():
                    continue

                actor = self._actors[replica_idx]
                try:
                    with trace_span("vllm.get_logprobs", args={"replica": replica_idx, "slot": slot_idx, "n_seqs": len(token_ids_list)}):
                        result = await actor.get_logprobs.remote(
                            token_ids_list,
                            prompt_lens,
                            temperature,
                        )
                    return result
                except asyncio.CancelledError:
                    # Request was cancelled - retry
                    continue
                except ray.exceptions.RayTaskError as e:
                    # Check if it's a retriable error (AbortedError) vs a real bug
                    if "AbortedError" in str(e):
                        continue
                    else:
                        # Real error - log and raise
                        logger.error(f"[get_logprobs_single] RayTaskError: {e}")
                        raise
            finally:
                await self._router.release_slot(replica_idx, slot_idx)
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._in_flight_zero.set()

    def get_replica_loads(self) -> list[int]:
        """Get current in-flight count per replica (for metrics/debugging)."""
        return self._router.get_load()

    async def log_stats(self):
        """Trigger vLLM's built-in stats logging on all replicas."""
        await asyncio.gather(*[a.log_stats.remote() for a in self._actors])

    async def get_metrics(self) -> dict:
        """Get aggregated metrics from all replicas and reset their counters."""
        results = await asyncio.gather(*[a.get_metrics.remote() for a in self._actors])
        # Aggregate across replicas
        total = {"output_tokens": 0, "prompt_tokens": 0, "calls": 0, "gen_tps": 0}
        for m in results:
            total["output_tokens"] += m["output_tokens"]
            total["prompt_tokens"] += m["prompt_tokens"]
            total["calls"] += m["calls"]
            total["gen_tps"] += m["gen_tps"]
        # Sum gen_tps across replicas (each replica's rolling window average)
        # No averaging needed - total throughput is the sum
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

        print(f"[VLLM SYNC recv_chunk] Received chunk: {len(chunk['params'])} params, total_numel={chunk['total_numel']:,}, dtype={dtype_str}", flush=True)

        offset = 0
        loaded_count = 0
        for p in chunk["params"]:
            weight = flat[offset : offset + p["numel"]].view(p["shape"])
            offset += p["numel"]
            self.model_runner.model.load_weights(weights=[(p["name"], weight)])
            loaded_count += 1
            # Log embed/lm_head weights for debugging
            if p["name"] in ["model.embed_tokens.weight", "lm_head.weight"]:
                print(f"[VLLM SYNC recv_chunk] Loaded {p['name']}: shape={p['shape']} numel={p['numel']:,} sample={weight[0,:3].tolist()}", flush=True)

        print(f"[VLLM SYNC recv_chunk] Loaded {loaded_count} params from chunk", flush=True)
