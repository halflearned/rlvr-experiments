"""GRPO training for code generation.

The algorithm is visible, the async machinery is hidden.
"""

import asyncio
import argparse

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_mbpp, load_humaneval, load_gsm8k, load_dummy
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.algorithms.grpo import grpo_samples
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.verifiers import VerifierPool, MBPPVerifier, HumanEvalVerifier, MathVerifier
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import trace_span

DATASETS = {
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
    "dummy": (load_dummy, MathVerifier),
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # --- Setup ---
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]

    verifier = VerifierPool(verifier_cls, **plan.verifier)

    data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **plan.data_iter)
    loss_fn = GRPOLoss(**plan.loss)

    # NOTE TO CLAUDE: DONT CHANGE THE VALUES BELOW!
    num_epochs = 1 #plan.training.get("num_epochs")
    max_steps = None # plan.training.get("max_steps")

    prompts_per_batch = 9 #plan.training.get("prompts_per_batch") or 1
    sync_ref_every = 10  # plan.training.get("sync_reference_every", 10)
    sync_model_every = 5 # plan.training.get("sync_model_every", 5)
    log_every = 5 #plan.training.get("log_every", 5)
    max_staleness = 1 # plan.training.get("max_staleness", 0)

    # Fixed sequence lengths to avoid dynamic shape recompilation
    max_completion_len = plan.sampling.get("max_tokens", 512)
    max_seq_len = plan.training.get("max_seq_len") or (max_completion_len + 256)

    # --- main loop ---
    async for step, epoch, batch in grpo_samples(
        rollout, data_iter, buffer,
        verifier_fn=verifier.verify_completions,
        pad_token_id=pad_token_id,
        prompts_per_batch=prompts_per_batch,
        sampling_params=plan.sampling,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_seq_len=max_seq_len,
        max_completion_len=max_completion_len,
        max_staleness=max_staleness,
    ):
        # ---- train step ----
        with trace_span("ref_logprobs"):
            ref_logprobs = await reference.compute_logprobs(
                batch.input_ids, batch.completion_ids, batch.prompt_lens
            )

        with trace_span("forward_backward"):
            loss, grpo_debug = await trainer.forward_backward(
                loss_fn,
                batch.input_ids,
                loss_args=(batch.completion_ids, ref_logprobs, batch.logprobs, batch.rewards),
                loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
            )

        with trace_span("optim_step"):
            grad_norm = await trainer.optim_step()

        # ---- maybe sync ----
        if step % sync_ref_every == 0:
            await sync_titan_to_titan(trainer, reference)

        if step % sync_model_every == 0:
            await sync_titan_to_vllm(trainer, rollout)


        # ---- maybe log ----
        titan_metrics = await trainer.log_metrics(loss, grad_norm, ntokens=batch.input_ids.numel())
        if titan_metrics:
            runtime.tracer.counter("titan.metrics", titan_metrics)

        if grpo_debug:
            runtime.tracer.counter("grpo.debug", grpo_debug)

        avg_reward = batch.rewards.mean().item()
        runtime.tracer.counter("metrics", {"loss": loss, "grad_norm": grad_norm, "avg_reward": avg_reward})

        if step % log_every == 0:
            await rollout.log_stats()

        # Collect vLLM metrics and emit to tracer
        vllm_metrics = await rollout.get_metrics()
        if vllm_metrics["calls"] > 0:
            runtime.tracer.counter("vllm.metrics", vllm_metrics)

        print(f"[epoch {epoch}] step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} reward={avg_reward:.2f}")

    # --- Save final checkpoint ---
    print("\n=== Training complete, saving checkpoint ===")
    checkpoint_path = "/efs/rlvr-experiments/checkpoints/grpo_final"
    await trainer.export_to_hf(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    asyncio.run(main())
