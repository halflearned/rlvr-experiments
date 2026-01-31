"""GRPO + GKD (Generalized Knowledge Distillation) training.

Combines GRPO's reward-based policy gradient with JSD-based distillation
from a frozen teacher model. The total loss is:

    L = alpha_grpo * GRPO_loss + alpha_gkd * JSD(student || teacher)

The teacher is a separate frozen model (e.g. an instruct variant or a larger
base model) that is never updated. The JSD is computed over the teacher's
top-K tokens at each position, following the GKD paper (Agarwal et al., 2024).
"""

import argparse
import asyncio
import os
import time
import traceback
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_TIMEOUT", "90")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, RewardStats
from rlvr_experiments.data import DataIterator, DATASET_LOADERS, load_apps, load_mbpp, load_humaneval, load_gsm8k, load_math, load_deepscaler, load_dummy, load_ifeval, load_if_multi_constraints, load_mixed
from rlvr_experiments.losses import GRPOLoss, DrGRPOLoss, DAPOLoss, compute_grpo_advantages, compute_drgrpo_advantages
from rlvr_experiments.ops import compute_logprobs
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed, get_checkpoint_dir, upload_traces_to_s3
from rlvr_experiments.verifiers import VerifierPool, APPSVerifier, MBPPVerifier, HumanEvalVerifier, MathVerifier, IFEvalVerifier, IFMultiConstraintsVerifier, MultiVerifier

DATASETS = {
    "apps": (load_apps, APPSVerifier),
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
    "math": (load_math, MathVerifier),
    "deepscaler": (load_deepscaler, MathVerifier),
    "dummy": (load_dummy, MathVerifier),
    "ifeval": (load_ifeval, IFEvalVerifier),
    "if_multi_constraints": (load_if_multi_constraints, IFMultiConstraintsVerifier),
    "mixed": (load_mixed, MultiVerifier),
}


# ---------------------------------------------------------------------------
# ChatML teacher template (for instruct teacher models)
# ---------------------------------------------------------------------------

def build_chatml_teacher_template(question: str) -> str:
    """Wrap a question in ChatML format for instruct teacher models."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# JSD loss for GKD (operates on teacher top-k logprobs)
# ---------------------------------------------------------------------------

class GKDJSDLoss(torch.nn.Module):
    """JSD loss between student (trainer) and teacher for GKD.

    JSD(alpha) = alpha * KL(teacher || M) + (1-alpha) * KL(student || M)
    where M = alpha * teacher + (1-alpha) * student

    This is computed over the teacher's top-K tokens at each position.
    """

    def __init__(self, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(self, logits, completion_ids, teacher_topk_ids, teacher_topk_lps,
                padding_mask, prompt_lens, temperature=1.0):
        B = logits.size(0)
        T = completion_ids.size(1)
        K = teacher_topk_ids.size(-1)

        topk_ids = teacher_topk_ids.to(completion_ids.device).long()
        topk_lps = teacher_topk_lps.to(completion_ids.device, dtype=torch.float32)

        # Get student logprobs for all K teacher tokens at each position
        student_topk_lps = torch.zeros(B, T, K, device=completion_ids.device, dtype=torch.float32)
        for k_idx in range(K):
            lps_k, _ = compute_logprobs(
                logits, topk_ids[:, :, k_idx], prompt_lens=prompt_lens, temperature=temperature,
            )
            student_topk_lps[:, :, k_idx] = lps_k

        # Student logprob for actual chosen token (for logging)
        student_chosen_lps, _ = compute_logprobs(
            logits, completion_ids, prompt_lens=prompt_lens, temperature=temperature,
        )

        # Renormalize teacher probs over top-k
        teacher_probs_raw = topk_lps.exp()
        teacher_probs_sum = teacher_probs_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        teacher_probs = teacher_probs_raw / teacher_probs_sum  # [B, T, K]

        student_probs = student_topk_lps.exp()  # [B, T, K]

        # Mixture: M = alpha * teacher + (1-alpha) * student
        alpha = self.alpha
        M = alpha * teacher_probs + (1.0 - alpha) * student_probs
        log_M = M.clamp_min(1e-12).log()

        # KL(teacher || M)
        teacher_log_probs = teacher_probs.clamp_min(1e-12).log()
        kl_teacher_M = (teacher_probs * (teacher_log_probs - log_M)).sum(dim=-1)

        # KL(student || M)
        student_log_probs = student_probs.clamp_min(1e-12).log()
        kl_student_M = (student_probs * (student_log_probs - log_M)).sum(dim=-1)

        # JSD = alpha * KL(teacher||M) + (1-alpha) * KL(student||M)
        jsd_per_token = alpha * kl_teacher_M + (1.0 - alpha) * kl_student_M

        mask = padding_mask.to(jsd_per_token.device, dtype=torch.float32)
        per_response_loss = (jsd_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        loss = per_response_loss.mean()

        with torch.no_grad():
            mean_student_lp = (student_chosen_lps.detach() * mask).sum() / mask.sum().clamp_min(1.0)
            mean_jsd = (jsd_per_token.detach() * mask).sum() / mask.sum().clamp_min(1.0)

        self._last_debug = {
            "gkd_loss": float(loss.detach()),
            "mean_jsd": float(mean_jsd),
            "mean_student_lp": float(mean_student_lp),
        }
        return loss


# ---------------------------------------------------------------------------
# Combined GRPO + GKD loss (single forward pass)
# ---------------------------------------------------------------------------

class CombinedGRPOGKDLoss(torch.nn.Module):
    """Wraps a GRPO loss and a GKD JSD loss into a single forward call.

    This avoids two separate forward_backward calls (which breaks torch.compile).
    The combined loss is: alpha_grpo * grpo_loss + alpha_gkd * gkd_loss

    Usage:
        combined = CombinedGRPOGKDLoss(grpo_loss_fn, gkd_loss_fn, alpha_grpo, alpha_gkd)
        loss = combined(logits, completion_ids, ref_logprobs, old_logprobs, advantages,
                        teacher_topk_ids, teacher_topk_lps,
                        padding_mask=..., prompt_lens=..., temperature=...)
    """

    def __init__(self, grpo_loss_fn, gkd_loss_fn, alpha_grpo: float, alpha_gkd: float):
        super().__init__()
        self.grpo_loss_fn = grpo_loss_fn
        self.gkd_loss_fn = gkd_loss_fn
        self.alpha_grpo = alpha_grpo
        self.alpha_gkd = alpha_gkd
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(self, logits, completion_ids, ref_logprobs, old_logprobs, advantages,
                teacher_topk_ids, teacher_topk_lps,
                padding_mask, prompt_lens, temperature=1.0):
        # GRPO loss
        grpo_loss = self.grpo_loss_fn(logits, completion_ids, ref_logprobs, old_logprobs,
                                       advantages, padding_mask=padding_mask,
                                       prompt_lens=prompt_lens, temperature=temperature)

        # GKD JSD loss
        gkd_loss = self.gkd_loss_fn(logits, completion_ids, teacher_topk_ids, teacher_topk_lps,
                                     padding_mask=padding_mask, prompt_lens=prompt_lens,
                                     temperature=temperature)

        combined = self.alpha_grpo * grpo_loss + self.alpha_gkd * gkd_loss

        # Collect debug metrics from both
        grpo_debug = self.grpo_loss_fn.get_debug_metrics() if hasattr(self.grpo_loss_fn, 'get_debug_metrics') else {}
        gkd_debug = self.gkd_loss_fn.get_debug_metrics() or {}

        self._last_debug = {
            "loss_grpo": float(grpo_loss.detach()),
            "loss_gkd": float(gkd_loss.detach()),
            **(grpo_debug or {}),
            **gkd_debug,
        }

        return combined


# ---------------------------------------------------------------------------
# Extended TrainSample with teacher logprobs
# ---------------------------------------------------------------------------

class GKDTrainSample:
    """TrainSample extended with per-completion teacher top-k logprobs."""

    def __init__(self, grpo_sample: TrainSample,
                 teacher_topk_ids: list[torch.Tensor],
                 teacher_topk_lps: list[torch.Tensor]):
        self.grpo_sample = grpo_sample
        # Lists of [comp_len, K] tensors, one per completion
        self.teacher_topk_ids = teacher_topk_ids
        self.teacher_topk_lps = teacher_topk_lps

    @property
    def trainer_version(self):
        return self.grpo_sample.trainer_version

    @property
    def item_id(self):
        return self.grpo_sample.item_id

    @property
    def rewards(self):
        return self.grpo_sample.rewards

    @property
    def dataset(self):
        return self.grpo_sample.dataset


def make_gkd_batch(samples: list[GKDTrainSample], pad_token_id: int,
                    seq_len_buckets=None, completion_len_buckets=None):
    """Make a GRPO batch + aligned teacher top-k tensors.

    Returns (grpo_batch, grpo_stats, teacher_topk_ids, teacher_topk_lps,
             item_ids, group_sizes, trained_meta)
    where teacher_topk_{ids,lps} are [total_completions, padded_comp_len, K].
    """
    grpo_samples = [s.grpo_sample for s in samples]
    batch, stats = make_batch(grpo_samples, pad_token_id,
                              seq_len_buckets=seq_len_buckets,
                              completion_len_buckets=completion_len_buckets)

    padded_comp_len = batch.completion_ids.size(1)

    # Flatten teacher top-k across all completions in all samples
    all_topk_ids = []
    all_topk_lps = []
    for s in samples:
        for tid, tlp in zip(s.teacher_topk_ids, s.teacher_topk_lps):
            K = tid.size(-1)
            # Pad to padded_comp_len
            if tid.size(0) < padded_comp_len:
                pad_rows = padded_comp_len - tid.size(0)
                tid = F.pad(tid, (0, 0, 0, pad_rows), value=0)
                tlp = F.pad(tlp, (0, 0, 0, pad_rows), value=-100.0)
            else:
                tid = tid[:padded_comp_len]
                tlp = tlp[:padded_comp_len]
            all_topk_ids.append(tid)
            all_topk_lps.append(tlp)

    teacher_topk_ids = torch.stack(all_topk_ids)  # [total_comp, comp_len, K]
    teacher_topk_lps = torch.stack(all_topk_lps)

    item_ids = [s.item_id for s in samples]
    group_sizes = [len(s.rewards) for s in samples]
    trained_meta = [
        {
            "item_id": s.item_id,
            "trainer_version": s.trainer_version,
            "dataset": s.dataset,
            "n_tokens": sum(s.grpo_sample.rollout.completion_lens),
        }
        for s in samples
    ]

    return batch, stats, teacher_topk_ids, teacher_topk_lps, item_ids, group_sizes, trained_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_schedule(training_cfg: dict) -> dict:
    p_rollout = training_cfg["prompts_per_rollout_sync"]
    p_reference = training_cfg["prompts_per_reference_sync"]
    p_optim = training_cfg["prompts_per_optim_step"]
    p_fwd_bwd = training_cfg["prompts_per_forward_backward"]
    accumulation_steps = p_optim // p_fwd_bwd
    sync_model_every = p_rollout // p_optim or 1
    sync_ref_every = p_reference // p_optim or 1
    return {
        "accumulation_steps": accumulation_steps,
        "sync_model_every": sync_model_every,
        "sync_ref_every": sync_ref_every,
        "max_staleness": training_cfg["max_staleness"],
    }


def _get_micro_batch_size(cfg: Any, seq_len: int) -> int:
    if isinstance(cfg, dict):
        for bucket in sorted(cfg.keys()):
            if seq_len <= bucket:
                return cfg[bucket]
        return cfg[max(cfg.keys())]
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    run_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    training = plan.training
    sampling = plan.sampling
    roles = plan.roles
    run_id = runtime.run_name or "grpo_gkd_run"

    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] Using seed: {seed}")

    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer
    tracer = runtime.tracer

    # Frozen teacher model (required for GKD)
    teacher_model = runtime.roles["teacher"]
    print(f"[init] Using frozen teacher model for GKD (not synced)")

    resume_step = training.get("resume_step", 0)
    if resume_step:
        trainer.version = resume_step
        rollout.set_trainer_version(resume_step)
        reference.set_trainer_version(resume_step)
        print(f"[init] Resuming from step {resume_step}")

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]
    data_iter_cfg = dict(plan.data_iter)
    data_iter_cfg.setdefault("skip_chat_template", True)
    if dataset_name == "mixed":
        ds, order = load_fn(**data_cfg)
        data_iter = DataIterator(ds, tokenizer=tokenizer, order=order, **data_iter_cfg)
    else:
        data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **data_iter_cfg)

    verifier_cfg = plan.verifier
    verifier = VerifierPool(verifier_cls, **verifier_cfg)
    verify_completions = verifier.verify_completions

    loss_cfg = dict(plan.loss)
    loss_name = loss_cfg.pop("name", "drgrpo")
    if loss_name == "grpo":
        loss_fn = GRPOLoss(**loss_cfg); compute_advantages = compute_grpo_advantages
    elif loss_name == "drgrpo":
        loss_fn = DrGRPOLoss(**loss_cfg); compute_advantages = compute_drgrpo_advantages
    elif loss_name == "dapo":
        loss_fn = DAPOLoss(**loss_cfg); compute_advantages = compute_grpo_advantages
    else:
        raise ValueError(f"Unknown loss name: {loss_name}.")
    print(f"[init] Using loss: {loss_name}")
    reward_stats = RewardStats()

    # GKD-specific config
    alpha_grpo = training.get("alpha_grpo", 1.0)
    alpha_gkd = training.get("alpha_gkd", 0.1)
    jsd_alpha = training.get("jsd_alpha", 0.9)
    teacher_top_k = training.get("teacher_top_k", 20)
    teacher_use_chat_template = training.get("teacher_use_chat_template", True)
    gkd_loss_fn = GKDJSDLoss(alpha=jsd_alpha)
    if loss_name == "dapo":
        raise ValueError("CombinedGRPOGKDLoss does not support DAPO yet; use grpo or drgrpo")
    combined_loss_fn = CombinedGRPOGKDLoss(loss_fn, gkd_loss_fn, alpha_grpo, alpha_gkd)
    print(f"[config] alpha_grpo={alpha_grpo}, alpha_gkd={alpha_gkd}, jsd_alpha={jsd_alpha}")
    print(f"[config] teacher_top_k={teacher_top_k}, teacher_use_chat_template={teacher_use_chat_template}")

    checkpoint_dir, _ = get_checkpoint_dir()

    schedule = _compute_schedule(training)
    accumulation_steps = schedule["accumulation_steps"]
    sync_model_every = schedule["sync_model_every"]
    sync_ref_every = schedule["sync_ref_every"]
    max_staleness = schedule["max_staleness"]

    max_completion_len = sampling["max_tokens"]
    seq_len_buckets = training["seq_len_buckets"]
    completion_len_buckets = training["completion_len_buckets"] or [max_completion_len]
    max_seq_len = seq_len_buckets[-1]

    sampling_params = {**sampling, "logprobs": 0}
    policy_temperature = sampling_params.get("temperature", 1.0)
    rollout_max_model_len = roles["rollout"].config.get("max_model_len")
    teacher_max_model_len = roles["teacher"].config.get("max_model_len")
    rollout_timeout_s = training.get("rollout_timeout_s", 9999)
    max_concurrent_tasks = training.get("max_concurrent_tasks", 64)

    print(f"[config] accumulation_steps={accumulation_steps}, sync_model_every={sync_model_every}, sync_ref_every={sync_ref_every}")

    def mark_filtered(prompt_id: str, trainer_version: int, dataset: str, reason: str) -> None:
        buffer.stats.record_filtered(trainer_version)
        log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason=reason, dataset=dataset)
        data_iter.mark_done(prompt_id)

    async def compute_ref_logprobs(rollout_sample: RolloutSample) -> torch.Tensor:
        with trace_span("ref_logprobs"):
            n = rollout_sample.input_ids.size(0)
            mb = training.get("completions_per_micro_batch_reference") or n
            chunks = []
            for i in range(0, n, mb):
                chunk = await reference.compute_logprobs(
                    rollout_sample.input_ids[i:i+mb],
                    rollout_sample.completion_ids[i:i+mb],
                    torch.tensor([rollout_sample.prompt_len] * min(mb, n - i)),
                    temperature=policy_temperature,
                )
                chunks.append(chunk)
            return torch.cat(chunks, dim=0)

    async def compute_teacher_topk(template: str, prompt_text: str,
                                   completion_token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute teacher top-k logprobs for one completion.

        Returns (topk_ids, topk_lps) each [comp_len, K], or None if too long.
        """
        if teacher_use_chat_template:
            teacher_tmpl = build_chatml_teacher_template(prompt_text)
        else:
            teacher_tmpl = template

        teacher_prompt_ids = tokenizer.encode(teacher_tmpl, add_special_tokens=False)
        teacher_seq = teacher_prompt_ids + completion_token_ids
        teacher_prompt_len = len(teacher_prompt_ids)

        if teacher_max_model_len is not None and len(teacher_seq) > teacher_max_model_len:
            return None

        with trace_span("teacher_topk_logprobs"):
            results = await teacher_model.get_logprobs_topk_single(
                [teacher_seq], [teacher_prompt_len],
                top_k=teacher_top_k, temperature=policy_temperature,
            )

        position_dicts = results[0]
        comp_len = len(completion_token_ids)
        K = teacher_top_k

        topk_ids = torch.zeros(comp_len, K, dtype=torch.long)
        topk_lps = torch.full((comp_len, K), -100.0, dtype=torch.float32)

        for t, lp_dict in enumerate(position_dicts[:comp_len]):
            if not lp_dict:
                continue
            sorted_items = sorted(lp_dict.items(), key=lambda x: x[1], reverse=True)[:K]
            for k_idx, (tid, lp) in enumerate(sorted_items):
                topk_ids[t, k_idx] = tid
                topk_lps[t, k_idx] = lp

        return topk_ids, topk_lps

    async def produce_epoch():
        async def process_one(item):
            prompt_id = item["problem"].get("prompt_id", "unknown")
            dataset = item["problem"].get("dataset_name", item["problem"].get("verifier_type", "unknown"))

            per_sample_max = item["problem"].get("max_completion_len")
            effective_max_tokens = per_sample_max if isinstance(per_sample_max, int) else max_completion_len
            sp = {**sampling_params, "max_tokens": effective_max_tokens}

            if rollout_max_model_len is not None:
                prompt_tokens = len(tokenizer.encode(item["template"], add_special_tokens=False))
                headroom = rollout_max_model_len - prompt_tokens
                if headroom <= 0:
                    mark_filtered(prompt_id, rollout.trainer_version, dataset, reason="prompt_too_long")
                    return
                if sp.get("max_tokens") and sp["max_tokens"] > headroom:
                    sp = {**sp, "max_tokens": headroom}

            trainer_version = rollout.trainer_version
            verify_problem = dict(item["problem"])
            verify_problem.setdefault("prompt", item["prompt"])
            verify_problem.setdefault("template", item["template"])

            with trace_span("generate"):
                response = await asyncio.wait_for(rollout.generate_single(item["template"], **sp), timeout=rollout_timeout_s)
            completions = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id, prompt_id=prompt_id)
            with trace_span("verify"):
                rewards = await verify_completions(verify_problem, completions)

            log_rollout(prompt_id=prompt_id, prompt=item["prompt"], completions=completions, rewards=rewards,
                        trainer_version=trainer_version, trainer_version_after=rollout.trainer_version, dataset=dataset)

            if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="zero_variance")
                return
            if rollout_sample.input_ids.shape[1] > max_seq_len:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="seq_too_long")
                return

            # Compute ref logprobs (for GRPO KL penalty)
            ref_logprobs = await compute_ref_logprobs(rollout_sample)

            # Compute teacher top-k logprobs for each completion (for GKD JSD)
            # Parallelize all completions with asyncio.gather for speed
            n_completions = rollout_sample.input_ids.size(0)

            async def _teacher_for_idx(idx):
                actual_len = rollout_sample.completion_lens[idx]
                comp_token_ids = rollout_sample.completion_ids[idx, :actual_len].tolist()
                result = await compute_teacher_topk(
                    item["template"], item["prompt"], comp_token_ids,
                )
                if result is None:
                    K = teacher_top_k
                    return (torch.zeros(actual_len, K, dtype=torch.long),
                            torch.full((actual_len, K), -100.0))
                return result

            teacher_results = await asyncio.gather(*[_teacher_for_idx(i) for i in range(n_completions)])
            teacher_topk_ids_list = [r[0] for r in teacher_results]
            teacher_topk_lps_list = [r[1] for r in teacher_results]

            reward_stats.record(rewards, used=True)
            grpo_sample = TrainSample(rollout_sample, rewards, ref_logprobs,
                                      item_id=prompt_id, trainer_version=trainer_version, dataset=dataset)
            gkd_sample = GKDTrainSample(grpo_sample, teacher_topk_ids_list, teacher_topk_lps_list)
            await buffer.put(gkd_sample, trainer_version, item_id=prompt_id)
            log_sample("buffered", prompt_id=prompt_id, version=trainer_version, dataset=dataset)

        async def safe_process_one(item):
            prompt_id = item["problem"].get("prompt_id", "unknown")
            try:
                await process_one(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                trainer_version = rollout.trainer_version
                tb = traceback.format_exc()
                print(f"[producer] prompt failed {prompt_id}: {e}\n{tb}")
                buffer.stats.record_failed(trainer_version)
                log_sample("failed", prompt_id=prompt_id, version=trainer_version, error=str(e), traceback=tb)
                data_iter.mark_failed(prompt_id)

        async def worker() -> None:
            while True:
                item = await data_iter.get_next_async()
                if item is None:
                    return
                await safe_process_one(item)

        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(max_concurrent_tasks):
                    tg.create_task(worker())
        finally:
            await buffer.mark_done()

    async def batches(producer_task):
        pending_batch = []
        prompts_per_forward_backward = training["prompts_per_forward_backward"]

        def evict_stale(sample_list):
            if not sample_list:
                return sample_list
            min_acceptable_version = trainer.version - max_staleness
            fresh = []
            for s in sample_list:
                if s.trainer_version < min_acceptable_version:
                    buffer.stats.record_wasted(s.trainer_version)
                    log_sample("evicted", prompt_id=s.item_id, trained_at_step=trainer.version,
                               trainer_version=s.trainer_version, reason="stale_after_pop")
                    data_iter.mark_pending(s.item_id)
                else:
                    fresh.append(s)
            return fresh

        def emit(sample_list):
            result = make_gkd_batch(sample_list, pad_token_id,
                                    seq_len_buckets=seq_len_buckets,
                                    completion_len_buckets=completion_len_buckets)
            return result

        while True:
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()
            if buffer.size() == 0 and data_iter.pending_count() == 0 and data_iter.in_flight_count() == len(pending_batch):
                entry = None
            else:
                entry = await buffer.pop()

            if entry is None:
                if pending_batch:
                    pending_batch = evict_stale(pending_batch)
                    if not pending_batch:
                        return
                    yield emit(pending_batch)
                return

            pending_batch.append(entry.item)

            if len(pending_batch) >= prompts_per_forward_backward:
                pending_batch = evict_stale(pending_batch)
                if len(pending_batch) < prompts_per_forward_backward:
                    continue
                yield emit(pending_batch)
                pending_batch = []

    accum_count = 0; accum_loss = 0.0; accum_ntokens = 0
    accum_grpo = 0.0; accum_gkd = 0.0
    trained_meta_accum = []

    for epoch in range(training.get("num_epochs") or 999999):
        if training["max_steps"] and trainer.version >= training["max_steps"]:
            break
        data_iter.new_epoch(seed=seed + epoch)
        producer = asyncio.create_task(produce_epoch())

        async for batch, stats, teacher_topk_ids, teacher_topk_lps, item_ids, group_sizes, trained_meta in batches(producer):
            accum_count += 1
            trained_meta_accum.extend(trained_meta)
            print(f"[fwd_bwd] accum={accum_count}/{accumulation_steps} completions={batch.input_ids.size(0)} "
                  f"seq_len={stats.padded_seq_len}")

            # --- Combined GRPO + GKD loss (single forward pass) ---
            advantages = compute_advantages(batch.rewards, group_sizes=group_sizes)
            with trace_span("forward_backward_combined"):
                _loss_args = (batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages,
                              teacher_topk_ids, teacher_topk_lps)
                _loss_kwargs = {"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens,
                                "temperature": policy_temperature}
                loss_combined, combined_debug = await trainer.forward_backward(
                    combined_loss_fn,
                    batch.input_ids,
                    loss_args=_loss_args,
                    loss_kwargs=_loss_kwargs,
                    scale_loss=1.0 / accumulation_steps,
                    micro_batch_size=_get_micro_batch_size(training["completions_per_micro_batch"], stats.padded_seq_len),
                )

            loss_grpo = combined_debug.get("loss_grpo", 0.0) if combined_debug else 0.0
            loss_gkd = combined_debug.get("loss_gkd", 0.0) if combined_debug else 0.0
            accum_grpo += loss_grpo
            accum_gkd += loss_gkd
            accum_loss += float(loss_combined)
            accum_ntokens += batch.input_ids.numel()
            del advantages
            for item_id in item_ids:
                data_iter.mark_done(item_id)
            if accum_count < accumulation_steps:
                continue

            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            avg_loss = accum_loss / accumulation_steps
            avg_grpo = accum_grpo / accumulation_steps
            avg_gkd = accum_gkd / accumulation_steps

            avg_reward = batch.rewards.mean().item()
            rw_metrics = reward_stats.get_metrics()
            stats.trace(tracer, step=trainer.version)

            # Debug metrics come from last combined_debug (from last micro-batch)
            grpo_debug = {k: v for k, v in (combined_debug or {}).items()
                          if k in ("kl_mean", "kl_max", "ratio_max", "entropy_mean", "clip_frac")}
            gkd_metrics = {k: v for k, v in (combined_debug or {}).items()
                           if k in ("gkd_loss", "mean_jsd", "mean_student_lp")}
            tracer.counter("metrics", {"loss": avg_loss, "loss_grpo": avg_grpo, "loss_gkd": avg_gkd,
                                       "grad_norm": grad_norm, "avg_reward": avg_reward})
            tracer.counter("grpo.debug", grpo_debug)
            tracer.counter("gkd.debug", gkd_metrics)
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)
            tracer.counter("reward_stats", rw_metrics)

            for meta in trained_meta_accum:
                buffer.stats.record_used(meta["trainer_version"])
                log_sample("trained", prompt_id=meta["item_id"], trained_at_step=trainer.version,
                           trainer_version=meta["trainer_version"], dataset=meta["dataset"], n_tokens=meta["n_tokens"])

            mean_jsd = gkd_metrics.get("mean_jsd", 0.0)
            print(f"[epoch {epoch}] step={trainer.version} loss={avg_loss:.4f} grpo={avg_grpo:.4f} "
                  f"gkd={avg_gkd:.4f} grad_norm={grad_norm:.4f} reward={avg_reward:.2f} "
                  f"reward_all={rw_metrics.get('reward_overall', avg_reward):.2f} mean_jsd={mean_jsd:.4f}")

            if trainer.version % sync_ref_every == 0:
                await sync_titan_to_vllm(trainer, reference, abort_in_flight=training["abort_in_flight"],
                                         trainer_version=trainer.version, wire_dtype="float16")
            if trainer.version % sync_model_every == 0:
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=training["abort_in_flight"],
                                         trainer_version=trainer.version, wire_dtype="float16")
            # teacher is NEVER synced — frozen
            if training["checkpoint_interval"] and trainer.version % training["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_step{trainer.version}")
                await trainer.export_to_hf(ckpt_path)
            if trainer.version % 10 == 0:
                upload_traces_to_s3(runtime.trace_dir, run_id)

            accum_count = 0; accum_loss = 0.0; accum_ntokens = 0
            accum_grpo = 0.0; accum_gkd = 0.0
            trained_meta_accum = []
            if training["max_steps"] and trainer.version >= training["max_steps"]:
                break

        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        buffer.reset()

    print("\n=== Training complete ===")
    await rollout.stop(abort=True)

    final_ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_final")
    print(f"Saving final checkpoint to {final_ckpt_path}")
    await trainer.export_to_hf(final_ckpt_path)

    upload_traces_to_s3(runtime.trace_dir, run_id)

    run_elapsed = time.perf_counter() - run_start_time
    print(f"\n=== Run Summary ===")
    print(f"Total time: {run_elapsed:.1f}s")

    import sys
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
