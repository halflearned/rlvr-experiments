"""On-Policy Self-Distillation (OPSD) training.

The student model generates completions conditioned on prompts. For completions
that get the wrong answer, a "self-teacher" (same model weights, but conditioned
on the prompt PLUS a hint about the correct answer) provides dense per-token
logprob targets. The student is trained to minimize the KL divergence toward
the teacher's distribution.
"""

import argparse
import asyncio
import os
import time
import traceback
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_TIMEOUT", "90")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, BatchStats, RewardStats, _bucket_size
from rlvr_experiments.data import DataIterator, load_if_multi_constraints, load_gsm8k
from rlvr_experiments.ops import compute_logprobs
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed, get_checkpoint_dir, upload_traces_to_s3
from rlvr_experiments.verifiers import VerifierPool, IFMultiConstraintsVerifier
from rlvr_experiments.verifiers.math import MathVerifier


# ---------------------------------------------------------------------------
# Constraint feedback helpers
# ---------------------------------------------------------------------------

def describe_constraint(instruction_id: str, kwargs: dict) -> str:
    """Generate a human-readable description of a constraint."""
    descriptions = {
        # Keywords
        "keywords:existence": lambda k: f"include the keywords: {k.get('keywords', [])}",
        "keywords:frequency": lambda k: f"use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:forbidden_words": lambda k: f"avoid these words: {k.get('forbidden_words', [])}",
        "keywords:letter_frequency": lambda k: f"use the letter '{k.get('letter', '')}' {k.get('let_relation', '')} {k.get('let_frequency', '')} times",
        "keywords:word_once": lambda k: f"use the word '{k.get('keyword', '')}' exactly once",
        "keywords:word_count_different_numbers": lambda k: f"use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:no_adjacent_consecutive": lambda k: "ensure no two adjacent words start with the same letter",
        "keywords:palindrome": lambda k: "include at least one palindrome word",
        "keywords:start_end": lambda k: "start and end the response with the same word",
        "keywords:exclude_word_harder": lambda k: f"do not use the word '{k.get('keyword', '')}'",
        "keywords:keyword_specific_position": lambda k: f"place the word '{k.get('keyword', '')}' at sentence {k.get('n', '')}, word position {k.get('m', '')}",
        # Language
        "language:response_language": lambda k: f"respond in {k.get('language', 'the specified language')}",
        # Length constraints
        "length_constraints:number_sentences": lambda k: f"use {k.get('relation', '')} {k.get('num_sentences', '')} sentences",
        "length_constraints:number_paragraphs": lambda k: f"use exactly {k.get('num_paragraphs', '')} paragraphs separated by ***",
        "length_constraints:number_words": lambda k: f"use {k.get('relation', '')} {k.get('num_words', '')} words",
        "length_constraints:nth_paragraph_first_word": lambda k: f"start paragraph {k.get('nth_paragraph', '')} with the word '{k.get('first_word', '')}'",
        # Detectable content
        "detectable_content:number_placeholders": lambda k: f"include at least {k.get('num_placeholders', '')} placeholders in [brackets]",
        "detectable_content:postscript": lambda k: f"include a postscript starting with {k.get('postscript_marker', 'P.S.')}",
        # Detectable format
        "detectable_format:number_bullet_lists": lambda k: f"use exactly {k.get('num_bullets', '')} bullet points",
        "detectable_format:constrained_response": lambda k: "end with 'My answer is yes/no/maybe.'",
        "detectable_format:number_highlighted_sections": lambda k: f"include at least {k.get('num_highlights', '')} highlighted sections with *asterisks*",
        "detectable_format:multiple_sections": lambda k: f"organize into {k.get('num_sections', '')} sections using '{k.get('section_spliter', '')}'",
        "detectable_format:json_format": lambda k: "format the response as valid JSON",
        "detectable_format:title": lambda k: "include a title wrapped in <<double angle brackets>>",
        "detectable_format:sentence_hyphens": lambda k: "separate every sentence with hyphens",
        "detectable_format:square_brackets": lambda k: "wrap every word in [square brackets]",
        "detectable_format:bigram_wrapping": lambda k: "wrap every pair of words in <<double angle brackets>>",
        # Combination
        "combination:two_responses": lambda k: "provide two different responses separated by ******",
        "combination:repeat_prompt": lambda k: f"start by repeating: '{k.get('prompt_to_repeat', '')[:80]}'",
        # Start/end
        "startend:end_checker": lambda k: f"end with the phrase '{k.get('end_phrase', '')}'",
        "startend:quotation": lambda k: "wrap the entire response in double quotes",
        # Change case
        "change_case:capital_word_frequency": lambda k: f"use {k.get('capital_relation', '')} {k.get('capital_frequency', '')} fully capitalized words",
        "change_case:english_capital": lambda k: "write the entire response in CAPITAL LETTERS",
        "change_case:english_lowercase": lambda k: "write the entire response in lowercase",
        # Punctuation
        "punctuation:no_comma": lambda k: "do not use any commas",
        "punctuation:punctuation_dot": lambda k: "do not use any periods/dots",
        "punctuation:punctuation_exclamation": lambda k: "do not use any exclamation marks",
        # Copy
        "copy:repeat_phrase": lambda k: f"repeat the phrase '{k.get('phrase', '')[:60]}' at least {k.get('small_n', '')} times",
        "copy:copy": lambda k: f"repeat exactly: '{k.get('prompt_to_repeat', '')[:80]}'",
        "copy:copying_simple": lambda k: f"repeat exactly: '{k.get('prompt_to_repeat', '')[:80]}'",
        "copy:copying_multiple": lambda k: f"repeat '{k.get('prompt_to_repeat', '')[:60]}' exactly {k.get('N', '')} times",
        "new:copy_span_idx": lambda k: f"copy characters {k.get('n_start', '')} to {k.get('n_end', '')} from: '{k.get('prompt_to_repeat', '')[:60]}'",
        # First/last word
        "first_word:first_word_sent": lambda k: f"start every sentence with the word '{k.get('first_word', '')}'",
        "first_word:first_word_answer": lambda k: f"start your response with the word '{k.get('first_word', '')}'",
        "last_word:last_word_sent": lambda k: f"end every sentence with the word '{k.get('last_word', '')}'",
        "last_word:last_word_answer": lambda k: f"end your response with the word '{k.get('last_word', '')}'",
        # Paragraphs
        "paragraphs:paragraphs": lambda k: "use exactly 2 paragraphs separated by ***",
        "paragraphs:paragraphs2": lambda k: "use exactly 2 paragraphs separated by blank lines",
        # Counting / letters
        "count:lowercase_counting": lambda k: f"use at most {k.get('N', '')} lowercase words",
        "count:counting_composition": lambda k: f"write exactly 3 paragraphs, each with {k.get('n_sent', '')} sentences of {k.get('n_words', '')} words",
        "count:count_unique": lambda k: "use every word at most once (all words must be unique)",
        "count:count_increment_word": lambda k: f"in each sentence, increase the count of '{k.get('keyword1', '')}' by one more than '{k.get('keyword2', '')}'",
        "letters:letter_counting": lambda k: f"use {k.get('relation', '')} {k.get('N', '')} letters total",
        "letters:letter_counting2": lambda k: f"use the letter '{k.get('letter', '')}' {k.get('let_relation', '')} {k.get('let_frequency', '')} times",
    }
    desc_fn = descriptions.get(instruction_id)
    if desc_fn:
        try:
            return desc_fn(kwargs)
        except Exception:
            pass
    return instruction_id.replace(":", " - ").replace("_", " ")


def format_failed_constraints_feedback(
    failed_instruction_ids: list[str],
    failed_kwargs: list[dict],
) -> str:
    """Format feedback about which constraints were failed."""
    if not failed_instruction_ids:
        return ""
    failed_descriptions = []
    for inst_id, kwargs in zip(failed_instruction_ids, failed_kwargs):
        desc = describe_constraint(inst_id, kwargs)
        failed_descriptions.append(f"- {desc}")
    feedback = (
        "\n\nIMPORTANT: In a previous attempt, you failed to satisfy these constraints. "
        "Please make sure to follow them this time:\n"
        + "\n".join(failed_descriptions)
        + "\n\nWrite your answer now."
    )
    return feedback


def build_teacher_template(
    original_template: str,
    per_constraint_results: list[bool],
    instruction_ids: list[str],
    kwargs_list: list[dict],
) -> str | None:
    """Build teacher prompt by appending constraint feedback to original template.

    Returns None if all constraints were satisfied (no feedback needed).
    """
    failed_ids = []
    failed_kwargs = []
    for passed, inst_id, kw in zip(per_constraint_results, instruction_ids, kwargs_list):
        if not passed:
            failed_ids.append(inst_id)
            failed_kwargs.append(kw)
    if not failed_ids:
        return None
    feedback = format_failed_constraints_feedback(failed_ids, failed_kwargs)
    return original_template + feedback


# ---------------------------------------------------------------------------
# GSM8k teacher template
# ---------------------------------------------------------------------------

def build_gsm8k_teacher_template(original_template: str, ground_truth_answer: str) -> str:
    """Build teacher prompt for GSM8k: original question + answer hint.

    The teacher sees the correct answer so it can provide better per-token
    targets for the student to learn from.
    """
    # original_template is "Q: {question}\nA:"
    # Strip the trailing "A:" and rebuild with hint
    base = original_template.rstrip()
    if base.endswith("A:"):
        base = base[:-2].rstrip()
    return (
        f"{base}\n"
        f"The answer to this question is {ground_truth_answer}. "
        f"With that in mind, please provide your reasoning.\nA:"
    )


# ---------------------------------------------------------------------------
# OPSD loss
# ---------------------------------------------------------------------------

class OPSDLoss(torch.nn.Module):
    """On-Policy Self-Distillation loss using top-k partial forward KL.

    Instead of single-sample KL (which fails when teacher assigns lower
    probability to student's sampled tokens), this computes a partial
    forward KL over the teacher's top-k tokens at each position:

        KL_partial(t) = sum_{i in top_k} q_i * (log q_i - log p_i)

    where q_i = renormalized teacher prob, p_i = student prob.
    This pushes the student to assign high probability to tokens the teacher
    considers likely — a proper distillation signal.

    The teacher's top-k are stored as:
        teacher_topk_ids: [B, T, K] — token IDs
        teacher_topk_lps: [B, T, K] — corresponding logprobs

    Uses compute_logprobs (F.cross_entropy) for DTensor compatibility.
    """

    def __init__(self):
        super().__init__()
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

        topk_ids = teacher_topk_ids.to(completion_ids.device).long()  # [B, T, K]
        topk_lps = teacher_topk_lps.to(completion_ids.device, dtype=torch.float32)  # [B, T, K]

        # Get student logprobs for all K teacher tokens at each position.
        # Loop over K and call compute_logprobs for each — this uses F.cross_entropy
        # which is DTensor-compatible and doesn't materialize full log_softmax.
        student_topk_lps = torch.zeros(B, T, K, device=completion_ids.device, dtype=torch.float32)
        for k_idx in range(K):
            lps_k, _ = compute_logprobs(
                logits, topk_ids[:, :, k_idx], prompt_lens=prompt_lens, temperature=temperature,
            )  # [B, T]
            student_topk_lps[:, :, k_idx] = lps_k

        # Also get student logprob for the actual chosen token (for logging)
        student_chosen_lps, _ = compute_logprobs(
            logits, completion_ids, prompt_lens=prompt_lens, temperature=temperature,
        )  # [B, T]

        # Renormalize teacher probs over top-k so partial distribution sums to 1
        teacher_probs_raw = topk_lps.exp()  # [B, T, K]
        teacher_probs_sum = teacher_probs_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        teacher_probs = teacher_probs_raw / teacher_probs_sum  # [B, T, K]

        # Forward KL(teacher || student) over top-k:
        # sum_i q_i * (log q_i - log p_i)
        teacher_log_probs = teacher_probs.log()
        kl_per_token = (teacher_probs * (teacher_log_probs - student_topk_lps)).sum(dim=-1)  # [B, T]

        mask = padding_mask.to(kl_per_token.device, dtype=torch.float32)
        per_response_loss = (kl_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        loss = per_response_loss.mean()

        with torch.no_grad():
            mean_student_lp = (student_chosen_lps.detach() * mask).sum() / mask.sum().clamp_min(1.0)
            mean_kl = (kl_per_token.detach() * mask).sum() / mask.sum().clamp_min(1.0)

        self._last_debug = {
            "opsd_loss": float(loss.detach()),
            "mean_kl": float(mean_kl),
            "mean_student_lp": float(mean_student_lp),
        }
        return loss


# ---------------------------------------------------------------------------
# OPSD-specific data structures
# ---------------------------------------------------------------------------

@dataclass
class OPSDBatch:
    """Batched data for OPSD training with top-k teacher logprobs."""
    input_ids: torch.Tensor          # [B, seq_len]
    completion_ids: torch.Tensor     # [B, completion_len]
    teacher_topk_ids: torch.Tensor   # [B, completion_len, K]
    teacher_topk_lps: torch.Tensor   # [B, completion_len, K]
    rewards: torch.Tensor            # [B]
    mask: torch.Tensor               # [B, completion_len]
    prompt_lens: torch.Tensor        # [B]


@dataclass
class OPSDCompletion:
    """A single completion with its teacher top-k logprobs. This is the atomic training unit."""
    input_ids: torch.Tensor          # [seq_len] - prompt + completion
    completion_ids: torch.Tensor     # [completion_len]
    logprobs: torch.Tensor           # [completion_len] - rollout logprobs
    teacher_topk_ids: torch.Tensor   # [completion_len, K] - teacher's top-k token IDs
    teacher_topk_lps: torch.Tensor   # [completion_len, K] - teacher's top-k logprobs
    prompt_len: int
    completion_len: int              # actual (unpadded) completion length
    reward: float
    item_id: str
    trainer_version: int
    dataset: str = "unknown"


def make_opsd_batch(
    completions: list[OPSDCompletion],
    pad_token_id: int,
    seq_len_buckets: list[int] | None = None,
    completion_len_buckets: list[int] | None = None,
) -> tuple[OPSDBatch, BatchStats]:
    """Build a Batch from a flat list of completions."""
    B = len(completions)

    natural_max_seq = max(c.input_ids.shape[0] for c in completions)
    natural_max_comp = max(c.completion_ids.shape[0] for c in completions)
    max_prompt_len = max(c.prompt_len for c in completions)

    # Bucket completion length
    comp_buckets = completion_len_buckets or [512, 1024, 1536, 2048]
    padded_completion_len = natural_max_comp
    for bucket in comp_buckets:
        if natural_max_comp <= bucket:
            padded_completion_len = bucket
            break
    else:
        padded_completion_len = comp_buckets[-1]

    # Bucket sequence length
    min_seq_len_needed = max_prompt_len + padded_completion_len
    seq_buckets = seq_len_buckets or [768, 1280, 1536, 2048, 2560]
    padded_seq_len = min_seq_len_needed
    for bucket in seq_buckets:
        if min_seq_len_needed <= bucket:
            padded_seq_len = bucket
            break
    else:
        padded_seq_len = seq_buckets[-1]

    max_completion_for_seq = padded_seq_len - max_prompt_len
    if padded_completion_len > max_completion_for_seq:
        padded_completion_len = max_completion_for_seq

    def pad1d(t, target_len, pad_value=0):
        if t.shape[0] >= target_len:
            return t[:target_len]
        return F.pad(t, (0, target_len - t.shape[0]), value=pad_value)

    K = completions[0].teacher_topk_ids.shape[-1]

    def pad2d(t, target_len, pad_value=0):
        """Pad a [T, K] tensor along dim 0 to target_len."""
        if t.shape[0] >= target_len:
            return t[:target_len]
        pad_rows = target_len - t.shape[0]
        return F.pad(t, (0, 0, 0, pad_rows), value=pad_value)

    input_ids = torch.stack([pad1d(c.input_ids, padded_seq_len, pad_token_id) for c in completions])
    completion_ids = torch.stack([pad1d(c.completion_ids, padded_completion_len, pad_token_id) for c in completions])
    logprobs = torch.stack([pad1d(c.logprobs, padded_completion_len) for c in completions])
    teacher_topk_ids = torch.stack([pad2d(c.teacher_topk_ids, padded_completion_len) for c in completions])  # [B, T, K]
    teacher_topk_lps = torch.stack([pad2d(c.teacher_topk_lps, padded_completion_len, -100.0) for c in completions])  # [B, T, K]
    rewards = torch.tensor([c.reward for c in completions], dtype=torch.float32)
    prompt_lens = torch.tensor([c.prompt_len for c in completions], dtype=torch.long)

    # Build mask from actual completion lengths
    mask = torch.zeros(B, padded_completion_len, dtype=torch.float32)
    for i, c in enumerate(completions):
        effective = min(c.completion_len, padded_completion_len)
        mask[i, :effective] = 1.0

    batch = OPSDBatch(
        input_ids=input_ids,
        completion_ids=completion_ids,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_lps=teacher_topk_lps,
        rewards=rewards,
        mask=mask,
        prompt_lens=prompt_lens,
    )

    # Build stats
    stats = BatchStats(
        seq_lens=[min(c.prompt_len + c.completion_len, padded_seq_len) for c in completions],
        completion_lens=[c.completion_len for c in completions],
        padded_seq_len=padded_seq_len,
        padded_completion_len=padded_completion_len,
        finish_reasons={},
        rewards=[c.reward for c in completions],
        datasets=[c.dataset for c in completions],
    )
    return batch, stats


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
    run_id = runtime.run_name or "opsd_run"

    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] Using seed: {seed}")

    await runtime.start()

    trainer = runtime.roles["trainer"]; rollout = runtime.roles["rollout"]
    buffer = runtime.buffer; tracer = runtime.tracer

    reference = runtime.roles.get("reference")

    resume_step = training.get("resume_step", 0)
    if resume_step:
        trainer.version = resume_step
        rollout.set_trainer_version(resume_step)
        if reference:
            reference.set_trainer_version(resume_step)
        print(f"[init] Resuming from step {resume_step}")

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    data_iter_cfg = dict(plan.data_iter)
    data_iter_cfg.setdefault("skip_chat_template", True)

    if dataset_name == "if_multi_constraints":
        data_iter = DataIterator(load_if_multi_constraints(**data_cfg), tokenizer=tokenizer, **data_iter_cfg)
        detail_verifier = IFMultiConstraintsVerifier()
        verifier_cls = IFMultiConstraintsVerifier
    elif dataset_name == "gsm8k":
        data_iter = DataIterator(load_gsm8k(**data_cfg), tokenizer=tokenizer, **data_iter_cfg)
        detail_verifier = MathVerifier()
        verifier_cls = MathVerifier
    else:
        raise ValueError(f"OPSD supports gsm8k or if_multi_constraints, got {dataset_name}")

    verifier_cfg = plan.verifier
    verifier = VerifierPool(verifier_cls, **verifier_cfg)
    verify_completions = verifier.verify_completions

    reward_stats = RewardStats()
    checkpoint_dir, _ = get_checkpoint_dir()

    # --- OPSD schedule: flat completion-based batching ---
    batch_size = training["batch_size"]           # completions per optimizer step
    minibatch_size = training["minibatch_size"]   # completions per forward-backward
    assert batch_size % minibatch_size == 0, f"batch_size ({batch_size}) must be divisible by minibatch_size ({minibatch_size})"
    accumulation_steps = batch_size // minibatch_size

    # How often to sync weights to rollout (in optimizer steps)
    sync_model_every = training.get("sync_every_n_steps", 1)
    max_staleness = training["max_staleness"]

    max_completion_len = sampling["max_tokens"]
    seq_len_buckets = training["seq_len_buckets"]
    completion_len_buckets = training["completion_len_buckets"] or [max_completion_len]

    sampling_params = {**sampling, "logprobs": 0}
    policy_temperature = sampling_params.get("temperature", 1.0)
    rollout_max_model_len = roles["rollout"].config.get("max_model_len")
    rollout_timeout_s = training.get("rollout_timeout_s", 9999)
    max_concurrent_tasks = training.get("max_concurrent_tasks", 64)

    opsd_loss_fn = OPSDLoss()
    print(f"[config] batch_size={batch_size}, minibatch_size={minibatch_size}, accumulation_steps={accumulation_steps}")
    print(f"[config] sync_every={sync_model_every}, max_staleness={max_staleness}")
    print(f"[config] OPSD mode: pure self-distillation, discard perfect completions")

    def mark_filtered(prompt_id: str, trainer_version: int, dataset: str, reason: str) -> None:
        buffer.stats.record_filtered(trainer_version)
        log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason=reason, dataset=dataset)
        data_iter.mark_done(prompt_id)

    teacher_top_k = training.get("teacher_top_k", 32)
    print(f"[config] teacher_top_k={teacher_top_k}")

    async def compute_teacher_topk_for_completion(
        teacher_template: str,
        completion_token_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute teacher top-k logprobs for a single completion.

        Returns (topk_ids, topk_lps) each of shape [comp_len, K],
        or None if teacher prompt doesn't fit.
        """
        teacher_prompt_ids = tokenizer.encode(teacher_template, add_special_tokens=False)
        teacher_seq = teacher_prompt_ids + completion_token_ids
        teacher_prompt_len = len(teacher_prompt_ids)

        if rollout_max_model_len is not None and len(teacher_seq) > rollout_max_model_len:
            print(f"[opsd] teacher seq too long: {len(teacher_seq)} > {rollout_max_model_len}, skipping")
            return None

        with trace_span("teacher_topk_logprobs"):
            # Returns list of list of dict {token_id: logprob}
            results = await rollout.get_logprobs_topk_single(
                [teacher_seq], [teacher_prompt_len],
                top_k=teacher_top_k, temperature=policy_temperature,
            )

        # results[0] is list of dicts, one per completion position
        position_dicts = results[0]
        comp_len = len(completion_token_ids)
        K = teacher_top_k

        topk_ids = torch.zeros(comp_len, K, dtype=torch.long)
        topk_lps = torch.full((comp_len, K), -100.0, dtype=torch.float32)

        for t, lp_dict in enumerate(position_dicts[:comp_len]):
            if not lp_dict:
                continue
            # Sort by logprob descending, take top K
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
            completions_text = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id, prompt_id=prompt_id)
            with trace_span("verify"):
                rewards = await verify_completions(verify_problem, completions_text)

            log_rollout(prompt_id=prompt_id, prompt=item["prompt"], completions=completions_text, rewards=rewards,
                        trainer_version=trainer_version, trainer_version_after=rollout.trainer_version, dataset=dataset)

            # --- OPSD-specific: compute teacher logprobs for incorrect completions ---

            # Filter: keep only completions with reward < 1.0
            kept_indices = [i for i, r in enumerate(rewards) if r < 1.0 - 1e-6]
            if not kept_indices:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="all_perfect_opsd")
                return

            # Build individual OPSDCompletion items for each kept completion
            opsd_completions = []

            # Pre-compute detailed results for IF dataset
            if dataset_name == "if_multi_constraints":
                ground_truth = item["problem"].get("ground_truth", "")
                detailed_results = [
                    detail_verifier.verify_detailed(comp, ground_truth) for comp in completions_text
                ]

            for idx in kept_indices:
                actual_len = rollout_sample.completion_lens[idx]
                comp_token_ids = rollout_sample.completion_ids[idx, :actual_len].tolist()

                # Build teacher template based on dataset type
                if dataset_name == "gsm8k":
                    answer = item["problem"].get("answer", "")
                    teacher_tmpl = build_gsm8k_teacher_template(item["template"], answer)
                else:
                    # IF: use constraint feedback
                    per_constraint, inst_ids, kw_list = detailed_results[idx]
                    teacher_tmpl = build_teacher_template(
                        item["template"], per_constraint, inst_ids, kw_list
                    )
                    if teacher_tmpl is None:
                        continue  # All constraints passed for this completion

                teacher_result = await compute_teacher_topk_for_completion(
                    teacher_tmpl, comp_token_ids,
                )
                if teacher_result is None:
                    continue
                topk_ids, topk_lps = teacher_result

                # Build per-completion tensors (unpadded — batching pads later)
                seq_len = rollout_sample.prompt_len + actual_len
                opsd_completions.append(OPSDCompletion(
                    input_ids=rollout_sample.input_ids[idx, :seq_len],
                    completion_ids=rollout_sample.completion_ids[idx, :actual_len],
                    logprobs=rollout_sample.logprobs[idx, :actual_len],
                    teacher_topk_ids=topk_ids[:actual_len],
                    teacher_topk_lps=topk_lps[:actual_len],
                    prompt_len=rollout_sample.prompt_len,
                    completion_len=actual_len,
                    reward=rewards[idx],
                    item_id=prompt_id,
                    trainer_version=trainer_version,
                    dataset=dataset,
                ))

            if not opsd_completions:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="teacher_all_skipped")
                return

            reward_stats.record(rewards, used=True)
            # Put each completion individually into the buffer
            for comp in opsd_completions:
                await buffer.put(comp, trainer_version, item_id=prompt_id)
            log_sample("buffered", prompt_id=prompt_id, version=trainer_version, dataset=dataset,
                        n_kept=len(opsd_completions), n_total=len(rewards))

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

    async def minibatches(producer_task):
        """Yield minibatches of exactly `minibatch_size` completions."""
        pending: list[OPSDCompletion] = []

        def evict_stale(comps):
            if not comps:
                return comps
            min_acceptable_version = trainer.version - max_staleness
            fresh = []
            for c in comps:
                if c.trainer_version < min_acceptable_version:
                    buffer.stats.record_wasted(c.trainer_version)
                    log_sample("evicted", prompt_id=c.item_id, trained_at_step=trainer.version,
                              trainer_version=c.trainer_version, reason="stale_after_pop")
                    # Re-queue the prompt for retry; otherwise it can remain stuck
                    # in_flight forever, deadlocking the producer.
                    data_iter.mark_pending(c.item_id)
                else:
                    fresh.append(c)
            return fresh

        while True:
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()

            # Check for end-of-data
            if buffer.size() == 0 and data_iter.pending_count() == 0 and data_iter.in_flight_count() == 0 and not pending:
                return

            entry = await buffer.pop()

            if entry is None:
                # End of epoch — flush remaining as a partial minibatch
                pending = evict_stale(pending)
                if pending:
                    yield pending
                return

            pending.append(entry.item)

            if len(pending) >= minibatch_size:
                pending = evict_stale(pending)
                if len(pending) >= minibatch_size:
                    yield pending[:minibatch_size]
                    pending = pending[minibatch_size:]

    accum_count = 0; accum_loss = 0.0; accum_ntokens = 0
    trained_items: set[str] = set()  # track unique prompt_ids trained this step

    for epoch in range(training.get("num_epochs") or 999999):
        if training["max_steps"] and trainer.version >= training["max_steps"]:
            break
        data_iter.new_epoch(seed=seed + epoch)
        producer = asyncio.create_task(produce_epoch())

        async for mb_completions in minibatches(producer):
            batch, stats = make_opsd_batch(
                mb_completions, pad_token_id,
                seq_len_buckets=seq_len_buckets,
                completion_len_buckets=completion_len_buckets,
            )
            accum_count += 1
            n_comp = len(mb_completions)

            print(f"[fwd_bwd] accum={accum_count}/{accumulation_steps} completions={n_comp} "
                  f"seq_len={stats.padded_seq_len} comp_len={stats.padded_completion_len}", flush=True)

            with trace_span("forward_backward"):
                loss_opsd, opsd_debug = await trainer.forward_backward(
                    opsd_loss_fn,
                    batch.input_ids,
                    loss_args=(batch.completion_ids, batch.teacher_topk_ids, batch.teacher_topk_lps),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": policy_temperature},
                    scale_loss=1.0 / accumulation_steps,
                )

            accum_loss += float(loss_opsd)
            accum_ntokens += sum(c.completion_len for c in mb_completions)
            for c in mb_completions:
                trained_items.add(c.item_id)
                buffer.stats.record_used(c.trainer_version)

            if accum_count < accumulation_steps:
                continue

            # --- Optimizer step ---
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            avg_loss = accum_loss / accumulation_steps

            avg_reward = batch.rewards.mean().item()
            rw_metrics = reward_stats.get_metrics()
            stats.trace(tracer, step=trainer.version)
            tracer.counter("metrics", {"loss": avg_loss, "grad_norm": grad_norm, "avg_reward": avg_reward})
            if opsd_debug:
                tracer.counter("opsd.debug", opsd_debug)
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)
            tracer.counter("reward_stats", rw_metrics)

            for item_id in trained_items:
                data_iter.mark_done(item_id)
                log_sample("trained", prompt_id=item_id, trained_at_step=trainer.version,
                          trainer_version=trainer.version, dataset=dataset_name)

            dbg = opsd_debug or {}
            print(f"[epoch {epoch}] step={trainer.version} loss={avg_loss:.4f} grad_norm={grad_norm:.4f} "
                  f"reward={avg_reward:.2f} n_completions={batch_size} n_prompts={len(trained_items)} "
                  f"reward_all={rw_metrics.get('reward_overall', avg_reward):.2f} "
                  f"mean_kl={dbg.get('mean_kl', 0):.4f} student_lp={dbg.get('mean_student_lp', 0):.4f}")

            if trainer.version % sync_model_every == 0:
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=training["abort_in_flight"],
                                         trainer_version=trainer.version, wire_dtype="float16")
            if training["checkpoint_interval"] and trainer.version % training["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_step{trainer.version}")
                await trainer.export_to_hf(ckpt_path)
            if trainer.version % 10 == 0:
                upload_traces_to_s3(runtime.trace_dir, run_id)

            accum_count = 0; accum_loss = 0.0; accum_ntokens = 0
            trained_items = set()
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
