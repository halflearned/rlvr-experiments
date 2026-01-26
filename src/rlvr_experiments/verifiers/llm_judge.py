"""LLM-as-a-judge verifier using a vLLM engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from transformers import AutoTokenizer

from rlvr_experiments.vllm_engine_actor import VLLMHandle


class LLMJudgeVerifier:
    """Verifier that prompts an LLM judge to score completions in [0, 1]."""

    def __init__(
        self,
        vllm: VLLMHandle,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        use_chat_template: bool = False,
        tokenizer_name: str = "Qwen/Qwen3-8B",
        regex: str | None = None,
        sampling: dict | None = None,
        default_score: float = 0.0,
        clamp: bool = True,
        force_no_think: bool = False,
    ) -> None:
        self._vllm = vllm
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._use_chat_template = use_chat_template
        self._tokenizer_name = tokenizer_name
        pattern = regex or r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b"
        self._regex = re.compile(pattern, re.DOTALL)
        self._sampling = dict(sampling or {})
        self._default_score = float(default_score)
        self._clamp = clamp
        self._force_no_think = force_no_think
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name) if use_chat_template else None

    def _format_prompt(self, problem: dict, completion: str) -> str:
        # Provide common placeholders for templates.
        payload: dict[str, Any] = {
            "prompt": problem.get("prompt", ""),
            "template": problem.get("template", ""),
            "completion": completion,
            "response": completion,
            "answer": problem.get("answer", ""),
            "ground_truth": problem.get("ground_truth", ""),
            "prompt_id": problem.get("prompt_id", ""),
            "dataset_name": problem.get("dataset_name", ""),
            "verifier_type": problem.get("verifier_type", ""),
        }
        payload["problem_json"] = json.dumps(problem, ensure_ascii=True, sort_keys=True)

        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return ""

        if self._use_chat_template:
            if not self._user_prompt:
                raise ValueError("LLMJudgeVerifier requires user_prompt when use_chat_template=True")
            system_prompt = (self._system_prompt or "").format_map(_SafeDict(payload))
            user_prompt = self._user_prompt.format_map(_SafeDict(payload))
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        if not self._prompt:
            raise ValueError("LLMJudgeVerifier requires prompt when use_chat_template=False")
        rendered = self._prompt.format_map(_SafeDict(payload))
        if self._force_no_think and "/no_think" not in rendered:
            user_tag = "<|im_start|>user"
            idx = rendered.find(user_tag)
            if idx != -1:
                insert_pos = rendered.find("\n", idx)
                if insert_pos != -1:
                    rendered = rendered[:insert_pos + 1] + "/no_think\n" + rendered[insert_pos + 1 :]
                else:
                    rendered = rendered + "\n/no_think\n"
            else:
                rendered = "/no_think\n" + rendered
        return rendered

    def _parse_score(self, text: str) -> float:
        match = self._regex.search(text or "")
        if not match:
            return self._default_score
        try:
            score = float(match.group(1))
        except Exception:
            return self._default_score
        if self._clamp:
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
        return score

    def _sampling_params(self) -> dict:
        sp = dict(self._sampling)
        sp.setdefault("temperature", 0.0)
        sp.setdefault("top_p", 1.0)
        sp.setdefault("max_tokens", 16)
        sp.setdefault("n", 1)
        sp.setdefault("logprobs", 0)
        return sp

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores in [0, 1]."""
        sampling_params = self._sampling_params()

        async def judge_one(completion: str) -> float:
            prompt = self._format_prompt(problem, completion)
            output = await self._vllm.generate_single(prompt, **sampling_params)
            text = output.outputs[0].text if output.outputs else ""
            return self._parse_score(text)

        return list(await asyncio.gather(*[judge_one(c) for c in completions]))

    async def verify_batch_with_timing(
        self,
        problems: list[dict],
        completions: list[str],
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        scores: list[float] = []
        durations: list[float] = []
        timing_spans: list[tuple[float, float]] = []
        offset_ms = 0.0

        for problem, completion in zip(problems, completions):
            t0 = time.perf_counter()
            score = (await self.verify_completions(problem, [completion]))[0]
            dur_ms = (time.perf_counter() - t0) * 1000.0
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset_ms, dur_ms))
            offset_ms += dur_ms

        return scores, durations, timing_spans
