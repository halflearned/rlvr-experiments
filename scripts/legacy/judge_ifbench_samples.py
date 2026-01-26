#!/usr/bin/env python3
"""
Score a small sample of IFBench/IFEval completions with the LLM judge.

Example:
    python3 scripts/judge_ifbench_samples.py \
        --config configs/qwen3-1.7B-ifeval-judge-mixed.yaml \
        --completions results/qwen3-1.7B-base/evals/ifeval/ifbench_completions_verified.jsonl \
        --n 12
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import ray
import yaml

from rlvr_experiments.vllm_engine_actor import VLLMEngineRank, VLLMHandle
from rlvr_experiments.verifiers import LLMJudgeVerifier


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _find_judge_role(cfg: dict) -> dict:
    for role in cfg.get("roles", []):
        if role.get("name") == "judge" and role.get("kind") == "vllm":
            return role.get("config", {})
    raise ValueError("No judge vLLM role found in config (name: judge, kind: vllm)")


def _load_completions(path: Path) -> list[dict]:
    items = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


async def _score_samples(
    judge: LLMJudgeVerifier,
    items: list[dict],
    debug: bool = False,
) -> list[dict]:
    results = []
    for item in items:
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        problem = {"prompt": prompt}
        if debug:
            judge_prompt = judge._format_prompt(problem, completion)
            sampling_params = judge._sampling_params()
            output = await judge._vllm.generate_single(judge_prompt, **sampling_params)
            text = output.outputs[0].text if output.outputs else ""
            score = judge._parse_score(text)
            results.append({**item, "judge_score": score, "judge_text": text})
        else:
            score = (await judge.verify_completions(problem, [completion]))[0]
            results.append({**item, "judge_score": score})
    return results


async def main():
    parser = argparse.ArgumentParser(description="Score IFBench/IFEval completions with LLM judge")
    parser.add_argument("--config", type=str, required=True, help="Config with judge role + prompt")
    parser.add_argument("--completions", type=str, required=True, help="JSONL completions file")
    parser.add_argument("--n", type=int, default=12, help="Number of samples to score")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default="", help="Optional JSONL output file")
    parser.add_argument("--debug", action="store_true", help="Print raw judge outputs")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    judge_role_cfg = _find_judge_role(cfg)

    verifier_cfg = cfg.get("verifier", {})
    judge_kwargs = (verifier_cfg.get("judge") or {}).get("kwargs", {})
    if not judge_kwargs:
        raise ValueError("Missing verifier.judge.kwargs in config")

    items = _load_completions(Path(args.completions))
    if not items:
        raise ValueError("No completions loaded")

    random.seed(args.seed)
    sample_n = min(args.n, len(items))
    sample = random.sample(items, sample_n)

    if not ray.is_initialized():
        ray.init()

    actor = VLLMEngineRank.remote(engine_kwargs=judge_role_cfg, replica_id=0)
    await asyncio.get_event_loop().run_in_executor(None, ray.get, actor.ready.remote())
    max_concurrent = judge_role_cfg.get("max_concurrent_per_replica", 8)
    handle = VLLMHandle([actor], name="judge", max_concurrent_per_replica=max_concurrent)
    judge = LLMJudgeVerifier(handle, **judge_kwargs)

    scored = await _score_samples(judge, sample, debug=args.debug)

    print("\n=== LLM Judge Sample Scores ===")
    for item in scored:
        base_score = item.get("verification", {}).get("score")
        print(f"- {item.get('id','?')}  judge={item['judge_score']:.2f}  base={base_score}")
        prompt_snip = item.get("prompt", "")[:140].replace("\n", " ")
        completion_snip = item.get("completion", "")[:140].replace("\n", " ")
        print(f"  prompt: {prompt_snip}")
        print(f"  completion: {completion_snip}")
        if args.debug and "judge_text" in item:
            judge_snip = item.get("judge_text", "")[:200].replace("\n", " ")
            print(f"  judge_text: {judge_snip}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w") as f:
            for item in scored:
                f.write(json.dumps(item) + "\n")
        print(f"\nWrote scored samples to {out_path}")

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
