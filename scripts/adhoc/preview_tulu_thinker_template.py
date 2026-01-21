#!/usr/bin/env python
"""Preview Tulu thinker R1-style chat formatting on dataset samples."""

from __future__ import annotations

import argparse
import itertools

from rlvr_experiments.chat_templates import CHAT_TEMPLATES
from rlvr_experiments.data import DATASET_LOADERS
from rlvr_experiments.kshot_parser import parse_kshot_messages


def build_messages(prompt: str, problem: dict, system_prompt: str) -> list[dict]:
    messages = problem.get("messages")
    if isinstance(messages, list) and messages:
        messages = [dict(m) for m in messages]
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai_rlvr")
    parser.add_argument("--subset", default=None, help="Dataset subset for allenai_rlvr (gsm8k, math, ifeval).")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--chat-template", default="tulu_thinker_r1_style")
    parser.add_argument("--summary", action="store_true", help="Only print summary statistics.")
    parser.add_argument("--parse-kshot", action="store_true", help="Parse k-shot plaintext into multi-turn messages.")
    args = parser.parse_args()

    if args.chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {args.chat_template}")

    loader = DATASET_LOADERS.get(args.dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader_kwargs = {}
    if args.dataset == "allenai_rlvr" and args.subset:
        loader_kwargs["datasets"] = [args.subset]

    ds = loader(**loader_kwargs)
    formatter = CHAT_TEMPLATES[args.chat_template]

    total = 0
    ok = 0
    failed = 0
    missing_messages = 0
    invalid_messages = 0
    has_system = 0
    has_assistant = 0
    role_counts = {}
    dataset_counts = {}
    missing_think_suffix = 0
    exceptions = []
    parsed_kshot = 0

    for row in itertools.islice(ds.iter_rows(), args.limit):
        total += 1
        problem = row.get("problem", {})
        source_dataset = problem.get("source_dataset", problem.get("dataset_name", "unknown"))
        dataset_counts[source_dataset] = dataset_counts.get(source_dataset, 0) + 1

        system_prompt = problem.get("system_prompt", "") or ""
        messages = build_messages(row.get("prompt", ""), problem, system_prompt)
        if args.parse_kshot:
            source_name = str(source_dataset).lower()
            parsed = parse_kshot_messages(messages[0]["content"], source_name)
            if parsed:
                messages = parsed
                parsed_kshot += 1

        if "messages" not in problem:
            missing_messages += 1
        elif not isinstance(problem.get("messages"), list):
            invalid_messages += 1

        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "system":
                has_system += 1
            if role == "assistant":
                has_assistant += 1

        try:
            formatted = formatter(messages, tokenizer=None, add_generation_prompt=True)
            ok += 1
            if not formatted.rstrip().endswith("<think>"):
                missing_think_suffix += 1
        except Exception as exc:
            failed += 1
            exceptions.append(f"{type(exc).__name__}: {exc}")
            formatted = None

        if not args.summary:
            prompt_id = problem.get("prompt_id", "unknown")
            print("=" * 80)
            print(f"prompt_id: {prompt_id}")
            print(f"source_dataset: {source_dataset}")
            print("- raw prompt")
            raw_prompt = row.get("prompt", "")[: args.max_chars]
            print(raw_prompt + ("..." if len(row.get("prompt", "")) > args.max_chars else ""))
            print("- formatted")
            if formatted is None:
                print("<formatting failed>")
            else:
                formatted = formatted[: args.max_chars]
                print(formatted + ("..." if len(formatted) >= args.max_chars else ""))

    print("=" * 80)
    print("summary:")
    print(f"  total={total} ok={ok} failed={failed}")
    print(f"  missing_messages={missing_messages} invalid_messages={invalid_messages}")
    print(f"  has_system_messages={has_system} has_assistant_messages={has_assistant}")
    print(f"  missing_think_suffix={missing_think_suffix}")
    print(f"  dataset_counts={dataset_counts}")
    print(f"  role_counts={role_counts}")
    if args.parse_kshot:
        print(f"  parsed_kshot={parsed_kshot}")
    if exceptions:
        print("  exceptions:")
        for exc in exceptions[:5]:
            print(f"    - {exc}")


if __name__ == "__main__":
    main()
