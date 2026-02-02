"""Minimal SFT training script using HuggingFace Trainer.

Usage:
    python scripts/sft_hf_minimal.py --lr 1e-5 --data data/ifeval_sft_20k_hardest.jsonl --gpu 0
    python scripts/sft_hf_minimal.py --lr 1e-5 --overfit 1  # overfit on 1 example to verify training works
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024, limit=None):
        with open(path) as f:
            data = [json.loads(l) for l in f]
        if limit:
            data = data[:limit]

        self.examples = []
        skipped = 0
        for d in data:
            prompt = d["prompt"]
            completion = d["completion"]

            # Tokenize prompt and completion separately
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            comp_ids = tokenizer.encode(completion, add_special_tokens=False)

            # Truncate if needed
            total = len(prompt_ids) + len(comp_ids)
            if total > max_len:
                # Keep prompt, truncate completion
                comp_ids = comp_ids[:max_len - len(prompt_ids)]
                if len(comp_ids) < 1:
                    skipped += 1
                    continue

            input_ids = prompt_ids + comp_ids
            # Labels: -100 for prompt tokens, actual ids for completion
            labels = [-100] * len(prompt_ids) + comp_ids

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        print(f"Loaded {len(self.examples)} examples (skipped {skipped} too long)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)  # pad with 0
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--data", default="/efs/rlvr-experiments/data/ifeval_sft_20k_hardest.jsonl")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overfit", type=int, default=0, help="If >0, use only N examples and train many epochs to verify overfitting")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.output_dir is None:
        args.output_dir = f"/tmp/sft_hf_lr{args.lr}"
        if args.overfit:
            args.output_dir += f"_overfit{args.overfit}"

    print(f"Loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Always load in float32; mixed precision handled by Trainer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    limit = args.overfit if args.overfit > 0 else None
    dataset = SFTDataset(args.data, tokenizer, max_len=args.max_len, limit=limit)

    if args.overfit > 0:
        # Overfit mode: repeat the tiny dataset many times
        args.epochs = 200
        args.max_steps = -1
        print(f"\n*** OVERFIT MODE: {args.overfit} examples, {args.epochs} epochs ***\n")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=1,
        save_strategy="epoch",
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print(f"\nStarting training: lr={args.lr}, batch={args.batch_size}x{args.grad_accum}, max_len={args.max_len}")
    print(f"Dataset: {len(dataset)} examples, epochs={training_args.num_train_epochs}")
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done!")


if __name__ == "__main__":
    main()
