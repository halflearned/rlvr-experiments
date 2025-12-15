from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import torch

# TODO: generalize
class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer):
        self.data = load_dataset("openai/gsm8k", "main")[split]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]["question"].strip()
        template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            template,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            #max_length=seq_len, TODO: needed?
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }


def build_dataloader_fn(*, dp_world_size, dp_rank, tokenizer, job_config):
    dataset = GSM8KDataset(split="train", tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
    )
    return DataLoader(
        dataset,
        batch_size=job_config.training.batch_size_per_rank,
        sampler=sampler,
        drop_last=True,
    )
