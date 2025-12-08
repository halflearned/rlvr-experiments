from torchtitan.datasets import hf_datasets
from torchtitan.datasets.hf_datasets import DatasetConfig
from datasets import load_dataset

def _load_gsm8k(dataset_path: str):
    return load_dataset(dataset_path, "main", split="train", streaming=False)

def _process_gsm8k(sample: dict) -> str:
    return f"Question: {sample['question']}\n\nSolution: {sample['answer']}"

def register_gsm8k():
    hf_datasets.DATASETS["gsm8k"] = DatasetConfig(
        path="openai/gsm8k",
        loader=_load_gsm8k,
        sample_processor=_process_gsm8k,
    )
