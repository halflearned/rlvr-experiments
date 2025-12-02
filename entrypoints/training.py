# ----- dataset stuff

from torchtitan.datasets import hf_datasets
from torchtitan.datasets.hf_datasets import DatasetConfig
from datasets import load_dataset

def _load_gsm8k(dataset_path: str):
    return load_dataset(dataset_path, "main", split="train", streaming=False)

def _process_gsm8k(sample: dict) -> str:
    # whatever you decided earlier
    return f"Question: {sample['question']}\n\nSolution: {sample['answer']}"

def register_gsm8k():
    hf_datasets.DATASETS["gsm8k"] = DatasetConfig(
        path="openai/gsm8k",
        loader=_load_gsm8k,
        sample_processor=_process_gsm8k,
    )

register_gsm8k()

# -------- trainer stuff
from torchtitan.train import Trainer
from torchtitan.tools.logging import init_logger
from torchtitan.config import ConfigManager

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args, _ = parser.parse_known_args()

init_logger()
config = ConfigManager().parse_args(["--job.config-file", args.config])
trainer = Trainer(config)
trainer.train()
