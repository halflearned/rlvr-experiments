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


import os
import requests
from openai import OpenAI
from time import sleep

def wait_for_vllm(max_retries=60, delay=2):
    base_url = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
    health_url = base_url.replace("/v1", "/health")

    for i in range(max_retries):
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        sleep(delay)

    raise RuntimeError(f"vLLM not ready after {max_retries * delay} seconds")


def call_vllm(prompt):
    base_url = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
    client = OpenAI(base_url=base_url, api_key="dummy")
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0,
        logprobs=True,
    )
    print("GOT RESPONSE FROM VLLM:", resp)
    return resp.choices[0].message.content



wait_for_vllm()
print(f"vLLM rollout: {call_vllm('Say hello from the RLVR trainer.')[:200]!r}")


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
