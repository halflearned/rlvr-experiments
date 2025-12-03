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


from rlvr_experiments.vllm_client import VLLMClient
print("Creating vLLM client...")
client = VLLMClient(base_url="http://vllm:8000", model="Qwen/Qwen3-0.6B")
print("Calling vLLM...")
output = client.generate("Say hello from the RLVR trainer.")
print("vLLM call succeeded.", output)



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

import torch
import torch.distributed as dist
import os

rank = dist.get_rank()


# 1) init communicator
#Need to map using this adapter (or its reverse, actually):
# https://github.com/pytorch/torchtitan/blob/b39377f9fe33865fefb9bf64a33f6d74a598be87/torchtitan/models/qwen3/model/state_dict_adapter.py#L28
if rank == 0:
    client.init_communicator(device=0) 
    print("INITIALIZED COMMUNICATOR") 
    name = "model.layers.0.self_attn.q_proj.weight"
    dummy = torch.randn(2048, 1024).cuda(0)
    client.update_named_param(name, dummy)
    print("UPDATED MODEL PARAMS") 
    client.close_communicator()
    print("CLOSED COMMUNICATOR!")
print("SUCCESS! Weight sync smoke test completed.")





trainer.train()
