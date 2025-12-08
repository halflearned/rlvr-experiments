
import os
from torchtitan.tools.logging import init_logger
from rlvr_experiments.datasets.gsm8k import register_gsm8k

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args, _ = parser.parse_known_args()

init_logger()
register_gsm8k()

from rlvr_experiments.train import TitanRLTrainer
from torchtitan.config import ConfigManager
from rlvr_experiments.weight_update import VLLMSyncWeightUpdate



def build_titan_dataloader(trainer):
    job_config = trainer.job_config
    train_spec = trainer.train_spec 

    # figure out dp_world_size / dp_rank exactly like original Trainer
    parallel_dims = trainer.parallel_dims
    world_mesh = parallel_dims.world_mesh

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_world_size = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_world_size, dp_rank = 1, 0

    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=trainer.tokenizer,
        job_config=job_config,
    )
    return dataloader

rank = int(os.environ.get("RANK", "0"))
print(f"Starting training on rank {rank}...")

job_config = ConfigManager().parse_args(["--job.config-file", args.config])
trainer = TitanRLTrainer(job_config)
print(f"[rank: {rank}] Trainer built, preparing dataloader...")
dataloader = build_titan_dataloader(trainer)
print(f"[rank: {rank}] Dataloader built, starting training loop...")
for input_dict, labels in dataloader:
    print(f"[rank: {rank}] Starting training step...")
    logits = trainer.forward_step(input_dict)
    print(f"[rank: {rank}] Forward step complete.")
    loss = logits.mean()
    trainer.backward_step(loss)
    trainer.optimizer_step()
    break

print("Training step complete!!!")

print("Initializing client to push weights to vLLM server...")
from trl.extras.vllm_client import VLLMClient
client = VLLMClient(base_url="http://vllm:8000", connection_timeout=3000)
print("Pushing weights to vLLM server...")
weight_updater = VLLMSyncWeightUpdate([client])
weight_updater.push_weights(trainer.hf_state_dict())
print("Weights pushed to vLLM server successfully (hopefully)!")