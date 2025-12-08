
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



job_config = ConfigManager().parse_args(["--job.config-file", args.config])
trainer = TitanRLTrainer(job_config)
print("Trainer built, preparing dataloader...")
dataloader = build_titan_dataloader(trainer)
print("Dataloader built, starting training loop...")
for input_dict, labels in dataloader:
    print("Starting training step...")
    logits = trainer.forward_step(input_dict)
    print("Forward step complete.")
    loss = logits.mean()
    trainer.backward_step(loss)
    trainer.optimizer_step()

print("Training step complete!!!")

# from trl.extras.vllm_client import VLLMClient
# client = VLLMClient(base_url="http://vllm:8000", connection_timeout=120)
