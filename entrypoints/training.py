import argparse
import torch

from torchtitan.tools.logging import init_logger
from torchtitan.config import ConfigManager

from rlvr_experiments.model import TitanModel
from rlvr_experiments.weight_update import VLLMSyncWeightUpdate
from rlvr_experiments.vllm_client import VLLMClient
from rlvr_experiments.datasets.gsm8k import register_gsm8k
from rlvr_experiments.data_utils import build_titan_dataloader
from rlvr_experiments.rewards import DummyReward
from rlvr_experiments.losses import GRPOLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args, _ = parser.parse_known_args()
    
    init_logger()
    
    # model
    job_config = ConfigManager().parse_args(["--job.config-file", args.config])
    trainer = TitanModel(job_config)

    # data
    register_gsm8k()
    dataloader = build_titan_dataloader(trainer)

    # just for now! we'll stand this up as a separate service later
    reference = trainer

    # init vllm client for inference
    client = VLLMClient(base_url="http://vllm:8000", connection_timeout=3000)
    weight_updater = VLLMSyncWeightUpdate([client])
    
    # other trainer stuff
    verifier = DummyReward()
    loss_fn = GRPOLoss(beta=0.1, eps=0.2)
    
    # 'GRPO' loop
    for i, (input_dict, labels) in enumerate(dataloader):

        # Generate from vllm server
        output = client.generate(
            prompts=["Write a 100-line poem about the sea."] * input_dict["input"].size(0),
            n=5,
            max_tokens=128,
            temperature=1.0,
        )

        # Parse vllm output
        output["completion_ids"] = torch.tensor(output["completion_ids"])
        input_dict = {
            "input": output["completion_ids"],
            "attention_mask": (output["completion_ids"] != trainer.tokenizer.pad_token_id).long()
        }

        # ----- EVERYTHING HERE CAN RUN ASYNCHRONOUSLY! ------

        # Trainer forward
        trainer_logits = trainer.forward_step(input_dict)

        # Reference forward
        with torch.no_grad():
            reference_logits = reference.forward_step(input_dict)
        
        # Verifiers
        rewards = [verifier("a", "b", "c") for _ in range(trainer_logits.size(0))]

        # ------ END OF ASYNC SECTION ------

        # Loss
        loss = loss_fn(
            rewards=torch.tensor(rewards, device=trainer_logits.device),
            trainer_logprobs=torch.log_softmax(trainer_logits, dim=-1),
            inference_logprobs=torch.log_softmax(reference_logits, dim=-1),
            reference_logprobs=torch.log_softmax(reference_logits, dim=-1),
            padding_mask=input_dict["attention_mask"],
        )

        # Backward and optimizer steps
        trainer.backward_step(loss)
        trainer.optimizer_step()

        # Push weights to vllm server
        weight_updater.push_weights(trainer.hf_state_dict())
        
        if i >= 2:  # early break for testing
            break
    
    print("all done!")


if __name__ == "__main__":
    main()