"""
Group Relative Policy Optimization (GRPO) for RLVR.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Any
import time

from rlvr_experiments.inference import VLLMClient


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_model(config: Dict[str, Any], device: torch.device):
    """Load model for training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = config["training"]["model_name"]
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Manual placement
    )
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def collect_rollouts(
    vllm_client: VLLMClient,
    prompts: List[str],
    n_samples: int = 4,
    temperature: float = 0.8,
) -> List[List[str]]:
    """
    Collect rollouts from vLLM inference server.
    
    Args:
        vllm_client: Client for vLLM server
        prompts: List of prompts to generate from
        n_samples: Number of samples per prompt
        temperature: Sampling temperature
    
    Returns:
        List of lists of completions
    """
    print(f"Collecting {n_samples} rollouts for {len(prompts)} prompts...")
    
    completions = vllm_client.generate(
        prompts=prompts,
        temperature=temperature,
        n=n_samples,
        max_tokens=512,
    )
    
    return completions


def compute_rewards(
    prompts: List[str],
    completions: List[List[str]],
    config: Dict[str, Any],
) -> List[List[float]]:
    """
    Compute rewards for each completion.
    
    For now, this is a placeholder that returns random rewards.
    You'll replace this with your actual verifier.
    
    Args:
        prompts: Original prompts
        completions: Generated completions
        config: Training config
    
    Returns:
        List of lists of rewards (one per completion)
    """
    import random
    
    print(f"Computing rewards for {len(prompts)} prompts...")
    
    rewards = []
    for prompt, prompt_completions in zip(prompts, completions):
        prompt_rewards = []
        for completion in prompt_completions:
            # Placeholder: random reward
            # TODO: Replace with actual verifier
            reward = random.random()
            prompt_rewards.append(reward)
        rewards.append(prompt_rewards)
    
    return rewards


def compute_grpo_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    completions: List[List[str]],
    rewards: List[List[float]],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute GRPO loss.
    
    GRPO uses group-relative advantages:
    For each group of samples from the same prompt, compute advantage
    relative to the group mean.
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        prompts: Original prompts
        completions: Generated completions
        rewards: Rewards for each completion
        device: Device to run on
    
    Returns:
        Loss tensor
    """
    total_loss = 0.0
    num_samples = 0
    
    for prompt, prompt_completions, prompt_rewards in zip(prompts, completions, rewards):
        # Compute group baseline (mean reward)
        baseline = sum(prompt_rewards) / len(prompt_rewards)
        advantages = [r - baseline for r in prompt_rewards]
        
        for completion, advantage in zip(prompt_completions, advantages):
            # Tokenize
            full_text = prompt + completion
            tokens = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            
            prompt_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            
            prompt_len = prompt_tokens.input_ids.shape[1]
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**tokens)
                logits = outputs.logits
                
                # Get log probs for completion tokens only
                completion_logits = logits[:, prompt_len-1:-1, :]
                completion_tokens = tokens.input_ids[:, prompt_len:]
                
                log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=2,
                    index=completion_tokens.unsqueeze(-1),
                ).squeeze(-1)
                
                # GRPO objective: maximize advantage-weighted log prob
                loss = -(token_log_probs.sum() * advantage)
                
                total_loss += loss
                num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0)


def train_grpo(config: Dict[str, Any]):
    """Main GRPO training loop."""
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"Rank {rank}/{world_size} on device {device}")
    
    # Load model
    model, tokenizer = load_model(config, device)
    
    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-5),
    )
    
    # Setup vLLM client
    vllm_base_url = config["training"].get("vllm_base_url", "http://localhost:8000/v1")
    vllm_client = VLLMClient(base_url=vllm_base_url)
    
    # Load dataset (placeholder)
    # TODO: Replace with your actual dataset
    train_prompts = [
        "Solve: 2+2=",
        "What is the capital of France?",
        "Write a Python function to compute fibonacci:",
    ] * 10  # Repeat for demo
    
    # Training config
    num_steps = config["training"].get("num_steps", 1000)
    batch_size = config["training"].get("batch_size", 8)
    n_samples_per_prompt = config["training"].get("n_samples_per_prompt", 4)
    temperature = config["training"].get("temperature", 0.8)
    save_frequency = config["training"].get("save_frequency", 100)
    
    print(f"Training for {num_steps} steps")
    print(f"Batch size: {batch_size}")
    print(f"Samples per prompt: {n_samples_per_prompt}")
    
    # Training loop
    for step in range(num_steps):
        step_start = time.time()
        
        # Sample batch of prompts
        batch_prompts = train_prompts[
            (step * batch_size) % len(train_prompts):
            ((step + 1) * batch_size) % len(train_prompts)
        ]
        
        # Collect rollouts from vLLM
        completions = collect_rollouts(
            vllm_client,
            batch_prompts,
            n_samples=n_samples_per_prompt,
            temperature=temperature,
        )
        
        # Compute rewards
        rewards = compute_rewards(batch_prompts, completions, config)
        
        # Compute loss and update
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            model,
            tokenizer,
            batch_prompts,
            completions,
            rewards,
            device,
        )
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        step_time = time.time() - step_start
        
        if rank == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | Time: {step_time:.2f}s")
        
        # Save checkpoint
        if rank == 0 and (step + 1) % save_frequency == 0:
            checkpoint_dir = f"/workspace/checkpoints/step-{step+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            print(f"Saved checkpoint to {checkpoint_dir}")
            
            # TODO: Optionally update vLLM server with new checkpoint
            # update_vllm_server(checkpoint_dir)
    
    print("Training complete!")
    
    if world_size > 1:
        dist.destroy_process_group()