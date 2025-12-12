from __future__ import annotations
import asyncio
import os
import ray
import torch
from typing import Any, Dict, List, Tuple
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from rlvr_experiments.weight_sync import WeightSyncManager

@ray.remote(num_gpus=1)
class TitanModelRank:
    def __init__(self, rank: int, world_size: int, config_path: str, group_name: str = "default") -> None:
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.config_path = config_path
        self.model = None
        self.sync_managers: Dict[str, WeightSyncManager] = {}
        self._sync_cache: Dict[str, Any] | None = None

    def initialize_process_group(self, master_addr: str, master_port: str) -> Dict[str, Any]:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        
        # Ray isolates 1 GPU per actor, so for the actor, its local index is 0
        os.environ["LOCAL_RANK"] = "0"

        from rlvr_experiments.model import TitanModel
        from torchtitan.config import ConfigManager
        job_config = ConfigManager().parse_args(["--job.config-file", self.config_path])
        self.model = TitanModel(job_config, trainable=(self.group_name == "trainer"))
        return {"status": "ready"}

    def add_sync_channel(self, channel_name: str, host: str, port: int, world_size: int, my_rank: int) -> None:
        manager = WeightSyncManager()
        manager.init_communicator(host=host, port=port, world_size=world_size, my_rank=my_rank, device=self.model.device)
        self.sync_managers[channel_name] = manager

    def get_hf_param_names_and_shapes(self) -> List[Tuple[str, str, Tuple[int, ...]]]:
        """COLLECTIVE: All ranks must participate."""
        hf_state = self.model.hf_state_dict()
        return [(k, str(v.dtype).split(".")[-1], tuple(v.shape)) for k, v in hf_state.items()]

    def prepare_sync_state(self) -> None:
        """
        COLLECTIVE: Run the expensive FSDP all-gather once and cache the result.
        This must be called on ALL ranks simultaneously.
        """
        if self.model is None: raise RuntimeError("Model not initialized")
        # --- FIX 1: Run FSDP All-Gather ONLY ONCE ---
        self._sync_cache = self.model.hf_state_dict()
        print(f"[{self.group_name} Rank {self.rank}] Sync state prepared (Size: {len(self._sync_cache)} tensors).")

    def clear_sync_state(self) -> None:
        """Frees up the large cached state dict."""
        self._sync_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def broadcast_hf_param(self, hf_name: str, channel: str) -> None:
        """
        FAST LANE: Pulls from the cached state instead of re-gathering the whole model.
        """
        if self._sync_cache is None:
            raise RuntimeError("Sync state not prepared. Call prepare_sync_state first.")
            
        # --- FIX 2: Pull from cached state (FAST) ---
        tensor = self._sync_cache[hf_name] 

        # 1. Collective TP gather (Still required if TP > 1)
        if isinstance(tensor, DTensor):
            # This is the only remaining collective call inside the loop.
            tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim).to_local()
        
        # 2. NCCL Broadcast (Rank 0 pushes)
        self.sync_managers[channel].communicator.broadcast(tensor.contiguous(), src=0)


    def recv_hf_param(self, hf_name: str, dtype_str: str, shape: Tuple[int, ...], channel: str) -> None:
        """RECEIVER: RECEIVE -> SHARD -> INGEST"""
        if self.model is None: raise RuntimeError("Model not initialized")
        
        # 1. Prepare global buffer
        dtype = getattr(torch, dtype_str)
        full_tensor = torch.empty(shape, dtype=dtype, device=self.model.device)
        
        # 2. Receive from Trainer Rank 0 (Dense Global Weight)
        self.sync_managers[channel].communicator.broadcast(full_tensor, src=0)
        torch.cuda.current_stream().synchronize()

        # 3. Shard and Load
        with torch.no_grad():
            # Map names using your adapter
            titan_sync_dict = self.model.sd_adapter.from_hf({hf_name: full_tensor})
            model_params = dict(self.model.model_parts[0].named_parameters())
            
            for titan_name, incoming_val in titan_sync_dict.items():
                if titan_name in model_params:
                    target = model_params[titan_name]
                    
                    if isinstance(target, DTensor):
                        # === THE FIX ===
                        # Distribute the global 'incoming_val' to match target's sharding
                        sharded_src = distribute_tensor(
                            incoming_val, 
                            target.device_mesh, 
                            target.placements
                        )
                        target.copy_(sharded_src)
                    else:
                        # Normal tensor copy
                        target.copy_(incoming_val)
                        

class DistributedModelHandle:
    def __init__(self, actors: List, name: str = "model"):
        self.actors = actors
        self.name = name

    async def get_hf_param_info(self):
        """Collective: Ensures all ranks participate in metadata gather."""
        futures = [a.get_hf_param_names_and_shapes.remote() for a in self.actors]
        results = await asyncio.gather(*futures)
        return results[0]

# --- Orchestrators ---

# In src/rlvr_experiments/distributed_titan_actor.py

async def sync_titan_to_vllm(trainer, vllm, channel="fast_vllm"):
    print(f"--- [{trainer.name} -> vLLM] Syncing ---")
    
    # 1. CRITICAL: Prepare the state once (Collective)
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in trainer.actors])
    
    params = await trainer.get_hf_param_info()
    for name, dtype, shape in params:
        await asyncio.gather(
            *[a.broadcast_hf_param.remote(name, channel) for a in trainer.actors],
            vllm._actor.recv_named_param.remote(name, dtype, shape, 0)
        )
    
    # 2. Cleanup the large cached state
    await asyncio.gather(*[a.clear_sync_state.remote() for a in trainer.actors])


async def sync_titan_to_titan(src, target, channel="slow_ref"):
    print(f"--- [{src.name} -> {target.name}] Syncing ---")
    
    # 1. CRITICAL: Prepare the state once (Collective)
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in src.actors])
    
    params = await src.get_hf_param_info()
    for name, dtype, shape in params:
        await asyncio.gather(
            *[a.broadcast_hf_param.remote(name, channel) for a in src.actors],
            *[a.recv_hf_param.remote(name, dtype, shape, channel) for a in target.actors]
        )
        
    # 2. Cleanup the large cached state
    await asyncio.gather(*[a.clear_sync_state.remote() for a in src.actors])
    
    print(f"✓ {src.name} -> {target.name} complete.")

    
def create_titan_group(config_path, name, ranks, port):
    master_addr = ray.util.get_node_ip_address()
    actors = [TitanModelRank.options(num_gpus=1).remote(r, ranks, config_path, name) for r in range(ranks)]
    ray.get([a.initialize_process_group.remote(master_addr, str(port)) for a in actors])
    return DistributedModelHandle(actors, name=name)