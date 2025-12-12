import asyncio
from typing import Tuple
import torch

# STRICT: We expect vLLM V1 structure.
from vllm.v1.worker.gpu_worker import Worker
from rlvr_experiments.weight_sync import WeightSyncManager

class WeightSyncVLLMWorker(Worker):
    """
    Custom vLLM V1 Worker.
    Strictly follows vLLM V1 architecture (subclassing v1.worker.gpu_worker).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_sync = WeightSyncManager()
        
    def init_weight_sync(self, host: str, port: int, world_size: int, rank: int):
        """
        RPC Method: Join the Global Weight Sync NCCL group.
        """
        # In V1 GPUWorker, 'self.rank' is the global rank within the engine instance.
        my_sync_rank = rank + self.rank
        
        print(f"[Worker {self.rank}] Joining Weight Sync Group as Rank {my_sync_rank}...")
        self.weight_sync.init_communicator(
            host=host, 
            port=port, 
            world_size=world_size, 
            my_rank=my_sync_rank, 
            device=self.device
        )
        return True

    def update_weight(self, name: str, dtype_str: str, shape: Tuple[int, ...], src_rank: int):
        dtype = getattr(torch, dtype_str)
        staging = torch.empty(shape, dtype=dtype, device=self.device)
        self.weight_sync.communicator.broadcast(staging, src=src_rank)
        torch.cuda.current_stream().synchronize()

        # In-place copy to preserve CUDA Graph memory addresses
        params_dict = dict(self.model_runner.model.named_parameters())
        if name in params_dict:
            with torch.no_grad():
                params_dict[name].copy_(staging)
        else:
            # Fallback for complex vLLM internal mappings
            self.model_runner.model.load_weights(weights=[(name, staging)])
        return True