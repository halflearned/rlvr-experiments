"""
weight_sync.py - Separate NCCL communicator for fast weight synchronization
Inspired by vLLM's weight update mechanism
"""
import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncManager:
    """
    Manages a separate NCCL communicator for fast weight synchronization.
    
    This creates an independent NCCL group (on a different port) from the main
    torch.distributed world, allowing fast parameter broadcasts between models
    that may be in separate training worlds.
    
    Example:
        # Trainer rank 0 and Reference rank 0 can sync weights even if they're
        # in different torch.distributed worlds (different MASTER_PORTs)
    """
    
    def __init__(self):
        self.communicator = None
        self.sync_group_rank = None
        self.device = None
        self.world_size = None
    
    def init_communicator(self, host: str, port: int, world_size: int, my_rank: int, device: torch.device):
        """
        Initialize a separate NCCL group for weight syncing.
        
        Args:
            host: Master host for sync group (usually same as main distributed)
            port: Port for sync group (MUST be different from torch.distributed port)
            world_size: Total ranks in sync group (e.g., 16 = 8 trainer + 8 reference)
            my_rank: This rank's position in sync group
                     - Trainer ranks: 0-7
                     - Reference ranks: 8-15 (for 8-rank models)
            device: Device to use for communication (e.g., cuda:0)
        """
        if self.communicator is not None:
            raise RuntimeError("Weight sync communicator already initialized. Call close() first.")
        
        # Create stateless process group (like vLLM does)
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=my_rank,
            world_size=world_size
        )
        
        # Wrap in PyNccl communicator for efficient GPU-GPU transfers
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.sync_group_rank = my_rank
        self.device = device
        self.world_size = world_size
        
        print(f"[WeightSync] Rank {my_rank}/{world_size} initialized on {device}")
    
    def broadcast_weights(self, model_parameters, src_rank: int):
        """
        Broadcast weights using NCCL.
        
        All ranks call this with the same src_rank:
        - The src_rank sends its parameters
        - All other ranks receive and overwrite their parameters
        
        Args:
            model_parameters: Iterable of torch.nn.Parameter objects
            src_rank: Source rank to broadcast from (in the sync group)
        """
        if self.communicator is None:
            raise RuntimeError("Weight sync communicator not initialized. Call init_communicator() first.")
        
        for param in model_parameters:
            # NCCL broadcast: src sends, all others receive
            self.communicator.broadcast(param.data, src=src_rank)
    
    def close(self):
        """Close the weight sync communicator and free resources"""
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.sync_group_rank = None
            self.device = None
            self.world_size = None