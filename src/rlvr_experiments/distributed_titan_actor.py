"""
distributed_titan_actor.py - Clean distributed model with weight sync support
"""
import ray
import torch
import torch.distributed as dist
import os
import asyncio
from typing import Tuple
from rlvr_experiments.weight_sync import WeightSyncManager


@ray.remote(num_gpus=1, num_cpus=2)
class TitanModelRank:
    """Single rank of distributed TitanModel with optional weight sync"""
    
    def __init__(self, rank: int, world_size: int, config_path: str, group_name: str = "default"):
        """
        Args:
            rank: Rank within this model's group (0-7)
            world_size: Size of this model's group (8)
            group_name: Name for logging ("trainer" or "reference")
        """
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.config_path = config_path
        self.model = None
        
        # Weight sync is optional and separate from model
        self.weight_sync_manager = WeightSyncManager()
        
        # Set env vars for this rank
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = '0'  # Always 0 - Ray gives us one GPU visible as cuda:0
        os.environ['WORLD_SIZE'] = str(world_size)
        
    def initialize_process_group(self, master_addr: str, master_port: str):
        """Initialize torch.distributed and TitanModel"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Initialize TitanModel - it sees its own world (e.g., 8 ranks)
        from rlvr_experiments.model import TitanModel
        from torchtitan.config import ConfigManager
        
        job_config = ConfigManager().parse_args(["--job.config-file", self.config_path])
        self.model = TitanModel(job_config, trainable=(self.group_name == "trainer"))
        
        print(f"[{self.group_name}] Rank {self.rank}/{self.world_size} initialized")
        
        return {"rank": self.rank, "status": "ready"}
    
    # ========== Weight Sync Methods (separate from model) ==========
    
    def initialize_with_weight_sync(
        self,
        master_addr: str,
        master_port: str,
        weight_sync_host: str,
        weight_sync_port: int,
        weight_sync_world_size: int,
        weight_sync_rank: int
    ):
        """
        Initialize both torch.distributed AND weight sync in one call.
        This is more efficient than calling initialize_process_group and init_weight_sync separately.
        
        Args:
            master_addr: Master address for torch.distributed
            master_port: Port for torch.distributed
            weight_sync_host: Host for weight sync group
            weight_sync_port: Port for weight sync group
            weight_sync_world_size: Total ranks in weight sync group
            weight_sync_rank: This rank's position in weight sync group
        """
        # 1. Initialize torch.distributed and model
        result = self.initialize_process_group(master_addr, master_port)
        
        # 2. Initialize weight sync
        self.weight_sync_manager.init_communicator(
            host=weight_sync_host,
            port=weight_sync_port,
            world_size=weight_sync_world_size,
            my_rank=weight_sync_rank,
            device=self.model.device
        )
        
        result["weight_sync_status"] = "ready"
        result["weight_sync_rank"] = weight_sync_rank
        return result
    
    def init_weight_sync(self, host: str, port: int, world_size: int, my_rank: int):
        """
        Initialize weight sync communicator (separate from torch.distributed).
        
        Note: Prefer using initialize_with_weight_sync() for single-pass initialization.
        
        Args:
            host: Master host for sync group
            port: Port for sync group (different from MASTER_PORT)
            world_size: Total ranks in sync group (16 for trainer+reference)
            my_rank: Position in sync group (0-7 for trainer, 8-15 for reference)
        """
        self.weight_sync_manager.init_communicator(
            host=host,
            port=port,
            world_size=world_size,
            my_rank=my_rank,
            device=self.model.device
        )
        return {"status": "weight_sync_ready", "rank": my_rank}
    
    def broadcast_weights_as_sender(self):
        """
        Broadcast this rank's weights to all ranks in the sync group.
        Called by trainer ranks to send their weights.
        """
        self.weight_sync_manager.broadcast_weights(
            model_parameters=self.model.model_parts[0].parameters(),
            src_rank=self.weight_sync_manager.sync_group_rank
        )
    
    def broadcast_weights_as_receiver(self, src_rank: int):
        """
        Receive weights from a source rank in the sync group.
        Called by reference ranks to receive weights from trainer.
        
        Args:
            src_rank: Source rank to receive from (trainer rank in sync group)
        """
        self.weight_sync_manager.broadcast_weights(
            model_parameters=self.model.model_parts[0].parameters(),
            src_rank=src_rank
        )
    
    # ========== Generic Model Access ==========
    
    def call_method(self, method_name: str, *args, **kwargs):
        """Generic method forwarder for model methods"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_process_group first.")
        method = getattr(self.model, method_name)
        return method(*args, **kwargs)
    
    def get_attr(self, attr_name: str):
        """Generic attribute getter for model attributes"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_process_group first.")
        return getattr(self.model, attr_name)
    
    def get_rank(self):
        """Get this actor's rank"""
        return self.rank


class DistributedModelHandle:
    """
    Handle for distributed model that automatically broadcasts method calls to all ranks.
    Uses __getattr__ magic to forward any method to all actors.
    """
    
    def __init__(self, actors, name: str = "model"):
        """
        Args:
            actors: List of TitanModelRank actors
            name: Name for this model (for logging)
        """
        self.actors = actors
        self.name = name
        self._world_size = len(actors)
        
        # Known simple attributes that should be fetched directly (not called as methods)
        # TODO: Could make this automatic via inspect
        self._simple_attrs = {'tokenizer', 'device', 'step', 'job_config', 'model_args'}
    
    def __getattr__(self, attr_name):
        """
        Intercept attribute/method access and broadcast to all ranks.
        Returns an awaitable for method calls, or the value for simple attributes.
        """
        # For known simple attributes, fetch directly from rank 0
        if attr_name in self._simple_attrs:
            return ray.get(self.actors[0].get_attr.remote(attr_name))
        
        # For everything else, assume it's a method and return async wrapper
        async def distributed_method(*args, **kwargs):
            # Call method on all ranks
            futures = [
                actor.call_method.remote(attr_name, *args, **kwargs)
                for actor in self.actors
            ]
            
            # Await all results
            results = await asyncio.gather(*[self._ray_to_async(f) for f in futures])
            
            # Return rank 0's result (convention for distributed training)
            return results[0]
        
        return distributed_method
    
    async def sync_weights_to(self, target_model: 'DistributedModelHandle'):
        """
        Fast NCCL weight sync from this model to target model.
        Both models must have initialized their weight_sync groups first.
        
        Pattern: Each trainer rank i broadcasts to reference rank i
        (they receive via the shared sync group communicator)
        
        Args:
            target_model: Target model to sync weights to
        """
        if len(self.actors) != len(target_model.actors):
            raise ValueError(
                f"Models must have same number of ranks: "
                f"{len(self.actors)} vs {len(target_model.actors)}"
            )
        
        # Trainer ranks broadcast their weights
        send_futures = [
            actor.broadcast_weights_as_sender.remote()
            for actor in self.actors
        ]
        
        # Reference ranks receive from corresponding trainer ranks
        # Reference rank i receives from trainer rank i
        recv_futures = [
            target_model.actors[i].broadcast_weights_as_receiver.remote(src_rank=i)
            for i in range(len(target_model.actors))
        ]
        
        # Wait for all transfers to complete
        await asyncio.gather(
            *[self._ray_to_async(f) for f in send_futures + recv_futures]
        )
    
    async def _ray_to_async(self, object_ref):
        """Convert Ray ObjectRef to async/await"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, object_ref)
    
    def __repr__(self):
        return f"DistributedModelHandle(name='{self.name}', world_size={self._world_size})"


def create_distributed_model(
    config_path: str,
    world_size: int = 8,
    group_name: str = "trainer",
    master_port: int = 29500
) -> DistributedModelHandle:
    """
    Create a distributed model across multiple Ray actors.
    
    Args:
        config_path: Path to torchtitan config
        world_size: Number of processes (GPUs)
        group_name: Name for this model group
        master_port: Port for torch.distributed coordination
        
    Returns:
        DistributedModelHandle with clean async interface
    """
    master_addr = ray.util.get_node_ip_address()
    
    print(f"Creating {group_name} with {world_size} ranks...")
    actors = []
    for rank in range(world_size):
        actor = TitanModelRank.remote(
            rank=rank,
            world_size=world_size,
            config_path=config_path,
            group_name=group_name
        )
        actors.append(actor)
    
    print(f"Initializing torch.distributed for {group_name}...")
    init_futures = [
        actor.initialize_process_group.remote(master_addr, str(master_port))
        for actor in actors
    ]
    ray.get(init_futures)
    
    print(f"✓ {group_name} initialized with {world_size} ranks")
    
    return DistributedModelHandle(actors, name=group_name)


def create_two_distributed_models_with_sync(
    config_path: str,
    ranks_per_model: int = 8,
    trainer_master_port: int = 29500,
    reference_master_port: int = 29600,
    weight_sync_port: int = 51216,
) -> Tuple[DistributedModelHandle, DistributedModelHandle]:
    """
    Create trainer and reference models in SEPARATE torch.distributed worlds,
    with a shared NCCL weight sync group (like vLLM).
    
    Single-pass initialization: all actors are initialized with both their
    torch.distributed groups AND weight sync in one parallel batch.
    
    This allows:
    - Each model to have its own parallelism config (no WORLD_SIZE conflicts)
    - Fast NCCL weight transfers (100+ GB/s) via separate sync communicator
    
    Args:
        config_path: Path to torchtitan config
        ranks_per_model: Number of ranks per model (e.g., 8)
        trainer_master_port: Port for trainer's torch.distributed
        reference_master_port: Port for reference's torch.distributed
        weight_sync_port: Port for the weight sync NCCL group
        
    Returns:
        Tuple of (trainer_handle, reference_handle)
    """
    master_addr = ray.util.get_node_ip_address()
    sync_world_size = ranks_per_model * 2  # 16 total (8 trainer + 8 reference)
    
    print(f"Creating {ranks_per_model * 2} actors...")
    
    # ========== Create all actors ==========
    trainer_actors = [
        TitanModelRank.remote(
            rank=rank,
            world_size=ranks_per_model,
            config_path=config_path,
            group_name="trainer"
        )
        for rank in range(ranks_per_model)
    ]
    
    reference_actors = [
        TitanModelRank.remote(
            rank=rank,
            world_size=ranks_per_model,
            config_path=config_path,
            group_name="reference"
        )
        for rank in range(ranks_per_model)
    ]
    
    # ========== Initialize everything in one parallel batch ==========
    print("Initializing torch.distributed and weight sync for all actors...")
    
    init_futures = []
    
    # Trainer: sync ranks 0-7
    for i, actor in enumerate(trainer_actors):
        init_futures.append(
            actor.initialize_with_weight_sync.remote(
                master_addr=master_addr,
                master_port=str(trainer_master_port),
                weight_sync_host=master_addr,
                weight_sync_port=weight_sync_port,
                weight_sync_world_size=sync_world_size,
                weight_sync_rank=i  # 0-7
            )
        )
    
    # Reference: sync ranks 8-15
    for i, actor in enumerate(reference_actors):
        init_futures.append(
            actor.initialize_with_weight_sync.remote(
                master_addr=master_addr,
                master_port=str(reference_master_port),
                weight_sync_host=master_addr,
                weight_sync_port=weight_sync_port,
                weight_sync_world_size=sync_world_size,
                weight_sync_rank=i + ranks_per_model  # 8-15
            )
        )
    
    # Single wait for everything
    results = ray.get(init_futures)
    
    print(f"✓ Trainer initialized ({ranks_per_model} ranks)")
    print(f"✓ Reference initialized ({ranks_per_model} ranks)")
    print(f"✓ Weight sync group initialized ({sync_world_size} ranks on port {weight_sync_port})")
    
    return (
        DistributedModelHandle(trainer_actors, name="trainer"),
        DistributedModelHandle(reference_actors, name="reference")
    )