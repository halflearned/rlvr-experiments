"""
distributed_titan_actor.py - Clean distributed model with __getattr__ magic
"""
import ray
import torch
import torch.distributed as dist
import os
import asyncio


@ray.remote(num_gpus=1, num_cpus=2)
class TitanModelRank:
    """Single rank of distributed TitanModel"""
    
    def __init__(self, rank: int, world_size: int, config_path: str, group_name: str = "default"):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.config_path = config_path
        self.model = None
        
        # Set env vars for this rank
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = '0'  # Always 0! Ray gives us one GPU visible as cuda:0
        os.environ['WORLD_SIZE'] = str(world_size)
        
    def initialize_process_group(self, master_addr: str, master_port: str):
        """Initialize torch.distributed after all ranks are created"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Now initialize TitanModel
        from rlvr_experiments.model import TitanModel
        from torchtitan.config import ConfigManager
        
        job_config = ConfigManager().parse_args(["--job.config-file", self.config_path])
        self.model = TitanModel(job_config, trainable=True)
        
        print(f"[{self.group_name}] Rank {self.rank}/{self.world_size} initialized")
        
        return {"rank": self.rank, "status": "ready"}
    
    def call_method(self, method_name: str, *args, **kwargs):
        """Generic method forwarder - this is what Ray will call"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_process_group first.")
        method = getattr(self.model, method_name)
        return method(*args, **kwargs)
    
    def get_attr(self, attr_name: str):
        """Generic attribute getter"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_process_group first.")
        return getattr(self.model, attr_name)
    
    def get_rank(self):
        """Get this actor's rank"""
        return self.rank
    

class DistributedModelHandle:
    """
    Handle for distributed model that automatically broadcasts method calls to all ranks.
    """
    
    def __init__(self, actors, name: str = "model"):
        self.actors = actors
        self.name = name
        self._world_size = len(actors)
        # Define known attributes that should be fetched directly
        self._simple_attrs = {'tokenizer', 'device', 'step', 'job_config', 'model_args'}
    
    def __getattr__(self, attr_name):
        """
        Intercept attribute/method access and broadcast to all ranks.
        """
        # For known simple attributes, fetch directly
        if attr_name in self._simple_attrs:
            return ray.get(self.actors[0].get_attr.remote(attr_name))
        
        # For everything else, assume it's a method and return async wrapper
        async def distributed_method(*args, **kwargs):
            futures = [
                actor.call_method.remote(attr_name, *args, **kwargs)
                for actor in self.actors
            ]
            results = await asyncio.gather(*[self._ray_to_async(f) for f in futures])
            return results[0]
        
        return distributed_method
    
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
    # Get master address from Ray
    master_addr = ray.util.get_node_ip_address()
    
    # Create all rank actors
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
    
    # Initialize all ranks (sets up torch.distributed)
    print(f"Initializing torch.distributed for {group_name}...")
    init_futures = [
        actor.initialize_process_group.remote(master_addr, str(master_port))
        for actor in actors
    ]
    results = ray.get(init_futures)
    
    print(f"✓ {group_name} initialized with {world_size} ranks")
    
    # Return convenient handle
    return DistributedModelHandle(actors, name=group_name)


# Helper function to add to generic_actor.py
def create_distributed_actor(
    cls: type,
    name: str,
    config_path: str,
    world_size: int = 8,
    master_port: int = 29500,
    actor_config_path: str = None
) -> DistributedModelHandle:
    """
    Generic function to create any distributed model as Ray actors.
    
    This is the distributed equivalent of create_named_actor.
    
    Args:
        cls: The class to distribute (e.g., TitanModel)
        name: Name for this distributed model
        config_path: Path to model config
        world_size: Number of ranks/GPUs
        master_port: Port for coordination
        actor_config_path: Optional Ray actor config (not used yet, but for future)
        
    Returns:
        DistributedModelHandle
    """
    # For now, this just wraps create_distributed_model
    # But it could be extended to read actor_config_path for customization
    return create_distributed_model(
        config_path=config_path,
        world_size=world_size,
        group_name=name,
        master_port=master_port
    )