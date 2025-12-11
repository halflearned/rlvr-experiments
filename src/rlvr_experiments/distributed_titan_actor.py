"""
distributed_titan_actor.py - Distributed Titan model actor with weight sync support.

This module defines:
  * TitanModelRank    - a Ray actor that hosts a single rank of a TitanModel.
  * DistributedModelHandle - a convenience wrapper to broadcast method calls
                             to all ranks in a distributed model.
  * ModelGroupSpec    - configuration for a group of ranks that form one model.
  * create_distributed_model / create_distributed_models_with_sync /
    create_trainer_and_reference_with_sync - helper constructors.

The design allows:
  * Each model group (trainer, reference, etc.) to have its own
    torch.distributed world (distinct MASTER_PORTs).
  * All groups to share a separate NCCL-based weight sync communicator,
    implemented by WeightSyncManager (see weight_sync.py).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import ray
import torch

from rlvr_experiments.weight_sync import WeightSyncManager


@ray.remote(num_gpus=1, num_cpus=2)
class TitanModelRank:
    """
    Single rank of a distributed TitanModel with optional weight synchronization.

    Each Ray actor:
      * Initializes its own torch.distributed process group, with WORLD_SIZE
        equal to the number of ranks in its model group.
      * Builds a TitanModel configured by a JobConfig.
      * Optionally initializes a separate NCCL communicator for weight sync.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        config_path: str,
        group_name: str = "default",
    ) -> None:
        """
        Args:
            rank: Rank within this model group (0..world_size-1).
            world_size: Size of this model group.
            config_path: Path to the Titan config file (TOML).
            group_name: Logical name for this model ("trainer", "reference", etc.).
        """
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.config_path = config_path
        self.model = None

        # Weight sync is optional and separate from torch.distributed.
        self.weight_sync_manager = WeightSyncManager()

        # Environment variables for torch.distributed.
        os.environ["RANK"] = str(rank)
        # Ray exposes one GPU per actor; treat that as local rank 0.
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize_process_group(self, master_addr: str, master_port: str) -> Dict[str, Any]:
        """
        Initialize torch.distributed and construct the TitanModel for this rank.

        This uses a full Titan JobConfig parsed from `config_path`.
        """
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        from rlvr_experiments.model import TitanModel
        from torchtitan.config import ConfigManager

        job_config = ConfigManager().parse_args(
            ["--job.config-file", self.config_path]
        )

        # Trainer vs reference: only trainer is trainable.
        self.model = TitanModel(job_config, trainable=(self.group_name == "trainer"))

        print(
            f"[{self.group_name}] Rank {self.rank}/{self.world_size} initialized "
            f"(MASTER_ADDR={master_addr}, MASTER_PORT={master_port})"
        )

        return {"rank": self.rank, "status": "ready"}

    def initialize_with_weight_sync(
        self,
        master_addr: str,
        master_port: str,
        weight_sync_host: str,
        weight_sync_port: int,
        weight_sync_world_size: int,
        weight_sync_rank: int,
    ) -> Dict[str, Any]:
        """
        Initialize both torch.distributed AND the weight-sync communicator.

        This is more efficient (and less error-prone) than calling
        initialize_process_group() and init_weight_sync() separately.

        Args:
            master_addr: Master address for torch.distributed.
            master_port: Port for torch.distributed.
            weight_sync_host: Host for sync group (usually same as master_addr).
            weight_sync_port: Port for sync NCCL group (MUST differ from master_port).
            weight_sync_world_size: Total ranks in sync group (e.g., 16).
            weight_sync_rank: This rank's index in the sync group (0..world_size-1).
        """
        # 1. Initialize torch.distributed and TitanModel.
        result = self.initialize_process_group(master_addr, master_port)

        # 2. Initialize weight sync communicator on the model's device.
        if self.model is None:
            raise RuntimeError("Model must be initialized before weight sync.")

        self.weight_sync_manager.init_communicator(
            host=weight_sync_host,
            port=weight_sync_port,
            world_size=weight_sync_world_size,
            my_rank=weight_sync_rank,
            device=self.model.device,
        )

        result["weight_sync_status"] = "ready"
        result["weight_sync_rank"] = weight_sync_rank
        return result

    def init_weight_sync(
        self,
        host: str,
        port: int,
        world_size: int,
        my_rank: int,
    ) -> Dict[str, Any]:
        """
        Initialize ONLY the weight-sync communicator.

        Prefer using initialize_with_weight_sync() when possible so that
        both torch.distributed and weight sync are set up in a single pass.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call initialize_process_group() first."
            )

        self.weight_sync_manager.init_communicator(
            host=host,
            port=port,
            world_size=world_size,
            my_rank=my_rank,
            device=self.model.device,
        )

        return {"status": "weight_sync_ready", "rank": my_rank}

    # ------------------------------------------------------------------ #
    # Weight sync
    # ------------------------------------------------------------------ #
    def call_broadcast_weights(self, src_rank: int) -> Dict[str, Any]:
        """
        Participate in a weight sync broadcast from a source rank.

        All ranks in the sync group must call this with the same `src_rank`.
        Under the hood this calls WeightSyncManager.sync_model_from_src(...)
        on this rank's TitanModel.

        Args:
            src_rank: Source rank in the sync group whose parameters are used.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # TitanModel stores its parallelized model as model_parts[0].
            self.weight_sync_manager.sync_model_from_src(
                self.model.model_parts[0],
                src_rank=src_rank,
            )
            return {
                "status": "ok",
                "rank": self.rank,
                "sync_group_rank": self.weight_sync_manager.sync_group_rank,
            }
        except Exception as e:
            print(
                f"[WeightSync ERROR group={self.group_name} rank={self.rank}] "
                f"Failed to sync weights: {e}"
            )
            import traceback

            traceback.print_exc()
            raise

    # ------------------------------------------------------------------ #
    # Generic accessors
    # ------------------------------------------------------------------ #
    def call_method(self, method_name: str, *args, **kwargs):
        """Generic method forwarder for methods on self.model."""
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call initialize_process_group() first."
            )
        method = getattr(self.model, method_name)
        return method(*args, **kwargs)

    def get_attr(self, attr_name: str):
        """Generic attribute getter for attributes on self.model."""
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call initialize_process_group() first."
            )
        return getattr(self.model, attr_name)

    def get_rank(self) -> int:
        """Return this actor's rank within its model group."""
        return self.rank


class DistributedModelHandle:
    """
    Handle for a distributed model that automatically broadcasts method calls to all ranks.

    Uses __getattr__ "magic" to:
      * Fetch simple attributes (tokenizer, device, etc.) from rank 0 only.
      * Turn method accesses into async functions that call all actors in parallel.
    """

    def __init__(self, actors: List[ray.actor.ActorHandle], name: str = "model") -> None:
        """
        Args:
            actors: List of TitanModelRank actor handles.
            name: Human-readable name for this model group ("trainer", "reference").
        """
        self.actors = actors
        self.name = name
        self._world_size = len(actors)

        # Attributes that should be fetched from rank 0 only (not invoked).
        self._simple_attrs = {"tokenizer", "device", "step", "job_config", "model_args"}

    def __getattr__(self, attr_name: str):
        """
        Intercept attribute/method access and broadcast to all ranks.

        If attr_name is in _simple_attrs, return its value from rank 0.
        Otherwise, assume it is a method name and return an async wrapper
        that calls that method on all ranks and returns rank 0's result.
        """
        # Simple attribute: fetch from rank 0 only.
        if attr_name in self._simple_attrs:
            return ray.get(self.actors[0].get_attr.remote(attr_name))

        # Otherwise assume method; return an async function.
        async def distributed_method(*args, **kwargs):
            futures = [
                actor.call_method.remote(attr_name, *args, **kwargs)
                for actor in self.actors
            ]
            results = await asyncio.gather(
                *[self._ray_to_async(f) for f in futures]
            )
            # Convention: return rank 0's result.
            return results[0]

        return distributed_method

    async def sync_weights_to(self, target_model: "DistributedModelHandle") -> None:
        """
        Synchronize weights from this model to target_model via NCCL weight sync.

        Current protocol:
          * All ranks in BOTH models participate in the same weight-sync world.
          * We broadcast from sync_group_rank=0 (the first trainer rank).
          * Each rank calls call_broadcast_weights(src_rank=0) in parallel.

        Args:
            target_model: The DistributedModelHandle to sync weights to.
        """
        if len(self.actors) != len(target_model.actors):
            raise ValueError(
                f"Models must have same number of ranks: "
                f"{len(self.actors)} vs {len(target_model.actors)}"
            )

        futures = []

        # Trainer ranks (this model): source is sync_group_rank 0.
        for actor in self.actors:
            futures.append(actor.call_broadcast_weights.remote(src_rank=0))

        # Reference ranks (target model): also participate, receive from src=0.
        for actor in target_model.actors:
            futures.append(actor.call_broadcast_weights.remote(src_rank=0))

        await asyncio.gather(*[self._ray_to_async(f) for f in futures])

    async def _ray_to_async(self, object_ref):
        """Convert a Ray ObjectRef into an awaitable result."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, object_ref)

    def __repr__(self) -> str:
        return f"DistributedModelHandle(name='{self.name}', world_size={self._world_size})"


# ---------------------------------------------------------------------- #
# Group creation helpers
# ---------------------------------------------------------------------- #
@dataclass
class ModelGroupSpec:
    """
    Specification for one distributed model group.

    This is used by create_distributed_models_with_sync to create multiple
    models (trainer, reference, eval, etc.) that share a single weight-sync
    communicator but have their own torch.distributed worlds.
    """

    name: str
    ranks: int
    master_port: int
    sync_rank_offset: int = 0  # starting rank index in the weight-sync world


def create_distributed_model(
    config_path: str,
    world_size: int = 8,
    group_name: str = "trainer",
    master_port: int = 29500,
) -> DistributedModelHandle:
    """
    Create a single distributed model (one torch.distributed world) across Ray actors.

    Args:
        config_path: Path to Titan config file.
        world_size: Number of ranks (actors) in this model group.
        group_name: Logical group name ("trainer", "reference", etc.).
        master_port: Port for torch.distributed coordination.

    Returns:
        DistributedModelHandle wrapping all ranks.
    """
    master_addr = ray.util.get_node_ip_address()

    print(f"Creating {group_name} with {world_size} ranks...")
    actors: List[ray.actor.ActorHandle] = [
        TitanModelRank.remote(
            rank=rank,
            world_size=world_size,
            config_path=config_path,
            group_name=group_name,
        )
        for rank in range(world_size)
    ]

    print(f"Initializing torch.distributed for {group_name}...")
    init_futures = [
        actor.initialize_process_group.remote(master_addr, str(master_port))
        for actor in actors
    ]
    ray.get(init_futures)

    print(f"✓ {group_name} initialized with {world_size} ranks")
    return DistributedModelHandle(actors, name=group_name)


def create_distributed_models_with_sync(
    config_path: str,
    group_specs: List[ModelGroupSpec],
    weight_sync_port: int,
) -> Dict[str, DistributedModelHandle]:
    """
    Generic creator for multiple distributed model groups that share a single
    NCCL weight-sync communicator.

    Each group has its own torch.distributed world (different MASTER_PORT),
    but all actors across all groups join the same weight-sync group.

    Args:
        config_path: Path to Titan config file.
        group_specs: List of ModelGroupSpec describing each group.
        weight_sync_port: Port for the NCCL weight-sync communicator.

    Returns:
        Dict mapping group name -> DistributedModelHandle.
    """
    master_addr = ray.util.get_node_ip_address()
    sync_world_size = sum(spec.ranks for spec in group_specs)

    print(
        f"Creating {sync_world_size} actors across {len(group_specs)} groups "
        f"with weight sync world_size={sync_world_size} on port {weight_sync_port}..."
    )

    all_actors: Dict[str, List[ray.actor.ActorHandle]] = {}
    init_futures = []

    for spec in group_specs:
        print(
            f"  - Group '{spec.name}': ranks={spec.ranks}, "
            f"master_port={spec.master_port}, sync_rank_offset={spec.sync_rank_offset}"
        )

        group_actors: List[ray.actor.ActorHandle] = []
        for local_rank in range(spec.ranks):
            sync_rank = spec.sync_rank_offset + local_rank

            actor = TitanModelRank.remote(
                rank=local_rank,
                world_size=spec.ranks,
                config_path=config_path,
                group_name=spec.name,
            )
            group_actors.append(actor)

            init_futures.append(
                actor.initialize_with_weight_sync.remote(
                    master_addr=master_addr,
                    master_port=str(spec.master_port),
                    weight_sync_host=master_addr,
                    weight_sync_port=weight_sync_port,
                    weight_sync_world_size=sync_world_size,
                    weight_sync_rank=sync_rank,
                )
            )

        all_actors[spec.name] = group_actors

    print("Initializing torch.distributed and weight sync for all groups...")
    ray.get(init_futures)
    print("✓ All groups initialized")
    print(
        f"✓ Weight sync group initialized (world_size={sync_world_size}, "
        f"port={weight_sync_port})"
    )

    # Wrap each group in a DistributedModelHandle.
    return {
        name: DistributedModelHandle(actors, name=name)
        for name, actors in all_actors.items()
    }


def create_trainer_and_reference_with_sync(
    config_path: str,
    ranks_per_model: int = 8,
    trainer_master_port: int = 29500,
    reference_master_port: int = 29600,
    weight_sync_port: int = 51216,
) -> Tuple[DistributedModelHandle, DistributedModelHandle]:
    """
    Convenience wrapper: create 'trainer' and 'reference' distributed models
    in separate torch.distributed worlds, with a shared NCCL weight-sync group.

    Layout:
        - trainer ranks:   0 .. ranks_per_model - 1
        - reference ranks: ranks_per_model .. 2 * ranks_per_model - 1
    """
    group_specs = [
        ModelGroupSpec(
            name="trainer",
            ranks=ranks_per_model,
            master_port=trainer_master_port,
            sync_rank_offset=0,
        ),
        ModelGroupSpec(
            name="reference",
            ranks=ranks_per_model,
            master_port=reference_master_port,
            sync_rank_offset=ranks_per_model,
        ),
    ]

    handles = create_distributed_models_with_sync(
        config_path=config_path,
        group_specs=group_specs,
        weight_sync_port=weight_sync_port,
    )

    return handles["trainer"], handles["reference"]
