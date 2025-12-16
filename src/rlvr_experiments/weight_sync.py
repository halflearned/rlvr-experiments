"""
weight_sync.py - Separate NCCL communicator for fast weight synchronization.

Inspired by vLLM's weight update mechanism in trl.

This module defines WeightSyncManager, which creates an independent NCCL
communicator (on a different port from torch.distributed) so that different
torch.distributed "worlds" (e.g., Titan trainer vs Titan reference) can still
synchronize weights efficiently over GPU-GPU links.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncManager:
    """
    Manages a separate NCCL communicator for fast weight synchronization.

    This creates an independent NCCL group (on a different port) from the main
    torch.distributed world, allowing fast parameter broadcasts between models
    that may be in separate training worlds.

    Example usage (per process):

        wsm = WeightSyncManager()
        wsm.init_communicator(
            host=master_addr,
            port=sync_port,
            world_size=sync_world_size,
            my_rank=sync_rank,
            device=torch.device("cuda", 0),
        )

        # Later, when we want to sync:
        wsm.sync_model_from_src(model, src_rank=0)
    """

    def __init__(self) -> None:
        self.communicator: Optional[PyNcclCommunicator] = None
        self.sync_group_rank: Optional[int] = None
        self.device: Optional[torch.device] = None
        self.world_size: Optional[int] = None

    # --------------------------------------------------------------------- #
    # Initialization / teardown
    # --------------------------------------------------------------------- #
    def init_communicator(
        self,
        host: str,
        port: int,
        world_size: int,
        my_rank: int,
        device: torch.device,
    ) -> None:
        """
        Initialize a separate NCCL group for weight syncing.

        Args:
            host: Master host for sync group (usually same as main distributed).
            port: Port for sync group (MUST be different from torch.distributed port).
            world_size: Total ranks in sync group (e.g., 16 = 8 trainer + 8 reference).
            my_rank: This rank's position in sync group.
            device: Device to use for communication (e.g., cuda:0).
        """
        if self.communicator is not None:
            raise RuntimeError(
                "Weight sync communicator already initialized. Call close() first."
            )

        # StatelessProcessGroup creates a standalone NCCL process group, similar to vLLM.
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=my_rank,
            world_size=world_size,
        )

        # Wrap in PyNcclCommunicator for efficient GPU-GPU transfers.
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.sync_group_rank = my_rank
        self.device = device
        self.world_size = world_size

        print(f"[WeightSync] Rank {my_rank}/{world_size} initialized on {device}")

    def close(self) -> None:
        """Close the weight sync communicator and free resources."""
        if self.communicator is not None:
            del self.communicator
            self.communicator = None

        self.sync_group_rank = None
        self.device = None
        self.world_size = None

    # --------------------------------------------------------------------- #
    # Core API
    # --------------------------------------------------------------------- #
    def sync_model_from_src(self, model: torch.nn.Module, src_rank: int) -> None:
        """
        Synchronize all parameters of `model` from a given source rank.

        All ranks in the sync group must call this with the same `src_rank`.
        The source rank sends its parameters; all others receive and overwrite.

        This method:
          * Handles both DTensors (by using their local shard) and regular tensors.
          * Ensures tensors are CUDA and contiguous before calling NCCL.

        Args:
            model: The model whose parameters will be synchronized.
            src_rank: Source rank to broadcast from (in the sync group).
        """
        if self.communicator is None:
            raise RuntimeError(
                "Weight sync communicator not initialized. "
                "Call init_communicator() first."
            )

        params: Iterable[torch.nn.Parameter] = model.parameters()
        my_rank = self.sync_group_rank
        assert my_rank is not None, "sync_group_rank must be set if communicator exists"

        params = list(params)
        print(
            f"[WeightSync Rank {my_rank}] Syncing {len(params)} parameters from src={src_rank}"
        )

        for i, param in enumerate(params):
            # In Titan, DTensors often expose a `_local_tensor` attribute.
            # If present, only broadcast the local shard.
            if hasattr(param, "_local_tensor"):
                local = param._local_tensor
                if not isinstance(local, torch.Tensor):
                    raise RuntimeError(
                        f"Rank {my_rank}: Param {i} has _local_tensor of type "
                        f"{type(local)}; expected torch.Tensor"
                    )
                if not local.is_cuda:
                    raise RuntimeError(
                        f"Rank {my_rank}: DTensor local shard {i} is on {local.device}, "
                        f"expected CUDA device"
                    )

                self.communicator.broadcast(local, src=src_rank)
            else:
                # Regular dense tensor
                data = param.data
                if not data.is_cuda:
                    raise RuntimeError(
                        f"Rank {my_rank}: Param {i} is on {data.device}, expected CUDA"
                    )

                if not data.is_contiguous():
                    data = data.contiguous()
                    param.data = data  # keep the contiguous copy attached

                self.communicator.broadcast(data, src=src_rank)

        print(f"[WeightSync Rank {my_rank}] Sync completed successfully")

    # --------------------------------------------------------------------- #
    # Convenience
    # --------------------------------------------------------------------- #
    @property
    def is_initialized(self) -> bool:
        """Return True if communicator has been set up."""
        return self.communicator is not None
