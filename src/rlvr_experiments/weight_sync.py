from __future__ import annotations

from typing import Optional

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncManager:
    """
    Manages a separate NCCL communicator for fast weight synchronization.

    This creates an independent NCCL group (on a different port) from the main
    torch.distributed world, allowing fast parameter broadcasts between models
    that may be in separate training worlds.

    The communicator is used directly via `manager.communicator.broadcast(...)`.
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
        rank: int,
        device: torch.device,
    ) -> None:
        """
        Initialize a separate NCCL group for weight syncing.

        Args:
            host: Master host for sync group (usually same as main distributed).
            port: Port for sync group (MUST be different from torch.distributed port).
            world_size: Total ranks in sync group (e.g., 16 = 8 trainer + 8 reference).
            rank: This rank's position in sync group.
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
            rank=rank,
            world_size=world_size,
        )

        # Wrap in PyNcclCommunicator for efficient GPU-GPU transfers.
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.sync_group_rank = rank
        self.device = device
        self.world_size = world_size

        print(f"[WeightSync] Rank {rank}/{world_size} initialized on {device}")

    def close(self) -> None:
        """Close the weight sync communicator and free resources."""
        if self.communicator is not None:
            del self.communicator
            self.communicator = None

        self.sync_group_rank = None
        self.device = None
        self.world_size = None

    @property
    def is_initialized(self) -> bool:
        """Return True if communicator has been set up."""
        return self.communicator is not None
