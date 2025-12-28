import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncManager:
    """NCCL communicator for weight sync between models in separate distributed worlds."""

    def __init__(self, host: str, port: int, world_size: int, rank: int, device: torch.device) -> None:
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.sync_group_rank = rank
        self.device = device
        self.world_size = world_size
