import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[rank {rank}] world_size = {world_size}")

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    print(f"[rank {rank}] mesh: {mesh}")


    # example of a sharded matrix multiplication
    # let's say we have a batch size of size 10
    # this is data parallel
    x_global = torch.randn(10, 4)
    x_sharded = distribute_tensor(
        x_global,
        device_mesh=mesh,
        placements=[Shard(0), Replicate()]
    )

    # we'll tp this along the column dimension
    w_global = torch.arange(16, dtype=torch.float32).view(4, 4)
    w_sharded = distribute_tensor(
        w_global,
        device_mesh=mesh,
        placements=[Replicate(), Shard(1)]
    )

    y = x_sharded @ w_sharded

    y_local = y.to_local()

    # broadcast the result to all devices
    y_full = y.redistribute(
        device_mesh=mesh,
        placements=[Replicate(), Replicate()]
    )

    if rank == 0:
        y_final = y_full.to_local()

    print(f"[rank {rank}] w_global:\n{w_global}")
    print(f"[rank {rank}] w_sharded:\n{w_sharded}")
    print(f"[rank {rank}] y_local:\n{y_local}")
    if rank == 0:
        print(f"[rank {rank}] y_final:\n{y_final}")

if __name__ == "__main__":
    main()
