import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser("Multi-node NCCL bandwidth benchmark")

    p.add_argument(
        "--op",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_gather"],
        help="Collective op to benchmark",
    )
    p.add_argument(
        "--size-gb",
        type=float,
        default=2.0,
        help="Tensor size per rank, in GB (GiB actually: 1 GB = 1024**3 bytes).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of timed iterations.",
    )
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations (not timed).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Tensor dtype.",
    )

    return p.parse_args()


def get_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def init_dist():
    # torchrun sets these
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # Use explicit init_method via rdzv_endpoint from torchrun (c10d).
    # torchrun will already have created the store; backend='nccl' is enough.
    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)

    rank, world_size, local_rank = init_dist()
    is_leader = rank == 0

    if is_leader:
        print(
            f"[leader] world_size={world_size}, "
            f"local_rank={local_rank}, op={args.op}, "
            f"size_gb={args.size_gb}, iters={args.iters}, warmup={args.warmup_iters}, "
            f"dtype={dtype}"
        )

    device = torch.device(f"cuda:{local_rank}")

    # Allocate tensor per rank
    bytes_per_gb = 1024**3
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elems = int((args.size_gb * bytes_per_gb) / bytes_per_elem)

    if is_leader:
        print(
            f"[leader] Allocating tensor with {num_elems} elements "
            f"({args.size_gb:.3f} GB per rank, dtype={dtype})"
        )

    tensor = torch.randn(num_elems, dtype=dtype, device=device)

    if args.op == "all_gather":
        # all_gather output buffer: list of tensors, one per rank
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    torch.cuda.synchronize()
    dist.barrier()

    # Warmup
    for _ in range(args.warmup_iters):
        if args.op == "all_reduce":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif args.op == "all_gather":
            dist.all_gather(gather_list, tensor)
        else:
            raise ValueError

    torch.cuda.synchronize()
    dist.barrier()

    # Timed loop: use CUDA events for on-GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.iters):
        if args.op == "all_reduce":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif args.op == "all_gather":
            dist.all_gather(gather_list, tensor)
        else:
            raise ValueError
    end_event.record()

    torch.cuda.synchronize()
    dist.barrier()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000.0
    avg_s = elapsed_s / args.iters

    # Compute algorithmic bytes, following NCCL tests conventions.

    bytes_per_rank = num_elems * bytes_per_elem  # per-rank tensor size in bytes

    if args.op == "all_reduce":
        # Ring all_reduce: each rank sends and receives ~2 * (N-1)/N * bytes_per_rank.
        # Total "network" bytes (sum over all ranks) per iter:
        #   B_total = 2 * (N - 1) * bytes_per_rank
        total_bytes_per_iter = 2.0 * (world_size - 1) * bytes_per_rank
    elif args.op == "all_gather":
        # all_gather: each rank sends (N-1)*bytes_per_rank,
        # total across all ranks:
        #   B_total = N * (N - 1) * bytes_per_rank
        total_bytes_per_iter = world_size * (world_size - 1) * bytes_per_rank
    else:
        raise ValueError

    total_bytes = total_bytes_per_iter * args.iters
    gb = total_bytes / (1024**3)
    gbps = gb / elapsed_s
    gbit_s = gbps * 8.0

    if is_leader:
        print("-" * 60)
        print(f"Collective       : {args.op}")
        print(f"World size       : {world_size}")
        print(f"Dtype            : {dtype}")
        print(f"Tensor size/rank : {args.size_gb:.3f} GB")
        print(f"Total alg. bytes : {gb:.2f} GB over {args.iters} iters")
        print(f"Total time       : {elapsed_s:.4f} s")
        print(f"Avg time/iter    : {avg_s:.6f} s")
        print(f"Aggregate BW     : {gbps:.2f} GB/s")
        print(f"Aggregate BW     : {gbit_s:.2f} Gbit/s")
        print("-" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
