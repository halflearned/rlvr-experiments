import os
import argparse
import time
import torch
import torch.distributed as dist

def get_args():
    parser = argparse.ArgumentParser(description="NCCL/EFA Bandwidth Benchmark")
    parser.add_argument("--role", type=str, required=True, choices=["sender", "receiver"],
                        help="'receiver' is Rank 0 (Server), 'sender' is Rank 1 (Trainer)")
    parser.add_argument("--master-addr", type=str, required=True, 
                        help="IP address of the receiver (Rank 0) node")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--size-gb", type=float, default=1.0, 
                        help="Size of the payload to transfer in GB")
    parser.add_argument("--iterations", type=int, default=10, 
                        help="Number of transfer iterations for averaging")
    return parser.parse_args()

def run_benchmark():
    args = get_args()
    
    # 1. Setup Distributed Environment
    # We assume a world_size of 2 for this simple point-to-point test.
    rank = 0 if args.role == "receiver" else 1
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    print(f"[{args.role}] Initializing Process Group (backend='nccl')...")
    
    # EFA/NCCL requires CUDA tensors, so we must bind to a GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. EFA/NCCL requires GPUs.")
    
    local_device = torch.device(f"cuda:{0}") # Simplification: always use GPU 0
    torch.cuda.set_device(local_device)

    dist.init_process_group(backend="nccl", world_size=2, rank=rank)
    print(f"[{args.role}] Connected!")

    # 2. Prepare Data
    # Calculate number of float16 elements needed for the target GB size
    # 2 bytes per float16
    num_elements = int((args.size_gb * 1024**3) / 2)
    
    print(f"[{args.role}] Allocating {args.size_gb} GB tensor ({num_elements} fp16 elements)...")
    data = torch.randn(num_elements, dtype=torch.float16, device=local_device)
    
    # Ensure all allocs are done
    torch.cuda.synchronize()

    # 3. Warmup
    print(f"[{args.role}] Warming up...")
    dist.broadcast(data, src=1) # Rank 1 (Sender) broadcasts to Rank 0
    torch.cuda.synchronize()

    # 4. Benchmark Loop
    print(f"[{args.role}] Starting benchmark ({args.iterations} iterations)...")
    
    # We use CUDA events for high-precision timing on GPU
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(args.iterations):
        # The Transfer: Sender (1) -> Receiver (0)
        dist.broadcast(data, src=1)
    end_event.record()
    
    # Wait for completion
    torch.cuda.synchronize()
    
    # 5. Calculate Stats
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_sec = elapsed_ms / 1000.0
    avg_time_per_iter = elapsed_sec / args.iterations
    
    # Throughput = (Size * Iterations) / Total Time
    total_data_gb = args.size_gb * args.iterations
    throughput_gb_s = total_data_gb / elapsed_sec
    throughput_gbits = throughput_gb_s * 8  # Convert to Gigabits/sec for network comparison

    print("-" * 40)
    print(f"Results for role: {args.role.upper()}")
    print("-" * 40)
    print(f"Payload Size    : {args.size_gb} GB")
    print(f"Avg Time/Iter   : {avg_time_per_iter:.4f} sec")
    print(f"Throughput      : {throughput_gb_s:.2f} GB/s")
    print(f"Throughput      : {throughput_gbits:.2f} Gbit/s")
    print("-" * 40)

    dist.destroy_process_group()

if __name__ == "__main__":
    run_benchmark()