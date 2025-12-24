#!/usr/bin/env python3
"""
Benchmark script for testing different weight save/load strategies.

Usage:
    python benchmark_weights.py --model-size 1.7  # Size in GB
"""

import argparse
import io
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader


def create_mock_state_dict(total_size_gb: float) -> Dict[str, torch.Tensor]:
    """Create a mock state dict that approximates a real model."""
    state_dict = {}
    
    # Approximate layer sizes (similar to Llama-style models)
    layer_configs = [
        ("model.embed_tokens.weight", 0.05),  # 5% of total
        ("model.layers.0.self_attn.q_proj.weight", 0.02),
        ("model.layers.0.self_attn.k_proj.weight", 0.02),
        ("model.layers.0.self_attn.v_proj.weight", 0.02),
        ("model.layers.0.self_attn.o_proj.weight", 0.02),
        ("model.layers.0.mlp.gate_proj.weight", 0.03),
        ("model.layers.0.mlp.up_proj.weight", 0.03),
        ("model.layers.0.mlp.down_proj.weight", 0.03),
    ]
    
    # Replicate layers
    num_layers = 32
    total_bytes = total_size_gb * 1024**3
    bytes_per_element = 4  # float32
    
    accumulated_size = 0
    
    # Add embedding
    name, fraction = layer_configs[0]
    num_elements = int((total_bytes * fraction) / bytes_per_element)
    state_dict[name] = torch.randn(num_elements, dtype=torch.float32)
    accumulated_size += num_elements * bytes_per_element
    
    # Add layers
    for layer_idx in range(num_layers):
        for name_template, fraction in layer_configs[1:]:
            name = name_template.replace("layers.0", f"layers.{layer_idx}")
            num_elements = int((total_bytes * fraction) / bytes_per_element)
            # Create 2D tensors for realism
            rows = int(num_elements ** 0.5)
            cols = num_elements // rows
            state_dict[name] = torch.randn(rows, cols, dtype=torch.float32)
            accumulated_size += rows * cols * bytes_per_element
    
    # Add final layer norm and output
    remaining = total_bytes - accumulated_size
    if remaining > 0:
        num_elements = int(remaining / bytes_per_element)
        state_dict["lm_head.weight"] = torch.randn(num_elements, dtype=torch.float32)
    
    actual_size_gb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024**3
    print(f"Created state dict with {len(state_dict)} tensors, total size: {actual_size_gb:.2f} GB")
    
    return state_dict


def benchmark_torch_save(state_dict: Dict[str, torch.Tensor], save_dir: Path) -> tuple[float, float]:
    """Benchmark torch.save() for the entire state dict."""
    save_path = save_dir / "model.pt"
    
    # Save
    start = time.perf_counter()
    torch.save(state_dict, save_path)
    save_time = time.perf_counter() - start
    
    # Load
    start = time.perf_counter()
    loaded = torch.load(save_path, weights_only=True)
    load_time = time.perf_counter() - start
    
    return save_time, load_time


def benchmark_safetensors(state_dict: Dict[str, torch.Tensor], save_dir: Path) -> tuple[float, float]:
    """Benchmark safetensors save/load."""
    try:
        from safetensors.torch import save_file, load_file
    except ImportError:
        print("⚠️  safetensors not installed. Install with: pip install safetensors")
        return None, None
    
    save_path = save_dir / "model.safetensors"
    
    # Save
    start = time.perf_counter()
    save_file(state_dict, save_path)
    save_time = time.perf_counter() - start
    
    # Load
    start = time.perf_counter()
    loaded = load_file(save_path)
    load_time = time.perf_counter() - start
    
    return save_time, load_time


def benchmark_dcp(state_dict: Dict[str, torch.Tensor], save_dir: Path, 
                  thread_count: int = 8, single_file: bool = False) -> tuple[float, float]:
    """Benchmark DCP (Distributed Checkpoint) save/load."""
    dcp_dir = save_dir / "dcp_checkpoint"
    
    # Save
    start = time.perf_counter()
    writer = FileSystemWriter(
        str(dcp_dir),
        single_file_per_rank=single_file,
        thread_count=thread_count,
    )
    dcp.save(state_dict=state_dict, storage_writer=writer)
    save_time = time.perf_counter() - start
    
    # Load
    start = time.perf_counter()
    reader = FileSystemReader(str(dcp_dir))
    loaded_state_dict = {k: torch.empty_like(v) for k, v in state_dict.items()}
    dcp.load(state_dict=loaded_state_dict, storage_reader=reader)
    load_time = time.perf_counter() - start
    
    return save_time, load_time


def benchmark_individual_files(state_dict: Dict[str, torch.Tensor], save_dir: Path) -> tuple[float, float]:
    """Benchmark saving each tensor as a separate file (similar to non-DCP torchstore)."""
    param_dir = save_dir / "params"
    param_dir.mkdir(exist_ok=True)
    
    # Save
    start = time.perf_counter()
    for name, tensor in state_dict.items():
        safe_name = name.replace("/", "_").replace(".", "_")
        torch.save(tensor, param_dir / f"{safe_name}.pt")
    save_time = time.perf_counter() - start
    
    # Load
    start = time.perf_counter()
    loaded = {}
    for name in state_dict.keys():
        safe_name = name.replace("/", "_").replace(".", "_")
        loaded[name] = torch.load(param_dir / f"{safe_name}.pt", weights_only=True)
    load_time = time.perf_counter() - start
    
    return save_time, load_time


def benchmark_in_memory_serialization(state_dict: Dict[str, torch.Tensor]) -> tuple[float, float]:
    """Benchmark in-memory serialization (simulates network transfer)."""
    
    # Serialize
    start = time.perf_counter()
    buffers = {}
    for name, tensor in state_dict.items():
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffers[name] = buffer.getvalue()
    serialize_time = time.perf_counter() - start
    
    # Deserialize
    start = time.perf_counter()
    loaded = {}
    for name, data in buffers.items():
        buffer = io.BytesIO(data)
        loaded[name] = torch.load(buffer, weights_only=True)
    deserialize_time = time.perf_counter() - start
    
    return serialize_time, deserialize_time


def print_results(name: str, save_time: float, load_time: float, size_gb: float):
    """Print benchmark results."""
    if save_time is None or load_time is None:
        print(f"\n❌ {name}: SKIPPED")
        return
    
    save_speed = size_gb / save_time if save_time > 0 else 0
    load_speed = size_gb / load_time if load_time > 0 else 0
    
    print(f"\n✅ {name}:")
    print(f"   Save: {save_time:.3f}s ({save_speed:.2f} GB/s)")
    print(f"   Load: {load_time:.3f}s ({load_speed:.2f} GB/s)")
    print(f"   Total: {save_time + load_time:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark weight save/load strategies")
    parser.add_argument("--model-size", type=float, default=1.7, 
                       help="Model size in GB (default: 1.7)")
    parser.add_argument("--use-tmpfs", action="store_true",
                       help="Use /tmp (often tmpfs/RAM disk) for testing")
    parser.add_argument("--test-dir", type=str, default=None,
                       help="Custom test directory (default: creates temp dir)")
    args = parser.parse_args()
    
    print(f"🔬 Benchmarking save/load strategies for {args.model_size} GB model\n")
    
    # Create state dict
    print("Creating mock state dict...")
    state_dict = create_mock_state_dict(args.model_size)
    actual_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024**3
    
    # Setup test directory
    if args.test_dir:
        test_dir = Path(args.test_dir)
        test_dir.mkdir(exist_ok=True, parents=True)
        cleanup = False
    elif args.use_tmpfs:
        test_dir = Path("/tmp/weight_benchmark")
        test_dir.mkdir(exist_ok=True)
        cleanup = True
    else:
        test_dir = Path(tempfile.mkdtemp(prefix="weight_benchmark_"))
        cleanup = True
    
    print(f"Test directory: {test_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Run benchmarks
        results = []
        
        # 1. torch.save (baseline)
        print("Running: torch.save (single file)...")
        save_time, load_time = benchmark_torch_save(state_dict, test_dir)
        print_results("torch.save (single file)", save_time, load_time, actual_size)
        results.append(("torch.save", save_time, load_time))
        shutil.rmtree(test_dir / "model.pt", ignore_errors=True)
        
        # 2. safetensors
        print("\nRunning: safetensors...")
        save_time, load_time = benchmark_safetensors(state_dict, test_dir)
        print_results("safetensors", save_time, load_time, actual_size)
        if save_time:
            results.append(("safetensors", save_time, load_time))
        shutil.rmtree(test_dir / "model.safetensors", ignore_errors=True)
        
        # 3. DCP with different configs
        print("\nRunning: DCP (thread_count=8, multi-file)...")
        save_time, load_time = benchmark_dcp(state_dict, test_dir, thread_count=8, single_file=False)
        print_results("DCP (8 threads, multi-file)", save_time, load_time, actual_size)
        results.append(("DCP-8-multi", save_time, load_time))
        shutil.rmtree(test_dir / "dcp_checkpoint", ignore_errors=True)
        
        print("\nRunning: DCP (thread_count=16, multi-file)...")
        save_time, load_time = benchmark_dcp(state_dict, test_dir, thread_count=16, single_file=False)
        print_results("DCP (16 threads, multi-file)", save_time, load_time, actual_size)
        results.append(("DCP-16-multi", save_time, load_time))
        shutil.rmtree(test_dir / "dcp_checkpoint", ignore_errors=True)
        
        print("\nRunning: DCP (thread_count=8, single-file)...")
        save_time, load_time = benchmark_dcp(state_dict, test_dir, thread_count=8, single_file=True)
        print_results("DCP (8 threads, single-file)", save_time, load_time, actual_size)
        results.append(("DCP-8-single", save_time, load_time))
        shutil.rmtree(test_dir / "dcp_checkpoint", ignore_errors=True)
        
        # 4. Individual files (simulates non-DCP torchstore)
        print("\nRunning: Individual parameter files...")
        save_time, load_time = benchmark_individual_files(state_dict, test_dir)
        print_results("Individual files (per-parameter)", save_time, load_time, actual_size)
        results.append(("Individual files", save_time, load_time))
        shutil.rmtree(test_dir / "params", ignore_errors=True)
        
        # 5. In-memory (simulates network transfer)
        print("\nRunning: In-memory serialization...")
        save_time, load_time = benchmark_in_memory_serialization(state_dict)
        print_results("In-memory (no I/O)", save_time, load_time, actual_size)
        results.append(("In-memory", save_time, load_time))
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY (sorted by total time)")
        print(f"{'='*60}")
        results.sort(key=lambda x: x[1] + x[2])
        
        for i, (name, save_time, load_time) in enumerate(results, 1):
            total = save_time + load_time
            print(f"{i}. {name:30s} {total:6.3f}s (save: {save_time:.3f}s, load: {load_time:.3f}s)")
        
    finally:
        if cleanup:
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    main()