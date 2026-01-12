#!/usr/bin/env python3
"""Test model loading and forward pass."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')

import torch
from torchtitan.config import ConfigManager
import tempfile
import tomli_w

print('Loading config...')
cfg = {
    'model': {'name': 'qwen3', 'flavor': '1.7B', 'hf_assets_path': '/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base'},
    'training': {'seq_len': 512, 'dtype': 'bfloat16', 'mixed_precision_param': 'bfloat16', 'mixed_precision_reduce': 'float32'},
    'parallelism': {'data_parallel_replicate_degree': 1, 'data_parallel_shard_degree': 1, 'tensor_parallel_degree': 1, 'context_parallel_degree': 1},
    'checkpoint': {'enable': False},
    'activation_checkpoint': {'mode': 'none'},
    'compile': {'enable': False},
    'optimizer': {'name': 'AdamW', 'lr': 1e-5},
    'job': {'dump_folder': '/tmp/test'},
}
with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
    tomli_w.dump(cfg, f)
    config_path = f.name

print(f'Config at: {config_path}')
job_config = ConfigManager().parse_args(['--job.config-file', config_path])
print('Config parsed successfully')

from rlvr_experiments.model import TitanModel
print('Creating model...')
model = TitanModel(job_config, trainable=False)
print(f'Model created on device: {model.device}')

# Test forward with small batch
x = torch.randint(0, 1000, (1, 64), device=model.device)
print(f'Running forward with input shape {x.shape}...')
with torch.no_grad():
    out = model.forward(x)
print(f'Output shape: {out.shape}')

# Test forward with larger batch like actual training (64 completions)
x = torch.randint(0, 1000, (64, 512), device=model.device)
print(f'Running forward with larger input shape {x.shape}...')
with torch.no_grad():
    out = model.forward(x)
print(f'Output shape: {out.shape}')

print('SUCCESS')
