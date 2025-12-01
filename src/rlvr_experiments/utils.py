
def unknown_tokens_to_overrides(tokens):
    """
    Example:
    >>> inputs = ["--hello", "world", "--param", "1", "--learning_rate", "1e-5", "--dictionary", "{'a': 1}", "--yes-flag"]
    >>> overrides = unknown_tokens_to_overrides(inputs)
    >>> overrides
    {'hello': 'world', 'param': 1, 'learning_rate': 1e-05, 'dictionary': {'a': 1}, 'yes_flag': None}
    """
    from omegaconf import OmegaConf
    dot = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t.lstrip("-").replace("-", "_")
            if "=" in key:               # --k=v   (after lstrip)
                dot.append(key)
            else:
                if i + 1 < len(tokens) and not tokens[i+1].startswith("--"):
                    # --k v
                    value = tokens[i+1]
                    dot.append(f"{key}={value}")
                    i += 1
                else:
                    # --flag  -> None  (this is a sagemaker quirk. we're assuming you can't pass a flag with no value)
                    dot.append(f"{key}=null")
        i += 1
    return OmegaConf.to_container(OmegaConf.from_dotlist(dot), resolve=True)



import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Expected a value parseable as boolean. Got {v!r}')
    


def set_seed(seed: int = 42):
    import os
    import random
    import numpy as np
    import torch
    # 1. Base-level determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 2. PyTorch backend control
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 3. Environment-level determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 4. Optional: make HuggingFace Transformers respect this
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


import hashlib

def stable_hash(obj) -> int:
    """
    Deterministic 64-bit hash for any JSON-serializable object (e.g. (video_stem, indices)).
    Always identical across processes/runs.
    """
    # convert to string first to ensure consistent serialization
    s = str(obj).encode("utf-8")
    # compute SHA1 and take lower 64 bits
    return int(hashlib.sha1(s).hexdigest(), 16) % (1 << 63)


import torch.distributed as dist
def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank()==0


def print0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)
