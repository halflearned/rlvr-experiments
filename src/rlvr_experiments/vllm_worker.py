from typing import Tuple, List, Any
import torch

from vllm.v1.worker.gpu_worker import Worker

from .weight_sync import WeightSyncManager
from .syncing import ChunkMeta



class WeightSyncVLLMWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_sync = WeightSyncManager()

    def init_weight_sync(self, host: str, port: int, world_size: int, rank: int):
        my_sync_rank = rank + self.rank
        print(f"[Worker {self.rank}] Joining Weight Sync Group as Rank {my_sync_rank}...")
        self.weight_sync.init_communicator(
            host=host,
            port=port,
            world_size=world_size,
            my_rank=my_sync_rank,
            device=self.device,
        )
        return True

    # ---------------------------------------------- #
    # Existing per-param update_weight(...) omitted
    # ---------------------------------------------- #

    def recv_chunk_from_hf(
        self,
        chunk: Any,          # may be ChunkMeta or a dict
        dtype_str: str,
        src_rank: int,
    ):
        """
        vLLM-side chunk receiver:
        - participates in NCCL broadcast over weight_sync communicator.
        - slices flat buffer and loads HF weights into vLLM model.

        `chunk` may arrive as a dict due to msgspec serialization, so we
        normalize both ChunkMeta and dict-shaped inputs.
        """
        if not self.weight_sync.is_initialized:
            raise RuntimeError(f"Worker {self.rank} sync not initialized.")

        # ---- Normalize chunk structure (ChunkMeta or dict) ----
        if hasattr(chunk, "total_numel") and hasattr(chunk, "params"):
            total_numel = int(chunk.total_numel)
            params_iter = chunk.params
        else:
            # msgspec / dict case
            total_numel = int(chunk["total_numel"])
            params_iter = chunk["params"]

        # Now each entry in params_iter might be ParamMeta or dict.
        def iter_params():
            for p in params_iter:
                if hasattr(p, "name"):
                    # ParamMeta
                    name = p.name
                    numel = int(p.numel)
                    shape = tuple(p.shape)
                else:
                    # dict
                    name = p["name"]
                    numel = int(p["numel"])
                    shape = tuple(p["shape"])
                yield name, numel, shape

        dtype = getattr(torch, dtype_str)
        flat = torch.empty(total_numel, dtype=dtype, device=self.device)

        # NCCL broadcast over the weight-sync communicator
        self.weight_sync.communicator.broadcast(flat, src=src_rank)
        torch.cuda.current_stream().synchronize()

        offset = 0
        for name, numel, shape in iter_params():
            flat_slice = flat[offset : offset + numel]
            weight = flat_slice.view(shape)
            offset += numel

            # Load into vLLM model by HF name
            self.model_runner.model.load_weights(weights=[(name, weight)])

        del flat
        return True
