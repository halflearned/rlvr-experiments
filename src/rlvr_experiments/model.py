from typing import Any, Dict

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.checkpoint.state_dict import get_model_state_dict

import torchtitan.protocols.train_spec as train_spec_module

# Monkey-patch torchtitan's reshape_for_broadcast to remove assert that breaks torch.compile
# with dynamic sequence lengths. The assert is redundant after the slice operation.
def _patched_reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # Removed: assert rope_cache.shape == (seqlen, head_dim * 2)
    # The slice guarantees the shape, and the assert breaks torch.compile with dynamic shapes
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)

try:
    import torchtitan.models.qwen3.model.model as qwen3_model
    qwen3_model.reshape_for_broadcast = _patched_reshape_for_broadcast
except ImportError:
    pass  # qwen3 model not available
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTManager
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class TitanModel(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, job_config: JobConfig, trainable=True):
        torch._C._log_api_usage_once("torchtitan.train")
        self.job_config = job_config
        self.trainable = trainable

        # 1. Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # 2. Distributed init
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )

        # 3. Parallel dimensions (TP/DP only; PP/CP disabled)
        world_size = int(os.environ["WORLD_SIZE"])
        self.parallel_dims = ParallelDims(
            dp_shard=job_config.parallelism.data_parallel_shard_degree,
            dp_replicate=job_config.parallelism.data_parallel_replicate_degree,
            cp=job_config.parallelism.context_parallel_degree,
            tp=job_config.parallelism.tensor_parallel_degree,
            pp=job_config.parallelism.pipeline_parallel_degree,
            ep=job_config.parallelism.expert_parallel_degree,
            etp=job_config.parallelism.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        # Enforce no PP / no CP for RL trainer
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "TitanModel does not support Pipeline Parallelism; "
                "set parallelism.pipeline_parallel_degree = 0."
            )
        if self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "TitanModel does not support Context Parallelism; "
                "set parallelism.context_parallel_degree = 0."
            )

        # 4. TrainSpec / model args / tokenizer / state dict adapter (BEFORE model creation)
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
        self.model_args = self.train_spec.model_args[job_config.model.flavor]
        self.model_args.update_from_config(job_config)

        # override dataloader to use our own
        # self.train_spec.build_dataloader_fn = rlvr_loader

        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )
        
        if self.train_spec.state_dict_adapter is None:
            raise ValueError(
                "TitanModel requires a StateDictAdapter for HF state dict support."
            )
        self.sd_adapter = self.train_spec.state_dict_adapter(self.model_args, job_config.model.hf_assets_path)

        # 5. Build model on meta device
        logger.info(f"Building {job_config.model.name}...")
        with torch.device("meta"), utils.set_default_dtype(
            TORCH_DTYPE_MAP[job_config.training.dtype]
        ):
            model = self.train_spec.model_cls(self.model_args)

        # 6. Apply parallelism (FSDP / TP, no PP/CP)
        model_converters = build_model_converters(job_config, self.parallel_dims)
        model_converters.convert(model)

        model = self.train_spec.parallelize_fn(model, self.parallel_dims, job_config)

        # 7. Initialize weights - use checkpointer to load HF if available
        init_device = "cpu" if job_config.training.enable_cpu_offload else device_type
        model.to_empty(device=init_device)
        
        # Initialize with random weights first (required for DCP loading)
        logger.info("Initializing model with random weights...")
        with torch.no_grad():
            model.init_weights(buffer_device=None)
        
       

        # Wrap in list for compatibility with Titan utilities
        self.model_parts = [model]

        # 8. Train context / AMP (CP is disabled for now)
        loss_parallel_enabled = (
            self.parallel_dims.tp_enabled
            and not job_config.parallelism.disable_loss_parallel
        )
        self.train_context_mgr = dist_utils.get_train_context(
            loss_parallel_enabled,
            job_config.parallelism.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            self.parallel_dims, job_config.training.mixed_precision_param, device_type
        )

        # 9. Step counter
        self.step = 0

        if self.trainable:
            model.train()

            # 10. Fault tolerance manager (trainable only)
            self.ft_manager = FTManager(job_config.fault_tolerance)
            self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

            # 11. Optimizers and LR schedulers (trainable only)
            self.optimizers = self.train_spec.build_optimizers_fn(
                self.model_parts, job_config.optimizer, self.parallel_dims, self.ft_manager
            )
            self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
                self.optimizers, job_config.lr_scheduler, job_config.training.steps
            )

            # Post-optimizer hook (e.g., float8 amax/scale updates)
            self.optimizers.register_step_post_hook(
                lambda *args, **kwargs: model_converters.post_optimizer_hook(
                    self.model_parts
                )
            )

            # 12. Checkpointer (trainable only)
            self.checkpointer = CheckpointManager(
                dataloader=None,
                model_parts=self.model_parts,
                optimizers=self.optimizers,
                lr_schedulers=self.lr_schedulers,
                states={"train_state": self},
                checkpoint_config=job_config.checkpoint,
                sd_adapter=self.sd_adapter,
                base_folder=job_config.job.dump_folder,
                ft_manager=self.ft_manager,
            )
            self.checkpointer.load(step=-1)  # -1 = auto-find latest or skip if none
        else:
            # Reference / non-trainable models should be deterministic and not build graphs.
            # Keep them in eval mode and disable gradients to save memory.
            self.model_parts[0].eval()
            self.model_parts[0].requires_grad_(False)

            # Non-trainable models don't need optimizers, schedulers, or full checkpointing
            self.ft_manager = None
            self.optimizers = None
            self.lr_schedulers = None
            self.checkpointer = CheckpointManager(
                dataloader=None,
                model_parts=self.model_parts,
                optimizers=None,
                lr_schedulers=None,
                states={"train_state": self},
                checkpoint_config=job_config.checkpoint,
                sd_adapter=self.sd_adapter,
                base_folder=job_config.job.dump_folder,
                ft_manager=None,
            )
            self.checkpointer.load(step=-1)  # -1 = auto-find latest or skip if none

        # Metrics tracking for MFU/TFLOPS (without using MetricsProcessor which calls empty_cache)
        _, self._num_flops_per_token = self.model_args.get_nparams_and_flops(
            self.model_parts[0], job_config.training.seq_len
        )
        self._gpu_peak_flops = utils.get_peak_flops(torch.cuda.get_device_name(self.device))
        self._metrics_log_freq = job_config.metrics.log_freq
        self._ntokens_since_last_log = 0
        self._time_last_log = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model, returning logits.

        Args:
            input_ids: [batch_size, seq_len] token ids tensor.

        Returns:
            logits: [batch_size, seq_len, vocab_size] tensor (may be DTensor with TP)
        """
        logger.debug(f"TitanModel.forward: input shape={input_ids.shape}")
        inputs = input_ids.to(self.device)

        # Pad sequence length to be divisible by TP degree to avoid losing tokens
        # during sequence sharding with Shard(dim=1)
        tp_degree = self.parallel_dims.tp
        original_seq_len = inputs.size(1)
        pad_len = 0
        if tp_degree > 1:
            remainder = original_seq_len % tp_degree
            if remainder != 0:
                pad_len = tp_degree - remainder
                inputs = torch.nn.functional.pad(inputs, (0, pad_len), value=0)

        extra_kwargs: Dict[str, Any] = {}

        # FlexAttention: build masks if requested by model args
        if getattr(self.model_args, "use_flex_attn", False):
            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
            )

        logger.debug("TitanModel.forward: calling model_parts[0]...")
        with self.train_context_mgr(None), self.maybe_enable_amp:
            logits = self.model_parts[0](inputs, **extra_kwargs)

            # If we padded the input, trim logits back to original sequence length
            # to maintain correct alignment with target tokens
            # Note: We slice DTensor directly to preserve sharding (don't call full_tensor())
            if pad_len > 0:
                logits = logits[:, :original_seq_len, :]

        logger.debug(f"TitanModel.forward: done, logits type={type(logits).__name__}")
        return logits

    def optim_step(self) -> float:
        import time
        import torch
        logger.debug("TitanModel.optim_step: starting...")
        if not self.trainable:
            raise RuntimeError("Model is non-trainable; cannot call optim_step().")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logger.debug("TitanModel.optim_step: clipping grads...")
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=None,  # PP disabled
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logger.debug(f"TitanModel.optim_step: grad_norm={grad_norm.item():.4f}")

        # Ensure async checkpoint staging (if any) is done
        self.checkpointer.maybe_wait_for_staging()
        t2 = time.perf_counter()

        logger.debug("TitanModel.optim_step: stepping optimizer...")
        self.optimizers.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        logger.debug("TitanModel.optim_step: stepping lr_scheduler...")
        self.lr_schedulers.step()
        t4 = time.perf_counter()
        logger.debug("TitanModel.optim_step: zeroing grads...")
        self.optimizers.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        self.step += 1

        print(f"[PROFILE optim_step] clip_grad={1000*(t1-t0):.0f}ms, wait_ckpt={1000*(t2-t1):.0f}ms, "
              f"optim.step={1000*(t3-t2):.0f}ms, lr.step={1000*(t4-t3):.0f}ms, zero_grad={1000*(t5-t4):.0f}ms, "
              f"TOTAL={1000*(t5-t0):.0f}ms", flush=True)

        logger.debug(f"TitanModel.optim_step: done, step={self.step}")
        return grad_norm.item()

    def log_metrics(self, loss: float, grad_norm: float, ntokens: int) -> dict | None:
        """Log metrics (MFU, TFLOPS, memory, etc.). Returns metrics dict if logged."""
        import time

        # Accumulate tokens for throughput calculation
        self._ntokens_since_last_log += ntokens

        # Check if we should log this step
        if self._metrics_log_freq <= 0 or self.step % self._metrics_log_freq != 0:
            return None

        now = time.perf_counter()
        if self._time_last_log is None:
            self._time_last_log = now
            return None  # Skip first log (no time delta yet)

        time_delta = now - self._time_last_log
        self._time_last_log = now

        # Tokens per second
        tps = self._ntokens_since_last_log / time_delta if time_delta > 0 else 0

        # TFLOPS and MFU
        tflops = self._num_flops_per_token * tps / 1e12
        mfu = 100 * self._num_flops_per_token * tps / self._gpu_peak_flops if self._gpu_peak_flops > 0 else 0

        # Memory stats
        mem_stats = torch.cuda.memory_stats(self.device)
        peak_reserved = mem_stats.get("reserved_bytes.all.peak", 0) / (1024**3)
        total_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        mem_pct = 100 * peak_reserved / total_mem if total_mem > 0 else 0

        print(
            f"[METRICS] step={self.step:3d}  loss={loss:.4f}  "
            f"memory={peak_reserved:.1f}GiB({mem_pct:.0f}%)  "
            f"tps={tps:.0f}  tflops={tflops:.1f}  mfu={mfu:.1f}%",
            flush=True
        )

        # Reset token counter after logging
        self._ntokens_since_last_log = 0

        # Return metrics for the main process to emit to tracer
        return {
            "tps": tps,
            "tflops": tflops,
            "mfu": mfu,
            "memory_gib": peak_reserved,
            "memory_pct": mem_pct,
        }

    # Checkpointing and state management

    def hf_state_dict(self) -> Dict[str, Any]:
        """
        Returns a HuggingFace-style state dict (values may still be DTensors).
        """
        titan_state = {
            k: v
            for sd in map(get_model_state_dict, self.model_parts)
            for k, v in sd.items()
        }
        if not self.sd_adapter:
            raise RuntimeError("No StateDictAdapter found for this model.")
        hf_sd = self.sd_adapter.to_hf(titan_state)
        return hf_sd

    def save_checkpoint(self, step: int | None = None, last_step: bool = False):
        curr_step = self.step if step is None else step
        self.checkpointer.save(curr_step, last_step=last_step)

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()

    def export_to_hf(self, output_path: str) -> None:
        """
        Export the model to HuggingFace format.

        Only rank 0 writes; all ranks must call this (collective for DTensors).
        Copies tokenizer/config from the original HF assets path.
        """
        from torch.distributed.tensor import DTensor
        import shutil

        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.debug(f"TitanModel.export_to_hf: starting, output_path={output_path}")

        logger.debug("TitanModel.export_to_hf: calling hf_state_dict()...")
        hf_sd = self.hf_state_dict()
        logger.debug(f"TitanModel.export_to_hf: hf_state_dict done, {len(hf_sd)} keys")

        # Materialize DTensors to full tensors (collective operation)
        logger.debug("TitanModel.export_to_hf: materializing DTensors...")
        materialized = {}
        for i, (k, v) in enumerate(hf_sd.items()):
            if isinstance(v, DTensor):
                materialized[k] = v.full_tensor().cpu()
            else:
                materialized[k] = v.cpu() if torch.is_tensor(v) else v
            if i % 50 == 0:
                logger.debug(f"TitanModel.export_to_hf: materialized {i}/{len(hf_sd)} tensors")
        logger.debug("TitanModel.export_to_hf: materialization done")

        # Only rank 0 writes
        if rank == 0:
            logger.debug(f"TitanModel.export_to_hf: saving to {output_path}...")
            os.makedirs(output_path, exist_ok=True)

            # Save model weights using safetensors
            from safetensors.torch import save_file
            logger.debug("TitanModel.export_to_hf: calling save_file...")
            save_file(materialized, os.path.join(output_path, "model.safetensors"))
            logger.debug("TitanModel.export_to_hf: save_file done")

            # Copy tokenizer and config from original HF assets
            hf_assets = self.job_config.model.hf_assets_path
            if hf_assets and os.path.isdir(hf_assets):
                logger.debug(f"TitanModel.export_to_hf: copying config files from {hf_assets}...")
                for fname in os.listdir(hf_assets):
                    if fname.endswith((".json", ".txt", ".model")):
                        src = os.path.join(hf_assets, fname)
                        dst = os.path.join(output_path, fname)
                        shutil.copy2(src, dst)

            logger.info(f"Exported HuggingFace model to {output_path}")

        logger.debug("TitanModel.export_to_hf: done")


