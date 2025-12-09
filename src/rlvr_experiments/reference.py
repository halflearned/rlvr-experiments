from typing import Any, Dict

import os
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.checkpoint.state_dict import get_model_state_dict

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class TitanReferenceModel(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")
        self.job_config = job_config

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
                "TitanRLTrainer does not support Pipeline Parallelism; "
                "set parallelism.pipeline_parallel_degree = 0."
            )
        if self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "TitanRLTrainer does not support Context Parallelism; "
                "set parallelism.context_parallel_degree = 0."
            )

        # 5. TrainSpec / model args / tokenizer
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
        self.model_args = self.train_spec.model_args[job_config.model.flavor]
        self.model_args.update_from_config(job_config)

        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        logger.info(f"Building {job_config.model.name}...")
        with torch.device("meta"), utils.set_default_dtype(
            TORCH_DTYPE_MAP[job_config.training.dtype]
        ):
            model = self.train_spec.model_cls(self.model_args)

        # 6. Apply parallelism (FSDP / TP, no PP/CP)
        model_converters = build_model_converters(job_config, self.parallel_dims)
        model_converters.convert(model)

        model = self.train_spec.parallelize_fn(model, self.parallel_dims, job_config)

        # 7. Initialize weights and move to init device
        init_device = "cpu" if job_config.training.enable_cpu_offload else device_type
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
        model.train()

        # Wrap in list for compatibility with Titan utilities
        self.model_parts = [model]
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # 9. Checkpointer / state
        self.step = 0
        if self.train_spec.state_dict_adapter is None:
            raise ValueError(
                "TitanRLTrainer requires a StateDictAdapter for HF state dict support."
            )
        self.sd_adapter = self.train_spec.state_dict_adapter(self.model_args, job_config.model.hf_assets_path)

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

        # 10. Train context / AMP (CP is disabled, so cp_ctx is always None)
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


    # training api
    def forward_step(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = input_dict["input"].to(self.device)

        extra_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in input_dict.items()
            if k != "input"
        }
        extra_kwargs: Dict[str, Any] = {}

        # FlexAttention: build masks if requested by model args
        if getattr(self.model_args, "use_flex_attn", False):
            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )

        with self.train_context_mgr(None):
            with self.maybe_enable_amp, torch.no_grad():
                logits = self.model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        return logits


    # checkpointing and state management

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

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

    def load_checkpoint(self, step: int | None = None):
        """
        Optional convenience wrapper around CheckpointManager.load.
        """
        self.checkpointer.load(step=step)

    def save_checkpoint(self, step: int | None = None, last_step: bool = False):
        """
        Optional convenience wrapper around CheckpointManager.save.
        """
        curr_step = self.step if step is None else step
        self.checkpointer.save(curr_step, last_step=last_step)

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()

