import torch
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor


class WeightUpdate:

    def push_weights(self, model):
        # If we're working with an nccl client this pushes weights immediately
        # Later on, if we're working with some fast kind of storage,
        # this will put the weights there
        pass



class VLLMSyncWeightUpdate(WeightUpdate):

    def __init__(self, vllm_clients):
        print("NEW VLLMSyncWeightUpdate created")
        self.vllm_clients = vllm_clients
        if dist.get_rank() == 0:
            for vllm_client in self.vllm_clients:
                vllm_client.init_communicator(device=0)

    def push_weights(self, state_dict):
        # TODO:
        # * REQUIRED: Needs to stop vllm engine until all weights are pushed, likely also clear kv cache
        # * Push batches of parameters together if they're small
        # * Don't gather on all ranks?
        # * Parallelize the pushing across gpus, not just rank 0
        # * When parallelizing, bin by size not just name
        for name, value in state_dict.items():
            # skip non-trainable params
            if not value.requires_grad:
                continue  
            
            # gather DTensor
            full_tensor = gather_full_tensor(value)

            # push to all clients
            if dist.get_rank() == 0:
                full_tensor = full_tensor.to(0, non_blocking=True)
                for vllm_client in self.vllm_clients:
                    vllm_client.update_named_param(name, full_tensor)

    def close(self):
        if dist.get_rank() == 0:
            for vllm_client in self.vllm_clients:
                vllm_client.close_communicator()
    
    def __del__(self):
        # fallback
        try:
            self.close()
        except:
            pass



def gather_full_tensor(value) -> torch.Tensor | None:
    if isinstance(value, DTensor):
        full = value.full_tensor()
    else:
        full = value

    if dist.get_rank() == 0:
        return full.detach().clone()
    else:
        return None


# TODO: use this later
import hashlib
def get_responsible_rank(param_name):
    # Deterministic hashing: Every node agrees on who owns 'layers.0.weight'
    hash_val = int(hashlib.sha256(param_name.encode('utf-8')).hexdigest(), 16)
    return hash_val % dist.get_world_size()