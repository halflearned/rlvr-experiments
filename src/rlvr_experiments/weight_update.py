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
        # * Don't gather on all ranks? [low priority]
        # * Parallelize the pushing across gpus, not just rank 0
        # * When parallelizing, bin by size not just name
        dist.barrier()  # ensure all ranks sync here
        updated_params = []
        from time import time
        if dist.get_rank() == 0:
            start_time = time()
            print(f"[rank 0] Starting to push weights to vLLM server...")

        for name, value in state_dict.items():
            
            # TODO: skip non-trainable params
            # Right now, all params are non-trainable once we grab them.
            # if not value.requires_grad:
            #     print("Skipping non-trainable param:", name)
            #     continue  
            
            # gather DTensor
            full_tensor = gather_full_tensor(value)

            # push to all clients
            if dist.get_rank() == 0:
                full_tensor = full_tensor.to(0, non_blocking=True)
                for vllm_client in self.vllm_clients:
                    vllm_client.update_named_param(name, full_tensor)   
                updated_params.append(name)


        if dist.get_rank() == 0:
            end_time = time()
            print(f"[rank 0] Finished pushing {len(updated_params)} weights to vLLM server in {end_time - start_time:.6f} seconds.")

        dist.barrier()  # ensure all ranks sync here

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

