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
        # * Push batches of parameters together if they're small
        # * When parallelizing, bin by size not just name
        # * Don't gather on all ranks? [low priority]
        # * Parallelize the pushing across gpus, not just rank 0

        dist.barrier()  # ensure all ranks sync here

        for name, value in state_dict.items():
            
            # gather DTensor on all gpus
            if isinstance(value, DTensor):
                full_tensor = value.full_tensor()
            else:
                full_tensor = value

            # first gpu pushes to all clients
            if dist.get_rank() == 0:
                full_tensor = full_tensor.detach().to(0, non_blocking=True)
                for vllm_client in self.vllm_clients:
                    vllm_client.update_named_param(name, full_tensor)   

        dist.barrier()  # ensure all ranks sync here again

    def close(self):
        if dist.get_rank() == 0:
            for vllm_client in self.vllm_clients:
                vllm_client.close_communicator()
    
    def __del__(self):  # fallback
        try:
            self.close()
        except:
            pass