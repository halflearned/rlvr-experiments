import torch
from torch.nn.utils.rnn import pad_sequence


class VLLMOutput:
    
    def __init__(self, request_output):
        self.request_output = request_output

    def all_tokens(self):
        prompt_token_ids = self.request_output.prompt_token_ids
        return [
            prompt_token_ids + out.token_ids for out in self.request_output.outputs
        ]

    def completion_token_ids(self):
        return [out.token_ids for out in self.request_output.outputs]
    
    def prompt_length(self):
        return len(self.request_output.prompt_token_ids)
    
    def num_completions(self):
        return len(self.request_output.outputs)
    
    def completion_logprobs(self):
        all_logprobs = []
        for completion_output in self.request_output.outputs:
            row_logprobs = []
            for i, token_id in enumerate(completion_output.token_ids):
                lp = completion_output.logprobs[i][token_id].logprob
                row_logprobs.append(lp)
            all_logprobs.append(row_logprobs)
        return all_logprobs
    
    def completion_texts(self):
        return [out.text for out in self.request_output.outputs]
    
    def completion_lengths(self):
        return [len(out.token_ids) for out in self.request_output.outputs]
    
    def completion_mask(self):
        masks = []
        for out in self.request_output.outputs:
            length = len(out.token_ids)
            mask = [1] * length 
            masks.append(mask)
        return masks

    # TODO: better name
    def get_tensors(self, tokenizer):
        full_batch = tokenizer.pad(
            {"input_ids": self.all_tokens()},
            padding=True,
            return_tensors="pt",
        )
        completion_batch = tokenizer.pad(
            {"input_ids": self.completion_token_ids()},
            padding=True,
            return_tensors="pt",
        )
        completion_logprobs = pad_sequence(
            [torch.tensor(lp) for lp in self.completion_logprobs()],
            batch_first=True,
            padding_value=0.0,
        )
        return (
            full_batch["input_ids"],
            completion_batch["input_ids"],
            completion_batch["attention_mask"],
            completion_logprobs,
        )
    
