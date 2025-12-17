import torch


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

        completion_token_ids = self.completion_token_ids()
        raw_completion_logprobs = self.completion_logprobs()
        max_completion_len = max((len(ids) for ids in completion_token_ids), default=0)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")

        batch_size = len(completion_token_ids)
        completion_ids = torch.full(
            (batch_size, max_completion_len),
            pad_token_id,
            dtype=torch.long,
        )
        completion_mask = torch.zeros((batch_size, max_completion_len), dtype=torch.long)
        completion_logprobs = torch.zeros((batch_size, max_completion_len), dtype=torch.float32)

        for row, (token_ids, logprobs) in enumerate(zip(completion_token_ids, raw_completion_logprobs)):
            length = len(token_ids)
            if length == 0:
                continue
            completion_ids[row, -length:] = torch.tensor(token_ids, dtype=torch.long)
            completion_mask[row, -length:] = 1
            completion_logprobs[row, -length:] = torch.tensor(logprobs, dtype=torch.float32)

        return (
            full_batch["input_ids"],
            completion_ids,
            completion_mask,
            completion_logprobs,
        )
    
