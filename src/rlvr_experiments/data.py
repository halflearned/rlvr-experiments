import torch

class ChatPromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, tokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.prompts[idx]}],
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            template,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0]