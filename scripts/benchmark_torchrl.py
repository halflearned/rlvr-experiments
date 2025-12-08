import os
import time
import torch

from tensordict import TensorDict
from torchrl.weight_update.llm import VLLMDoubleBufferTransport


# Directory for double-buffer storage.
# Use a fast local NVMe path or your shared FS mount here.
REMOTE_ADDR = "/tmp/vllm_double_buffer_bench"
os.makedirs(REMOTE_ADDR, exist_ok=True)

# Transport used by TorchRL's double-buffer scheme underneath.
transport = VLLMDoubleBufferTransport(remote_addr=REMOTE_ADDR)


# ----- Dummy model to approximate your weight size -----
class TinyGPT(torch.nn.Module):
    def __init__(self, hidden=4096, vocab=32000, layers=24):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden, bias=False) for _ in range(layers)]
        )
        self.ln_f = torch.nn.LayerNorm(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for l in self.layers:
            x = l(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# Construct model. You can do this on GPU if you want,
# we will explicitly move to CPU for the "already gathered" state.
model = TinyGPT(hidden=4096, vocab=32000, layers=24)
model.eval()

# ----- Pretend we already did the global gather -----
# Extract full CPU state dict.
with torch.no_grad():
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# Wrap in TensorDict, which is what the transport expects.
weights_td = TensorDict(state, batch_size=[])

# Compute total size in GB for reference.
total_bytes = sum(t.numel() * t.element_size() for t in weights_td.values())
total_gb = total_bytes / (1024 ** 3)
print(f"Total parameter size: {total_gb:.3f} GiB")

# Optional warmup (touch filesystem once).
transport.send_weights("warmup", weights_td)
_ = transport.receive_weights()

# ----- Actual timed run -----
# Send (write) side: this is the trainer writing to the double buffer.
t0 = time.time()
transport.send_weights("benchmark", weights_td)
t1 = time.time()

# Receive (read) side: this is the vLLM worker reading from the double buffer.
t2 = time.time()
received_td = transport.receive_weights()
t3 = time.time()

send_time = t1 - t0
recv_time = t3 - t2

print(f"Send (memmap write) time : {send_time:.3f} s")
print(f"Recv (memmap read) time  : {recv_time:.3f} s")
print(f"Throughput write         : {total_gb / send_time:.3f} GiB/s")
print(f"Throughput read          : {total_gb / recv_time:.3f} GiB/s")

# Sanity check on one or two random keys (full allclose on all keys can be slow).
for i, (k, v) in enumerate(weights_td.items()):
    assert torch.allclose(v, received_td[k])
    if i >= 2:
        break
print("Sanity check on a few params: OK")
