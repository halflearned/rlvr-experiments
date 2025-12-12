import json
from safetensors import safe_open

# Adjust path to where your model.safetensors actually is
model_path = "./assets/hf/Qwen3-0.6B"
safetensors_file = "model.safetensors"

# 1. Read all keys from the single file
with safe_open(f"{model_path}/{safetensors_file}", framework="pt") as f:
    keys = f.keys()

# 2. Create the weight map (All keys -> "model.safetensors")
weight_map = {key: safetensors_file for key in keys}

# 3. Create the index structure
index_data = {
    "metadata": {"total_size": 0},  # Size doesn't strictly matter for loading logic usually
    "weight_map": weight_map
}

# 4. Save model.safetensors.index.json
with open(f"{model_path}/model.safetensors.index.json", "w") as f:
    json.dump(index_data, f, indent=2)

print(f"Generated index file with {len(keys)} keys.")