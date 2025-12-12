import os
import shutil
import json
from safetensors import safe_open

# Verify paths
src_dir = "./assets/hf/Qwen3-0.6B"
src_file = "model.safetensors"
full_src_path = os.path.join(src_dir, src_file)

if not os.path.exists(full_src_path):
    print(f"Error: {full_src_path} does not exist.")
    exit(1)

# 1. Define new filename pattern
# This matches what torchtitan expects: model-00001-of-00001.safetensors
new_filename = "model-00001-of-00001.safetensors"
dst_path = os.path.join(src_dir, new_filename)

print(f"Renaming {src_file} -> {new_filename}...")
shutil.move(full_src_path, dst_path)

# 2. Generate the index file mapping to this new name
print("Generating model.safetensors.index.json...")
with safe_open(dst_path, framework="pt") as f:
    keys = f.keys()

index_data = {
    "metadata": {"total_size": 0},
    "weight_map": {k: new_filename for k in keys}
}

index_path = os.path.join(src_dir, "model.safetensors.index.json")
with open(index_path, "w") as f:
    json.dump(index_data, f, indent=2)

print("✅ Checkpoint sharded successfully.")
print(f"   - File: {new_filename}")
print(f"   - Index: model.safetensors.index.json")