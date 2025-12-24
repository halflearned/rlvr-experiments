"""
Create model.safetensors.index.json for HuggingFace model directories.

Handles both single-file models (model.safetensors) and sharded models
(model-00001-of-00017.safetensors, etc.)

Usage:
    python scripts/create_index_json.py ./assets/hf/Qwen3-32B
    python scripts/create_index_json.py ./assets/hf/Qwen3-0.6B
"""
import argparse
import glob
import json
import os

from safetensors import safe_open


def create_index_json(model_path: str) -> None:
    # Find all safetensor files
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))

    # Filter out the index file itself if it exists
    safetensor_files = [f for f in safetensor_files if not f.endswith(".index.json")]

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    print(f"Found {len(safetensor_files)} safetensor file(s)")

    weight_map = {}
    total_size = 0

    for filepath in safetensor_files:
        filename = os.path.basename(filepath)
        print(f"  Processing {filename}...")

        with safe_open(filepath, framework="pt") as f:
            keys = list(f.keys())
            for key in keys:
                weight_map[key] = filename

            # Calculate size from tensor metadata
            metadata = f.metadata()
            if metadata:
                # Try to get size from metadata if available
                pass

        # Get file size
        total_size += os.path.getsize(filepath)

    # Create the index structure
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    # Save model.safetensors.index.json
    output_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(output_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Generated {output_path} with {len(weight_map)} keys, total_size={total_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create model.safetensors.index.json")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    args = parser.parse_args()

    create_index_json(args.model_path)