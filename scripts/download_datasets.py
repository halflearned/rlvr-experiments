from huggingface_hub import snapshot_download
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

for flavor in ["Qwen3-0.6B"]:
    repo_id = f"Qwen/{flavor}"
    local_dir = PROJECT_DIR / f"assets/hf/{flavor}"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Dataset '{repo_id}' downloaded to '{local_dir}'")