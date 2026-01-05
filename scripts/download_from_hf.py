"""
Download a HuggingFace model snapshot and ensure a safetensors index exists.

Handles both single-file models (model.safetensors) and sharded models
(model-00001-of-00017.safetensors, etc.). Single-file models are renamed to
model-00001-of-00001.safetensors to match torchtitan expectations.

Usage:
  python scripts/download_from_hf.py Qwen/Qwen3-0.6B
  python scripts/download_from_hf.py Qwen/Qwen3-32B --revision main
  python scripts/download_from_hf.py Qwen/Qwen3-0.6B --local-dir /path/to/assets
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = PROJECT_DIR / "assets" / "hf"
SINGLE_SHARD_NAME = "model-00001-of-00001.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF model snapshot and create model.safetensors.index.json"
    )
    parser.add_argument("repo_id", type=str, help="HF repo id, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Destination directory (defaults to assets/hf/<repo-name>)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (branch, tag, or commit)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF access token (optional, for gated/private repos)",
    )
    return parser.parse_args()


def download_snapshot(repo_id: str, local_dir: Path, revision: str | None, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
    )


def list_safetensors_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("model*.safetensors"), key=lambda p: p.name)


def normalize_single_checkpoint(model_dir: Path, safetensors_files: list[Path]) -> list[Path]:
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensor files found in {model_dir}")

    if len(safetensors_files) == 1 and safetensors_files[0].name == "model.safetensors":
        src = safetensors_files[0]
        dst = model_dir / SINGLE_SHARD_NAME
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite existing {dst}")
        print(f"Renaming {src.name} -> {dst.name}...")
        shutil.move(str(src), str(dst))
        return [dst]

    if any(p.name == "model.safetensors" for p in safetensors_files):
        raise ValueError(
            "Found model.safetensors alongside sharded files. "
            "Expected either a single model.safetensors or sharded model-00001-of-*.safetensors."
        )

    return safetensors_files


def create_index_json(model_dir: Path, safetensors_files: list[Path]) -> Path:
    weight_map: dict[str, str] = {}
    total_size = 0

    for filepath in safetensors_files:
        filename = filepath.name
        print(f"  Indexing {filename}...")
        with safe_open(str(filepath), framework="pt") as f:
            for key in f.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate tensor key found: {key}")
                weight_map[key] = filename
        total_size += filepath.stat().st_size

    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }

    output_path = model_dir / "model.safetensors.index.json"
    with output_path.open("w") as f:
        json.dump(index_data, f, indent=2)

    return output_path


def main() -> None:
    args = parse_args()

    if args.local_dir:
        model_dir = Path(args.local_dir)
    else:
        repo_name = args.repo_id.rstrip("/").split("/")[-1]
        model_dir = DEFAULT_ASSETS_DIR / repo_name

    print(f"Downloading {args.repo_id} -> {model_dir}...")
    download_snapshot(args.repo_id, model_dir, args.revision, args.token)

    safetensors_files = list_safetensors_files(model_dir)
    safetensors_files = normalize_single_checkpoint(model_dir, safetensors_files)
    safetensors_files = sorted(safetensors_files, key=lambda p: p.name)

    print("Creating model.safetensors.index.json...")
    index_path = create_index_json(model_dir, safetensors_files)
    print(f"✅ Done. Index written to {index_path}")


if __name__ == "__main__":
    main()
