"""Download the pinned official FISHER-small weights and inference code."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/fisher-small"))
    parser.add_argument("--config", type=Path, default=Path("fisher_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshot_download(
        config["model_repo"], revision=config["revision"], local_dir=args.out,
        allow_patterns=["*.py", "config.json", "model.safetensors", "README.md"],
        max_workers=3,
    )
    weights_hash = hashlib.sha256((args.out / "model.safetensors").read_bytes()).hexdigest()
    if weights_hash != config["weights_sha256"]:
        raise ValueError("FISHER weight hash does not match the pinned official checkpoint")
    print(f"Downloaded {config['model_repo']} at {config['revision']}; SHA-256 {weights_hash}", flush=True)


if __name__ == "__main__":
    main()
