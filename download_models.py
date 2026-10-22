"""Download exact model revisions; set HF_ENDPOINT to a reachable mirror if needed."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


BEATS_REV = "5b53b0404df452a3a607d7e67687227730e5bad1"
QWEN_REV = "0a095220c30b7b31434169c3086508ef3ea5bf0a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(hf_hub_download(
        "lpepino/beats_ckpts", "BEATs_iter3.pt", revision=BEATS_REV,
        local_dir=args.out / "beats",
    ), flush=True)
    print(snapshot_download(
        "Qwen/Qwen2-Audio-7B-Instruct", revision=QWEN_REV,
        local_dir=args.out / "qwen2-audio",
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.tiktoken", "*.model"],
        max_workers=5,
    ), flush=True)


if __name__ == "__main__":
    main()
