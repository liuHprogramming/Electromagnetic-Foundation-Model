"""Official frozen FISHER-small readout with nearest-normal cosine scoring."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import shlex
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

from run_beats_knn import digest, load_audio


def make_spectrogram(audio, sample_rate, config, band_width):
    if config["waveform_mean_center"]:
        audio = audio - audio.mean()
    transform = torchaudio.transforms.Spectrogram(
        n_fft=config["frame_length_ms"] * sample_rate // 1000,
        win_length=None,
        hop_length=config["frame_shift_ms"] * sample_rate // 1000,
        power=config["spectrogram_power"],
        center=config["spectrogram_center"],
    )
    spec = torch.log(torch.abs(transform(audio)) + config["log_epsilon"])
    spec = spec.transpose(-2, -1)
    spec = (spec - config["normalization_mean"]) / (
        config["normalization_std"] * config["normalization_std_multiplier"]
    )
    # Follow the model card: truncate long inputs; do not pad the time axis.
    spec = spec[:, :config["max_time_frames"]]
    if config["time_padding"]:
        raise ValueError("This protocol follows the model card without time padding")
    if spec.shape[-1] < band_width:
        spec = F.pad(spec, (0, band_width - spec.shape[-1]))
    return spec.unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fisher_knn"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--fisher-config", type=Path, default=Path("fisher_config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fisher = json.loads(args.fisher_config.read_text(encoding="utf-8"))
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if digest(args.model / "model.safetensors") != fisher["weights_sha256"]:
        raise ValueError("FISHER checkpoint SHA-256 mismatch")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # The pinned official custom code is downloaded and reviewed separately.
    model, loading = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32, output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete FISHER checkpoint load: {loading}")
    model = model.eval().to(args.device)
    band_width = model.cfg.band_width
    embeddings = {}
    input_shapes = set()
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, item in enumerate(manifest, 1):
            path = args.root / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Audio SHA-256 mismatch: {path}")
            audio = load_audio(path, config["sample_rate"])
            spec = make_spectrogram(audio, config["sample_rate"], fisher, band_width)
            input_shapes.add(tuple(spec.shape))
            embedding = model.extract_features(spec.to(args.device)).float()[0]
            if not torch.isfinite(embedding).all() or embedding.norm() == 0:
                raise ValueError(f"Invalid embedding: {path}")
            embeddings[item["path"]] = F.normalize(embedding, dim=0).cpu().numpy()
            if index == 1 or index % 50 == 0:
                print(f"Embedded {index}/{len(manifest)}; dim={embedding.numel()}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    rows, metrics = [], []
    for machine in config["machines"]:
        for machine_id in config["machine_ids"]:
            group = [x for x in manifest if x["machine"] == machine and x["machine_id"] == machine_id]
            train = [x for x in group if x["split"] == "train"]
            test = [x for x in group if x["split"] == "test"]
            if not train or any(x["condition"] != "normal" for x in train):
                raise ValueError("The nearest-neighbor bank must contain only normal training audio")
            bank = np.stack([embeddings[x["path"]] for x in train])
            labels, scores = [], []
            for item in test:
                score = float(1 - np.max(bank @ embeddings[item["path"]]))
                label = int(item["condition"] == "anomaly")
                labels.append(label)
                scores.append(score)
                rows.append({"path": item["path"], "machine": machine, "machine_id": machine_id,
                             "label": label, "anomaly_score": score})
            metrics.append({"machine": machine, "machine_id": machine_id,
                            "n_train_normal": len(train), "n_test_normal": labels.count(0),
                            "n_test_anomaly": labels.count(1),
                            "auroc": float(roc_auc_score(labels, scores)),
                            "pauc_0_1_standardized": float(roc_auc_score(labels, scores, max_fpr=config["pauc_max_fpr"]))})
    (args.out / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    paths = list(embeddings)
    np.savez_compressed(args.out / "embeddings.npz", paths=np.array(paths),
                        embeddings=np.stack([embeddings[x] for x in paths]))
    result = {"dataset": config["dataset"], "model": "FISHER-small-0723",
              "model_repo": fisher["model_repo"], "model_revision": fisher["revision"],
              "checkpoint_sha256": fisher["weights_sha256"], "metrics": metrics,
              "pauc_convention": "sklearn standardized pAUC, max_fpr=0.1",
              "test_labels_used_for": "metrics only; not embedding, bank, score or threshold"}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    metadata = {"completed_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                "python": platform.python_version(), "device": args.device, "dtype": "float32",
                "threads": torch.get_num_threads(), "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                "versions": {p: version(p) for p in ("torch", "torchaudio", "transformers", "numpy", "scikit-learn", "timm", "omegaconf", "einops", "safetensors")},
                "model_repo": fisher["model_repo"], "model_revision": fisher["revision"],
                "model_files_sha256": {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
                "script_sha256": digest(Path(__file__)), "config_sha256": digest(args.config),
                "fisher_config_sha256": digest(args.fisher_config), "manifest_sha256": digest(manifest_path),
                "embedding_dimension": len(next(iter(embeddings.values()))),
                "input_shapes": [list(x) for x in sorted(input_shapes)], "band_width": band_width,
                "loading_info": loading, "command": shlex.join([sys.executable, *sys.argv])}
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
