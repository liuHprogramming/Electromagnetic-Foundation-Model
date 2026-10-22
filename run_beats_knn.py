"""Frozen BEATs embeddings and per-machine-ID 1-nearest-normal scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
import torch


def digest(path):
    hash_ = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def load_audio(path, sample_rate):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        from math import gcd
        factor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, sr // factor).astype("float32")
    return torch.from_numpy(audio.copy())[None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--beats-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(args.beats_source))
    from BEATs import BEATs, BEATsConfig

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.0 has no weights_only argument
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(args.device)
    if model.predictor is not None:
        raise ValueError("Use a pre-trained BEATs checkpoint, not an AudioSet classifier")

    embeddings = {}
    with torch.inference_mode():
        for i, item in enumerate(manifest, 1):
            path = args.root / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Audio hash mismatch: {path}")
            audio = load_audio(path, config["sample_rate"]).to(args.device)
            features, _ = model.extract_features(audio)
            pooled = features.mean(dim=1).float()[0]
            pooled = torch.nn.functional.normalize(pooled, dim=0)
            embeddings[item["path"]] = pooled.cpu().numpy()
            if i % 50 == 0:
                print(f"Embedded {i}/{len(manifest)}", flush=True)

    rows = []
    metrics = []
    for machine in config["machines"]:
        for machine_id in config["machine_ids"]:
            group = [e for e in manifest if e["machine"] == machine and e["machine_id"] == machine_id]
            train = [e for e in group if e["split"] == "train"]
            test = [e for e in group if e["split"] == "test"]
            bank = np.stack([embeddings[e["path"]] for e in train])
            labels, scores = [], []
            for item in test:
                score = float(1 - np.max(bank @ embeddings[item["path"]]))
                label = int(item["condition"] == "anomaly")
                labels.append(label)
                scores.append(score)
                rows.append({"path": item["path"], "machine": machine,
                             "machine_id": machine_id, "label": label, "anomaly_score": score})
            if set(labels) != {0, 1}:
                raise ValueError(f"Both test classes required for {machine}/{machine_id}")
            metrics.append({"machine": machine, "machine_id": machine_id,
                            "n_train_normal": len(train),
                            "n_test_normal": labels.count(0),
                            "n_test_anomaly": labels.count(1),
                            "auroc": float(roc_auc_score(labels, scores)),
                            "pauc_0_1_standardized": float(roc_auc_score(labels, scores, max_fpr=0.1))})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = {"dataset": config["dataset"], "model": "BEATs_iter3", "checkpoint_sha256": digest(args.checkpoint),
              "metrics": metrics, "pauc_convention": "sklearn standardized pAUC, max_fpr=0.1",
              "test_labels_used_for": "metrics only; not embedding, bank, score or threshold"}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
