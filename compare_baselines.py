"""Verify paired baseline results and reproduce the comparison from saved scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(directory):
    rows = [json.loads(line) for line in (directory / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    indexed = {row["path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate scored paths")
    return indexed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.results
    manifest_path = root / "dcase2020/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test = {x["path"]: x for x in manifest if x["split"] == "test"}
    methods = {name: load_scores(root / name) for name in ("beats_knn", "fisher_knn")}
    tables = {}
    for name, rows in methods.items():
        require(set(rows) == set(test), f"Test path mismatch for {name}")
        saved = json.loads((root / name / "metrics.json").read_text(encoding="utf-8"))
        for path, row in rows.items():
            item = test[path]
            require(row["label"] == int(item["condition"] == "anomaly"), f"Label mismatch: {path}")
            require(row["machine"] == item["machine"] and row["machine_id"] == item["machine_id"], f"Group mismatch: {path}")
            require(np.isfinite(row["anomaly_score"]), f"Nonfinite score: {path}")
        tables[name] = {}
        for metric in saved["metrics"]:
            group = [r for r in rows.values() if r["machine"] == metric["machine"] and r["machine_id"] == metric["machine_id"]]
            labels = [r["label"] for r in group]
            scores = [r["anomaly_score"] for r in group]
            recomputed = {"auroc": float(roc_auc_score(labels, scores)),
                          "pauc_0_1_standardized": float(roc_auc_score(labels, scores, max_fpr=0.1))}
            for key, value in recomputed.items():
                require(abs(metric[key] - value) < 1e-12, f"Metric mismatch: {name}/{key}")
            tables[name][(metric["machine"], metric["machine_id"])] = recomputed

    # Reconstruct FISHER scores via squared Euclidean distance between unit vectors,
    # independently of the inference script's matrix-product implementation.
    with np.load(root / "fisher_knn/embeddings.npz", allow_pickle=False) as archive:
        paths = archive["paths"].tolist()
        vectors = archive["embeddings"]
    require(len(set(paths)) == len(paths) == len(manifest), "Embedding path/count mismatch")
    require(set(paths) == {x["path"] for x in manifest}, "Embedding coverage mismatch")
    require(np.isfinite(vectors).all(), "Nonfinite embeddings")
    require(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6), "Embeddings are not unit vectors")
    features = dict(zip(paths, vectors.astype(np.float64)))
    max_score_error = 0.0
    for path, item in test.items():
        bank = [features[x["path"]] for x in manifest if x["split"] == "train"
                and x["condition"] == "normal" and x["machine"] == item["machine"]
                and x["machine_id"] == item["machine_id"]]
        score = float(np.min(np.sum((np.stack(bank) - features[path]) ** 2, axis=1)) / 2)
        max_score_error = max(max_score_error, abs(score - methods["fisher_knn"][path]["anomaly_score"]))
    require(max_score_error < 1e-6, "Saved FISHER scores do not match the saved embeddings")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for name in methods:
        metadata = json.loads((root / name / "run_metadata.json").read_text(encoding="utf-8"))
        require(metadata["manifest_sha256"] == manifest_hash, f"Different manifest used for {name}")
    comparison = []
    for machine, machine_id in sorted(tables["beats_knn"]):
        beats, fisher = (tables[name][(machine, machine_id)] for name in methods)
        comparison.append({"machine": machine, "machine_id": machine_id,
                           "beats": beats, "fisher": fisher,
                           "fisher_minus_beats": {key: fisher[key] - beats[key] for key in beats}})
    output = {"verification": "passed", "n_manifest": len(manifest), "n_test": len(test),
              "manifest_sha256": manifest_hash, "fisher_embedding_shape": list(vectors.shape),
              "max_fisher_score_reconstruction_error": max_score_error,
              "comparison": comparison,
              "scope": "Descriptive differences on one fixed small subset; no significance or full-benchmark claim"}
    (root / "baseline_comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
