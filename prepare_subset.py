"""Download and extract a fixed, compact DCASE 2020 Task 2 dev subset."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import random
import zipfile

import requests


URL = "https://zenodo.org/api/records/3678171/files/dev_data_{machine}.zip/content"
ARCHIVE_MD5 = {"fan": "649bdfc06263ae7a838963f43b6641e6",
               "pump": "90e7091ef722b7238a7f1009365779cd"}
NAME = re.compile(r"^(normal|anomaly)_id_(\d\d)_.*\.wav$")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    if path.exists():
        print(f"Using existing archive: {path}", flush=True)
        return
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with open(tmp, "wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    file.write(chunk)
    tmp.replace(path)


def check_archive(path, expected_md5):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"Archive MD5 mismatch or incomplete download: {path}")


def parse_member(name, machine, machine_id):
    parts = Path(name).parts
    if len(parts) < 3:
        return None
    for i in range(len(parts) - 2):
        if parts[i] != machine or parts[i + 1] not in {"train", "test"}:
            continue
        match = NAME.fullmatch(parts[i + 2])
        if not match or match.group(2) != machine_id:
            continue
        condition = match.group(1)
        split = parts[i + 1]
        if split == "train" and condition != "normal":
            continue
        return split, condition, parts[i + 2]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = random.Random(config["seed"])
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for machine in config["machines"]:
        archive = root / f"dev_data_{machine}.zip"
        download(URL.format(machine=machine), archive)
        check_archive(archive, ARCHIVE_MD5[machine])
        with zipfile.ZipFile(archive) as zf:
            for machine_id in config["machine_ids"]:
                groups = {}
                for member in zf.namelist():
                    parsed = parse_member(member, machine, machine_id)
                    if parsed:
                        split, condition, filename = parsed
                        groups.setdefault((split, condition), []).append((filename, member))
                for (split, condition), limit in {
                    ("train", "normal"): config["train_normal_per_id"],
                    ("test", "normal"): config["test_normal_per_id"],
                    ("test", "anomaly"): config["test_anomaly_per_id"],
                }.items():
                    available = sorted(groups.get((split, condition), []))
                    if len(available) < limit:
                        raise RuntimeError(f"Too few files for {machine} {machine_id} {split} {condition}: {len(available)}")
                    selected = sorted(rng.sample(available, limit))
                    for filename, member in selected:
                        relative = Path(machine) / split / filename
                        destination = root / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists():
                            with zf.open(member) as source, open(destination, "wb") as output:
                                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                    output.write(chunk)
                        entries.append({
                            "path": relative.as_posix(), "machine": machine,
                            "machine_id": machine_id, "split": split,
                            "condition": condition, "sha256": sha256(destination),
                        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {manifest}", flush=True)


if __name__ == "__main__":
    main()
