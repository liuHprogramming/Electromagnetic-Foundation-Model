"""Ten deterministic audio questions; store every prompt, audio reference and raw reply."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import transformers
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def audio_array(path, target_rate):
    array, rate = sf.read(path, dtype="float32")
    if array.ndim == 2:
        array = array.mean(axis=1)
    if rate != target_rate:
        from math import gcd
        divisor = gcd(rate, target_rate)
        array = resample_poly(array, target_rate // divisor, rate // divisor).astype("float32")
    return array


def select_cases(manifest, config):
    test = [x for x in manifest if x["split"] == "test"]
    cases = []
    per_group = config["qwen_per_type_per_class"]
    for condition in ("normal", "anomaly"):
        for machine in ("fan", "pump"):
            matches = sorted((x for x in test if x["condition"] == condition and x["machine"] == machine), key=lambda x: x["path"])
            for item in matches[:per_group]:
                cases.append((item, condition, None))
    for machine, condition in (("fan", "normal"), ("pump", "anomaly")):
        matches = sorted((x for x in test if x["condition"] == condition and x["machine"] == machine), key=lambda x: x["path"])
        cases.append((matches[per_group], "insufficient", config["insufficient_duration_seconds"]))
    assert len(cases) == 10 and len({x[0]["path"] for x in cases}) == 10
    return cases


def label_from_response(response):
    leading = response.strip().upper()
    for token in ("INSUFFICIENT EVIDENCE", "ANOMALOUS", "NORMAL"):
        if re.match(r"^[\s`*#:\-]*" + re.escape(token) + r"\b", leading):
            return token.lower().replace(" evidence", "")
    return "unparseable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    device = "cuda:0" if args.device == "cuda" else "cpu"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    sample_rate = processor.feature_extractor.sampling_rate
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "qwen_raw.jsonl"
    completed = set()
    if output.exists():
        completed = {json.loads(line)["case_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    for index, (item, expected, short_seconds) in enumerate(select_cases(manifest, config), 1):
        case_id = f"case_{index:02d}"
        if case_id in completed:
            print(f"Skipping completed {case_id}", flush=True)
            continue
        source_path = args.root / item["path"]
        audio = audio_array(source_path, sample_rate)
        start_sample = 0
        if short_seconds is not None:
            length = round(short_seconds * sample_rate)
            start_sample = max(0, (len(audio) - length) // 2)
            audio = audio[start_sample:start_sample + length]
        prompt = config["qwen_prompt"]
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "file://" + str(source_path)},
            {"type": "text", "text": prompt},
        ]}]
        rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=rendered, audios=[audio], sampling_rate=sample_rate,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        record = {
            "case_id": case_id, "source_path": item["path"],
            "source_sha256": item["sha256"], "original_condition": item["condition"],
            "expected_answer": expected, "segment_start_seconds": start_sample / sample_rate,
            "segment_duration_seconds": len(audio) / sample_rate,
            "segment_pcm_f32_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
            "sample_rate": sample_rate, "prompt": prompt, "rendered_chat_template": rendered,
            "raw_output": response, "parsed_answer": label_from_response(response),
            "model": "Qwen/Qwen2-Audio-7B-Instruct",
            "model_revision": "0a095220c30b7b31434169c3086508ef3ea5bf0a",
            "transformers_version": transformers.__version__,
            "generation": {"max_new_tokens": 96, "do_sample": False,
                           "device": device, "dtype": str(dtype)},
        }
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{index}/10 {item['path']} -> {response}", flush=True)
    if args.device == "cuda":
        print("peak_cuda_memory_gib", round(torch.cuda.max_memory_allocated() / 2**30, 3), flush=True)


if __name__ == "__main__":
    main()
