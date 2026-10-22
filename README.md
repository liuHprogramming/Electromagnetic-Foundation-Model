# Frozen audio model baselines on DCASE 2020 Task 2

This repository records the 17–23 October 2026 pilot in the accompanying research plan: frozen BEATs features with nearest-normal-neighbor anomaly scoring, and ten Qwen2-Audio questions on machine recordings. It is a small, reproducible feasibility study, not a submission to the 2020 challenge. BEATs was released after the challenge's external-resource cutoff.

## Data and fixed protocol

The source is the [DCASE 2020 Task 2 development set](https://zenodo.org/records/3678171) (CC BY-NC-SA 4.0). We use **fan** and **pump**, machine ID **00** for each. A fixed Python RNG seed (`20261022`) samples, without replacement, 100 normal training clips, 50 normal test clips, and 50 anomalous test clips per type. `prepare_subset.py` checks the official archive MD5 values and records the SHA-256 of every extracted clip in `results/dcase2020/manifest.json`. Each recording is approximately 10 seconds of 16 kHz mono audio. The raw audio and model weights are downloaded separately and are not committed here.

The detector uses Microsoft's [BEATs implementation](https://github.com/microsoft/unilm/tree/master/beats) at source commit `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` and its **pre-trained BEATs_iter3** checkpoint ([checkpoint mirror](https://huggingface.co/lpepino/beats_ckpts), revision `5b53b0404df452a3a607d7e67687227730e5bad1`). It mean-pools time-frame embeddings and L2-normalizes each clip. For a test clip, the anomaly score is `1 - max(cosine_similarity(test, training_normal))` within the same type and ID. No model fine-tuning or score calibration is performed.

Test labels are used only to construct the balanced evaluation subset and calculate metrics; they never enter the embedding, nearest-neighbor bank, scoring, or any threshold. This experiment selects **no decision threshold**. We report AUROC and the scikit-learn standardized pAUC with `max_fpr=0.1`, matching the [official baseline configuration](https://github.com/y-kawagu/dcase2020_task2_baseline/blob/master/baseline.yaml). Higher anomaly scores mean more likely anomalous. The [challenge task page](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) explains the task and low-FPR metric.

## BEATs nearest-neighbor results

| Machine type | ID | Normal train | Normal test | Anomalous test | AUROC | pAUC, FPR ≤ 0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fan | 00 | 100 | 50 | 50 | 0.5932 | 0.5242 |
| pump | 00 | 100 | 50 | 50 | 0.7176 | 0.5832 |

`results/beats_knn/metrics.json` contains the exact numbers and checkpoint SHA-256 (`8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`); `scores.jsonl` retains every filename, label, and raw anomaly score. The 200 scored test clips independently reproduce the metrics above. Both types show only modest separation, especially at low FPR. Because this is only one machine ID per type with a balanced 100-clip test sample, these results should not be compared directly with full DCASE leaderboard numbers.

## Qwen2-Audio questions

We use [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), revision `0a095220c30b7b31434169c3086508ef3ea5bf0a`, in audio-analysis mode without fine-tuning. The ten **distinct source recordings** comprise four normal and four anomalous full-length clips (two of each condition from each machine type), plus two separate 0.25-second center excerpts. The excerpts are intentionally too short for a reliable condition judgment; “insufficient evidence” is a protocol expectation, not a DCASE annotation. The DCASE normal/anomaly labels are hidden from the model prompt.

The fixed prompt asks for `NORMAL`, `ANOMALOUS`, or `INSUFFICIENT EVIDENCE` followed by a brief audible reason, and forbids guessing an exact fault or operating setting. `results/qwen/qwen_raw.jsonl` retains the source path and SHA-256, segment timing and PCM hash, literal prompt and rendered chat template, unedited response, parsed answer, model revision, and generation settings for each case. The audio is supplied as a waveform array; the rendered template contains an audio token, not the source filename or condition label. `results/qwen/error_summary.json` groups basic label errors, abstentions, and format failures; `results/qwen/manual_review.json` records additional qualitative observations tied to case IDs.

| Error type | Count |
| --- | ---: |
| Wrong condition on a full clip | 4 |
| Correct condition on a full clip | 4 |
| Unnecessary abstention on a full clip | 0 |
| False certainty on a short excerpt | 2 |
| Response format failure | 0 |

All ten raw replies begin with `ANOMALOUS` (some use mixed case). This gives four false alarms among four normal full clips, four correct tags among four anomalous full clips, and no abstention on either 0.25-second excerpt. The model recycled nearly the same pitch/loudness explanation for both normal and anomalous fan clips (cases 01 and 05). Case 07 also asserted that a pump should sound different “at idle,” although the prompt supplied no operating-state reference. These are answer-level observations, not verified physical diagnoses. The ten questions are a diagnostic probe, not an accuracy estimate. A short excerpt's “insufficient” expectation is an assessment rule, not a claim that its parent recording lacks a DCASE condition label.

## Reproduce

The recorded A800 environment used Python 3.10, PyTorch `2.0.1+cu118`, TorchAudio `2.0.2+cu118`, Transformers `4.45.2`, Accelerate `0.34.2`, and Hugging Face Hub `0.25.2`. TorchAudio must match PyTorch's CUDA build. Other Python dependencies are listed in `requirements.txt`. BEATs extraction ran on the A800 host's CPU with eight threads, and Qwen2-Audio ran there in float32 on 16 CPU threads, because the GPU was occupied by another user's training job.

```bash
python prepare_subset.py --root data/dcase2020 --config config.json
HF_ENDPOINT=https://hf-mirror.com python download_models.py --out models
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git external/unilm
git -C external/unilm sparse-checkout set beats
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python run_beats_knn.py --root data/dcase2020 --checkpoint models/beats/BEATs_iter3.pt --beats-source external/unilm/beats --out results/beats_knn --config config.json --device cpu
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python run_qwen_audio.py --root data/dcase2020 --model models/qwen2-audio --out results/qwen --config config.json --device cpu
python analyze_qwen.py --input results/qwen/qwen_raw.jsonl --out results/qwen/error_summary.json
```

For exact source reproduction, check out the BEATs source commit above. The Qwen and BEATs checkpoint revisions are pinned in `download_models.py`. `run_qwen_audio.py` writes after each case and skips completed case IDs on restart. All computed results and the selected-file manifest are committed under `results/`; raw licensed WAV files and model checkpoints stay outside Git.

## Limits

The subset is small and drawn from only one ID per type. Clip-level mean pooling can hide brief anomalies. Training and test clips come from the official development archive, with its original split preserved, but this pilot does not measure domain shift or cross-device generalization. Qwen replies are evaluated only against the recorded DCASE condition or the deliberately short-clip abstention rule; they are not evidence of a physical fault diagnosis.
