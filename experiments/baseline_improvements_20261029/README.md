# Sound audit and magnetic improvements — 29 October 2026

The magnetic input and sample manifest remain exactly those of
`../magnetic_rebenchmark_20261028/`. No raw waveforms are reinterpreted, no
new acquisition-clock repair is claimed, and no GPU inference is necessary.
All numerical jobs run on a scheduler-selected remote CPU node with four
threads per process. Preserve the original data and embeddings as read-only.

## Reproduce

`BASE` must contain the unchanged `data/`, `features/` and `results/` from the
28 October run, including its SHA receipts. `OUT` is a new writable directory.

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python evaluate_magnetic.py --base "$BASE" --out "$OUT" --stage nested
python evaluate_magnetic.py --base "$BASE" --out "$OUT" --stage temporal
python evaluate_logistic.py --base "$BASE" --out "$OUT"
python summarize_verify.py --base "$BASE" --out "$OUT"
python audit_asd_metrics.py --input "$OLD_AUDIO_RESULTS/asd2026" --out "$OUT" --bootstrap 2000
```

`vendor/minirocket_multivariate.py` is the author's unchanged reference
implementation at commit `0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6`;
its original license is included. Pin its SHA to
`e48c16b9b9b122d3794638d4e01303396c2db76f44b561b6cd6b21331035d940`.
Dependencies are NumPy, SciPy, scikit-learn, Numba and threadpoolctl; actual
remote versions are included in the runtime receipts. No aeon/sktime install
or modification to shared environments is needed.

## Evaluation boundaries

- Same outer 3 held-device and 44 held-date folds; device-level OOF macro-F1
  is computed first, then the three devices receive equal final weight.
- Nested search compares three Ridge strengths and three RBF-SVM strengths.
  Its inner validation holds out source devices for strict device transfer,
  or groups training dates into three folds for within-device evaluation.
  Scalers and classifiers are fitted separately inside each training partition.
- The automatic selector chooses representation and head using inner scores
  alone; all family-specific results are also retained. No target examples,
  including unlabeled examples, are fitted in the strict held-device protocol.
- MiniRocket fits its random convolution biases on outer training samples
  only. Both its NumPy and Numba RNGs are seeded. Raw and sample-normalized
  views are predeclared alternatives; neither reconstructs a magnetic waveform.
- Fixed logistic probes use `StandardScaler + LogisticRegression(C=1,
  solver='lbfgs', max_iter=500)`. We test the author's default unweighted
  Mantis head and balanced heads for the three pretrained models plus a
  matched Mantis random-seed-42 control. Convergence warnings are retained.
- All alternatives and negative results remain in the report. Previous outer
  test results were already known, so these are exploratory comparisons on an
  existing benchmark, not fresh confirmatory generalization evidence.
- The sound script recomputes official domain AUROCs using normal samples of
  that domain plus **all** anomalies. It checks the old statistics and an
  independent pairwise-ranking calculation. It never changes model scores.

Large embeddings and source arrays stay on the remote archive. Git contains
the code, exact sample references, outer predictions, compressed split/selection
audit, metrics, source receipts and report. Intervals condition on these
three devices and the fitted folds; they do not estimate a hardware-population
interval or retrain the full search in each bootstrap draw.
