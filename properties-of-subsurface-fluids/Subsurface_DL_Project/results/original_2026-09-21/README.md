# The original results (2026-09-21) and their reproduction (2026-10-04)

The results of the version published in September 2026 are kept **where they
were**, unchanged:

* `data/flash_dataset.npz`, the dataset and its split;
* `results/checkpoints/ffn_phys{0,1}_s{0,1,2}.pt`, the six reported models;
* `results/metrics/*.json`, including `evaluation.json` with its 185 metrics;
* `results/figures/fig1`–`fig5`;
* `logs/`.

The October 2026 revision writes only to new places:
`results/analysis/`, `results/capacity/`, `results/guarded/`,
`results/figures/fig6`–`fig9`, `configs/prediction_domain.json`,
`configs/capacity.json`. It reran nothing in place.

This folder holds the evidence that identifies and reproduces those originals.

| file | what it is |
|---|---|
| `MANIFEST_original.sha256` | SHA-256 of all 54 files of the project at commit `e782d2e` (the last change before the revision), as they were published |
| `baseline_reproduction.json` | the reproduction record described below, with hashes of the dataset arrays, the split, and every model's weights |
| `rerun_2026-10-04/*.log` | console output of that reproduction |

## The reproduction, before any change

* **Code and environment.** An unmodified copy of the project at
  repository commit `1afdd61` (`main`), outside the working tree. Python 3.11.15
  on Linux x86-64 with 4 CPUs, NumPy 2.4.4, PyTorch 2.14.0 (the PyPI build,
  run on the CPU, as in the recorded run) and Matplotlib 3.10.9.
* **`python scripts/verify_flash.py`.** PASS on all three worked examples;
  `results/metrics/verify_flash.json` was byte-identical.
* **`python tests/run_tests.py`.** 14 passed, 0 failed.
* **`python scripts/evaluate.py`, with the recorded run's 2 torch threads.**
  All **185** numerical metrics of `results/metrics/evaluation.json` (the 8
  timing values excluded) were identical bit for bit, as were the 6 booleans.
* **The same, with 4 threads.** All 185 within a relative 1.8e−09: the float32
  sums are accumulated in a different order. This is why CI uses tolerances
  rather than exact equality.
* **Figures and dataset report.** `fig1`–`fig5` and `dataset.json` were
  byte-identical.
* **No retraining.** The dataset and the six saved models were reused.

Split, checked from the saved arrays: 36,000 / 12,000 / 12,000 rows from
1,800 / 600 / 600 mixtures, with **no mixture in more than one part**.
