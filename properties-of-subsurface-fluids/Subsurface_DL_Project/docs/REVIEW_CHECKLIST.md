# October 2026 extension — record and open items (flash surrogate)

What the extension of 2026-10-04 added, where it is implemented and what shows
that it holds. No original result was changed and no baseline model was
retrained.

| Part | Implementation | Evidence |
|---|---|---|
| Baseline reproduced before any change; originals preserved and identifiable | `results/original_2026-09-21/` (`baseline_reproduction.json`, `MANIFEST_original.sha256`) | 185 of 185 metrics identical bit for bit with 2 threads (within 1.8e−9 relative with 4); SHA-256 of the dataset, the split index arrays and all six checkpoints recorded |
| Grouped split and training-only standardisation kept | unchanged `sfp/data.py`; every new script uses the saved split and each checkpoint's saved scaler | `test_group_split_keeps_mixtures_together`, `test_standardiser_uses_training_statistics_only` |
| Guarded prediction: input validation, the course phase test first, the network only inside its training domain, a labelled solver fallback otherwise | `src/sfp/predict.py`; `configs/prediction_domain.json`; [`GUARDED_PREDICTION.md`](GUARDED_PREDICTION.md) | 34 tests in `tests/test_predict.py`; 16 defects found in the first version, all fixed and tested; 14 worked cases in `results/guarded/demo_cases.json`, compared in CI |
| Training domain checked against the data | `scripts/derive_domain.py` | Pressure and temperature limits confirmed; composition limited to the realised per-component ranges (maxima 0.43–0.48) rather than the generator's 0.87 |
| Error across the two-phase window, per seed, in range and on extrapolation | `scripts/analyse_errors.py`; [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) | The dew band holds 45–58 % of the squared error; the physics loss is better in range in 3 of 3 seeds and worse on extrapolation in 2 of 3 |
| Capacity comparison, frozen before scoring | `configs/capacity.json`; `scripts/capacity.py`; [`CAPACITY.md`](CAPACITY.md) | Freeze commit `d7cd786` precedes training and scoring; 3 × 64 selected by the validation rule; the 3 × 128 benchmark and its 185 metrics unchanged |
| CI with measured tolerances and no training | `.github/workflows/subsurface-dl.yml`; `requirements-ci.txt`; `scripts/compare_results.py`, `scripts/ulp_sensitivity.py` | 185 metrics compared at rtol = atol = 1e−6, justified by one- and three-ULP float32 perturbations (`results/reproducibility/ulp_sensitivity.json`) |

## Not done, by design

* No new baseline training, no change to the dataset or the original metrics.
* No equation-of-state comparison (kept as a separate future study).
* No search beyond the three specified widths.
