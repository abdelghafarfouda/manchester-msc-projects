# Completion report — 2026-09-20

A factual record of what was changed, what was executed, what passed or
failed, and what remains open. Numbers come from `results/demo/` and
`results/study/` produced on 2026-09-20.

## 1. Starting point (verified, not assumed)

* `SubsurfaceML/` was byte-identical to the author's archive of the original
  version (86 files). The `Models/` folder = 6 PDFs identical to its archive, plus
  `3-Advanced BL.pptx` (not in the zip). The ML folder = 70 notebooks,
  `results.csv` and `utils.py`, identical to its zip; 6 byte-identical
  `(1)` duplicates.
* The 60 original tests passed.
* The earlier source map cited files that are **absent** (an upscaling
  "practical session" PDF, two MRST textbooks, the M3O paper) and misnamed
  the others. These attributions have been corrected (`docs/SOURCE_MAP.md`).

## 2. What changed

**The five reported defects** — each reproduced first, then fixed, with a
test that fails on the old behaviour:

| | Reproduction | Resolution |
|---|---|---|
| a. Well coupling | implied BHPs 103.3 vs 38.1 MPa (2 layers) | common BHP solved implicitly with pressure; injector-only completions; shut-in closes all completions. 7 tests + V11 |
| b. Feature mismatch | optimiser used `krg0 = 0.4` | single feature path (`features.py`); bit-identical inputs through 4 routes |
| c. Capillary pressure | outputs identical for 3 `P_c` values | switch removed; `P_c = 0` stated (`4-CO2 BL.pdf` p.19), because the material gives no CO2–brine `P_c` curve |
| d. Calibration folds | ungrouped `CalibratedClassifierCV` | calibration removed (not taught); grouped out-of-fold scores everywhere; leakage test |
| e. Missing models | no `models/` directory | models + SHA-256 manifest + verifying loader; notebooks executed |

**Further defects found and fixed:** near-well odd–even saturation
oscillation (relaxed time stepping); pressure round-off drift; a rate floor
of 1 kg/s that clipped the designed schedules of more than half the
scenarios (and produced duplicate inputs).

**Scope rule applied** (only methods in the supplied folders): removed or
replaced the Theis benchmark, Latin-hypercube sampling, correlated
within-layer fields, Lorenz coefficient, geometric-mean comparison,
conformal intervals, probability calibration, forecasting stage,
HistGradientBoosting/MLP, Nelder–Mead, and the retrieval assistant.

**Added:** course-based ML components and studies (see coverage), reusable
`Predictor` and CLI (`predict`, `simulate`, `validate`, `coverage`),
configuration validation, run logs, a worked example, six notebooks built
from one script (one walkthrough, `notebooks/00_START_HERE.ipynb`, plus five
supporting notebooks), `docs/TECHNICAL_REPORT.md`, `docs/WALKTHROUGH.md`,
`docs/CHANGELOG.md`.

The previous results were moved, unchanged and labelled, to
`results/historical/2026-09-11_pre-correction_demo/` in the author's local copy
(not published; see `results/historical/README.md`).

## 3. What was executed

| Command | Result |
|---|---|
| `pytest -q` | **72 passed**, 0 failed (80 s) |
| `python scripts/run_pipeline.py --config config/demo.yaml` | completed, **9.3 min** wall (validation 4.0, dataset 1.7, surrogates 1.7, studies 1.1, rest 0.8) |
| `python scripts/run_pipeline.py --config config/study.yaml` | completed, **47.6 min** wall (validation 7.4, dataset 24.4, surrogates 10.4, rest 5.4) |
| `python scripts/build_coverage.py --config config/demo.yaml` (and study) | 41/42 verified in both runs |
| `python scripts/make_notebooks.py --execute` | 6 notebooks executed without error (83 s; the walkthrough alone 24 s) |
| `subsurfaceml predict / simulate --input examples/worked_example_input.json` | ran; predicted/simulated build-up 6.80 / 7.01 MPa |
| Streamlit interface (`streamlit.testing` AppTest: load, Predict, Simulate, Screen) | no exceptions |

Machine: 2-core Linux cloud VM, Python 3.11.15, versions in
`requirements-lock.txt`. The local machine's shell lacked SciPy and has a
3-minute limit, so runs were done in the cloud workspace and the results
copied into the project folder.

## 4. Checks

* **Simulator verification: 18/18 pass** in both configurations (steady
  radial and well equation to ~1e-13, closed tank 6e-13, Buckley–Leverett
  front within 0.5% at 400 cells with first-order L1 convergence, CO2 mass
  balance ~1e-15, no clipping, no odd–even oscillation, upscaling bounds and
  exact limits, common BHP with 0 Pa spread and 2e-14 rate error).
* **Not converged (reported, not failed):** demo well block (`Δp_bh,max` 12%
  above the 5 m value), demo radial grid (`r_95` 9% vs 120 cells).
* **Data:** 300/300 and 880/880 simulations succeeded; no duplicates or
  missing values in either dataset.

## 5. Verified ML coverage

**41 / 42 = 97.6%** of the frozen, deduplicated topic inventory (42 topics,
all 73 files traced). The only uncovered topic is categorical encoding (P07):
the physical inputs contain no categorical variable, and inventing one would
be artificial. Every covered topic has a source, a purpose, code found in the
named file, executed evidence present in `summary.json` and a stated finding
(`docs/COVERAGE_MATRIX.md`).

## 6. Corrected performance (unseen reservoirs; synthetic data)

| | Demo (25 test reservoirs) | Study (55 test reservoirs) |
|---|---|---|
| `Δp_bh,max` RMSE / MAE / R² | 0.83 / 0.41 MPa / 0.970 (lasso) | 1.24 / 0.42 MPa / 0.941 (SVR) |
| `r_95` RMSE / R² | 7.6 m / 0.997 (OLS) | 10.5 m / 0.995 (elastic net) |
| sweep RMSE / R² | 0.0020 / 0.970 | 0.0012 / 0.989 |
| P5–P95 band coverage (measured) | 95 / 92 / 89% | 88 / 91 / 83% |
| pressure screen, test | AUC 0.988, 1 miss of 16, 0 false alarms | AUC 0.995, 3 misses of 51, 3 false alarms |
| screening vs constant rate (re-simulated) | median +2.8%, 0/15 violations | median +1.4%, **3/24 violations** (one reservoir) |
| speed-up batched / single call / end-to-end | ~11,600× / ~176× / ~52× | ~74,000× / ~1,000× / ~300× |

Speed-ups exclude data generation (103 s / 1460 s) and training (99 s / 623
s), and are specific to this machine. The historical ~10,900× / ~179× and
+7.3% figures are superseded.

## 7. Unresolved limitations

* Physics: no gravity/buoyancy (largest omission), dissolution, residual
  trapping, `P_c`, crossflow; radial symmetry — not 3-D field behaviour.
* The surrogate fails most on low-permeability reservoirs; on one such
  reservoir (study, reservoir 36) all screened schedules violated the stated
  pressure limit when re-simulated. The error band does not protect against
  reservoirs unlike the calibration set.
* The schedule design scales rates by a permeability-dependent reference
  rate, so rate and permeability are linked in the data.
* 25–55 test reservoirs cannot separate the top model families.
* Error bands are empirical; no coverage guarantee.
* No field data, no field validation, no deployment testing. The pressure
  and plume limits are stated assumptions, not safety limits.
* Model files load only with the recorded scikit-learn version (the loader
  enforces this and prints the rebuild command).
