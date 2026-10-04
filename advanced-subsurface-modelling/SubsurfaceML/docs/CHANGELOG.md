# Changelog

## 2026-10-04 — completion pass (no pipeline results changed)

A bounded pass to close correctness and publication issues. The pipeline's
results, models and figures are unchanged. Two checks were added and are
recorded separately from the original evaluation.

* **Calibration independence.** The 41 interval-calibration reservoirs had
  taken part in the development experiments that chose the design, so the
  pipeline's intervals carry measured coverage only, and every guarantee
  wording was corrected. A protocol addendum, pushed before the data existed
  (commit `4717b96`), recalibrated the fixed design on 41 fresh reservoirs:
  97 % / 91 % of test cases / reservoirs covered for peak build-up (original
  95 % / 86 %), at most 73 % of shift reservoirs
  (`results/study/experiments/calibration_check.json`).
* **Data roles.** The calibration reservoirs also train the pressure-limit
  classifier and serve as the domain check's reference set; "interval
  calibration only" was corrected.
* **Reproducibility.** A re-run of the ablation whose inputs were read from the stored feature table
  (equal to 2e-13) gave the same decisions and RMSEs within 0.014 MPa, but moved one prediction by
  4.36 MPa: a near-tied SVR/ridge choice flipped. The experiments now always rebuild their inputs from
  `scenarios.csv` and record their hash. Re-run this way, all six pressure variants are bit-identical
  to the record. Tolerances: `docs/TECHNICAL_REPORT.md` §8.
* **Documentation.** `docs/REVIEW_CHECKLIST.md` records each review
  recommendation as completed, remaining, portfolio-wide or future
  extension. The Supervisor Overview, README, technical report, assumptions
  and notebooks now state the measured coverages, the out-of-distribution
  limits, what simulator verification checks, and that the pressure limit is
  not a validated fracture or caprock criterion. PR #1's SubsurfaceML edits
  were reconciled: its "how to run" line and its note on the classifier's
  training data were adopted; its earlier split wording is superseded, and
  its attribution sentence was not adopted because it does not describe
  this revision.
* `subsurfaceml predict` / `simulate`: malformed input files give a one-line error and exit code 2,
  where they had raised a traceback.
* Tests: 103 → 107 (calibration-check set; experiment input source; command-line input handling).

## 2026-10 — revision: reliable surrogate-assisted screening (results regenerated)

Implemented after two portfolio reviews (only the English review was
available; see the README's version history). **All results in
`results/study/` and `results/demo/` were regenerated** with the code of this
version; the 2026-09-20 run was first reproduced end to end from a clean
environment (all 2,961 recorded values identical,
`results/published_2026-09-20/baseline_reproduction.json`).

### Findings that drove the changes

* **Pressure failures had one cause.** The published surrogate saw each
  reservoir through the parameters of its sampling prior (median
  permeability, target V_DP). With four layers per realisation, the realised
  rock can be several times tighter or more permeable than those parameters
  imply; the worst under-prediction (R0172: 32.4 MPa simulated, 17.6 MPa
  predicted) and the reservoir on which every screened schedule failed (R0036)
  both have realised mean permeability about one third of what their prior
  parameters imply.
* **No verification check covered the surrogate target.** V1–V3 verified the
  separate single-phase solver; the two-phase bottom-hole pressure had no
  analytical check.

### Added

* V12/V13: the two-phase simulator's bottom-hole pressure against the
  bounded-reservoir PSS solution (one layer, two commingled layers).
* `rom.py`: analytical multi-layer sealed-tank PSS model with a common BHP.
* Realised-layer inputs and ROM inputs; `hybrid.py` (ROM × learned
  correction).
* `numerics.py`: discretisation error of the dataset targets themselves, and a
  screen of every case for a peak set by a well-block transient
  (`scripts/run_experiments.py --only peak_screen`).
* `rz.py` + `scripts/run_model_form.py`: r–z reference model with gravity and
  vertical crossflow, measuring the model-form error of the layered model.
* `intervals.py`: reservoir-grouped split-conformal intervals (and three
  alternatives, compared); `domain.py`: applicability-domain check.
* `screening.py`: verification-gated screening; nothing is recommended
  without simulator verification; explicit no-recommendation outcomes.
* `final_eval.py`: fresh final-test and distribution-shift reservoirs,
  generated after the design was fixed (`docs/EVALUATION_PROTOCOL.md`).
* `experiments.py` + `scripts/run_experiments.py`: nested reservoir-grouped
  ablation and interval selection on development data only.
* `figures.py` / `labels.py`: every figure with readable axis labels.
* Tests: 72 → 103 (ROM, hybrid, intervals, domain, screening invariants,
  evaluation design and leakage, r–z model, discretisation study).

### Changed

* The published run's 55 test reservoirs (inspected during development) are
  now training data; testing uses fresh reservoirs.
* The old screening (`optimise_schedule`, which ranked candidates that were
  only predicted to be feasible) is removed; `optimise.py` keeps the
  candidate sampler.
* `pipeline.py` split into stages; plotting moved to `figures.py`.
* The course-coverage record of 2026-09-20 moved to
  `docs/archive/2026-09-20_course_coverage/` (not regenerated).

### Ablation (development reservoirs, nested grouped CV, out-of-fold RMSE of peak build-up)

| Variant | RMSE [MPa] | Worst under-prediction [MPa] |
|---|---|---|
| V0 published inputs | 1.138 | 15.8 |
| V1 + realised layers | 0.844 | 12.8 |
| V2 + ROM as an input | 0.458 | 8.4 |
| V3 hybrid (selected) | 0.266 | 4.6 |
| V4 ROM alone | 0.559 | 4.7 |
| V5 published inputs, 3× search | 1.169 | 15.8 |

Plume radius: 9.57 → 6.51 m with the new inputs (adopted); swept fraction:
0.00109 → 0.00112 (not adopted). Interval method: adaptive conformal
(whole-reservoir coverage 0.90 at the smallest width).

### Results

On 100 fresh test reservoirs (400 cases), scored once after the design was
fixed: peak build-up RMSE 0.31 MPa (published approach retrained on the same
reservoirs: 0.70; published models as released: 0.67), worst
under-prediction 2.7 MPa (6.7); interval 95 % of cases and 86 % of whole
reservoirs covered at 0.55 MPa mean width (published band 90 % / 85 % at
1.50 MPa). On 60 lower-permeability shift reservoirs: 1.95 vs 3.81 MPa; all
flagged out of domain. Screening at four simulations per reservoir: every
recommendation simulator-verified; median +2.9 % mass over the best constant
rate (published screening −9.6 %; ROM-ranked shapes +3.4 %). Discretisation
error of the training data (41 cases re-simulated with refined grid, well
block and time step): peak build-up median 0.06 %, no pressure-limit label
changes; at production resolution the plume radius is about 4 % and the swept
fraction about 17 % too large. A screen of all 1,520 cases found peaks set
by a start-up transient of the well block in 4 development, 0 test and 32
shift cases (2–14 % too high; one shift label changes). Omitting gravity and crossflow: peak build-up
−4 %, plume radius +88 % (medians). Details: `docs/TECHNICAL_REPORT.md` §4–5.
The full pipeline was run twice from clean data with identical machine-learning,
interval and screening results; re-running the development ablation reproduced
every design decision (`results/study/experiments/ablation_reproduction.json`).

## 2026-09-27 — publication review (no results changed)

Reviewed before publication in the Manchester MSc collection. **No simulation,
dataset, model, metric or figure was regenerated.**

Corrections:

* **Entry notebook, verification count.** It reported "19 of 19" checks because
  it counted the aggregate `ALL_PASS` flag as a check; it now reports the 18
  checks, as the rest of the documentation does.
* **Entry notebook, model families.** Its list named polynomial regression and
  extra trees, which are not among the compared families, and omitted AdaBoost.
  It now lists the eleven families actually compared, plus the mean baseline;
  "12 families" in the README and technical report is worded the same way.
* **Course-material path.** Both configurations held a Windows path, which on
  Linux was resolved as a relative path, so the supporting notebook s1 printed a
  meaningless location. The configurations no longer contain a local path; the
  optional check uses `SUBSURFACEML_COURSE_DIR` (read by the config loader).
* **Locked environment.** `requirements-lock.txt` lacked `ipykernel`, so in an
  environment built from it the notebooks ran in a different Python installation
  or not at all. It is now pinned.
* **Technical report scope.** Its opening said every number came from the demo
  run, although §12 reports the study run; the two are now distinguished.
* `LICENSE` (MIT) added — `pyproject.toml` already declared MIT; README rewritten
  in the collection's format with an execution record of the clean-copy checks;
  references to the author's local archives reworded; the superseded run is
  described but its files are not published.

## 2026-09-20b — presentation for publication (no results changed)

Preparation for a public repository. **No model, dataset, metric or figure was
regenerated**; `results/study/` and `results/demo/` are the runs of
2026-09-20 and the documentation quotes the same files.

* One entry point: `notebooks/00_START_HERE.ipynb` — purpose and assumptions →
  simulation → dataset and split → surrogate training → held-out evaluation →
  screening → conclusions and limitations. It reads the finished study run,
  simulates one case live, and recomputes the held-out metrics from the saved
  models as a check (recomputed values match the reported ones exactly).
* Every notebook output is labelled `[LOADED from the completed … run]` or
  `[COMPUTED NOW in this notebook session]`.
* The five earlier notebooks became optional supporting notebooks in
  `notebooks/supporting/` (`s1`–`s5`). All six are still generated and executed
  by `scripts/make_notebooks.py`.
* `README.md` rewritten: what the project is, results in one table, exact
  environment setup, reproduction commands with measured runtimes, folder map,
  limitations.
* `.gitignore` extended, including rules that keep the supplied teaching
  materials out of the repository.

## 2026-09-20 — corrected and scoped version (this release)

The original project is preserved unchanged in the author's local archive.
Its saved results are kept locally, clearly labelled, in
`results/historical/2026-09-11_pre-correction_demo/`; they are superseded and
are not published with this repository.

### Defects reproduced and fixed

| # | Finding | Reproduction (before) | Fix | Test |
|---|---|---|---|---|
| a | **Multi-layer well did not have one bottom-hole pressure.** Rates were split in proportion to `WI·λ_t` at the old level; the implied per-layer BHPs differed whenever well-block pressures differed, and the maximum was reported. | two layers, 500 mD/φ 0.25 vs 20 mD/φ 0.08, sealed: implied BHPs 103.3 vs 38.1 MPa | `p_bh` is now one extra unknown solved **implicitly** with the pressure field (exact elimination by superposition); prescribed total rate enforced; injector-only completions (active set); shut-in closes all completions | `tests/test_well_coupling.py` (7 tests: rate conservation, common BHP to < 1 mPa, old rule shown wrong, kh split in the open single-phase limit, pore-volume split in the sealed limit, shut-in with no crossflow, no back-flow); validation V11 |
| b | **Training and inference features differed.** Training used each realisation's `krg0` in the mobility ratio; the optimiser used 0.4. | `log10_mobility_ratio` off by up to 0.2 decades | one feature path, `features.py`, used by dataset, training, evaluation, screening, CLI and app | `tests/test_features.py` (identical inputs through 4 routes, bit-for-bit) |
| c | **Capillary-pressure switch was disconnected.** `RelPerm.pc` existed; the simulator never called it. | `pc_entry` 0, 5e4, 2e5 Pa gave bit-identical outputs | switch **removed**; `P_c = 0` stated as the course assumption (`4-CO2 BL.pdf` p.19). A working Pc implementation was built and verified first, then removed because the supplied material gives no CO2–brine Pc curve (scope rule) | `tests/test_simulator.py::test_no_dead_capillary_switch_remains` |
| d | **Classifier calibration folds were not grouped.** `CalibratedClassifierCV(cv=3)` mixed rows of one realisation across its folds. | code inspection | probability calibration removed (not in the course material); every score used for model or threshold choice is out-of-fold from `GroupKFold` on realisation | `tests/test_classifier_grouping.py` (a memorisation task: leaky folds AUC > 0.95, grouped folds < 0.75) |
| e | **No model files; notebooks unexecuted.** | `results/demo/models/` absent | `artifacts.py` writes models + `manifest.json` (SHA-256, feature list, versions, seeds, dataset hash, rebuild command) and refuses missing/modified/incompatible files with a rebuild message; notebooks executed | `tests/test_artifacts.py` |

### Further defects found during this work

* **Odd–even saturation oscillations near the well** (up to ~0.03–0.06 in
  `S_g`): the relaxed time-step controller accepted steps beyond the CFL
  limit and the maximum-principle check was one-sided with tolerance 0.02.
  Now: strict local CFL (0.9), two-sided check with tolerance 1e-3; V6
  counts profile turning points (strict: 1; relaxed: 9). Scalar QoIs changed
  by < 0.01 %, runtime about 2×.
* **Pressure round-off drift** of ~10 Pa over two years in a static sealed
  column (solving for absolute pressures with tiny storage); fixed by
  solving for pressure increments.
* **Schedule design collapsed for small compartments.** The absolute rate
  floor `q_min_kg_s = 1.0` exceeded the designed rates of many small or tight
  reservoirs (reference rates go down to 0.3 kg/s): 162 of 300 demo
  scenarios had at least one period clipped to 1 kg/s, 50 were flat at
  1 kg/s regardless of their sampled level and shape, and 16 were exact
  input duplicates of another schedule on the same reservoir. The floor is
  now 0.01 kg/s (a numerical guard only); the data-quality report counts
  rates on the floor/ceiling and exact duplicates are dropped before
  learning (`pipeline.deduplicate`). The historical run had the same flaw.
* `split_by_realisation` checked only the triple intersection of the three
  partitions; the pipeline's leakage check (pairwise) is the one relied upon.

### Scope reduction (material outside the supplied folders removed)

Theis line-source benchmark; Latin-hypercube sampling (→ Monte Carlo);
within-layer correlated random fields (exponential covariance); Lorenz
coefficient; geometric-mean upscaling comparison; transcription of an absent
"practical session" PDF (→ lecture 2×2 layout); split-conformal intervals
(→ empirical residual-percentile bands with measured coverage); isotonic /
sigmoid calibration and Brier score; time-series forecasting stage with
persistence/drift baselines; HistGradientBoosting and MLP (→ course
families: linear, ridge, lasso, elastic net, KNN, SVR, tree, random forest,
gradient boosting, AdaBoost, XGBoost); Nelder–Mead schedule refinement
(→ Monte Carlo screening + re-simulation); TF-IDF retrieval assistant.

### Added (all from the course notebooks)

Data-quality report; grouped model comparison over 11 families plus a mean baseline with physical
unit errors and difficult-case analysis; classifier comparison, imbalance
handling inside folds, recall-targeted threshold, ROC/PR; green/amber/red
multiclass screen; plume-shape PCA, kernel PCA, k-means, DBSCAN, t-SNE, UMAP;
permutation importance, impurity importance, SHAP, LIME-style local
explanation, what-if analysis; ML-method studies (learning curve, scalers,
missing descriptors, outliers, collinearity, filter/wrapper/embedded
selection, polynomial features, ColumnTransformer, grid/random/Bayesian
search, normal equation and gradient descent, grouped vs random split,
XGBoost early stopping, voting/stacking/bagging); frozen topic inventory and
automatic coverage matrix; `predict`/`simulate` CLI; reusable `Predictor`;
config validation; run log.

### Results

All results in `results/demo/` (and `results/study/` if run) come from this
version. Numbers from the historical run are not re-used.
