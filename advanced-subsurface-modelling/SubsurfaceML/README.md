# SubsurfaceML — CO₂ injection into a layered saline aquifer: simulator and surrogates

A radial IMPES simulator for CO₂ injection into a sealed, layered saline
aquifer, written in Python and verified against analytical and conservation
results, is used to generate 880 synthetic injection cases. Machine-learning
surrogates trained on those cases are then tested on reservoirs they have never
seen, and used — with re-simulation as the final check — to screen injection
schedules against an assumed pressure limit.

**All data are synthetic.** Every case is produced by the simulator in this
repository. Nothing is calibrated to, or validated against, a real site, and the
pressure and plume limits are stated assumptions, not safety limits.

## Question

In a sealed, layered saline-aquifer compartment, how do permeability,
heterogeneity, fluid mobility and the injection schedule control the peak
pressure build-up at the well and the spread of the CO₂ plume — and how
accurately can an inexpensive surrogate, trained on simulations, predict those
outcomes for a reservoir outside its training set?

## Source and scope

The project uses only methods, equations and algorithms from two sets of MSc
course material for *Advanced Subsurface Modelling*:

* **`Models`** — lecture notes on transmissibility, upscaling, IMPES, CO₂
  Buckley–Leverett and uncertainty (the physics and numerics);
* **`Data Science and machine learning`** — Jupyter notebooks on preprocessing,
  model families, evaluation, tuning, interpretation, imbalance and unsupervised
  learning (the machine learning).

[`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md) traces every main component to a
specific file and page or notebook, and marks each choice the material does not
dictate as a *project choice*. [`docs/COVERAGE_MATRIX.md`](docs/COVERAGE_MATRIX.md)
records which machine-learning topics from the notebooks are used, where, and
with what evidence. No external dataset, pretrained model or additional physics
is used. The course material itself is not included in this repository.

## Start here

**[`notebooks/00_START_HERE.ipynb`](notebooks/00_START_HERE.ipynb)** — the
whole project in order: purpose and assumptions → one simulation run live →
simulator verification → dataset and grouped split → surrogate training →
held-out evaluation (recomputed in the notebook from the saved models) →
schedule screening → conclusions and limitations. It runs in well under a minute
because it reads a finished run and simulates only one case.

Every output in the notebooks is labelled with its provenance:

| Label | Meaning |
|---|---|
| `[LOADED from the completed study run]` | read from `results/study/`, produced by `scripts/run_pipeline.py` |
| `[COMPUTED NOW in this notebook session]` | calculated while the cells run |

Optional, deeper analyses are in [`notebooks/supporting/`](notebooks/supporting):
s1 sources and ML-topic coverage · s2 simulator verification in detail · s3
machine-learning methods and studies · s4 schedule screening on an unseen
reservoir · s5 a worked example from raw inputs to prediction. These read the
smaller demo run.

## Model

One rate-controlled injector sits at the centre of a sealed cylindrical
compartment divided into horizontal layers. Within each layer the flow is
radial, isothermal, immiscible two-phase flow of CO₂ displacing brine. All
layers share **one bottom-hole pressure**, solved together with the pressure
field, so the rate split between layers follows from the physics rather than
being imposed.

| Deliberately left out | Consequence |
|---|---|
| Gravity / buoyancy | the largest omission: no gravity override, so plume rise under the caprock is not modelled |
| Dissolution and residual trapping | storage security cannot be assessed |
| Capillary pressure (`P_c = 0`) | the material gives no CO₂–brine capillary curve (`4-CO2 BL.pdf` p.19) |
| Vertical crossflow between layers | layers communicate only through the well |
| Non-radial geometry, faults, more than one well | not a field model |

Every assumption that can change a number is listed in
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

## Method

* **Simulator** (`src/subsurfaceml/impes.py`). IMPES — implicit pressure,
  explicit saturation — on a log-spaced radial grid, with upstream mobilities and
  harmonic face transmissibilities (`3-IMPES.pdf`, `1-Transmissibility.pdf`). The
  CO₂ saturation update is conservative, so CO₂ mass balances to round-off. Time
  steps obey a local CFL limit and are rejected and halved if a saturation
  overshoots.
* **Verification** (`validation.py`). 18 checks, run before any data are
  generated; the pipeline stops if one fails (see below).
* **Dataset** (`scenarios.py`, `features.py`). Reservoir descriptions and
  four-period injection schedules are drawn by Monte Carlo sampling from assumed
  generic ranges (`5-Uncertainty.pdf` p.42–45) and simulated. One feature path
  builds the model inputs for training, the notebooks, the command line and the
  screening.
* **Grouped split** (`splits.py`). The unit of learning is the reservoir
  realisation: its schedules share one reservoir, so rows are not independent.
  Each realisation is placed wholly in training, calibration or test
  (`Lecture08.ipynb`, `E03_geographicalspliting.ipynb`).
* **Surrogates** (`models.py`). Eleven regression families from the notebooks —
  linear, ridge, lasso, elastic net, k-nearest neighbours, SVR, decision tree,
  random forest, gradient boosting, AdaBoost and XGBoost — plus a mean-value
  baseline, each in a pipeline with median imputation and standard scaling, are
  tuned by randomised search inside grouped 4-fold cross-validation on the
  training reservoirs. The family with the lowest cross-validated RMSE is
  selected per target. Pressure build-up and plume radius are fitted on a log
  scale.
* **Error bands** (`uncertainty.py`). Empirical P5–P95 residual percentiles from
  the calibration reservoirs; their coverage on the test reservoirs is measured
  and reported, not assumed.
* **Pressure screen** (`classify.py`). A classifier for "exceeds the assumed
  limit", selected on grouped out-of-fold scores, with imbalance handling inside
  the folds and a recall-targeted threshold.
* **Schedule screening** (`optimise.py`). Monte Carlo screening of candidate
  schedules with the surrogates, then **re-simulation** of the shortlist and of
  a constant-rate baseline; violations are reported.

The full derivation and discussion are in
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md); a guided tour of the
code is in [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

## Verification

This is numerical verification — it checks that the equations are solved
correctly. It is **not validation**: no measurements exist for this problem.
All 18 checks pass in both recorded runs (`results/*/metrics/validation.json`).

| Checks | Result (study run) | What it establishes |
|---|---|---|
| Steady radial pressure and the well equation, three well-block sizes | relative error ≤ 5 × 10⁻¹³ | the radial transmissibilities and the well index are assembled correctly |
| Closed-tank material balance | relative error 6 × 10⁻¹³ | compressible storage and the sealed boundary are consistent |
| Buckley–Leverett / Welge front, 100 / 200 / 400 cells | L1 error 0.012 / 0.006 / 0.003; front within 0.5 % at 400 cells | the two-phase transport converges to the analytical solution at first order |
| Radial two-phase: CO₂ mass balance, saturation bounds, no clipping, no odd–even oscillation, insensitivity to the time-step controller | mass error ~ 10⁻¹⁵; all pass | the IMPES time stepping is conservative and stable on the production grid |
| Upscaling: exact series and parallel limits, bounding chain on random and 2 × 2 fields | limits exact to 10⁻¹⁶; bounds hold | the upscaling formulae of `2-Upscaling.pdf` are implemented correctly |
| Multi-layer well: common bottom-hole pressure, total rate, shut-in | BHP spread 0 Pa; rate error 2 × 10⁻¹⁴; no crossflow at shut-in | the layers share one well pressure and the prescribed rate is honoured |

Grid, well-block and time-step convergence are also measured (not pass/fail).
In the verification case, the study configuration's 5 m well block still gives
a peak build-up 2.6 % higher than a 2.5 m block, and the demo configuration's
10 m block one 12 % higher than a 5 m block. These discretisation errors are
carried into the error-source table of the technical report.

## Results

### Study run — the reference results (`results/study/`)

220 reservoir realisations × 4 injection schedules = **880 simulations**, 0
failed. Split by realisation into 124 training, 41 calibration and **55 test
reservoirs** (496 / 164 / 220 cases); no reservoir appears in two partitions.

| Target (220 held-out cases) | Selected family | RMSE | MAE | R² | Worst case | Mean-baseline RMSE | P5–P95 band coverage |
|---|---|---|---|---|---|---|---|
| Peak pressure build-up | SVR | **1.24 MPa** | 0.42 MPa | **0.941** | 14.8 MPa | 5.08 MPa | 88 % |
| Plume radius (95 % of CO₂ mass) | elastic net | 10.5 m | 6.5 m | 0.995 | 57 m | 142 m | 91 % |
| Sweep efficiency | SVR | 0.0012 | 0.0008 | 0.989 | 0.0052 | 0.0108 | 83 % |

The exact values (e.g. pressure RMSE 1.2373 MPa, R² 0.9406) are in
`results/study/metrics/summary.json` and `results/study/reports/RESULTS.md`.
The error bands were calibrated to cover 90 % on the calibration reservoirs;
their measured coverage on the test reservoirs is 83–91 %.

![Surrogate parity on held-out reservoirs](results/study/figures/06_surrogate_parity.png)

**Where the surrogate is weak.** Pressure errors concentrate in
low-permeability reservoirs (mean absolute error 0.78 MPa in the lowest
permeability tercile against 0.24 and 0.21 MPa in the other two). One case
(45 mD; 32.4 MPa simulated, 17.6 MPa predicted) dominates the RMSE.

**Pressure screen.** Assumed limit: 9 MPa build-up. On the test reservoirs the
classifier scores ROC-AUC 0.995 with 3 missed exceedances out of 51 and 3 false
alarms out of 169. Thresholding the regression surrogate does as well (3 missed,
1 false alarm).

**Schedule screening — including where it fails.** For 8 test reservoirs,
surrogate-screened schedules were re-simulated together with a constant-rate
baseline. On the 7 reservoirs where a comparison was possible the median gain in
injected mass was only **+1.4 %**. On the eighth, a 40 mD reservoir (realisation
36), **all 3 screened schedules and the constant-rate baseline violated the
assumed pressure limit** when re-simulated (10.6–14.8 MPa against 9 MPa), although
the surrogate predicted 7.6–7.7 MPa with an upper band edge of 8.9 MPa. The
surrogate had learned the common case and the band, calibrated on other
reservoirs, did not cover this one: the screening can shortlist schedules, but
only the simulator can decide.

![Schedule screening, re-simulated](results/study/figures/15_schedule_screening.png)

**Cost.** On the recording machine a simulation took 5.5 s and the three
surrogates about 18 ms per case end to end (≈ 300× faster; ≈ 74 000× in a
batch). This excludes generating the data (1460 s) and training (623 s), and it
applies to three scalar outputs only; the simulator also gives full fields and
time series.

### Demo run (`results/demo/`)

A smaller configuration (100 realisations × 3 schedules = 300 simulations,
coarser grid, 25 test reservoirs) that reproduces the whole pipeline in about
ten minutes. Its numbers — for example pressure RMSE 0.83 MPa, R² 0.970 — come
from a different, coarser configuration and a smaller test set; they are not
the reference results and should not be combined with the study's.

### Superseded run

An earlier run made before the corrections of 2026-09-20 is **not included**
here; its known defects are listed in [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
and [`results/historical/README.md`](results/historical/README.md), and none of
its numbers is used.

## Limitations

* **Synthetic data only**; no field data, history match or field validation.
* **Physics**: no gravity (the largest omission for CO₂ plumes), dissolution,
  residual trapping or capillary pressure; no vertical crossflow; isothermal;
  radially symmetric with one well. The results do not describe 3-D field
  behaviour.
* **Numerics**: the well block is not fully converged (in the verification
  case, 2.6 % on peak build-up for the study's 5 m block, 12 % for the demo's
  10 m block).
* **Surrogates**: weakest on low-permeability reservoirs; the empirical error
  band gives no protection for a reservoir unlike the calibration set, as the
  screening failure shows. 55 test reservoirs cannot separate the best model
  families from one another.
* **Design**: schedules are scaled by a permeability-dependent reference rate,
  so rate and permeability are correlated by construction in the dataset.
* **Limits** (9 MPa build-up, 400 m plume radius) are stated assumptions; nothing
  about fracture pressure, caprock integrity or leakage is modelled.
* **Saved models** load only with the scikit-learn version recorded in their
  manifest; otherwise the loader stops and prints the command that rebuilds them.

## Run it

The saved models are version-specific, so use the pinned versions. Python 3.11:

```bash
git clone https://github.com/abdelghafarfouda/manchester-msc-projects.git
cd manchester-msc-projects/advanced-subsurface-modelling/SubsurfaceML
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

(`environment.yml` is the conda equivalent; `requirements.txt` lists the same
dependencies unpinned.) To open the notebooks interactively, use any Jupyter
front end — for example `python -m pip install notebook` and then
`jupyter notebook`, or VS Code.

Runtimes below were measured on a 2-core Linux machine (details in the next
section); the two pipeline runtimes are those of the runs that produced the
recorded results.

| Command | What it does | Measured runtime |
|---|---|---|
| `jupyter nbconvert --to notebook --execute notebooks/00_START_HERE.ipynb --output-dir /tmp` | runs the entry notebook without changing it | 19 s |
| `subsurfaceml predict --config config/demo.yaml --input examples/worked_example_input.json` | loads and verifies the saved models, predicts one case | 1.8 s |
| `pytest` | 72 tests: simulator, well coupling, features, grouping, artifacts, splits, upscaling | 97 s |
| `python scripts/run_pipeline.py --config config/demo.yaml` | regenerates the whole demo run: verification → data → models → figures → report | 9.3 min (recording run) |
| `python scripts/run_pipeline.py --config config/study.yaml` | regenerates the study run (optional) | 47.6 min (recording run) |
| `python scripts/make_notebooks.py --execute` | rebuilds and executes all six notebooks | 60 s |

Running a pipeline command overwrites that configuration's folder in `results/`.
Other entry points: `subsurfaceml simulate`, `subsurfaceml validate` and
`streamlit run app/streamlit_app.py -- --config config/demo.yaml`.

## Execution record

**Recorded results.** `results/study/` and `results/demo/` were produced on
2026-09-20 by `python scripts/run_pipeline.py --config config/study.yaml` (47.6
min) and `--config config/demo.yaml` (9.3 min) on a 2-core Linux x86-64 cloud
machine (`Linux-6.18.44-fc-v37-x86_64-with-glibc2.39`) with Python 3.11.15,
NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0 and XGBoost 3.2.0.
The full version list is in `requirements-lock.txt`; each run's
`metrics/summary.json` and `reports/run.log` record its configuration, versions
and timings. The commit named in those records is an internal one that predates
some code already in the working copy at run time; the reproduction checks below
were therefore run with the code as published.

**Checks from a clean copy (2026-09-27).** A copy of exactly the published files
was installed into a new virtual environment from `requirements-lock.txt`
(Python 3.11.15, same machine type), and:

* `pytest` — 72 passed;
* the entry notebook executed without error; the held-out metrics it recomputes
  from the saved models equal the recorded ones (pressure RMSE 1.2373 MPa, R²
  0.9406; plume radius 10.5104 m, R² 0.9945; sweep 0.0012, R² 0.9885);
* all six notebooks executed without error, and `subsurfaceml predict` loaded and
  hash-verified the saved models;
* four saved study simulations, including the worst pressure case and the
  reservoir where screening failed, were re-simulated: their outputs matched the
  saved values to a relative difference below 3 × 10⁻¹⁴;
* the feature table rebuilt from `scenarios.csv` matched the saved one (maximum
  difference 2 × 10⁻¹³);
* re-training the peak-pressure surrogate on the saved training reservoirs
  selected the same family (SVR) and gave a held-out RMSE of 1.2372 MPa against
  the recorded 1.2373 MPa, with the same R² of 0.9406.

The full simulation and training pipeline was not re-run for this check. The
project has not been run natively on Windows.

## Files

```
notebooks/00_START_HERE.ipynb  the walkthrough (entry point)
notebooks/supporting/          five optional notebooks (s1–s5)
src/subsurfaceml/              simulator, verification, data, features, models, screening, inference
scripts/                       run_pipeline.py · run_validation.py · build_coverage.py · make_notebooks.py
config/                        study.yaml (reference run) · demo.yaml (small run)
results/study/                 the reference run: data, figures, metrics, models (+ manifest), report, log
results/demo/                  the small run, same layout
results/historical/README.md   what the superseded run was and why it is not used
tests/                         72 tests
app/streamlit_app.py           optional interface: results, predict, simulate, screen
examples/                      input for the worked example
docs/                          TECHNICAL_REPORT · WALKTHROUGH · SOURCE_MAP · ASSUMPTIONS · MAPPING_TABLE
                               COVERAGE_MATRIX · CHANGELOG · COMPLETION_REPORT · coverage/ (frozen topic list)
```

## Author and attribution

Abdelghafar Fouda. The physics, numerical methods and machine-learning methods
follow the CHEN60482 *Advanced Subsurface Modelling* course materials (lectures by
Dr Masoud Babaei; uncertainty material by Dr Lin Ma) and the accompanying *Data
Science and Machine Learning* notebooks, as traced in
[`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md); no course material is redistributed
here. The code uses NumPy, SciPy, pandas, scikit-learn, Matplotlib, joblib, PyYAML,
XGBoost, SHAP, imbalanced-learn, scikit-optimize, umap-learn and Streamlit, all
under permissive open-source licences (BSD, MIT or Apache 2.0; Matplotlib under
its own PSF-style licence). AI assistance was used substantially in writing,
correcting, testing and documenting the code, including the corrections recorded
in `docs/CHANGELOG.md`; every number in this README is read from files in
`results/` produced by the code in this repository.

Licence: MIT (see [`LICENSE`](LICENSE)).
