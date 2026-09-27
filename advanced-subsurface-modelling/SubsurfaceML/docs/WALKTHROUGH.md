# Walkthrough — how the pipeline works and why it is built this way

This is the "explain it to someone" guide: what happens, in which file, and
the reasoning behind each choice. Equations and results are in
`TECHNICAL_REPORT.md`; sources in `SOURCE_MAP.md`.

## 0. The one-sentence version

We simulate CO₂ injection into many plausible (synthetic) layered reservoirs
with a verified IMPES simulator, learn fast surrogates of the three outputs
an operator cares about, measure honestly how well those surrogates work on
reservoirs they have never seen, and use them to screen injection schedules
that are then checked again with the simulator.

```
config/demo.yaml
     │
     ▼
validation.py ──(18 checks must pass)──► scenarios.py + dataset.py ──► scenarios.csv
                                                   │  (simulate 300 cases)
                                                   ▼
                                        features.py  (the ONE feature path)
                                                   │
         ┌──────────────────────┬─────────────────┼──────────────────┬───────────────┐
         ▼                      ▼                 ▼                  ▼               ▼
   models.py (surrogates)  classify.py      interpret.py        studies.py     uncertainty.py
         │                                                                          │
         └──────────► artifacts.py (models + manifest) ◄───────────────────────────┘
                                   │
                    predict.py (Predictor) ──► cli.py · app/streamlit_app.py · notebooks
                                   │
                         optimise.py (screen) ──► re-simulate with impes.py
```

## 1. The simulator (`grid.py`, `fluids.py`, `impes.py`)

**Geometry.** Each layer is a 1-D radial grid from the wellbore (0.15 m) to
the compartment radius. Radial Darcy flow between two radii has the exact
transmissibility `2πh k / ln(r₂/r₁)` (series radial flow, `2-Upscaling.pdf`
p.8), so cell centres are geometric means of the face radii and neighbouring
permeabilities are combined harmonically (`1-Transmissibility.pdf` p.19).
*Why:* with these choices the steady single-phase solution is exact on any
grid, which gives verification checks V1/V2 a machine-precision reference.

**Near-well block.** Cell 0 spans `[r_w, r_near]`. A fully logarithmic grid
down to the wellbore would have tiny cells near the well and the explicit
saturation update would need second-long time steps. The well equation
(`3-IMPES.pdf` p.16) carries the pressure drop from the block to the
wellbore analytically. V7 measures what this costs in accuracy.

**IMPES step** (`TwoPhaseModel._step`):

1. Evaluate mobilities with **upstream** weighting at the old time level
   (`3-IMPES.pdf` p.7–8, p.13).
2. Solve the **pressure** equation implicitly: the sum of the brine and CO₂
   equations with the compressible storage term (`3-IMPES.pdf` p.14–15;
   with `B = 1` the weighting factor `β` is 1). It is solved for the pressure
   *increment*, which avoids round-off drift.
3. **Well coupling** — the key correction of this version. Every layer's
   well block connects to the same wellbore:
   `q_l = J_l (p_bh − p_l0)`, `Σ q_l = Q`. `p_bh` is one extra unknown.
   Because everything is linear, each layer is solved twice (once for the
   right-hand side, once for a unit well source) and `p_bh` follows from the
   rate constraint — no iteration, no approximation (`_solve_pressure`).
   A completion that would *produce* (its pressure above `p_bh`) is closed,
   because an injector does not produce. At shut-in all completions close.
4. Update the CO₂ saturation **explicitly** from the CO₂ mass equation in a
   conservative form, so the CO₂ mass balance holds to round-off.
5. Accept the step only if the saturation change is small, stays in range and
   creates no new local extremum; otherwise halve it (`run`). The step also
   respects the local CFL limit — without it, near-well odd–even
   oscillations appeared (found and fixed in this version).

*Why IMPES and not fully implicit?* It is the method the course teaches, it
is transparent (a tridiagonal solve per layer), and the time-step limit it
needs is exactly what `3-IMPES.pdf` p.20 warns about — so it is verified
rather than assumed away.

**What is deliberately left out:** gravity/buoyancy, capillary pressure,
dissolution, hysteresis, vertical crossflow, geomechanics. These are scope
decisions listed in `ASSUMPTIONS.md`, not bugs.

## 2. Verification before any data (`validation.py`)

The pipeline stops if any of 18 checks fails. They compare the code with
results derived in the lectures (steady radial pressure, closed-tank
material balance, Buckley–Leverett/Welge), with conservation (CO₂ mass
balance, total well rate, common BHP), and with itself (grid, well-block and
time-step convergence; strict vs relaxed stepping). *Why first:* a
surrogate can only be as good as the data; training on an unverified
simulator would make every ML number meaningless.

## 3. Scenarios and data (`scenarios.py`, `dataset.py`)

A **realisation** is one reservoir: median permeability (log-uniform), the
Dykstra–Parsons coefficient of the layer permeabilities, porosity,
thickness, compartment radius, relative-permeability shape and end point,
CO₂ viscosity and density — drawn by plain Monte Carlo from assumed ranges
(`5-Uncertainty.pdf` p.42–45). Each realisation gets three **schedules**:
a level (a multiple of a closed-form reference rate) times a shape (front- or
back-loaded, bowed). Level and shape are sampled independently so the model
can learn the effect of each.

*Why a reference rate?* The same absolute rate is trivial in a thick
high-permeability unit and absurd in a thin tight one. Scaling by a
closed-form injectivity/storage estimate keeps the sample in a physically
sensible range. It uses no simulator output, so it is a legitimate input.

Every row records its seed; failed runs are logged; a data-quality report
checks duplicates, missing values, bounds, mass balance and the common-BHP
condition.

## 4. The one feature path (`features.py`)

`raw_inputs(cfg, realisation, rates)` → `engineer(df)` → the 29 model
inputs. The same two functions are used for the training data, for
evaluation, for the schedule screening, for the command line and for the
app. `tests/test_features.py` pushes one reservoir and schedule through all
routes and requires bit-identical inputs. *Why this matters:* the earlier
version computed one input differently at inference (fixed `krg0 = 0.4`),
so the model was asked questions in a different language from the one it
was trained in.

The engineered features are physical scales: log permeability, flow
capacity `kh`, pore volume, the rate relative to the reference rate, the
fraction of mass delivered early, the radius the injected volume would fill
(`4-CO2 BL.pdf` p.30: displaced brine volume = injected CO₂ volume).

## 5. Learning (`splits.py`, `models.py`, `uncertainty.py`)

**Grouped splits.** All rows of a reservoir stay together — train (56
reservoirs in the demo), calibration (19) and test (25). This is the
geographic-splitting idea of `Lecture08` / `E03_geographicalspliting` with
the reservoir as the "region". A random row split would let the model see
another schedule on the same reservoir; the split-comparison study measures
how optimistic that would be.

**Model choice.** Twelve families from the course (mean baseline, linear,
ridge, lasso, elastic net, KNN, SVR, decision tree, random forest, gradient
boosting, AdaBoost, XGBoost), each inside a pipeline (imputer, scaler,
model) and tuned by randomised search within grouped 4-fold CV on the
training reservoirs only. The lowest CV error wins. The test reservoirs are
used once. Pressure and plume radius are fitted on a log scale (their
drivers act multiplicatively); all errors are reported in MPa, m and
fraction.

**Error band.** The spread of residuals on the calibration reservoirs
(P5–P95) gives a band around each prediction; its coverage on the test
reservoirs is measured and reported — not promised.

## 6. Screening classifier (`classify.py`)

A yes/no "will this exceed the pressure limit?" model, because operators
often need a quick screen rather than a number. Families are compared on
grouped out-of-fold scores; class imbalance is handled inside the folds;
the threshold is set on out-of-fold scores to reach high recall (missing an
exceedance is the costly error). A green/amber/red version tests
multiclass methods. Both are compared with simply thresholding the
regression surrogate — which turns out to be a strong competitor.

## 7. Interpretation and studies (`interpret.py`, `studies.py`)

These answer "what did the model learn and how robust is the setup?" with
course methods: permutation importance on unseen reservoirs, SHAP for a tree
model, a local explanation and a what-if curve for one case near the limit,
PCA/k-means of reservoirs and of simulated plume shapes, and a set of short
experiments (how many reservoirs are enough, which scaler, what if a
descriptor is unknown, are extreme cases outliers, do correlated features
matter, how many inputs are needed, does a quadratic model help, grid vs
random vs Bayesian search, normal equation vs gradient descent, grouped vs
random split, early stopping, ensembles). None of them touches the test
reservoirs.

## 8. Screening schedules (`optimise.py`)

For each of five unseen reservoirs: draw 3000 candidate schedules from the
training design, predict pressure build-up and plume radius, keep those whose
**upper band edge** is below both stated limits, rank by injected mass
(computed exactly from the rates), and do the same for constant-rate
schedules as the baseline. Then **re-simulate** the best three and the
baseline and report what the simulator says — including violations and any
failed runs. *Why re-simulate:* the surrogate proposes; only the simulator
decides.

## 9. Using the trained models (`artifacts.py`, `predict.py`)

`save_bundle` writes each model with a SHA-256 hash, the feature list, the
package versions, seeds and the dataset hash. `Predictor(cfg)` refuses files
that are missing, modified, trained on a different feature list or saved with
a different scikit-learn version, and tells you the rebuild command. The CLI
(`subsurfaceml predict / simulate`), the app and the worked-example notebook (`notebooks/supporting/s5_worked_example.ipynb`) all go through
it.

## 10. Where to look when explaining a number

| Question | File |
|---|---|
| Did the simulator pass? | `results/demo/metrics/validation.json` |
| How good are the surrogates? | `results/demo/reports/RESULTS.md`, `summary.json → ml.targets` |
| Where do they fail? | `summary.json → ml.targets.<t>.difficult_cases`, `figures/08_errors_*.png` |
| How fast? | `summary.json → speed` |
| Did the screened schedules hold up? | `metrics/optimisation_resimulation.csv` |
| Which course topics were used, and how? | `docs/COVERAGE_MATRIX.md` |
| What was run, exactly? | `reports/run.log`, `models/manifest.json` |
