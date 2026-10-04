# Walkthrough — how the pipeline works and why it is built this way

This is the "explain it to someone" guide: what happens, in which file, and
the reasoning behind each choice. Equations and results are in
`TECHNICAL_REPORT.md`; sources in `SOURCE_MAP.md`.

## 0. The one-sentence version

We simulate CO₂ injection into many plausible (synthetic) layered reservoirs
with a verified IMPES simulator, learn fast surrogates of the outputs an
operator cares about, measure how well they work on reservoirs generated
*after* every modelling choice was fixed (including reservoirs from outside
the training population), and use them only to *propose* injection schedules
that the simulator must verify before anything is recommended.

```
config/study.yaml
     │
     ▼
validation.py ──(21 checks must pass)──► scenarios.py + dataset.py ──► development data (220 reservoirs)
                                                   │
                         final_eval.py ──► fresh test reservoirs (same prior) + shift reservoirs (10-30 mD)
                                                   │
                       features.py (the ONE feature path) + rom.py (analytical ROM)
                                                   │
       ┌───────────────┬────────────────┬─────────┼───────────┬──────────────┬─────────────┐
       ▼               ▼                ▼         ▼           ▼              ▼             ▼
  models.py +     intervals.py     domain.py  classify.py  interpret.py  studies.py   numerics.py
  hybrid.py       (conformal)                                                         (dataset-case
       │               │                │                                              discretisation)
       └──► artifacts.py (models + manifest) ◄──┘
                       │
        predict.py (Predictor) ──► cli.py · app/streamlit_app.py · notebooks
                       │
        screening.py: surrogate proposes ──► impes.py verifies ──► recommendation or explicit "none"

  experiments.py (scripts/run_experiments.py): nested grouped-CV ablation on development data only
  rz.py (scripts/run_model_form.py): r-z model with gravity + crossflow, measures the model-form error
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

The pipeline stops if any of 21 checks fails. They compare the code with
results derived in the lectures (steady radial pressure, closed-tank
material balance, Buckley–Leverett/Welge), with conservation (CO₂ mass
balance, total well rate, common BHP), and with itself (grid, well-block and
time-step convergence; strict vs relaxed stepping). Since 2026-10, V12 and
V13 also check the *two-phase* simulator's bottom-hole pressure — the
quantity the surrogates learn — against the pseudo-steady-state solution of
a bounded reservoir, for one layer and for two commingled layers, and check
the analytical ROM in the same limit. *Why first:* a
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

## 5. Learning (`splits.py`, `models.py`, `rom.py`, `hybrid.py`)

**Data roles.** The 220 development reservoirs are split by whole reservoir
into training (179) and calibration (41, used only for the intervals). The
published run's 55 test reservoirs had been inspected while this revision was
developed, so they are now training data; the test reservoirs are fresh
(`final_eval.py`, seed 20261104), and a second fresh set comes from a
lower-permeability prior to test behaviour under distribution shift. The
order — decisions first, test data second — is fixed in
`docs/EVALUATION_PROTOCOL.md`.

**Why the published surrogate failed where it did.** Its inputs described a
reservoir by the *parameters of its sampling prior* (median permeability and
target Dykstra–Parsons coefficient). The simulator, however, uses four
*realised* layers drawn from that prior, and with only four draws the
realised layers can be several times tighter (or more permeable) than the
parameters suggest. The largest pressure errors were exactly those
reservoirs. The realised layers are known before simulating, so they are
legitimate inputs (`features.ROCK_FEATURES`).

**The analytical ROM.** `rom.py` treats each realised layer as a sealed tank
in pseudo-steady state connected to one common bottom-hole pressure — the
same well rule as the simulator, solved analytically in seconds for
thousands of schedules. It is exact in the single-phase limit (V12, V13) and
is already a strong predictor on its own.

**The hybrid surrogate.** `hybrid.ROMOffsetRegressor` learns only the
*correction factor* the ROM needs: `ln(Δp) = ln(Δp_ROM) + g(x)`. Every course
family is tuned for `g` exactly as before (pipeline with imputer and scaler,
randomised search inside grouped 4-fold CV on the training reservoirs,
lowest CV error wins). The ablation (`scripts/run_experiments.py`) compares
this with the published inputs, the realised-layer inputs, the ROM as an
ordinary input, the ROM alone, and a three-times-larger search, with
identical folds and budgets.

## 5a. Intervals and the applicability domain (`intervals.py`, `domain.py`)

**Intervals.** Four constructions are compared; the one used was chosen on
development data by a pre-declared rule. The reservoir-level conformal
interval takes one score per calibration reservoir (its worst schedule),
which gives a statement about whole reservoirs drawn like the calibration
reservoirs — and nothing more: it is not a guarantee for a reservoir from a
different population (tested on the shift set) or for thousands of screened
candidates.

**Domain check.** A reservoir whose descriptors fall outside the training
range (with a 2 % tolerance) or far from every training reservoir is flagged;
the screening then does not use the surrogate for it.

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

## 8. Screening schedules (`screening.py`)

The surrogate proposes; only the simulator decides. `Recommendation` cannot
hold a schedule that has not been simulated and found within both stated
limits — its constructor raises — so an attractive *prediction* can never be
presented as a feasible schedule. Each reservoir gets a simulation budget
(4 runs). The revised method screens 4000 candidates with the upper edge of
the interval, simulates the best proposal, and if it violates (or leaves
margin) scales it along its own shape with further simulator runs. If the
reservoir is out of domain, or if the interval upper edges exclude every
candidate (`upper_bound_excludes_all`: the uncertainty is too large to
decide), it falls back to a simulator search. The outcomes are explicit:
`VERIFIED_FEASIBLE`, `NO_FEASIBLE_SCHEDULE_FOUND` or
`NO_CANDIDATE_PREDICTED_FEASIBLE`, with flags saying why. The pipeline
compares five methods at the same budget, including a simulator-only
constant-rate search (the baseline) and a ROM-ranked control that isolates
the learned correction. The 9 MPa build-up limit is a modelling assumption.

## 8a. Numerical and model-form error (`numerics.py`, `rz.py`)

`numerics.py` re-simulates a stratified sample of the *dataset* cases with
the grid, the well block and the time step refined, one at a time and
together, and reports how much each target and each pressure-limit label
changes. `startup_peak_screen` checks *every* case for a peak build-up set by
a transient of the brine-filled well block (`scripts/run_experiments.py
--only peak_screen`, which also refines the flagged cases). `rz.py` is a separate 2-D radial–vertical model with buoyancy and
vertical crossflow that reduces to the layered model when both are switched
off; `scripts/run_model_form.py` uses it to measure how much the layered,
no-gravity assumption changes the targets.

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
| Did the simulator pass? | `results/study/metrics/validation.json` |
| How large is the discretisation error of the training data? | `results/study/metrics/numerics_summary.json` |
| Is any peak build-up a well-block transient? | `results/study/experiments/peak_screen.json` |
| How good are the surrogates on fresh reservoirs, and under shift? | `results/study/reports/RESULTS.md`, `summary.json → ml.targets` |
| Which change helped, by how much? | `results/study/experiments/ablation_summary.json` |
| Do the intervals cover? | `summary.json → ml.targets.<t>.designs.revised.evaluation.<set>.intervals` |
| Where do the surrogates fail? | `summary.json → ml.targets.<t>.difficult_cases`, `figures/08_errors_*.png` |
| Did the screening recommend anything unverified? | `metrics/screening_recommendations.csv`, `metrics/screening_details.json` |
| How much do gravity and crossflow matter? | `results/study/experiments/model_form_summary.json` |
| What was decided before the test data existed? | `docs/EVALUATION_PROTOCOL.md` |
| What was run, exactly? | `reports/run.log`, `models/manifest.json`, `experiments/experiments_run.json` |
