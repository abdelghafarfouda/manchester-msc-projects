# Evaluation protocol (fixed before the independent test sets were generated)

Written on 2026-10-04, during the revision of the project, **before** the
final-test and distribution-shift reservoirs were simulated or scored. It is
committed to the repository ahead of the results so the order can be checked
in the history.

## Why a new test set

The published run (2026-09-20) held out 55 reservoirs for testing. Those
reservoirs have since been inspected in detail — their worst cases, their
schedule-screening failure (reservoir 36) and their error patterns motivated
this revision. Scoring the revised method on them again would reward fitting
to known failures. They are therefore **re-labelled as development data**
and join the training reservoirs; fresh reservoirs take their place.

## Data roles

| Role | Reservoirs | Use |
|---|---|---|
| Development – training | 179 (the published run's 124 training + 55 former test reservoirs) | fitting, hyper-parameter search, model selection, all development experiments |
| Development – calibration | 41 (the published run's calibration reservoirs, same seed) | interval calibration only |
| **Final test** | 100 fresh reservoirs, same prior, seed 20261104, ids 10000+ | scored once, at the end |
| **Distribution shift** | 60 fresh reservoirs, seed 20261105, ids 20000+; median permeability 10–30 mD instead of 30–1000 mD, everything else unchanged | scored once, at the end |

Each reservoir carries 4 sampled schedules (same design as the training
data). Every split is by whole reservoir. The pipeline asserts that no
realisation id and no reservoir description occurs in two roles
(`final_eval.check_disjoint`, `pipeline.make_split`,
`tests/test_evaluation_design.py`).

## Decisions taken on development data only

All of these were made with `scripts/run_experiments.py` on the 220
development reservoirs (nested, reservoir-grouped cross-validation) before
any independent reservoir existed.

1. **Inputs and target form of the pressure surrogate** — rule: the learned
   variant with the lowest out-of-fold RMSE in the ablation (outer 5-fold
   grouped CV, inner 4-fold grouped CV, 20 search iterations per family,
   identical folds and seeds for every variant). Candidates: V0 published
   inputs, V1 + realised layers, V2 + ROM as an input, V3 hybrid (ROM ×
   learned factor), V5 published inputs with three times the search budget.
2. **Inputs of the plume-radius and sweep surrogates** — rule, applied to
   each target separately: V2 inputs (realised layers + ROM outputs) replace
   the published inputs only if their out-of-fold RMSE is lower.
3. **Interval construction** — rule: among `empirical`, `case_conformal`,
   `reservoir_conformal` and `adaptive_conformal`, the method with the
   smallest mean relative width whose mean whole-reservoir coverage over 200
   random calibration/evaluation splits of the out-of-fold residuals is at
   least 0.88 (nominal 0.90 minus 0.02); if none qualifies, the method with
   the highest whole-reservoir coverage. Whole-reservoir coverage is the
   criterion because the screening relies on every schedule of a reservoir
   being covered.
4. **Applicability-domain check** — fixed rule (`domain.py`): a reservoir is
   out of domain if any of its 16 descriptors lies outside the training range
   widened by 2 % of that range, or its standardised nearest-neighbour
   distance exceeds the 99th percentile of the training reservoirs'
   leave-one-out distances. The 2 % tolerance was set on development data:
   a strict range test flags 12.7 % of development reservoirs
   leave-one-out, the 2 % tolerance 1.8 %.
5. **Screening protocol** — fixed before scoring (`screening.py`,
   `pipeline.stage_screening`): budget of 4 simulator runs per reservoir and
   method (3 for the published method, as published); no schedule is
   recommended without simulator verification. Methods, on the first 20
   final-test and the first 10 shift reservoirs (by id), 4000 candidates each:
   * `simulator_constant` — constant rate, simulator-only proportional search
     started at the reference rate (the baseline every gain is measured
     against);
   * `rom_constant` — the same search started at the ROM's limiting rate;
   * `surrogate_published` — the published approach: published-approach
     surrogates with the empirical P5–P95 band, the three highest-mass
     predicted-feasible candidates simulated, no repair, no fallback;
   * `rom_shaped` — candidates ranked by the ROM pressure (no learned
     correction) and the revised plume model, best one simulated and repaired
     along its shape (control isolating the learned pressure correction);
   * `surrogate_verified` — the revised method: revised surrogates, upper edge
     of the selected interval, applicability-domain check, best proposal
     simulated and repaired along its shape, ROM-started constant-rate search
     as fallback.

The outcome of rules 1–3 is recorded in `config/study.yaml` (`ml` section)
and in `results/study/experiments/ablation_summary.json` (`decisions`):

| Rule | Outcome | Development evidence (out-of-fold, 220 reservoirs, 880 cases) |
|---|---|---|
| 1 | V3 hybrid, all inputs (`pressure_model: hybrid`, `feature_set: all`) | RMSE 1.138 (V0) · 0.844 (V1) · 0.458 (V2) · **0.267 (V3)** · 0.559 (V4, ROM alone) · 1.169 (V5) MPa |
| 2 | plume radius: V2 inputs (`feature_set_plume: all`); sweep: published inputs (`feature_set_sweep: baseline`) | plume 9.57 → 6.51 m; sweep 0.001089 → 0.001116 (worse, so not adopted) |
| 3 | `adaptive_conformal` | mean whole-reservoir coverage 0.899 at mean relative width 0.098, against 0.900 at 0.109 for `reservoir_conformal`; `case_conformal` 0.754 and `empirical` 0.720 fail the 0.88 criterion |

The per-target reading of rule 2 was fixed before the sweep result was
known (the plume result was known). The domain-check tolerance of rule 4
was set on development data before any independent reservoir existed.

## What is reported on the independent sets

For each target and each of (a) the revised design, (b) the published
approach retrained on the same reservoirs, (c) the published models as
released:

* RMSE, MAE, R², 95th-percentile absolute error, worst under- and
  over-prediction, mean signed error;
* the same within subgroups fixed in advance: median-permeability tercile
  (edges from the development reservoirs), realised-vs-prior permeability
  ratio (< 0.5, 0.5–2, > 2), and cases within 20 % of the assumed pressure
  limit;
* for the pressure target: missed exceedances and false alarms of the
  assumed 9 MPa build-up limit, using the point prediction and using the
  interval's upper edge;
* for every interval method: case coverage, whole-reservoir coverage, mean
  width, misses above and below;
* a paired, reservoir-level bootstrap (2000 resamples) of the RMSE, MAE and
  maximum-error differences between the revised and the published designs;
* screening: share of reservoirs with a simulator-verified schedule, number
  of first proposals that violated a limit when simulated, verified mass
  relative to the simulator-only constant-rate search, simulations used.

Results on the distribution-shift set are reported separately from the
in-distribution test set and are not pooled with it.

## What would count against the revision

The revision is not claimed to help unless, on the final test set, the
revised pressure surrogate's RMSE and worst under-prediction are both lower
than those of the published approach retrained on the same reservoirs, with
the bootstrap interval of the RMSE difference excluding zero. Whatever the
outcome, all numbers are reported.

## Addendum (2026-10-04, after the final evaluation): independence of the interval calibration

Written and pushed **before** the reservoirs it describes were generated.

**Finding.** The 41 calibration reservoirs are part of the 220 development
reservoirs on which rules 1–3 were applied. The surrogates and the difficulty
models were fitted without them. But their out-of-fold residuals entered the
ablation RMSEs (rules 1–2) and the 200 random calibration/evaluation splits
that chose the interval method (rule 3). The choice of score function —
inputs, target form, interval construction — therefore depended on the
outcomes of the reservoirs that later calibrated it. The split-conformal
argument needs a score function fixed independently of the calibration data,
so the original intervals carry **measured coverage only**, not a
finite-sample guarantee. The final-test and shift sets are not affected: they
were generated after every choice, so their numbers remain held-out
measurements of the procedure as it was run.

**Correction (fixed here, before the data exist).**

* 41 fresh reservoirs, the *calibration check*: development prior, seed
  20261106, ids 30000+, four schedules each from the same design; disjoint
  from every other set (asserted).
* Nothing is refitted or re-selected. The saved surrogates (fitted on the 179
  training reservoirs) and difficulty models (fitted on their out-of-fold
  residuals) are reused. Only the conformal quantile is recomputed from the
  fresh reservoirs, for each target's selected method (`adaptive_conformal`).
  The other three methods are recomputed on the same reservoirs for
  comparison only.
* Reported once on the final-test and shift sets, labelled as a subsequent
  evaluation, next to the original results, which stay unchanged: case and
  whole-reservoir coverage, mean width, and missed exceedances using the
  upper edge.
* The pipeline, its saved models and `subsurfaceml predict` keep the original
  calibration.

**What can then be claimed.** For a reservoir drawn from the development
prior, with four schedules from the same design, the recalibrated interval
covers all four schedules with probability at least 0.90. The probability is
over the draw of the calibration reservoirs and the new reservoir. For one
fixed set of 41 calibration reservoirs the coverage itself varies: it follows
Beta(38, 4), with mean 0.905, and 90 % of calibration sets give between 0.82
and 0.97. Measuring it on 100 test reservoirs adds about ±3 percentage points
of binomial noise. Nothing is claimed for the shift set, or for the thousands
of candidate schedules screened per reservoir.

### Outcome of the addendum (added after the check was run)

Run with `python scripts/run_experiments.py --config config/study.yaml
--only calibration_check` (`results/study/experiments/calibration_check.json`).
All 164 cases were simulated, and the set is disjoint from every other set.
Re-scoring the saved models reproduces the pipeline's original numbers
exactly. Selected method (`adaptive_conformal`), recalibrated on the fresh
reservoirs:

| Target | Test: cases / whole reservoirs covered (original → recalibrated) | Test mean width | Shift: whole reservoirs covered |
|---|---|---|---|
| Peak build-up | 95 % / 86 % → **97 % / 91 %** | 0.55 → 0.72 MPa | 70 % → 73 % |
| Plume radius r95 | 98 % / 97 % → 94 % / 88 % | 20.8 → 16.4 m | 78 % → 63 % |
| Swept fraction | 92 % / 78 % → 95 % / 84 % | 0.0029 → 0.0033 | 80 % → 82 % |

The recalibrated results are consistent with the stated property (one
calibration draw; Beta(38, 4) spread). The pipeline's intervals keep measured
coverage only.

**Correction to the data-roles table above.** "Interval calibration only"
was inaccurate. Together with the training reservoirs, the calibration
reservoirs also train the pressure-limit classifier and form the domain
check's reference set, the permeability-tercile edges and the prior Monte
Carlo sample. None of these uses their interval outcomes. The table is kept
as written so that the record is unchanged.
