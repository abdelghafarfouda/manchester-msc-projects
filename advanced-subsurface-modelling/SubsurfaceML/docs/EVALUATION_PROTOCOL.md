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
