# Revision checklist — Subsurface Fluids / Deep Learning (2026-10-04)

**Source.** The task specification for this revision: eight sections, given in
full, with no separate review document. Each row below maps one requirement to
where it is implemented and to the evidence that it holds.

**Status key.** *Completed*: done in this repository and checked. *Not done*:
deliberately left out, with the reason.

## 1. Scope and objective

| requirement | status | implementation | evidence |
|---|---|---|---|
| Clear, reproducible benchmark linking deep learning with subsurface-fluid calculations, on the central question (reproduce the flash for unseen mixtures; where it fails; when the RR residual helps) | Completed | README *Supervisor overview*; notebook header and §§4.2, 6, 7, 8 | the three parts are answered in the overview with measured numbers |
| Keep its identity as coursework later extended and re-verified | Completed | README first line and *Version history*; notebook header | coursework → rebuilt 2026-09-21 → extended 2026-10-04 |
| Describe the synthetic reference accurately; distinguish it from experimental or field validation | Completed | README overview and §1; `docs/LIMITATIONS.md` §1, §9 | "synthetic benchmark … nothing here is checked against measurements" |
| Change only this project, its CI workflow and its index entry | Completed | `properties-of-subsurface-fluids/Subsurface_DL_Project/`, `.github/workflows/subsurface-dl.yml`, one row of the top-level `README.md` | the merged diff touches nothing else (checked before merge) |

## 2. Reproduce and preserve the baseline

| requirement | status | implementation | evidence |
|---|---|---|---|
| Run `verify_flash.py`, `run_tests.py`, `evaluate.py` before changing behaviour | Completed | an unmodified copy of `main` at `1afdd61` | `results/original_2026-09-21/rerun_2026-10-04/*.log` |
| Verify the 14 tests and reproduce the six models' 185 metrics, timing excluded | Completed | `scripts/compare_results.py --expect-count 185` | 14 passed; 185 of 185 identical bit for bit with 2 threads, within 1.8e−9 relative with 4 (`baseline_reproduction.json`) |
| Record commit, environment, dataset/split hashes, model hashes | Completed | `results/original_2026-09-21/baseline_reproduction.json`, `MANIFEST_original.sha256` | SHA-256 of the dataset file and of every array (including the three split index arrays), of all six checkpoints and their weights, and of all 54 original files |
| Preserve the original results and distinguish them from the extension | Completed | originals left in place and never rerun in place; extension writes only to new paths | `results/original_2026-09-21/README.md`; CI compares the originals with fresh runs |
| Reuse the dataset and weights; do not retrain the six baseline models | Completed | — | no baseline model was retrained; the dataset is unchanged (hash in the record) |
| Keep the grouped split and training-only standardisation | Completed | unchanged `sfp/data.py`; all new code uses the saved split and each checkpoint's saved scaler | split overlap 0/0/0 recorded; `test_group_split_keeps_mixtures_together`, `test_standardiser_uses_training_statistics_only` |

## 3. Guarded prediction

| requirement | status | implementation | evidence |
|---|---|---|---|
| Validate composition, component order, pressure, temperature, finite values and units; reject malformed input concisely; never change units or normalise silently | Completed | `src/sfp/predict.py::validate_inputs` | 12 validation tests in `tests/test_predict.py`; an adversarial review confirmed 16 defects in the first version, all fixed and tested (`docs/GUARDED_PREDICTION.md` §5) |
| Apply the course phase test first; single-phase states get the phase result, not the network | Completed | `classify_phase`; `FlashSurrogate.predict_many` | `test_single_phase_states_use_the_phase_test_not_the_network`, `test_network_is_never_called_for_single_phase_or_boundary_states` (spy) |
| Phase-boundary and degenerate cases explicit, using the existing flash | Completed | bubble/dew point within 1e−9; indeterminate when both sums equal 1 (`sfp.flash` sums) | `test_bubble_and_dew_points_are_handled_explicitly`, `test_pure_component_at_its_vapour_pressure_is_indeterminate`, `test_phase_test_does_not_depend_on_rounding_of_the_composition` |
| Check the network's documented domain before inference | Completed | `TrainingDomain`, `configs/prediction_domain.json` | `test_every_training_row_is_inside_the_domain`, `test_lower_and_upper_domain_edges` |
| Verify the review's limits (2000 psia, 610–680 R, 0.004–0.87) against saved data and configuration; state that marginal checks do not establish joint coverage | Completed | `scripts/derive_domain.py` | pressure and temperature confirmed. Composition: 0.004–0.87 is what the configuration allows, but the training rows reach only 0.43–0.48, so the realised ranges are used (`review_limits_checked` in the domain file). The marginal-check caveat is in the module, the docs and the README |
| Unsupported two-phase inputs: labelled solver fallback by default, or unsupported status; extrapolation clearly marked | Completed | `on_unsupported = "solver" \| "status" \| "extrapolate"` | `test_out_of_domain_two_phase_states_fall_back_to_the_solver_by_default`, `test_each_route_gets_its_value_from_the_right_source`, `test_metadata_of_every_route` |
| Metadata distinguishing network, single-phase calculation and fallback | Completed | `Prediction.method`, `phase`, `status`, `in_training_domain`, `domain_violations`, `model` | `results/guarded/demo_cases.json` (14 cases, compared in CI) |
| Explain that the binary example verifies the reference but lies outside the network's domain | Completed | `InputError` message; `docs/GUARDED_PREDICTION.md` §6; `docs/VERIFICATION.md` | `test_binary_worked_example_is_solved_by_the_reference_flash_only` |
| Meaningful tests: invalid inputs, phase boundaries, out-of-domain cases | Completed | `tests/test_predict.py` (34 tests) | 48 of 48 pass locally and in CI |

## 4. Error analysis (six saved models, no training)

| requirement | status | implementation | evidence |
|---|---|---|---|
| Position in the two-phase window from the existing Wilson bubble/dew calculation; definition and direction documented | Completed | `sfp.flash.wilson_saturation_pressures`, `window_position`; `docs/ERROR_ANALYSIS.md` §1 | `ξ = ln(p/p_d)/ln(p_b/p_d)`, 0 at the dew point, 1 at the bubble point; tests check both ends and the identity `p_d < p < p_b` ⇔ two-phase |
| Error across the window, especially near the boundaries | Completed | `scripts/analyse_errors.py` | `fig6_error_across_window.png`, `window_profile.csv` |
| Data-only versus physics loss for every seed | Completed | same | `physics_vs_data_only.csv`, `fig8_physics_vs_data.png` |
| RMSE, MAE, large-error quantiles, Rachford-Rice residuals | Completed | same | `metrics_by_model.csv` (95th, 99th, 99.9th percentiles, max, mean `h²`, median and 99th-percentile \|h\|) |
| Share of squared error and of the worst errors near each boundary | Completed | same | `boundary_bands.csv`, `fig7_boundary_share.png` |
| In-range and extrapolation reported separately | Completed | same | separate tables and figure panels; bands defined once from the test set |
| Recalculate the "≈ half the squared error near the dew point" and "most improvement there" observations, not hard-coded | Completed | `docs/ERROR_ANALYSIS.md` §2–3 | dew band 45–58 % of squared error (data loss): confirmed. Improvement from the dew band 135 %, 17 %, 64 % by seed: true in two of three seeds only |
| Keep the negative result visible | Completed | README overview, §3; error analysis §3 | physics loss better in range in 3/3 seeds, worse on extrapolation in 2/3 |
| Seed std is not a confidence interval; any added uncertainty resamples whole mixtures and is explained | Completed | mixture bootstrap, 2,000 resamples | `docs/ERROR_ANALYSIS.md` §3 explains what each spread measures; the method is flagged as outside the course material in `SOURCE_MAP.md` §5 |

## 5. Capacity

| requirement | status | implementation | evidence |
|---|---|---|---|
| Compare 3 × 64, 3 × 128, 3 × 192 data-only networks, seeds 0–2, same split, preprocessing, optimiser and checkpoint rules | Completed | `configs/capacity.json`, `scripts/train.py --out-dir/--tag/--threads`, `scripts/capacity.py` | `results/capacity/` |
| Reuse the 128-unit runs when provenance matches; at most six new runs | Completed | `capacity.py provenance` | 36 of 36 settings match (`provenance_128.json`); exactly six new runs |
| Record parameters, training time, inference time, training/validation behaviour, test and extrapolation error | Completed | `capacity_results.json`, `capacity_table.csv`, `epoch_timing.json` | `docs/CAPACITY.md` §2 |
| Freeze the configuration before scoring; select on validation; call the test set an established benchmark | Completed | freeze commit `d7cd786` precedes training and scoring | `docs/CAPACITY.md` §1 |
| Keep the smaller choice if larger offers little; otherwise select by the rule and preserve the original benchmark | Completed | rule: smallest width within 10 % of the best mean validation MSE | 3 × 64 selected; 3 × 128 benchmark and its 185 metrics unchanged |
| Tie the physics-loss finding to the architecture it was tested on | Completed | `docs/CAPACITY.md` §3, README, notebook §§6, 8 | stated in each |
| No broad search; cache and reuse runs | Completed | three widths only; checkpoints committed and rescored by CI without training | — |

## 6. CI

| requirement | status | implementation | evidence |
|---|---|---|---|
| Python 3.11 and a pinned, working CPU-only PyTorch | Completed | `requirements-ci.txt` (`torch==2.14.0+cpu` from the PyTorch CPU index) | the install step asserts a `+cpu` build without CUDA; green on GitHub |
| Run flash verification and all tests | Completed | workflow steps 1–2 | — |
| Evaluate the saved models; compare the 185 metrics with `rtol=1e-6` and a justified absolute tolerance, timing excluded | Completed | `compare_results.py --expect-count 185` (`atol = 1e-6`) | `atol` measured: one-/three-ULP float32 perturbations move these metrics by at most 1.7e−7 / 4.6e−7 (`results/reproducibility/ulp_sensitivity.json`); GitHub's CPU build differed by up to 5.5e−8 |
| Check the new analysis outputs, required keys and non-numerical statuses | Completed | domain file, guarded cases, analysis tables, capacity results, all compared | any missing or extra key fails; statuses, phases, methods and labels must match exactly |
| Exclude timestamps, runtime measurements and image pixels | Completed | `timing`, `environment`, `run` subtrees skipped; PNGs never compared | — |
| No retraining in CI; verify the commands locally once; fix genuine failures | Completed | — | the commands passed locally. The first GitHub run found three medians of \|h\| outside `atol = 1e-12`; that was a real float32 effect, measured, and the tolerance justified from it, not loosened by guesswork. The local install used the PyPI PyTorch build, because this container cannot reach the PyTorch CPU index |

## 7. Presentation

| requirement | status | implementation | evidence |
|---|---|---|---|
| Supervisor Overview written last, about one page, covering every listed topic, with the deep-learning-to-subsurface link visible | Completed | README top | written after CI was green on the final results |
| Concise README; details in linked docs; notebook and outputs updated consistently | Completed | `docs/VERIFICATION.md`, `ERROR_ANALYSIS.md`, `CAPACITY.md`, `GUARDED_PREDICTION.md`; `build_notebook.py` → `notebooks/flash_surrogate.ipynb` | the notebook was executed end to end, 25 of 25 code cells, with its new sections recomputing the numbers |
| Keep the approved-source rules; EoS comparison as a separate future study | Completed | `docs/SOURCE_MAP.md`; README §1 | no new physical input; the EoS route is listed as outside scope / future study |
| No invented contribution, novelty, field accuracy or speed-up | Completed | README §9 | speed is stated as not a benefit |
| This checklist | Completed | this file | — |

## 8. Publication

Recorded in the pull request and the final report: the commit, the CI result,
and the check that the other three projects are unchanged.

## Not done, by design

* No new baseline training, no change to the dataset or the original metrics.
* No equation-of-state comparison (kept as a separate future study).
* No search beyond the three specified widths.
