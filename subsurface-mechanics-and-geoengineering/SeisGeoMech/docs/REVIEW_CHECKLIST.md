# Revision checklist — SeisGeoMech (2026-10-04)

**Source.** The task specification for this revision. No separate review document was
supplied. The specification as received ended partway through section 4, at "Supplied-Gardner
results on the same evaluation b…". The rows for the rest of section 4 and for CI, presentation and
publication follow the same pattern as the other projects in this repository, and are marked
*inferred*.

**Status key.** *Completed*: done in this repository and checked. *Not done*: deliberately left
out, with the reason.

## 1. Objective and scope

| requirement | status | implementation | evidence |
|---|---|---|---|
| Strengthen the question (Gardner over the measured interval; does a locally fitted relation predict a separate interval; what the density errors imply for the qualified ρg calculations) | Completed | README title and supervisor overview; notebook §1 and §4.8 | each part is answered with measured numbers in the overview |
| Keep the seismic–geomechanics link immediately visible | Completed | README overview, first paragraph ("Why seismic and geomechanics meet here") | the same `rho` feeds the synthetic seismogram and `d(sigma_zz)/dz = rho g` |
| Preserve the identity as coursework later extended and re-verified | Completed | README opening line and *Version history*; notebook §1 | coursework → 3.0.0 published (2026-09-27) → 3.1.0 extended and re-verified (2026-10-04) |
| Keep measured log evidence, synthetic seismic calculations and worked-example verification distinct | Completed | README overview, "Three kinds of evidence, kept apart"; *Data* | unchanged separation in code (`analysis` stages 2–7) |
| One focused branch from current `main`; change only SeisGeoMech, its CI workflow and its index row | Completed | branch from `3236c15`; `subsurface-mechanics-and-geoengineering/SeisGeoMech/`, `.github/workflows/seisgeomech.yml`, one row of the top-level `README.md` | diff checked before merge; other projects and workflows unchanged |

## 2. Reproduce and preserve the baseline

| requirement | status | implementation | evidence |
|---|---|---|---|
| Run the 126 tests and `run_all.py` in a clean, working environment | Completed | fresh Python 3.11.15 environment from `requirements-lock.txt`, on an untouched copy of `3236c15` | `results/original_2026-09-27/logs/1_baseline_*`; 126 passed |
| Reproduce the 12 CSV tables and the numerical content of `summary.json` | Completed | byte comparison and value-by-value comparison | 12 of 12 byte-identical; 65 of 65 values identical (timestamp excluded); 6 of 6 figures identical |
| Record starting commit, environment, hashes of the raw LAS and the original results | Completed | `results/original_2026-09-27/baseline_reproduction.json`, `MANIFEST_original.sha256` | 22 files hashed, LAS included; pip freeze of every environment |
| Preserve the unmodified LAS, original calculations and in-sample refit | Completed | `analysis.py` unchanged; original tables and figures left in place; the depth-block test writes only to `results/depth_blocks/` | the manifest; CI compares the original tables on every change. The notebook was extended, so its manifest line is expected to differ |
| Verify the reference findings by recalculation | Completed | values read from the rerun's own `summary.json` | 1,105 samples over 220.8 m; bias −0.0932, RMSE 0.1179; refit RMSE 0.0690; 24.973 vs 25.887 MPa/km (`reference_findings_check`) |
| Keep baseline and extension results identifiable | Completed | `results/tables/`, `results/figures/` (2026-09-27); `results/depth_blocks/` (2026); version 3.0.0 → 3.1.0 | README *Headline numbers* labels the 2026 rows |

## 3. Clean-install defect

| requirement | status | implementation | evidence |
|---|---|---|---|
| Reproduce the failure once in a disposable clean environment | Completed | fresh Python 3.13.14 venv, `requirements.txt` as committed | `ModuleNotFoundError: No module named 'pkg_resources'`; `run_all.py` exit 1; 7 failed, 17 errors (`logs/2_defect_*`) |
| `setuptools<81` in `requirements.txt` and `environment.yml` | Completed | both files | — |
| An exact compatible setuptools version in `requirements-lock.txt` | Completed | `setuptools==79.0.1`, the version the recorded environment had | the lock reproduces every result byte for byte on Python 3.13 (`logs/4_*`) |
| Explain the dependency briefly in the README | Completed | README *Reproducing the results*, "Why `setuptools<81` is listed" | — |
| Keep the course-cited `bruges` wavelet | Completed | `seismic.ricker_wavelet` unchanged | — |
| Verify the fix preserves the original numbers; record before and after | Completed | `docs/REPRODUCIBILITY.md` §2; `baseline_reproduction.json` runs 2–4 | after: 126 passed; tables within 9.1e−13 absolute and 8.7e−16 relative (newer NumPy), identical with the lock |
| Both final CI jobs pass; no intentionally failing job | Completed | `locked` and `fresh` jobs; `fresh` first proves the new venv has no setuptools | both green on GitHub |
| (beyond the listed files) | Completed | `setuptools<81` also in `pyproject.toml` | `pip install .` failed the same way before the fix |

## 4. Depth-block prediction test

| requirement | status | implementation | evidence |
|---|---|---|---|
| Correct "no out-of-sample test is possible": within-well test possible, another well unavailable | Completed | README *Scope* and *Limitations*; `SOURCE_MAP.md` §7.8; notebook §4.1, §4.8, §7 | wording guard in `tests/test_depth_blocks.py`, over the package module, script, configuration, README, SOURCE_MAP, `docs/` and the notebook's text |
| Sort the valid paired samples by the logged coordinate | Completed | `depth_blocks.paired_samples` (same selection as `stage_load`) | `test_paired_samples_are_the_original_overlap` |
| Two contiguous blocks, ~20 m exclusion gap near the midpoint | Completed | midpoint of the coordinate range; samples within 10 m excluded | A 502, excluded 100, B 503; last A to first B 20.2 m (`split.json`) |
| Fit on A, evaluate on B; reverse | Completed | `score_direction` | `depth_block_results.json` |
| Compare with the fixed supplied relation on exactly the same samples | Completed | the same evaluation rows; identifiers checked against the frozen ones | `test_both_relations_are_scored_on_the_same_samples` |
| Original fitting method, units and functional form; no searches, no new correlations | Completed | `fit_gardner_form` = `np.polyfit(log Vp, log RHOB, 1)` | bit-identical to the original refit on all samples (`test_fit_reproduces_the_original_in_sample_refit_exactly`) |
| Freeze and commit the split configuration before scoring | Completed | `configs/depth_blocks.json` frozen in `193eb36` (after the work-in-progress `251ae90`); scores first computed after it and committed in `3bea7e9` | `score` refuses to run unless the configuration is frozen and the derived split matches it; a test pins the configuration's hash |
| Record exact boundaries, exclusions, counts and identifiers or hashes | Completed | `split.json` (boundaries, gap edges with signed distances, LAS row ranges, per-block SHA-256 of `las_row:dept_ft`), `split_samples.csv` | — |
| Do not optimise the gap or split | Completed | one midpoint, one gap, two directions, fixed in the configuration | no other split is scored anywhere |
| 20 m gap as a design choice, not independence; logged coordinate | Completed | configuration, docstrings, `docs/DEPTH_BLOCK_TEST.md` §1, README | residual correlation at 20 m reported as context (0.08 and 0.10) |
| Report training and evaluation depth ranges and velocity ranges | Completed | `training` / `evaluation` blocks of each direction | `docs/DEPTH_BLOCK_TEST.md` §1 |
| Report fitted coefficients for each direction | Completed | `0.2362 Vp^0.2875` (A), `0.1859 Vp^0.3143` (B) | — |
| Report held-out bias, RMSE and MAE | Completed | `depth_block_table.csv` | A → B: +0.036, 0.088, 0.069; B → A: −0.035, 0.061, 0.050 g/cm³ |
| Report the supplied relation on the same evaluation blocks | Completed | same table | A → B: −0.076, 0.113, 0.092; B → A: −0.119, 0.129, 0.120 g/cm³ |
| *(inferred)* What the density errors imply for the qualified ρg calculation | Completed | mean ρg gradient over each evaluation block, all three densities, 1-D qualification kept | block fit +0.35 / −0.34 MPa/km; supplied −0.75 / −1.17 MPa/km (`docs/DEPTH_BLOCK_TEST.md` §3) |
| *(inferred)* A reading rule fixed in advance; negative and mixed outcomes reported | Completed | `reading_rule` in the frozen configuration | both directions "block fit better"; the opposite-sign held-out biases and differing coefficients are reported as limits |
| *(inferred)* Tests | Completed | `tests/test_depth_blocks.py` (40 tests), `tests/test_compare_results.py` (18) | 184 passed locally and in both CI jobs |

## 5–8. CI, presentation and publication *(inferred from the other projects)*

| requirement | status | implementation | evidence |
|---|---|---|---|
| Project CI: tests, regenerate every result, compare with the recorded results with a justified tolerance, no timestamps or pixels | Completed | `.github/workflows/seisgeomech.yml`; `scripts/compare_results.py` (rtol 1e−9, atol 1e−12) | tolerance set from measured differences, including GitHub's (`docs/REPRODUCIBILITY.md` §3) |
| Commands verified locally once | Completed | the CI commands run in both the locked and the fresh environment here | PASS in both |
| Supervisor overview written last, about one page, at the top of the README | Completed | README | written after the results and CI were final |
| Concise README; details in linked documents; notebook updated and executed | Completed | `docs/DEPTH_BLOCK_TEST.md`, `docs/REPRODUCIBILITY.md`; notebook §4.8 and updated §1, §6–§8 | executed end to end, 22 code cells, no errors; also executed in the `locked` CI job |
| Source rule kept; new design choices recorded | Completed | `SOURCE_MAP.md` §4 rows for the split, metrics, reading rule and lag correlation | no new physical input |
| No invented contribution, generality or field claims | Completed | README *Licence and attribution* | — |
| Repository index row | Completed | top-level `README.md`, SeisGeoMech row | — |
| This checklist | Completed | this file | — |
| Commit, focused PR, merge after checks; verify `main`; other projects unchanged | Recorded in the pull request and the final report | — | — |

## Not done, by design

* No trajectory, TVD conversion or vertical-stress claim.
* No second well and no other correlation.
* No hole-condition filtering.
* No change to the original calculations or their results.
* No confidence interval for the depth-block result. Two directions are reported as two numbers.
