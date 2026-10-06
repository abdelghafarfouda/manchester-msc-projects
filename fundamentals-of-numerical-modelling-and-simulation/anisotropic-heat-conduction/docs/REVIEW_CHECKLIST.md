# Review recommendations — status for the anisotropic heat-conduction project

**Source.** The accepted recommendations of an independent review of this
project, as set out in the task specification for the October 2026 revision
(seven items). No separate review document was used. The work was done on
2026-10-04.

**Status key.** *Completed*: done in this repository and checked.
*Remaining*: applies to this project but is not done.

| # | Recommendation | Status | Where it is addressed | Evidence |
|---|---|---|---|---|
| 1 | Reproduce the existing baseline and its six verification checks before changing anything; keep the original results identifiable | Completed | [`results/published_2026-09-26/`](../results/published_2026-09-26/) | The unmodified code passed 6 of 6 checks and reproduced all 12 numerical files exactly (largest difference 0) and the four figures byte for byte. The original outputs, their hashes and the re-run record are kept |
| 2 | Correct the heat-input interpretation: separate the boundary-flux heat input from the stored-energy increase, which includes the heated boundary nodes' half-cells; recalculate, do not hard-code | Completed | README *Heat-input accounting* and results table; [WALKTHROUGH](WALKTHROUGH.md) 3.8 and 7.1; `results/heat_accounting.csv` | Recalculated with the existing control volumes: boundary-flux heat input 507.3 vs 537.1 MJ/m (−5.5 %); half-cell energy 58.0 vs 3.2 MJ/m; stored-energy increase 565.3 vs 540.3 MJ/m (+4.6 %). The identity is checked to 5.8 × 10⁻¹² (S1). MJ/m is explained as energy per metre of out-of-plane depth |
| 3 | Separate spatial sensitivity at a fixed fine time step from temporal sensitivity on a fixed fine grid; measure the opposing effects | Completed | README *Space and time separated*; WALKTHROUGH 7.2; `run_studies.py`; `results/space_time_study.csv`, `results/coarse_grid_difference.csv`; figure 5 | Grid effect +8.23 K at Δt = 5.29 s and time-step effect −1.83 K on 13 × 12 (other order: +8.08 / −1.68 K; interaction −0.15 K). Spatial changes about first order in the grid spacing; temporal first order in Δt. Fixed step adequate by S3 (0.70 % < 1 %). V2's second- and first-order evidence is kept and limited to the smooth problem |
| 4 | Analyse temperature and boundary heat-input differences near the heated-segment ends at matching nested-grid locations, labelled as differences between grids | Completed | README *Segment ends*; WALKTHROUGH 7.3; `results/segment_end_temperatures.csv`, `results/segment_heat_input.csv`; figure 6 | Largest differences at the insulated node beyond each bottom-segment end (3.07 K, 97 × 89 vs 193 × 177, against 1.09 K elsewhere). They shrink about 1.5 times per refinement near the ends and 2 times elsewhere. Coarse grids put more heat through the end intervals. Locations are identical on all grids (S2) |
| 5 | GitHub Actions CI for this project only: run the six checks and compare numerical CSV/JSON results with justified tolerances; exclude timing, timestamps, logs and image pixels | Completed | `.github/workflows/heat-conduction.yml`; `compare_results.py`; `requirements-lock.txt` | Python 3.11 with the pinned environment; runs V1–V5 and S1–S3, then compares 19 numerical files with the recorded results (rtol 10⁻⁹, atol 10⁻⁹, justified in the script and the README). It also checks the original 2026-09-26 outputs. A local run of the same commands reproduced every value exactly |
| 6 | Correct the Matplotlib licence description against its upstream licence | Completed | README *Author and attribution* | Matplotlib is described by its own licence, the "License agreement for matplotlib versions 1.3.0 and later", which is based on the PSF licence agreement (checked against the licence file shipped with Matplotlib 3.10.9). NumPy and SciPy: BSD 3-Clause |
| 7 | After the results are verified, write a one-page Supervisor Overview, update the detailed documentation consistently, and record completion | Completed | README *Supervisor overview*, *Version history*, *Limitations*; WALKTHROUGH 3.8, 5, 6, 7, 8; SOURCE_MAP 3–4; this checklist | Written after the recorded run and the local CI check passed. The coursework is distinguished from the 2026 extension. Authorship and source attribution are stated |

## Unchanged by this revision

The governing equation, the assignment's inputs, the boundary and initial
conditions, the solver, the five verification checks and their tolerances, and
the original 2026-09-26 outputs. The only scientific sources are still the
course materials in [`SOURCE_MAP.md`](../SOURCE_MAP.md); no extrapolation or
error-bound method from outside them is introduced.

## Remaining limitations (not review items)

* The plate results are first order in the grid spacing. The segment ends are
  the measured location of the largest differences, but the local mechanism is
  not analysed.
* No error bound is given for the 193 × 177 values, and nothing is validated
  against measurements.
