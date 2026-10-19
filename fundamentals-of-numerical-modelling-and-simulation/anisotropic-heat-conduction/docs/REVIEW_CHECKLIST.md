# October 2026 revision — record and open items (anisotropic heat conduction)

What the revision of 2026-10-04 changed, where it is documented and what shows
that it holds.

| Change | Where | Evidence |
|---|---|---|
| Baseline reproduced before any change; the original results kept identifiable | [`results/published_2026-09-26/`](../results/published_2026-09-26/) | The unmodified code passed 6 of 6 checks and reproduced all 12 numerical files exactly (largest difference 0) and the four figures byte for byte |
| Boundary-flux heat input separated from the stored-energy increase, which includes the heated boundary nodes' half-cells | README *Heat-input accounting*; [WALKTHROUGH](WALKTHROUGH.md) 3.8 and 7.1; `results/heat_accounting.csv` | Heat input 507.3 vs 537.1 MJ/m (−5.5 %); half-cell energy 58.0 vs 3.2 MJ/m; stored-energy increase 565.3 vs 540.3 MJ/m (+4.6 %); the identity holds to 5.8 × 10⁻¹² (S1) |
| Spatial sensitivity at a fixed fine time step separated from temporal sensitivity on a fixed fine grid | README *Space and time separated*; WALKTHROUGH 7.2; `run_studies.py`; figure 5 | Grid effect +8.23 K and time-step effect −1.83 K on 13 × 12 (other order: +8.08 / −1.68 K; interaction −0.15 K); spatial changes about first order in the grid spacing, temporal first order in Δt |
| Temperature and heat-input differences near the heated-segment ends compared at matching nested-grid locations | README *Segment ends*; WALKTHROUGH 7.3; figure 6 | Largest differences at the insulated node beyond each bottom-segment end (3.07 K, 97 × 89 vs 193 × 177, against 1.09 K elsewhere) |
| CI for this project | `.github/workflows/heat-conduction.yml`; `compare_results.py`; `requirements-lock.txt` | Runs V1–V5 and S1–S3, then compares 19 numerical files with the recorded results (rtol and atol 10⁻⁹; a measured 10⁻⁷ for the 21 convergence ratios after the 2026-10-06 CI fix) |
| Matplotlib licence described from its upstream licence | README *Author and attribution* | Checked against the licence file shipped with Matplotlib 3.10.9 |

## Unchanged by this revision

The governing equation, the assignment's inputs, the boundary and initial
conditions, the solver, the five verification checks and their tolerances, and
the original 2026-09-26 outputs. The only scientific sources are still the
course materials in [`SOURCE_MAP.md`](../SOURCE_MAP.md); no extrapolation or
error-bound method from outside them is introduced.

## Open items

* The plate results are first order in the grid spacing. The segment ends are
  the measured location of the largest differences, but the local mechanism is
  not analysed.
* No error bound is given for the 193 × 177 values, and nothing is validated
  against measurements.
