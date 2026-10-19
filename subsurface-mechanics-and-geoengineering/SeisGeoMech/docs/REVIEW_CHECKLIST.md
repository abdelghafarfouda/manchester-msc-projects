# October 2026 revision — record and open items (SeisGeoMech 3.1.0)

What version 3.1.0 (2026-10-04) changed, where it is implemented and what shows that it holds.
No original calculation or result changed.

| Part | Implementation | Evidence |
|---|---|---|
| Baseline reproduced before any change | fresh Python 3.11.15 environment from `requirements-lock.txt` | 126 tests passed; 12 of 12 tables byte-identical; 65 of 65 summary values and 6 of 6 figures identical (`results/original_2026-09-27/`) |
| Clean-install defect fixed | `setuptools<81` in `requirements.txt`, `environment.yml` and `pyproject.toml`; `setuptools==79.0.1` in `requirements-lock.txt` | The failure (`No module named 'pkg_resources'`) was reproduced first. After the fix the tables agree within 9.1e−13 absolute, and the lock reproduces them byte for byte on Python 3.13 ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §2) |
| Two-direction depth-block test, frozen before scoring | `configs/depth_blocks.json` (frozen in `193eb36`); `src/seisgeomech/depth_blocks.py`; [`DEPTH_BLOCK_TEST.md`](DEPTH_BLOCK_TEST.md) | Scores first computed after the freeze (`3bea7e9`). Held-out RMSE 0.088 and 0.061 g/cm³, against 0.113 and 0.129 for the supplied relation on the same samples |
| No tuning of the gap or split on held-out scores | One midpoint, one gap and two directions, fixed in the frozen configuration | No other split is scored anywhere |
| Tests | `tests/test_depth_blocks.py` (40 tests), `tests/test_compare_results.py` (18 tests) | 184 tests pass |
| Project CI with two jobs | `.github/workflows/seisgeomech.yml`; `scripts/compare_results.py` (rtol 1e−9, atol 1e−12) | A locked Python 3.11 job and a fresh Python 3.13 job; the tolerance is set from measured differences ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §3) |

## Not done, by design

* No trajectory, TVD conversion or vertical-stress claim.
* No second well and no other correlation.
* No hole-condition filtering.
* No change to the original calculations or their results.
* No confidence interval for the depth-block result. Two directions are reported as two numbers.
