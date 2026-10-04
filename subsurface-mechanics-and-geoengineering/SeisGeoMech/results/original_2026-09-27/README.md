# The published results of 2026-09-27, reproduced before the October 2026 revision

The original results are kept where they were published: `results/tables/` (12 CSV
tables and `summary.json`), `results/figures/` and `notebooks/SeisGeoMech.ipynb`. The
revision changes none of them. Its own outputs are written to `results/depth_blocks/`.

| file | what it records |
|---|---|
| `MANIFEST_original.sha256` | SHA-256 of the unmodified LAS file and of every original result file, as committed at the starting commit `3236c15` |
| `baseline_reproduction.json` | starting commit, environments, package versions, the comparison of each rerun with the committed results, and the reference findings recalculated from the rerun |
| `logs/` | pytest and `run_all.py` output and `pip freeze --all` for each environment below |

## Reproduction, 2026-10-04

| run | environment | tests | `run_all.py` | against the committed results |
|---|---|---|---|---|
| 1. baseline | Python 3.11.15, `requirements-lock.txt` as committed | 126 passed | ok | 12 of 12 CSV byte-identical, 65 of 65 `summary.json` values identical, 6 of 6 figures identical |
| 2. defect | Python 3.13.14, fresh, `requirements.txt` as committed | 7 failed, 17 errors | exit 1 | — |
| 3. fixed | Python 3.13.14, fresh, `requirements.txt` with `setuptools<81` | 126 passed | ok | 9 of 12 CSV byte-identical; the other three differ by at most 2.2e−15 absolute (8.7e−16 relative), from newer NumPy/SciPy releases |
| 4. fixed lock | Python 3.13.14, fresh, `requirements-lock.txt` with `setuptools==79.0.1` | 126 passed | ok | 12 of 12 byte-identical, 65 of 65 identical |

`generated_utc` in `summary.json` is a timestamp and is not compared.

**The defect.** `bruges` 0.5.4 runs `from pkg_resources import ...` on import, and
`pkg_resources` is part of setuptools. A Python 3.11 virtual environment seeds
setuptools itself (79.0.1 here), which is why the original runs worked although no
requirements file listed it. A Python 3.12+ environment seeds no setuptools, and
setuptools 82.0.0 and later no longer contain `pkg_resources`. Either way the import
fails: `ModuleNotFoundError: No module named 'pkg_resources'`. `pip install .` from
`pyproject.toml` failed the same way.

**The fix.** `setuptools<81` in `requirements.txt`, `environment.yml` and
`pyproject.toml`, and `setuptools==79.0.1` — the version of the recorded environment —
in `requirements-lock.txt`. 80.10.2 and 81.0.0 also still provide `pkg_resources`, but
they print a warning that advises pinning `setuptools<81`. The fix changes no code and no
result (runs 3 and 4).

**Reference findings, recalculated (run 1).** 1,105 paired samples over 220.8 m
(3,614.2–3,835.0 m); supplied Gardner bias −0.0932 g/cm³ and RMSE 0.1179 g/cm³; in-sample
refit RMSE 0.0690 g/cm³; ρg gradients 24.973 (Gardner) against 25.887 MPa/km (measured).
All agree with the published values.
