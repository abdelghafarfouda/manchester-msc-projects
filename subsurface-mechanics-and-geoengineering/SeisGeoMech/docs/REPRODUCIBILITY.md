# Reproducing the published results, the clean-install defect, and CI

`results/original_2026-09-27/` (record) · `scripts/compare_results.py` ·
`.github/workflows/seisgeomech.yml` (repository root)

## 1. The baseline, reproduced before any change

The published results date from 2026-09-27. Before any scientific change, the starting commit
`3236c15` was extracted into a clean folder and run in a fresh Python 3.11.15 virtual environment
built from `requirements-lock.txt` as it then stood:

| check | result |
|---|---|
| `python -m pytest tests -q` | **126 passed** |
| `python scripts/run_all.py` | completed; 12 tables, `summary.json`, 6 figures |
| 12 CSV tables against the committed ones | **12 of 12 byte-identical** |
| `summary.json` | **65 of 65 values identical**; only the `generated_utc` timestamp differs |
| 6 figures | byte-identical |

The reference findings were recalculated from that rerun, not copied:

| finding | recalculated | published |
|---|---|---|
| paired samples, overlap | 1,105 samples, 3,614.2–3,835.0 m (220.8 m) | 1,105, ~221 m |
| supplied Gardner bias, RMSE | −0.09321, 0.11793 g/cm³ | −0.0932, 0.1179 |
| in-sample refit RMSE | 0.06904 g/cm³ | 0.0690 |
| ρg gradient, Gardner vs measured (1-D) | 24.9733 vs 25.8873 MPa/km | 24.973 vs 25.887 |

`results/original_2026-09-27/` records the following:
* the starting commit and environments;
* SHA-256 hashes of 22 files as committed at `3236c15` (`MANIFEST_original.sha256`): the
  unmodified LAS file and 21 original result files (12 tables, `summary.json`,
  `environment.txt`, 6 figures and the notebook);
* the comparisons, in `baseline_reproduction.json`;
* the logs.

The original tables, figures and in-sample refit are kept where they were, and the revision
writes its own outputs to `results/depth_blocks/` only. The notebook is the exception: it was
extended (§4.8) and re-executed, so its manifest line no longer matches. The original is
`git show 3236c15:subsurface-mechanics-and-geoengineering/SeisGeoMech/notebooks/SeisGeoMech.ipynb`.

## 2. The clean-install defect

**Reproduced once, in a disposable environment.** Python 3.13.14, fresh virtual environment,
`pip install -r requirements.txt` as committed:

```
ModuleNotFoundError: No module named 'pkg_resources'
  bruges/__init__.py, line 20: from pkg_resources import get_distribution, DistributionNotFound
```

`run_all.py` exited with status 1. pytest gave 7 failures and 17 errors, and 102 tests passed.
`pip install .` from `pyproject.toml` failed in the same way.

**Cause.** `bruges` 0.5.4 supplies the course practical's Ricker wavelet, and it imports
`pkg_resources`, which belongs to setuptools. No requirements file listed setuptools. A
Python 3.11 virtual environment seeds setuptools itself (79.0.1 in the recorded one), which is why
the original runs worked. A Python 3.12+ environment seeds none. Checked version by version:

| setuptools | `import pkg_resources` |
|---|---|
| 79.0.1 | works (DeprecationWarning) |
| 80.10.2, 81.0.0 | works, with a warning that advises `setuptools<81` |
| 82.0.0, 82.0.1, 83.0.0, 84.0.0 | `No module named 'pkg_resources'` |

**Fix.** `setuptools<81` is listed in `requirements.txt` and `environment.yml`. It is also in
`pyproject.toml`, because `pip install .` failed too. `requirements-lock.txt` pins
`setuptools==79.0.1`, the version of the recorded environment. The course-cited `bruges` wavelet
is kept unchanged.

**After the fix.**

| environment | tests | `run_all.py` | against the recorded results |
|---|---|---|---|
| Python 3.13.14, fresh, `requirements.txt` (setuptools 80.10.2, NumPy 2.5.3) | 126 passed | ok | 9 of 12 tables byte-identical; the other three within 9.1e−13 absolute (a Merivale velocity of ~6,750 m/s) and 8.7e−16 relative; 59 of 65 summary values identical, the other six within 1.7e−12 relative (the largest on the near-zero refit bias) |
| Python 3.13.14, fresh, `requirements-lock.txt` (setuptools 79.0.1) | 126 passed | ok | 12 of 12 byte-identical, 65 of 65 identical |

The differences in the first row come from newer NumPy and SciPy releases, not from setuptools:
with the locked versions, everything is identical. The logs are in
`results/original_2026-09-27/logs/`.

## 3. CI

`.github/workflows/seisgeomech.yml` runs on every change to this folder. It has two jobs, and
both are expected to pass:

| job | environment | steps |
|---|---|---|
| `locked` | Python 3.11, `requirements-lock.txt` | tests; `run_all.py --out-dir ci`; compare `results/tables` and `results/depth_blocks` with `ci/`; execute the notebook |
| `fresh` | Python 3.13 virtual environment, `requirements.txt` only | first checks that the new environment has **no** setuptools, then installs, tests, regenerates and compares |

So the `fresh` job is the install that failed before the fix. It now passes only because
`requirements.txt` supplies setuptools.

**Comparison** (`scripts/compare_results.py`). Every CSV and JSON file in the recorded folder must
exist in the regenerated one, with the same columns, rows and keys. Integers, strings and
booleans must match exactly. Floats must agree within `1e-12 + 1e-9 × |recorded|`. The
`generated_utc` timestamp is skipped, and figures are not compared.

**Why those tolerances.** They are set from measured differences:

| where | largest absolute difference | largest relative difference |
|---|---|---|
| this container, locked environment | 0 (byte-identical) | 0 |
| this container, fresh environment | 9.1e−13 (a Merivale velocity, ~6,750 m/s) | 1.7e−12 (the in-sample refit bias, ~−9e−4) |
| GitHub, locked job (largest over runs 37242795223, 37243041054, 37243230211) | 5.6e−9 (a Gardner impedance, ~1.2e7) | 7.1e−12 (a small reflection coefficient) |
| GitHub, fresh job (largest over the same runs) | 5.6e−9 (the same impedance) | 7.1e−12 (tables); between 1.7e−12 and 6.8e−12 for the depth-block outputs, depending on the runner |

On GitHub's runners even the locked environment is not bit-identical: 7 of 13 original files
match byte for byte. The differences are last-bit rounding, which the runner's processor changes.
They are largest in absolute terms on large values, such as impedances of order 1e7, and in
relative terms on small differences of nearly equal numbers. The tolerance is more than 100 times
the largest observed relative difference. It is still far below the four significant figures at
which any result is reported.

## 4. Running the checks yourself

```bash
python -m pytest tests -q
python scripts/run_all.py --out-dir check
python scripts/compare_results.py results/tables check/tables --expect-files 13
python scripts/compare_results.py results/depth_blocks check/depth_blocks --expect-files 6
```

`run_all.py` without `--out-dir` writes into `results/` itself. All tables regenerate to the
same values, but `summary.json` receives a new `generated_utc` timestamp. The notebook
recomputes its results in a temporary folder and leaves `results/` untouched. The CI's locked
job checks this at the end, with `git diff --exit-code -- results`.
