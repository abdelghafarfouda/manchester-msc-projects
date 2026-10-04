# Original results of 2026-09-26 (baseline)

This folder keeps the outputs of the project as first published, so that they
stay identifiable after the October 2026 revision.

* **What.** The numerical outputs of `python run_project.py` recorded on
  2026-09-26 (Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.9),
  copied unchanged from commit `ea480d8`, with their `run_log.txt`,
  `environment.txt` and `summary.json`. The four figures are not copied: they are
  identical to `results/fig1`–`fig4` (byte for byte), and
  `SHA256SUMS_original.txt` lists the hashes of all 18 original files.
* **Baseline re-run.** Before any change, the unmodified code on `main` (commit
  `68e532f`) was re-run on 2026-10-04 with Python 3.11.15 and the same NumPy,
  SciPy and Matplotlib versions. It passed 6 of 6 checks, and
  `compare_results.py` found all 12 numerical files identical to these, with
  57,066 numbers and 22,188 text fields compared and a largest difference of 0.
  The four figures were also identical byte for byte. The `.npz` file differed
  only in its zip timestamps, not in its arrays. The records are in
  [`rerun_2026-10-04/`](rerun_2026-10-04/): `run_log.txt`, `environment.txt` and
  `comparison.json`.
* **After the revision.** The revised `run_project.py` writes the same 12 files.
  The CI workflow checks on every change that they still match this folder:
  `python compare_results.py results/published_2026-09-26 <new results> --subset`.

The revision does not change any of these numbers or their definitions. It adds
new quantities and studies (`run_studies.py`), and it corrects how the heat input
is interpreted in the documentation (README, *Heat-input accounting*).
