# A neural surrogate for the two-phase flash calculation

**CHEN60492 *Properties of Subsurface Fluids* + the Deep Learning module,
University of Manchester**

A **synthetic surrogate benchmark** of the flash calculation taught in the
module: Wilson $K$-values fed into Rachford-Rice. Every label is produced by
that calculation, on mixtures generated for this study. There are no
measurements in it, and nothing here is a claim about real vapour–liquid
equilibrium.

### Summary

* **Objective.** Test whether a small feed-forward network can predict the
  vapour fraction `F_V` of the module's two-phase flash directly from
  composition, pressure and temperature for mixtures it has never seen, and
  whether adding the Rachford-Rice residual to the loss helps.
* **Method.** The taught flash (Wilson K-values and Rachford-Rice), checked
  against the notes' three worked examples, labels 60,000 two-phase states from
  3,000 generated mixtures, split by mixture into 1,800 training, 600
  validation and 600 test mixtures. A PyTorch network with 34,433 parameters is
  trained with the data loss alone and with the Rachford-Rice residual added,
  three seeds each.
* **Key result.** On the 600 held-out mixtures the test RMSE in `F_V` (mean ±
  s.d. over three seeds) is 0.00636 ± 0.00026 with the data loss and
  0.00535 ± 0.00029 with the physics term, a 15.9 % reduction. On a
  2000–4000 psia pressure-extrapolation set the physics term did not improve the
  result: 0.00835 → 0.00972, a difference about the size of the seed spread
  (± 0.00197).
* **Main limitation.** This is a synthetic benchmark: the labels come from the
  Wilson correlation, not from measurements, for one fixed set of seven
  components and two-phase states only. Three seeds are not a significance
  test, and the network is only about 1.2–3.1 times faster than the cheap 1-D
  bisection it replaces.
* **How to run.** `pip install -r requirements.txt`, then
  `jupyter notebook notebooks/flash_surrogate.ipynb`. The dataset and trained
  weights are included, so the notebook recomputes every reported metric in
  under a minute without training (see *Start here* below and §4).

### The question

> The taught calculation finds the vapour fraction `F_V` by iterating on
> Rachford-Rice. **Can a small feed-forward network predict `F_V` directly
> from composition, pressure and temperature for mixtures it has never seen,
> and does adding the Rachford-Rice equation itself to the loss function
> change the result?**

### Start here

```bash
git clone https://github.com/abdelghafarfouda/manchester-msc-projects.git
cd manchester-msc-projects/properties-of-subsurface-fluids/Subsurface_DL_Project
pip install -r requirements.txt
jupyter notebook notebooks/flash_surrogate.ipynb
```

The notebook is saved with its outputs from the run reported below, so it can
also be read directly on GitHub. It states in its header which numbers are
loaded from disk and which are recomputed while it runs.

---

## 1. Assumptions and scope

* **Ground truth is a model, not data.** The notes give Wilson's original
  validity as below 500 psia and then apply it in their own worked examples up
  to 4000 psia. This project follows the notes, so the correlation *defines*
  the target.
* **Only the Wilson route is modelled.** The equation-of-state route is named
  in the notes (p. 6) without any equation of state being written down. It is
  **outside the scope chosen here**, not unfinished work.
* **The component constants are used exactly as printed** in the module's own
  table, and are not checked against or corrected by any outside source.
* **One fixed set of seven components** in varying proportions; the network is
  never told what the components are, so nothing transfers to another set.
* **Two-phase states only.** The module's phase test is applied before the
  data is built; the surrogate does not decide whether a mixture splits.

### Where every physical input comes from

| input | value | source |
|---|---|---|
| seven components with `Pc` (psia), `Tc` (R) and **acentric factor** | CO2, C1, C2, C3, C4, C5, C10 — `src/sfp/components.py` | `Models/3 - Two-phase flash calculation.pdf`, **pp. 14–15** |
| K-value correlation | `K_i = (p_ci/p)·exp[5.37(1+ω_i)(1 − T_ci/T)]` | same file, p. 4 and the box on p. 14 |
| Rachford-Rice, phase test, phase compositions | `h(F_V) = Σ z_i(K_i−1)/[F_V(K_i−1)+1] = 0` | same file, pp. 2, 7, 8, 11 |
| composition rule | `z_i = n_i / Σ n_j` from integer mole charges | same file, p. 2, and the `ni` → `zi` columns on p. 14 |
| pressure / temperature window | 610–680 R (150–220 °F), 2–4000 psia | span of the flash states printed on pp. 14–18 |
| network, training loop, standardisation | `simpleFFN`, `train`/`validate`, `apply_standardization` | `Day01-Intro_DL_FFNs_and_Colab_morning_solutions.ipynb`, cells 21–41 |
| physics-informed loss | data MSE + `λ·mean[h(NN(x))²]` | `Day13-SciML_morning.pdf`, slide 37 |

No external dataset, property table, equation of state, scientific method or
model architecture is used.
[`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md) traces each item to a page or a
notebook cell; its **§0** records the page-by-page OCR sweep of all 139 pages
of `Models/` behind those references, and its **§5** lists the ordinary
experiment settings that are project choices — mole-charge range, sample
counts, split proportions, seeds, network width — documented as choices, not
as anything the course prescribed.
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is the full caveat list.

---

## 2. Verification against the notes' worked examples

`scripts/verify_flash.py` → `results/metrics/verify_flash.json`. Differences
are reported as measured.

| case | what the notes print | measured here |
|---|---|---|
| **A** p. 14 — bubble/dew point of the 7-component mixture at 150 °F | `p_b = 1590.8769` psia, `p_d = 1.9995691` psia | `1590.8767` psia (rel. diff **1.3e−07**), `1.9995763` psia (rel. diff **3.6e−06**); closure `Σ z_iK_i = 1.000000` at `p_b`, `Σ z_i/K_i = 1.000000` at `p_d` |
| **B** p. 15 — full flash at 796.43821 psia, 180 °F | seven `K_i`; `f_v = 0.1917145`; `x_i`, `y_i` | every `K_i` within 1e−05 of the printed 5 d.p. (one component differs by one unit in the last place); `F_V = 0.1916985`, difference **−1.6e−05**; `max\|x − x_notes\| = 7.1e−06`, `max\|y − y_notes\| = 3.9e−05` |
| **C** pp. 16–18 — C1/nC10 binary | phase labels; `F_V = 0.457`, `x = (0.263, 0.737)`, `y = (0.999, 0.001)` | labels agree; bisection root `0.4588879`, matching **both** closed forms the notes derive (pp. 17, 18) to 1e−09; `x = (0.2626, 0.7374)` |

### Case B: the cause of the 1.6e−05 difference is not established

The sheet's own `Target` cell prints a non-zero residual, `9.89E-06`, so it
stopped short of the root — consistent with a difference of this size. But the
residual measured here at the sheet's `f_v` is `−5.19e−05`, and that value is
not reproduced by any combination of the printed (5 d.p.) or recomputed `K`
with the printed (8 d.p.) or recomputed `z`. The sheet's own `y` column also
sums to 0.99996 rather than 1. Something beyond the stopping tolerance
differs; the evidence here does not identify it, so the difference is reported
rather than attributed.

### Case C: neglecting K₂ reproduces the printed 0.457

The two closed forms the notes derive (pp. 17, 18), evaluated with both
K-values, agree with the bisection at `0.4588879`. `K₂ = 0.0029` is about
0.3 % of 1. Neglecting it against 1 in those forms gives

```
(z₁K₁ − 1)/(K₁ − 1) = (0.6×3.8 − 1)/(3.8 − 1) = 1.28/2.8 = 0.4571428…
```

which to three decimals is **0.457**, the value the notes print. The full
expression gives 0.4588879, which to three decimals is 0.459, so the printed
figure is not the full expression rounded. The same neglect is also consistent
with their `x = (0.263, 0.737)`, and their `y₁ = 0.999` is `0.263 × 3.8`
recomputed from the already-rounded `x₁`, with `y₂ = 1 − 0.999`.

**The notes do not state this approximation.** It is offered as the numerical
explanation consistent with every figure they print, not as a claim about what
was done. Nothing was adjusted to force agreement: the inputs are the notes'
own `z` and `K`.
`tests/test_flash.py::test_neglecting_K2_reproduces_the_printed_0457` asserts
the arithmetic.

---

## 3. Results

**Dataset.** 60,000 two-phase states generated from **3,000 mixtures**, split
**by mixture** 60/20/20:

| partition | states | mixtures |
|---|---|---|
| train | 36,000 | 1,800 |
| validation | 12,000 | 600 |
| **test (held out)** | **12,000** | **600** |
| pressure extrapolation, 2000–4000 psia | 7,931 | 401 |

Zero mixtures are shared between train and test
(`results/metrics/dataset.json`).

**Model.** `simpleFFN`, 9 inputs → 3 × 128 hidden → 1, **34,433 parameters**,
Adam, `nn.MSELoss`, 200 epochs, three seeds per variant. `F_V` is
dimensionless, so every error below is in absolute vapour-fraction units.

| model | test RMSE | test R² | extrapolation RMSE | extrapolation R² |
|---|---|---|---|---|
| training-mean `F_V` (baseline) | 0.27869 | −0.000 | 0.56222 | −68.71 |
| FFN, data loss only | 0.00636 ± 0.00026 | 0.99948 | **0.00835 ± 0.00197** | 0.9838 |
| FFN, data + Rachford-Rice loss | **0.00535 ± 0.00029** | 0.99963 | 0.00972 ± 0.00197 | 0.9783 |

**What `mean ±` means.** The mean is the arithmetic mean over seeds 0, 1, 2 of
the per-seed metric; the ± is the **population standard deviation
(`ddof = 0`)** of those three values. It is not a standard error and not a
confidence interval: the dataset and the split are identical across seeds, so
it measures only variability from a different random initialisation and
shuffling order. Per-seed numbers are in `results/metrics/evaluation.json`
under `models`; the baseline is scored on exactly the same rows by exactly the
same function as the networks.

### Findings, both kept visible

1. **In range, the physics term improved the test result**: RMSE 0.00636 →
   0.00535, a **15.9 % reduction** on 600 held-out mixtures, about three times
   the seed-to-seed spread. The Rachford-Rice residual of the predictions
   falls with it (mean `h²` 6.0e−04 → 3.3e−04).
2. **On the reported pressure-extrapolation set, it did not**: RMSE 0.00835 →
   0.00972, **16.4 % worse**. That difference is about the same size as the
   seed spread on either variant (± 0.00197), so it is a change of sign rather
   than a firmly separated effect — but it is the measured result.
3. **More training mixtures reduced the error at every size tried**: 225 / 450
   / 900 / 1,800 training mixtures give test RMSE 0.02329 / 0.01554 / 0.01065
   / 0.00629, still falling at the largest. Only one width and depth were
   tried, so this does **not** establish that network capacity is unimportant.
4. **Speed is not the reason to build this surrogate.** Over all 12,000 test
   states on the same 2-core CPU the bisection takes ~30 ms and one forward
   pass ~10–24 ms; repeated measurements on this container span about 1.2× to
   3.1×, moving with machine load. A 1-D bisection on a scalar monotone
   equation is already cheap.

Figures in `results/figures/`, every number in `results/metrics/*.json`, every
run's record in `logs/`.

---

## 4. Setup

Python 3.10+ (3.11 used). From this folder, either:

```bash
conda env create -f environment.yml
conda activate flash-surrogate
```

or

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` and `results/metrics/environment.json` record the exact
versions the reported run used (numpy 2.4.4, torch 2.14.0, matplotlib 3.10.9).

### Starting command

From this folder (`properties-of-subsurface-fluids/Subsurface_DL_Project`):

```bash
jupyter notebook notebooks/flash_surrogate.ipynb
```

Nothing needs to be trained first: the dataset and the six model weights are in
the repository, and the notebook recomputes every reported metric from them in
under a minute. The quick checks, without the notebook:

```bash
python scripts/verify_flash.py     # solver vs the notes' three worked examples
python tests/run_tests.py          # 14 tests
python scripts/evaluate.py         # recomputes results/metrics/evaluation.json from the saved models
```

`evaluate.py` overwrites `results/metrics/evaluation.json` and the figures;
every metric except the timing reproduces exactly.

### Reproducing the whole thing from nothing

About **45 minutes on two CPU cores**, no GPU:

```bash
python scripts/verify_flash.py          # <1 s   solver vs the three worked examples
python tests/run_tests.py               # ~40 s  14 tests
python scripts/make_dataset.py          # ~60 s  rewrites data/flash_dataset.npz

# the six reported models, ~4 min each
python scripts/train.py --physics 0 --seed 0 --epochs 200 --hidden 128 128 128
python scripts/train.py --physics 1 --seed 0 --epochs 200 --hidden 128 128 128
python scripts/train.py --physics 0 --seed 1 --epochs 200 --hidden 128 128 128
python scripts/train.py --physics 1 --seed 1 --epochs 200 --hidden 128 128 128
python scripts/train.py --physics 0 --seed 2 --epochs 200 --hidden 128 128 128
python scripts/train.py --physics 1 --seed 2 --epochs 200 --hidden 128 128 128

python scripts/learning_curve.py        # ~8 min
python scripts/evaluate.py              # ~30 s  metrics + figures
python scripts/record_environment.py    # records the versions actually used
```

The commands work the same on Windows, macOS and Linux.
`configs/experiment.json` holds the same settings in one place;
`scripts/run_all.sh` runs the sequence end to end where `bash` is available,
and `build_notebook.py` regenerates the notebook from source.

---

## 5. Repository layout

```
notebooks/flash_surrogate.ipynb   the project, start here (executed, with outputs)
configs/experiment.json           the settings behind every reported number
src/sfp/components.py             the component table, transcribed from pp. 14-15
src/sfp/flash.py                  Wilson K-values, Rachford-Rice, bisection, phase test
src/sfp/data.py                   composition rule, sampling window, split-by-mixture, scaler
src/sfp/nn.py                     simpleFFN, training loop, Rachford-Rice residual in torch
scripts/verify_flash.py           solver vs the three worked examples in the notes
scripts/make_dataset.py           dataset + grouped split
scripts/train.py                  one model (--physics 0|1, --seed)
scripts/learning_curve.py         test error against number of training mixtures
scripts/evaluate.py               metrics and figures
scripts/record_environment.py     writes requirements-lock.txt + environment.json
data/flash_dataset.npz            dataset, incl. `moles`, `realisation` and the split indices
results/checkpoints/*.pt          six model weights + their fitted scaler
results/metrics/*.json            per-seed and summary metrics, dataset report, verification
tests/                            14 tests: tests/run_tests.py (or pytest)
docs/SOURCE_MAP.md                every equation and property value traced to a page or cell
docs/LIMITATIONS.md               assumptions, caveats, and what was superseded
logs/                             console output of the reported run
LICENSE                           MIT, for the code in this folder
```

The course material is **not** in this repository. It is cited by page and
notebook cell instead (see §8).

---

## 6. Limitations

Summarised here; the full list is in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

* Synthetic benchmark: no measurements, no claim about real VLE.
* The EoS route to K-values is outside the chosen scope.
* Component constants used exactly as printed, including three that look
  unusual next to the other four; correcting them would need an outside table.
* One fixed component set; two-phase states only.
* The extrapolation set is **selected** — at 2000–4000 psia many mixtures have
  no two-phase state, so its composition distribution is shifted and its mean
  `F_V` is much lower than in training.
* Three seeds indicate whether an effect exceeds seed scatter; they are not a
  significance test.
* The learning curve shows that more mixtures helped at this architecture; it
  does not separate data from capacity.

---

## 7. Where the results were produced and verified

**Reported run.** Every number in this README and in the notebook was produced
on **21 September 2026** in a cloud Linux container (2 vCPU, no GPU,
Python 3.11, PyTorch 2.14.0 on CPU). The exact package versions are in
`requirements-lock.txt` and `results/metrics/environment.json`; every metrics
file also records its `torch_version` and `device`.

**Publication check.** Before publication, a fresh copy of this folder was run
in a new virtual environment built only from `requirements.txt` (which
resolved torch 2.14.0, numpy 2.4.6, matplotlib 3.11.2), again in a cloud Linux
container:

* `scripts/verify_flash.py` — PASS on all three worked examples;
* `tests/run_tests.py` — 14 passed, 0 failed;
* `scripts/evaluate.py` — recomputed from the packaged dataset and the six
  saved models, all 185 metric values in `results/metrics/evaluation.json`
  matched the published file exactly; only the timing differs between runs;
* `notebooks/flash_surrogate.ipynb` — executed end to end, 19 of 19 code
  cells, no errors, same results table.

The project has not been run natively on Windows. The NumPy-only verification
script was additionally run in the Linux workspace of the author's Windows
desktop and gave identical output.

Earlier runs of this project were superseded twice — first when the acentric
factors were found in the teaching material, then when Dirichlet composition
sampling was replaced by the notes' own `z_i = n_i / Σ n_j` construction. Both
times the whole workflow was re-run and the metrics replaced; neither earlier
run is included here. `docs/LIMITATIONS.md` §8 records what changed.

---

## 8. Author and attribution

Abdelghafar Fouda — MSc Subsurface Energy Engineering, University of
Manchester. I developed the original project myself. AI tools were subsequently
used to help publish it on GitHub and make minor quality improvements.

The flash model, the component table and the worked examples come from the
CHEN60492 *Properties of Subsurface Fluids* notes (Dr Masoud Babaei,
University of Manchester); the notes credit their seven-component example
(pp. 14–15) to a Texas A&M PETE 310 spreadsheet. The network, training loop and
standardisation follow the Deep Learning module's PyTorch notebooks, and the
physics-informed loss follows its *Introduction to Scientific Machine
Learning* slides (Dr Ben Moseley). No teaching material is redistributed here;
[`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md) cites each item by file, page or
notebook cell so it can be checked against the originals.

NumPy, PyTorch, Matplotlib and Jupyter are used under their own open-source
licences.

Licence: MIT (see [`LICENSE`](LICENSE)).
