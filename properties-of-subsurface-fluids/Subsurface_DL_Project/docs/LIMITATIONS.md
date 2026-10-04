# Assumptions, limitations and honest caveats

Read this before quoting any number from this project.

---

## 1. What this project is

A **synthetic surrogate benchmark** of the flash calculation taught in the
module. The labels are **not measurements**: they are the output of Wilson /
Whitson K-values (`Models/3 - Two-phase flash calculation.pdf`, p. 4 and the
green box on p. 14) fed into Rachford-Rice (p. 8) and solved by bisection, on
mixtures generated for this study. The question is:

> can a small feed-forward network reproduce *this* calculation, and does
> adding the Rachford-Rice equation to the loss change the result?

Nothing here is a claim that the Wilson route predicts real vapour–liquid
equilibrium well. The notes give Wilson's original validity as "below 3.5 MPa
(500 psia)" (p. 4) and then apply the correlation themselves at 796.44 psia
(p. 15) and at 1000–4000 psia (pp. 16–18). This project follows the notes and
stays inside the `(p, T)` states they print, so the correlation *defines* the
target. The supplied material contains no VLE measurements against which it
could be checked, and no external data was used, so no such check is reported.

**Scope.** Only the Wilson route is modelled. The equation-of-state route is
named in the notes (p. 6, fugacity equality `f_Li = f_Vi`) without any equation
of state, mixing rule or interaction parameter being written down, so
modelling it would mean importing one. It is **outside the scope chosen here**
— a boundary of the study, not unfinished work.

## 2. The source restriction: what is and is not satisfied

Every physical input — the seven components, their `Pc`, `Tc` and acentric
factors, and the `(p, T)` window — comes from pp. 14–15 and 16–18 of
`Models/3 - Two-phase flash calculation.pdf`. Nothing physical is invented,
substituted or imported. `docs/SOURCE_MAP.md` §0 records the page-by-page OCR
sweep that established this, and §5 lists every choice that is *not* sourced.

Compositions are built the way the notes build theirs:
`z_i = n_i / sum(n_j)` from integer mole charges (p. 2, and the `ni`/`zi`
columns of the p. 14 table). An earlier version used a flat Dirichlet draw;
"Dirichlet" appears in neither supplied source, so it was removed and the
dataset rebuilt. Mole fractions are non-negative and sum to one by
construction, and no distribution over the composition simplex is assumed.

What remains unsourced are **ordinary experiment settings**: the mole-charge
range, the number of mixtures and states, the log-uniform pressure draw, the
60/20/20 split proportions, the seeds, and the network width and depth. They
are documented as project choices in `SOURCE_MAP.md` §5 and are not presented
as anything the course prescribed. None of them is a property value or a
physical law. The October 2026 revision adds one statistical method from
outside the course material, a resampling of whole test mixtures. It is used
only to describe how a comparison of two trained models depends on which
mixtures happen to be in the evaluation set (`ERROR_ANALYSIS.md` §3), and it
adds no physics.

## 3. The component constants are used exactly as printed

Four of the seven printed `Pc`/`Tc` pairs sit where a reader familiar with
these components would expect; three do not. They are nevertheless used
**exactly as printed**, and deliberately not corrected, because correcting
them would mean importing an external property table — the thing the
restriction forbids. The consequence is stated plainly: this project
reproduces the flash calculation *as the module teaches it*, on *the module's
own numbers*. It does not claim those numbers describe real C3, C4 or C10.

The transcription itself is checked the only legitimate way, by reproducing
the two calculations the notes perform with the table — the bubble and dew
points on p. 14 and the seven-component flash on p. 15. Both reproduce to the
precision the notes print (see `README.md` §"Verification").

## 4. What the dataset contains and does not contain

* **One fixed component set.** Every mixture is the same seven components in
  different proportions, with the mole charges drawn as uniform integers in
  1–40 and normalised. Nothing here says how the surrogate behaves for a
  different component set — and because the network is never told what the
  components are, it could not transfer to one.
* Only **two-phase** states are in the dataset. States failing the module's
  own physical-root test (`Σ z_iK_i > 1` and `Σ z_i/K_i > 1`, pp. 8 and 11)
  have no vapour fraction to predict and are discarded. The network itself
  therefore never decides whether a mixture splits. The guarded prediction
  path (`src/sfp/predict.py`, 2026 revision) applies the phase test *before*
  the network and answers single-phase and saturated states without it
  (`GUARDED_PREDICTION.md`).
* The pressure-extrapolation set is **selected**: at 2000–4000 psia many
  mixtures have no two-phase state at all, so the mixtures that survive into
  it are not a random sample of the composition simplex, and their mean `F_V`
  is much lower than in training. Its scores test extrapolation in pressure
  under a shifted composition distribution — not general robustness.

## 5. Splitting

The split is **by mixture (realisation)**, 60/20/20, never by row. All
`(p, T)` states of one mixture stay together. A row-wise split would put the
same mixture at neighbouring pressures on both sides and would report a
smaller error than the model deserves. `results/metrics/dataset.json` records
`leakage_check_train_test_realisation_overlap`, which is 0, and
`data/flash_dataset.npz` carries the mixture index of every row
(`realisation`) plus the three index arrays, so a reviewer can re-check the
split without re-running anything.

Standardisation statistics are computed on the training rows only and applied
unchanged to validation, test and extrapolation
(`sfp/data.py::Standardiser`).

## 6. Model selection, the physics weight, and what ± means

* Parameters are kept at the epoch with the lowest **validation data MSE** —
  the same quantity for both variants, so the comparison is not biased by the
  selection criterion. The physics term never enters model selection.
* `λ` is **measured, not tuned**: the value that makes the data term and the
  physics term equal for the training-mean predictor, computed on the training
  split only. It is recorded in every `results/metrics/train_*.json`. No
  hyper-parameter was chosen using test data.
* Architecture (3 × 128 hidden units), epoch budget, batch size and learning
  rate were fixed once and are identical for both variants.
* **Every reported ± is the population standard deviation across the three
  seeds (0, 1, 2) of the per-seed metric**, computed by
  `numpy.std(..., ddof=0)` in `scripts/evaluate.py`. It is *not* a standard
  error, *not* a confidence interval, and *not* a spread over data
  resamplings: the dataset and the split are identical across seeds, so it
  measures only the variability of training from a different random
  initialisation and shuffling order. The reported mean is the plain arithmetic
  mean of the three per-seed values. Per-seed numbers are in
  `results/metrics/evaluation.json` under `models`, so a reviewer never has to
  take the summary on trust.
* Three seeds show whether an effect exceeds seed scatter; they are not a
  significance test. Differences smaller than the reported spread are reported
  as "no measured effect", not as a result.

## 6b. What the learning curve and the capacity comparison establish

Test error fell at every step as training mixtures were added — 225, 450, 900,
1,800 — and was still falling at the largest size. That shows **more training
mixtures reduced the error at this architecture**.

The October 2026 revision added a bounded width comparison (`CAPACITY.md`):
3 × 64, 3 × 128 and 3 × 192, three seeds each, the data-only loss, selection by
validation MSE under a rule fixed in advance. It selected 3 × 64. On the test
set 3 × 64 and 3 × 128 are indistinguishable, and 3 × 192 is slightly worse,
so a larger network offers no benefit here. It does **not** establish
anything about depth, other architectures or other training settings. Nor
does it extend the physics-loss comparison, which was run at 3 × 128 only.

## 7. Timing

The speed comparison in `results/metrics/evaluation.json` is between a
vectorised NumPy bisection to a tolerance of 1e-12 (40 iterations) and one
float32 forward pass, both over the whole test set on the same CPU, median of
15 repetitions after 3 warm-ups. It is a fair comparison of *these two
implementations on this machine*, not a general statement about flash solvers.
A production flash also runs a stability test and a successive-substitution or
Newton loop on the K-values, which this bisection does not, so the real cost
being replaced would be higher. The measurement is also sensitive to what else
is running on the box: for the final models `results/metrics/evaluation.json`
records 31.1 ms against 10.0 ms (3.1×), the executed notebook records its own
measurement, and repeated measurements during the work ranged from about 1.2×
to 3.1× with machine load. The capacity comparison's inference times (about
3, 11 and 21 ms for 3 × 64, 128 and 192) show that the forward pass itself
scales with width.

## 8. Where the numbers came from, and what was superseded

*The October 2026 revision changed none of the numbers below. Its new results
are in `results/analysis/`, `results/capacity/` and `results/guarded/`, and
`results/original_2026-09-21/` records the reproduction of the originals.*

Every number in `README.md` and in the notebook was produced by the run
recorded in `results/metrics/*.json` and `logs/`, executed on
**21 September 2026** in a cloud Linux container (2 vCPU, no GPU, Python 3.11,
PyTorch 2.14.0 on CPU). Every metrics file records `torch_version` and
`device`. `README.md` §7 records where the published copy was re-verified.

**Superseded, twice.** Earlier runs were overwritten rather than kept
alongside, because each answers the same question with different inputs and
keeping several sets invites exactly the confusion this file exists to
prevent.

1. *20 September 2026* — sampled `T_c`, `p_c` and `ω` over a declared range,
   because the acentric factors had not yet been found in the material. They
   are on pp. 14–15. The component constants became the cited table, the
   `(p, T)` window became the supplied one in psia and Rankine, and the
   network inputs became `(z₁…z₇, p, T)`.
2. *21 September 2026, first run* — compositions from a flat Dirichlet draw.
   Dirichlet appears in neither supplied source, so compositions were rebuilt
   as `z_i = n_i / sum(n_j)` from uniform integer mole charges and the whole
   workflow re-run.

Only the current numbers should be quoted. The earlier runs, their metrics
and the project's original CO2-storage version (which used external reference
data) are not part of this repository and are not comparable with anything
here.

## 9. The October 2026 revision: what its additions do not establish

* **Guarded prediction.** The training-domain checks are marginal: every
  variable is compared with its own range in the training data. Passing them
  does not show that a particular *combination* of composition, pressure and
  temperature was covered. The composition ranges are the realised training
  ranges (mole fractions up to 0.43–0.48), much narrower than the 0.004–0.87
  the generator permits. The test rows outside them had no larger error, so
  the check is conservative rather than calibrated.
* **Error analysis.** It locates the errors — next to the dew point in range,
  away from the bubble point in the extrapolation set — but does not explain
  them. Its window coordinate is built from the same Wilson K-values that
  define the labels.
* **Physics loss.** It improved the in-range test error in all three seeds.
  Where in the window that improvement came from differs by seed. It worsened
  the pressure-extrapolation error in two of three seeds. Three seeds cannot
  turn this into a probability. The mixture resampling describes only the
  dependence on the evaluation mixtures, not training-run variability.
* **Capacity.** One depth, three widths, three seeds, fixed training settings.
  The test set had already been inspected for the 3 × 128 models, so it is
  reported as an established benchmark and was not used to choose.
* **Reproducibility.** The 185 original metrics were reproduced bit for bit
  with the recorded thread count, and to 1.8e−9 relative with another.
  CI compares them within measured float32 tolerances on GitHub's CPU-only
  PyTorch build. The revision was run in a cloud Linux container, not on the
  author's Windows machine.
