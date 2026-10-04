# Source map

Every physical model, equation, correlation, property value, network
architecture and loss term used in this project, traced to a supplied file.
"Supplied" means a file from the two sets of teaching material provided for
the work: `Models/` (the CHEN60492 *Properties of Subsurface Fluids* notes by
Dr M. Babaei, University of Manchester) and `Deep Learning/` (the Deep Learning
module notebooks and slides). Neither is redistributed in this repository;
file names, pages and notebook cells are cited so each item can be checked
against the originals.

---

## 0. The source audit that produced this map

Every PDF in `Models\` was rendered page by page at 150 dpi (`pdftoppm`) and
OCR'd (`tesseract`) — 139 pages in total, including the scanned OneNote prints
(`Part 2.pdf`, `Part 3.pdf`, `Part 4.pdf`, `4 - Ternary Diagram.pdf`,
`CO2 Storage Capacity.pdf`) whose text layer is empty. The `.docx` and the
three `.pptx` files were unpacked and their text extracted. Pages that matched
on property-table keywords were then read as images.

The word **"Accentric" / "acentric" occurs on exactly four pages of the whole
`Models\` set**: p. 4 of `3 - Two-phase flash calculation.pdf`, where the
acentric factor is defined in words, and **pp. 14–15 of the same file**, which
carry a numeric table. The same two pages appear again as scans on pp. 14–15
of `Part 3.pdf`. Nowhere else in the supplied material is any acentric factor,
critical pressure or critical temperature tabulated for a named component.

This corrects an earlier version of this project, which stated that no
acentric factor existed in the material and sampled one. It does exist. The
dataset was rebuilt on the table below and every result re-run.

The same sweep was repeated for the **composition-sampling method**. The word
"Dirichlet" appears nowhere in `Models/` and in none of the 45 supplied Deep
Learning notebooks. The random draws those notebooks do demonstrate are
uniform ones — `np.random.rand`, `np.random.randint`, `np.random.choice`,
`torch.rand`, `torch.randn`, `torch.randint`. An earlier version of this
project drew compositions from a flat Dirichlet; that has been removed and
replaced by the construction the notes themselves use,
`z_i = n_i / sum(n_j)` from integer mole charges (p. 2, and the `ni`/`zi`
columns on p. 14), with the integers drawn uniformly. The dataset was rebuilt
and every dependent result re-run again.

---

## 1. Physical inputs — the component table

`Models/3 - Two-phase flash calculation.pdf`, **p. 14**
("Example [from .../PETE310/goodies/VLE_Pete310.xls]", headed *Whitson's
K-value Correlation*) and **p. 15** ("Model: Flash vaporization calculation
using K-factor correlation"). Transcribed verbatim in `src/sfp/components.py`:

| component | `ni` | `zi` | `Pc` (psia) | `Tc` (R) | `ω` |
|---|---|---|---|---|---|
| CO2 | 5 | 0.0387597 | 1071.00 | 547.91 | 0.26670 |
| C1 | 25 | 0.1937984 | 666.40 | 343.33 | 0.01040 |
| C2 | 10 | 0.0775194 | 706.50 | 549.92 | 0.09790 |
| C3 | 15 | 0.1162791 | 539.25 | 750.04 | 0.19235 |
| C4 | 14 | 0.1085271 | 481.00 | 818.68 | 0.22523 |
| C5 | 20 | 0.1550388 | 488.60 | 845.80 | 0.25140 |
| C10 | 40 | 0.3100775 | 359.84 | 1129.20 | 0.38900 |

Units are the source's own: psia and degrees Rankine, with `T[R] = T[F] + 460`
(the sheet's own conversion — CO2 is printed as 87.91 °F = 547.91 R). Every
temperature and pressure in this project is in those units and on that
convention.

**The numbers are used exactly as printed and are not checked against,
corrected by, or supplemented from any outside source** — doing that is what
the source restriction forbids. `scripts/verify_flash.py` checks the
transcription the only legitimate way: by reproducing the two calculations the
notes themselves perform with it (§3 below).

## 1b. Physical inputs — the pressure and temperature window

The `(p, T)` states printed in `Models/3 - Two-phase flash calculation.pdf`:

| `T` | `p` | where |
|---|---|---|
| 610 R (150 °F) | 1590.8769 psia | p. 14, bubble point of the example mixture |
| 610 R (150 °F) | 1.9995691 psia | p. 14, dew point of the example mixture |
| 640 R (180 °F) | 796.43821 psia | p. 15, flash of the example mixture |
| 680 R (220 °F) | 4000 psia | p. 16, C1/nC10 example |
| 620 R (160 °F) | 1000 psia | p. 17, C1/nC10 example |

The sampling window is the span of those states: **T ∈ [610, 680] R,
p ∈ [2, 4000] psia** (`src/sfp/components.py::SUPPLIED_STATES`). No pressure
or temperature outside the supplied material is used.

## 2. Physical models and equations

| what | where used | supplied source |
|---|---|---|
| `K_i = y_i/x_i`; `z_i = x_i F_L + y_i F_V`; `F_V + F_L = 1` | definition of the predicted quantity | `Models/3 - ...`, p. 2 |
| `z_i = n_i / n` from component moles — the rule every generated composition is built with | `sfp/data.py::build_dataset` | `Models/3 - ...`, p. 2 (definition) and the `ni` → `zi` columns of the p. 14 table (5, 25, 10, 15, 14, 20, 40 moles → the printed mole fractions) |
| Wilson / Whitson K-value correlation `K_i = (p_ci/p) exp[5.37(1+ω_i)(1 − T_ci/T)]` | `sfp/flash.py::wilson_k` — the K-values of every row | `Models/3 - ...`, p. 4 (Wilson), and written out again in the green box on p. 14 |
| reduced properties `T_ri = T/T_ci`, `p_ri = p/p_ci` (corresponding states) | inside `wilson_k` | `Models/1 - Definitions and thermodynamic properties.pdf`, p. 7; `Models/3 - ...`, p. 4 |
| Rachford-Rice `h(F_V) = Σ z_i(K_i−1)/[F_V(K_i−1)+1] = 0` | `sfp/flash.py::rachford_rice`, `solve_fv`; the physics loss term | `Models/3 - ...`, p. 8 |
| physical-root test `Σ z_iK_i > 1` **and** `Σ z_i/K_i > 1` | `sfp/flash.py::is_two_phase` — selects which states enter the dataset | `Models/3 - ...`, p. 8 and the table on p. 11 |
| `x_i = z_i/[F_V(K_i−1)+1]`, `y_i = z_i K_i/[F_V(K_i−1)+1]` | `sfp/flash.py::phase_compositions` | `Models/3 - ...`, p. 7 (and the boxed formula on p. 15) |
| bubble point `F_V = 0`, `Σ z_iK_i = 1`, `p_b = Σ z_i p_ci exp[5.37(1+ω_i)(1−T_ci/T)]` | verification case A | `Models/3 - ...`, p. 12 |
| dew point `F_V = 1`, `Σ z_i/K_i = 1` | verification case A | `Models/3 - ...`, p. 13 |
| binary closed forms `F_V = (1 − z₁K₁ − z₂K₂)/[(K₁−1)(K₂−1)]` and `F_V = [z₁(K₁−K₂)/(1−K₂) − 1]/(K₁−1)` | verification case C, independent check on the bisection | `Models/3 - ...`, pp. 17–18 |
| K-values span ~10² to ~10⁻² over reservoir pressures, converging to 1 at the convergence pressure | context for the pressure window; not used numerically | `Models/3 - ...`, p. 5 (chart) |
| bubble- and dew-point pressures as the sums `Σ z_i p_ci exp[…]` and `1/Σ z_i/(p_ci exp[…])` (the `Bpi`, `Dpi` columns), so that the phase test is `p_d < p < p_b` | `sfp/flash.py::wilson_saturation_pressures`; the guarded predictor's phase labels; the window position of the error analysis (2026-10-04) | `Models/3 - ...`, pp. 12–14 |
| phase test before any prediction: liquid if `Σ z_iK_i ≤ 1`, vapour if `Σ z_i/K_i ≤ 1`, saturated at equality | `sfp/predict.py::classify_phase` (2026-10-04) | `Models/3 - ...`, p. 8 and the table on p. 11; bubble and dew points pp. 12–13 |

## 3. Verification cases — all from the supplied material

`scripts/verify_flash.py`, `results/metrics/verify_flash.json`:

* **Case A**, p. 14 — bubble and dew point of the seven-component mixture at
  150 °F, from the printed `Pc`, `Tc`, `ω`. Notes: `p_b = 1590.8769 psia`,
  `p_d = 1.9995691 psia`.
* **Case B**, p. 15 — a full flash of the same mixture at 796.43821 psia and
  180 °F. Notes: all seven `K_i`, `f_v = 0.1917145`, and all `x_i`, `y_i`.
* **Case C**, pp. 16–18 — the C1/nC10 binary at two states, with the phase
  labels and `F_V = 0.457`, `x = (0.263, 0.737)`, `y = (0.999, 0.001)`.

Measured agreement is in `README.md` §"Verification".

## 4. Deep learning

| what | where used | supplied source |
|---|---|---|
| `set_seed` (python / numpy / torch seeds, cuDNN autotuner off) | `sfp/nn.py::set_seed` | `Day01-Intro_DL_FFNs_and_Colab_morning_solutions.ipynb`, cell 21 |
| `class simpleFFN(nn.Module)` — `nn.Linear` stack built by `setattr`, read by `getattr`, activation after every layer | `sfp/nn.py::simpleFFN` | `Day01-...morning_solutions.ipynb`, cell 23 |
| standardisation fitted on the training data and applied unchanged to validation and test (`apply_standardization`) | `sfp/data.py::Standardiser` | `Day01-...morning_solutions.ipynb`, cell 29 |
| `train(model, optimizer, criterion, data_loader)` / `validate(...)`, loss accumulated as `loss * X.size(0)` then divided by `len(data_loader.dataset)` | `sfp/nn.py::train_one_epoch`, `validate` | `Day01-...morning_solutions.ipynb`, cell 35 |
| `evaluate(model, data_loader)` under `torch.no_grad()` | `sfp/nn.py::predict` | `Day01-...morning_solutions.ipynb`, cell 37 |
| `train_loop`: seed, optimiser, criterion, `DataLoader`s, epoch loop | `scripts/train.py::main` | `Day01-...morning_solutions.ipynb`, cell 41 |
| `TensorDataset` + `DataLoader`, `nn.MSELoss`, `torch.optim.Adam` | `scripts/train.py` | `Day02-Intro_to_Pytorch_afternoon_solutions.ipynb`; `Day03-Intro_to_Pytorch_morning_solutions.ipynb` |
| `nn.ReLU` | hidden activation | `Day04-CNNs(1)_morning_solutions.ipynb` |
| `nn.Sigmoid` | output activation (target is bounded, `0 < F_V < 1`) | `Day01-...morning_solutions.ipynb`, cell 23 (`activation=nn.Sigmoid`) |
| `torch.save(model.state_dict(), path)` and reload | `scripts/train.py`, `scripts/evaluate.py` | `Day03-Intro_to_Pytorch_morning_solutions.ipynb`, cells 22–26 |
| **physics-informed loss** `L = (1/N) Σ (NN(x_i) − u_i)² + (λ/M) Σ [D(NN(x_j))]²`, `D` the governing equation evaluated at the network's own output, no label in the second term | `sfp/nn.py::rachford_rice_torch` and the `lam * phys_loss` term in `train_one_epoch` | `Deep Learning/Day13-SciML_morning.pdf`, slide 37 (framing on slides 34–39) |

`D` here is Rachford-Rice — the algebraic equilibrium equation of
`Models/3 - ...` p. 8 — in place of the differential operator in the slide's
damped-oscillator and wave-equation examples.

---

## 5. What is **not** from the supplied material

Ordinary experiment settings. They are documented here as project choices, not
as anything the course prescribed. None of them is a physical property value
or a physical law. Apart from the mixture bootstrap of the 2026 revision
(marked below), none is a scientific method or a model architecture from
outside the course material.

| choice | value used | what it is |
|---|---|---|
| mole-charge range for generated mixtures | integers 1–40 per component | sets how varied the compositions are. The construction `z_i = n_i / sum(n_j)` is the notes' (p. 2, p. 14); only the range is chosen here, and the example's own charges run 5–40 |
| number of mixtures / states | 3,000 mixtures × 20 two-phase states; 600 more mixtures for the extrapolation band | sample size |
| pressure drawn log-uniformly | across the supplied window | the example mixture's own two-phase window spans three decades (2.0 to 1590.9 psia, p. 14), so a linear draw would starve the low-pressure end |
| where the training band ends | 2000 psia, with 2000–4000 psia held out | a split of the supplied window, not a physical boundary |
| train / validation / test proportions | 60 / 20 / 20, **by mixture** | standard practice |
| network width and depth | 3 × 128 hidden units | fixed once, identical for both variants |
| epochs, batch size, learning rate | 200, 64, Adam `lr = 1e-3` | fixed once, identical for both variants |
| model selection | parameters at the lowest validation **data** MSE | same criterion for both variants |
| seeds | 0, 1, 2 | reproducibility |
| `λ` in the physics loss | measured, not tuned: the value making the two loss terms equal for the training-mean predictor, from the training rows only | a scaling measurement |
| bisection instead of a library root-finder | tolerance 1e-12 | makes the replaced iteration visible |
| file layout, JSON/PNG formats, plot styling | — | presentation |

Added in the October 2026 revision (extension and re-verification), also project choices:

| choice | value used | what it is |
|---|---|---|
| training domain of the guarded predictor | 2–2000 psia and 610–680 R (the training band of the window), and each mole fraction's range in the training rows | measured from the saved training split by `scripts/derive_domain.py`; marginal checks only |
| tolerances of the guarded predictor | composition must sum to 1 within 1e-6; a phase-test sum within 1e-9 of 1 is a saturation point | input checking and boundary labelling |
| default model of the guarded predictor | the saved model with the lowest validation data MSE | a validation rule; no test data |
| window position `xi` | `ln(p/p_d)/ln(p_b/p_d)`, from the notes' own `p_b`, `p_d` | an analysis coordinate defined here |
| boundary bands, groups, worst errors | 5 % of test rows nearest each boundary; 20 equal-count groups; largest 1 % of errors | how the error is summarised |
| **mixture bootstrap** | 2,000 resamples of whole test mixtures, seed 20261004 | a standard statistical resampling, **not taken from the course material**. It is used only to describe how a comparison of two fixed models depends on which mixtures are in the evaluation set. It adds no physics and no model |
| capacity comparison | widths 3 × 64 and 3 × 192 beside the original 3 × 128; seeds 0–2; selection by mean validation MSE with a 10 % margin (`configs/capacity.json`) | a bounded check of the original width choice |
| CI tolerances | original 185 metrics and other reported metrics: relative 1e-6, absolute 1e-6; derived analysis tables (bands, groups, bootstrap): relative 1e-4, absolute 2e-6 | reproducibility checking (`scripts/compare_results.py`); the absolute values are set from measured one- and three-ULP float32 sensitivity (`results/reproducibility/ulp_sensitivity.json`) |

Two further facts belong here rather than in a footnote:

* **This is a synthetic benchmark.** The labels are the course's
  Wilson–Rachford-Rice calculation, not measurements. The notes give Wilson's
  original validity as "below 3.5 MPa (500 psia)" (p. 4) and then apply it in
  their own worked examples at 796 psia and up to 4000 psia. This project
  follows the notes, so the correlation *defines* the target and no claim
  about real vapour–liquid equilibrium follows from any number here.
* **Three of the printed `Pc`/`Tc` values look unusual next to the other
  four.** They are used exactly as printed and deliberately not "corrected",
  because correcting them would require an outside property table. Recorded in
  `docs/LIMITATIONS.md` §3.

## 6. Deliberately left out

Present in earlier, unpublished versions of this project, removed because it
could not be traced to the supplied material.

| removed | why |
|---|---|
| `data/reference/nist_co2_h2.csv`, `data/reference/co2_brine_solubility.csv` | external datasets (NIST WebBook; a published solubility paper) |
| Peng-Robinson equation of state (`src/sfp/eos.py`) | no cubic EoS appears in the supplied files; the flash notes name the EoS route (p. 6) but write no equation of state |
| the old `src/sfp/components.py` table | its own docstring described it as "standard DIPPR/NIST recommended constants" — an external property table. It is replaced by the table on pp. 14–15 above |
| Henry's law with Poynting and Setschenow salting-out (`src/sfp/brine.py`) | none of these appear in the supplied files |
| reservoir simulator, relative permeability, capillary J-function, material balance, ternary/MMP, schedule optimisation | each depends on correlations, published results or the removed property tables |
| the NumPy autodiff engine `src/sfp/minitorch` | not needed: real PyTorch runs here, and the supplied notebooks use PyTorch |
| sampled `T_c`, `p_c` and `ω` (an earlier version of *this* project) | superseded by the cited table on pp. 14–15 |
| Dirichlet composition sampling (an earlier version of *this* project) | appears in neither supplied source; replaced by `z_i = n_i / sum(n_j)` from uniform integer mole charges |
| the equation-of-state route to `K`-values | **outside the chosen scope**, not unfinished work: the notes name it (p. 6, `f_Li = f_Vi`) without writing down an equation of state, so modelling it would mean importing one |

None of it is included in this repository.
