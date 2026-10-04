# SOURCE_MAP

Every scientific statement in this project traces to one of two supplied folders:

* `Models/` — EART35102 *Hydrogeology and Geomechanics* notes, exercises and worked solutions
* `Seismic/` — seismic imaging practical notebooks and the LAS file they use

Nothing else is a source. Earlier versions of this project, previous review documents and any
prior output are implementation history, not scientific sources, and none of their content is
carried forward unless it is re-derived from the two folders below.

Page numbers are the printed page footers of the PDFs. Cell numbers are positional indices of
cells in the supplied `.ipynb` files.

---

## 1. Equations

| # | Relation | Used in | Exact source |
|---|---|---|---|
| E1 | `sigma = E eps`; isotropic Hooke's law; stress and strain as 2nd-rank tensors | framing | `Models/1_Elasticity.pdf` s.1, s.3–s.4, pp.1–3 |
| E2 | Conversion table between `K, E, lambda, G, nu` — the `(E, nu)` and `(K, G)` rows | `elasticity.bulk_modulus_from_E_nu`, `shear_modulus_from_E_nu`, `youngs_modulus_from_K_G`, `poissons_ratio_from_K_G` | `Models/1_Elasticity.pdf` s.7, table on p.10 |
| E3 | `E = 2G(1 + nu)` | verification identity | `Models/1_Elasticity.pdf` s.7, p.9 |
| E4 | Voigt / Reuss averages: `K_V=(A+2B)/3`, `G_V=(A−B+3C)/5`, `K_R=1/(3a+6b)`, `G_R=5/(4a−4b+3c)`, with `A,B,C` from `c_ij` and `a,b,c` from `s_ij` | `elasticity.voigt_reuss_hill` | `Models/1_Elasticity.pdf` s.8, table on p.11 |
| E5 | Voigt–Reuss–Hill average is the mean of the two bounds | `VRHResult.K_hill`, `G_hill` | `Models/1_Elasticity.pdf` s.8, p.11 ("an average of the Voigt and the Ruess … is called the Voigt-Ruess-Hill average") |
| E6 | `S = C^-1` (compliance is the inverse of stiffness) | `elasticity.voigt_reuss_hill` | `Models/1_Elasticity.pdf` s.6, p.6 |
| E7 | `Vp = sqrt((K + 4G/3)/rho)` | `elasticity.vp_from_K_G_rho`, `p_wave_modulus_from_vp_rho` | `Models/1_Elasticity.pdf` s.10, p.12 |
| E8 | `Vs = sqrt(G/rho)` | `elasticity.vs_from_G_rho` | `Models/1_Elasticity.pdf` s.10, p.13 |
| E9 | `d(sigma_zz)/dz = rho g` | `stress.rho_g_integral`, `mean_rho_g_gradient` | `Models/Elasticity_exercise.pdf` Q5, p.2 |
| E10 | `sigma_xx = nu/(1 − nu) · sigma_zz` under no lateral strain | `stress.horizontal_over_vertical_ratio` | `Models/Elasticity_exercise.pdf` Q5 + full derivation in `Models/Elasticity Exercise Solutions.pdf`, sheet "1 of 2" |
| E11 | `eps_zz = (sigma_zz/E)(1 − 2nu²/(1 − nu))`; hydrostatic only if `nu = 1/2` | `stress.vertical_strain` | `Models/Elasticity Exercise Solutions.pdf`, sheet "2 of 2" |
| E12 | `(sigma_ij)_eff = sigma_ij − alpha P_fluid delta_ij`, `alpha ≈ 1` for rock strength | `stress.effective_stress` — **implemented, never called** | `Models/2_Hard_rocks.pdf` s.8, p.7 |
| E13 | Six stages of a compression test; stage III (crack propagation) begins "at about half the ultimate failure strength" | criterion for the Merivale linear-elastic window | `Models/2_Hard_rocks.pdf` s.3, p.2 |
| S1 | Gardner: `rho = 0.31 Vp^0.25` | `seismic.gardner_density_gcc` | `Seismic/Ex1.ipynb` markdown cell 35 |
| S2 | Acoustic impedance `Z = rho · Vp` | `seismic.acoustic_impedance` | `Seismic/Ex1.ipynb` s.1.3, markdown cell 10 and code cell 8 |
| S3 | `R = (rho2 V2 − rho1 V1)/(rho2 V2 + rho1 V1)` at normal incidence | `seismic.reflection_coefficients` | `Seismic/Ex1.ipynb` s.1.4, markdown cell 14 and code cell 15 |
| S4 | Ricker wavelet via `bruges.filters.ricker(duration, dt, f)` | `seismic.ricker_wavelet` | `Seismic/Ex1.ipynb` s.1.6, markdown cell 21 and code cell 23 |
| S5 | Synthetic trace by `np.convolve(rc, wavelet, mode='same')` | `seismic.convolve_trace` | `Seismic/Ex1.ipynb` code cell 25 |
| S6 | Wavelet-frequency comparison posed as an experiment: change the wavelet frequency and compare the trace with the reflectivity series | `seismic.resolution_sweep` | `Seismic/Ex1.ipynb` learning objective 3 and markdown cells 28, 30 |
| S7 | LAS files are read with `lasio` | `las_io.read_las` | `Seismic/Unsupervised-las_answers.ipynb` cells 29, 33, 37 |

---

## 2. Data

| Dataset | What is used | Source |
|---|---|---|
| Well 48/10b-9 log | `DEPT` (ft), `DT` (µs/ft), `RHOB` (g/cm³); `GR`, `NPHI`, `CALI`, `DRHO`, `PEF` read for context only | `Seismic/48_10b-_9_jwl_JWL_FILE_1682139.las`, copied verbatim to `data/raw/` |
| Olivine stiffness tensor | 6×6 `C_ij` in GPa; `rho = 3355 kg/m³` | `Models/Elasticity_exercise.pdf` Q3, p.2 |
| Merivale granite compression record | 16 rows of axial stress (MPa), axial strain, circumferential strain, volumetric strain; cylinder 15 mm × 40 mm, mass 18.73 g | `Models/Elasticity_exercise.pdf` Q4, p.2 |

No other dataset is read, imported or generated.

---

## 3. Constants and unit factors

| Symbol | Value | Class | Justification |
|---|---|---|---|
| foot | 0.3048 m | exact unit definition | required to read `DEPT.F` and `DT.US/F` |
| g | 9.80665 m s⁻² | defined SI constant | E9 writes `g` as "the gravitational acceleration" and gives no number. A universal constant, not a rock property or correlation. |
| g/cm³ → kg/m³ | 1000 | exact | LAS `RHOB.G/C3` |
| µs → s | 1e-6 | exact | LAS `DT.US/F` |
| Gardner coefficient | 0.31 | supplied | S1 |
| Gardner exponent | 0.25 | supplied | S1 |

---

## 4. Choices that are not scientific assumptions

These are experimental design or presentation choices. Each is stated in the code and in the
notebook, and none introduces a rock property, a correlation or a physical parameter range.

| Choice | Value | Why it is not a scientific assumption |
|---|---|---|
| Frequency sweep values | 25–300 Hz, nine values | S6 asks the student to "try different frequencies". The lower bound is imposed by the data, not chosen: the overlap is 97.5 ms of travel time along the logged path, so a Ricker below ~20.5 Hz is longer than the whole trace (`seismic.min_supportable_frequency`). None of these values is claimed to represent a field survey. |
| Ricker duration | `2/f` | A Ricker is effectively zero beyond ±1/f, so this contains the wavelet without truncating it. Every sweep result is checked against the trace length. |
| Regular time sampling | 0.1 ms | Convolution needs a regular axis. Coarser than the log's native ~0.086 ms time step, so nothing is interpolated up. |
| Coordinate→time interpolation | linear | Uses only the time–coordinate pairs derived from `DT` itself. |
| Merivale fit window | above the first loading step, below half the peak stress | The half-peak criterion is E13, taken from the supplied notes rather than chosen by eye. Sensitivity to the window is reported in `results/tables/merivale_window_sensitivity.csv`. |
| Poisson's ratio sweep | 0.05–0.45 | A plain sweep of the admissible range for an isotropic solid. **No value is claimed for this well.** |
| Gardner refit | least squares on `log rho` vs `log Vp` over the 1105 measured points | Fits the *supplied functional form* to the *supplied data*, changing no equation, and is not used anywhere else in the workflow. It is an **in-sample calibration**: no samples were held out; the coefficients were fitted and evaluated on the same 1,105 paired samples, so its 41.5 % RMSE reduction describes those points. It is not a prediction test and not evidence that the functional form is correct. |
| Gardner refit, as used in the depth-block test (October 2026) | the same least squares, on one depth block only | The original method, units and functional form, fitted on block A and evaluated on block B, then the reverse (`configs/depth_blocks.json`, `src/seisgeomech/depth_blocks.py`). No other density–velocity relation is introduced. The original in-sample refit is unchanged. |
| Depth-block split | midpoint of the overlap's logged-coordinate range; samples within 10 m of it (a 20 m gap) excluded; A above, B below | A design choice, frozen and committed before any held-out score (`configs/depth_blocks.json`, `results/depth_blocks/split.json`). The ~20 m gap follows an earlier review's estimate of residual correlation along the log, which is implementation history, not a source; it does **not** make the blocks independent. It is measured along the logged coordinate, whose vertical convention is unresolved (§5). |
| Held-out metrics and reading rule | bias, RMSE, MAE; reductions `100 (supplied − block fit)/supplied`; better/worse only if RMSE and MAE agree, each direction read separately | Summary statistics of residuals, fixed with the split before scoring. They are not physical parameters. |
| Residual lag correlation | Pearson correlation of the residual series with itself shifted by 0.2–40 m | Context for the gap only. It is not a test of independence and not a decorrelation length. It was computed after the gap was fixed and played no part in choosing it. |

---

## 5. The depth coordinate, and what follows from it

### 5.1 What the supplied files say

The LAS depth curve is labelled `DEPT`. That label alone does not establish true vertical depth,
and nothing else in either folder does either. `las_io.depth_convention_evidence()` runs the check
and `results/tables/depth_convention_evidence.csv` records its output:

| field / curve | meaning | value |
|---|---|---|
| `LMF` | logs measured from | `UNKNOWN` |
| `EKB`, `EDF`, `EPD`, `EGL`, `ELZ` | elevation references | all `.0` |
| `WDMS` | water depth | `.0 M` |
| `TVD`, `TVDSS`, `DEVI`, `INCL`, `AZIM` | vertical-depth or trajectory curves | **all absent** |

The only statement about depth convention anywhere in the supplied material is in
`Seismic/Unsupervised-las_answers.ipynb` (markdown cell beginning "Well log plots are a common
visualization tool..."), which says logs are displayed against "measure depth **or** true vertical
depth". It names the distinction and does not resolve it for this file.

No external trajectory data are obtained, no inclination is invented, and measured depth is not
silently assumed to equal vertical depth.

### 5.2 What the project therefore claims

* The ρg integration is an **explicitly one-dimensional application** of E9 to the logged
  coordinate. The absolute gradient is reported as such and is **not** presented as a verified
  field stress profile.
* The functions are named for what they compute — `rho_g_integral`, `mean_rho_g_gradient` — not for
  a vertical stress they cannot be shown to represent. A test enforces that no function in
  `stress.py` is named `overburden` or `sigma_v`.
* The **comparison** between the measured-density and Gardner-density gradients is **invariant
  under uniform scaling of the logged coordinate**: both are integrated over the same array, so one
  constant factor applied to every interval cancels from their ratio. This is verified numerically
  — a uniform rescaling leaves the ratio unchanged to 1.3e-15 — and it is why the −3.53 %
  difference, not the absolute 25.887 MPa/km, is the project's headline result.
* That invariance is **not** shown to extend to a depth-dependent trajectory correction. Such a
  correction would reweight the two integrals sample by sample, and because the Gardner density is
  not a constant multiple of the measured density the ratio would in general change. A test,
  `test_uniform_scaling_invariance_does_not_extend_to_a_varying_correction`, asserts that limit.
  No trajectory correction is applied, and the one-dimensional qualification on the gradient
  calculations stands regardless.

### 5.3 Travel time: arithmetic, not a model

`DT` is logged in microseconds per foot. It *is* a one-way travel time per unit length, so
integrating it along the logged coordinate yields one-way time along that path directly, and twice
that is the round-trip time along the same path:

    t_ow(s) = integral of DT ds        t = 2 t_ow

This is **travel time along the logged path, not vertical two-way time**. No velocity model,
checkshot, datum correction or drift correction is applied, and none is supplied. The result is
used only to place the reflectivity on a time axis so a wavelet with a frequency in hertz can be
convolved with it. It is not a seismic-to-well tie; no field seismic data are used anywhere in this
project.

### 5.4 Two RMS ratios that are not the same quantity

The reflection-coefficient RMS ratio (Gardner / measured) is **1.2017**, computed on the reflection
coefficients *before* any wavelet is applied. The synthetic-trace RMS ratio, computed *after*
convolution, is **0.884 to 0.991** across the tested frequencies — below one everywhere. The two
are reported separately in `results/tables/resolution_sweep.csv` and in figure 5, and the first is
never presented as an amplitude statement about a synthetic trace or about field seismic data.

### 5.5 The correlation is a similarity measure

The Pearson correlation between a synthetic trace and its input reflectivity is reported as a
similarity measure for this synthetic experiment. It is not a fraction of information, variability
or structure recovered; its square is not either; it is not converted into a resolvable bed
thickness, because no quarter-wavelength or tuning rule is supplied; and no frequency in the sweep
is claimed to represent a field survey.

---

## 6. Deliberately excluded

Each of the following appeared in an earlier version of this project and has been removed because
no supplied source supports it. Listing them is part of the audit.

| Removed | Why |
|---|---|
| CO₂ storage scenario: reservoir depth, thickness, caprock, injection depth, fault strike and dip, throw, water depth | invented geometry; nothing in either folder describes this structure |
| Biot coefficient (0.88) | not measured; `Models/2_Hard_rocks.pdf` s.8 gives `alpha ≈ 1` for *strength* only, not a matrix stress-path value |
| Pore-pressure profile, seawater density 1025 kg/m³, mud weight | no pore-pressure or mud measurement in the supplied data; LAS `DFD` is 0.000 |
| Thermal expansion coefficient, cooling scenario, thermoelastic stress terms | no thermal property or temperature measurement supplied |
| Kirsch borehole wall stresses, mud-weight window, UCS = 60 MPa | Kirsch is not in the supplied notes; the UCS came from an external table |
| Cohesion, friction coefficient as an active parameter, slip and dilation tendency, critical pressure | `Models/5_Friction.docx` describes Byerlee's law, but applying it needs a fault and a stress state neither folder provides for this well |
| Monte Carlo over invented parameter distributions; Sobol/Spearman sensitivity | every distribution was an assumed range |
| Halite creep fit (Carter et al.) | the data *are* supplied in `Models/`, but the result has no connection to the seismic side and does not serve the question |
| Lithosphere strength envelope, geotherm 25 °C/km, strain rate 1e-15 s⁻¹ | conventional values from external literature |
| Slope stability, thrust-sheet, friction-angle 32° examples | no site data; the friction angle was assumed |
| Facies machine learning on `training_data.csv` | the file is supplied in `Seismic/`, but it is an unrelated dataset from other wells and does not serve the question |
| NMO, semblance velocity analysis, CMP gathers | Ex2's gathers are not in the supplied folder |
| SEG-Y volumes, attribute analysis, coherency, RMS amplitude | Ex3/Ex4 volumes are not in the supplied folder |
| Quarter-wavelength / tuning-thickness resolution rule | not stated anywhere in the supplied material |
| Any claim about what a field seismic survey would image | no field seismic data are used; the supplied material gives no representative survey frequency |
| Any well trajectory, inclination or TVD conversion | not supplied, and not obtainable from the permitted sources |
| Generic property tables used as inputs (Young's modulus and Poisson's ratio by rock type) | the table on p.2 of `1_Elasticity.pdf` exists, but assigning a row of it to a depth interval of this well would require a lithology assignment the data cannot support |
| Streamlit dashboard, PDF report generator | presentation layers built on the removed science |

---

## 7. Known source gaps

Genuine gaps, stated rather than filled.

1. **No shear sonic (`DTS`) in the supplied LAS.** `K` and `G` cannot be separated, so no `E`,
   `nu`, `Vs` or horizontal stress can be computed for this well. This is the binding constraint on
   the project.
2. **No pore-pressure measurement and no mud weight.** All reported stresses are total.
3. **Density is logged over 236 m of a 3,850 m well.** No absolute vertical stress can be computed;
   only increments and gradients are reported.
4. **Gardner's unit convention is not stated in `Ex1.ipynb`.** The m/s → g/cm³ reading is inferred
   and then confirmed against the measured log, but the notebook itself does not say it.
5. **`Ex1.ipynb` cites "Lecture 1 slide 22" and "slide 24" for a velocity table and for Gardner.**
   The lecture slides are not in the supplied `Seismic` folder. Gardner survives because the
   expression itself is written out in the notebook; the velocity table does not, and no rock-type
   velocity is used anywhere in this project.
6. **No hole-condition cut-off is supplied** for `CALI` or `DRHO`, so none is applied. Part of the
   0.072 g/cm³ residual scatter may be borehole effect rather than model error, and this dataset
   cannot separate the two.
7. **The depth convention is unresolved.** Section 5.1 sets out the evidence. A deviation survey
   would settle it; none is available from the permitted sources, and none is assumed.
8. **The original Gardner refit is in-sample.** Its coefficients were fitted and evaluated on the
   same 1,105 paired samples. The October 2026 revision adds a within-well, two-direction
   depth-block test (`configs/depth_blocks.json`, `results/depth_blocks/`): coefficients fitted on
   one block predict the other. The two blocks are adjacent parts of one 221 m interval of one
   log, either side of a 20 m gap, so they are not independent data. Validation on another well
   is unavailable: the supplied material contains paired sonic and density data for this one
   well only.
