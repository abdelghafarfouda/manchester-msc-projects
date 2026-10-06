# SeisGeoMech

**How well does Gardner's velocity–density relation describe the measured interval at well
48/10b-9, does a locally fitted relation predict a separate depth interval, and what do the
density errors imply for the qualified ρg calculations?**

MSc coursework for *Subsurface Mechanics and Geoengineering* (University of Manchester), published
in September 2026 and extended and re-verified in October 2026. It is a small, fully
source-traceable study that links one seismic method to one geomechanical consequence.

## Supervisor overview

**Why seismic and geomechanics meet here.** When a synthetic seismogram is built without a density
log, density is predicted from velocity with Gardner's relation, `rho = 0.31 Vp^0.25` (the seismic
practical). The geomechanics notes give the overburden relation `d(sigma_zz)/dz = rho g`, which
integrates that same density. A density prediction made for the seismic model is therefore also a
statement about stress. This project tests the prediction where a measurement exists, then follows
its error into both subjects.

**Three kinds of evidence, kept apart.**
* **Measured well-log evidence.** UK well 48/10b-9, where the sonic and density logs overlap over
  221 m (1,105 paired samples, 3,614–3,835 m along the logged coordinate). Every Gardner and ρg
  result comes from these samples.
* **Synthetic seismic calculations.** Impedance, reflection coefficients and Ricker synthetics built
  from those logs. No field seismic data are used anywhere.
* **Worked-example verification.** The exercise sheet's olivine (Q3) and Merivale granite (Q4)
  problems check the elasticity code. They are not data from this well.

**Findings.**
* **The supplied relation under-predicts density here.** Over the 1,105 samples the bias is
  **−0.093 g/cm³ (−3.5 %)** and the RMSE 0.118 g/cm³. Refitting the same form to the same samples
  gives `rho = 0.159 Vp^0.333` with RMSE 0.069 g/cm³. That is an in-sample description, not a
  prediction.
* **A locally fitted relation does predict a separate depth interval better, with limits.** In a
  two-direction depth-block test, the coefficients are fitted on one ~100 m block and scored
  on the other. The design and split were committed before any held-out score; the design was
  pre-specified but not blind, because the full-interval residuals had been published in
  September. A 20 m gap separates the
  blocks. The held-out RMSE is **0.088 and 0.061 g/cm³**, against 0.113 and 0.129 for the supplied
  relation on the same samples: better in both directions under the rule fixed in advance. Most
  of the gain is the removal of the supplied relation's bias. Yet the
  two blocks give different coefficients (exponents 0.288 and 0.314), and each block fit
  mis-predicts the other block by about ±0.035 g/cm³. The blocks are adjacent parts of one well, so
  this is a within-well test. Validation on another well is not possible with the supplied data.
* **The density error carries straight into ρg.** The gradient is linear in density, so the bias
  reappears as a gradient error of the same percentage. Over the whole overlap the result is
  **24.97 against 25.89 MPa/km**. Over each 100 m block the supplied relation is 0.75 and
  1.17 MPa/km low, and the block-fitted relation is +0.35 and −0.34 MPa/km off. Every gradient is a
  **one-dimensional application** along the logged coordinate, because the file does not establish
  that coordinate as vertical. The measured-versus-predicted differences are invariant under a
  uniform scaling of it, not under a depth-varying trajectory correction.
* **The same substitution acts differently on the seismic side.** Gardner makes impedance a
  function of velocity alone. That raises the RMS of the reflection coefficients by 20 %
  (correlation 0.935). After wavelet convolution, however, the synthetic-trace RMS ratio is
  0.88–0.99, below one at every frequency tested. The two ratios are different quantities.

**Limits.** One well and one 221 m interval near total depth. There is no shear sonic, so no
Young's modulus, Poisson's ratio or horizontal stress for the well. There is no pore pressure, so
every stress is total. The depth convention is unresolved. No hole-condition filter is applied,
because the material supplies no threshold. The depth-block gap is a design choice: it does not
make the blocks independent.

**Re-verification and a fixed install defect.** Before any change, the 126 tests passed and the 12
tables, 65 summary values and 6 figures reproduced exactly from the recorded lock file. A clean
Python 3.12+ install failed, because `bruges` imports `pkg_resources` and such environments ship
without setuptools. Listing `setuptools<81` fixes it without changing any result. CI now runs a
locked Python 3.11 job and a fresh Python 3.13 job, and both compare every table with the recorded
one. Details: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```bash
pip install -r requirements.txt     # Python >= 3.10; requirements-lock.txt for the exact 3.11 set
python scripts/run_all.py           # every table and figure, including the depth-block test (~5 s)
python -m pytest tests -q           # 184 tests
```

Details: [depth-block test](docs/DEPTH_BLOCK_TEST.md) ·
[reproducibility](docs/REPRODUCIBILITY.md) · [sources](SOURCE_MAP.md) ·
[review checklist](docs/REVIEW_CHECKLIST.md) · notebook `notebooks/SeisGeoMech.ipynb`.

---

## Result in one paragraph

Over the 221 m of well 48/10b-9 where the sonic and density logs overlap (1,105 samples,
3,614–3,835 m), the Gardner relation as supplied, `rho = 0.31 Vp^0.25`, **under-predicts bulk
density by 0.093 g/cm³, a systematic 3.5 %**, with an RMS error of 0.118 g/cm³. Because the
supplied overburden model integrates that same density, the error transfers exactly: a ρg gradient
of **24.97 against 25.89 MPa/km**, 0.91 MPa/km lower.

On the seismic side, the same substitution makes acoustic impedance a function of velocity alone
(`Z = 0.31 Vp^1.25`). That preserves the shape of the **reflection-coefficient series**
(correlation 0.935) while raising its RMS by 20 %. After wavelet convolution, the
**synthetic-trace** RMS ratio is 0.88–0.99, below one at every frequency tested, so the two ratios
are different quantities and behave differently.

Refitting the same functional form to all 1,105 points gives `rho = 0.159 Vp^0.333` and lowers
RMSE by 41.5 % on the very data it was fitted to. The October 2026 depth-block test asks the
predictive question that refit could not. Coefficients fitted on one block of the interval
predict the other block with RMSE 0.088 and 0.061 g/cm³, against 0.113 and 0.129 for the
supplied relation on the same samples.

---

## Two things the numbers depend on

**The logged depth coordinate is not established as vertical.** The depth curve is labelled
`DEPT`, which by itself says nothing about the convention. The file has no TVD curve and no
deviation survey, `LMF` ("logs measured from") is `UNKNOWN`, and every elevation and water-depth
field is zero. The only statement about depth convention in the supplied material names the
measured-depth / true-vertical-depth distinction without resolving it.

The absolute ρg gradients below are therefore reported as an explicitly **one-dimensional
application of the supplied model** to the logged coordinate, not as a verified field stress
profile. Likewise, the two-way times are travel time along the logged path: not vertical TWT, and
not a seismic tie. No trajectory is obtained, invented or assumed. The depth-block gap and block
lengths are also measured along this coordinate. `las_io.depth_convention_evidence()` prints the
evidence, which is saved to `results/tables/depth_convention_evidence.csv`, and a test asserts that
every vertical-depth and trajectory curve is still absent.

**The measured-versus-Gardner difference is invariant under uniform coordinate scaling.** Both
density profiles are integrated over the same coordinate array, so multiplying every interval by
one constant factor rescales both integrals identically, and the factor cancels from their ratio.
This is verified numerically: a uniform rescaling leaves the ratio unchanged to 1.3e−15.

That claim is deliberately narrow. It does **not** establish invariance under a depth-dependent
trajectory correction. Such a correction would reweight the two integrals sample by sample, and
because the Gardner density is not a constant multiple of the measured density, it would in general
change the ratio. A test asserts that limit explicitly. No trajectory correction is applied, and
the gradient calculations keep their one-dimensional qualification throughout.

---

## Data

One dataset, copied unmodified from the module's seismic practical folder:
`data/raw/48_10b-_9_jwl_JWL_FILE_1682139.las`. It is UK well **48/10b-9**, southern North Sea,
operator BP, spudded 1990-08-26: LAS 2.0, 18,975 depth samples, NULL = −999.25.

Two facts about that file set the shape of the whole project:

* `RHOB` is logged over only ~236 m near total depth (about 6 % of the well), while `DT` covers
  766–3,835 m. Their **221 m overlap (1,105 samples)** is the only place Gardner can be tested
  against a measurement.
* There is **no shear sonic (`DTS`)**, so K and G cannot be separated, and no Young's modulus,
  Poisson's ratio or horizontal stress is computed for this well anywhere.

Two further datasets are used only to verify the elasticity code. Both are transcribed from the
module's exercise sheet rather than measured here: the olivine stiffness tensor (Q3) and the
Merivale granite uniaxial-compression record (Q4). Measured log data, synthetic calculations and
these worked examples are kept separate throughout.

## Methods

| Step | What is done |
|---|---|
| Read the log | `lasio`, rejecting only the header NULL. No caliper or DRHO cut-off, because the material supplies no threshold |
| Velocity | `Vp = 10⁶ × 0.3048 / DT`: the definition of the logged unit µs/ft, not a model |
| Gardner test | `rho = 0.31 Vp^0.25` evaluated against measured `RHOB` over the 1,105 paired samples |
| Refit | the same functional form refitted to those points: an **in-sample calibration** |
| Depth-block test (2026) | the same refit on one depth block, scored on the other, both ways, against the supplied relation on the same samples. Split frozen before scoring ([docs/DEPTH_BLOCK_TEST.md](docs/DEPTH_BLOCK_TEST.md)) |
| ρg gradient | `d(sigma_zz)/dz = rho g` integrated along the logged coordinate, with measured, Gardner and (2026) block-fitted density in turn |
| Forward model | `Z = rho·Vp`, `R = (Z₂−Z₁)/(Z₂+Z₁)`, Ricker wavelet, `np.convolve` |
| Frequency sweep | 25–300 Hz, reporting Pearson similarity **and**, separately, synthetic-trace RMS |
| Elasticity | `M = rho Vp² = K + 4G/3`, the only elastic modulus this well supports |
| Verification | the two supplied worked examples, plus analytical identities and unit checks |

Every one of these is traced to a specific page, section or notebook cell in `SOURCE_MAP.md`, or
listed there as a design choice.

---

## Headline numbers

| Quantity | Value |
|---|---|
| Overlap interval where Gardner can be tested | 3,614.2 – 3,835.0 m, 1,105 samples |
| Gardner bias vs measured density | **−0.0932 g/cm³ (−3.53 %)** |
| Gardner RMSE | 0.1179 g/cm³ |
| Correlation, Vp vs measured RHOB | 0.758 |
| Same form refitted, **in-sample** on these points | `rho = 0.159 Vp^0.333`, RMSE 0.0690 g/cm³ (41.5 % lower on the calibration data) |
| ρg gradient, measured density (1-D application) | **25.887 MPa/km** |
| ρg gradient, Gardner density (1-D application) | **24.973 MPa/km** |
| Gradient difference (invariant under uniform coordinate scaling) | **−0.914 MPa/km (−3.53 %)** |
| P-wave modulus `M = rho Vp²` over the overlap | 28.0 – 83.1 GPa (mean 55.9) |
| Reflection-coefficient RMS ratio, **before** convolution | 1.202 |
| Synthetic-trace RMS ratio, **after** convolution | 0.884 – 0.991 across the tested frequencies |
| Reflection-coefficient correlation, Gardner vs measured | 0.935 |
| Pearson r, synthetic trace vs input reflectivity | 0.066 at 25 Hz to 0.447 at 300 Hz (similarity measure for this experiment only) |
| Verification: olivine VRH (Models Q3) | K 129.45, G 77.97 GPa; Vp 8,341 m/s, Vs 4,821 m/s |
| Verification: Merivale granite (Models Q4) | E 99.95 GPa, ν 0.2534; Vp 6,750 m/s, Vs 3,879 m/s |
| **2026:** fitted on block A (3,614–3,714 m), predicting B (3,735–3,835 m) | `0.236 Vp^0.288`: held-out RMSE **0.088**, bias +0.036 g/cm³; supplied relation 0.113, −0.076 |
| **2026:** fitted on block B, predicting A | `0.186 Vp^0.314`: held-out RMSE **0.061**, bias −0.035 g/cm³; supplied relation 0.129, −0.119 |
| **2026:** 1-D ρg gradient error over the predicted block | block fit +0.35 / −0.34 MPa/km; supplied relation −0.75 / −1.17 MPa/km |

The September 2026 numbers are in `results/tables/` and the 2026 extension's numbers are in
`results/depth_blocks/`. All are regenerated by `python scripts/run_all.py`.

![Gardner's relation tested against the measured density log](results/figures/fig02_gardner_vs_measured.png)

*Gardner's relation against the measured density log over the 1,105 paired samples. The panels show
the profile comparison, the Vp–density crossplot with the supplied and refitted curves, and the
residual distribution with its −0.093 g/cm³ bias.*

![Depth-block test](results/depth_blocks/fig07_depth_blocks.png)

*The 2026 depth-block test. Left: the frozen split, with the 20 m excluded gap shaded. Centre: each
block's fitted relation against the supplied one. Right: the held-out residuals, each block
predicted from the other; the supplied relation is shown in grey.*

![Ricker frequency experiment](results/figures/fig05_resolution.png)

*The wavelet-frequency experiment. Left: Pearson similarity between the synthetic trace and its
input reflectivity, a similarity measure for this synthetic experiment only. Right: the two RMS
ratios that must not be conflated: 1.202 on the reflection coefficients before convolution, but
0.88–0.99 on the synthetic traces after it.*

### How to read the Pearson r

It is the linear correlation between a synthetic trace and the reflectivity series that produced
it, inside this synthetic experiment. It is **not** a fraction of the log's information,
variability or structure recovered, and neither is its square: it compares a broadband input with
a band-limited output of that same input, so it is not a decomposition of anything. It ranks
frequencies against each other here and does nothing else. No quarter-wavelength or tuning rule is
supplied, so it is not converted into a bed thickness. It supports no claim about what a field
survey would image, because no field seismic data are used anywhere in this project.

---

## What each subject contributes

| | contribution |
|---|---|
| **Seismic** | the Gardner relation; acoustic impedance; normal-incidence reflection coefficients; the Ricker wavelet and convolution; the wavelet-frequency experiment; and the LAS file itself |
| **Models** | `d(sigma_zz)/dz = rho g`; the uniaxial-strain result `sigma_xx = nu/(1−nu) sigma_zz`; the link between elastic moduli and Vp/Vs; and the two worked examples that verify the elasticity code |

The connection is one-directional: a seismic-side density model feeds a mechanics-side ρg
gradient, because they are the same `rho`. Nothing is forced in the other direction.

---

## Reproducing the results

Python 3.10 or newer. The recorded results were produced on Python 3.11.15. In October 2026 they
were re-run on Python 3.11 and 3.13, and CI runs both.

```bash
cd SeisGeoMech
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or requirements-lock.txt: the exact Python 3.11 set
```

or, with conda:

```bash
conda env create -f environment.yml
conda activate seisgeomech
```

**Why `setuptools<81` is listed.** The Ricker wavelet comes from `bruges`, the library the course
practical uses, and `bruges` 0.5.4 runs `from pkg_resources import ...` when it is imported.
`pkg_resources` is part of setuptools. A Python 3.12+ virtual environment contains no
setuptools, and setuptools 82 removed `pkg_resources`, so a clean install failed with
`No module named 'pkg_resources'`. The pin restores it and changes no result. A
`pkg_resources is deprecated` warning from `bruges` on import is expected and harmless.

## Run

```bash
python scripts/run_all.py                 # into results/
python scripts/run_all.py --out-dir DIR   # elsewhere; compare with scripts/compare_results.py
```

This writes 12 CSV tables plus `summary.json` to `results/tables/` and 6 figures to
`results/figures/`, the September 2026 workflow, unchanged. It then writes the depth-block test
to `results/depth_blocks/`. The whole run takes about 5 seconds.

```bash
python -m pytest tests -q
```

184 tests: the original 126, 40 for the depth-block test and 18 for the result comparison.
They take about 5 seconds.

The single starting notebook is:

```bash
jupyter lab notebooks/SeisGeoMech.ipynb
```

It runs top to bottom in the order
**purpose → supplied inputs → assumptions → calculations → verification → results → limitations**,
and is the recommended entry point.

Every table and figure in `results/tables/`, `results/figures/` and `results/depth_blocks/` is
generated by `run_all.py`, except `results/original_2026-09-27/`, which is the record of the
baseline reproduction. Of the generated files, `results/depth_blocks/split.json` and
`split_samples.csv` were first committed with the frozen design, before scoring, and
`run_all.py` regenerates them identically.

**Reproducibility.** With `requirements-lock.txt` the tables and summary values reproduce
**byte for byte** in this project's own environments (Python 3.11 and 3.13). On GitHub's runners
the same locked versions agree to the last few bits only, so CI compares numbers with a stated
tolerance, not bytes. Figures differ by a few pixels between Matplotlib versions. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Layout

```
SeisGeoMech/
├── README.md                  this file
├── SOURCE_MAP.md              every equation, dataset and constant, with its exact source
├── configs/depth_blocks.json  the frozen depth-block design (2026)
├── docs/                      depth-block test, reproducibility, review checklist (2026)
├── notebooks/SeisGeoMech.ipynb   the one starting notebook
├── scripts/run_all.py         regenerates every table and figure
├── scripts/depth_blocks.py    the depth-block test on its own (--split-only: the split, no scoring)
├── scripts/compare_results.py compares regenerated results with the recorded ones (CI)
├── src/seisgeomech/
│   ├── units.py               unit definitions and standard gravity
│   ├── las_io.py              reads the supplied LAS; reports coverage and depth-convention evidence
│   ├── elasticity.py          isotropic constants, VRH averaging, Vp and Vs
│   ├── stress.py              rho g integration along the logged coordinate, uniaxial-strain ratio
│   ├── seismic.py             Gardner, impedance, reflectivity, Ricker, frequency sweep
│   ├── worked_examples.py     olivine (Q3) and Merivale granite (Q4)
│   ├── analysis.py            the seven-stage workflow
│   ├── depth_blocks.py        the frozen two-direction depth-block test (2026)
│   └── figures.py             presentation only
├── tests/                     184 tests
├── data/raw/                  the supplied LAS file, unmodified
└── results/                   tables/ and figures/ (2026-09-27 workflow), depth_blocks/ (2026),
                               original_2026-09-27/ (the baseline reproduction record)
```

---

## Source rule

Every scientific equation, parameter, dataset and method in this project is traceable to a
specific page, section or notebook cell in the two supplied teaching folders (`Models/` and
`Seismic/`). `SOURCE_MAP.md` gives the reference for each one. It also lists the design choices
that are *not* scientific assumptions, including the depth-block split and its reading rule,
records what was removed and why, and states the genuine source gaps.

Nothing is imported from outside those folders: no external property table, correlation, paper or
dataset; no assumed rock property; no invented scenario; no well trajectory. Where the data run
out, the project says so rather than filling the gap. Tests in `tests/test_workflow.py` and
`tests/test_depth_blocks.py` guard this mechanically. They cover an information-recovered
reading of the correlation and any field-seismic calibration claim. In the depth-block files,
README, SOURCE_MAP, `docs/` and the notebook's text, a test checks that independence and
validation on another well are only mentioned to say that they do not apply.

---

## Scope: what is implemented, and what is not

Everything in the **Headline numbers** table is computed by the code in this repository and
regenerated by `python scripts/run_all.py`. Nothing in this project is a placeholder, a stub or a
figure copied in by hand.

**Implemented and verified here**

| | |
|---|---|
| LAS reading, QC and curve-coverage reporting | `src/seisgeomech/las_io.py` |
| Gardner's relation tested against the measured density log, 1,105 paired samples | `results/tables/gardner_vs_rhob.csv` |
| In-sample refit of the same functional form | `summary.json` → `gardner_refit_*` |
| Two-direction depth-block test of the refit, design frozen before scoring (2026) | `results/depth_blocks/` |
| ρg integration along the logged coordinate, measured vs Gardner density | `results/tables/rho_g_profile.csv` |
| P-wave modulus `M = rho Vp²` profile | `results/tables/p_wave_modulus_profile.csv` |
| Acoustic impedance, normal-incidence reflectivity, Ricker convolution | `results/tables/impedance_and_reflectivity.csv` |
| Wavelet-frequency sweep, 25–300 Hz, similarity **and** synthetic-trace RMS | `results/tables/resolution_sweep.csv` |
| Depth-convention audit of the supplied LAS | `results/tables/depth_convention_evidence.csv` |
| Olivine (Q3) and Merivale granite (Q4) worked examples | `results/tables/worked_example_*.csv` |
| 184 tests, including analytical identities and source-compliance guards | `tests/` |

**Deliberately not implemented.** Each item would need a measurement or a source that the module
material does not supply, so none of it is attempted, approximated or presented as a result:

| Not done | Why |
|---|---|
| Young's modulus, Poisson's ratio, Vs or horizontal stress for this well | no shear sonic (`DTS`) in the supplied LAS; `M` is the only modulus Vp and density can give |
| Effective stress | no pore-pressure measurement and no mud weight. The law is coded in `stress.effective_stress` and never called, and a test enforces that |
| Absolute vertical stress, or a true-vertical-depth conversion | density is unlogged above 3,614 m, and the file establishes no depth convention. Only increments and gradients are reported, with a one-dimensional qualification |
| Lithology, porosity or facies | no GR cut-off and no matrix density are supplied |
| No validation of the Gardner refit on another well | the supplied material contains paired sonic and density data for this one well only. The within-well depth-block test is the only held-out check |
| A resolvable bed thickness | no quarter-wavelength or tuning rule is supplied, so the frequency sweep is reported as a similarity measure only |
| NMO, semblance, migration, AVO, attributes, seismic-to-well tie | those practicals use SEG-Y volumes and CMP gathers that are not part of the supplied material; no field seismic data are used anywhere |

**Possible extensions, not claimed as results.** A shear sonic log would close the largest gap and
make E, ν and the uniaxial-strain horizontal stress computable at this well. A deviation survey
would settle the depth convention. Paired sonic and density data from a second well would allow a
test on data from outside this well. None of these are attempted here, and no part of the repository
assumes them.

---

## Limitations

1. **The test interval is 221 m near total depth**: 6 % of the well, one lithological setting.
   Everything said about Gardner's performance is measured there, and no extrapolation beyond the
   logged interval is offered.
2. **No shear sonic.** Without Vs, K and G cannot be separated, so no E, ν or horizontal stress is
   computed for this well. `sigma_h/sigma_v = nu/(1−nu)` is shown parametrically only. This is the
   single measurement that would most extend the project.
3. **No pore pressure, no mud weight.** All stresses are total. The effective stress law is
   implemented and deliberately never called.
4. **No absolute vertical stress.** Density is unlogged above 3,614 m, so only integrals and
   gradients are reported.
5. **No lithology or porosity.** No GR cut-off and no matrix density are supplied.
6. **No hole-condition filtering.** CALI and DRHO are reported, not applied, because no threshold
   is supplied. Some residual scatter may be borehole effect rather than model error.
7. **Gardner's unit convention is inferred**, then confirmed against measurement; `Ex1.ipynb` does
   not state it.
8. **The frequency-sweep metric is a similarity measure, not a resolution limit.** See *How to
   read the Pearson r* above.
9. **Zero-offset, normal incidence only.** No NMO, migration, AVO or attribute work. No field
   seismic data are used anywhere, so there is no seismic-to-well tie and no amplitude calibration.
10. **The depth convention is unresolved**, so absolute gradients are a one-dimensional application
    of the supplied model rather than a field stress profile. The measured-versus-Gardner
    difference is unaffected by a uniform rescaling. A deviation survey would resolve this.
11. **The original refit is in-sample; the depth-block test is within one well.** The two blocks
    are adjacent parts of the same 221 m interval. The 20 m gap between them is a design choice,
    and the residuals are still correlated at 0.08–0.10 across it, so the blocks are not
    independent. Two directions give two numbers, not a distribution, and no confidence interval
    is claimed.
12. **One well.** This establishes what Gardner does at 48/10b-9 between 3,614 and 3,835 m, and
    nothing more general.

---

## Version history

| version | date | what it is |
|---|---|---|
| Coursework | during the MSc | the project for *Subsurface Mechanics and Geoengineering*. Earlier versions included scenarios and parameters that the supplied sources could not support; they were removed before publication (`SOURCE_MAP.md` §6) |
| 3.0.0, published | results generated 2026-09-27 | the Gardner test, ρg gradients, reflectivity, frequency sweep and worked examples; 126 tests |
| **3.1.0, extended and re-verified** | **2026-10-04** | baseline reproduced exactly; clean-install defect fixed; frozen two-direction depth-block test; project CI with two jobs. No original result changed |

## Licence and attribution

Code, documentation and generated results in this folder are released under the MIT Licence; see
[`LICENSE`](LICENSE). Author: **Abdelghafar Fouda**.

**Data attribution.** `data/raw/48_10b-_9_jwl_JWL_FILE_1682139.las` is a public-domain UK
continental-shelf well log (well 48/10b-9, operator BP, spudded 1990-08-26). It was supplied with
the module's seismic practical material and is redistributed here unmodified, so that the workflow
runs without external downloads. See [`data/raw/README.md`](data/raw/README.md).

**Teaching material.** No course teaching material is redistributed in this repository. The
`Models` and `Seismic` folders referenced throughout `SOURCE_MAP.md` are the University of
Manchester EART35102 notes, exercises and practical notebooks. `SOURCE_MAP.md` cites the specific
page, section or notebook cell behind every equation, dataset and constant without reproducing any
of it.

**Results and records.** The results in
`results/tables/` and `results/depth_blocks/` are produced by `scripts/run_all.py` and compared
with the recorded files in CI. The reproduction records in `results/original_2026-09-27/` come
from runs whose logs are kept there. The GitHub-runner differences quoted in
`docs/REPRODUCIBILITY.md` come from CI runs whose logs are on GitHub, not in the repository.

Module: **Subsurface Mechanics and Geoengineering**, MSc Subsurface Energy Engineering,
University of Manchester.
