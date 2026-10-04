# Source map — what came from which supplied file

**Scope.** The version published on 2026-09-20 followed a strict rule: only
methods, equations and algorithms supplied or demonstrated in the two
course folders below.  The October 2026 revision keeps every component of
that version and adds a small number of components **from outside the course
material**; they are listed, with their primary sources, in §4, so nothing
taken from elsewhere is presented as course content.

The course folders are:

* **`Models`** — six PDF lecture handouts and `3-Advanced BL.pptx`
  (listed in §1);
* **`Data Science and machine learning`** — 70 Jupyter notebooks, `results.csv`
  and `utils.py` (§3).

Both folders are the author's copies of the course material and are **not**
included in this repository.

No external dataset, paper, tutorial, pretrained model or downloaded code is
used as a methodological source. Standard software (NumPy, SciPy, pandas,
scikit-learn, XGBoost, SHAP, imbalanced-learn, scikit-optimize, umap-learn,
matplotlib, Streamlit) is used to *implement* those methods. The course
material itself is not copied into the project.

Where the project has to make a choice that the material does not dictate,
it is labelled **project choice** below and in `docs/ASSUMPTIONS.md`.

## 1. The modelling folder — actual files (checked 2026-09-20 and 2026-09-27)

| File | Pages | Used for | Where in the code |
|---|---|---|---|
| `1-Transmissibility.pdf` | 28 | mass conservation (p.3–4), Darcy (p.5), rock/fluid compressibility (p.6, 15–16), boundary and initial conditions (p.11–13), discretisation and transmissibility with the harmonic face permeability (p.17–19), storage term (p.20), 2-D practice case (p.21–28) | `grid.py`, `single_phase.py`, `impes.py`; V1, V3 |
| `2-Upscaling.pdf` | 23 | series/parallel, linear/radial averaging (p.5–9), HA/AH averaging (p.10–11), bounding chain (p.14), flow-based upscaling in x and y (p.15–16), SPE10/ALG (p.17–23, described only) | `upscaling.py`; V9, V10; radial transmissibility in `grid.py` (p.8) |
| `3-IMPES.pdf` | 20 | two-phase continuity and Darcy (p.2–3), kr/Pc curve shapes, drainage vs imbibition (p.4), discretisation and upstream mobility (p.5–8), storage coefficients (p.9–10), IMPES approximation (p.12–13), pressure equation (p.14–15), well equation and constant-rate injector (p.16), explicit saturation update (p.18), applicability/time-step limits (p.20) | `impes.py` (the whole module), `grid.RadialGrid.well_index_geom` |
| `4-CO2 BL.pdf` | 37 | fractional flow with viscous/capillary/gravity terms (p.4–10), method of characteristics (p.11–13), Welge construction (p.14–17), end-point mobility ratio table and normalised saturation (p.18), CO2 BL assumptions incl. "no capillary and/or gravity forces" (p.19), cylindrical flow equation (p.20), compositional CO2–brine formulation and two-shock structure (p.21–29, **explained, not implemented**), displaced brine volume = injected CO2 volume (p.30–32) | `fluids.py` (`f_g`, `welge_shock`, `bl_profile_1d`, `endpoint_mobility_ratio`), V5, `features.r_fill_est_m` |
| `3-Advanced BL.pptx` | 21 slides | slide version of `4-CO2 BL.pdf` p.1–8 and p.19–32 (pressure/saturation/concentration equations, effects of gravity and capillarity, mobility-ratio curves, CO2 two-shock analysis). **Not in `Models.zip`**; no content beyond the PDF was found. | cited alongside `4-CO2 BL.pdf` |
| `5-Uncertainty.pdf` | 62 | uncertainty sources: input data, model bias, numerical error (p.8–31); heterogeneity: variance, CV, Dykstra–Parsons (p.19–23); combining uncertainties (p.32); probabilistic approach, PDFs (uniform, normal, triangular, lognormal), Monte Carlo steps (p.38–45); P10/P50/P90 (p.46); static-to-dynamic model selection (p.47–50) | `petrophysics.py`, `scenarios.py` (Monte Carlo sampling), `uncertainty.py` |
| `Uncertainty_LM_09.05.2022.pdf` | 36 | earlier version of `5-Uncertainty.pdf`; no unique content used | — |

### References in the earlier documentation that are **not** in the supplied material

The previous `SOURCE_MAP.md` cited files that are not present in these
folders. Their attributions have been removed or replaced:

| Cited before | Status here | Action taken |
|---|---|---|
| `2-Upscaling (1).pdf`, `3-IMPES (2).pdf`, `4-CO2 BL (1).pdf` | present without the `(1)`/`(2)` suffix | file names corrected everywhere |
| `Upscaling practical session.pdf` (scanned 2×2 case, "k1=1, k2=1.25, k3=2, k4=1.75") | **absent** — cannot be verified | test and validation replaced by the four-block layout of `2-Upscaling.pdf` p.15 with values labelled *illustrative project choice* |
| MRST textbooks (*Introduction to Reservoir Simulation Using MATLAB/GNU Octave*, *Advanced Modeling with MRST*) | **absent** | "background reference" claims removed |
| Giuliani et al. (2016) M3O paper | **absent**, and external | optimisation-methodology attribution removed; schedule search reduced to Monte Carlo screening (`5-Uncertainty.pdf` p.44–45) |
| scikit-learn online documentation (grouped CV) | external | grouped splitting now attributed to `Lecture08.ipynb` and `E03_geographicalspliting.ipynb` |

## 2. Component traceability (every main component → a supplied file)

| Component | Supplied source | Project choices (not dictated by the material) |
|---|---|---|
| Radial grid, geometric transmissibility, harmonic face k | `1-Transmissibility.pdf` p.18–19; `2-Upscaling.pdf` p.8 (series radial) | log-spaced cells; a single near-well block `[r_w, r_near]` |
| Single-phase implicit solver | `1-Transmissibility.pdf` p.10–20 | — |
| Two-phase IMPES | `3-IMPES.pdf` p.2–18 | saturation advanced from the CO2 (injected-phase) equation in conservative form so CO2 mass is conserved exactly; brine-equation residual reported |
| Upstream mobility | `3-IMPES.pdf` p.7–8 | — |
| Well: rate-controlled injector, `WC = 2πkh/ln(r_e/r_w)` | `3-IMPES.pdf` p.16 | common BHP across layers solved implicitly; total well-block mobility; injector-only completions; shut-in closes all completions |
| Time-step control | `3-IMPES.pdf` p.20 ("if time steps are kept small, IMPES provides accurate and stable solutions") | local CFL limit enforced, reject/retry, two-sided discrete maximum-principle check |
| Relative permeability | `4-CO2 BL.pdf` p.18 (normalised saturation, end-point kr, mobility ratio); `3-IMPES.pdf` p.4 (curve shapes) | power-law (Corey-type) exponents `n_g`, `n_a` — the material shows curves but no formula |
| Capillary pressure | `4-CO2 BL.pdf` p.19 (neglected) | `P_c = 0`; no curve supplied, none modelled |
| Buckley–Leverett / Welge benchmark | `4-CO2 BL.pdf` p.11–17 | — |
| Steady radial and closed-tank benchmarks | `2-Upscaling.pdf` p.8; `1-Transmissibility.pdf` p.6, 15–16 | — |
| Upscaling (analytical, flow-based, bounds) | `2-Upscaling.pdf` p.5–16 | illustrative 2×2 values |
| Layered heterogeneity, Dykstra–Parsons, CV | `5-Uncertainty.pdf` p.19–23, lognormal p.43 | layers homogeneous in r; porosity–permeability trend (idea from `Lecture03/08.ipynb` poro-perm examples; that data file is not supplied) |
| Scenario sampling | `5-Uncertainty.pdf` p.42–45 (distributions, Monte Carlo) | prior ranges are assumed generic values (`ASSUMPTIONS.md` A11, C2) |
| Uncertainty sources / P10-P50-P90 / error table | `5-Uncertainty.pdf` p.8–9, 27–32, 46 | empirical residual-percentile band with *measured* coverage |
| All ML components (2026-09-20) | notebooks listed in `docs/archive/2026-09-20_course_coverage/COVERAGE_MATRIX.md` | see §3 |
| Schedule screening | Monte Carlo (`5-Uncertainty.pdf` p.44–45) + surrogates | limits are stated assumptions; verification gate and simulator search added in 2026-10 (§4) |

## 3. Data Science and machine learning folder

73 files: 70 notebooks, `results.csv` (a `y_test`/`y_pred` table used by
`E03_Evaluationmetrics`, not reservoir data) and `utils.py` (plot helpers for
the UMAP/t-SNE notebook). Six notebooks are byte-identical duplicates
(`E02_SHAP`, `E02_clustering`, `E03_geographicalspliting`, `E04_Imbalance`,
`E0_Instruction`, `OptionalE03` with a `(1)` copy); most other `(1)` files
are the executed versions of unexecuted originals. The data files the
notebooks load (`../Excercise_data/...`, `../Lecture_data/...`) are **not**
supplied, so no notebook can be re-run as is and no course dataset is used.

The frozen, deduplicated topic inventory of the 2026-09-20 version (42
topics, file-by-file traceability including duplicates, exclusions) and its
executed coverage are archived in `docs/archive/2026-09-20_course_coverage/`
(not regenerated for the 2026-10 revision).

*Small language models / transformers:* not present in the material
(`Extra_transformer.ipynb` is scikit-learn `SimpleImputer` preprocessing).
The earlier TF-IDF "retrieval assistant" had no basis in the material and has
been removed from the project (it remains only in the author's archive of the original version).


## 4. Components added in the October 2026 revision — beyond the course material

| Component | What it uses | Source |
|---|---|---|
| Two-phase bottom-hole pressure checked against the bounded-reservoir pseudo-steady-state (PSS) solution (V12, V13) | `p_w − p_i = Q t/(c_t V_p) + Q μ [ln(r_e/r_w) − 3/4]/(2π k h)` and its commingled-layer limit | standard well-test theory, e.g. Dake, L.P. (1978), *Fundamentals of Reservoir Engineering*, Elsevier, ch. 6; the closed-tank balance and the radial well equation are course material (`1-Transmissibility.pdf` p.6, 15–16; `3-IMPES.pdf` p.16) |
| Analytical reduced-order model (`rom.py`) | one sealed PSS tank per realised layer, a common bottom-hole pressure and injector-only completions | the same PSS theory; analytical pressure build-up in closed aquifers is established, e.g. Zhou, Birkholzer, Tsang & Rutqvist (2008), *Int. J. Greenhouse Gas Control* 2, 626–639; Mathias, González Martínez de Miguel, Thatcher & Zimmerman (2011), *Transport in Porous Media* 89, 383–397. **No novelty is claimed for the ROM.** |
| Hybrid surrogate (`hybrid.py`) | ROM × learned multiplicative correction (residual learning) | a standard way of combining physics-based models with machine learning; see the survey of Willard, Jia, Xu, Steinbach & Kumar (2022), *ACM Computing Surveys* 55(4), doi:10.1145/3514228 |
| Realised-layer inputs (`features.ROCK_FEATURES`) | arithmetic, harmonic, minimum and maximum layer permeability of the generated rock | course averages (`2-Upscaling.pdf` p.5–9) applied to the realised layers; using them as inputs is a project choice motivated by the error analysis |
| Prediction intervals (`intervals.py`) | split-conformal prediction, case-level and with one score per reservoir; a locally adaptive variant | Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*, Springer; Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), *JASA* 113(523), 1094–1111; grouped/hierarchical data: Dunn, Wasserman & Ramdas (2023), *JASA* 118(544), 2491–2502. The per-reservoir maximum score is a simple conservative construction, not one of Dunn et al.'s estimators. Split-conformal intervals had been removed from the 2026-09-20 version under the course-only rule; they return here with this attribution. |
| Applicability-domain check (`domain.py`) | training-range and nearest-neighbour distance tests on the reservoir descriptors | project choice (standard practice; no specific source) |
| Verification-gated screening (`screening.py`) | proposals become recommendations only after simulation; proportional search along a schedule shape | project choice, using the near-linearity of the build-up in the rate verified by V12 |
| Nested reservoir-grouped cross-validation, paired reservoir bootstrap (`experiments.py`, `evaluation.py`) | outer/inner `GroupKFold`; bootstrap over reservoirs | grouped K-fold is course material (`Lecture05`, `Lecture08`, `E03_geographicalspliting`); nesting it and the cluster bootstrap are standard statistical practice |
| r–z reference model with gravity and vertical crossflow (`rz.py`) | phase-potential upwinding, gravity in the fractional-flow terms, vertical transmissibility | the gravity and capillary terms of the fractional-flow derivation are in `4-CO2 BL.pdf` p.4–10 and `3-Advanced BL.pptx` slides 4–5; their 2-D finite-volume implementation is a project extension. Verified against hydrostatic equilibrium, the analytical end state of gravity segregation, and the layered model in the no-gravity, no-crossflow limit (`tests/test_rz.py`). |
