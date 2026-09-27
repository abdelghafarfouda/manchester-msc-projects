# Source map — what came from which supplied file

**Scope rule (since 2026-09-20).** The project uses only the methods,
equations and algorithms supplied or demonstrated in two folders:

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
| All ML components | notebooks listed in `docs/COVERAGE_MATRIX.md` | see §3 |
| Schedule screening | Monte Carlo (`5-Uncertainty.pdf` p.44–45) + surrogates | limits are stated assumptions |

## 3. Data Science and machine learning folder

73 files: 70 notebooks, `results.csv` (a `y_test`/`y_pred` table used by
`E03_Evaluationmetrics`, not reservoir data) and `utils.py` (plot helpers for
the UMAP/t-SNE notebook). Six notebooks are byte-identical duplicates
(`E02_SHAP`, `E02_clustering`, `E03_geographicalspliting`, `E04_Imbalance`,
`E0_Instruction`, `OptionalE03` with a `(1)` copy); most other `(1)` files
are the executed versions of unexecuted originals. The data files the
notebooks load (`../Excercise_data/...`, `../Lecture_data/...`) are **not**
supplied, so no notebook can be re-run as is and no course dataset is used.

The frozen, deduplicated topic inventory (42 topics, file-by-file
traceability including duplicates, exclusions) is
`docs/coverage/ml_topics.yaml`; the executed coverage is
`docs/COVERAGE_MATRIX.md`.

*Small language models / transformers:* not present in the material
(`Extra_transformer.ipynb` is scikit-learn `SimpleImputer` preprocessing).
The earlier TF-IDF "retrieval assistant" had no basis in the material and has
been removed from the project (it remains only in the author's archive of the original version).
