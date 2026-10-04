# Assumptions and deliberate omissions

Every modelling choice that can change a number. "Omitted physics" is a
deliberate scope decision, not an implementation error; genuine defects that
were found and fixed are listed in `docs/CHANGELOG.md`.

## A. Physics

| # | Assumption | Consequence | Source / status |
|---|---|---|---|
| A1 | Isothermal | no energy equation | `1-Transmissibility.pdf` p.2 |
| A2 | **Immiscible** CO2 / brine — no dissolution, no water vaporisation | the two-shock CO2–brine structure of `4-CO2 BL.pdf` p.25–29 is *not* reproduced; dissolution trapping not modelled; retained mass is free-phase mass | omitted physics |
| A3 | `B_a = B_g = 1`; densities at reservoir conditions | IMPES `β = 1`, total-mobility pressure equation | `4-CO2 BL.pdf` p.2 (no FVF), p.21 |
| A4 | Slight compressibility only in storage, `c_t = c_r + S_a c_a + S_g c_g`; CO2 accumulation written conservatively | CO2 mass balance exact to round-off; brine-equation splitting residual reported | `1-Transmissibility.pdf` p.6, 15–16 |
| A5 | **Capillary pressure `P_c = 0`** | no capillary smearing of fronts, no capillary entry barrier | `4-CO2 BL.pdf` p.19; no CO2–brine `P_c` curve is supplied (the former unused switch was removed) |
| A6 | **Gravity neglected**; 1-D radial flow in each layer | **no buoyant override** — the largest unmodelled effect: real plumes rise and spread under the caprock. Quantified since 2026-10 with the r–z reference model (`rz.py`, `scripts/run_model_form.py`): see the model-form section of `docs/TECHNICAL_REPORT.md` | omitted physics (quantified, not removed) |
| A7 | Layers isolated vertically (no crossflow); they communicate only through the well | stratified idealisation that makes Dykstra–Parsons meaningful; its effect is quantified together with A6 | project choice |
| A14 | In the r–z reference model, `k_v/k_h = 0.1` and five rows per layer (ten in a resolution check) | the size of the model-form error depends on this assumed anisotropy | assumption (r–z model only) |
| A8 | **One rate-controlled injector, one common bottom-hole pressure**, solved implicitly with the pressure field; total well-block mobility; a completion whose well-block pressure exceeds the BHP is closed (an injector does not produce); at shut-in every completion is closed (no wellbore crossflow) and the reported BHP is a zero-net-rate diagnostic | the split between layers follows kh in open systems and pore volume in sealed ones at late time (tested) | `3-IMPES.pdf` p.16 + project choices |
| A9 | Drainage kr only — no hysteresis, residual trapping not modelled | post-injection redistribution over-estimated | `3-IMPES.pdf` p.4 notes both curves may be needed |
| A10 | Power-law relative permeability on normalised saturation | shape parameters `n_g`, `n_a`, end points `krg0`, `kra0 = 1`, `S_ar`, `S_gr` | `4-CO2 BL.pdf` p.18; functional form is a project choice |
| A11 | Base case outer boundary **closed** (sealed compartment) | retained = injected mass exactly; pressure accumulates so the schedule shape matters; the open boundary (Dirichlet, `1-Transmissibility.pdf` p.11) is run as a variant | project choice |
| A12 | Property values (ρ, μ, c, kr end points, prior ranges) are generic assumed values for a deep saline aquifer | not measured; not from the course material | assumption |
| A13 | Layers homogeneous in the radial direction; heterogeneity is between layers (lognormal, `σ_lnk = −ln(1−V_DP)`) | Dykstra–Parsons controls the contrast | `5-Uncertainty.pdf` p.23, p.43 |

## A′. The analytical reduced-order model (ROM, `rom.py`)

| # | Assumption | Consequence |
|---|---|---|
| R1 | Each realised layer is a sealed tank in pseudo-steady state with the PSS productivity index `2πkh/(μ_a(ln(r_e/r_w) − 3/4))` | exact in the single-phase limit (V12, V13); the early transient after each rate change is ignored |
| R2 | Brine viscosity throughout (no two-phase mobility near the well) | the residual left by this simplification is what the hybrid surrogate learns; a two-region variant was tried and was less accurate (`docs/TECHNICAL_REPORT.md`) |
| R3 | Total compressibility `c_r + c_a`, plus the compressibility of the injected CO2 volume | as in the simulator, to first order |

## B. Numerics

| # | Choice | Quantified where |
|---|---|---|
| B1 | Near-well block `[r_w, r_near]`; pressure drop to the wellbore carried by the well equation | exact for steady single phase (V2); two-phase sensitivity in V7 (`r_near` sweep). Since 2026-10: where the peak build-up is a start-up transient (well block still brine-filled) its height depends on `r_near`; every case was screened (`experiments/peak_screen.json`): 0 of 400 test, 4 of 880 development and 32 of 240 shift cases, 2–14 % too high (conservative) |
| B2 | Strict local CFL limit + reject/retry on saturation change and on a two-sided discrete maximum principle | V6 counts turning points of the saturation profile (odd–even oscillation detector); V8 compares with the relaxed controller the earlier version used |
| B3 | No silent clipping; clipping would be counted | `n_clipped = 0` in every validation case and every dataset scenario (data-quality report) |
| B4 | Pressure solved in increment form | removes a round-off drift of ~10 Pa/yr seen when solving for absolute pressure with small storage |
| B5 | Discretisation error is reported, not assumed small | V7 on the verification case; since 2026-10 measured on a stratified sample of the dataset cases themselves (`numerics.py`, `metrics/numerics_summary.json`: 41 cases; peak build-up median 0.06 %, no pressure-limit label changes; plume radius about 4 % and swept fraction about 17 % too large at production resolution) and carried into the error-source table |

## C. Statistics and machine learning

| # | Assumption | Note |
|---|---|---|
| C1 | Data are **synthetic**, generated by the simulator in this repository | no field data, **no field validation** |
| C2 | Prior ranges express assumed uncertainty for a generic aquifer | P10/P50/P90 describe that prior, not a site |
| C3 | The grouping unit of every split is the **reservoir realisation** | `Lecture08`, `E03_geographicalspliting` (realisation ≙ geographic block); nested grouped CV for development decisions |
| C4 | Intervals (2026-10): split-conformal with one score per calibration reservoir (the largest of its schedules), scaled by a difficulty model | **The pipeline's intervals: measured coverage only.** Their 41 calibration reservoirs took part in the development experiments that chose the design, so the conformal argument does not apply (protocol addendum). **Recalibrated on 41 fresh reservoirs** (`experiments/calibration_check.json`): for a reservoir drawn like them, with four schedules from the same design, all four are covered with probability ≥ 90 % over the draw of calibration and new reservoirs. For one calibration set the coverage varies (Beta(38, 4); 90 % of sets 0.82–0.97). **Not** claimed for either: a reservoir from a different population (measured on the shift set), or the thousands of candidate schedules screened per reservoir |
| C4′ | The published method (empirical P5–P95 residual band) is kept for comparison | measured coverage only |
| C8 | Test reservoirs (2026-10) are fresh, generated after the design was fixed; the published run's 55 test reservoirs were inspected during development and are now training data | `docs/EVALUATION_PROTOCOL.md` |
| C9 | The distribution-shift set changes one prior range (median permeability 10–30 mD instead of 30–1000 mD) | a deliberately simple shift; other kinds of shift are not tested |
| C5 | Extreme scenarios are kept | the outlier study shows they are physically meaningful, not errors |
| C6 | Importance and SHAP describe model behaviour, not causality | stated with every figure |
| C7 | Classifier scores are used for ranking and thresholding only | probability calibration is not in the course material |

## D. Decision limits — inputs, not findings

| Quantity | Value (both configurations) | Status |
|---|---|---|
| `p_limit_MPa` | 24 (initial 15 → 9 MPa buildup) | **modelling assumption**: chosen to make the screening question bind for part of the prior, not derived from a fracture gradient, caprock test or any site; no fracture, caprock or leakage mechanism is modelled |
| `r_plume_limit_m` | 400 | stated assumption standing in for an areal review boundary |
| traffic-light amber band | above 80 % of the allowed buildup | stated assumption |

## E. Not claimed

No deployment, no industrial use, no field validation, no demonstrated
fracture/caprock/leakage safety, no causal effect of any feature, no history
match. A simplified radial, layered model does not reproduce 3-D field
behaviour.
