# Technical report — SubsurfaceML (corrected version, 2026-09-20)

**Question.** In a sealed, layered saline-aquifer compartment, how do
permeability, heterogeneity, fluid mobility and the injection schedule
control the bottom-hole pressure build-up and the CO₂ distribution; and how
accurately do inexpensive surrogates, trained on simulations, predict those
outcomes for reservoirs they have never seen?

**Status of the evidence.** Sections 1–11 quote the **demo** run,
`results/demo/metrics/*.json`, written by `scripts/run_pipeline.py --config
config/demo.yaml` on 2026-09-20; §12 quotes the larger **study** run,
`results/study/`, which is the reference run of the README (Python 3.11,
scikit-learn 1.8, 2 CPU cores). The two runs use different grids and test sets,
so their numbers are not combined. The data are **synthetic**. Simulator
accuracy (§3) and surrogate accuracy (§5) are separate questions. Results of
the pre-correction run are superseded, not published and not used here.

---

## 1. Physical model (sources: `1-Transmissibility`, `3-IMPES`, `4-CO2 BL`)

A single injector in a cylindrical compartment of radius `r_e`, split into
`L = 4` horizontal layers of equal thickness with no vertical crossflow.
Within each layer, 1-D radial, isothermal, immiscible two-phase flow of CO₂
(`g`) displacing brine (`a`):

```
∂/∂t (φ ρ_l S_l) = (1/r) ∂/∂r ( r ρ_l k k_rl/μ_l ∂p/∂r ) + q_l ,   S_a + S_g = 1
```

with `P_c = 0` and no gravity (`4-CO2 BL.pdf` p.19). Relative permeabilities
are power laws of the normalised brine saturation `S_e = (S_a − S_ar)/(1 −
S_ar − S_gr)` with end points `k_rg⁰`, `k_ra⁰ = 1` (`4-CO2 BL.pdf` p.18; the
power-law form is a project choice). Slight compressibility enters the
storage term, `c_t = c_r + S_a c_a + S_g c_g` (`1-Transmissibility.pdf`
p.6, 15–16).

**Discretisation.** Radial cells with geometric-mean centres; face
transmissibility `T = 2πh k̄ λ^up / ln(r_{i+1}/r_i)`, `k̄` the
log-radius-weighted harmonic mean (series radial flow, `2-Upscaling.pdf`
p.8; `1-Transmissibility.pdf` p.19); mobilities upstream (`3-IMPES.pdf`
p.8). The first cell is a well block `[r_w, r_near]`.

**IMPES** (`3-IMPES.pdf` p.12–18). Pressure implicit, from the sum of the
phase equations (`β = 1` since `B = 1`):

```
Σ_j (T_a + T_g)_ij (p_j − p_i) + Q_i = (V_p c_t / Δt)_i (p_i − p_i^n)
```

solved per layer as a tridiagonal system for the increment `p − p^n`.
Saturation explicit, from the CO₂ equation written in conservative form
`V_p [S_g B(p)]^{n+1} − V_p [S_g B(p)]^n = Δt · flux`, `B = (1+c_r Δp)(1+c_g
Δp)`, so CO₂ mass is conserved to round-off; the brine-equation residual is
reported as the splitting error.

**Well.** One rate-controlled injector completed in all layers
(`3-IMPES.pdf` p.16):

```
q_l = J_l (p_bh − p_{l,0}),   J_l = [2π k_{l,0} h_l / ln(r_0/r_w)] · λ_{t,l,0},   Σ_l q_l = Q
```

`p_bh` is solved **implicitly with the pressure field**: each layer is
solved for the right-hand side (`x_l`) and for a unit well source (`y_l`);
then `p_bh = p_init + [Q + Σ J_l (x_{l,0} + p^n_{l,0} − p_init)] / Σ J_l (1 −
y_{l,0})`. A layer whose well-block pressure exceeds `p_bh` is closed
(injector-only completions); at shut-in all completions close (no wellbore
crossflow) and the reported BHP is the zero-net-rate diagnostic
`Σ J_l (p_ws − p_l0) = 0`. `Δp_bh,max` is taken over injecting steps.

**Time step.** Local CFL limit (0.9) using the cell throughput and the local
`df_g/dS_g`; steps are rejected and halved if the saturation change exceeds
0.05, leaves `[0, 1 − S_ar]`, or creates a new local extremum (tolerance
1e-3). `3-IMPES.pdf` p.20 motivates this: IMPES is accurate "if time steps
are kept small".

**Quantities of interest.** `Δp_bh,max` [MPa]; the radius enclosing 95% of
the free-phase CO₂ mass, `r_95` [m] (an integral measure, far less
grid-sensitive than a saturation contour); sweep efficiency = pore-volume
fraction with `S_g > 0.05`; retained mass (equal to injected mass under the
sealed boundary).

## 2. What was wrong before, and what changed

| Defect | Evidence before | Now |
|---|---|---|
| Rate split `∝ J_l` did not give one BHP | 103.3 vs 38.1 MPa implied BHPs (2 layers) | implicit common BHP; spread 0 Pa, rate error 2e-14 (V11) |
| Optimiser inputs used `k_rg⁰ = 0.4` | `log10 M` off by up to 0.2 decades | one feature path; identical inputs through 4 routes (test) |
| Capillary switch unused | bit-identical outputs for 3 values of `P_c` | removed; `P_c = 0` stated |
| Ungrouped calibration folds | code | calibration removed (not in course); grouped OOF everywhere |
| No model files | missing directory | models + SHA-256 manifest + loader checks |
| Near-well odd–even oscillation (found now) | 9 turning points per profile, relaxed stepping | strict CFL: 1 turning point; QoIs unchanged (< 0.01%) |
| Rate floor of 1 kg/s distorted the schedule design (found now) | 159–162 of 300 scenarios clipped, 50–61 flat at 1 kg/s, 16 duplicates | floor 0.01 kg/s; 0 clipped, 0 duplicates; duplicates would be dropped |

Full list: `docs/CHANGELOG.md`. Components outside the supplied material
were removed or replaced (`docs/SOURCE_MAP.md`).

## 3. Simulator verification (18/18 checks pass; `validation.json`, 238 s)

| Check | Reference | Result |
|---|---|---|
| V1 steady radial pressure | `p_e + qμ/(2πkh) ln(r_e/r)` | rel. error ≤ 5e-13 for `r_near` = 0.15, 5, 20 m |
| V2 well equation BHP | analytic `p(r_w)` | rel. error ≤ 5e-13 |
| V3 closed tank | `p_i + qt/(c_t V_p)` | rel. error 6e-13 |
| V5 Buckley–Leverett | Welge: `S_gf = 0.362`, `f_gf = 0.792`, `M = 4.36` | front error 1.2% / 0.9% / 0.5% and L1 error 0.012 / 0.006 / 0.003 at 100 / 200 / 400 cells (first order) |
| V6 radial two-phase | mass balance, bounds, clipping, oscillation | mass error 3e-15; no clipping; 1 profile turning point |
| V7 convergence (60 cells, 5 m block, reference case) | refinement | `n_r` 30/60/120: `Δp` 5.63/5.63/5.62 MPa, `r_95` 981/925/899 m; `r_near` 20/10/5 m: `Δp` 6.99/6.31/5.63 MPa; `max ΔS` 0.1/0.05/0.02: `Δp` 5.60/5.63/5.64 MPa |
| V8 strict vs relaxed stepping | same QoIs | differences ≤ 1.3e-5; strict needs 2.6× the steps |
| V9 upscaling | `2-Upscaling.pdf` p.14 chain; exact series/parallel limits | limits exact to 1e-16; `K_H ≤ K_HA ≤ K* ≤ K_AH ≤ K_A` for σ_lnk = 0.5, 1, 1.75 and the 2×2 layout (`K* = 1.4917` in `[1.4889, 1.5]`) |
| V10 upscaling a two-phase problem | 4 layers vs 1 upscaled layer | `Δp` +7%, **threshold plume radius −39%**, sweep +10%, mass exact, 4.7× faster |
| V11 multi-layer well | common BHP, Σq = Q | spread 0 Pa; rate error 2e-14; shut-in rates exactly 0; 80% of the rate in the 500 mD layer at the end of injection |

**Numerical error that matters for interpretation.** Two discretisation
choices of the demo are *not* converged:

* the **well block** (`r_near` = 10 m in the demo): `Δp_bh,max` is 12% higher
  than with 5 m in the V7 case (and 15% higher than with 2.5 m in the study
  configuration's extra level) — the coarse block overstates the near-well
  pressure drop;
* the **radial grid** (`n_r` = 40): `r_95` changes by 3% from 60 to 120 cells
  and 9% from 30 to 120.

The study configuration's extra levels confirm the trends: `r_near` = 2.5 m
gives `Δp` = 5.48 MPa (−2.6% vs 5 m, still decreasing), and `n_r` = 240 gives
`r_95` = 889 m (−1.1% vs 120). Both errors enter the error-source table
(§6). The study configuration (`r_near` = 5 m, `n_r` = 60) reduces them. Surrogate errors below are errors
against *this* simulator; they are smaller than the simulator's own
well-block discretisation error for pressure.

## 4. Data

100 reservoirs × 3 schedules = 300 simulations, 0 failures, 103 s wall on 2
cores (200 s CPU, 0.67 s per simulation). Reservoir inputs by Monte Carlo
(`5-Uncertainty.pdf` p.42–45): median k 30–1000 mD (log-uniform), `V_DP`
0.05–0.85, φ 0.10–0.28, h 20–80 m, `r_e` 1.2–4 km, Corey exponents, `k_rg⁰`
0.2–0.65, `S_ar` 0.15–0.35, `μ_g` 0.04–0.08 cP, `ρ_g` 600–780 kg/m³ — assumed
generic values. Schedules: four 1.25-year periods, then 5 years shut-in;
level = multiple (0.15–4, log-uniform) of a closed-form reference rate
(harmonic combination of the steady radial injectivity and the
compartment-storage rate for 5 MPa), times a shape with front/back loading
and bow. Data quality: no duplicates, no missing values, no rate on the
floor or ceiling, mass balance < 1e-9, no clipping, one BHP in every run.

## 5. Surrogates (unseen reservoirs)

Split by reservoir: 56 training, 19 calibration, 25 test reservoirs (168 / 57
/ 75 rows). Eleven course families plus a mean-value baseline, each a
pipeline (median imputer, standard scaler, model), tuned by randomised search (12 draws) inside grouped
4-fold CV; pressure and plume radius fitted on log scale.

| Target | Selected | Test MAE | Test RMSE | R² | Max error | Mean baseline RMSE | P5–P95 band coverage |
|---|---|---|---|---|---|---|---|
| `Δp_bh,max` | lasso | 0.41 MPa | 0.83 MPa | 0.970 | 3.6 MPa | 4.8 MPa | 95% (width 2.6 MPa) |
| `r_95` | linear (OLS) | 5.2 m | 7.6 m | 0.997 | 30 m | 142 m | 92% (29 m) |
| sweep | elastic net | 0.0014 | 0.0020 | 0.970 | 0.009 | 0.011 | 89% (0.005) |

*Why linear wins.* On the engineered inputs — log kh, log pore volume,
rate relative to the reference rate, dimensionless injected mass, fill
radius — the responses are close to log-linear; the polynomial study shows
quadratic terms over-fit. For pressure the top families are within ~10% on
the test set (gradient boosting 0.77 MPa, XGBoost 0.79, ridge 0.81,
elastic net 0.82, lasso 0.83); CV and test rankings agree less for pressure
(Spearman 0.80) than for plume radius and sweep (0.94–0.95). Twenty-five
test reservoirs cannot separate the top families.

*Where it fails.* Pressure errors grow with schedule aggressiveness (MAE
0.08 / 0.26 / 0.88 MPa for gentle / moderate / aggressive terciles) and
toward low permeability (0.78 / 0.29 / 0.11 MPa for low / mid / high k); the
worst 10% of cases carry 55% of the absolute error. The worst case,
`R0036_S+02` (40 mD, 32 m, 11.8 MPa simulated vs 8.2 predicted), is a
low-permeability unit where the near-well pressure drop is a large part of
the build-up; the log-linear surrogate under-predicts it. The same case is
the classifier's only miss (§7) and, in the study configuration, the
reservoir on which screened schedules violated the limit (§12).

*Error bands.* The residual-percentile band calibrated on the 19 calibration
reservoirs covered 95%, 92% and 89% of test cases for a nominal 90% on the
calibration set. This is a measurement on 25 reservoirs, not a guarantee.

## 6. Uncertainty (`5-Uncertainty.pdf` p.8–9, 32, 44–46)

Monte Carlo of the assumed prior through the pressure surrogate: P90 (low
case) 1.0 MPa, P50 3.9 MPa, P10 (high case) 14.3 MPa. Error sources
(quadrature only as a rough total):

| Target | Input (prior spread) | Surrogate (test RMSE) | Numerical (V7) | Shares input / surrogate / numerical |
|---|---|---|---|---|
| `Δp_bh,max` | 5.1 MPa | 0.83 MPa | 0.48 MPa (well block 10 vs 5 m, 12%) | 97 / 3 / 1% |
| `r_95` | 148 m | 7.6 m | 18 m (30 vs 120 cells, 9%) | 98 / 0.3 / 1.5% |
| sweep | 0.011 | 0.0020 | 0.0025 (30 vs 120 cells, 29%) | 92 / 3 / 5% |

The input spread dominates because the prior is wide; the relevant
comparison for trusting the surrogate is surrogate vs numerical error, and
for plume radius and sweep the simulator's own discretisation error is the
larger of the two. Model bias from omitted physics (buoyancy, dissolution,
trapping, `P_c`, crossflow, geomechanics) is listed, not quantified;
buoyant override is expected to be the largest.

## 7. Pressure-limit screen

Limit: `Δp > 9 MPa` (25% of training rows, 21% of test rows). On grouped
out-of-fold scores the random forest (average precision 0.990) edges
logistic regression (0.985), SVC and XGBoost; the single tree is weakest
(0.893). Imbalance handling inside the folds (class weights, over/under-
sampling) raises default-threshold recall from 0.96 to 0.98 at some
precision cost, without changing ROC-AUC. The recall-targeted threshold
(0.48) gives OOF recall 0.96 and precision 0.93. Test: ROC-AUC 0.988;
confusion matrix `[[59, 0], [1, 15]]` — the only miss is the regression
surrogate's worst case. The green/amber/red variant: the random forest
scores macro-F1 0.89 on test (amber recalled 3/4); banding the regression
surrogate's prediction does better (0.95). For this problem a good regression
surrogate plus a threshold is at least as useful as a dedicated classifier.

## 8. What drives the outcomes (model behaviour, not causality)

The selected pressure surrogate (lasso) is sparse: permutation importance is
dominated by the rate relative to the reference rate, followed by the
fill-radius ratio and front loading; SHAP on gradient boosting tells the same
story (rate multiple 0.57, dimensionless mass 0.16, fill ratio 0.09). Since
the reference rate combines injectivity (kh) and storage (pore volume ×
compressibility), this says that peak build-up is governed by how hard the
schedule pushes *relative to the reservoir's capacity*. Heterogeneity
(`V_DP`), `k_rg⁰` and Corey exponents matter little for peak pressure
(masking them changes the error by < 1%). They matter for the plume: its
surrogate leans on flow capacity and schedule shape, and the final-saturation
profiles fall into two regimes (k-means on PCA, silhouette 0.42; 4 PCs
explain 95%) — a compact, high-saturation plume and a lower-saturation,
farther-reaching one, the mobility-ratio effect of `4-CO2 BL.pdf` p.18.
Plume regime and pressure risk are nearly independent (AMI 0.27). V10 shows
why layering matters for the plume: upscaling four layers to one keeps
pressure within 7% but changes the threshold plume radius by 39%.

## 9. ML-method studies (training reservoirs only; log-pressure RMSE)

Learning curve: 0.26 → 0.12 by 54 rows, 0.10 at 126 (variance-limited).
Random row split vs grouped: only 0.7% optimistic here (§12: the sign
reverses in the study configuration). Scaling: unscaled KNN/SVR 1.6–2.2×
worse. Missing permeability: imputation 0.50 (5×) vs a model without it
0.126 — because the sampled rates encode permeability through the reference
rate. Collinearity filter at |r| > 0.75: removes kh, pore volume and the
rate multiple, 4.5× worse — use domain knowledge or regularisation. Feature
selection: Lasso-embedded (10 inputs) matches all 29; 5 inputs cost ~10%.
Polynomial terms: worse. Grid / random / Bayesian search at 12 candidates:
0.100 / 0.102 / 0.102 (Bayesian 2× slower). Normal equation vs scikit-learn:
7e-11; cond(XᵀX) ≈ 8e15 (mean rate ∝ planned mass). XGBoost early stopping at
~260 rounds. Voting lasso + elastic net + gradient boosting: 0.092 → 0.088.
Details: `docs/COVERAGE_MATRIX.md`.

## 10. Schedule screening (re-simulated)

Five unseen reservoirs; 3000 Monte Carlo candidates each; feasibility on the
upper band edge for both limits (build-up 9 MPa, `r_95` 400 m); 68% of
candidates predicted feasible on average; screening 0.14 s per reservoir.
Re-simulation of the best three screened schedules and of the best
constant-rate schedule: **0 of 15 screened schedules and 0 of 5 baselines
violated a limit; 0 failed runs**. Injected mass vs constant rate: −0.3%,
+3.0%, +4.3%, +2.8%, +0.9% (median **+2.8%**, 4/5 improved). Where the
screening constraint on pressure is active (reservoirs 8, 9, 10) the best
schedules are front-loaded (70–80% of the mass in the first half) and gain
3–4%: the rate-dependent near-well pressure drop is paid while the
compartment pressure is still low. Where the 400 m plume limit binds
(reservoirs 5 and 11, `r_95` ≈ 370–386 m) the plume radius is fixed by the
injected mass and no schedule shape adds mass (−0.3%, +0.9%). The screening
is conservative: re-simulated build-ups stay ~1.3 MPa below the 9 MPa limit
because feasibility was judged on the upper band edge. (The uncorrected run
reported +7.3%; that number is superseded.)

## 11. Cost (measured on this machine; not a general claim)

Timing setup: `time.perf_counter` in one Python process on a 2-core cloud
VM; surrogates with default scikit-learn/XGBoost threading; the simulator run
serially on 5 test scenarios. Final demo run:

| Quantity | Value |
|---|---|
| simulator, per case (serial, in-process) | 0.90 s (0.59 s mean inside the parallel dataset run) |
| surrogates, 3 QoIs, per case in a 75-case batch | 7.8e-5 s → **~11,600×** |
| surrogates, 3 QoIs, one case per call (median of 30) | 5.1 ms → **~176×** |
| one case end-to-end (raw inputs → features → 3 surrogates) | 17 ms → **~52×** |
| data generation (300 simulations, 2 workers) | 103 s wall, 200 s CPU — not amortised above |
| training + tuning of 11 families + baseline × 3 targets | 99 s — not amortised above |

The single-call figure is dominated by per-call overhead (pandas and
pipeline dispatch), not arithmetic; the batch figure is what a Monte Carlo or
screening loop sees. In the study configuration (finer grid, 5 m well block)
the simulator costs 5.5 s per case and the surrogates give ~74,000× /
~1,000× / ~300× (§12). The earlier ~10,900× / ~179× figures were measured on the
uncorrected code and are superseded.

## 12. The larger study configuration (`config/study.yaml`, measured 47.6 min)

220 reservoirs × 4 schedules = 880 simulations (0 failed; 24 min wall on 2
cores, 3.3 s per simulation with 60 cells and a 5 m well block), validation
with the full convergence levels (7.4 min), 124/41/55 train/calibration/test
reservoirs (496/164/220 rows).

| Target | Selected | Test RMSE (MAE) | R² | Max error | Band coverage |
|---|---|---|---|---|---|
| `Δp_bh,max` | SVR | 1.24 MPa (0.42) | 0.941 | 14.8 MPa | 88% |
| `r_95` | elastic net | 10.5 m (6.5) | 0.995 | 57 m | 91% |
| sweep | SVR | 0.0012 (0.0008) | 0.989 | 0.005 | 83% |

The median error is as small as in the demo, but with a converged well block
the pressure response of **low-permeability reservoirs** is harder to learn:
MAE by permeability tercile 0.78 / 0.24 / 0.21 MPa, and one case (45 mD,
32.4 MPa simulated vs 17.6 predicted) dominates the RMSE. The pressure screen
scores ROC-AUC 0.995 with confusion matrix `[[166, 3], [3, 48]]`.

Screening on 8 unseen reservoirs: 0 failed re-simulations; median
improvement over constant rate +1.4% (6/7 comparable reservoirs improved,
best +6.6%). **3 of 24 screened schedules and 1 of 8 baselines violated the
pressure limit after re-simulation** — all on reservoir 36 (40.5 mD, the
demo's worst case too). For all four schedules the surrogate predicted
7.6–7.7 MPa with an upper band edge of 8.9 MPa; the simulator gave 10.6–14.8
MPa, and the front-loaded proposals (78–82% of the mass in the first half)
were the worst. In this tight reservoir the rate-dependent near-well pressure
drop dominates the build-up, so the high early rate sets the peak; in most
reservoirs the accumulated-storage term dominates and front-loading helps
(§10). The surrogate learned the common case and is blind to schedule shape
here, and the band calibrated on other reservoirs does not cover it. This is exactly why proposals are
re-simulated: the surrogate screens, the simulator decides.

The random-vs-grouped split comparison is not stable across configurations
(demo: random split 0.7% optimistic; study: 19% *pessimistic*, driven by
which heavy-tailed cases land in five random test sets); the grouped split is
kept because it is the only one that measures unseen-reservoir performance.

Timing on the same machine: simulator 5.5 s per case serial; surrogates
7.4e-5 s per case batched (~74,000×), 5.2 ms single call (~1,000×), 18 ms
end-to-end (~300×); data generation 1460 s wall and training 623 s excluded.

## 13. Limitations

* **Physics scope**: no gravity (the dominant omission for CO₂ plumes), no
  dissolution, no residual trapping/hysteresis, `P_c = 0`, no crossflow,
  isothermal, radial symmetry. The model does not represent 3-D field
  behaviour.
* **Numerics**: demo well block and grid not converged (§3).
* **Data**: synthetic, assumed prior ranges; the schedule design ties rates
  to permeability; 100 reservoirs.
* **ML**: 25 test reservoirs; top model families not statistically
  separable; error-band coverage measured, not guaranteed; importances are
  model behaviour.
* **Decision limits** are stated inputs; no safety claim.
* **Not done**: field validation, deployment testing, history matching.
