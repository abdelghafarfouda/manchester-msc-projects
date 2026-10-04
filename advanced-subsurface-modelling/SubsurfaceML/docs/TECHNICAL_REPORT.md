# Technical report — SubsurfaceML (revision of 2026-10-04)

**Research question.** In a sealed, layered saline aquifer, can a cheap
surrogate trained on simulations predict the peak bottom-hole pressure
build-up for reservoirs it has never seen — accurately enough, and with
honest enough uncertainty, to screen injection schedules against a pressure
limit, with the simulator confirming every recommendation?

**Status of the evidence.** All numbers are read from files written by the
code of this revision: `results/study/metrics/*.json` (pipeline,
`config/study.yaml`), `results/study/experiments/*.json` (development
experiments and model-form study). The data are **synthetic**. Verification
(is the equation solved correctly?), numerical error, model-form error and
surrogate error are kept separate. The design decisions were taken on
development reservoirs only, recorded in `docs/EVALUATION_PROTOCOL.md` and
pushed to GitHub (commit `561948e`) before the independent test reservoirs
were generated.

---

## 1. Contribution, and what is not claimed

The project is a reproducible case study. Its contribution is the evidence
chain, not a new method:

1. the published surrogate's large pressure errors are explained by a
   **problem-formulation error** — the inputs described each reservoir by
   the parameters of its sampling prior, not by the realised layers the
   simulator used;
2. a physics-based reduced-order model (ROM) with a learned multiplicative
   correction (a hybrid surrogate) removes most of that error on
   reservoirs generated after every modelling choice was fixed, while a
   three-times larger hyper-parameter search on the published inputs does
   not;
3. honest uncertainty needs calibration **by reservoir**: residual bands
   calibrated on pooled cases under-cover whole reservoirs, and no
   calibration survives a shift in the reservoir population. A
   finite-sample statement also needs calibration reservoirs that took no
   part in choosing the design. The pipeline's did, so its coverage is
   reported as measured, and a recalibration on fresh reservoirs is
   reported separately (§10);
4. a screening protocol in which the surrogate only proposes and the
   simulator must verify cannot present an unverified schedule as feasible,
   and states explicitly when it has no recommendation;
5. the error budget is quantified: discretisation error of the training
   data, model-form error of the layered no-gravity assumption, and
   surrogate error, on the same cases.

No novelty is claimed for the components: pseudo-steady-state well and tank
models, analytical pressure build-up in closed aquifers (Zhou et al., 2008;
Mathias et al., 2011), physics–ML hybrids (Willard et al., 2022) and
split-conformal prediction (Vovk et al., 2005; Lei et al., 2018; Dunn et al.,
2023) are established. Sources: `docs/SOURCE_MAP.md` §4.

## 2. Physical model (sources: `1-Transmissibility`, `3-IMPES`, `4-CO2 BL`)

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


## 3. Verification (21/21 checks pass; `metrics/validation.json`)

The 18 checks of the published version (steady radial pressure and the well
equation to ~1e-13, closed tank, Buckley–Leverett convergence, CO₂ mass
balance, oscillation detector, upscaling bounds, common bottom-hole pressure)
still pass. Three checks are new:

| Check | Configuration | Result |
|---|---|---|
| V12 | two-phase simulator, single-phase-equivalent fluids (equal viscosities and compressibilities, linear kr, zero residuals), one 100 mD layer, sealed, 3 years at 10 kg/s | bottom-hole build-up 25.15949 MPa vs PSS solution 25.15950 MPa (relative 4.3e-7) |
| V13 | the same with two commingled layers (500 mD / φ 0.25 and 20 mD / φ 0.08) sharing one bottom-hole pressure | 96.8706 vs 96.8818 MPa (1.2e-4, the not-yet-decayed inter-layer transient); late rate split 0.758 / 0.242 vs the pore-volume limit 0.758 / 0.242 |
| ROM in the same limit | `rom.bhp_buildup_layers` | 0.0 (one layer) and 1.2e-4 (two layers) relative; simulator and ROM agree to 3e-6 |

V1–V3 verified the *separate* single-phase solver; V12/V13 verify the
quantity the surrogates actually learn — the two-phase simulator's
bottom-hole pressure — including the well index, the units, the storage term
and the sealed boundary.

## 4. Numerical error of the training targets (`metrics/numerics_summary.json`)

V7 measures grid, well-block and time-step convergence on one verification
case. That does not show whether the discretisation error of the *training
targets* is small against the surrogate error, or whether it can change a
pressure-limit label. `numerics.py` answers this on the dataset cases: 41
development cases — three per permeability tercile × schedule-intensity
quartile, plus five cases singled out in §7 and in the diagnosis of the
published surrogate (R0172_S+00, R0036_S+02, R0039_S+03, R0150_S+03,
R0149_S+00) — were re-simulated at five levels: production; twice the radial
cells (`n_r_x2`); half the well-block radius (`r_near_half`); half the
saturation-change limit and a 30-day maximum step (`time_fine`); and all three
together (`fine`). Ten of them (cases with at most 12 000 production time
steps, spread by permeability) were also run at `finer` (four times the
cells, a quarter of the well-block radius and of the saturation-change limit,
15-day maximum step). All 215 runs succeeded (the refined runs need up to
553 000 time steps; each level has its own step guard), CO₂ mass balance holds
to 8e-14 in every run, and the production level reproduces the dataset values
exactly.

| Target | Production vs `fine` (41 cases): median / P90 / max relative | max absolute | `fine` vs `finer` (10): median / max | Production vs `finer` (10): median / max |
|---|---|---|---|---|
| Peak build-up | 0.06 % / 0.48 % / 0.54 % | 0.17 MPa (R0172_S+00, 32.4 MPa) | 0.03 % / 0.36 % | 0.07 % / 0.91 % |
| Plume radius r95 | 2.7 % / 3.5 % / 4.1 % | 20 m | 1.6 % / 1.9 % | 4.3 % / 5.3 % |
| Swept fraction | 8.4 % / 15 % / 27 % | 0.0058 | 4.2 % / 6.5 % | 17 % / 26 % |

Median relative change from refining one parameter at a time (41 cases):
twice the radial cells 0.07 % / 3.0 % / 8.4 % (build-up / plume radius /
swept fraction); half the well-block radius 0.02 % / 0.8 % / 4.0 %; finer
time step 0.00 % / 0.00 % / 0.00 % (at most 0.02 % / 0.13 % / 5.5 %). The
production level over-predicts the plume radius in all 41 cases and the
swept fraction in 39 (against `finer`: all ten, for both), and the
production-to-`fine` change is 1.9 (plume radius) and 2.6 (swept fraction)
times the `fine`-to-`finer` change on the same ten cases — consistent with
roughly first-order convergence in the cell size.

Readings:

* **Peak build-up**: the numerical error of the training data is at most
  about 1 % and its median is 0.06 %; no pressure-limit label changes (0 of
  41; one case within 5 % of the limit). It is negligible against the
  surrogate error (median relative error 1.1 %) and against the model-form
  error (§5).
* **Plume radius**: the production grid places r95 about 4 % too far out
  (median against `finer`; 3–5 %), more than the surrogate's median error. The surrogate reproduces the
  production-resolution simulator, not the converged solution.
* **Swept fraction**: the pore-volume fraction above a saturation threshold
  depends on the cell size; at production resolution it is about 17 % too
  large (median against `finer`; 5–26 %) and is not converged. Its surrogate (median relative error 5.3 %) is more
  accurate than its training data.

The time step contributes nothing measurable at production settings; the
radial cell size dominates.

**Screen of every case for a transient peak**
(`experiments/peak_screen.json`, `scripts/run_experiments.py --only
peak_screen`). A stratified sample of 41 cases cannot reveal a failure mode
that affects a few per cent of cases. V7 points to one: in that
verification case (a large compartment, 30 kg/s from the start) the peak
build-up depends strongly on the well-block size — 6.99 / 6.31 / 5.63 /
5.48 MPa for a 20 / 10 / 5 / 2.5 m block (`metrics/validation.json`) —
because it is a transient shortly after injection starts, while the well
block still holds brine of low mobility. In a sealed compartment the
build-up within a schedule period is otherwise largest at the period's end,
which the stored quarter-yearly series samples. So every case whose
time-step peak exceeds the largest stored value during injection by more
than 1 % was flagged (`numerics.startup_peak_screen`) and re-simulated at the
`production`, `r_near_half`, `n_r_x2` and `fine` levels. The final-test and
shift cases, generated without series, were re-simulated to record them;
this reproduced their stored targets exactly.

| Set | Flagged | Peak build-up at `fine` vs production | Halving the well block alone | Pressure-limit labels changed |
|---|---|---|---|---|
| Development | 4 of 880 (3 reservoirs; R0112, R0150, R0195) | −1.9 to −11.8 % (median −6.5 %) | median −5.8 % | 0 |
| Final test | 0 of 400 | — | — | 0 |
| Shift | 32 of 240 (20 of 60 reservoirs) | −1.7 to −13.9 % (median −11.3 %) | median −11.4 % | 1 (R20009_S+00, 9.18 → 8.68 MPa) |

Only one development case and no test or shift case lies between 0.1 % and
the 1 % flagging threshold, so the result does not depend on that threshold.
Doubling the radial cells changes the flagged peaks by at most 0.3 %; the
well block causes the error. In low-permeability reservoirs the near-well
pressure drop of the brine-filled well block at start-up is large relative
to the compartment build-up, which is why the shift set is affected and the
test set is not.

Consequences. (i) The reported test evaluation is unaffected. (ii) On the
shift set the targets of the flagged cases are 2–14 % too high; re-scoring
the revised surrogate against the refined targets changes its shift RMSE
from 1.952 to 1.944 MPa and its MAE from 0.605 to 0.584 MPa. Its shift
errors are therefore not a numerical artefact. (iii) The error always
over-states the peak. A schedule verified by the simulator at production
resolution is therefore checked conservatively, but the limit is binding
earlier than it would be at convergence for such reservoirs. (iv) A finer well
block, or a peak defined after the well block has filled, would remove the
artefact. It was not adopted because the evaluation design had been fixed;
changing the target after the test would invalidate the comparison.

## 5. Model-form error: gravity and vertical crossflow (`experiments/model_form_summary.json`)

The layered simulator omits buoyancy and crossflow. `rz.py` adds exactly those
two processes (radial–vertical grid, phase-potential upwinding, vertical
transmissibility with an **assumed** `k_v/k_h = 0.1`, five rows per layer, the
same well model and fluids) and reduces to the layered model when both are
switched off (build-up within 1e-5 relative on the dataset cases checked;
test tolerance 1e-4, `tests/test_rz.py`); it holds
hydrostatic equilibrium exactly and reaches the analytical end state of
gravity segregation (top zone at `1 − S_ar`, residual `S_gr` below, height
within one cell). Twenty-six development cases, stratified by permeability
and schedule intensity, were re-run:

| Change vs the layered model | Peak build-up | Plume radius r95 | Swept fraction |
|---|---|---|---|
| gravity + crossflow, median (P10–P90) | −4.3 % (−9.7 to −0.5 %) | **+88 %** (+19 to +213 %) | 0 % (−23 to +23 %) |
| crossflow only, median | −2.4 % | +10 % | +0.2 % |
| gravity + crossflow, 10 rows per layer (8 cases), median | −6.1 % | +101 % | −0.5 % |

Pressure-limit labels: 1 of 26 changes (an exceedance in the layered model
falls below the limit). Buoyancy carries 90 % of the CO₂ into the top quarter
of the formation (25 % without gravity). Two consequences:

* for **pressure**, the layered model is mildly conservative, and its
  model-form error (a few per cent) is now of the same order as the hybrid
  surrogate's error — further surrogate accuracy would not be meaningful
  without better physics;
* for **plume radius**, model-form error dominates every other error source
  (factor ~2, and the estimate grows with vertical resolution, so it is a
  lower bound): the plume surrogate, and the 400 m plume limit, describe the
  layered model only.

## 6. Data and evaluation design

Synthetic data only. Reservoir descriptions (11 parameters, assumed generic
prior ranges, `config/study.yaml`) and four-period injection schedules
(level × shape design scaled by a closed-form reference rate) are drawn by
Monte Carlo sampling and simulated for 5 years of injection and 5 years of
shut-in in a sealed compartment.

| Role | Reservoirs | Cases | Seed / ids | Use |
|---|---|---|---|---|
| Development — training | 179 | 716 | 20260909, ids 0–219 | fitting, tuning, model selection, all development experiments |
| Development — calibration | 41 | 164 | same | interval calibration (and, with the training reservoirs, the pressure-limit classifier, the domain check's reference set, the permeability-tercile edges and the prior Monte Carlo); they also took part in the development experiments that chose the design |
| **Final test** | **100** | **400** | 20261104, ids 10000+ | scored once |
| **Distribution shift** | **60** | **240** | 20261105, ids 20000+; median permeability 10–30 mD (training: 30–1000 mD) | scored once |
| Calibration check (after the evaluation; protocol addendum) | 41 | 164 | 20261106, ids 30000+; development prior | re-calibrating the fixed design's intervals only (§10) |

The development calibration reservoirs are those of the published run (same
split seed); its 55 former test reservoirs were inspected during development
and now train. No realisation id or reservoir description occurs in two
roles (asserted by the pipeline). All 1,520 simulations succeeded; CO₂ mass
balance holds to 3e-13 in every one.

**Leakage audit.** Every input is built by `features.py` from quantities
known before simulating — a test builds the full input table with the
simulator disabled; imputation and scaling are fitted inside every fold;
every cross-validation fold, nested or not, is grouped by reservoir; the
interval calibration uses only calibration reservoirs (a test perturbs test
targets and checks that the calibrated interval does not change); the final
and shift sets were generated after the decisions were pushed.

## 7. Why the published surrogate failed where it did

The published inputs described each reservoir by `k_median` and the target
`V_DP` of its sampling prior. The simulator uses four layers drawn from that
lognormal prior; with four draws, the realised layers can differ several-fold
from what the parameters imply. Writing `E[k] = k_median exp(σ²/2)`,
`σ = −ln(1 − V_DP)`, for the expected arithmetic mean:

| Development case | Simulated build-up | Published prediction | Realised / expected mean permeability |
|---|---|---|---|
| R0172_S+00 (worst under-prediction) | 32.4 MPa | 17.6 MPa | 0.30 |
| R0036_S+02 (the reservoir on which every screened schedule failed) | 11.7 MPa | 7.9 MPa | 0.31 |
| R0039_S+03 (worst over-prediction) | 13.1 MPa | 18.8 MPa | 12.2 |

On the out-of-fold predictions of the development reservoirs, the published
inputs give RMSE 2.12 MPa where the realised mean permeability is below half
the prior expectation (104 cases) and 2.06 MPa where it is more than twice
it (24 cases), against 0.87 MPa where it lies within a factor of two (752
cases). The realised
layers are generated deterministically from the realisation's seed before any
simulation, so using them as inputs is legitimate (`tests/test_evaluation_design.py`
builds every input with the simulator disabled).

## 8. The ROM, the hybrid surrogate, and the ablation

**ROM** (`rom.py`). Each realised layer is a sealed tank of pore volume `V_l`
with the PSS productivity index `J_l = 2πk_l h_l/(μ_a(ln(r_e/r_w) − 3/4))`; all
layers share one bottom-hole pressure with the simulator's injector-only rule;
the system is integrated with backward Euler (100 steps per schedule period,
converged to < 0.5 %). It uses no simulator output. On all 880 development
cases it gives RMSE 0.56 MPa on its own. A two-region variant with the CO₂
end-point mobility inside the volume-balance plume radius was *less*
accurate (RMSE 1.26 MPa) and was not used.

**Hybrid** (`hybrid.py`). `ln Δp = ln Δp_ROM + g(x)`; `g` is learned by the
same course model families, pipelines and grouped randomised search as
before.

**Ablation** (`experiments/ablation_summary.json`): nested cross-validation on
the 220 development reservoirs (outer 5-fold, inner 4-fold, both grouped by
reservoir), 20 search iterations per family, identical folds and seeds; the
bootstrap resamples reservoirs.

| Variant (peak build-up) | RMSE [MPa] | MAE | P95 abs. error | Worst under-prediction | ΔRMSE vs V0, 95 % CI |
|---|---|---|---|---|---|
| V0 published inputs | 1.138 | 0.471 | 1.93 | 15.85 | — |
| V1 + realised layers | 0.844 | 0.301 | 1.17 | 12.83 | −0.29 [−0.46, −0.11] |
| V2 + ROM as an input | 0.458 | 0.176 | 0.75 | 8.35 | −0.68 [−0.85, −0.50] |
| **V3 hybrid (ROM × learned factor)** | **0.266** | **0.103** | **0.42** | **4.64** | **−0.87 [−1.09, −0.66]** |
| V4 ROM alone (no learning) | 0.559 | 0.258 | 1.18 | 4.74 | −0.58 [−0.83, −0.33] |
| V5 published inputs, 3× search budget | 1.169 | 0.475 | 1.85 | 15.85 | +0.03 [−0.03, +0.13] |

Within the low-permeability tercile the RMSE falls from 1.62 (V0) to 0.36 MPa
(V3), and where the realised layers are less than half as permeable as the
prior suggests from 2.12 to 0.50 MPa. The larger search (V5) does not help:
the failure was not one of model capacity or tuning. The one case the hybrid
still misses by more than 3 MPa is R0172_S+00 (32.4 MPa simulated, the most
extreme build-up in the data set; 27.8 predicted), where the ROM itself is
4.7 MPa low and a single tail case cannot teach the correction.

For the plume radius the realised-layer and ROM inputs lower the out-of-fold
RMSE from 9.57 to 6.51 m (ΔRMSE −3.06 m, CI [−4.15, −2.04]); for the swept
fraction they do not help (0.00109 → 0.00112) and were not adopted.

**Reproduction and tolerances** (`experiments/ablation_reproduction.json`).
The documented command was re-run twice.

1. *Inputs read from the stored `scenarios_features.csv`.* The recorded run
   had rebuilt its inputs from `scenarios.csv`. The two tables agree only to
   2e-13 relative, because pandas' default CSV parser is not round-trip
   exact. Every decision was reproduced, and the plume, swept-fraction and
   ROM-only results agree to 1e-9. The learned pressure variants agree within
   0.014 MPa RMSE (V0: 1.138 → 1.124), but individual predictions do not:
   R0033_S+00 moved by **4.36 MPa**. In that outer fold the inner search
   chose ridge instead of SVR. Their inner-CV RMSEs differ by 0.05 %, and the
   tiny input change reversed their order. The ablation is
   model-selection-sensitive in this sense.
2. *Inputs pinned.* `experiments.load_dev_table` now always rebuilds the
   inputs from `scenarios.csv` with a named parser, and each run records a
   hash of its input table. Re-run this way, the out-of-fold predictions of
   all six pressure variants are bit-identical to the record. These are the
   only variants that had differed; the run was stopped before the plume and
   swept-fraction variants, which had already agreed to 1e-9.

Tolerances a re-run should meet:
* with the same inputs (same hash) and the pinned environment, identical
  results;
* with inputs that differ in the last digits (another source, platform or
  library build), identical design decisions and aggregate RMSE within
  about 1.5 %, while single predictions can differ by the gap between
  near-tied model families — several MPa, as observed;
* identity across platforms is not claimed.

The final evaluation depends on the ablation's decisions only, not on its
individual predictions.

## 9. Results on the untouched test reservoirs and under distribution shift

Three models are scored on the same reservoirs: the **revised** design, the
**published approach retrained** on the same 179 development reservoirs
(published inputs, log target, all families, empirical P5–P95 band
calibrated on the same 41 reservoirs), and the **published models as
released** (trained on 124 reservoirs, hash-checked).

| Target | Set | Model | RMSE | MAE | R² | P95 abs. error | Worst under-prediction |
|---|---|---|---|---|---|---|---|
| Peak build-up [MPa] | test | **revised (hybrid SVR)** | **0.306** | **0.125** | **0.996** | **0.53** | **2.70** |
| | test | published approach, retrained (SVR) | 0.701 | 0.298 | 0.980 | 1.14 | 6.65 |
| | test | published models as released (SVR) | 0.670 | 0.303 | 0.982 | 1.18 | 5.51 |
| | shift | revised | 1.95 | 0.605 | 0.907 | 2.03 | 15.1 |
| | shift | published approach, retrained | 3.81 | 1.53 | 0.643 | 6.32 | 31.2 |
| Plume radius r95 [m] | test | **revised (SVR, new inputs)** | **6.36** | **3.49** | **0.998** | **11.5** | **30.3** |
| | test | published approach, retrained | 11.3 | 5.91 | 0.994 | 23.5 | 98.2 |
| | shift | revised | 8.47 | 4.89 | 0.992 | 17.3 | 12.0 |
| | shift | published approach, retrained | 12.5 | 6.87 | 0.982 | 25.7 | 96.0 |
| Swept fraction [-] | test | revised = published approach (same inputs and model by rule 2) | 0.00092 | 0.00061 | 0.992 | 0.0020 | 0.0032 |

Mean-value baselines on the test set: 4.99 MPa, 142 m, 0.0104.

**Paired reservoir-level bootstrap** (2000 resamples; negative = revised
better):

| Target, set | ΔRMSE vs published approach retrained (95 % CI) | ΔMAE (95 % CI) | vs published models: ΔRMSE |
|---|---|---|---|
| Peak build-up, test | −0.395 MPa [−0.634, −0.185] | −0.173 [−0.247, −0.116] | −0.365 [−0.537, −0.209] |
| Peak build-up, shift | −1.86 MPa [−2.75, −0.97] | −0.92 [−1.32, −0.58] | −1.81 [−2.75, −0.91] |
| Plume radius, test | −4.95 m [−6.83, −2.89] | −2.42 [−3.45, −1.48] | −7.47 [−9.49, −5.45] |
| Plume radius, shift | −4.06 m [−7.22, −0.19] | −1.98 [−3.77, −0.32] | −5.31 [−9.03, −0.79] |

The protocol's criterion is met: on the final test set the revised pressure
surrogate has both a lower RMSE and a lower worst under-prediction than the
published approach retrained on the same reservoirs, and the bootstrap
interval of the RMSE difference excludes zero. The published approach scores
better here (0.70 MPa) than in its development cross-validation (1.14 MPa):
this test set happens to contain fewer extreme cases; the comparison between
models on the same reservoirs is unaffected.

**Subgroups (peak build-up, test).** Low-permeability tercile (edges from
the development reservoirs): RMSE 0.47 (revised) vs 1.03 MPa (published
approach); realised permeability below half the prior expectation (52 cases):
0.35 vs 1.25 MPa; cases within 20 % of the assumed limit (47): 0.30 vs 0.68
MPa. The revised surrogate's worst case on the test set is R10029_S+02
(37 mD; 21.6 MPa simulated, 18.9 predicted).

**Pressure-limit decisions (test, 90 exceedances of the assumed 9 MPa
build-up among 400 cases).** Revised point prediction: 0 missed, 0 false
alarms; published approach: 2 missed, 2 false alarms. Under shift (51
exceedances among 240): revised 6 missed with the point prediction, 2 with
the interval's upper edge; published approach 14 and 13.

**Under distribution shift** every model degrades — the revised pressure
surrogate's worst under-prediction rises from 2.7 to 15.1 MPa — which is why
the screening does not use it there (§11): the domain check flags all 60
shift reservoirs and 4 of the 100 test reservoirs.

## 10. Intervals

Nominal 90 %, two-sided, calibrated on the 41 calibration reservoirs of the
development set. The selected method (`adaptive_conformal`, chosen on
development data) is shown first; the others are reported for comparison
(peak build-up). These are the original results; the recalibration on fresh
reservoirs follows below.

| Model | Set | Method | Case coverage | Reservoirs fully covered | Mean width [MPa] | Missed exceedances with the upper edge |
|---|---|---|---|---|---|---|
| revised | test | **adaptive conformal** | **95 %** | **86 %** | 0.55 | 0 / 90 |
| revised | test | reservoir conformal | 92 % | 85 % | 0.56 | 0 / 90 |
| revised | test | case conformal | 88 % | 78 % | 0.42 | 0 / 90 |
| revised | test | empirical P5–P95 | 87 % | 77 % | 0.40 | 0 / 90 |
| published approach | test | empirical P5–P95 (published method) | 90 % | 85 % | 1.50 | 0 / 90 |
| revised | shift | **adaptive conformal** | **80 %** | **70 %** | 1.15 | 2 / 51 |
| revised | shift | reservoir conformal | 62 % | 47 % | 0.54 | 5 / 51 |
| revised | shift | empirical P5–P95 | 55 % | 33 % | 0.38 | 5 / 51 |
| published approach | shift | empirical P5–P95 | 47 % | 23 % | 1.37 | 13 / 51 |

Readings:

* On the test reservoirs the revised intervals are about **three times
  narrower** than the published band (0.55 vs 1.50 MPa mean width) at
  similar whole-reservoir coverage.
* **Pooled calibration under-covers whole reservoirs**: the empirical and
  case-conformal intervals cover 87–88 % of cases but only 77–78 % of
  reservoirs entirely, because a reservoir's schedules share their errors.
  The reservoir-level constructions reach 85–86 %.
* 86 % whole-reservoir coverage is below the nominal 90 %. With 100 test
  reservoirs the binomial standard error is about 3 percentage points. These
  intervals carry no finite-sample guarantee (next subsection), so the
  measured value is the claim.
* **Under shift no construction holds its level** (47–80 % of cases). The
  adaptive interval degrades least because its width grows where the
  training residuals were large. The conformal argument applies to
  reservoirs drawn like the calibration reservoirs, and the shift set is not
  such a population; this is the reason for the domain check.

Plume radius (test): adaptive conformal 98 % of cases, 97 % of reservoirs
(mean width 20.8 m); shift 85 % / 78 %. Swept fraction (test): 92 % / 78 %
(adaptive) against 88 % / 66 % for the empirical band with the same model.

### Calibration independence (protocol addendum; `experiments/calibration_check.json`)

The 41 calibration reservoirs belonged to the 220 development reservoirs on
which the design and the interval method were chosen (rules 1–3 of
`docs/EVALUATION_PROTOCOL.md`). Their out-of-fold residuals entered the
ablation RMSEs and the 200 calibration/evaluation splits of rule 3. The
surrogates and difficulty models never saw them, but the choice of score
function depended on their outcomes. The split-conformal argument therefore
does not apply to the intervals above: their coverage is **measured, not
guaranteed**. The test and shift sets were generated after every choice, so
the numbers above remain valid held-out measurements of the procedure as
run.

As fixed in the addendum before the data existed, 41 fresh reservoirs from
the development prior re-calibrated the saved design (seed 20261106, ids
30000+, 164 cases, all simulated, disjoint from every other set). Nothing was
refitted or re-selected. Scored once on the same test and shift sets, with
the selected method:

| Target | Calibration | Test: cases / whole reservoirs covered | Test mean width | Shift: cases / whole reservoirs covered |
|---|---|---|---|---|
| Peak build-up | original (41 development reservoirs) | 95 % / 86 % | 0.55 MPa | 80 % / 70 % |
| | **fresh (41 independent reservoirs)** | **97 % / 91 %** | 0.72 MPa | 84 % / 73 % |
| Plume radius r95 | original | 98 % / 97 % | 20.8 m | 85 % / 78 % |
| | fresh | 94 % / 88 % | 16.4 m | 78 % / 63 % |
| Swept fraction | original | 92 % / 78 % | 0.0029 | 92 % / 80 % |
| | fresh | 95 % / 84 % | 0.0033 | 93 % / 82 % |

Using the recalibrated upper edge, no exceedance of the assumed limit is
missed on the test set (0 of 90) and 2 of 51 are missed under shift, as
before. Re-scoring the saved models reproduces the pipeline's original
numbers exactly, which checks that the comparison is like for like.

Readings:

* **What the recalibrated intervals support.** Take a reservoir drawn from the
  development prior with four schedules from the same design. All four
  schedules are covered with probability at least 0.90, over the draw of the
  calibration reservoirs and the new reservoir. For one fixed set of 41
  calibration reservoirs the coverage varies: it follows Beta(38, 4), and 90 %
  of calibration sets give 0.82–0.97. Measuring on 100 test reservoirs adds
  about ±3 points. The measured 91 % (pressure), 88 % (plume) and 84 % (swept
  fraction) are consistent with this.
* **No evidence either way on bias.** The recalibrated pressure interval is
  30 % wider, and the plume interval 21 % narrower. With one calibration draw
  for each, these changes lie within the spread between calibration sets.
  The check does not show whether the original selection made the original
  calibration optimistic; it removes the dependence.
* **Outside the training distribution no calibration helps.** Under shift,
  63–82 % of reservoirs are fully covered whichever calibration is used. The
  conformal argument needs reservoirs drawn like the calibration reservoirs,
  and the shift set is not such a population.
* **Which interval the tool reports.** The pipeline, its saved models and
  `subsurfaceml predict` keep the original calibration (measured coverage
  only). The fresh calibration is an evaluation record and does not replace
  them.

## 11. Verification-gated schedule screening

Thirty reservoirs (the first 20 test and the first 10 shift reservoirs by
id), five methods, four simulator runs per reservoir and method (three for
the published method, as published), 4000 candidate schedules from the
training design. The assumed limits are a 9 MPa build-up and a 400 m plume
radius. Mass is compared with the simulator-only constant-rate search on the
same reservoir.

| Method | Test: verified | first simulated proposal violated | median (range) mass vs constant rate | Shift: verified | violated | median mass vs constant |
|---|---|---|---|---|---|---|
| simulator-only constant rate (baseline) | 20/20 | 0 | 0 % | 10/10 | 0 | 0 % |
| ROM-started constant rate | 20/20 | 13 | +0.1 % (−1.0 to +2.0) | 10/10 | 4 | +0.0 % |
| published screening (no repair) | 19/20 | 1 | **−9.6 %** (−18 to −0.5) | 9/10 | 2 | −11.8 % |
| ROM-ranked shapes + repair (control) | 20/20 | 5 | **+3.4 %** (−0.2 to +12.7) | 10/10 | 2 | **+11.1 %** |
| **revised: verified screening** | **20/20** | **0** | +2.9 % (−0.2 to +16.9) | 10/10 | 4 | +0.0 % (fallback) |

What this shows, and what it does not:

* **The gate works.** No method recommended a schedule that the simulator
  had not verified; every "verified" schedule met both limits. "Verified"
  means that the layered simulator, with its stated assumptions, keeps the
  schedule within the *assumed* 9 MPa and 400 m limits. It is not a check of
  real fracture or caprock integrity. The published
  method returned no verified schedule for one test and one shift
  reservoir — under the old code those reservoirs would have been reported
  as shortlisted but not recommended; here they get the explicit outcome
  `NO_FEASIBLE_SCHEDULE_FOUND`.
* **Schedule shape is worth a few per cent.** Verified schedules that vary
  the rate inject a median 3 % more than the best constant rate found with
  the same simulator budget (up to 17 %); the published method, which ranks
  by a wide band and does not repair, loses 10 %.
* **The learned surrogate is not what makes screening good.** Ranking
  candidate shapes by the analytical ROM alone and letting the simulator
  repair them gives the same median gain as the hybrid surrogate on the test
  reservoirs (+3.4 % vs +2.9 %). The hybrid's advantage is that its first
  proposal never violated a limit (0 of 20, against 5 of 20 for the ROM
  ranking) — the value of accurate predictions with calibrated intervals is
  in fewer wasted simulations and no unsafe first proposals, not in more
  injected mass.
* **Under shift the revised method behaves as designed but leaves mass
  unclaimed.** All 10 shift reservoirs were flagged out of domain, so the
  surrogate was not used and the method fell back to a ROM-started
  constant-rate search (+0.0 %). The ROM, which does not depend on the
  training population, would have found +11 % with shaped schedules. A
  ROM-ranked fallback is therefore the obvious next step; it was not
  adopted here because the method was fixed before the test, and changing
  it after seeing the result would invalidate the comparison.

## 12. Other analyses (classifier, interpretation, method studies)

* **Pressure-limit classifier** (revised inputs, XGBoost selected on grouped
  out-of-fold average precision): test ROC-AUC 0.9996; at the
  recall-targeted threshold 3 missed and 1 false alarm among 400 cases.
  Thresholding the revised regression surrogate does better (0 missed, 0
  false alarms), and its interval's upper edge gives 0 missed with 3 false
  alarms — a dedicated classifier adds nothing here.
* **Interpretation** (model behaviour, not causality). For the hybrid,
  permutation importance on the test reservoirs is dominated by the ROM
  input; SHAP on the best tree model of the *correction* `ln(Δp/ROM)` ranks
  the fill-radius ratio, the harmonic and arithmetic layer permeability,
  the CO₂ Corey exponent and the mobility ratio highest — the two-phase and
  heterogeneity effects the single-phase ROM leaves out.
* **Method studies** (development reservoirs, on the learned correction):
  the learning curve still falls slowly at 536 cases (grouped-CV RMSE of
  `ln(Δp/ROM)` 0.059 → 0.041); a random row split would look 14 %
  optimistic against the grouped split; voting or stacking the best three
  families is worse than the best single family (grouped-CV RMSE of the
  correction 0.031–0.033 against 0.026 for SVR).

## 13. Error budget

One case, one unit, side by side (peak build-up; median relative values):

| Source | Size | How measured |
|---|---|---|
| discretisation error of the training data | 0.06 % median, 0.54 % max (0.9 % against the finest level); no label changes | dataset cases, all discretisation refined (§4) |
| start-up transient set by the well block | 0 of 400 test cases; 4 of 880 development and 32 of 240 shift cases, 2–14 % too high | screen of every case (§4) |
| surrogate error, test reservoirs | 0.31 MPa RMSE (median relative error 1.1 %) | §9 |
| model-form error of the layered no-gravity model | −4.3 % median (to −15 %) | r–z model, assumed `k_v/k_h = 0.1` (§5) |
| surrogate error under distribution shift | 1.95 MPa RMSE | §9 |
| input uncertainty (spread of the assumed prior) | 5.1 MPa (standard deviation) | not reducible by the surrogate |

For plume radius the order is reversed: model-form error (median +88 %)
dwarfs the discretisation error (about 4 % against the finest level) and the surrogate error
(median relative error 1.1 %). For the swept fraction, the discretisation error of the
training data (8 % median against `fine`, 17 % against `finer`) is larger
than the surrogate error (5.3 % median), so that target is only as good as
the grid.

## 14. Cost

All times were measured on one 4-core virtual machine (Intel Xeon 2.1 GHz,
15 GB RAM), Python 3.11, using all cores (`n_jobs: -1`).

| Step | Compute budget | Wall time |
|---|---|---|
| Verification suite (21 checks, full V7 levels) | — | 7 min |
| Development data | 880 simulations | 11 min (42 min CPU) |
| Fresh test and shift data | 400 + 240 simulations | 6 min |
| Discretisation study of dataset cases (§4) | 215 simulations at refined levels | 17 min |
| Surrogates: three designs × three targets | randomised search over 10 model families, 40 settings each, 4-fold reservoir-grouped CV | 14 min |
| Method studies, interpretation, classifier, speed, open boundary | — | 4 min |
| Screening (§11) | 30 reservoirs × 5 methods × at most 4 simulations; 4000 candidates per reservoir | 6 min |
| **Full pipeline** (`run_pipeline.py --config config/study.yaml`) | | **66 min** |
| Development experiments (§8) | nested CV, 5 outer × 4 inner folds, 20 settings per family (V5: 60), 10 variant–target pairs; 200 calibration resamples | 71 min (83 min in the original run, which shared the machine with other jobs) |
| Model-form study (§5) | 26 cases × 2 r–z variants + 8 cases at double vertical resolution (60 r–z simulations) | 16 min |
| Screen for transient peaks (§4) | all 1,520 cases screened; 640 re-simulated with series; 36 flagged cases × 4 levels | 7 min |
| Calibration check (§10, protocol addendum) | 164 simulations (41 fresh reservoirs); intervals recalibrated, nothing refitted | 2 min |

**Inference.** One simulation takes 5.0 s (mean of five, one core, serial);
the surrogate pipeline takes 0.049 s for one case end to end (inputs,
analytical ROM and the three surrogates; median) — about 100 times faster —
and 9 × 10⁻⁵ s per case in a batch of 400. The surrogates return three
numbers; the simulator also returns fields and time series. In the screening
the simulator, not the surrogate, sets the cost: at most four simulations
per reservoir and method (mean 2.4–3.4), against 4000 surrogate
evaluations per reservoir.

## 15. Limitations and unresolved questions

* **Synthetic data, assumed prior.** No measurement constrains any number;
  the distribution-shift test changes one prior range only.
* **Physics.** No gravity or crossflow in the data-generating model — their
  effect is measured (§5) but not removed; no dissolution, residual trapping,
  capillary pressure, temperature effects or geomechanics; radial symmetry;
  one well. Plume-radius results describe the layered model only.
* **Grid dependence.** At production resolution the plume radius is about
  4 % and the swept fraction about 17 % too large (§4); their surrogates
  learn the production grid. The peak build-up is converged to within 1 %
  except where it is a start-up transient set by the well block (13 % of the
  low-permeability shift cases, 0.5 % of development cases, no test case):
  there it is 2–14 % too high.
* **Limits.** The 9 MPa build-up and 400 m plume-radius limits are modelling
  assumptions chosen to make the screening question bind. They are not
  validated fracture-pressure or caprock criteria, so a simulator-verified
  schedule is verified only against these assumptions.
* **Intervals.** The pipeline's intervals (also those `subsurfaceml
  predict` reports) have measured coverage only (§10). The recalibrated
  intervals' statement holds on average over calibration draws, only for
  reservoirs drawn like the calibration reservoirs, and only for their four
  sampled schedules. It does not transfer to the shift set or to thousands
  of screened candidates.
* **Domain check** sees only the reservoir descriptors it is given.
* **Tail.** The most extreme build-up case remains under-predicted by about
  15 %; the ROM's residual there is not learnable from one case.
* **Unresolved.** Whether the hybrid approach survives the move to physics
  with gravity and crossflow (where the plume, not the pressure, carries the
  model-form error); whether a two-phase near-well correction to the ROM
  would remove its tail residual; whether the small schedule-shape gains
  (§11) persist when the plume constraint is evaluated with buoyancy.

## References

* Dake, L.P. (1978). *Fundamentals of Reservoir Engineering*. Elsevier.
* Dunn, R., Wasserman, L. & Ramdas, A. (2023). Distribution-free prediction
  sets for two-layer hierarchical models. *JASA* 118(544), 2491–2502.
* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. & Wasserman, L. (2018).
  Distribution-free predictive inference for regression. *JASA* 113(523),
  1094–1111.
* Mathias, S.A., González Martínez de Miguel, G.J., Thatcher, K.E. &
  Zimmerman, R.W. (2011). Pressure buildup during CO₂ injection into a closed
  brine aquifer. *Transport in Porous Media* 89, 383–397.
* Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a
  Random World*. Springer.
* Willard, J., Jia, X., Xu, S., Steinbach, M. & Kumar, V. (2022). Integrating
  scientific knowledge with machine learning for engineering and
  environmental systems. *ACM Computing Surveys* 55(4).
* Zhou, Q., Birkholzer, J.T., Tsang, C.-F. & Rutqvist, J. (2008). A method for
  quick assessment of CO₂ storage capacity in closed and semi-closed saline
  formations. *Int. J. Greenhouse Gas Control* 2, 626–639.
* Course material (CHEN60482): `docs/SOURCE_MAP.md` §1–3.
