# Mapping table — course concept → project use → implementation → evidence

"V*n*" = checks in `results/<config>/metrics/validation.json`. The ML rows of
the 2026-09-20 version are in
`docs/archive/2026-09-20_course_coverage/COVERAGE_MATRIX.md`; components
added in the 2026-10 revision (some from outside the course material) are
listed with their sources in `docs/SOURCE_MAP.md` §4.

## Advanced Subsurface Modelling

| Course concept | Source (file, page) | Project use | Implementation | Evidence |
|---|---|---|---|---|
| Mass conservation, Darcy | `1-Transmissibility.pdf` p.3–5 | governing equations | `single_phase.py`, `impes.py` | V1 (steady radial, machine precision) |
| Rock/fluid compressibility, storage term | `1-Transmissibility.pdf` p.6, 15–16, 20 | pressure build-up in a sealed compartment | `impes._step` (`C = V_p c_t / Δt`) | V3 closed-tank balance |
| Transmissibility, harmonic face permeability | `1-Transmissibility.pdf` p.18–19 | every inter-cell flux | `grid.harmonic_k`, `impes` | V1, V5, `test_harmonic_face_permeability_matches_the_lecture_formula` |
| Dirichlet / Neumann boundaries | `1-Transmissibility.pdf` p.11–13 | `outer_bc = closed | constant_pressure`; rate-controlled well | `impes`, `single_phase` | V1 (Dirichlet), V3 (closed); open-boundary variant in the pipeline |
| Series radial flow (harmonic), cylindrical equation | `2-Upscaling.pdf` p.8; `4-CO2 BL.pdf` p.20 | radial geometry and exact radial transmissibility | `grid.RadialGrid` | V1, V2 |
| Two-phase continuity + Darcy per phase | `3-IMPES.pdf` p.2–3 | CO2–brine flow | `impes.py` | V5, V6 |
| kr curves; drainage vs imbibition | `3-IMPES.pdf` p.4; `4-CO2 BL.pdf` p.18 | drainage kr on normalised saturation (power law = project choice) | `fluids.RelPerm` | V5 |
| Upstream mobility | `3-IMPES.pdf` p.7–8 | all face mobilities | `impes._upstream` | V5 |
| IMPES: implicit pressure, explicit saturation | `3-IMPES.pdf` p.12–18 | the time stepping | `impes.TwoPhaseModel.run/_step` | V5–V8 |
| Well equation, constant-rate injector | `3-IMPES.pdf` p.16 | one injector in all layers, **one common BHP** (implicit coupling) | `impes._solve_pressure`, `grid.well_index_geom` | V2, V11, `tests/test_well_coupling.py` |
| IMPES applicability, small time steps | `3-IMPES.pdf` p.20 | strict local CFL + reject/retry | `impes.run` | V6 (no odd–even oscillation), V8 |
| Fractional flow; viscous/capillary/gravity terms | `4-CO2 BL.pdf` p.4–10; `3-Advanced BL.pptx` slides 4–5 | `f_g`, `df_g/dS_g`; capillary and gravity terms **omitted** (A5, A6) | `fluids.RelPerm.f_g` | V5 |
| Method of characteristics, Welge | `4-CO2 BL.pdf` p.11–17 | analytical benchmark | `fluids.welge_shock`, `fluids.bl_profile_1d` | V5 (front position, first-order L1 convergence) |
| End-point mobility ratio | `4-CO2 BL.pdf` p.18 | ML input `log10_mobility_ratio` (realisation's own `krg0`) | `RelPerm.endpoint_mobility_ratio`, `features.raw_inputs` | `tests/test_features.py` |
| CO2–brine two-shock structure | `4-CO2 BL.pdf` p.21–29 | **explained, not implemented** (immiscible model, A2) | — | stated limitation |
| Displaced brine volume = injected CO2 volume | `4-CO2 BL.pdf` p.30 | closed-form fill radius feature | `features.engineer` (`r_fill_est_m`) | permutation importance |
| Arithmetic/harmonic, HA/AH, bounding chain | `2-Upscaling.pdf` p.5–14 | analytical upscaling | `upscaling.k_*` | V9 |
| Flow-based upscaling | `2-Upscaling.pdf` p.15–16 | `K*` from a pressure solve with sealed sides | `upscaling.upscale_flow_based`, `lecture_2x2_case` | V9 (exact limits, bounds) |
| Does single-phase upscaling preserve two-phase answers? | `2-Upscaling.pdf` p.3–4 | 4-layer vs upscaled 1-layer | `validation.v10_upscaling_two_phase` | V10 |
| Variance, CV, Dykstra–Parsons | `5-Uncertainty.pdf` p.19–23 | sampled heterogeneity + ML input | `petrophysics.py` | `test_dykstra_parsons_recovers_the_target` |
| PDFs, Monte Carlo, P10/P50/P90 | `5-Uncertainty.pdf` p.42–46 | scenario sampling; prior propagation through the surrogate; candidate schedules for the screening | `scenarios.sample_realisations`, `uncertainty.monte_carlo_propagate`, `optimise.sample_candidates`, `screening` | `summary.json → uncertainty`, `screening` |
| Sources of uncertainty (input, model bias, numerical) | `5-Uncertainty.pdf` p.8–9, 27–32 | error-source table per target; numerical error now measured on dataset cases; model bias of the no-gravity assumption measured with the r–z model | `uncertainty.error_source_table`, `numerics`, `rz` | `09_uncertainty_sources.png`, `17_numerics_dataset_cases.png`, `19_model_form_error.png` |
