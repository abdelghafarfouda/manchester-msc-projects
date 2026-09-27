# SubsurfaceML results - configuration `demo`

Generated Sun Sep 20 14:51:47 2026 by `scripts/run_pipeline.py` from `metrics/summary.json`. Total wall time 9.29 min.

All data are **synthetic**, produced by the simulator in this repository. No field data are used and no field validation is claimed. Pressure and plume limits are stated assumptions, not safety limits.

| Section | Metric | Value | Notes |
|---|---|---|---|
| Simulator verification | All checks | **PASS** | 18/18 checks pass (validation.json) |
| Simulator verification | steady radial pressure | **PASS** |  |
| Simulator verification | well index bhp | **PASS** |  |
| Simulator verification | closed tank | **PASS** |  |
| Simulator verification | buckley leverett front | **PASS** |  |
| Simulator verification | buckley leverett L1 | **PASS** |  |
| Simulator verification | bl no clipping | **PASS** |  |
| Simulator verification | radial mass balance | **PASS** |  |
| Simulator verification | radial no clipping | **PASS** |  |
| Simulator verification | radial saturation bounds | **PASS** |  |
| Simulator verification | no odd even oscillation | **PASS** |  |
| Simulator verification | time step control insensitive | **PASS** |  |
| Simulator verification | upscaling 2x2 bounds | **PASS** |  |
| Simulator verification | upscaling parallel exact | **PASS** |  |
| Simulator verification | upscaling series exact | **PASS** |  |
| Simulator verification | upscaling bounds random | **PASS** |  |
| Simulator verification | well common bhp | **PASS** |  |
| Simulator verification | well total rate conserved | **PASS** |  |
| Simulator verification | well shut in no crossflow | **PASS** |  |
| Dataset | Scenarios | **300 ok / 0 failed** | SYNTHETIC - produced by the in-repo IMPES simulator; no field data, no field validation |
| Dataset | Generation cost | **103 s wall, 200 s CPU** | mean 0.665 s per simulation |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: selected model | **lasso** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: MAE / RMSE / R2 | **0.409 / 0.826 MPa / 0.9702** | mean-baseline RMSE 4.81 MPa; n = 75 |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: P5-P95 error band | **94.7% measured coverage** | mean width 2.64 MPa; calibrated on 57 rows; no guarantee |
| Surrogates (unseen reservoirs) | r_plume_m95_m: selected model | **linear** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | r_plume_m95_m: MAE / RMSE / R2 | **5.21 / 7.64 m / 0.9971** | mean-baseline RMSE 142 m; n = 75 |
| Surrogates (unseen reservoirs) | r_plume_m95_m: P5-P95 error band | **92% measured coverage** | mean width 29.2 m; calibrated on 57 rows; no guarantee |
| Surrogates (unseen reservoirs) | sweep_efficiency: selected model | **elasticnet** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | sweep_efficiency: MAE / RMSE / R2 | **0.00138 / 0.00195 - / 0.9695** | mean-baseline RMSE 0.0112 -; n = 75 |
| Surrogates (unseen reservoirs) | sweep_efficiency: P5-P95 error band | **89.3% measured coverage** | mean width 0.0046 -; calibrated on 57 rows; no guarantee |
| Pressure-limit screen | selected / test ROC-AUC | **random_forest / 0.988** | limit dp > 9 MPa |
| Pressure-limit screen | recall / precision at chosen threshold | **0.938 / 1** | threshold chosen on out-of-fold scores for recall >= 0.95 |
| Pressure-limit screen | traffic light macro-F1 (test) | **0.89** | regression surrogate banded: 0.952 |
| Cost | simulator per case (serial) | **0.902 s** | 5 test scenarios re-run in-process |
| Cost | surrogate batched / single call / single end-to-end | **7.77e-05 / 0.00513 / 0.0174 s** | 3 QoIs; end-to-end includes feature building |
| Cost | speed-up batched / single / end-to-end | **1.16e+04 / 176 / 51.8** | excludes 103 s data generation and 99.3 s training |
| Schedule screening | median improvement vs constant rate | **2.76%** | 4/5 unseen reservoirs improved (re-simulated) |
| Schedule screening | screened schedules violating a limit | **0/15** | baselines violating: 0; failed re-simulations: 0 |
| Physics checks | retained vs planned mass | **4.10e-13** | max relative difference, sealed boundary |
| Physics checks | open boundary retention (median) | **0.809** | 7/10 runs lost CO2 across r_e |
