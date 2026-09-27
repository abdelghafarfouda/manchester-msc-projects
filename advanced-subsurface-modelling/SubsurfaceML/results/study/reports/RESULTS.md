# SubsurfaceML results - configuration `study`

Generated Sun Sep 20 15:39:27 2026 by `scripts/run_pipeline.py` from `metrics/summary.json`. Total wall time 47.6 min.

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
| Dataset | Scenarios | **880 ok / 0 failed** | SYNTHETIC - produced by the in-repo IMPES simulator; no field data, no field validation |
| Dataset | Generation cost | **1.46e+03 s wall, 2.91e+03 s CPU** | mean 3.31 s per simulation |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: selected model | **svr** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: MAE / RMSE / R2 | **0.42 / 1.24 MPa / 0.9406** | mean-baseline RMSE 5.08 MPa; n = 220 |
| Surrogates (unseen reservoirs) | dp_bh_max_MPa: P5-P95 error band | **87.7% measured coverage** | mean width 1.55 MPa; calibrated on 164 rows; no guarantee |
| Surrogates (unseen reservoirs) | r_plume_m95_m: selected model | **elasticnet** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | r_plume_m95_m: MAE / RMSE / R2 | **6.53 / 10.5 m / 0.9945** | mean-baseline RMSE 142 m; n = 220 |
| Surrogates (unseen reservoirs) | r_plume_m95_m: P5-P95 error band | **90.9% measured coverage** | mean width 29.4 m; calibrated on 164 rows; no guarantee |
| Surrogates (unseen reservoirs) | sweep_efficiency: selected model | **svr** | lowest grouped-CV RMSE among the course model families |
| Surrogates (unseen reservoirs) | sweep_efficiency: MAE / RMSE / R2 | **8.08e-04 / 0.00116 - / 0.9885** | mean-baseline RMSE 0.0108 -; n = 220 |
| Surrogates (unseen reservoirs) | sweep_efficiency: P5-P95 error band | **83.2% measured coverage** | mean width 0.00266 -; calibrated on 164 rows; no guarantee |
| Pressure-limit screen | selected / test ROC-AUC | **random_forest / 0.995** | limit dp > 9 MPa |
| Pressure-limit screen | recall / precision at chosen threshold | **0.941 / 0.941** | threshold chosen on out-of-fold scores for recall >= 0.95 |
| Pressure-limit screen | traffic light macro-F1 (test) | **0.793** | regression surrogate banded: 0.89 |
| Cost | simulator per case (serial) | **5.45 s** | 5 test scenarios re-run in-process |
| Cost | surrogate batched / single call / single end-to-end | **7.39e-05 / 0.00524 / 0.0179 s** | 3 QoIs; end-to-end includes feature building |
| Cost | speed-up batched / single / end-to-end | **7.38e+04 / 1.04e+03 / 305** | excludes 1.46e+03 s data generation and 623 s training |
| Schedule screening | median improvement vs constant rate | **1.41%** | 6/7 unseen reservoirs improved (re-simulated) |
| Schedule screening | screened schedules violating a limit | **3/24** | baselines violating: 1; failed re-simulations: 0 |
| Physics checks | retained vs planned mass | **3.23e-13** | max relative difference, sealed boundary |
| Physics checks | open boundary retention (median) | **0.812** | 6/10 runs lost CO2 across r_e |
