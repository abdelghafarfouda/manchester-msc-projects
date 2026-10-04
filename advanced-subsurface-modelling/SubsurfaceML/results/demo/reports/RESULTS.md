# SubsurfaceML results - configuration `demo`

Generated Sun Oct  4 08:16:52 2026 by `scripts/run_pipeline.py` from `metrics/summary.json`. Total wall time 11.7 min.

All data are **synthetic**, produced by the simulator in this repository. No field data are used and no field validation is claimed. Pressure and plume limits are stated modelling assumptions, not safety limits. 'test' = fresh reservoirs from the training prior, untouched until this run; 'shift' = fresh reservoirs with median permeability 10-30 mD, below the training range.

| Section | Metric | Value | Notes |
|---|---|---|---|
| Verification | All checks | **PASS** | 21/21 checks pass (validation.json) |
| Verification | steady radial pressure | **PASS** |  |
| Verification | well index bhp | **PASS** |  |
| Verification | closed tank | **PASS** |  |
| Verification | buckley leverett front | **PASS** |  |
| Verification | buckley leverett L1 | **PASS** |  |
| Verification | bl no clipping | **PASS** |  |
| Verification | radial mass balance | **PASS** |  |
| Verification | radial no clipping | **PASS** |  |
| Verification | radial saturation bounds | **PASS** |  |
| Verification | no odd even oscillation | **PASS** |  |
| Verification | time step control insensitive | **PASS** |  |
| Verification | upscaling 2x2 bounds | **PASS** |  |
| Verification | upscaling parallel exact | **PASS** |  |
| Verification | upscaling series exact | **PASS** |  |
| Verification | upscaling bounds random | **PASS** |  |
| Verification | well common bhp | **PASS** |  |
| Verification | well total rate conserved | **PASS** |  |
| Verification | well shut in no crossflow | **PASS** |  |
| Verification | pss bhp single layer | **PASS** |  |
| Verification | pss bhp two layers | **PASS** |  |
| Verification | rom matches pss limit | **PASS** |  |
| Data | Development scenarios | **300 ok / 0 failed** | SYNTHETIC - produced by the in-repo IMPES simulator; no field data, no field validation |
| Data | final test scenarios | **90 ok / 0 failed** | 30 fresh reservoirs, seed 20261204, median k [30.0, 1000.0] mD |
| Data | shift scenarios | **45 ok / 0 failed** | 15 fresh reservoirs, seed 20261205, median k [10.0, 30.0] mD |
| Data | overlap between sets | **0 ids / 0 descriptions** | must be 0 / 0 |
| Discretisation error of dataset cases | peak build-up, production vs refined | **median 0.1 %, max 0.6 %** | 13 development cases re-simulated |
| Discretisation error of dataset cases | pressure-limit labels changed | **0/13** | 0 cases within 5 % of the limit |
| Discretisation error of dataset cases | Plume radius (95 % mass) | **median 3.2 %, max 5.3 %** |  |
| Discretisation error of dataset cases | Swept pore-volume fraction | **median 8.1 %, max 21.7 %** |  |
| Peak pressure build-up - test | revised (support-vector regression) | **RMSE 0.528 MPa** | MAE 0.166, R2 0.9918, worst under-prediction 4.23 MPa, P95 |error| 0.644; adaptive_conformal interval: case coverage 100.0 %, reservoirs fully covered 100.0 %, mean width 1.69 MPa |
| Peak pressure build-up - shift (10-30 mD) | revised (support-vector regression) | **RMSE 0.636 MPa** | MAE 0.397, R2 0.985, worst under-prediction 1.83 MPa, P95 |error| 1.63; adaptive_conformal interval: case coverage 100.0 %, reservoirs fully covered 100.0 %, mean width 2.45 MPa |
| Peak pressure build-up - test | published approach, retrained (lasso) | **RMSE 1.65 MPa** | MAE 0.654, R2 0.9202, worst under-prediction 9.62 MPa, P95 |error| 4; empirical interval: case coverage 87.8 %, reservoirs fully covered 80.0 %, mean width 3.02 MPa |
| Peak pressure build-up - shift (10-30 mD) | published approach, retrained (lasso) | **RMSE 1.39 MPa** | MAE 0.887, R2 0.9286, worst under-prediction 3.4 MPa, P95 |error| 3.18; empirical interval: case coverage 64.4 %, reservoirs fully covered 40.0 %, mean width 2.58 MPa |
| Peak pressure build-up - paired bootstrap | revised vs published approach test | **dRMSE -1.12 MPa** | 95 % CI [-1.5102, -0.6719], resampling 30 reservoirs |
| Peak pressure build-up - paired bootstrap | revised vs published approach shift | **dRMSE -0.754 MPa** | 95 % CI [-1.0962, -0.4178], resampling 15 reservoirs |
| Plume radius (95 % mass) - test | revised (elastic net) | **RMSE 13.2 m** | MAE 8.01, R2 0.9928, worst under-prediction 24.3 m, P95 |error| 23; adaptive_conformal interval: case coverage 94.4 %, reservoirs fully covered 90.0 %, mean width 34 m |
| Plume radius (95 % mass) - shift (10-30 mD) | revised (elastic net) | **RMSE 9.21 m** | MAE 6.95, R2 0.9914, worst under-prediction 28.8 m, P95 |error| 15; adaptive_conformal interval: case coverage 88.9 %, reservoirs fully covered 73.3 %, mean width 24.1 m |
| Plume radius (95 % mass) - test | published approach, retrained (linear regression) | **RMSE 16 m** | MAE 8.82, R2 0.9894, worst under-prediction 74.7 m, P95 |error| 29.8; empirical interval: case coverage 92.2 %, reservoirs fully covered 86.7 %, mean width 33.8 m |
| Plume radius (95 % mass) - shift (10-30 mD) | published approach, retrained (linear regression) | **RMSE 12.7 m** | MAE 9.68, R2 0.9837, worst under-prediction 36.5 m, P95 |error| 26.3; empirical interval: case coverage 73.3 %, reservoirs fully covered 60.0 %, mean width 21.9 m |
| Plume radius (95 % mass) - paired bootstrap | revised vs published approach test | **dRMSE -2.81 m** | 95 % CI [-6.9758, 0.2424], resampling 30 reservoirs |
| Plume radius (95 % mass) - paired bootstrap | revised vs published approach shift | **dRMSE -3.44 m** | 95 % CI [-5.7099, -1.2331], resampling 15 reservoirs |
| Swept pore-volume fraction - test | revised (elastic net) | **RMSE 0.0017 -** | MAE 0.00123, R2 0.9781, worst under-prediction 0.00709 -, P95 |error| 0.00343; adaptive_conformal interval: case coverage 98.9 %, reservoirs fully covered 96.7 %, mean width 0.0112 - |
| Swept pore-volume fraction - shift (10-30 mD) | revised (elastic net) | **RMSE 0.00157 -** | MAE 0.00121, R2 0.9575, worst under-prediction 0.00219 -, P95 |error| 0.00301; adaptive_conformal interval: case coverage 97.8 %, reservoirs fully covered 93.3 %, mean width 0.00954 - |
| Swept pore-volume fraction - test | published approach, retrained (elastic net) | **RMSE 0.0017 -** | MAE 0.00123, R2 0.9781, worst under-prediction 0.00709 -, P95 |error| 0.00343; empirical interval: case coverage 91.1 %, reservoirs fully covered 73.3 %, mean width 0.00529 - |
| Swept pore-volume fraction - shift (10-30 mD) | published approach, retrained (elastic net) | **RMSE 0.00157 -** | MAE 0.00121, R2 0.9575, worst under-prediction 0.00219 -, P95 |error| 0.00301; empirical interval: case coverage 91.1 %, reservoirs fully covered 73.3 %, mean width 0.00529 - |
| Swept pore-volume fraction - paired bootstrap | revised vs published approach test | **dRMSE 0 -** | 95 % CI [0.0, 0.0], resampling 30 reservoirs |
| Swept pore-volume fraction - paired bootstrap | revised vs published approach shift | **dRMSE 0 -** | 95 % CI [0.0, 0.0], resampling 15 reservoirs |
| Pressure-limit screen (test) | classifier / ROC-AUC | **random_forest / 0.999** | limit: build-up > 9 MPa |
| Pressure-limit screen (test) | recall / precision at chosen threshold | **0.962 / 0.962** | threshold from out-of-fold scores for recall >= 0.95 |
| Pressure-limit screen (test) | interval upper edge: missed exceedances | **0/26** | false alarms 3 |
| Cost | simulator per case (serial) | **0.608 s** | 5 test scenarios re-run in-process |
| Cost | surrogates, single case end to end | **0.0431 s** | speed-up 14.1x; batched 1.04e-04 s/case |
| Screening - final test | rom constant | **6/6 verified** | first simulated proposal violated: 2; no feasible found: 0; median verified mass vs simulator-only constant rate 0.866 %; mean simulations 2.17 |
| Screening - final test | rom shaped | **6/6 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 2.72 %; mean simulations 2.33 |
| Screening - final test | simulator constant | **6/6 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 0 %; mean simulations 3.67 |
| Screening - final test | surrogate published | **6/6 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate -9.73 %; mean simulations 3 |
| Screening - final test | surrogate verified | **6/6 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 1.81 %; mean simulations 3.33 |
| Screening - shift | rom constant | **4/4 verified** | first simulated proposal violated: 1; no feasible found: 0; median verified mass vs simulator-only constant rate -0.0365 %; mean simulations 3 |
| Screening - shift | rom shaped | **4/4 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 10.6 %; mean simulations 2 |
| Screening - shift | simulator constant | **4/4 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 0 %; mean simulations 3.5 |
| Screening - shift | surrogate published | **4/4 verified** | first simulated proposal violated: 2; no feasible found: 0; median verified mass vs simulator-only constant rate -12.7 %; mean simulations 3 |
| Screening - shift | surrogate verified | **4/4 verified** | first simulated proposal violated: 1; no feasible found: 0; median verified mass vs simulator-only constant rate -0.0365 %; mean simulations 3 |
| Physics checks | retained vs planned mass | **4.10e-13** | max relative difference, sealed boundary |
| Physics checks | open boundary retention (median) | **0.809** | 7/10 runs lost CO2 across r_e |
