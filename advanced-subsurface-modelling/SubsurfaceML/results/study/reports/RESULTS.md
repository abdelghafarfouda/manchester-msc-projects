# SubsurfaceML results - configuration `study`

Generated Sun Oct  4 08:02:11 2026 by `scripts/run_pipeline.py` from `metrics/summary.json`. Total wall time 66 min.

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
| Data | Development scenarios | **880 ok / 0 failed** | SYNTHETIC - produced by the in-repo IMPES simulator; no field data, no field validation |
| Data | final test scenarios | **400 ok / 0 failed** | 100 fresh reservoirs, seed 20261104, median k [30.0, 1000.0] mD |
| Data | shift scenarios | **240 ok / 0 failed** | 60 fresh reservoirs, seed 20261105, median k [10.0, 30.0] mD |
| Data | overlap between sets | **0 ids / 0 descriptions** | must be 0 / 0 |
| Discretisation error of dataset cases | peak build-up, production vs refined | **median 0.1 %, max 0.5 %** | 41 development cases re-simulated |
| Discretisation error of dataset cases | pressure-limit labels changed | **0/41** | 1 cases within 5 % of the limit |
| Discretisation error of dataset cases | Plume radius (95 % mass) | **median 2.7 %, max 4.1 %** |  |
| Discretisation error of dataset cases | Swept pore-volume fraction | **median 8.3 %, max 26.8 %** |  |
| Peak pressure build-up - test | revised (support-vector regression) | **RMSE 0.306 MPa** | MAE 0.125, R2 0.9962, worst under-prediction 2.7 MPa, P95 |error| 0.526; adaptive_conformal interval: case coverage 95.0 %, reservoirs fully covered 86.0 %, mean width 0.552 MPa |
| Peak pressure build-up - shift (10-30 mD) | revised (support-vector regression) | **RMSE 1.95 MPa** | MAE 0.605, R2 0.9066, worst under-prediction 15.1 MPa, P95 |error| 2.03; adaptive_conformal interval: case coverage 79.6 %, reservoirs fully covered 70.0 %, mean width 1.15 MPa |
| Peak pressure build-up - test | published approach, retrained (support-vector regression) | **RMSE 0.701 MPa** | MAE 0.298, R2 0.9802, worst under-prediction 6.65 MPa, P95 |error| 1.14; empirical interval: case coverage 90.5 %, reservoirs fully covered 85.0 %, mean width 1.5 MPa |
| Peak pressure build-up - shift (10-30 mD) | published approach, retrained (support-vector regression) | **RMSE 3.81 MPa** | MAE 1.53, R2 0.6433, worst under-prediction 31.2 MPa, P95 |error| 6.32; empirical interval: case coverage 46.7 %, reservoirs fully covered 23.3 %, mean width 1.37 MPa |
| Peak pressure build-up - test | published models as released (support-vector regression) | **RMSE 0.67 MPa** | MAE 0.303, R2 0.9819, worst under-prediction 5.51 MPa, P95 |error| 1.18; empirical interval: case coverage 89.8 %, reservoirs fully covered 83.0 %, mean width 1.5 MPa |
| Peak pressure build-up - shift (10-30 mD) | published models as released (support-vector regression) | **RMSE 3.76 MPa** | MAE 1.52, R2 0.6531, worst under-prediction 31.5 MPa, P95 |error| 6.42; empirical interval: case coverage 47.1 %, reservoirs fully covered 21.7 %, mean width 1.36 MPa |
| Peak pressure build-up - paired bootstrap | revised vs published approach test | **dRMSE -0.395 MPa** | 95 % CI [-0.634, -0.1847], resampling 100 reservoirs |
| Peak pressure build-up - paired bootstrap | revised vs published models test | **dRMSE -0.365 MPa** | 95 % CI [-0.5372, -0.2087], resampling 100 reservoirs |
| Peak pressure build-up - paired bootstrap | revised vs published approach shift | **dRMSE -1.86 MPa** | 95 % CI [-2.752, -0.9746], resampling 60 reservoirs |
| Peak pressure build-up - paired bootstrap | revised vs published models shift | **dRMSE -1.81 MPa** | 95 % CI [-2.7458, -0.9142], resampling 60 reservoirs |
| Plume radius (95 % mass) - test | revised (support-vector regression) | **RMSE 6.36 m** | MAE 3.49, R2 0.998, worst under-prediction 30.3 m, P95 |error| 11.5; adaptive_conformal interval: case coverage 98.0 %, reservoirs fully covered 97.0 %, mean width 20.8 m |
| Plume radius (95 % mass) - shift (10-30 mD) | revised (support-vector regression) | **RMSE 8.47 m** | MAE 4.89, R2 0.9915, worst under-prediction 12 m, P95 |error| 17.3; adaptive_conformal interval: case coverage 85.4 %, reservoirs fully covered 78.3 %, mean width 16.5 m |
| Plume radius (95 % mass) - test | published approach, retrained (support-vector regression) | **RMSE 11.3 m** | MAE 5.91, R2 0.9936, worst under-prediction 98.2 m, P95 |error| 23.5; empirical interval: case coverage 89.2 %, reservoirs fully covered 82.0 %, mean width 21.9 m |
| Plume radius (95 % mass) - shift (10-30 mD) | published approach, retrained (support-vector regression) | **RMSE 12.5 m** | MAE 6.87, R2 0.9815, worst under-prediction 96 m, P95 |error| 25.7; empirical interval: case coverage 79.6 %, reservoirs fully covered 66.7 %, mean width 16.1 m |
| Plume radius (95 % mass) - test | published models as released (elastic net) | **RMSE 13.8 m** | MAE 8.5, R2 0.9905, worst under-prediction 106 m, P95 |error| 26.5; empirical interval: case coverage 86.5 %, reservoirs fully covered 80.0 %, mean width 28.8 m |
| Plume radius (95 % mass) - shift (10-30 mD) | published models as released (elastic net) | **RMSE 13.8 m** | MAE 7.57, R2 0.9776, worst under-prediction 107 m, P95 |error| 23.7; empirical interval: case coverage 82.1 %, reservoirs fully covered 73.3 %, mean width 20.8 m |
| Plume radius (95 % mass) - paired bootstrap | revised vs published approach test | **dRMSE -4.95 m** | 95 % CI [-6.8266, -2.8875], resampling 100 reservoirs |
| Plume radius (95 % mass) - paired bootstrap | revised vs published models test | **dRMSE -7.47 m** | 95 % CI [-9.485, -5.4492], resampling 100 reservoirs |
| Plume radius (95 % mass) - paired bootstrap | revised vs published approach shift | **dRMSE -4.06 m** | 95 % CI [-7.2182, -0.1891], resampling 60 reservoirs |
| Plume radius (95 % mass) - paired bootstrap | revised vs published models shift | **dRMSE -5.31 m** | 95 % CI [-9.0338, -0.7885], resampling 60 reservoirs |
| Swept pore-volume fraction - test | revised (support-vector regression) | **RMSE 9.21e-04 -** | MAE 6.12e-04, R2 0.9921, worst under-prediction 0.00321 -, P95 |error| 0.002; adaptive_conformal interval: case coverage 92.2 %, reservoirs fully covered 78.0 %, mean width 0.00288 - |
| Swept pore-volume fraction - shift (10-30 mD) | revised (support-vector regression) | **RMSE 6.17e-04 -** | MAE 4.65e-04, R2 0.9908, worst under-prediction 0.00229 -, P95 |error| 0.00121; adaptive_conformal interval: case coverage 91.7 %, reservoirs fully covered 80.0 %, mean width 0.00206 - |
| Swept pore-volume fraction - test | published approach, retrained (support-vector regression) | **RMSE 9.21e-04 -** | MAE 6.12e-04, R2 0.9921, worst under-prediction 0.00321 -, P95 |error| 0.002; empirical interval: case coverage 88.5 %, reservoirs fully covered 66.0 %, mean width 0.00261 - |
| Swept pore-volume fraction - shift (10-30 mD) | published approach, retrained (support-vector regression) | **RMSE 6.17e-04 -** | MAE 4.65e-04, R2 0.9908, worst under-prediction 0.00229 -, P95 |error| 0.00121; empirical interval: case coverage 95.4 %, reservoirs fully covered 85.0 %, mean width 0.00261 - |
| Swept pore-volume fraction - test | published models as released (support-vector regression) | **RMSE 0.00103 -** | MAE 7.36e-04, R2 0.9901, worst under-prediction 0.00334 -, P95 |error| 0.00221; empirical interval: case coverage 84.2 %, reservoirs fully covered 56.0 %, mean width 0.00266 - |
| Swept pore-volume fraction - shift (10-30 mD) | published models as released (support-vector regression) | **RMSE 9.53e-04 -** | MAE 7.22e-04, R2 0.9781, worst under-prediction 0.00204 -, P95 |error| 0.00207; empirical interval: case coverage 87.9 %, reservoirs fully covered 68.3 %, mean width 0.00266 - |
| Swept pore-volume fraction - paired bootstrap | revised vs published approach test | **dRMSE 0 -** | 95 % CI [0.0, 0.0], resampling 100 reservoirs |
| Swept pore-volume fraction - paired bootstrap | revised vs published models test | **dRMSE -1.12e-04 -** | 95 % CI [-0.0002, -0.0], resampling 100 reservoirs |
| Swept pore-volume fraction - paired bootstrap | revised vs published approach shift | **dRMSE 0 -** | 95 % CI [0.0, 0.0], resampling 60 reservoirs |
| Swept pore-volume fraction - paired bootstrap | revised vs published models shift | **dRMSE -3.36e-04 -** | 95 % CI [-0.0005, -0.0002], resampling 60 reservoirs |
| Pressure-limit screen (test) | classifier / ROC-AUC | **xgboost / 1** | limit: build-up > 9 MPa |
| Pressure-limit screen (test) | recall / precision at chosen threshold | **0.967 / 0.989** | threshold from out-of-fold scores for recall >= 0.95 |
| Pressure-limit screen (test) | interval upper edge: missed exceedances | **0/90** | false alarms 3 |
| Cost | simulator per case (serial) | **5 s** | 5 test scenarios re-run in-process |
| Cost | surrogates, single case end to end | **0.0487 s** | speed-up 103x; batched 9.10e-05 s/case |
| Screening - final test | rom constant | **20/20 verified** | first simulated proposal violated: 13; no feasible found: 0; median verified mass vs simulator-only constant rate 0.0736 %; mean simulations 2.65 |
| Screening - final test | rom shaped | **20/20 verified** | first simulated proposal violated: 5; no feasible found: 0; median verified mass vs simulator-only constant rate 3.42 %; mean simulations 2.7 |
| Screening - final test | simulator constant | **20/20 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 0 %; mean simulations 3.4 |
| Screening - final test | surrogate published | **19/20 verified** | first simulated proposal violated: 1; no feasible found: 1; median verified mass vs simulator-only constant rate -9.56 %; mean simulations 3 |
| Screening - final test | surrogate verified | **20/20 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 2.88 %; mean simulations 3.25 |
| Screening - shift | rom constant | **10/10 verified** | first simulated proposal violated: 4; no feasible found: 0; median verified mass vs simulator-only constant rate 0.0232 %; mean simulations 2.5 |
| Screening - shift | rom shaped | **10/10 verified** | first simulated proposal violated: 2; no feasible found: 0; median verified mass vs simulator-only constant rate 11.1 %; mean simulations 2.4 |
| Screening - shift | simulator constant | **10/10 verified** | first simulated proposal violated: 0; no feasible found: 0; median verified mass vs simulator-only constant rate 0 %; mean simulations 3.2 |
| Screening - shift | surrogate published | **9/10 verified** | first simulated proposal violated: 2; no feasible found: 1; median verified mass vs simulator-only constant rate -11.8 %; mean simulations 3 |
| Screening - shift | surrogate verified | **10/10 verified** | first simulated proposal violated: 4; no feasible found: 0; median verified mass vs simulator-only constant rate 0.0232 %; mean simulations 2.5 |
| Physics checks | retained vs planned mass | **3.23e-13** | max relative difference, sealed boundary |
| Physics checks | open boundary retention (median) | **0.812** | 6/10 runs lost CO2 across r_e |
