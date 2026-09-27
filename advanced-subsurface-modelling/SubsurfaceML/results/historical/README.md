# Historical runs (superseded)

## `2026-09-11_pre-correction_demo/`

Results produced on 2026-09-11 by the original version of the project,
**before** the corrections of 2026-09-20. The author keeps them unchanged, in a
local copy only, for comparison; **the files of this run are not included in the
public repository, and none of its numbers should be quoted as results of the
current code.** This note records what the run was and why it is superseded.

Known problems with this run (details in `docs/CHANGELOG.md`):

* the multi-layer rate allocation did not enforce a common bottom-hole
  pressure, so `dp_bh_max` and the per-layer CO2 distribution of every
  multi-layer scenario are affected;
* the optimiser built its inputs with a fixed `krg0 = 0.4` instead of each
  reservoir's value, so its surrogate inputs differed from the training inputs;
* the absolute rate floor of 1 kg/s clipped the designed schedule of 159 of
  the 300 scenarios (61 were flat at 1 kg/s whatever their sampled level and
  shape);
* the relaxed time-step controller let odd-even saturation oscillations
  develop near the well (scalar QoIs were barely affected);
* the classifier's probability calibration used folds that were not grouped
  by realisation;
* several methods used are not in the supplied course material (conformal
  intervals, Latin-hypercube sampling, Theis benchmark, forecasting,
  isotonic calibration, Nelder-Mead refinement, retrieval assistant);
* the `models/` directory was not included, so its inference results could
  not be reproduced.

Its reported numbers (e.g. ~10,900x batched / ~179x single-call speed-up,
7.3% median optimisation improvement, test R2 0.93 / 0.98 / 0.97) are
therefore **historical and unverified against the corrected code**; they are
replaced by `results/demo/` (and `results/study/`).
