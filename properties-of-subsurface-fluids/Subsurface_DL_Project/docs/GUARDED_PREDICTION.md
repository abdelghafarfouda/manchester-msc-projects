# Guarded prediction: when the network is allowed to answer

`src/sfp/predict.py` is the public prediction path. Tests:
`tests/test_predict.py`. Worked cases: `python scripts/guarded_demo.py` →
`results/guarded/demo_cases.json`.

## 1. The problem it fixes

The network ends in a sigmoid, so it returns a number between 0 and 1 for
*any* nine inputs. That includes states it was never meant to see: a
single-phase mixture, a pressure far above the training data, a composition
that does not sum to one. The training data contain only two-phase states,
because the module's phase test was applied before the dataset was built. So
the network has never been asked whether a mixture splits, and its answer for
a single-phase state is meaningless — yet it looks plausible (§4).

## 2. Three gates, in order

```python
from sfp.predict import FlashSurrogate
s = FlashSurrogate()                       # default model: see §3
r = s.predict(z, p_psia, T_R)              # one state; predict_many() for a batch
r.FV, r.status, r.phase, r.method, r.domain_violations, r.model, r.message
```

1. **Input validation** — `validate_inputs`. Malformed input raises `InputError`
   with a one-line explanation. Nothing is normalised, reordered or converted.
   * The composition must be seven mole fractions in the order of the notes'
     table (pp. 14–15): `CO2, C1, C2, C3, C4, C5, C10`. Alternatively, give a
     mapping from exactly those names, with one value per component, or one
     sequence of values per component for a batch. A different number of
     components is refused. The message says that other mixtures, such as the
     notes' C1/nC10 binary, can be flashed with `sfp.flash`.
   * Only real numbers are accepted. Booleans, text, complex numbers, `None`
     and ragged batches are refused, not converted.
   * A stated `components_order` must equal the notes' order.
   * Values must be finite. Mole fractions must be non-negative and sum to one
     within 1e−6; they are **not** normalised.
   * Pressure and temperature must be positive and in **psia** and **degrees
     Rankine**. Any other unit is refused with the conversion to apply
     (`T[R] = T[°F] + 460`, the notes' convention).
2. **The course phase test** — `classify_phase`, with Wilson K-values (notes
   p. 4, p. 8, p. 11). Because Wilson's `K` is proportional to `1/p`,
   `Σ z_iK_i = p_b/p` and `Σ z_i/K_i = p/p_d`. The sums are compared with
   `Σ z_i` rather than with 1, which is the Rachford-Rice function at its two
   ends, `h(0)` and `h(1)` (p. 8). A composition accepted within the 1e−6
   tolerance is therefore classified exactly as its normalised counterpart, and
   it is still not normalised. Components with `z_i = 0` contribute nothing.
   Conditions so extreme that a present component's K-value overflows or
   underflows are refused.
   * `Σ z_iK_i < 1`: **liquid**, `F_V = 0`.
   * `Σ z_i/K_i < 1`: **vapour**, `F_V = 1`.
   * `|Σ z_iK_i − 1| ≤ 1e−9`: **bubble point**, `F_V = 0` (p. 12).
   * `|Σ z_i/K_i − 1| ≤ 1e−9`: **dew point**, `F_V = 1` (p. 13).
   * Both sums equal to one (within 1e−9): **indeterminate**, returned as
     `unsupported` with no value. The bubble and dew points then coincide:
     every `K_i` of a component present is 1, as for an (effectively) pure
     component at its own Wilson vapour pressure, and `p` and `T` do not fix a
     vapour fraction.
   * The network is never called for any of these.
3. **The training domain** — `TrainingDomain`, from
   `configs/prediction_domain.json` (§3). A two-phase state inside every range
   goes to the network (`method = network`). A two-phase state outside any
   range is handled according to `on_unsupported`:
   * `"solver"` (default): answered by the reference solver (bisection on
     Rachford-Rice), `method = reference_solver_fallback`;
   * `"status"`: `status = unsupported`, no value;
   * `"extrapolate"`: the network's answer, `method = network_extrapolation`,
     with a message beginning `EXTRAPOLATION`.

| `method` | what produced `FV` |
|---|---|
| `network` | the network, inside the training domain |
| `network_extrapolation` | the network outside its training domain, on request only |
| `single_phase` | the phase test: liquid (0) or vapour (1) |
| `phase_boundary` | the phase test: bubble point (0) or dew point (1) |
| `reference_solver_fallback` | the bisection solver of `sfp.flash`, not the network |
| `none` | no value (`status = unsupported`) |

Every result also carries `phase`, `in_training_domain`, the list of
`domain_violations`, the two phase-test sums, the Wilson `p_bubble_psia` and
`p_dew_psia`, the `model` used (only for network routes), and a message.

## 3. The training domain, checked against the data

`python scripts/derive_domain.py` derives the domain from the training split of
`data/flash_dataset.npz` and from `configs/experiment.json`, and checks the
limits suggested by the earlier review:

| variable | suggested limit | configuration | training rows | used |
|---|---|---|---|---|
| pressure | ≤ 2000 psia (extrapolation test to 4000) | 2–2000 psia training band; 2000–4000 psia held out for the extrapolation test | 2.10–1999.5 psia | **2–2000 psia** (confirmed) |
| temperature | 610–680 R | 610–680 R | 610.00–680.00 R | **610–680 R** (confirmed) |
| each mole fraction | about 0.004–0.87 | integer charges 1–40 allow 1/241 = 0.00415 to 40/46 = 0.870 | 0.0048–0.0060 (minima), **0.427–0.476** (maxima) | **the per-component training ranges** |

The suggested composition range is what the generator *allows*, not what the
network *saw*. The extremes need one component at 40 moles and the other six
at 1, which practically never happens: no training row has any mole fraction
above 0.476. The guard therefore uses the realised per-component ranges.
Pressure and temperature use the sampling window, which the continuous draws
fill. 180 test rows (9 mixtures) and 100 extrapolation rows fall outside these
composition ranges. The network's error on those test rows is no larger than
elsewhere ([`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) §5), so the check errs on
the side of caution.

**The checks are marginal.** Each variable is compared with its own range.
Passing all of them does not establish that a particular *combination* — say
a high C1 fraction with a low C10 fraction at a given pressure — was covered by
the training data. The checks reject what is certainly outside; they do not
certify what is inside.

**Default model.** `results/checkpoints/ffn_phys1_s2.pt`. It has the lowest
best validation data MSE (4.05e−05) of the six reported models
(`results/metrics/train_*.json`); no test data were used to choose it. Any
other checkpoint can be passed as `FlashSurrogate(checkpoint=...)`.

## 4. The worked cases

`results/guarded/demo_cases.json`; the last column is what the bare network
returns for the same nine inputs.

| case | status | phase | method | `F_V` | solver | bare network |
|---|---|---|---|---|---|---|
| notes p. 15 flash, 796.43821 psia, 180 °F | ok | two-phase | network | 0.1923 | 0.1917 | 0.1923 |
| p. 14 mixture at its bubble point, 150 °F | ok | bubble point | phase boundary | 0 | – | 0.0052 |
| p. 14 mixture at its dew point, 150 °F | ok | dew point | phase boundary | 1 | – | **0.8988** |
| p. 14 mixture at 2500 psia (liquid) | ok | liquid | single phase | 0 | – | 0.0000 |
| p. 14 mixture at 1 psia (vapour) | ok | vapour | single phase | 1 | – | **0.9246** |
| two-phase at 2667 psia, default | ok | two-phase | solver fallback | 0.0649 | 0.0649 | 0.0551 |
| the same, `on_unsupported="status"` | unsupported | two-phase | none | – | 0.0649 | 0.0551 |
| the same, `on_unsupported="extrapolate"` | ok | two-phase | network extrapolation | 0.0551 | 0.0649 | 0.0551 |
| p. 15 state at 700 R | ok | two-phase | solver fallback | 0.2547 | 0.2547 | 0.2649 |
| CO2-rich, `z_CO2 = 0.60` | ok | two-phase | solver fallback | 0.8231 | 0.8231 | 0.8115 |
| pure C1 at its Wilson vapour pressure, 640 R | unsupported | indeterminate | none | – | – | **0.0000** |
| C1/nC10 binary of pp. 16–18 | rejected | – | – | – | – | – |
| composition summing to 0.98 | rejected | – | – | – | – | – |
| temperature given in °F | rejected | – | – | – | – | – |

Two examples show the hazard. The bare network calls an all-vapour state
92 % vapour, and it gives a definite answer for a pure component at its
vapour pressure, where no vapour fraction is defined.

## 5. How it was checked

`tests/test_predict.py` holds 34 tests of this path; the whole suite has 48.
After the first version, an adversarial review was run. Four reviewers
covered input validation, phase logic, the domain and metadata, and the tests
themselves; a sceptic then re-ran every finding against the code. It
confirmed 16 defects, all fixed:

* named compositions holding per-state values were read transposed;
* booleans and complex numbers were converted instead of refused;
* the phase test compared with 1 rather than `Σ z_i`;
* underflowed K-values of absent components produced NaN;
* scalar pressures were not applied to every row of a batch in the domain
  check;
* several messages were imprecise;
* tests were not tied to the source of each route's value. Mutation testing
  showed that a network answer computed by the solver, or a batch with
  misaligned rows, would still have passed.

Tests for each of these are now included.

## 6. The notes' binary example

The C1/nC10 binary of notes pp. 16–18 verifies the **reference calculation**
(`scripts/verify_flash.py`, case C: the bisection reproduces both of the notes'
closed forms to 1e−9). It is **not** inside the network's domain: the network
takes the seven components of the p. 14 table in that order, and has never seen
a two-component mixture. The guarded predictor therefore refuses it. The
message points to `sfp.flash`, which solves it directly.
