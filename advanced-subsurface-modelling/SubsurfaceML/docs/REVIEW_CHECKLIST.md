# October 2026 revision — record and open items (SubsurfaceML)

What the revision of 2026-10-04 changed, where the evidence is, and what
remains open. The full change log is [`CHANGELOG.md`](CHANGELOG.md); section
numbers below refer to [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

## Changes

| Change | Evidence |
|---|---|
| The project is organised around one research question (reliable, simulator-verified screening) instead of coverage of course topics | README *Supervisor overview*; the earlier topic-coverage record is archived in `docs/archive/2026-09-20_course_coverage/` |
| Out-of-distribution test on 60 lower-permeability reservoirs, scored once | §9–10: RMSE 1.95 MPa; at most 73 % of reservoirs fully covered by any interval |
| Cause of the earlier screening failure identified: the inputs described reservoirs by their prior parameters, not the realised layers | §7; the hybrid ROM surrogate removes the failure and a three-times larger search does not (§8) |
| Omitted gravity and vertical crossflow quantified with a separate r–z model | §5: peak build-up −4 %, plume radius +88 % (medians) |
| Data roles stated exactly: 880 generated cases, the first public run's 124 / 41 / 55 reservoir split (496 / 164 / 220 cases) and the current split | README *Data and evaluation design* and *Version history* |
| GitHub Actions workflow for the tests | `.github/workflows/subsurfaceml-tests.yml` (107 tests and the `subsurfaceml predict` example) |
| Course and outside sources attributed item by item | `docs/SOURCE_MAP.md` §1–3 (course material) and §4 (outside sources with primary references) |
| Readable figure labels and a one-page overview | `figures.py`, `labels.py`; README *Supervisor overview* |
| The 41 interval-calibration reservoirs had helped choose the design | Coverage is reported as measured only. A protocol addendum, fixed before the data existed, recalibrated the design on 41 fresh reservoirs: 97 % / 91 % of test cases / reservoirs covered (`docs/EVALUATION_PROTOCOL.md`, `results/study/experiments/calibration_check.json`) |
| "Interval calibration only" was inaccurate for the calibration reservoirs | Corrected in the README and the technical report: they also train the pressure-limit classifier and form the domain check's reference set |
| One prediction moved by 4.36 MPa between two sources of the experiment inputs | The inputs now come from one source; the pressure ablation is bit-identical to the record (§8, tolerances) |
| A malformed `subsurfaceml predict` input raised a traceback | One-line error and exit code 2, checked before any model loads (`tests/test_cli.py`) |

## Open items

| Item | Status |
|---|---|
| Capillary pressure, dissolution and residual trapping are omitted | Future extension; stated in the README and `docs/ASSUMPTIONS.md` |
| All data are synthetic | Inherent; no field data are available to this project |
| The targets are three scalars | Future extension (roadmap item 6) |
| `pipeline.py` is one 48 kB file | Partly addressed: plotting moved to `figures.py` |
| The peak is set by a well-block start-up transient in 32 of 240 shift cases | Documented (`results/study/experiments/peak_screen.json`); the error over-states the peak and the test set is unaffected; fix is roadmap item 5 |

## Future-work roadmap (not started)

1. Train the hybrid surrogate on data with gravity and vertical crossflow,
   using the verified r–z model.
2. Capillary pressure; dissolution and residual trapping.
3. Priors calibrated to a real formation, and validation against field or
   benchmark data.
4. A ROM-ranked fallback for out-of-domain reservoirs. The ROM-ranked control
   found +11 % on the shift reservoirs; it was not adopted after the test.
5. A finer well block, or a peak defined after the well block has filled, to
   remove the start-up transient.
6. Spatio-temporal targets (pressure and saturation fields) and a
   deep-learning surrogate for them.
7. A preprint built on items 1–6 or on the out-of-distribution result.
8. A later research integration connecting computational mathematics, data
   science and machine learning, deep learning and seismic monitoring in one
   testable workflow.
