# Review recommendations — status for SubsurfaceML

**Source.** "Portfolio Quick Review" (English, 4 October 2026), supplied as a
PDF during this revision. The Arabic version of the review was not available.
No recommendation that appears only in it is claimed as addressed.

**Status key.**
* **Completed**: done in this repository and checked.
* **Remaining**: applies to SubsurfaceML but is not done.
* **Portfolio-wide**: outside this project (CV, other repositories, the
  author).
* **Future extension**: on the roadmap below, not started.

## Recommendations that concern SubsurfaceML

| # | Recommendation (review section) | Status | Evidence or note |
|---|---|---|---|
| 1 | Organise the project around one research question, not coverage of course topics (§4) | Completed | README *Supervisor overview*; the topic-coverage record is archived in `docs/archive/2026-09-20_course_coverage/` |
| 2 | Test on reservoirs outside the training distribution — the test reservoirs came from the training distribution (§4) | Completed | 60 lower-permeability shift reservoirs, scored once (`docs/TECHNICAL_REPORT.md` §9–10). Out-of-distribution performance is reported as a limit: RMSE 1.95 MPa, at most 73 % of reservoirs covered |
| 3 | Explain the screening failure (§4, §6) | Completed | The cause is that the inputs described reservoirs by their prior parameters, not the realised layers (§7). The hybrid ROM surrogate removes it; the larger search does not (§8) |
| 4 | Gravity, the main driver of plume migration, is omitted; so is crossflow (§4) | Completed as a measurement; training on that physics is a future extension | A separate r–z model quantifies the omission: peak build-up −4 %, plume radius +88 % (§5) |
| 5 | Capillary pressure and dissolution are omitted (§4) | Future extension | Stated as limitations in the README and `docs/ASSUMPTIONS.md` |
| 6 | All data are synthetic (§1, §4) | Remaining (inherent) | No field data is available to this project; stated as the first limitation everywhere |
| 7 | The targets are three scalars (§4) | Future extension | Unchanged; spatio-temporal targets are on the roadmap |
| 8 | Make the "880 cases" wording accurate (§5, §8.1) | Completed in the repository; CV: Portfolio-wide | The README states 880 generated cases and the published run's 124 / 41 / 55 reservoir split (496 / 164 / 220 cases), and the revision's split |
| 9 | Say that the GitHub versions were extended and re-verified in 2026 (§5, §8.1) | Completed in the repository; CV: Portfolio-wide | README *Version history and dates*; top-level index row |
| 10 | A GitHub Actions workflow that runs the existing tests (§8.1) | Completed for SubsurfaceML; other projects: Portfolio-wide | `.github/workflows/subsurfaceml-tests.yml` runs the 105 tests and the `subsurfaceml predict` example |
| 11 | A short "My contribution" statement beside the AI-use note (§1, §4, §8.5) | Remaining (needs the author) | It needs the author's own words and is not written here. The README's AI-assistance statement is accurate and kept |
| 12 | Precise attribution of each course and teaching resource (§8.5) | Completed | `docs/SOURCE_MAP.md` §1–3 (course material), §4 (outside sources with primary references) |
| 13 | Move COMPLETION_REPORT, COVERAGE_MATRIX and frozen hashes out of the main view (§3, §8.5) | Completed | `docs/archive/2026-09-20_course_coverage/` |
| 14 | Readable axis labels instead of raw variable names (§3, §8.5) | Completed | `figures.py` / `labels.py`; every figure regenerated |
| 15 | A one-screen summary for a busy reader (§3) | Completed | README *Supervisor overview* (about one page) |
| 16 | The documentation is long (§3) | Partly completed | The one-page overview is the entry point; the full README is still long (about 440 lines) |
| 17 | `pipeline.py` is one 48 kB file (§3) | Remaining (minor) | Plotting moved to `figures.py` and the stages are separate functions, but the file is still about 48 kB |
| 18 | A citable output, e.g. a preprint on when a calibrated surrogate fails outside its training range (§8.4) | Future extension | The repository provides the evidence; no manuscript has been started |

## Items found during the revision (not in the review)

| Item | Status | Evidence |
|---|---|---|
| The 41 interval-calibration reservoirs had taken part in choosing the design | Completed | The pipeline's coverage is now reported as measured only. A protocol addendum, pushed before the data existed, recalibrated the fixed design on 41 fresh reservoirs: 97 % / 91 % of test cases / reservoirs covered (`docs/EVALUATION_PROTOCOL.md` addendum, `results/study/experiments/calibration_check.json`) |
| "Calibration reservoirs: interval calibration only" was inaccurate | Completed | Corrected in the README and technical report: they also train the pressure-limit classifier and serve as the domain check's reference set |
| A reproduction of the ablation differed in one prediction by 4.36 MPa | Completed | The experiment inputs now come from one source. The re-run reproduces the record, and the tolerances are documented (`docs/TECHNICAL_REPORT.md` §8) |
| Peak build-up set by a well-block start-up transient in 32 of 240 shift cases | Documented; fix is a future extension | `results/study/experiments/peak_screen.json`. The error over-states the peak (conservative), and the test set is unaffected |

## Portfolio-wide (outside this project)

CV wording, dates and skill claims; repository links and referees on the CV;
the research-interest repository's README, licence and stray branch; a
trained deep-learning result on spatio-temporal CO₂ data; "My contribution"
statements for the other projects; unit tests for the heat-conduction
project; readiness to explain any line at interview.

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
8. The later research integration: connect computational mathematics, data
   science and machine learning, deep learning and seismic monitoring in one
   testable workflow. Deliberately not started in this project.
