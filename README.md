# MSc Subsurface Energy Engineering Projects — University of Manchester

This repository collects projects developed from my MSc studies in Subsurface
Energy Engineering at the University of Manchester. Each module has its own
folder, and each project sits in its own folder inside that module, keeping its
own documentation, dependencies and results so that it can be read and run on
its own.

Version note: The GitHub versions of these projects were extended and their results re-verified in 2026.

## Start here

Each project's README opens with a one-page **Supervisor overview**: the question, the method,
the main measured findings and their limits.

| Project | What it is | Overview |
|---|---|---|
| **Screening CO₂ injection with a verified surrogate**<br>`SubsurfaceML` | Surrogate-assisted screening of injection schedules for a sealed, layered saline aquifer, in which the simulator verifies every recommendation. Synthetic data. | [Supervisor overview](advanced-subsurface-modelling/SubsurfaceML/README.md#supervisor-overview) |
| **Heat conduction in an anisotropic plate**<br>`anisotropic-heat-conduction` | Implicit finite-difference solution of transient 2-D heat conduction, verified against exact and worked solutions. A numerical exercise from the module's assignment. | [Supervisor overview](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#supervisor-overview) |
| **A neural surrogate for the two-phase flash**<br>`Subsurface_DL_Project` | PyTorch networks trained to reproduce the course's Wilson / Rachford-Rice flash calculation for unseen mixtures. A synthetic benchmark. | [Supervisor overview](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#supervisor-overview) |
| **Gardner's density relation at a real well**<br>`SeisGeoMech` | Gardner's velocity–density relation is tested on the measured logs of UK well 48/10b-9, and its error is followed into a ρg gradient and synthetic seismic reflectivity. | [Supervisor overview](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#supervisor-overview) |

## Projects

| Module | Project | Summary |
|---|---|---|
| [Advanced Subsurface Modelling](advanced-subsurface-modelling) | [SubsurfaceML](advanced-subsurface-modelling/SubsurfaceML) | A radial IMPES simulator generates **synthetic** reservoirs. A hybrid surrogate (a reduced-order model with a learned correction) predicts peak pressure build-up on 100 test reservoirs generated after the design was frozen. It achieves an RMSE of 0.31 MPa, against 0.70 MPa for the published approach. Outside the training distribution the screening falls back to the simulator. There is no field validation, and the pressure and plume limits are modelling assumptions. |
| [Fundamentals of Numerical Modelling and Simulation](fundamentals-of-numerical-modelling-and-simulation) | [anisotropic-heat-conduction](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction) | Backward-Euler finite differences with five verification checks. After 5 h, the mean temperature on the assignment's 13 × 12 grid is 6.4 K above the 193 × 177 result, and grid and time step act in opposite directions. The checks verify the code; no measurements exist for this problem, and no error bound is claimed for the finest grid. |
| [Properties of Subsurface Fluids](properties-of-subsurface-fluids) | [Subsurface_DL_Project](properties-of-subsurface-fluids/Subsurface_DL_Project) | A **synthetic benchmark** built from the course's flash calculation. Test RMSE in vapour fraction is 0.00636 with the data loss and 0.00535 with the Rachford-Rice residual added. On pressure extrapolation, however, the residual made results worse in two of three seeds. Errors concentrate near the dew point, and a guarded predictor falls back to the reference solver outside the training domain. Three seeds; nothing is checked against measured phase behaviour. |
| [Subsurface Mechanics and Geoengineering](subsurface-mechanics-and-geoengineering) | [SeisGeoMech](subsurface-mechanics-and-geoengineering/SeisGeoMech) | Gardner's relation is tested on **measured** sonic and density logs: 1,105 paired samples over 221 m of UK well 48/10b-9. It under-predicts density by 3.5 %, giving a one-dimensional ρg gradient of 24.97 against 25.89 MPa/km. In a frozen within-well depth-block test, locally fitted coefficients predict the adjacent block better than the supplied relation, but the two blocks give different coefficients. The seismic results are synthetic. One well; the depth coordinate is not established as vertical. |

This index lists only the projects currently in the repository. Further module
projects will be added in the same module-folder / project-folder structure.

## Running a project

Each project is self-contained and is run from its own folder:

```bash
git clone https://github.com/abdelghafarfouda/manchester-msc-projects.git
cd manchester-msc-projects/<module-folder>/<project-folder>
```

Install that project's dependencies in a fresh virtual environment, as its README describes:
`requirements.txt` gives the packages and `requirements-lock.txt` the exact recorded versions.
Each quick check below takes at most a few minutes. On Windows, use the same commands in a
terminal such as an Anaconda Prompt.

| Project | Python | Quick check (in the project folder) | Installation and full instructions |
|---|---|---|---|
| SubsurfaceML | 3.11 (pinned versions) | `pytest` (107 tests, about 2 min) | [Reproduce](advanced-subsurface-modelling/SubsurfaceML/README.md#reproduce) |
| anisotropic-heat-conduction | 3.10 or later | `python run_project.py` (checks V1–V5 and the original study, about 1 min; exits with status 1 if a check fails) | [Run it](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#run-it) |
| Subsurface_DL_Project | 3.11 (as in its CI) | `python tests/run_tests.py` (48 tests) | [5. Reproduce and check](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#5-reproduce-and-check) |
| SeisGeoMech | 3.10 or later | `python -m pytest tests -q` (184 tests, about 5 s) | [Reproducing the results](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#reproducing-the-results) and [Run](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#run) |

Each project also has its own GitHub Actions workflow, which runs on changes to that project's
folder: [SubsurfaceML](.github/workflows/subsurfaceml-tests.yml) ·
[heat conduction](.github/workflows/heat-conduction.yml) ·
[flash surrogate](.github/workflows/subsurface-dl.yml) ·
[SeisGeoMech](.github/workflows/seisgeomech.yml).

## Structure

```
<module-folder>/
└── <project-folder>/
    ├── README.md         supervisor overview first, then method, results, limitations and how to run
    ├── LICENSE           the project's licence
    ├── requirements.txt  the dependencies for that project (requirements-lock.txt: exact versions)
    ├── docs/             detailed write-ups and the review checklist
    └── results/          the recorded tables, figures and run records
```

## Licences and attribution

All four projects carry the **MIT Licence** (© 2026 Abdelghafar Fouda). It covers each project's
own code and documentation, and in SeisGeoMech also its generated results. It does not cover
course teaching material, none of which is redistributed here: each project cites the teaching
material it relies on in its source map. The third-party Python libraries the projects use keep
their own open-source licences.

| Project | Licence | Data | Teaching sources | Details |
|---|---|---|---|---|
| SubsurfaceML | [MIT](advanced-subsurface-modelling/SubsurfaceML/LICENSE) | Synthetic, generated by the project's simulator. | CHEN60482 *Advanced Subsurface Modelling* materials (Dr Masoud Babaei; uncertainty material by Dr Lin Ma) and the *Data Science and Machine Learning* notebooks. Components from outside the course are listed with their primary sources. See [source map](advanced-subsurface-modelling/SubsurfaceML/docs/SOURCE_MAP.md). | [Author and attribution](advanced-subsurface-modelling/SubsurfaceML/README.md#author-and-attribution) |
| anisotropic-heat-conduction | [MIT](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/LICENSE) | None: all inputs come from the module's assignment. | Course materials for the equations, inputs, methods and benchmark values. See [source map](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/SOURCE_MAP.md). | [Author and attribution](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#author-and-attribution) |
| Subsurface_DL_Project | [MIT](properties-of-subsurface-fluids/Subsurface_DL_Project/LICENSE) | Synthetic, generated from the course notes' seven-component table. The notes credit that example to a Texas A&M PETE 310 spreadsheet. | CHEN60492 *Properties of Subsurface Fluids* notes (Dr Masoud Babaei), and the Deep Learning module's notebooks and *Introduction to Scientific Machine Learning* slides (Dr Ben Moseley). See [source map](properties-of-subsurface-fluids/Subsurface_DL_Project/docs/SOURCE_MAP.md). | [Author and attribution](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#9-author-and-attribution) |
| SeisGeoMech | [MIT](subsurface-mechanics-and-geoengineering/SeisGeoMech/LICENSE) for code, documentation and generated results | **Measured:** a public-domain UK continental-shelf well log (48/10b-9, operator BP), supplied with the module's practical and redistributed unmodified with its own attribution ([data note](subsurface-mechanics-and-geoengineering/SeisGeoMech/data/raw/README.md)). Two worked examples are transcribed from the exercise sheet. | University of Manchester EART35102 notes, exercises and practical notebooks. See [source map](subsurface-mechanics-and-geoengineering/SeisGeoMech/SOURCE_MAP.md). | [Licence and attribution](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#licence-and-attribution) |
