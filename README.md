# Personal Computational Projects — 2026

Four computational projects that I developed in 2026, informed by my MSc studies in Subsurface
Energy Engineering at the University of Manchester. Each project takes a topic, method or
dataset from an MSc module and develops it into a self-contained study with its own code,
documentation, verification and recorded results.

**What the published results are.** All code, analyses and results in this repository come
from the 2026 versions of these projects. They are not assessed coursework, and no module mark
refers to them.

**Titles and folders.** Each project is listed below by its displayed title, which its README
also uses. The name in code font is the project's folder, the path used in links and commands.
Each module folder is named after the MSc module whose teaching material the project draws on.

## Start here

Each project's README opens with a one-page **Supervisor overview**: the question, the method,
the main measured findings and their limits.

| Project | What it is | Overview |
|---|---|---|
| **Surrogate-Assisted Screening of CO₂ Injection Schedules with Simulator Verification**<br>`SubsurfaceML` | A radial CO₂–brine simulator and a hybrid surrogate screen injection schedules for a sealed, layered saline aquifer; the simulator verifies every recommendation. Synthetic data. | [Supervisor overview](advanced-subsurface-modelling/SubsurfaceML/README.md#supervisor-overview) |
| **Implicit Finite-Difference Modelling of Transient Heat Conduction in an Anisotropic Plate**<br>`anisotropic-heat-conduction` | A sparse backward-Euler solver for transient 2-D conduction with different conductivities in x and y, verified against exact and worked solutions. The problem statement comes from a course assignment. | [Supervisor overview](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#supervisor-overview) |
| **Neural Surrogate for the Two-Phase Flash Calculation**<br>`Subsurface_DL_Project` | PyTorch networks trained to reproduce the Wilson / Rachford-Rice flash calculation taught in the course, for unseen mixtures. A synthetic benchmark. | [Supervisor overview](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#supervisor-overview) |
| **Testing Gardner's Velocity–Density Relation against Measured Well Logs**<br>`SeisGeoMech` | Gardner's relation is compared with the measured sonic and density logs of UK well 48/10b-9, and its density error is followed into a one-dimensional ρg gradient and synthetic seismic reflectivity. | [Supervisor overview](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#supervisor-overview) |

## Projects

| Project | Informed by (MSc module) | Summary |
|---|---|---|
| [Surrogate-Assisted Screening of CO₂ Injection Schedules with Simulator Verification](advanced-subsurface-modelling/SubsurfaceML) | [Advanced Subsurface Modelling](advanced-subsurface-modelling) | A radial IMPES simulator generates **synthetic** reservoirs. A hybrid surrogate (a reduced-order model with a learned correction) predicts peak pressure build-up on 100 test reservoirs generated after the design was frozen. It achieves an RMSE of 0.31 MPa, against 0.70 MPa for the approach of the project's first public version (September 2026), retrained on the same reservoirs. Outside the training distribution the screening falls back to the simulator. There is no field validation, and the pressure and plume limits are modelling assumptions. |
| [Implicit Finite-Difference Modelling of Transient Heat Conduction in an Anisotropic Plate](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction) | [Fundamentals of Numerical Modelling and Simulation](fundamentals-of-numerical-modelling-and-simulation) | Backward-Euler finite differences with five verification checks. After 5 h, the mean temperature on the 13 × 12 grid specified in the course assignment is 6.4 K above the 193 × 177 result, and grid and time step act in opposite directions. The checks verify the code; no measurements exist for this problem, and no error bound is claimed for the finest grid. |
| [Neural Surrogate for the Two-Phase Flash Calculation](properties-of-subsurface-fluids/Subsurface_DL_Project) | [Properties of Subsurface Fluids](properties-of-subsurface-fluids), with Deep Learning | A **synthetic benchmark** built from the course's flash calculation. Test RMSE in vapour fraction is 0.00636 with the data loss and 0.00535 with the Rachford-Rice residual added. On pressure extrapolation, however, the residual made results worse in two of three seeds. Errors concentrate near the dew point, and a guarded predictor falls back to the reference solver outside the training domain. Three seeds; nothing is checked against measured phase behaviour. |
| [Testing Gardner's Velocity–Density Relation against Measured Well Logs](subsurface-mechanics-and-geoengineering/SeisGeoMech) | [Subsurface Mechanics and Geoengineering](subsurface-mechanics-and-geoengineering) | Gardner's relation is tested on **measured** sonic and density logs: 1,105 paired samples over 221 m of UK well 48/10b-9. It under-predicts density by 3.5 %, giving a one-dimensional ρg gradient of 24.97 against 25.89 MPa/km. In a frozen within-well depth-block test, locally fitted coefficients predict the adjacent block better than the supplied relation, but the two blocks give different coefficients. The seismic results are synthetic. One well; the depth coordinate is not established as vertical. |

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

| Project folder | Python | Quick check (in the project folder) | Installation and full instructions |
|---|---|---|---|
| `SubsurfaceML` | 3.11 (pinned versions) | `pytest` (107 tests, about 2 min) | [Reproduce](advanced-subsurface-modelling/SubsurfaceML/README.md#reproduce) |
| `anisotropic-heat-conduction` | 3.10 or later | `python run_project.py` (checks V1–V5 and the original study, about 1 min; exits with status 1 if a check fails) | [Run it](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#run-it) |
| `Subsurface_DL_Project` | 3.11 (as in its CI) | `python tests/run_tests.py` (48 tests) | [5. Reproduce and check](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#5-reproduce-and-check) |
| `SeisGeoMech` | 3.10 or later | `python -m pytest tests -q` (184 tests, about 5 s) | [Reproducing the results](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#reproducing-the-results) and [Run](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#run) |

Each project also has its own GitHub Actions workflow, which runs on changes to that project's
folder: [`SubsurfaceML`](.github/workflows/subsurfaceml-tests.yml) ·
[`anisotropic-heat-conduction`](.github/workflows/heat-conduction.yml) ·
[`Subsurface_DL_Project`](.github/workflows/subsurface-dl.yml) ·
[`SeisGeoMech`](.github/workflows/seisgeomech.yml).

## Structure

```
<module-folder>/          named after the MSc module whose material the project draws on
└── <project-folder>/
    ├── README.md         supervisor overview first, then method, results, limitations and how to run
    ├── LICENSE           the project's licence
    ├── requirements.txt  the dependencies for that project (requirements-lock.txt: exact versions)
    ├── docs/             detailed write-ups and the record of the October 2026 revision
    └── results/          the recorded tables, figures and run records
```

## Licences and attribution

All four projects carry the **MIT Licence** (© 2026 Abdelghafar Fouda). It covers each project's
own code and documentation, and in `SeisGeoMech` also its generated results. It does not cover
course teaching material, none of which is redistributed here: each project cites the teaching
material it relies on in its source map. The third-party Python libraries the projects use keep
their own open-source licences.

| Project folder | Licence | Data | Teaching sources | Details |
|---|---|---|---|---|
| `SubsurfaceML` | [MIT](advanced-subsurface-modelling/SubsurfaceML/LICENSE) | Synthetic, generated by the project's simulator. | CHEN60482 *Advanced Subsurface Modelling* materials (Dr Masoud Babaei; uncertainty material by Dr Lin Ma) and the *Data Science and Machine Learning* notebooks. Components from outside the course are listed with their primary sources. See [source map](advanced-subsurface-modelling/SubsurfaceML/docs/SOURCE_MAP.md). | [Author and attribution](advanced-subsurface-modelling/SubsurfaceML/README.md#author-and-attribution) |
| `anisotropic-heat-conduction` | [MIT](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/LICENSE) | None: all inputs come from the course assignment. | Course materials for the equations, inputs, methods and benchmark values. See [source map](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/SOURCE_MAP.md). | [Author and attribution](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md#author-and-attribution) |
| `Subsurface_DL_Project` | [MIT](properties-of-subsurface-fluids/Subsurface_DL_Project/LICENSE) | Synthetic, generated from the course notes' seven-component table. The notes credit that example to a Texas A&M PETE 310 spreadsheet. | CHEN60492 *Properties of Subsurface Fluids* notes (Dr Masoud Babaei), and the Deep Learning module's notebooks and *Introduction to Scientific Machine Learning* slides (Dr Ben Moseley). See [source map](properties-of-subsurface-fluids/Subsurface_DL_Project/docs/SOURCE_MAP.md). | [Author and attribution](properties-of-subsurface-fluids/Subsurface_DL_Project/README.md#9-author-and-attribution) |
| `SeisGeoMech` | [MIT](subsurface-mechanics-and-geoengineering/SeisGeoMech/LICENSE) for code, documentation and generated results | **Measured:** a public-domain UK continental-shelf well log (48/10b-9, operator BP), supplied with the module's practical and redistributed unmodified with its own attribution ([data note](subsurface-mechanics-and-geoengineering/SeisGeoMech/data/raw/README.md)). Two worked examples are transcribed from the exercise sheet. | University of Manchester EART35102 notes, exercises and practical notebooks. See [source map](subsurface-mechanics-and-geoengineering/SeisGeoMech/SOURCE_MAP.md). | [Licence and attribution](subsurface-mechanics-and-geoengineering/SeisGeoMech/README.md#licence-and-attribution) |
