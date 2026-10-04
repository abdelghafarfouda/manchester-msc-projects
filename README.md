# MSc Subsurface Energy Engineering Projects — University of Manchester

This repository collects projects developed from my MSc studies in Subsurface
Energy Engineering at the University of Manchester. Each module has its own
folder, and each project sits in its own folder inside that module, keeping its
own documentation, dependencies and results so that it can be read and run on
its own.

## Projects

| Module | Project | What it does |
|---|---|---|
| [Fundamentals of Numerical Modelling and Simulation](fundamentals-of-numerical-modelling-and-simulation) | [anisotropic-heat-conduction](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction) | Transient 2-D heat conduction in a plate whose thermal conductivity differs in x and y, solved in Python with an implicit finite-difference scheme: five-point central differences in space, backward Euler in time, and one sparse LU factorisation per run. Verified by five checks — a worked benchmark, an exact solution, an energy balance, a unit conversion and an anisotropy-orientation test — and used for a grid-refinement study of the mean plate temperature. The October 2026 revision separates the grid and time-step effects (on the assignment grid, +8.2 K and −1.8 K in the mean temperature), distinguishes the boundary-flux heat input (−5.5 % on the coarse grid) from the stored-energy increase (+4.6 %), locates the largest grid differences at the heated-segment ends, and adds a CI check of the recorded results. |
| [Advanced Subsurface Modelling](advanced-subsurface-modelling) | [SubsurfaceML](advanced-subsurface-modelling/SubsurfaceML) | Surrogate-assisted screening of CO₂ injection schedules for a sealed, layered saline aquifer, with every recommendation verified by the simulator. A radial IMPES simulator (21 verification checks) generates synthetic data; a hybrid surrogate — an analytical reduced-order model times a learned correction — predicts peak bottom-hole pressure build-up on 100 fresh test reservoirs, generated after the design was fixed, with an RMSE of 0.31 MPa, against 0.70 MPa for the previously published approach trained on the same reservoirs; reservoir-calibrated intervals, a distribution-shift test, and a separate radial–vertical model that measures the error of omitting gravity and crossflow (peak build-up −4 %, plume radius +88 %). Extended and re-verified in October 2026. |
| [Properties of Subsurface Fluids](properties-of-subsurface-fluids) | [Subsurface_DL_Project](properties-of-subsurface-fluids/Subsurface_DL_Project) | A neural surrogate for the module's two-phase flash calculation (Wilson K-values with Rachford-Rice), built as a synthetic benchmark from the course notes. PyTorch networks trained on 1,800 of 3,000 generated mixtures (60,000 two-phase states) predict the vapour fraction of 600 mixtures held out by mixture with a mean test RMSE of 0.00636; adding the Rachford-Rice residual to the loss reduced it to 0.00535, but did not improve a higher-pressure extrapolation set. The October 2026 revision reproduces the original 185 metrics exactly; adds a guarded prediction path that validates inputs, applies the course phase test first and answers states outside the training domain with the reference solver; shows that the 5 % of test states nearest the dew point hold 45–58 % of the squared error; finds that a 3 × 64 network performs as well as the original 3 × 128; and adds a CI comparison of the saved models' results. |
| [Subsurface Mechanics and Geoengineering](subsurface-mechanics-and-geoengineering) | [SeisGeoMech](subsurface-mechanics-and-geoengineering/SeisGeoMech) | Does the Gardner velocity-density relation hold at a real well, and what does its error cost? Over the 221 m of UK well 48/10b-9 where the sonic and density logs overlap (1,105 paired samples), Gardner under-predicts bulk density by 0.093 g/cm³ (3.5 %, RMSE 0.118). Because the module's overburden model integrates that same density, the error carries straight through to a one-dimensional ρg gradient of 24.97 against 25.89 MPa/km. On the seismic side the same substitution makes impedance a function of velocity alone, raising reflection-coefficient RMS by 20 % before convolution while the synthetic-trace RMS ratio stays below one after it. Every equation, dataset and constant is traced to a specific page or notebook cell in SOURCE_MAP.md. The October 2026 revision reproduces the published results exactly and fixes a clean-install failure on Python 3.12+ (`bruges` needs `setuptools<81`). It also adds a two-direction depth-block test whose design was frozen before scoring. Coefficients fitted on one 100 m block predict the other with held-out RMSE 0.088 and 0.061 g/cm³, against 0.113 and 0.129 for the supplied relation, but the two blocks need different coefficients. 160 tests and a two-job CI. |

This index lists only the projects currently in the repository. Further module
projects will be added in the same module-folder / project-folder structure.

## Opening and running a project

Each project is self-contained and is run from its own folder. For the
heat-conduction project:

```bash
git clone https://github.com/abdelghafarfouda/manchester-msc-projects.git
cd manchester-msc-projects/fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction
python -m pip install -r requirements.txt
python run_project.py
python run_studies.py
```

On Windows, open a terminal (for example an Anaconda Prompt) in that same folder
and run the last three commands. The scripts need Python 3.10 or later with NumPy,
SciPy and Matplotlib, write every output to `results/`, and exit with status 1
if any check fails.

The project's own
[README](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction/README.md)
sets out the problem, the numerical method, what each verification check does and
does not establish, the results and the limitations, and records the environment
in which the published results were produced.

## Structure

```
<module-folder>/
└── <project-folder>/
    ├── README.md         the problem, method, verification, results and limitations
    ├── requirements.txt  the dependencies for that project
    ├── docs/             derivations and a full walkthrough
    └── results/          the recorded tables, figures and run log
```

## Licence and attribution

Each project carries its own licence and its own source attribution. The
heat-conduction project is MIT licensed, and its `SOURCE_MAP.md` traces every
input, equation, numerical method and benchmark value to the specific teaching
material it came from. No course teaching material is redistributed here.
