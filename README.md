# MSc Subsurface Energy Engineering Projects — University of Manchester

This repository collects projects developed from my MSc studies in Subsurface
Energy Engineering at the University of Manchester. Each module has its own
folder, and each project sits in its own folder inside that module, keeping its
own documentation, dependencies and results so that it can be read and run on
its own.

I developed the original projects myself. AI tools were subsequently used to
help publish them on GitHub and make minor quality improvements.

## Projects

| Module | Project | In one line |
|---|---|---|
| [Fundamentals of Numerical Modelling and Simulation](fundamentals-of-numerical-modelling-and-simulation) | [anisotropic-heat-conduction](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction) | A verified implicit finite-difference solver for transient heat conduction in an anisotropic plate, with a grid-refinement study |
| [Advanced Subsurface Modelling](advanced-subsurface-modelling) | [SubsurfaceML](advanced-subsurface-modelling/SubsurfaceML) | A radial IMPES simulator for CO₂ injection into a layered saline aquifer, and machine-learning surrogates tested on held-out reservoirs |
| [Properties of Subsurface Fluids](properties-of-subsurface-fluids) | [Subsurface_DL_Project](properties-of-subsurface-fluids/Subsurface_DL_Project) | A neural surrogate for the module's two-phase flash calculation, trained with and without a Rachford-Rice physics loss |

Each project README opens with a summary of the same five points, condensed
below.

### Anisotropic heat conduction

- **Objective.** The temperature field, mean temperature and heat input of a
  partially heated anisotropic plate after 5 h, and the grid needed before the
  mean temperature changes by less than 0.2 %.
- **Method.** Implicit finite differences in Python (five-point central
  differences in space, backward Euler in time, one sparse LU factorisation per
  run), checked against a course worked solution, an exact solution, an energy
  balance, a units test and an anisotropy-orientation test.
- **Key result.** All six verification checks pass. The finest grid, 193 × 177
  nodes, gives a mean temperature of 438.5 K after 5 h (444.9 K on the
  assignment's 13 × 12 grid), and refining both directions together first meets
  the 0.2 % criterion there.
- **Main limitation.** Verification only, with no measurements; the mean
  temperature converges at only about first order, and the 0.2 % criterion
  measures the change between grids, not the error.
- **How to run.** From the project folder, `python -m pip install -r requirements.txt`,
  then `python run_project.py` (about a minute).

### SubsurfaceML

- **Objective.** How permeability, heterogeneity, fluid mobility and the
  injection schedule control peak pressure build-up and plume spread in a
  sealed, layered saline aquifer, and how accurately a surrogate predicts them
  for reservoirs it has not seen.
- **Method.** A radial, multi-layer IMPES simulator with one shared bottom-hole
  pressure, verified by 18 analytical and conservation checks, generates 880
  cases: 496 for training, 164 for calibrating error bands and 220 for testing,
  the test cases coming from 55 reservoirs held out by realisation. Regression
  families from the course are tuned by grouped cross-validation, and
  surrogate-screened injection schedules are re-simulated.
- **Key result.** On the 220 test cases, peak pressure build-up is predicted
  with an RMSE of 1.24 MPa and R² = 0.941 (mean-value baseline: 5.08 MPa).
- **Main limitation.** Synthetic data, and no gravity, dissolution, residual
  trapping or capillary pressure. On one 40 mD test reservoir every screened
  schedule exceeded the assumed pressure limit when re-simulated, although the
  surrogate and its error band predicted they would not.
- **How to run.** From the project folder, with Python 3.11,
  `python -m pip install -r requirements-lock.txt` and
  `python -m pip install -e . --no-deps`, then run
  `notebooks/00_START_HERE.ipynb` (under a minute, from the saved study run).

### Flash surrogate (Subsurface_DL_Project)

- **Objective.** Whether a small network can predict the vapour fraction `F_V`
  of the module's flash calculation for mixtures it has not seen, and whether a
  Rachford-Rice term in the loss helps.
- **Method.** Wilson K-values with Rachford-Rice, checked against the notes'
  three worked examples, label 60,000 two-phase states from 3,000 generated
  mixtures, split by mixture into 1,800 training, 600 validation and 600 test
  mixtures. PyTorch networks are trained with and without the Rachford-Rice
  residual, three seeds each.
- **Key result.** On the 600 held-out mixtures the mean test RMSE is 0.00636
  with the data loss alone and 0.00535 with the physics term, a 15.9 %
  reduction.
- **Main limitation.** A synthetic benchmark whose labels come from a
  correlation, not measurements; the physics term did not improve a
  higher-pressure extrapolation set.
- **How to run.** From the project folder, `pip install -r requirements.txt`,
  then `jupyter notebook notebooks/flash_surrogate.ipynb`. The trained models
  are included, so nothing needs training and the notebook runs in under a
  minute.

This index lists only the projects currently in the repository. Further module
projects will be added in the same module-folder / project-folder structure.

## Opening and running a project

Each project is self-contained and is run from its own folder:

```bash
git clone https://github.com/abdelghafarfouda/manchester-msc-projects.git
cd manchester-msc-projects/<module-folder>/<project-folder>
```

Then follow the *How to run* line above. On Windows, open a terminal (for
example an Anaconda Prompt) in the project folder and run the same commands.
Each project's README has a full *Run it* section, with the Python version and
runtimes, and an execution record of the environment in which its published
results were produced.

## Structure

```
<module-folder>/
└── <project-folder>/
    ├── README.md         summary, problem, method, verification, results and limitations
    ├── LICENSE           MIT licence for the project's code
    ├── requirements.txt  the dependencies for that project
    ├── SOURCE_MAP.md     where every input, equation and method comes from (in docs/ for some projects)
    ├── docs/             walkthroughs, derivations and limitations
    └── results/          the recorded tables, figures and metrics
```

## Licence, attribution and credits

Each project is MIT licensed (see its `LICENSE`) and carries its own source
attribution. Its source map — `SOURCE_MAP.md` in the heat-conduction project,
`docs/SOURCE_MAP.md` in the other two — traces every input, equation, numerical
method and benchmark to the specific teaching material it came from. No course
teaching material is redistributed here.

Course material credited in the project READMEs and source maps:

- **Fundamentals of Numerical Modelling and Simulation.** The assignment,
  slides, tutorials and MATLAB material of the module, and the accompanying
  *Computational Mathematics* notebooks.
- **Advanced Subsurface Modelling (CHEN60482).** Lectures by Dr Masoud Babaei,
  uncertainty material by Dr Lin Ma, and the accompanying *Data Science and
  Machine Learning* notebooks.
- **Properties of Subsurface Fluids (CHEN60492).** Notes by Dr Masoud Babaei,
  whose seven-component example (pp. 14–15) the notes credit to a Texas A&M
  PETE 310 spreadsheet; the Deep Learning module's PyTorch notebooks and its
  *Introduction to Scientific Machine Learning* slides (Dr Ben Moseley).

Open-source libraries (NumPy, SciPy, pandas, scikit-learn, Matplotlib, PyTorch
and the others listed in each project) are used under their own licences.
