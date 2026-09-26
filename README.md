# MSc Subsurface Energy Engineering Projects — University of Manchester

This repository collects projects developed from my MSc studies in Subsurface
Energy Engineering at the University of Manchester. Each module has its own
folder, and each project sits in its own folder inside that module, keeping its
own documentation, dependencies and results so that it can be read and run on
its own.

## Projects

| Module | Project | What it does |
|---|---|---|
| [Fundamentals of Numerical Modelling and Simulation](fundamentals-of-numerical-modelling-and-simulation) | [anisotropic-heat-conduction](fundamentals-of-numerical-modelling-and-simulation/anisotropic-heat-conduction) | Transient 2-D heat conduction in a plate whose thermal conductivity differs in x and y, solved in Python with an implicit finite-difference scheme: five-point central differences in space, backward Euler in time, and one sparse LU factorisation per run. Verified by five checks — a worked benchmark, an exact solution, an energy balance, a unit conversion and an anisotropy-orientation test — and used for a grid-refinement study of the mean plate temperature. |

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
```

On Windows, open a terminal (for example an Anaconda Prompt) in that same folder
and run the last two commands. The script needs Python 3.10 or later with NumPy,
SciPy and Matplotlib, writes every output to `results/`, and exits with status 1
if any verification check fails.

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
