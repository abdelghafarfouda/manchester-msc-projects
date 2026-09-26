# Transient heat conduction in an anisotropic plate

Transient 2-D heat conduction in a plate whose conductivity differs in x and y,
solved in Python with an implicit finite-difference scheme (central differences in
space, backward Euler in time, one sparse LU factorisation per run) and verified
against a worked benchmark, exact solutions, an energy balance and a units check.
The study then measures how the predicted mean temperature depends on refinement
of the grid and time step.

## Question

A 1 m x 1 m plate is initially at 300 K. Its conductivity is anisotropic (kx = 16,
ky = 20 W m⁻¹ K⁻¹), with ρ = 7800 kg m⁻³ and cp = 500 J kg⁻¹ K⁻¹. It is heated for
5 h through two centred boundary segments, each half the plate width. The top
segment is held at 500 K and the bottom one at 300 sin(πx/L) + 400 K; every other
boundary is insulated.

**What are the temperature field, the mean temperature and the heat conducted into
the plate after 5 h, and how fine must the grid be before the mean temperature
changes by less than the course's 0.2 % criterion?**

## Source and scope

The problem is Assignment (3) of an MSc module in numerical modelling and
simulation, solved with methods from that module and from the accompanying
*Computational Mathematics* notebooks. [SOURCE_MAP.md](SOURCE_MAP.md) traces every
input, equation, method and benchmark to a specific file and page, slide or
notebook cell.

The material properties are the assignment's values; the project does not claim
they describe any rock, reservoir or geothermal system. The course materials
contain no geothermal model or data, so none is used.

## Model

ρ cp ∂T/∂t = kx ∂²T/∂x² + ky ∂²T/∂y², on 0 ≤ x ≤ L, 0 ≤ y ≤ W:

| | |
|---|---|
| top edge, L/4 ≤ x ≤ 3L/4 | T = 500 K |
| bottom edge, L/4 ≤ x ≤ 3L/4 | T = 300 sin(πx/L) + 400 K |
| rest of the boundary | insulated, ∂T/∂n = 0 |
| initial condition | T = 300 K |
| L = W | 1 m |
| thermal diffusivities αx = kx/(ρcp), αy = ky/(ρcp) | 4.103 × 10⁻⁶, 5.128 × 10⁻⁶ m² s⁻¹ |

The plate is two-dimensional, so heat quantities are per metre of depth.

## Method

* **Grid.** Nodes include the boundaries, with dx = L/(nx − 1), as in the course
  codes. The assignment grid is 13 × 12 nodes.
* **Space.** Second-order central differences, using the five-point stencil.
* **Insulated boundary.** A ghost node is mirrored across the boundary (T_ghost =
  T_opposite). This is the central difference of ∂T/∂n = 0, so it is second order
  like the interior stencil. The course hint uses a one-sided difference instead;
  test V2 compares the two.
* **Heated segments.** Their nodes hold the prescribed temperature and enter the
  right-hand side.
* **Time.** The method of lines gives dT/dt = −K T + g, which is integrated with
  backward (implicit) Euler: (I + Δt K) Tⁿ⁺¹ = Tⁿ + Δt g. The matrix is constant,
  so it is LU-factorised once and reused every step.
* **Time step.** The assignment's Equation 3, Δt = min(Δx², Δy²) ρ cp / max(kx, ky),
  rounded down so that a whole number of steps lands exactly on 5 h. On the
  assignment grid this gives 14 steps of 1286 s, so r_x = αxΔt/Δx² = 0.76 and
  r_y = αyΔt/Δy² = 0.80. Each already exceeds the explicit scheme's 1-D limit of
  1/2 on its own; backward Euler is unconditionally stable, so this step is
  allowed. Because Equation 3 ties Δt to Δx², refining the grid also shortens the
  time step.
* **Mean temperature and heat input.**
  * The mean temperature is an area average by the composite trapezoidal rule.
  * The heat input is taken from the solver's own discrete fluxes out of the
    heated-segment nodes.

The derivations and the reasons for each choice are in
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Verification

This is numerical verification: it checks that the equations are solved
correctly. It is **not validation**; no measurements exist for this problem.

| Check | Result | Tolerance (set before the run) | What it establishes |
|---|---|---|---|
| V1: course worked solution (implicit rod, 4 steps) | max difference 4.7 × 10⁻⁵ °C | ≤ 5 × 10⁻⁵ °C | the implicit Euler / central-difference assembly reproduces the course's hand calculation |
| V2: exact solution, space refinement | order 1.97 (RMS), 1.97 (max) | 2 ± 0.1 | second-order spatial accuracy and the ghost-node insulated boundary, on a smooth test problem with the whole top and bottom edges at a fixed temperature, not on the partially heated plate. The course hint's one-sided boundary gives order 0.96. The test uses a square grid and the same wavenumber in x and y, so it cannot detect an exchange of αx and αy — that is what V5 is for |
| V2: exact solution, time refinement | order 0.98 | 1 ± 0.1 | first-order backward Euler |
| V3: energy balance, every plate run | worst closure 5.2 × 10⁻¹² (relative) | ≤ 10⁻¹⁰ | stored energy change = heat conducted in, to round-off: the scheme is conservative and the heat bookkeeping is consistent |
| V4: units, same case in (cm, h) vs (m, s) | max ΔT 1.0 × 10⁻¹² K | ≤ 10⁻⁸ K | no hidden unit assumption; Equation 3 is dimensionally consistent and gives 1354.17 s in both systems |
| V5: anisotropy orientation, 33 × 17 grid (Δx ≠ Δy) | max ΔT 1.6 × 10⁻¹² K as implemented; 4.7 K with αx and αy exchanged in the assembly | ≤ 10⁻⁹ K as implemented, and the exchanged case must exceed it | αx and αy act on the correct derivatives. The solver is compared with the analytic solution of the discrete scheme, which is derived independently of the assembly, so the deliberate exchange is caught |

V2's spatial order depends slightly on the time step it is measured with: the fitted
order is 1.971 at Δt = 0.25 s (the recorded benchmark) and 1.989 at Δt = 0.0625 s,
as the finest-grid RMS error falls from 2.506 × 10⁻³ to 2.405 × 10⁻³ K. Both runs are
in `results/verification_exact_space.csv`.

## Results

![Temperature after 5 h](results/fig1_temperature_5h.png)

| After 5 h | Assignment grid, 13 × 12 nodes, Δt = 1286 s | Finest grid, 193 × 177 nodes, Δt = 5.29 s | 13 × 12 minus 193 × 177 |
|---|---|---|---|
| mean temperature | 444.9 K | 438.5 K | +6.4 K (+1.5 %) |
| heat conducted in (per m depth) | 507 MJ/m | 537 MJ/m (top 191, bottom 346) | −30 MJ/m (−5.5 %) |
| bottom corners | 465.9 K | 453.8 K | +12.1 K |
| top corners | 406.6 K | 395.1 K | +11.5 K |

The percentages are relative to the 193 × 177 values. Those values are the most
refined numerical estimates, not exact answers. The table comes from
`results/grid_sensitivity.csv` and `results/corner_temperatures_5h.csv`; every
number quoted below is in `results/grid_sensitivity.csv`, `results/run_log.txt` or
`results/summary.json`.

**Grid sensitivity (Equation 4, criterion 0.2 %).**

* The assignment's study refines only nx, with ny fixed at 12. It meets the
  criterion already at nx = 17 (a change of 0.05 %). Because dy never changes, it
  cannot show the effect of the y-resolution.
* Refining dx and dy together, with Equation 3 quartering Δt at each step, gives
  changes of 0.72, 0.41, 0.21 and 0.11 %. Successive estimates therefore first
  agree within 0.2 % at 193 × 177 nodes.
* Each change is about half the previous one (ratios 1.79, 1.91 and 1.96), so under
  this combined refinement the mean temperature converges at about first order in
  the grid spacing. Second-order spatial errors, or backward Euler's first-order
  error in Δt, would each shrink about four-fold per step, since Δx is halved and Δt
  quartered.
* A diagnostic run with the heated segments extended over the whole edges gives
  mean-temperature ratios of 3.6 and 3.9, close to four. This is consistent with
  the loss of order coming from the four segment ends, where a fixed temperature
  meets an insulated boundary part-way along an edge.

![Grid sensitivity](results/fig3_grid_sensitivity.png)

![History and centreline](results/fig2_history_and_centreline.png)

![Verification](results/fig4_verification.png)

## Limitations

* **Scope.** Conduction only, with the assignment's constant properties, in 2-D per
  unit depth. There is no fluid flow, and no rock or reservoir data; the results say
  nothing about any geothermal system.
* **Segment ends.** Temperature is prescribed on one side of each end and insulated
  on the other. The full-edge diagnostic is consistent with these points causing the
  first-order convergence of the mean temperature, but it is not proof: extending
  the segments also changes the heated length and the boundary temperatures. The
  local error was not mapped, so where it is largest is not known.
* **Sudden heating at t = 0.** The heated segments jump from 300 K to 500–700 K, so
  heat enters fastest at first. The time resolution of this early period was not
  studied separately.
* **What Equation 4 measures.**
  * It measures the change between successive grids, not the error, so meeting the
    0.2 % criterion does not bound the error of the 193 × 177 result. No
    extrapolated or error-bounded value is given, because no such method is in the
    course materials.
  * The size of the change depends on how different the two grids are: the
    assignment's last step, 65 → 73 nodes, reduces dx by only 11 %.
  * It is relative to the absolute temperature in kelvin. The final change of 0.48 K
    is 0.11 % of 438.5 K but 0.35 % of the 138.5 K temperature rise.
* **Verification, not validation.** Nothing is compared with measurements.

## Run it

Needs Python 3.10 or later with NumPy, SciPy and Matplotlib.

```bash
python -m pip install -r requirements.txt
python run_project.py
```

On Windows, open a terminal (for example an Anaconda Prompt) in the project folder and
run the same two commands. The script takes about a minute, writes everything to
`results/`, and exits with status 1 if any verification check fails.

## Execution record

The results in `results/` were produced with `python run_project.py` on
2026-09-26 (00:06 UTC). The run took 51.5 s and passed 6 of 6 checks.

* **Platform.** `Linux-6.8.0-138-generic-x86_64-with-glibc2.35`, x86-64.
* **Software.** Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.9.
* **Records.** `results/run_log.txt` is the full console log and
  `results/environment.txt` the platform and versions.
* **Reproduced.** Re-running the workflow from a clean copy of this repository in
  the same environment reproduced all 18 result files byte for byte except the
  three that record timestamps (`environment.txt`, `run_log.txt`, `summary.json`).
  Running it in a different environment (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1,
  Matplotlib 3.11.2, different Linux kernel) reproduced the tables byte for byte,
  apart from one verification error in its sixteenth digit, and gave visually
  identical figures whose PNG bytes differ through the Matplotlib version. The
  workflow has not been run natively on Windows.

## Files

```
run_project.py      the whole study, top to bottom (entry point)
heat2d.py           grid, assembly, backward Euler, averages, heat flow, norms
SOURCE_MAP.md       source of every input, equation, method and benchmark
docs/WALKTHROUGH.md question -> equations -> method -> verification -> results
results/            CSV tables, figures, summary.json, run_log.txt, environment.txt
```

## Author and attribution

Abdelghafar Fouda — problem set-up, implementation, verification and analysis.
The equations, physical inputs, numerical methods and benchmark values come from
the course materials cited in [SOURCE_MAP.md](SOURCE_MAP.md); no course material is
redistributed here. The code uses NumPy, SciPy and Matplotlib (BSD licences). AI
assistance was used while writing and reviewing the code and documentation; every
number in this README is produced by `run_project.py` and stored in `results/`.

Licence: MIT (see `LICENSE`).
