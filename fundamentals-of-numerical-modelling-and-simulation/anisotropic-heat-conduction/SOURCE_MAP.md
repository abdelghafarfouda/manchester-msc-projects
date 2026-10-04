# Source map

Every model input, equation, numerical method and benchmark used by this
project comes from one of the two course folders below. Nothing else is used
as a scientific source; standard libraries (NumPy, SciPy, Matplotlib) only
implement the methods listed here.

* **Models**: the materials of an MSc module in numerical modelling and
  simulation — slides, assignments, tutorials, MATLAB files and a recorded MATLAB
  session. The files are the author's own copies and are not redistributed here.
* **Computational Mathmatics**: Jupyter notebooks (the folder name is spelled
  this way on disk). Cell numbers are 0-based positions in the `.ipynb` file,
  followed by the section heading.

Slide numbers refer to the slide number printed on the slide; for the PDF
handouts with two slides per page, the PDF page is given too.

## 1. Problem and physical inputs

| Item | Value used | Source |
|---|---|---|
| Governing equation | dT/dt = alpha * Laplacian(T) (Eq. 1), alpha = k / (rho cp) (Eq. 2) | `Models/Assignment (3).docx`, p. 1 |
| Anisotropic conductivity | kx = 16, ky = 20 W m^-1 K^-1, so rho cp dT/dt = kx T_xx + ky T_yy | `Assignment (3).docx`, p. 1 (kx, ky listed separately; Eq. 3 uses max(kx, ky)) |
| Specific heat, density | cp = 500 J kg^-1 K^-1, rho = 7800 kg m^-3 | `Assignment (3).docx`, p. 1 |
| Domain | 1 m x 1 m | `Assignment (3).docx`, p. 1 |
| Heated segments | top: T = 500 K; bottom: T = 300 sin(pi x / L) + 400 K; each half the domain width and centred | `Assignment (3).docx`, p. 1: text ("equal 1/2 of the domain width") and figure (position and values) |
| Other boundaries | insulated, grad T = 0 | `Assignment (3).docx`, p. 1, figure |
| Initial condition, duration | 300 K everywhere; 5 h | `Assignment (3).docx`, p. 1 |
| Heat flux (for the heat-input bookkeeping) | Fourier's law q = -k dT/dz, k in W m^-1 K^-1 | `Models/01_What_is_modelling [Auto-saved].pdf`, slide 28 (p. 14) |
| Context only | course focus on subsurface energy engineering (no geothermal model, inputs or data are given anywhere) | `Models/00_about_this_course (10).pptx`, slide 3 |

## 2. Numerical method

| Component | Where it is used | Source |
|---|---|---|
| Central difference method with implicit Euler, requested for this problem | whole solver | `Assignment (3).docx`, Q1 |
| Grid of nodes that includes the boundaries, dx = L/(nx - 1) | `heat2d.Grid` | `Models/MATLAB Assignment (Steady-state heat distribution) (5).pptx`, slide 1 ("nx and ny are number of nodes"); `Models/Matlab_Topic4_Q1 (13).m` (dx = Lx/(nx - 1)); `pdes1 (1).ipynb` cell 26, *Defining a grid* (nodes on the boundary named as an option) |
| Node counts 13 x 12 and 13, 17, 33, 65, 73 | base case and assignment grid study | `Assignment (3).docx`, p. 1 and Q5. For every listed nx, nx - 1 is a multiple of 4, so the segment ends at L/4 and 3L/4 fall on nodes; the code enforces this |
| Taylor series and finite differences; second-derivative stencil (f(x+dx) - 2f(x) + f(x-dx))/dx^2 with O(dx^2) error | `heat2d._stencil`, `heat2d.assemble` | `Models/Session2_course(1) (1).pdf`, slides 15-21 (pp. 8-11); `pdes1 (1).ipynb` cells 9, 15, 18-19, *Taylor series*, *The central difference*, *Approximating second derivatives* |
| Five-point stencil in 2-D | `heat2d._stencil` | `MATLAB Assignment (Steady-state heat distribution) (5).pptx`, slide 2 |
| Insulated boundary with a ghost node | `heat2d._stencil` | Central form, T_ghost = T_opposite from (T_ghost - T_opposite)/(2 dx) = 0: the central first-derivative formula, `Session2_course(1) (1).pdf` slide 18 (p. 9) and `pdes1 (1).ipynb` cell 15. Boundary approximations should match the interior order: `pdes1 (1).ipynb` cell 27, *Dealing with Dirichlet boundary conditions*. One-sided form, T_ghost = T_node, from the course hint (compared in V2 only): `Models/MATLAB_Session_17112022(1).mp4` at about 48:10 ("Hints for MATLAB Topic 4 Q2") and `Models/Matlab_Topic4_Q2 (9).m` |
| Prescribed-temperature nodes hold their values ("boundary as knowns") | `heat2d.assemble` | `Models/Tutorial_FDM_PDE_Solution(2) (10).docx`, pp. 3 and 8 (boundary node values inserted as knowns) |
| Method of lines: semi-discrete system dT/dt = -K T + g | `heat2d.Operator`, `heat2d.assemble` | `pdes1 (1).ipynb` cells 46, 51-52, *The Method of Lines*, *The semi-discrete matrix system* |
| Backward (implicit) Euler | `heat2d.backward_euler` | `ode-time-stepping-1 (1).ipynb` cell 24, *Four simple schemes*; `pdes1 (1).ipynb` cell 77, *Other time stepping options* (BTCS); `Models/Practical1 (26).docx` p. 2 (implicit formulation); `MATLAB_Session_17112022(1).mp4` about 50-53 min (central differences in space, implicit Euler in time) |
| Stability: explicit FTCS needs r <= 1/2; backward Euler is unconditionally stable | justification for Eq. 3's step (r about 0.8) | `Models/Topic 6- stability_and_more.pdf`, slides 8-12 (pp. 4-6); `pdes1 (1).ipynb` cell 76, *The r-number*; `ode-time-stepping-1 (1).ipynb` cells 57, 61 |
| Time step dt = min(dx^2, dy^2) rho cp / max(kx, ky) | `heat2d.time_step_eq3` | `Assignment (3).docx`, Eq. 3 |
| Coefficient matrix constant in time, right-hand side updated each step; LU decomposition, then forward and back substitution | `heat2d.backward_euler` (factorised once) | `Models/Practical1 (26).docx`, p. 4, step 5; `Models/Session3_course_final.pdf`, slides 36-48 (pp. 18-24); sparse matrices: `pdes1 (1).ipynb` cell 1 (scipy.sparse) and cell 78, *Final comments* |
| Area average by the composite trapezoidal rule (applied in x, then y) | `heat2d.area_average`, `heat2d.Grid.weights` | `interpolation-regression-quadrature (1).ipynb` cell 95, *Trapezoidal rule* |
| Conservation of the flux form | energy balance, `heat2d.heat_from_fixed_nodes`, `heat2d.stored_energy` | `Models/Topic 6- stability_and_more.pdf`, slide 24 (p. 12); divergence theorem: `Models/02_Tutorial_solution.pdf`, slides 7-9; conservative form: `pdes1 (1).ipynb` cell 44 |
| Centreline value at x = L/2 (middle node, or the mean of the two middle nodes) | `run_project.centreline` | `Assignment (3).docx`, Q4 |
| Corner temperatures | results table | `Assignment (3).docx`, Q3 |

## 3. Verification and benchmark

| Check | Source |
|---|---|
| Verification versus validation | `errors-verification-validation (1).ipynb` cells 12-13, 21, 30-32; `Models/01_What_is_modelling [Auto-saved].pdf`, slides 8 and 24 |
| Kinds of verification test (benchmark, conservation, exact solution) | `errors-verification-validation (1).ipynb` cell 23, *A code verification testing hierarchy* |
| **V1** course worked solution: aluminium rod, 10 cm, dx = 2 cm, dt = 0.1 s, diffusivity 0.835 cm^2/s, T(0) = 100 C, T(10 cm) = 50 C, lambda = 0.020875, nodal values at 0.1-0.4 s | `Models/Tutorial PDE FDM.pdf`, p. 1, Question 2; `Models/Tutorial_FDM_PDE_Solution(2) (10).docx`, pp. 8-10 |
| **V2** exact solution of a simplified, smooth problem with the temperature fixed along the whole top and bottom edges (a manufactured solution with zero source term, checked by substitution) | `errors-verification-validation (1).ipynb` cells 62, 78-85, *Comparing algorithms against analytic solutions* and *The Method of Manufactured Solutions* |
| RMS and maximum norms; observed order from a least-squares line on a log-log plot | `errors-verification-validation (1).ipynb` cells 49, 53-56, 74-75 |
| Expected orders: central differences 2, backward Euler 1 | `pdes1 (1).ipynb` cell 19; `ode-time-stepping-1 (1).ipynb` cell 31 |
| **V5** anisotropy orientation: exact solution of the discrete scheme on a grid with dx != dy, compared with the assembled solver, plus a control run with alpha_x and alpha_y exchanged | discrete symbol 4 alpha sin^2(k dx/2)/dx^2 from the von Neumann analysis in `Models/Topic 6- stability_and_more.pdf`, slides 9-11 (pp. 5-6); central differences and the mirrored ghost node as in the rows above; backward Euler damping factor 1/(1 + dt lambda) from `ode-time-stepping-1 (1).ipynb` cell 24; exact-solution testing from `errors-verification-validation (1).ipynb` cells 62, 78-85 |
| **V3** energy balance (conservation test) | `errors-verification-validation (1).ipynb` cell 23; `Topic 6- stability_and_more.pdf`, slide 24 |
| **V4** units: dimensional analysis | `pdes1 (1).ipynb` cell 20 ("perform a dimensional analysis") and cell 76 ("check units"); Eq. 3 of `Assignment (3).docx` |
| **Grid study**: Eq. 4 relative change between successive grids, 0.2 % criterion, nx = 13, 17, 33, 65, 73 with ny = 12 | `Assignment (3).docx`, Q5 |
| Solution verification by sensitivity testing | `errors-verification-validation (1).ipynb` cells 24, 28 |
| Halving dx while quartering dt for diffusion (what Eq. 3 does automatically) | `pdes1 (1).ipynb` cell 76, comment 2 |
| **Revision (2026-10-04) studies**: refining the grid at a fixed time step and the time step on a fixed grid, reported as changes between successive levels and their ratios (no extrapolation) | sensitivity testing: `errors-verification-validation (1).ipynb` cells 24, 28; expected orders as for V2: `pdes1 (1).ipynb` cell 19, `ode-time-stepping-1 (1).ipynb` cell 31 |
| **S1** heat accounting: stored-energy increase = boundary-flux heat input + half-cell energy of the prescribed nodes | the same trapezoidal weights and control volumes as V3: `interpolation-regression-quadrature (1).ipynb` cell 95; conservation: `Topic 6- stability_and_more.pdf`, slide 24 |
| **S2**, **S3** matching nested-grid locations; adequacy of the fixed time step | design checks of this project; S3 uses the same sensitivity-testing idea as the row above |

## 4. Numerical settings chosen in this project (not physical inputs)

* Steps: n = ceil(5 h / dt_Eq3) equal steps of 5 h / n, so the run ends exactly
  at 5 h with a step no larger than Eq. 3 gives.
* Refinement sequence with dx and dy halved together: 13 x 12, 25 x 23, 49 x 45,
  97 x 89, 193 x 177 nodes. dt comes from Eq. 3 each time, so every step also
  quarters dt.
* Diagnostic in section 4c of the log: the same problem with both heated
  segments extended over the full edges. It is used only to test whether the
  segment ends could explain the observed convergence rate of the mean temperature.
* V2 settings: T = 300 + 100 cos(pi x) sin(pi y) exp(-gamma t) on the plate
  properties, compared at t = 1 h; space study on 9, 17, 33, 65 nodes per side with
  dt = 0.25 s; time study on 129 x 129 nodes with dt = 900 to 56.25 s. The space
  study is repeated at dt = 0.0625 s to measure how much of the fitted order comes
  from the time step; the dt = 0.25 s run stays the recorded benchmark.
* V5 settings: 33 x 17 nodes on the same square (dx = 1/32, dy = 1/16), initial
  field T = 300 + 100 cos(pi x) sin(2 pi y), insulated left and right edges,
  T = 300 K on the whole top and bottom edges, Equation 3 time step (19 steps of
  189.47 s to 1 h). Tolerance 1e-9 K on the maximum difference from the analytic
  discrete solution, with the alpha_x / alpha_y exchange required to exceed it.
* Revision studies (`run_studies.py`, 2026-10-04), with the same plate, boundary
  conditions and solver:
  * spatial study: the five nested grids above at a fixed dt = 5.2895 s (3403
    steps), the 193 x 177 grid's Eq. 3 step, so the original 193 x 177 run is its
    finest member; repeated at dt = 1285.71 s (14 steps), the 13 x 12 grid's Eq. 3
    step;
  * temporal study: 14, 28, 56, 112, 224, 448, 896, 1792, 3403 and 6806 steps to
    5 h (dt = 1285.71 s halved seven times, then 5.29 and 2.64 s), on the 193 x 177
    grid and on the 13 x 12 grid;
  * segment ends: every grid compared with the 193 x 177 grid (and with the next
    finer grid) at its own nodes, which the finer grids share exactly; "near an
    end" = within 0.1 m of one of the four segment ends; boundary heat input
    summed over seven intervals per segment, one around each 13 x 12 segment node;
  * study-check tolerances, fixed before the recorded run: S1 <= 1e-10 relative
    (as V3); S2 <= 1e-12 m and <= 1e-12 relative; S3: halving the fixed step must
    change the 193 x 177 mean temperature by < 1 % of the smallest grid-to-grid
    change.
* Comparison with the recorded results (`compare_results.py`, CI): relative
  tolerance 1e-9 and absolute tolerance 1e-9 in each value's unit; text exact.
* Tolerances, fixed before the recorded run: V1 <= 5e-5 C (half the last digit
  of the tabulated solution); V2 orders within 0.1 of 2 and 1; V3 <= 1e-10
  relative; V4 <= 1e-8 K and <= 1e-10 relative; V5 <= 1e-9 K, with the
  exchanged-coefficient case required to exceed it.

## 5. Not used

The earlier geothermal doublet model and its inputs (wells, reservoir and
fluid properties, well spacing, injection rates, pressure limits, tracer
transport, heterogeneity, uncertainty sampling) have no support in these
folders and are not part of this project. The advection-diffusion,
root-finding, regression, probability and machine-learning material in the
course folders is not needed for this problem and is not used.
