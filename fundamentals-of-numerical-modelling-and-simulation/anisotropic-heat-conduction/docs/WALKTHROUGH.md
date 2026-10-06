# Walkthrough

The project follows one line of reasoning: **question, equations and
assumptions, numerical method, implementation, verification, results,
limitations.** Numbers quoted here come from `results/run_log.txt`,
`results/studies_run_log.txt` and the CSV files beside them. Source references
(file, slide or notebook cell) are in [`SOURCE_MAP.md`](../SOURCE_MAP.md).
Section 7 and the heat accounting in section 3.8 were added in the October 2026
revision (`run_studies.py`); the rest describes the study as first published on
2026-09-26 (`run_project.py`), whose results are unchanged.

Energies are per metre of out-of-plane depth (J/m or MJ/m), because the plate is
two-dimensional: 1 MJ/m is 1 MJ for each metre of plate thickness.

---

## 1. The question

The course assignment defines a 1 m x 1 m plate, initially at 300 K, heated for 5 h
through two centred segments that are each half the plate width:

* the top segment is held at 500 K;
* the bottom segment is held at 300 sin(pi x / L) + 400 K (612 K at the segment
  ends, 700 K at its centre);
* the rest of the boundary is insulated.

The assignment asks for:

* the temperature field;
* the corner temperatures;
* the temperature along the vertical centreline;
* a grid-sensitivity study: refine nx at fixed ny = 12 until the mean temperature
  changes by less than 0.2 % (its Equation 4).

The project answers these, adds the heat conducted into the plate (which the
energy balance provides anyway), and then asks whether the assignment's
sensitivity study is enough to trust the answer.

## 2. Equations and assumptions

Energy conservation, with Fourier's law q = -k grad T applied separately in x
and y, gives

```
rho cp dT/dt = kx d2T/dx2 + ky d2T/dy2        <=>   dT/dt = ax d2T/dx2 + ay d2T/dy2
ax = kx/(rho cp) = 4.103e-6 m^2/s              ay = ky/(rho cp) = 5.128e-6 m^2/s
```

All of these assumptions come from the assignment:

* constant, uniform properties;
* a conductivity tensor aligned with x and y;
* no internal heat source;
* a two-dimensional plate, so energies and heat flows are per metre of depth;
* boundary temperatures that are constant in time and switched on at t = 0.

The diffusion length after 5 h is sqrt(ay t) = 0.30 m, and ay t / W^2 = 0.09. The
plate is therefore still far from steady state at 5 h: the answer depends on the
transient, not just on the boundary values.

## 3. Numerical method

### 3.1 Grid

Nodes sit at x_i = i dx, y_j = j dy with dx = L/(nx - 1) and dy = W/(ny - 1), so
the outer nodes lie on the boundary. This is the grid used by the course codes
and the one the assignment's node counts imply. For all the assignment's values
(13, 17, 33, 65, 73), nx - 1 is a multiple of 4, which puts the segment ends
x = L/4 and 3L/4 exactly on nodes. The code refuses any other nx.

### 3.2 Central differences

Adding the Taylor expansions of T(x + dx) and T(x - dx) gives

```
d2T/dx2 = (T[i+1] - 2 T[i] + T[i-1]) / dx^2 + O(dx^2)
```

and likewise in y. Each unknown node therefore obeys the semi-discrete equation

```
dT[i,j]/dt = ax (T[i+1,j] - 2T[i,j] + T[i-1,j])/dx^2 + ay (T[i,j+1] - 2T[i,j] + T[i,j-1])/dy^2
```

### 3.3 Insulated boundaries: the ghost node

At a node on an insulated edge, say x = 0, the stencil needs T[-1,j], which lies
outside the plate. Writing the boundary condition dT/dx = 0 with the **central**
difference (T[1,j] - T[-1,j]) / (2 dx) = 0 gives T[-1,j] = T[1,j], so

```
d2T/dx2 at x = 0  ~  2 (T[1,j] - T[0,j]) / dx^2        (second order)
```

The course hint for its steady-state Question 2 uses the **one-sided** difference
(T_ghost - T_node)/dx = 0 instead. That gives T_ghost = T_node and a term
(T[1,j] - T[0,j]) / dx^2, which in effect moves the insulated wall half a cell
outside the plate. That is first order.

The notebooks advise matching the boundary approximation to the order of the
interior scheme, so the central form is used. Verification V2 measures both
forms: 1.97 against 0.96.

The central form has a useful consequence. It is exactly the energy balance of a
control volume of half size (dx/2 x dy on an edge, a quarter at a corner) around
each boundary node. This is why the energy balance in 3.8 closes exactly.

### 3.4 Heated segments

Nodes on the heated segments hold their prescribed temperatures. They are not
unknowns: their contributions move to the right-hand side, as in the course's
worked tutorial solutions.

### 3.5 Method of lines

Collecting the unknown nodes into a vector T gives a linear system of ODEs

```
dT/dt = -K T + g
```

* K is a sparse matrix with five non-zeros per row at most.
* g holds the heated-segment contributions.
* Both are constant in time.

### 3.6 Backward Euler, stability and the linear solve

Backward Euler evaluates the right-hand side at the new time level:

```
(T^{n+1} - T^n)/dt = -K T^{n+1} + g     =>     (I + dt K) T^{n+1} = T^n + dt g
```

* **Order.** It is first order in time.
* **Stability.** It is unconditionally stable for this problem, whose eigenvalues
  are real and negative, so the step size is limited by accuracy only.
* **Why not explicit.** The explicit FTCS scheme needs r = a dt/dx^2 <= 1/2 in 1-D.
  The assignment's time step (below) gives r_x = 0.76 and r_y = 0.80 on the 13 x 12
  grid, so FTCS would not be stable there.
* **Linear solve.** I + dt K does not change from step to step (the course
  practical makes the same observation), so it is LU-factorised once by SciPy's
  sparse LU. Every step is then one forward and one back substitution.

### 3.7 Time step: Equation 3

```
dt = min(dx^2, dy^2) * rho * cp / max(kx, ky)
units: m^2 * (kg m^-3) * (J kg^-1 K^-1) / (W m^-1 K^-1) = s
```

Equation 3 does not generally divide 5 h into whole steps. The code therefore uses
n = ceil(5 h / dt_Eq3) equal steps of 5 h / n, which is never larger than the
Equation 3 value. On the 13 x 12 grid that is 14 steps of 1285.7 s, against
Equation 3's 1354.2 s.

Note that Equation 3 ties dt to dx^2. Refining the grid therefore also refines
the time step, and by a factor of four per halving. Every "grid" study below is
really a space-time study.

### 3.8 Mean temperature, heat input and the energy balance

**Mean temperature.** The composite trapezoidal rule, applied in x and then in y,
gives weights 1 inside, 1/2 on edges and 1/4 at corners:

```
T_avg = sum(w[i,j] T[i,j]) dx dy / (L W)
```

A plain average of nodal values would over-weight the boundary nodes, including
the hot heated-segment nodes.

**Boundary-flux heat input.** Take each unknown node's equation, multiply it by its
control-volume area w dx dy and by rho cp, then sum over all unknown nodes.

* An exchange between two unknown nodes appears twice, with opposite signs and
  equal weights, so it cancels. This is the discrete version of the divergence
  theorem.
* What remains is the heat conducted in from the heated-segment nodes. For an
  interior node below the top segment, for example, that is
  rho cp dx dy ay/dy^2 (T_seg - T) = ky dx (T_seg - T)/dy: Fourier's law across
  one face.

Backward Euler satisfies this balance exactly at every step:

```
change in stored energy of the unknown nodes = dt * (heat flow in at the new time level)
```

The run checks it in check V3, and sums the heat flows into the reported
boundary-flux heat input, Q_b.

**Heat-input accounting: three quantities** (revision of 2026-10-04). The heat
flows above cross the *inner* faces of the heated-segment nodes' control volumes,
half a cell inside the edge. The segment nodes' own control volumes - strips
dx wide and dy/2 thick on the edge (`heat2d.fixed_node_energy`) - are not unknowns:
they hold the boundary temperature from t = 0. So the energy in the plate has two
parts:

```
Q_b    boundary-flux heat input       = sum over time of dt x (heat flow from the segment nodes)
E_seg  heated-segment half-cell energy = rho cp sum over segment nodes of w dx dy (T_b - 300 K)
dE     stored-energy increase          = rho cp L W (T_avg - 300 K) = Q_b + E_seg
```

The last identity holds because the trapezoidal weights of the mean temperature are
the control-volume areas of *all* nodes, unknown and prescribed. It is checked to
round-off in every revision run (check S1). Since every other edge is insulated, dE
is the total heat the discrete plate has taken in through the segments.

* E_seg is proportional to dy. It is 58.0 MJ/m on 13 x 12 and 3.2 MJ/m on
  193 x 177, halving with each refinement (ratios 2.14, 2.07, 2.04, 2.02).
* So Q_b and dE converge towards each other from opposite sides: Q_b is 5.5 % low
  on the 13 x 12 grid (507.3 against 537.1 MJ/m), while dE, and the mean
  temperature with it, is 4.6 % high (565.3 against 540.3 MJ/m). The two
  percentages describe different quantities; neither is "the" grid effect on the
  heat taken in. The mean-temperature difference, +6.4 K, is the same +4.6 % of
  the 138.5 K temperature rise.
* The original version of this section said the gap between the two was "the
  energy of the heated-segment nodes' own half-cells", which is right, but its
  README compared only Q_b between grids. Section 7.1 gives the full table.

## 4. Implementation map

| Step | Code |
|---|---|
| inputs (all from the assignment) | `run_project.PLATE`, `run_project.plate_boundary` |
| grid and trapezoidal weights | `heat2d.Grid`, `heat2d.area_average` |
| ghost nodes and the five-point stencil | `heat2d._stencil` |
| K, g (as C times the prescribed temperatures) | `heat2d.assemble` |
| Equation 3 and the whole number of steps | `heat2d.time_step_eq3`, `heat2d.steps_to_reach` |
| backward Euler with one LU factorisation | `heat2d.backward_euler` |
| heat flow and stored energy | `heat2d.heat_from_fixed_nodes`, `heat2d.stored_energy` |
| one plate run and its diagnostics | `run_project.solve_plate` |
| checks V1-V5, grid studies, figures, summary | `run_project.main` and the `verify_*` / `figure_*` functions |
| V5 reference solution of the discrete scheme | `run_project.aniso_reference`, `run_project.verify_anisotropy` |
| half-cell energy of the prescribed nodes | `heat2d.fixed_node_energy` |
| runs with a fixed number of steps; per-node heat input | `run_project.solve_plate(..., n_steps=)` |
| heat accounting (section 7.1) | `run_studies.accounting` |
| spatial and temporal studies, split of the coarse-grid difference (7.2) | `run_studies.main` (part B), `run_studies.split_difference` |
| nested-grid comparison and boundary intervals (7.3) | `run_studies.matched`, `run_studies.temperature_differences`, `run_studies.segment_heat` |
| checks S1-S3, figures 5 and 6 | `run_studies.main`, `run_studies.figure_*` |
| comparison with the recorded results (CI) | `compare_results.py` |

## 5. Verification

The notebooks distinguish **verification** (are the equations solved correctly?)
from **validation** (are they the right equations for a real system, judged
against measurements?). Everything here is verification. The five checks, V1 to
V5, are chosen to cover different failure modes. The run log counts six passes,
because V2's space and time studies are reported and judged separately.

**V1: course worked solution.**

* *Set-up.* The tutorial's implicit solution for a 10 cm aluminium rod: dx = 2 cm,
  dt = 0.1 s, diffusivity 0.835 cm^2/s, ends at 100 C and 50 C, initially 0 C. It
  runs on the same 2-D code, with two node rows and insulated top and bottom, so
  the solution does not vary in y. The code reproduces the tutorial's
  lambda = 0.020875.
* *Result.* The largest difference from the 16 tabulated values (nodes 1-4 at
  0.1-0.4 s) is 4.7e-5 C, within the rounding of the four-decimal table.
* *Establishes.* The implicit matrix, the right-hand-side update and the
  prescribed-node handling match the course's hand calculation.
* *Does not establish.* Anything about 2-D, anisotropy, insulated boundaries or
  accuracy under refinement.

**V2: exact solution.**

* *Set-up.* The same equation, with the plate's properties, on a simplified
  problem: insulated left and right edges, T = 300 K on the whole top and bottom
  edges. It has the exact solution

  ```
  T = 300 + 100 cos(pi x/L) sin(pi y/W) exp(-gamma t),   gamma = pi^2 (ax/L^2 + ay/W^2)
  ```

  which satisfies the PDE and both boundary types (check by substitution). It is a
  manufactured solution whose source term happens to be zero.
* *Space refinement* (9 to 65 nodes per side, dt = 0.25 s): the RMS error falls
  with observed order 1.97 (maximum norm 1.97). The time step is not small enough
  for the time error to be negligible; see *Time-step sensitivity* below.
* *Time refinement* (129 x 129 nodes, dt = 900 to 56.25 s): order 0.98.
* *Comparison.* The same space study with the one-sided insulated boundary gives
  order 0.96.
* *Establishes.* Second-order central differences and the ghost-node boundary, and
  first-order backward Euler: the orders the theory predicts, for smooth data.
* *Time-step sensitivity.* The spatial study is run at dt = 0.25 s (the recorded
  benchmark, order 1.971) and again at dt = 0.0625 s (order 1.989). The finest-grid
  RMS error falls by 4 %, from 2.506e-3 to 2.405e-3 K, so part of the benchmark's
  error is time error and the benchmark order is slightly depressed by it. Both
  sets of runs are in `results/verification_exact_space.csv`.
* *Does not establish.* Two things. First, second-order convergence for the
  partially heated plate, whose boundary condition changes type at the four segment
  ends; the plate's convergence is examined separately, in the grid study
  (section 6). Second, the orientation of the anisotropy: the test uses a square
  domain, a square grid and the same wavenumber in each direction, so alpha_x and
  alpha_y enter the discrete solution symmetrically. Exchanging them changes no V2
  error by more than 1.5e-8 relative - about eight significant figures, which is
  round-off for these runs - and the fitted order by 3e-9. V5 closes that gap.

**V3: energy balance.**

* *Result.* In every plate run, the change in stored energy equals the heat
  conducted in to a relative 5.2e-12 at worst.
* *Establishes.* The discrete scheme conserves energy, and the heat-flow
  bookkeeping (J/m against W/m x s) is consistent.
* *Does not establish.* That the heat input is accurate: a conservative scheme can
  still be inaccurate.

**V4: units.**

* *Set-up.* The 13 x 12 case is solved again in centimetres and hours: L = 100 cm,
  k in J h^-1 cm^-1 K^-1, rho in kg cm^-3, t = 5 h.
* *Result.* Equation 3 gives the same 1354.17 s. The temperature fields agree to
  1.0e-12 K, and the heat inputs, converted to J/m, to 2e-15.
* *Establishes.* The code has no hidden assumption about units, such as a
  hard-coded 3600 or 1 m.
* *Does not establish.* Anything about accuracy: the check compares the code with
  itself in two unit systems.

**V5: anisotropy orientation (exact solution of the discrete scheme).**

* *Why.* V1 is one-dimensional, V3 is an identity satisfied by any coefficients
  and V4 compares the code with itself, so before V5 an exchange of alpha_x and
  alpha_y passed every check. Injecting the exchange into `heat2d.assemble` and
  running the whole script confirms it: V1 to V4 still pass, while the base-case
  mean temperature moves from 444.947 to 434.512 K. That fault-injection run is a
  one-off experiment, not part of the recorded run, so its numbers are not in
  `results/`; V5's own control, which is, appears below.
* *Set-up.* A 33 x 17 grid on the same unit square, so dx = 1/32 and dy = 1/16.
  Insulated left and right edges, T = 300 K on the whole top and bottom edges, and
  the initial field T = 300 + 100 cos(pi x/L) sin(2 pi y/W) K. The Equation 3 step
  gives 19 steps of 189.47 s to reach 1 h.
* *The reference, derived by hand.* Substituting the grid function
  phi[i,j] = cos(pi x_i/L) sin(2 pi y_j/W) into the five-point stencil gives, at
  every node,

  ```
  (K phi)[i,j] = lambda_h phi[i,j],
  lambda_h = 4 ax sin^2(pi dx / 2L) / dx^2 + 4 ay sin^2(pi dy / W) / dy^2
  ```

  which is the von Neumann symbol of the scheme (Topic 6). The relation also holds
  on the insulated edges, where the mirrored ghost node reproduces the cosine
  exactly, and the prescribed rows sit where the sine vanishes, so the boundary
  values are exactly 300 K. Backward Euler multiplies an eigenvector by
  1/(1 + dt lambda_h) each step, so

  ```
  T^n = 300 + 100 phi (1 + dt lambda_h)^-n
  ```

  is the exact solution of the discrete system. It is computed from the formula
  alone, without the assembled matrix or the linear solve.
* *Result.* The solver matches that reference to 1.6e-12 K (max norm), which is
  round-off for 19 backward Euler solves. Repeating the run with alpha_x and
  alpha_y exchanged in the assembly, against the same reference, gives 4.7 K.
* *Tolerance.* 1e-9 K: about 600 times the observed round-off and nine orders of
  magnitude below the error produced by the exchange, so the check has no useful
  freedom to pass for the wrong reason. The check requires both the correct run to
  pass and the exchanged run to fail.
* *Establishes.* That alpha_x multiplies the x-derivative and alpha_y the
  y-derivative, and that the assembled operator and the time stepping reproduce the
  scheme they are supposed to implement, to round-off, on a grid where the two
  directions are not interchangeable.
* *Does not establish.* Anything about the accuracy of the discretisation itself:
  it compares the code with the exact solution of the discrete equations, not of
  the PDE. V2 covers that.

**Study checks S1-S3 (revision).** `run_studies.py` adds three checks on its own
calculations. S1: in all 29 of its runs the stored-energy increase equals Q_b +
E_seg, and the V3 energy balance closes, to 5.8e-12 at worst (tolerance 1e-10).
S2: the nested-grid comparisons use identical node coordinates (mismatch 0 m), the
boundary-interval sums equal the segment totals (1.2e-15 relative) and the segment
ends lie on nodes. S3: halving the spatial study's fixed time step changes the
193 x 177 mean temperature by 0.0035 K, 0.70 % of the smallest grid-to-grid change
(tolerance 1 %). These check bookkeeping and study design; they say nothing about
the accuracy of the plate results.

## 6. Results

### Temperature field (figure 1)

After 5 h the whole plate is above 375 K: with a diffusion length of about 0.3 m
from each of the two segments, no part of the 1 m plate is still at its initial
temperature. The warmest region is
above the bottom segment's centre (700 K). The coolest points are on the left and
right edges at y = 0.61 m (375.5 K on the finest grid). The field is symmetric about
x = L/2, as the boundary conditions are.

| After 5 h | 13 x 12 nodes (14 steps) | 193 x 177 nodes (3403 steps) |
|---|---|---|
| mean temperature | 444.95 K | 438.54 K |
| boundary-flux heat input, top / bottom / total | 181.9 / 325.4 / 507.3 MJ/m | 190.9 / 346.3 / 537.1 MJ/m |
| bottom corners (x = 0 and L, y = 0) | 465.88 K | 453.76 K |
| top corners (x = 0 and L, y = W) | 406.59 K | 395.05 K |

About two thirds of the heat enters through the bottom segment, which is hotter:
612-700 K along its length against 500 K for the top one.

### Grid sensitivity (figure 3)

**(a) The assignment's study** refines nx only (13, 17, 33, 65, 73) with ny = 12:

* The mean temperature changes by 0.048, 0.066, 0.023 and 0.002 %.
* The 0.2 % criterion is met immediately, at nx = 17.
* But the y-spacing never changes, so the study shows only the effect of refining
  dx (and, through Equation 3, dt) at a fixed and rather coarse dy = 1/11 m.
* The last step (65 to 73 nodes) reduces dx by only 11 %, which by itself makes a
  small change likely.

**(b) nx and ny refined together**, dx and dy halved each time, with dt from
Equation 3 (so quartered):

| Nodes | 13 x 12 | 25 x 23 | 49 x 45 | 97 x 89 | 193 x 177 |
|---|---|---|---|---|---|
| mean temperature (K) | 444.947 | 441.745 | 439.959 | 439.023 | 438.544 |
| change (Eq. 4, %) | – | 0.72 | 0.41 | 0.21 | 0.11 |
| boundary-flux heat input (MJ/m) | 507.3 | 525.7 | 532.7 | 535.8 | 537.1 |

Successive estimates first agree within 0.2 % at 193 x 177, whose values are the
most refined numerical estimates, not exact answers. Compared with them, the
13 x 12 run (dt = 1286 s) gives a mean temperature 6.4 K higher (1.5 % of 438.5 K),
a boundary-flux heat input 5.5 % lower and a stored-energy increase 4.6 % higher
(section 3.8).

**(c) Observed convergence rate.**

* Successive changes in (b) fall by factors of 1.79, 1.91 and 1.96, approaching 2.
  Each step halves dx and dy and quarters dt, so this is first-order convergence in
  the grid spacing for the combined space-time refinement. A second-order spatial
  error, or backward Euler's first-order error in dt, would each fall by a factor of
  about 4 per step.
* As a diagnostic, the heated segments are extended over the full top and bottom
  edges. This removes the four points where the boundary condition changes type
  part-way along an edge; it still changes type at the corners, as in V2. The
  ratios for the mean temperature then become 3.64 and 3.87, approaching 4.
* This is consistent with the segment ends causing the first-order behaviour of the
  mean temperature, but it does not prove it, because the extension also changes
  the heated length and the boundary temperatures. In the diagnostic, Q_b still
  changes at first order (by 49.2, 23.4 and 11.3 MJ/m) while dE changes by 5.9, 1.6
  and 0.4 MJ/m: the half-cell energy cancels the first-order part of Q_b. With the
  partial segments that cancellation is incomplete.
* Section 7.2 separates the grid and time-step parts of these changes, and section
  7.3 maps where the grids differ.

### Centreline and history (figure 2)

* **Centreline.** The profile at x = L/2 changes little with refinement. At the 12
  heights both grids share, it differs by at most 3.2 K between the 13 x 12 and
  193 x 177 grids, against 6.4 K for the mean temperature.
* **Boundary-flux heat input.** Enters fastest at first, then more slowly:
  231 MJ/m after 1.25 h and 537 MJ/m after 5 h, roughly proportional to t^0.6.
* **Mean temperature.** Rises from 300 K to 438.5 K.

## 7. Revision studies (October 2026)

`python run_studies.py` reruns the same plate - inputs, boundary conditions, solver -
with chosen grids and time steps, and writes the tables named below. All
comparisons are **differences between numerical results**, not errors: there is no
exact solution of the plate problem, and the 193 x 177 grid is only the most
refined one. Settings and their reasons are in `SOURCE_MAP.md` section 4.

### 7.1 Heat-input accounting (`results/heat_accounting.csv`, figure 5 left)

Nested grids, Equation 3 time step, MJ per metre of depth:

| Grid | 13 x 12 | 25 x 23 | 49 x 45 | 97 x 89 | 193 x 177 |
|---|---|---|---|---|---|
| boundary-flux heat input Q_b | 507.3 | 525.7 | 532.7 | 535.8 | 537.1 |
| heated-segment half-cell energy E_seg | 58.0 | 27.1 | 13.1 | 6.4 | 3.2 |
| stored-energy increase dE = Q_b + E_seg | 565.3 | 552.8 | 545.8 | 542.2 | 540.3 |
| mean temperature (K) | 444.95 | 441.74 | 439.96 | 439.02 | 438.54 |

13 x 12 against 193 x 177: Q_b -29.8 MJ/m (-5.5 %), E_seg +54.8 MJ/m, dE +25.0 MJ/m
(+4.6 %). The definitions and the reason for the difference are in section 3.8.

### 7.2 Space and time separated (`results/space_time_study.csv`, `results/coarse_grid_difference.csv`, figure 5)

**Design.**

* *Spatial study.* The five nested grids at a fixed dt = 5.2895 s (3403 steps to
  5 h). This is the 193 x 177 grid's Equation 3 step, the smallest in the original
  study, so the finest member of the spatial study is the original 193 x 177 run,
  reused. Check S3 confirms the step is small enough: halving it changes the
  193 x 177 mean temperature by 0.0035 K, 0.70 % of the smallest grid-to-grid change
  (0.50 K). The same halving changes Q_b by 1.06 % of its smallest grid-to-grid
  change. The study is repeated at dt = 1285.71 s, the 13 x 12 grid's Equation 3
  step (14 steps).
* *Temporal study.* On the fixed 193 x 177 grid: dt = 1285.71 s halved seven times
  to 10.04 s, then 5.29 s and 2.64 s (14, 28, ..., 1792, 3403 and 6806 steps). The
  same steps on the 13 x 12 grid show how the time-step effect depends on the grid.
  193 x 177 is the finest grid of the original study; the time-step effect at
  dt = 1285.71 s differs by 0.011 K (0.7 %) between it and 97 x 89.

**Spatial study, dt = 5.29 s.**

| Grid | 13 x 12 | 25 x 23 | 49 x 45 | 97 x 89 | 193 x 177 |
|---|---|---|---|---|---|
| mean temperature (K) | 446.773 | 442.199 | 440.067 | 439.044 | 438.544 |
| change from previous grid (K) | - | -4.574 | -2.132 | -1.023 | -0.500 |
| ratio of successive changes | - | - | 2.15 | 2.08 | 2.05 |
| bottom / top corners (K) | 468.91 / 407.87 | 460.39 / 400.72 | 456.53 / 397.42 | 454.67 / 395.83 | 453.76 / 395.05 |
| Q_b (MJ/m) | 514.5 | 527.4 | 533.2 | 535.8 | 537.1 |
| dE (MJ/m) | 572.4 | 554.6 | 546.3 | 542.3 | 540.3 |

At dt = 1285.71 s the changes are -4.501, -2.090, -1.000 and -0.489 K (ratios 2.15,
2.09, 2.05). Either way the mean temperature changes at about **first order in the
grid spacing**. The first-order part is visible in both energy terms: E_seg halves
with each refinement, and Q_b changes by 13.0, 5.7, 2.7 and 1.3 MJ/m.

**Temporal study, 193 x 177.** Halving dt from 1285.71 s changes the mean
temperature by +0.836, +0.422, +0.212, +0.106, +0.053, +0.027 and +0.013 K (ratios
1.98-2.00): **first order in the time step**, as V2 found for backward Euler on the
smooth problem. The next two steps (1792 -> 3403 -> 6806) change it by 0.0063 and
0.0035 K. On 13 x 12 the changes are 9 % larger (+0.910, +0.460, ...), with the same
ratios.

**Why the original joint study shows ratios of 1.79-1.96.** Each joint step halves
dx and quarters dt. The joint changes (-3.202, -1.786, -0.937, -0.479 K) are the
spatial changes at a fixed dt plus a time-step part of the opposite sign, roughly
+1.37, +0.35, +0.09 and +0.02 K. That part shrinks four-fold per step, because it is
first order in dt and dt is quartered. As it fades, the ratio approaches the spatial
value of 2.

**Splitting the coarse-grid difference.** For 13 x 12 at dt = 1285.71 s minus
193 x 177 at dt = 5.2895 s, two exact identities hold:

```
total = grid effect at the fine dt   + time-step effect on the coarse grid
      = grid effect at the coarse dt + time-step effect on the fine grid
```

| Quantity | total | grid at 5.29 s | time step on 13 x 12 | grid at 1286 s | time step on 193 x 177 | interaction |
|---|---|---|---|---|---|---|
| mean temperature (K) | +6.403 | +8.229 | -1.826 | +8.081 | -1.677 | -0.149 |
| bottom corners (K) | +12.125 | +15.150 | -3.025 | +15.060 | -2.935 | -0.089 |
| top corners (K) | +11.541 | +12.818 | -1.277 | +12.609 | -1.067 | -0.210 |
| Q_b (MJ/m) | -29.80 | -22.68 | -7.12 | -23.26 | -6.54 | -0.58 |
| E_seg (MJ/m) | +54.77 | +54.77 | 0 | +54.77 | 0 | 0 |
| dE (MJ/m) | +24.97 | +32.09 | -7.12 | +31.51 | -6.54 | -0.58 |

The interaction is the difference between the two orders: the time-step effect
depends on the grid (for the mean temperature, -1.826, -1.753, -1.711, -1.689 and
-1.677 K from 13 x 12 to 193 x 177). The grid and time-step effects are therefore not
exactly additive. On this problem the interaction is small (2 % of the grid effect),
but that is a measurement for these grids and steps, not a general property.

### 7.3 Segment ends (`results/segment_end_temperatures.csv`, `results/segment_heat_input.csv`, figure 6)

All five grids at dt = 5.2895 s, so the differences come from the grid. The grids are
nested: each grid's nodes are nodes of every finer grid at exactly the same
coordinates (check S2). Physical boundary locations are therefore the same on every
grid; in particular the segment ends x = L/4 and 3L/4 are nodes on all of them.

**Temperature at matching nodes.** T(grid) - T(193 x 177) at the grid's own nodes.
"Near an end" means within 0.1 m of one of the four segment ends. The field is
symmetric about x = L/2, so each largest difference occurs at a mirror-image pair of
nodes whose values agree to round-off (at most 1.1e-12 relative); the right-hand node
is reported, so the location does not depend on the CPU's floating-point kernels.

| Grid | nodes | largest difference (x, y) | within 0.1 m of an end | elsewhere: largest / RMS | mean |
|---|---|---|---|---|---|
| 13 x 12 | 156 | +19.71 K (0.833, 0) | 19.71 K | 16.10 / 8.20 K | +7.54 K |
| 25 x 23 | 575 | +11.68 K (0.792, 0) | 11.68 K | 7.60 / 3.71 K | +3.51 K |
| 49 x 45 | 2205 | +6.73 K (0.771, 0) | 6.73 K | 3.33 / 1.56 K | +1.49 K |
| 97 x 89 | 8633 | +3.07 K (0.760, 0) | 3.07 K | 1.09 / 0.51 K | +0.49 K |

* On every grid the largest difference is at the first insulated node beyond the
  end of the bottom segment, on the bottom edge (y = 0), where the segment
  temperature is 612 K. The coarser grid is warmer there.
* Between *successive* grids (each against the next finer one) the largest
  difference more than 0.1 m from an end is 9.10, 4.46, 2.23 and 1.09 K: it halves
  with each refinement. Within 0.1 m of an end it is 10.86, 6.75, 4.48 and 3.07 K,
  falling by only 1.61, 1.51 and 1.46 times per refinement. The ends therefore
  dominate more and more: between the two finest grids the largest near-end
  difference is 2.8 times the largest elsewhere.
* Figure 6 (left) shows T(97 x 89) - T(193 x 177) over the plate: up to 3.1 K at the
  four ends, about 0.5 K over most of the plate. The middle panel shows the largest
  difference in 0.05 m bands of distance from the nearest end.

**Boundary heat input along the segments.** The heat each segment node supplies
over 5 h (its share of Q_b) is summed over the same seven stretches of edge on every
grid: one interval around each 13 x 12 segment node, so the two end intervals are
L/24 = 0.042 m long and the others L/12 = 0.083 m. A node on the border between two
intervals is shared equally (it occurs only on the finer grids).

| Bottom segment, minus 193 x 177 (MJ/m) | 13 x 12 | 25 x 23 | 49 x 45 | 97 x 89 |
|---|---|---|---|---|
| each end interval | +15.32 | +7.99 | +3.50 | +1.18 |
| other intervals | -8.53 to -10.83 | -4.20 to -5.37 | -1.82 to -2.30 | -0.61 to -0.77 |

| Top segment, minus 193 x 177 (MJ/m) | 13 x 12 | 25 x 23 | 49 x 45 | 97 x 89 |
|---|---|---|---|---|
| each end interval | +8.64 | +4.11 | +1.79 | +0.60 |
| other intervals | -3.31 to -6.19 | -1.51 to -2.83 | -0.65 to -1.19 | -0.22 to -0.39 |

* Coarse grids put more heat in through the end intervals and less along the rest
  of each segment. Per metre of edge, the end-interval difference is about three
  times the largest elsewhere (bottom, 13 x 12: 368 against 130 MJ/m per m).
* Between successive grids these interval differences shrink about two-fold per
  refinement, in the end intervals and elsewhere.
* The four end nodes supply 38.8 % of Q_b on 13 x 12 and 8.7 % on 193 x 177. Each
  end node's control volume reaches half a cell beyond the segment end, over the
  insulated part of the edge, and also exchanges heat sideways with its insulated
  neighbour.
* The interior intervals' first-order differences in Q_b are largely the
  half-cell effect of section 3.8 (heat measured dy/2 inside the edge). Adding each
  interval's half-cell energy reduces them by a factor of about 2 to 3, while the
  end intervals' differences grow (bottom, 13 x 12: from +15.3 to +19.8 MJ/m).
  `segment_heat_input.csv` gives both columns.

**Interpretation.** At each segment end the boundary condition changes abruptly
along a straight edge, from a prescribed temperature to zero heat flow, so the
temperature need not be smooth there. The Taylor-series argument behind the
second-order stencil, and V2's evidence, assume a smooth solution. The
measurements agree with this picture: the largest differences sit at the ends and
shrink more slowly there. Removing the ends (the full-edge diagnostic, section 6)
restores near-second-order changes in the mean temperature. The local behaviour
near an end is not analysed, and no exact local solution is used.

### 7.4 Reproducibility and continuous integration

* The unmodified 2026-09-26 code was re-run first, on Python 3.11.15 with the
  original NumPy, SciPy and Matplotlib versions. It passed 6 of 6 checks and
  reproduced its 12 numerical output files exactly and its four figures byte for
  byte (`results/published_2026-09-26/rerun_2026-10-04/`).
* `compare_results.py` compares two result folders value by value: relative
  tolerance 1e-9, absolute tolerance 1e-9 in each value's unit, text exactly. Its
  docstring gives the reasons for the tolerances. Logs, environment records,
  timestamps, run times and figure pixels are excluded.
* `.github/workflows/heat-conduction.yml` runs on changes to this project only. It
  installs `requirements-lock.txt` on Python 3.11, runs both scripts (V1-V5 and
  S1-S3 must pass), compares the outputs with `results/` and checks that the
  original outputs are still reproduced.

## 8. Limitations

See the README. In short:

* conduction only, with the assignment's properties, in 2-D;
* first-order changes of the mean temperature in the grid spacing at a fixed time
  step, and in the time step on a fixed grid; second-order accuracy is shown only
  for the smooth V2 problem;
* the largest grid differences at the segment ends, with a measured but
  unanalysed local mechanism;
* the early transient after the sudden heating is not resolved separately; the
  temporal study measures only its effect at 5 h;
* a grid criterion that measures successive changes, not the error, and no error
  estimate or extrapolated value for the 193 x 177 results;
* no validation data: the course materials contain no measurements.
