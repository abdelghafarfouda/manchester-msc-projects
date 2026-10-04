"""Helper functions for transient 2-D anisotropic heat conduction.

Model (course Assignment (3); see SOURCE_MAP.md)::

    rho*cp * dT/dt = kx * d2T/dx2 + ky * d2T/dy2      on [0, Lx] x [0, Ly]

Every boundary node either has a prescribed temperature (Dirichlet) or is
insulated (zero normal temperature gradient).

Discretisation (details and reasons in docs/WALKTHROUGH.md):

* uniform vertex-centred grid: x_i = i*dx, i = 0..nx-1, dx = Lx/(nx-1), so the
  first and last nodes lie on the boundary (the course codes use the same grid);
* second-order central differences in space (three-point stencil per direction);
* an insulated boundary node uses a ghost node mirrored across the boundary,
  T_ghost = T_opposite.  This is the central difference of dT/dn = 0, second-order
  like the interior stencil.  The one-sided alternative T_ghost = T_node (the
  course hint) is kept only so the verification can compare the two;
* method of lines: dT/dt = -K T + g for the unknown nodes, integrated with
  backward (implicit) Euler, (I + dt*K) T^{n+1} = T^n + dt*g.  K and g do not
  change in time, so I + dt*K is LU-factorised once and reused every step.

Fields are numpy arrays of shape (ny, nx) indexed [j, i]; the linear node index
is k = j*nx + i (C order, the same as field.ravel()).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    """Uniform vertex-centred grid with nx x ny nodes on [0, Lx] x [0, Ly]."""

    nx: int
    ny: int
    Lx: float
    Ly: float

    def __post_init__(self):
        if self.nx < 3 or self.ny < 2:
            raise ValueError("need nx >= 3 and ny >= 2 nodes")

    @property
    def dx(self) -> float:
        return self.Lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.ny - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, self.Lx, self.nx)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(0.0, self.Ly, self.ny)

    def weights(self) -> np.ndarray:
        """Composite trapezoidal-rule weights (1 inside, 1/2 on edges, 1/4 at corners).

        They are also the areas of the nodes' control volumes divided by dx*dy,
        which is why the same weights appear in the energy balance.
        """
        wx = np.ones(self.nx)
        wx[[0, -1]] = 0.5
        wy = np.ones(self.ny)
        wy[[0, -1]] = 0.5
        return np.outer(wy, wx)


def area_average(grid: Grid, field: np.ndarray) -> float:
    """Area-averaged value of a nodal field (2-D composite trapezoidal rule)."""
    total = np.sum(grid.weights() * field) * grid.dx * grid.dy
    return float(total / (grid.Lx * grid.Ly))


# ----------------------------------------------------------------- operator
@dataclass
class Operator:
    """Semi-discrete system dT_free/dt = -K T_free + C T_fixed."""

    grid: Grid
    free: np.ndarray      # linear indices of unknown nodes
    fixed: np.ndarray     # linear indices of prescribed-temperature nodes
    T_fixed: np.ndarray   # prescribed temperatures at `fixed` (K)
    K: sp.csr_matrix      # (n_free, n_free), 1/s
    C: sp.csr_matrix      # (n_free, n_fixed), 1/s

    @property
    def g(self) -> np.ndarray:
        return self.C @ self.T_fixed

    def full_field(self, T_free: np.ndarray) -> np.ndarray:
        """Unknown and prescribed values combined into a (ny, nx) field."""
        T = np.empty(self.grid.nx * self.grid.ny)
        T[self.free] = T_free
        T[self.fixed] = self.T_fixed
        return T.reshape(self.grid.ny, self.grid.nx)


def _stencil(i, j, nx, ny, cx, cy, neumann):
    """Neighbours of node (i, j) with their coefficients (1/s).

    A neighbour outside the grid is a ghost node on an insulated boundary.
    "central":   (T_ghost - T_opposite)/(2 dx) = 0  ->  T_ghost = T_opposite,
                 so the ghost's coefficient is added to the opposite neighbour.
    "one_sided": (T_ghost - T_node)/dx = 0  ->  T_ghost = T_node, so the term
                 vanishes (first-order; used only in the verification).
    """
    neighbours = []
    for di, dj, c in ((-1, 0, cx), (1, 0, cx), (0, -1, cy), (0, 1, cy)):
        ii, jj = i + di, j + dj
        if 0 <= ii < nx and 0 <= jj < ny:
            neighbours.append((ii, jj, c))
        elif neumann == "central":
            neighbours.append((i - di, j - dj, c))
        elif neumann != "one_sided":
            raise ValueError(f"unknown insulated-boundary treatment {neumann!r}")
    return neighbours


def assemble(grid: Grid, alpha_x: float, alpha_y: float, dirichlet: np.ndarray,
             neumann: str = "central") -> Operator:
    """Assemble K and C for the unknown nodes.

    dirichlet : (ny, nx) array holding the prescribed temperature at Dirichlet
                nodes and NaN elsewhere; every other boundary node is insulated.
    alpha_x, alpha_y : thermal diffusivities k/(rho*cp), m^2/s.
    """
    nx, ny = grid.nx, grid.ny
    is_fixed = np.isfinite(dirichlet).ravel()
    free = np.flatnonzero(~is_fixed)
    fixed = np.flatnonzero(is_fixed)
    row_of = np.full(nx * ny, -1)
    row_of[free] = np.arange(free.size)
    col_of = np.full(nx * ny, -1)
    col_of[fixed] = np.arange(fixed.size)

    cx, cy = alpha_x / grid.dx ** 2, alpha_y / grid.dy ** 2
    k_rows, k_cols, k_vals = [], [], []
    c_rows, c_cols, c_vals = [], [], []
    for k in free:
        j, i = divmod(int(k), nx)
        diagonal = 0.0
        for ii, jj, c in _stencil(i, j, nx, ny, cx, cy, neumann):
            kn = jj * nx + ii
            diagonal += c
            if is_fixed[kn]:
                c_rows.append(row_of[k]); c_cols.append(col_of[kn]); c_vals.append(c)
            else:
                k_rows.append(row_of[k]); k_cols.append(row_of[kn]); k_vals.append(-c)
        k_rows.append(row_of[k]); k_cols.append(row_of[k]); k_vals.append(diagonal)

    K = sp.csr_matrix((k_vals, (k_rows, k_cols)), shape=(free.size, free.size))
    C = sp.csr_matrix((c_vals, (c_rows, c_cols)), shape=(free.size, fixed.size))
    return Operator(grid, free, fixed, dirichlet.ravel()[fixed].copy(), K, C)


# -------------------------------------------------------------- time stepping
def time_step_eq3(grid: Grid, kx: float, ky: float, rho: float, cp: float) -> float:
    """Assignment (3), Equation 3: dt = min(dx^2, dy^2) * rho * cp / max(kx, ky)."""
    return min(grid.dx ** 2, grid.dy ** 2) * rho * cp / max(kx, ky)


def steps_to_reach(t_end: float, dt_max: float) -> tuple[int, float]:
    """Smallest number of equal steps with dt <= dt_max that lands exactly on t_end."""
    n = int(np.ceil(t_end / dt_max - 1e-9))
    return n, t_end / n


def backward_euler(op: Operator, T0: np.ndarray, dt: float, n_steps: int,
                   observe=None) -> np.ndarray:
    """Integrate dT/dt = -K T + g with backward Euler; return the final (ny, nx) field.

    observe(n, T_free) is called at n = 0 and after every step, e.g. to record
    heat flows.  The matrix I + dt*K is factorised once by sparse LU
    decomposition and reused; each step is then one forward and one back
    substitution.  (SciPy's splu reorders the unknowns to limit fill-in; that
    changes the cost, not the solution.)
    """
    n = op.K.shape[0]
    lu = spla.splu((sp.identity(n, format="csc") + dt * op.K).tocsc())
    rhs_source = dt * op.g
    T = T0.ravel()[op.free].astype(float).copy()
    if observe is not None:
        observe(0, T)
    for step in range(1, n_steps + 1):
        T = lu.solve(T + rhs_source)
        if observe is not None:
            observe(step, T)
    return op.full_field(T)


# ----------------------------------------------------- heat flow and energy
def heat_from_fixed_nodes(op: Operator, T_free: np.ndarray, rho_cp: float) -> np.ndarray:
    """Conductive heat flow (W per metre of depth) from each prescribed node into
    the unknown nodes' control volumes, using the solver's own discrete fluxes."""
    g = op.grid
    w = g.weights().ravel()[op.free]
    Cw = sp.diags(w) @ op.C
    into_free = op.T_fixed * np.asarray(Cw.sum(axis=0)).ravel() - Cw.T @ T_free
    return rho_cp * g.dx * g.dy * into_free


def stored_energy(op: Operator, T_free: np.ndarray, rho_cp: float) -> float:
    """rho*cp * integral of T over the unknown nodes' control volumes (J per metre of depth)."""
    g = op.grid
    w = g.weights().ravel()[op.free]
    return float(rho_cp * g.dx * g.dy * np.sum(w * T_free))


def fixed_node_energy(op: Operator, rho_cp: float, T_ref: float) -> np.ndarray:
    """rho*cp * (T - T_ref) over each prescribed node's own control volume (J per
    metre of depth).  These half-cells (quarter-cells at corners) lie on the edge
    and hold the boundary temperature; their area is proportional to the grid
    spacing normal to the edge."""
    g = op.grid
    w = g.weights().ravel()[op.fixed]
    return rho_cp * g.dx * g.dy * w * (op.T_fixed - T_ref)


# -------------------------------------------------------------- error norms
def rms(e) -> float:
    """Root-mean-square norm (two-norm divided by sqrt(N)), so grids of
    different sizes can be compared."""
    e = np.asarray(e, dtype=float).ravel()
    return float(np.sqrt(np.mean(e ** 2)))


def fitted_order(h, err) -> float:
    """Observed order: least-squares slope of log(err) against log(h)."""
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])
