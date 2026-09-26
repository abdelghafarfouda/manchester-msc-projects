"""Transient 2-D anisotropic heat conduction in a 1 m x 1 m plate.

The problem is Assignment (3) of an MSc module in numerical modelling and
simulation; SOURCE_MAP.md lists the source of every input, equation, method and
benchmark.  Run the whole study with

    python run_project.py

The script works top to bottom and writes everything to results/:

  1. Problem definition  - inputs exactly as given in the assignment
  2. Verification        - V1 course worked solution, V2 exact solution (smooth,
                           square grid), V3 energy balance, V4 units,
                           V5 anisotropy orientation (exact discrete solution)
  3. Base case           - the assignment's 13 x 12 grid
  4. Grid sensitivity    - the assignment's nx study, then nx, ny (and dt)
                           refined together
  5. Results             - CSV tables, four figures, summary.json, run log

It exits with status 1 if any verification check fails.
"""

from __future__ import annotations

import csv
import datetime as dt_
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import heat2d as h  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

# --------------------------------------------------------------------------
# 1. Problem definition - every value below is given in Assignment (3)
# --------------------------------------------------------------------------
PLATE = dict(
    Lx=1.0, Ly=1.0,          # m, domain length and width
    kx=16.0, ky=20.0,        # W/(m K), thermal conductivity in x and y
    rho=7800.0,              # kg/m^3, density
    cp=500.0,                # J/(kg K), specific heat capacity
    T_initial=300.0,         # K, initial temperature everywhere
    T_top=500.0,             # K, top (green) segment
    t_end=5 * 3600.0,        # s, total simulated time (5 h)
)
# Bottom (red) segment: T = 300 sin(pi x / L) + 400  [K].  Both heated segments
# are centred and half the domain width long; every other boundary is insulated.
BASE_GRID = (13, 12)                           # nodes in x and y (assignment)
COURSE_NX = (13, 17, 33, 65, 73)               # assignment Q5, ny fixed at 12
JOINT_GRIDS = ((13, 12), (25, 23), (49, 45), (97, 89), (193, 177))  # dx, dy halved, dt quartered
CRITERION_PERCENT = 0.2                        # assignment Q5, Equation 4

# The same plate in centimetres and hours (J, kg, K unchanged) for the units check.
PLATE_CM_H = dict(PLATE, Lx=100.0, Ly=100.0,
                  kx=16.0 * 3600 / 100, ky=20.0 * 3600 / 100,   # J/(h cm K)
                  rho=7800.0e-6,                              # kg/cm^3
                  t_end=5.0)                                  # h

# Plot styling (validated categorical and ordinal colours; ink and grid tones).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORDINAL = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")


class Log:
    """Print to the console and to results/run_log.txt at the same time."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text)
        self.file.write(text + "\n")
        self.file.flush()


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------
# The plate problem
# --------------------------------------------------------------------------
def plate_boundary(grid: h.Grid, p: dict, full_edges: bool = False) -> np.ndarray:
    """Prescribed temperatures on the heated segments; NaN = insulated or interior.

    full_edges=True extends both heated segments over the whole top and bottom
    edges.  It is used only as a numerical diagnostic (section 4c), to test whether
    the four segment ends could explain the observed convergence rate; it is not a
    second application.
    """
    if (grid.nx - 1) % 4:
        raise ValueError("nx - 1 must be a multiple of 4 so that each heated "
                         "segment (x from L/4 to 3L/4) starts and ends on a node")
    D = np.full((grid.ny, grid.nx), np.nan)
    x = grid.x
    on_segment = (x >= 0.25 * grid.Lx - 1e-9 * grid.Lx) & (x <= 0.75 * grid.Lx + 1e-9 * grid.Lx)
    if full_edges:
        on_segment[:] = True
    D[-1, on_segment] = p["T_top"]
    D[0, on_segment] = 300.0 * np.sin(np.pi * x[on_segment] / grid.Lx) + 400.0
    return D


def centreline(field: np.ndarray) -> np.ndarray:
    """Temperature on the vertical line x = L/2 (the middle node for odd nx,
    the average of the two middle nodes for even nx)."""
    nx = field.shape[1]
    if nx % 2:
        return field[:, nx // 2].copy()
    return 0.5 * (field[:, nx // 2 - 1] + field[:, nx // 2])


def solve_plate(nx: int, ny: int, p: dict = PLATE, keep_history: bool = False,
                full_edges: bool = False) -> dict:
    """Run the plate problem with the Equation 3 time step on an nx x ny grid."""
    grid = h.Grid(nx, ny, p["Lx"], p["Ly"])
    rho_cp = p["rho"] * p["cp"]
    ax, ay = p["kx"] / rho_cp, p["ky"] / rho_cp
    op = h.assemble(grid, ax, ay, plate_boundary(grid, p, full_edges))
    dt_eq3 = h.time_step_eq3(grid, p["kx"], p["ky"], p["rho"], p["cp"])
    n_steps, dt = h.steps_to_reach(p["t_end"], dt_eq3)
    on_top = (op.fixed // nx) == ny - 1          # prescribed nodes on the top edge

    energy, q_top, q_bottom, t_avg = [], [], [], []

    def observe(n, T_free):
        q = h.heat_from_fixed_nodes(op, T_free, rho_cp)
        q_top.append(q[on_top].sum())
        q_bottom.append(q[~on_top].sum())
        energy.append(h.stored_energy(op, T_free, rho_cp))
        if keep_history:
            # at t = 0 the whole plate is at the initial temperature
            t_avg.append(p["T_initial"] if n == 0 else h.area_average(grid, op.full_field(T_free)))

    T0 = np.full((ny, nx), p["T_initial"])
    t0 = time.perf_counter()
    T = h.backward_euler(op, T0, dt, n_steps, observe)
    wall = time.perf_counter() - t0

    q_top, q_bottom = np.array(q_top), np.array(q_bottom)
    H_top, H_bottom = dt * q_top[1:].sum(), dt * q_bottom[1:].sum()   # backward Euler
    change_in_storage = energy[-1] - energy[0]
    result = dict(
        grid=grid, T=T, dt_eq3=dt_eq3, dt=dt, n_steps=n_steps,
        r_x=ax * dt / grid.dx ** 2, r_y=ay * dt / grid.dy ** 2,
        T_avg=h.area_average(grid, T), H_top=H_top, H_bottom=H_bottom,
        energy_closure=(change_in_storage - (H_top + H_bottom)) / (H_top + H_bottom),
        corners=dict(bottom_left=T[0, 0], bottom_right=T[0, -1],
                     top_left=T[-1, 0], top_right=T[-1, -1]),
        centreline=centreline(T), wall_s=wall,
    )
    if keep_history:
        result["history"] = dict(
            t=np.arange(n_steps + 1) * dt, T_avg=np.array(t_avg),
            Q_top=q_top, Q_bottom=q_bottom,
            H_top=np.concatenate([[0.0], np.cumsum(dt * q_top[1:])]),
            H_bottom=np.concatenate([[0.0], np.cumsum(dt * q_bottom[1:])]))
    return result


def relative_change_percent(new: float, old: float) -> float:
    """Assignment (3), Equation 4: |T_avg_new - T_avg_old| / T_avg_new x 100 %."""
    return abs(new - old) / abs(new) * 100.0


# --------------------------------------------------------------------------
# 2. Verification
# --------------------------------------------------------------------------
# V1: Tutorial "Finite difference method / PDE", Question 2 - aluminium rod,
# 10 cm long, dx = 2 cm, dt = 0.1 s, diffusivity 0.835 cm^2/s, T(0) = 100 C,
# T(10 cm) = 50 C, initially 0 C.  Worked implicit solution at nodes 1-4:
ROD_REFERENCE = {
    0.1: (2.0047, 0.0406, 0.0209, 1.0023),
    0.2: (3.9305, 0.1190, 0.0618, 1.9653),
    0.3: (5.7815, 0.2325, 0.1219, 2.8909),
    0.4: (7.5612, 0.3787, 0.2004, 3.7810),
}


def verify_rod() -> dict:
    kappa, dx, dt = 0.835, 2.0, 0.1                     # cm^2/s, cm, s
    grid = h.Grid(6, 2, 10.0, 2.0)                      # 1-D rod: 2 node rows, insulated top/bottom
    D = np.full((2, 6), np.nan)
    D[:, 0], D[:, -1] = 100.0, 50.0
    op = h.assemble(grid, kappa, kappa, D)
    profiles = {}

    def observe(n, T_free):
        if n:
            profiles[round(n * dt, 10)] = op.full_field(T_free)[0].copy()

    h.backward_euler(op, np.zeros((2, 6)), dt, 4, observe)
    rows, worst = [], 0.0
    for t, ref in ROD_REFERENCE.items():
        for node, T_ref in enumerate(ref, start=1):
            T_code = profiles[t][node]
            worst = max(worst, abs(T_code - T_ref))
            rows.append((t, node, node * dx, f"{T_code:.8f}", T_ref, f"{T_code - T_ref:+.2e}"))
    return dict(lam=kappa * dt / dx ** 2, max_abs_diff=worst, rows=rows,
                profiles=profiles, x=grid.x)


# V2: exact solution of the same PDE with simplified boundaries (insulated
# left/right, T = T0 on the full top and bottom edges), plate properties:
#   T = T0 + A cos(pi x/L) sin(pi y/W) exp(-gamma t),
#   gamma = pi^2 (alpha_x/L^2 + alpha_y/W^2)   (check by substitution).
MODE_T0, MODE_A, MODE_T_END = 300.0, 100.0, 3600.0     # K, K, s


def mode_exact(grid: h.Grid, t: float, ax: float, ay: float) -> np.ndarray:
    X, Y = np.meshgrid(grid.x, grid.y)
    gamma = np.pi ** 2 * (ax / grid.Lx ** 2 + ay / grid.Ly ** 2)
    return (MODE_T0 + MODE_A * np.cos(np.pi * X / grid.Lx)
            * np.sin(np.pi * Y / grid.Ly) * np.exp(-gamma * t))


def mode_error(n: int, dt: float, neumann: str = "central") -> tuple[float, float]:
    rho_cp = PLATE["rho"] * PLATE["cp"]
    ax, ay = PLATE["kx"] / rho_cp, PLATE["ky"] / rho_cp
    grid = h.Grid(n, n, PLATE["Lx"], PLATE["Ly"])
    D = np.full((n, n), np.nan)
    D[0, :] = D[-1, :] = MODE_T0
    op = h.assemble(grid, ax, ay, D, neumann)
    steps = int(round(MODE_T_END / dt))
    T = h.backward_euler(op, mode_exact(grid, 0.0, ax, ay), MODE_T_END / steps, steps)
    e = T - mode_exact(grid, MODE_T_END, ax, ay)
    return h.rms(e), float(np.abs(e).max())


def verify_exact() -> dict:
    space_nodes, space_dt = (9, 17, 33, 65), 0.25      # dt small: spatial error dominates
    space_dt_check = 0.0625        # the same study four times finer in time, to size the time error
    time_nodes, time_dts = 129, (900.0, 450.0, 225.0, 112.5, 56.25)   # fine grid: time error dominates
    space = {}
    for treatment in ("central", "one_sided"):
        rows = [(n, 1.0 / (n - 1), space_dt, *mode_error(n, space_dt, treatment)) for n in space_nodes]
        dx = [r[1] for r in rows]
        space[treatment] = dict(rows=rows,
                                order_rms=h.fitted_order(dx, [r[3] for r in rows]),
                                order_max=h.fitted_order(dx, [r[4] for r in rows]))
    # Time-step sensitivity of the spatial order (reported, not part of the V2 tolerance).
    rows = [(n, 1.0 / (n - 1), space_dt_check, *mode_error(n, space_dt_check)) for n in space_nodes]
    dx = [r[1] for r in rows]
    space_fine_dt = dict(rows=rows, dt=space_dt_check,
                         order_rms=h.fitted_order(dx, [r[3] for r in rows]),
                         order_max=h.fitted_order(dx, [r[4] for r in rows]))
    rows = [(time_nodes, 1.0 / (time_nodes - 1), d, *mode_error(time_nodes, d)) for d in time_dts]
    temporal = dict(rows=rows, order_rms=h.fitted_order(time_dts, [r[3] for r in rows]),
                    order_max=h.fitted_order(time_dts, [r[4] for r in rows]))
    return dict(space=space, space_fine_dt=space_fine_dt, time=temporal)


# V5: anisotropy orientation.  On a grid with dx != dy, the grid function
#   phi[i,j] = cos(pi x_i / L) sin(2 pi y_j / W)
# is an exact eigenvector of the discretisation used here (five-point central
# differences, mirrored ghost nodes on the insulated x-edges, T = T0 prescribed on
# the whole y-edges, where sin vanishes).  Substituting it into the stencil gives
#   lambda_h = 4 alpha_x sin^2(pi dx / 2L) / dx^2 + 4 alpha_y sin^2(pi dy / W) / dy^2,
# the von Neumann symbol of the scheme, and backward Euler then damps it by exactly
# (1 + dt lambda_h) per step, so
#   T^n = T0 + A phi (1 + dt lambda_h)^(-n)
# is an independent reference: it uses only the analytic symbol, not the assembled
# matrix or the linear solve.  Unlike V2 (square grid, same wavenumber in x and y)
# the two directions enter differently, so exchanging alpha_x and alpha_y in the
# assembly changes the solution while the reference stays fixed.
ANISO_GRID, ANISO_MODE_Y = (33, 17), 2       # nodes (nx, ny); y wavenumber (x uses 1)
ANISO_TOL = 1e-9                             # K; round-off level is about 2e-12 K


def aniso_reference(grid: h.Grid, ax: float, ay: float, dt: float, n_steps: int) -> tuple[np.ndarray, float]:
    """Exact solution of the discrete scheme for the mode above, and its eigenvalue."""
    X, Y = np.meshgrid(grid.x, grid.y)
    phi = np.cos(np.pi * X / grid.Lx) * np.sin(ANISO_MODE_Y * np.pi * Y / grid.Ly)
    lam = (4.0 * ax / grid.dx ** 2 * np.sin(np.pi * grid.dx / (2 * grid.Lx)) ** 2
           + 4.0 * ay / grid.dy ** 2 * np.sin(ANISO_MODE_Y * np.pi * grid.dy / (2 * grid.Ly)) ** 2)
    return MODE_T0 + MODE_A * phi * (1.0 + dt * lam) ** (-n_steps), float(lam)


def verify_anisotropy() -> dict:
    nx, ny = ANISO_GRID
    rho_cp = PLATE["rho"] * PLATE["cp"]
    ax, ay = PLATE["kx"] / rho_cp, PLATE["ky"] / rho_cp
    grid = h.Grid(nx, ny, PLATE["Lx"], PLATE["Ly"])
    dt_eq3 = h.time_step_eq3(grid, PLATE["kx"], PLATE["ky"], PLATE["rho"], PLATE["cp"])
    n_steps, dt = h.steps_to_reach(MODE_T_END, dt_eq3)
    reference, lam = aniso_reference(grid, ax, ay, dt, n_steps)      # fixed for both cases below
    X, Y = np.meshgrid(grid.x, grid.y)
    T_start = MODE_T0 + MODE_A * np.cos(np.pi * X / grid.Lx) * np.sin(ANISO_MODE_Y * np.pi * Y / grid.Ly)
    D = np.full((ny, nx), np.nan)
    D[0, :] = D[-1, :] = MODE_T0
    rows = []
    for label, (a_x, a_y) in (("as implemented", (ax, ay)),
                              ("alpha_x and alpha_y exchanged", (ay, ax))):
        op = h.assemble(grid, a_x, a_y, D)
        e = h.backward_euler(op, T_start, dt, n_steps) - reference
        rows.append((label, nx, ny, f"{grid.dx:.6f}", f"{grid.dy:.6f}", f"{dt:.4f}", n_steps,
                     f"{lam:.6e}", f"{h.rms(e):.3e}", f"{np.abs(e).max():.3e}"))
    correct, exchanged = float(rows[0][-1]), float(rows[1][-1])
    return dict(rows=rows, lam=lam, dt=dt, n_steps=n_steps, nx=nx, ny=ny,
                max_abs_error=correct, max_abs_error_exchanged=exchanged,
                passed=correct <= ANISO_TOL < exchanged)


# V4: the same base case in (cm, h) instead of (m, s).
def verify_units(base_si: dict) -> dict:
    cm_h = solve_plate(*BASE_GRID, p=PLATE_CM_H)
    H_si = base_si["H_top"] + base_si["H_bottom"]                 # J per m of depth
    H_cm = (cm_h["H_top"] + cm_h["H_bottom"]) * 100.0             # J per cm -> J per m
    return dict(dt_eq3_s=base_si["dt_eq3"], dt_eq3_h_in_s=cm_h["dt_eq3"] * 3600.0,
                max_abs_dT=float(np.abs(base_si["T"] - cm_h["T"]).max()),
                H_rel_diff=abs(H_si - H_cm) / H_si)


# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_figure(ncols, width, height):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE,
                             constrained_layout=True)
    return fig, np.atleast_1d(axes)


def figure_fields(base: dict, fine: dict, path: Path) -> None:
    fig, axes = new_figure(2, 10.0, 4.4)
    levels = np.arange(300, 701, 25)
    for ax, run in zip(axes, (base, fine)):
        g, T = run["grid"], run["T"]
        filled = ax.contourf(g.x, g.y, T, levels=levels, cmap="YlOrRd")
        lines = ax.contour(g.x, g.y, T, levels=levels[::2], colors=INK, linewidths=0.5)
        ax.clabel(lines, fmt="%d", fontsize=7, colors=INK)
        ax.plot([0.25, 0.75], [g.Ly, g.Ly], color=BLUE, linewidth=4, solid_capstyle="butt")
        ax.plot([0.25, 0.75], [0.0, 0.0], color=BLUE, linewidth=4, solid_capstyle="butt")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        dt_text = f"{run['dt']:.0f}" if run["dt"] >= 100 else f"{run['dt']:.2f}"
        ax.set_title(f"{g.nx} x {g.ny} nodes, dt = {dt_text} s:  "
                     f"mean T = {run['T_avg']:.1f} K", fontsize=10, color=INK)
        ax.tick_params(colors=INK2, labelsize=9)
    bar = fig.colorbar(filled, ax=axes, shrink=0.9)
    bar.set_label("Temperature (K)", color=INK)
    fig.suptitle("Temperature after 5 h (blue bars: heated segments; all other "
                 "boundaries insulated)", fontsize=11, color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_history(fine: dict, joint: list, path: Path) -> None:
    fig, axes = new_figure(3, 14.0, 4.2)
    hist = fine["history"]
    hours = hist["t"] / 3600.0
    ax = axes[0]
    ax.plot(hours, hist["T_avg"], color=BLUE, linewidth=1.8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("area-averaged temperature (K)")
    ax.set_title("Mean plate temperature", fontsize=10)
    ax.annotate(f"{hist['T_avg'][-1]:.1f} K", (hours[-1], hist["T_avg"][-1]),
                textcoords="offset points", xytext=(-40, 6), color=INK, fontsize=9)
    ax = axes[1]
    for key, colour, label in (("H_bottom", ORANGE, "bottom segment"),
                               ("H_top", BLUE, "top segment")):
        ax.plot(hours, hist[key] / 1e6, color=colour, linewidth=1.8, label=label)
        ax.annotate(f"{hist[key][-1] / 1e6:.0f} MJ/m", (hours[-1], hist[key][-1] / 1e6),
                    textcoords="offset points", xytext=(-58, 6), color=INK, fontsize=9)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("cumulative heat input (MJ per m depth)")
    ax.set_title("Heat conducted in through each segment", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax = axes[2]
    for colour, run in zip(ORDINAL, joint):
        g = run["grid"]
        ax.plot(run["centreline"], g.y, color=colour, linewidth=1.8,
                label=f"{g.nx} x {g.ny}")
    ax.set_xlabel("temperature at x = 0.5 m after 5 h (K)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vertical centreline, refining the grid", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="nodes",
              title_fontsize=9)
    for ax in axes:
        style_axes(ax)
    fig.suptitle(f"Finest grid ({fine['grid'].nx} x {fine['grid'].ny} nodes) unless labelled",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def nx_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(values)))
    ax.set_xticklabels([str(v) for v in sorted(set(values))])
    ax.minorticks_off()


def figure_grid(course: list, joint: list, diagnostic: list, path: Path) -> None:
    fig, axes = new_figure(2, 11.5, 4.4)
    series = (
        (course, BLUE, "o", "nx refined, ny = 12 (assignment Q5)"),
        (joint, ORANGE, "s", "nx and ny refined together"),
    )
    ax = axes[0]
    for runs, colour, marker, label in series:
        ax.plot([r["grid"].nx for r in runs], [r["T_avg"] for r in runs], marker + "-",
                color=colour, linewidth=1.8, markersize=6, markeredgecolor=SURFACE, label=label)
    nx_ticks(ax, [r["grid"].nx for r in course + joint])
    ax.set_xlabel("nodes in x, nx")
    ax.set_ylabel("mean temperature after 5 h (K)")
    ax.set_title("Mean temperature against grid size", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax = axes[1]
    for runs, colour, marker, label in series + (
            (diagnostic, AQUA, "^", "diagnostic: heated over full edges"),):
        ax.plot([r["grid"].nx for r in runs[1:]], [r["change_percent"] for r in runs[1:]],
                marker + "-", color=colour, linewidth=1.8, markersize=6,
                markeredgecolor=SURFACE, label=label)
    ax.axhline(CRITERION_PERCENT, color=INK2, linewidth=1.0)
    ax.annotate("assignment criterion, 0.2 %", (17, CRITERION_PERCENT),
                textcoords="offset points", xytext=(0, 5), color=INK2, fontsize=9)
    nx_ticks(ax, [r["grid"].nx for r in course[1:] + joint[1:]])
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 6.0)                    # headroom for the legend
    ax.set_xlabel("nodes in x, nx")
    ax.set_ylabel("change from previous grid, Equation 4 (%)")
    ax.set_title("Change in mean temperature between successive grids", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    for ax in axes:
        style_axes(ax)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_verification(rod: dict, exact: dict, path: Path) -> None:
    fig, axes = new_figure(3, 14.5, 4.4)
    ax = axes[0]
    rod_colours = (ORDINAL[0], ORDINAL[1], ORDINAL[3], ORDINAL[4])
    for colour, t in zip(rod_colours, ROD_REFERENCE):
        ax.semilogy(rod["x"], rod["profiles"][t], color=colour, linewidth=1.8, label=f"t = {t:.1f} s")
        ax.semilogy(rod["x"][1:5], ROD_REFERENCE[t], "o", color=colour, markersize=6,
                    markeredgecolor=INK, markeredgewidth=0.8)
    ax.set_xticks(rod["x"])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("temperature (°C, log scale)")
    ax.set_title(f"V1 course worked solution (max difference {rod['max_abs_diff']:.1e} °C)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
              title="lines: this code; dots: tutorial", title_fontsize=8)
    ax = axes[1]
    for key, colour, marker, label in (("central", BLUE, "o", "central ghost node (used)"),
                                       ("one_sided", ORANGE, "s", "one-sided ghost node")):
        s = exact["space"][key]
        dx = [r[1] for r in s["rows"]]
        ax.loglog(dx, [r[3] for r in s["rows"]], marker + "-", color=colour, linewidth=1.8,
                  markersize=6, markeredgecolor=SURFACE,
                  label=f"{label}: order {s['order_rms']:.2f}")
    ax.set_xticks(dx)
    ax.set_xticklabels([f"1/{round(1 / d)}" for d in dx])
    ax.minorticks_off()
    ax.set_xlabel("grid spacing dx = dy (m)")
    ax.set_ylabel("RMS error at t = 1 h (K)")
    ax.set_title("V2 exact solution: space (dt = 0.25 s)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax = axes[2]
    s = exact["time"]
    dts = [r[2] for r in s["rows"]]
    ax.loglog(dts, [r[3] for r in s["rows"]], "o-", color=BLUE,
              linewidth=1.8, markersize=6, markeredgecolor=SURFACE,
              label=f"backward Euler: order {s['order_rms']:.2f}")
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.minorticks_off()
    ax.set_xlabel("time step dt (s)")
    ax.set_ylabel("RMS error at t = 1 h (K)")
    ax.set_title(f"V2 exact solution: time ({s['rows'][0][0]} x {s['rows'][0][0]} nodes)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    for ax in axes:
        style_axes(ax)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(exist_ok=True)
    log = Log(OUT / "run_log.txt")
    started = dt_.datetime.now(dt_.timezone.utc)
    clock = time.perf_counter()
    env = dict(
        started_utc=started.isoformat(timespec="seconds"),
        command="python " + " ".join(sys.argv),
        python=sys.version.split()[0], implementation=platform.python_implementation(),
        platform=platform.platform(), machine=platform.machine(),
        numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
    )
    log("Transient 2-D anisotropic heat conduction (course Assignment (3))")
    log(f"started {env['started_utc']} on {env['platform']}")
    log(f"Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, "
        f"matplotlib {env['matplotlib']}")

    # 1. problem --------------------------------------------------------------
    rho_cp = PLATE["rho"] * PLATE["cp"]
    ax, ay = PLATE["kx"] / rho_cp, PLATE["ky"] / rho_cp
    log("\n1. PROBLEM (all inputs from Assignment (3))")
    log(f"   plate {PLATE['Lx']} m x {PLATE['Ly']} m, kx = {PLATE['kx']}, ky = {PLATE['ky']} W/(m K), "
        f"rho = {PLATE['rho']} kg/m^3, cp = {PLATE['cp']} J/(kg K)")
    log(f"   alpha_x = {ax:.4e} m^2/s, alpha_y = {ay:.4e} m^2/s; T0 = {PLATE['T_initial']} K; "
        f"t_end = {PLATE['t_end']:.0f} s (5 h)")
    log("   top segment 500 K, bottom segment 300 sin(pi x/L) + 400 K, "
        "x in [L/4, 3L/4]; all other boundaries insulated")

    checks = []

    def check(name, value, tolerance, passed, establishes):
        checks.append(dict(check=name, value=value, tolerance=tolerance,
                           passed=bool(passed), establishes=establishes))
        log(f"   [{'PASS' if passed else 'FAIL'}] {name}: {value}  (tolerance {tolerance})")

    # 2. verification ---------------------------------------------------------
    log("\n2. VERIFICATION")
    rod = verify_rod()
    write_csv(OUT / "verification_rod.csv",
              ("t_s", "node", "x_cm", "T_this_code_C", "T_tutorial_C", "difference_C"), rod["rows"])
    check("V1 course worked solution (rod, 4 steps)", f"max |dT| = {rod['max_abs_diff']:.2e} C",
          "<= 5e-5 C (half the last tabulated digit)", rod["max_abs_diff"] <= 5e-5,
          "the implicit-Euler / central-difference assembly reproduces the course's hand solution")
    log(f"   (lambda = kappa dt / dx^2 = {rod['lam']:.6f}; the tutorial quotes 0.020875)")

    exact = verify_exact()
    rows = [(k, *r) for k in ("central", "one_sided") for r in exact["space"][k]["rows"]]
    rows += [("central", *r) for r in exact["space_fine_dt"]["rows"]]
    write_csv(OUT / "verification_exact_space.csv",
              ("insulated_boundary_treatment", "nodes_per_side", "dx_m", "dt_s", "rms_error_K",
               "max_error_K"), rows)
    write_csv(OUT / "verification_exact_time.csv",
              ("nodes_per_side", "dx_m", "dt_s", "rms_error_K", "max_error_K"), exact["time"]["rows"])
    c, o, t, cf = (exact["space"]["central"], exact["space"]["one_sided"], exact["time"],
                   exact["space_fine_dt"])
    check("V2 exact solution, spatial order (RMS)", f"{c['order_rms']:.3f} (max norm {c['order_max']:.3f})",
          "2 +/- 0.1", abs(c["order_rms"] - 2) <= 0.1,
          "second-order central differences and the ghost-node insulated boundary for smooth data; "
          "the square grid makes x and y interchangeable here, so this test cannot detect an "
          "exchange of alpha_x and alpha_y (V5 covers that)")
    check("V2 exact solution, temporal order (RMS)", f"{t['order_rms']:.3f} (max norm {t['order_max']:.3f})",
          "1 +/- 0.1", abs(t["order_rms"] - 1) <= 0.1, "first-order backward Euler")
    log(f"   (for comparison, the one-sided insulated-boundary treatment converges at order "
        f"{o['order_rms']:.3f} RMS / {o['order_max']:.3f} max)")
    log(f"   (time-step sensitivity of the spatial order: {c['order_rms']:.3f} at dt = "
        f"{c['rows'][0][2]:g} s, {cf['order_rms']:.3f} at dt = {cf['dt']:g} s; the finest-grid RMS "
        f"error falls from {c['rows'][-1][3]:.3e} to {cf['rows'][-1][3]:.3e} K)")

    aniso = verify_anisotropy()
    write_csv(OUT / "verification_anisotropy.csv",
              ("case", "nx", "ny", "dx_m", "dy_m", "dt_s", "steps", "lambda_h_per_s",
               "rms_error_K", "max_error_K"), aniso["rows"])
    check("V5 anisotropy orientation (exact discrete solution, dx != dy)",
          f"max |dT| = {aniso['max_abs_error']:.1e} K as implemented, "
          f"{aniso['max_abs_error_exchanged']:.1e} K with alpha_x and alpha_y exchanged",
          f"<= {ANISO_TOL:g} K as implemented, and the exchanged case must exceed it",
          aniso["passed"],
          "alpha_x and alpha_y act on the correct derivatives: the solver reproduces the analytic "
          "solution of the discrete scheme, and exchanging them in the assembly breaks the match")
    log(f"   (analytic eigenvalue lambda_h = {aniso['lam']:.6e} 1/s on {aniso['nx']} x {aniso['ny']} "
        f"nodes, {aniso['n_steps']} steps of {aniso['dt']:.4f} s)")

    # 3. base case -------------------------------------------------------------
    log("\n3. BASE CASE: assignment grid 13 x 12 nodes, Equation 3 time step")
    base = solve_plate(*BASE_GRID)
    log(f"   Equation 3 dt = {base['dt_eq3']:.2f} s -> {base['n_steps']} equal steps of "
        f"{base['dt']:.2f} s to reach exactly 5 h; r_x = {base['r_x']:.3f}, r_y = {base['r_y']:.3f}")
    log(f"   mean temperature after 5 h = {base['T_avg']:.3f} K")
    for name, value in base["corners"].items():
        log(f"   corner {name:12s} T = {value:.3f} K")
    log(f"   heat conducted in: top {base['H_top'] / 1e6:.2f} MJ/m, bottom "
        f"{base['H_bottom'] / 1e6:.2f} MJ/m (per metre of depth)")

    units = verify_units(base)
    log(f"   units check: Equation 3 gives {units['dt_eq3_s']:.4f} s in (m, s) and "
        f"{units['dt_eq3_h_in_s']:.4f} s in (cm, h)")
    check("V4 units: same case in (cm, h) vs (m, s)",
          f"max |dT| = {units['max_abs_dT']:.1e} K, heat input rel. diff {units['H_rel_diff']:.1e}",
          "<= 1e-8 K and <= 1e-10", units["max_abs_dT"] <= 1e-8 and units["H_rel_diff"] <= 1e-10,
          "no hidden unit assumption; Equation 3 gives seconds (m^2 kg/m^3 J/(kg K) / W/(m K) = s)")

    # 4. grid sensitivity --------------------------------------------------------
    log("\n4. GRID SENSITIVITY (Equation 4, criterion 0.2 %)")
    log("   (a) assignment study: nx refined, ny = 12")
    course = []
    for nx in COURSE_NX:
        run = base if nx == BASE_GRID[0] else solve_plate(nx, 12)
        run["change_percent"] = (relative_change_percent(run["T_avg"], course[-1]["T_avg"])
                                 if course else float("nan"))
        course.append(run)
        log(f"      {nx:3d} x 12: dt = {run['dt']:8.2f} s, steps = {run['n_steps']:4d}, "
            f"T_avg = {run['T_avg']:.4f} K, change = {run['change_percent']:.4f} %")
    log("   (b) nx and ny refined together (dx, dy halved; dt from Equation 3)")
    joint = []
    for nx, ny in JOINT_GRIDS:
        run = base if (nx, ny) == BASE_GRID else solve_plate(nx, ny, keep_history=(nx, ny) == JOINT_GRIDS[-1])
        run["change_percent"] = (relative_change_percent(run["T_avg"], joint[-1]["T_avg"])
                                 if joint else float("nan"))
        joint.append(run)
        log(f"      {nx:3d} x {ny:3d}: dt = {run['dt']:8.3f} s, steps = {run['n_steps']:5d}, "
            f"T_avg = {run['T_avg']:.4f} K, change = {run['change_percent']:.4f} %, "
            f"heat in = {(run['H_top'] + run['H_bottom']) / 1e6:.2f} MJ/m, "
            f"energy closure = {run['energy_closure']:.1e}, {run['wall_s']:.1f} s")

    def first_meeting(runs):
        for run in runs[1:]:
            if run["change_percent"] < CRITERION_PERCENT:
                return run
        return None

    def change_ratios(runs):
        """Ratio of successive changes in T_avg when dx and dy are halved (Equation 3
        then quarters dt): about 4 for second-order and about 2 for first-order
        convergence in the grid spacing."""
        d = np.abs(np.diff([r["T_avg"] for r in runs]))
        return [float(v) for v in d[:-1] / d[1:]]

    course_ok, joint_ok = first_meeting(course), first_meeting(joint)
    log(f"   assignment study first meets 0.2 % at nx = "
        f"{course_ok['grid'].nx if course_ok else 'none'} (ny fixed at 12)")
    log(f"   joint refinement first meets 0.2 % at "
        f"{(str(joint_ok['grid'].nx) + ' x ' + str(joint_ok['grid'].ny)) if joint_ok else 'none'}")
    log(f"   ratios of successive changes (joint): "
        f"{', '.join(f'{v:.2f}' for v in change_ratios(joint))}  (2 = first order, 4 = second order)")

    log("   (c) diagnostic: same problem with both heated segments extended over the full")
    log("       top and bottom edges, which removes the four segment ends")
    diagnostic = []
    for nx, ny in JOINT_GRIDS[:4]:
        run = solve_plate(nx, ny, full_edges=True)
        run["change_percent"] = (relative_change_percent(run["T_avg"], diagnostic[-1]["T_avg"])
                                 if diagnostic else float("nan"))
        diagnostic.append(run)
        log(f"      {nx:3d} x {ny:3d}: T_avg = {run['T_avg']:.4f} K, change = {run['change_percent']:.4f} %")
    log(f"   ratios of successive changes (diagnostic): "
        f"{', '.join(f'{v:.2f}' for v in change_ratios(diagnostic))}")

    all_runs = course + joint[1:] + diagnostic
    worst_closure = max(abs(r["energy_closure"]) for r in all_runs)
    check("V3 energy balance (every plate run)", f"worst relative closure {worst_closure:.1e}",
          "<= 1e-10", worst_closure <= 1e-10,
          "the scheme conserves energy: storage change = heat conducted in through the segments")

    # 5. results -------------------------------------------------------------------
    fine = joint[-1]
    grid_rows = []
    for study, runs in (("assignment: nx refined, ny = 12", course), ("nx and ny refined together", joint),
                        ("diagnostic: heated segments over the full edges", diagnostic)):
        for r in runs:
            g = r["grid"]
            grid_rows.append((study, g.nx, g.ny, f"{g.dx:.6f}", f"{g.dy:.6f}", f"{r['dt_eq3']:.4f}",
                              f"{r['dt']:.4f}", r["n_steps"], f"{r['r_x']:.4f}", f"{r['r_y']:.4f}",
                              f"{r['T_avg']:.6f}", f"{r['change_percent']:.6f}",
                              f"{r['H_top'] / 1e6:.6f}", f"{r['H_bottom'] / 1e6:.6f}",
                              f"{r['energy_closure']:.3e}"))
    write_csv(OUT / "grid_sensitivity.csv",
              ("study", "nx", "ny", "dx_m", "dy_m", "dt_eq3_s", "dt_used_s", "steps", "r_x", "r_y",
               "T_avg_5h_K", "change_eq4_percent", "heat_in_top_MJ_per_m",
               "heat_in_bottom_MJ_per_m", "energy_closure_relative"), grid_rows)
    write_csv(OUT / "corner_temperatures_5h.csv", ("grid", "corner", "T_K"),
              [(f"{r['grid'].nx}x{r['grid'].ny}", k, f"{v:.4f}") for r in (base, fine)
               for k, v in r["corners"].items()])
    write_csv(OUT / "centreline_5h.csv", ("grid", "y_m", "T_K"),
              [(f"{r['grid'].nx}x{r['grid'].ny}", f"{y:.6f}", f"{T:.4f}") for r in joint
               for y, T in zip(r["grid"].y, r["centreline"])])
    hist = fine["history"]
    write_csv(OUT / f"history_{fine['grid'].nx}x{fine['grid'].ny}.csv",
              ("t_s", "T_avg_K", "Q_top_W_per_m", "Q_bottom_W_per_m", "H_top_J_per_m", "H_bottom_J_per_m"),
              [(f"{a:.3f}", f"{b:.6f}", f"{c_:.6e}", f"{d:.6e}", f"{e:.6e}", f"{f:.6e}")
               for a, b, c_, d, e, f in zip(hist["t"], hist["T_avg"], hist["Q_top"], hist["Q_bottom"],
                                            hist["H_top"], hist["H_bottom"])])
    np.savetxt(OUT / "field_5h_13x12.csv", base["T"], delimiter=",", fmt="%.6f",
               header="T (K) after 5 h; rows j = 0..11 (y = 0 to 1 m), columns i = 0..12 (x = 0 to 1 m)")
    np.savez_compressed(OUT / f"field_5h_{fine['grid'].nx}x{fine['grid'].ny}.npz",
                        x=fine["grid"].x, y=fine["grid"].y, T=fine["T"])
    checks.sort(key=lambda c_: c_["check"])          # V1 .. V4 in label order
    write_csv(OUT / "checks.csv", ("check", "value", "tolerance", "passed", "establishes"),
              [(c_["check"], c_["value"], c_["tolerance"], c_["passed"], c_["establishes"])
               for c_ in checks])

    figure_fields(base, fine, OUT / "fig1_temperature_5h.png")
    figure_history(fine, joint, OUT / "fig2_history_and_centreline.png")
    figure_grid(course, joint, diagnostic, OUT / "fig3_grid_sensitivity.png")
    figure_verification(rod, exact, OUT / "fig4_verification.png")

    def run_summary(r):
        g = r["grid"]
        return dict(nx=g.nx, ny=g.ny, dt_s=r["dt"], steps=r["n_steps"], T_avg_5h_K=r["T_avg"],
                    heat_in_top_MJ_per_m=r["H_top"] / 1e6, heat_in_bottom_MJ_per_m=r["H_bottom"] / 1e6,
                    corners_K=r["corners"], energy_closure=r["energy_closure"])

    wall = time.perf_counter() - clock
    env["finished_utc"] = dt_.datetime.now(dt_.timezone.utc).isoformat(timespec="seconds")
    env["wall_time_s"] = round(wall, 1)
    summary = dict(
        question=("Temperature field, mean temperature and heat uptake of the plate after 5 h, "
                  "and the grid needed for the mean temperature to change by < 0.2 % (Equation 4)"),
        inputs=dict(PLATE, bottom_segment="300 sin(pi x / L) + 400 K"),
        base_case_13x12=run_summary(base),
        finest_grid=run_summary(fine),
        assignment_study_first_below_0p2_percent_nx=course_ok["grid"].nx if course_ok else None,
        joint_refinement_first_below_0p2_percent=(
            [joint_ok["grid"].nx, joint_ok["grid"].ny] if joint_ok else None),
        joint_refinement_changes_percent=[r["change_percent"] for r in joint[1:]],
        joint_refinement_change_ratios=change_ratios(joint),
        diagnostic_full_edges_change_ratios=change_ratios(diagnostic),
        verification=dict(
            rod_max_abs_diff_C=rod["max_abs_diff"],
            exact_space_order_rms=c["order_rms"], exact_space_order_max=c["order_max"],
            exact_space_order_rms_one_sided=o["order_rms"],
            exact_space_dt_sensitivity=dict(
                benchmark_dt_s=c["rows"][0][2], benchmark_order_rms=c["order_rms"],
                finer_dt_s=cf["dt"], finer_order_rms=cf["order_rms"],
                finest_grid_rms_error_K=dict(benchmark=c["rows"][-1][3], finer=cf["rows"][-1][3])),
            exact_time_order_rms=t["order_rms"], exact_time_order_max=t["order_max"],
            worst_energy_closure=worst_closure, units=units,
            anisotropy=dict(nx=aniso["nx"], ny=aniso["ny"], steps=aniso["n_steps"], dt_s=aniso["dt"],
                            lambda_h_per_s=aniso["lam"], tolerance_K=ANISO_TOL,
                            max_abs_error_K=aniso["max_abs_error"],
                            max_abs_error_exchanged_K=aniso["max_abs_error_exchanged"])),
        checks=checks, environment=env)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(OUT / "environment.txt", "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}: {v}\n")

    n_fail = sum(not c_["passed"] for c_ in checks)
    log(f"\n5. RESULTS written to {OUT.name}/")
    log(f"   finest grid {fine['grid'].nx} x {fine['grid'].ny}: T_avg(5 h) = {fine['T_avg']:.3f} K, "
        f"heat in = {(fine['H_top'] + fine['H_bottom']) / 1e6:.1f} MJ/m "
        f"(top {fine['H_top'] / 1e6:.1f}, bottom {fine['H_bottom'] / 1e6:.1f})")
    log(f"   checks: {len(checks) - n_fail} of {len(checks)} passed; wall time {wall:.1f} s")
    log.file.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
