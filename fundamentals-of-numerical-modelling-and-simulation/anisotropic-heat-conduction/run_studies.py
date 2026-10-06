"""Studies added in the October 2026 revision of the anisotropic-plate project.

    python run_studies.py                 # writes to results/
    python run_studies.py --out DIR       # writes to DIR instead

The plate problem, its inputs, boundary conditions and solver are exactly those
of run_project.py (Assignment (3)); only the grids and time steps are varied.
The script is self-contained (it does not read run_project.py's outputs) and
works top to bottom:

  A. Heat-input accounting   - for the five nested grids with the Equation 3
                               time step: boundary-flux heat input, energy of the
                               heated-segment half-cells, and the stored-energy
                               increase, which is their sum
  B. Space and time separated - spatial study: nested grids at a fixed time step;
                               temporal study: time step halved on a fixed grid;
                               the coarse-grid difference split into the two
  C. Segment ends            - temperature and boundary heat input compared at
                               matching physical locations on the nested grids
  D. Checks S1-S3, CSV tables, two figures, studies_summary.json, log

Every comparison between grids here is a difference between numerical results,
not an error: no exact solution of the plate problem is available, and the
193 x 177 grid is only the most refined result.  It exits with status 1 if a
study check fails.
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import heat2d as h  # noqa: E402
import run_project as rp  # noqa: E402

P = rp.PLATE
RHO_CP = P["rho"] * P["cp"]                          # J/(m^3 K)
AREA = P["Lx"] * P["Ly"]                             # m^2 (energies are per metre of depth)
GRIDS = rp.JOINT_GRIDS                               # 13 x 12 ... 193 x 177, dx and dy halved each time
COARSE, FINE = GRIDS[0], GRIDS[-1]

# Fixed time step of the spatial study: 3403 equal steps to 5 h, dt = 5.2895 s.
# It is the Equation 3 step of the 193 x 177 grid, the smallest step of the
# original study, so the finest member of the spatial study is the original
# 193 x 177 run.  Check S3 tests that it is small enough.
N_FINE = 3403
# The Equation 3 step of the 13 x 12 assignment grid: 14 steps of 1285.71 s.
N_COARSE = 14
# Temporal study: dt halved seven times from 1285.71 s, then the fixed fine step
# and half of it.  Run on the finest grid, and on the assignment grid for contrast.
TEMPORAL_STEPS = tuple(N_COARSE * 2 ** k for k in range(8)) + (N_FINE, 2 * N_FINE)
TEMPORAL_GRIDS = (FINE, COARSE)

# Segment ends, as fractions of (Lx, Ly): the four points where a heated segment
# meets an insulated part of the same edge.
SEGMENT_ENDS = ((0.25, 0.0), (0.75, 0.0), (0.25, 1.0), (0.75, 1.0))
NEAR_END_M = 0.1                                     # "near an end": within 0.1 m of one
# The plate and its boundary conditions are symmetric about x = Lx/2, so the largest
# |T(coarse) - T(fine)| occurs at a mirror-image pair of nodes whose values differ only by
# round-off (at most 1.1e-12 relative, measured on every grid pair). Nodes within TIE_RTOL of
# the maximum are treated as equivalent, and the right-hand one (largest x, then smallest y)
# is reported, so the reported location does not depend on the CPU's floating-point kernels.
TIE_RTOL = 1e-9
DISTANCE_BAND_M = 0.05                               # distance bands in figure 6
# Boundary intervals for the heat input: one per heated node of the 13 x 12 grid
# (spacing L/12), so every grid is compared over the same stretches of edge.
SEGMENT_BIN_CENTRES = 0.25 + np.arange(7) / 12.0     # x / Lx

# Study checks, with tolerances fixed before the recorded run.
ACCOUNTING_TOL = 1e-10      # S1, relative; the same bound as V3
MATCH_TOL = 1e-12           # S2, m for coordinates and relative for heat sums
TIME_STEP_CRITERION = 0.01  # S3: halving the fixed step changes T_avg by < 1 % of the smallest grid change

SPACE_LABELS = {N_FINE: f"spatial, fixed dt = {P['t_end'] / N_FINE:.4f} s",
                N_COARSE: f"spatial, fixed dt = {P['t_end'] / N_COARSE:.2f} s"}


def grid_label(g) -> str:
    return f"{g[0]} x {g[1]}"


def eq3_steps(nx: int, ny: int) -> int:
    grid = h.Grid(nx, ny, P["Lx"], P["Ly"])
    return h.steps_to_reach(P["t_end"], h.time_step_eq3(grid, P["kx"], P["ky"], P["rho"], P["cp"]))[0]


class Runs:
    """Plate runs keyed by (nx, ny, steps); each one is solved once and reused."""

    def __init__(self, log):
        self.cache, self.log = {}, log

    def __call__(self, nx: int, ny: int, n_steps: int) -> dict:
        key = (nx, ny, n_steps)
        if key not in self.cache:
            run = rp.solve_plate(nx, ny, n_steps=n_steps)
            run.update(accounting(run))
            self.cache[key] = run
            self.log(f"      solved {nx:3d} x {ny:3d}, {n_steps:5d} steps of {run['dt']:9.4f} s: "
                     f"T_avg = {run['T_avg']:.4f} K ({run['wall_s']:.1f} s)")
        return self.cache[key]


# --------------------------------------------------------------------------
# A. Heat-input accounting
# --------------------------------------------------------------------------
def accounting(run: dict) -> dict:
    """The three energy quantities of one run, in J per metre of depth.

    Q   boundary-flux heat input: heat conducted from the heated-segment nodes
        into the rest of the plate over 5 h, from the solver's own fluxes.  It
        crosses the inner faces of the segment nodes' half-cells, dy/2 inside
        the edge (run_project.py reports it as "heat conducted in").
    E   heated-segment half-cell energy: rho cp (T_b - T0) over the segment
        nodes' own control volumes (dx by dy/2 strips on the edge), which hold
        the boundary temperature from t = 0.  Proportional to dy.
    dE  stored-energy increase of the whole plate, rho cp A (T_avg - T0), the
        quantity the mean temperature measures.  dE = Q + E to round-off.
    """
    Q = run["H_top"] + run["H_bottom"]
    E = run["E_seg_top"] + run["E_seg_bottom"]
    dE = RHO_CP * AREA * (run["T_avg"] - P["T_initial"])
    return dict(Q=Q, E_seg=E, dE=dE, accounting_closure=(dE - (Q + E)) / dE)


def accounting_row(run: dict) -> tuple:
    g = run["grid"]
    return (g.nx, g.ny, f"{run['dt']:.4f}", run["n_steps"], f"{run['T_avg']:.6f}",
            f"{run['H_top'] / 1e6:.6f}", f"{run['H_bottom'] / 1e6:.6f}", f"{run['Q'] / 1e6:.6f}",
            f"{run['E_seg_top'] / 1e6:.6f}", f"{run['E_seg_bottom'] / 1e6:.6f}",
            f"{run['E_seg'] / 1e6:.6f}", f"{run['dE'] / 1e6:.6f}", f"{run['accounting_closure']:.3e}")


ACCOUNTING_HEADER = ("nx", "ny", "dt_s", "steps", "T_avg_5h_K",
                     "boundary_flux_heat_input_top_MJ_per_m", "boundary_flux_heat_input_bottom_MJ_per_m",
                     "boundary_flux_heat_input_MJ_per_m", "segment_half_cell_energy_top_MJ_per_m",
                     "segment_half_cell_energy_bottom_MJ_per_m", "segment_half_cell_energy_MJ_per_m",
                     "stored_energy_increase_MJ_per_m", "accounting_closure_relative")


# --------------------------------------------------------------------------
# B. Space and time separated
# --------------------------------------------------------------------------
def changes_and_ratios(values, halvings):
    """Change from the previous level, and the ratio of successive changes where
    both steps halve the refined quantity (about 2 for first order, 4 for second)."""
    change = [float("nan")] + [b - a for a, b in zip(values[:-1], values[1:])]
    ratio = [float("nan")] * len(values)
    for k in range(2, len(values)):
        if halvings[k] and halvings[k - 1]:
            ratio[k] = change[k - 1] / change[k]
    return change, ratio


def study_rows(study: str, runs: list, halvings: list) -> list:
    change, ratio = changes_and_ratios([r["T_avg"] for r in runs], halvings)
    rows = []
    for r, c, q in zip(runs, change, ratio):
        g = r["grid"]
        rows.append((study, g.nx, g.ny, f"{g.dx:.6f}", f"{g.dy:.6f}", r["n_steps"], f"{r['dt']:.4f}",
                     f"{r['T_avg']:.6f}", f"{c:.6f}", f"{q:.4f}",
                     f"{r['corners']['bottom_left']:.6f}", f"{r['corners']['top_left']:.6f}",
                     f"{r['Q'] / 1e6:.6f}", f"{r['E_seg'] / 1e6:.6f}", f"{r['dE'] / 1e6:.6f}"))
    return rows


STUDY_HEADER = ("study", "nx", "ny", "dx_m", "dy_m", "steps", "dt_s", "T_avg_5h_K",
                "T_avg_change_from_previous_K", "ratio_of_successive_changes",
                "T_bottom_corners_K", "T_top_corners_K", "boundary_flux_heat_input_MJ_per_m",
                "segment_half_cell_energy_MJ_per_m", "stored_energy_increase_MJ_per_m")

QUANTITIES = (  # name in the tables, and how to read it from a run
    ("T_avg_5h_K", lambda r: r["T_avg"]),
    ("T_bottom_corners_K", lambda r: r["corners"]["bottom_left"]),
    ("T_top_corners_K", lambda r: r["corners"]["top_left"]),
    ("boundary_flux_heat_input_MJ_per_m", lambda r: r["Q"] / 1e6),
    ("segment_half_cell_energy_MJ_per_m", lambda r: r["E_seg"] / 1e6),
    ("stored_energy_increase_MJ_per_m", lambda r: r["dE"] / 1e6),
)


def split_difference(run) -> dict:
    """Split (13 x 12, Eq. 3 dt) minus (193 x 177, Eq. 3 dt) into grid and time-step parts.

    Two orders are possible; each is an exact identity:
        total = grid effect at the fine dt + time-step effect on the coarse grid
              = grid effect at the coarse dt + time-step effect on the fine grid.
    The two orders differ by the interaction, (time-step effect on the coarse
    grid) - (time-step effect on the fine grid), so the parts are not additive in
    general.
    """
    cc, cf = run(*COARSE, N_COARSE), run(*COARSE, N_FINE)        # coarse grid, coarse / fine dt
    fc, ff = run(*FINE, N_COARSE), run(*FINE, N_FINE)            # fine grid, coarse / fine dt
    out = {}
    for name, get in QUANTITIES:
        total = get(cc) - get(ff)
        grid_fine_dt, time_coarse_grid = get(cf) - get(ff), get(cc) - get(cf)
        grid_coarse_dt, time_fine_grid = get(cc) - get(fc), get(fc) - get(ff)
        out[name] = dict(coarse=get(cc), reference=get(ff), total=total,
                         grid_effect_at_fine_dt=grid_fine_dt,
                         time_step_effect_on_coarse_grid=time_coarse_grid,
                         grid_effect_at_coarse_dt=grid_coarse_dt,
                         time_step_effect_on_fine_grid=time_fine_grid,
                         interaction=time_coarse_grid - time_fine_grid)
    return out


# --------------------------------------------------------------------------
# C. Segment ends
# --------------------------------------------------------------------------
def matched(coarse: dict, fine: dict):
    """Fine-grid temperatures at the coarse grid's nodes (nested grids), and the
    largest coordinate mismatch, which must be round-off."""
    gc, gf = coarse["grid"], fine["grid"]
    sx, sy = (gf.nx - 1) // (gc.nx - 1), (gf.ny - 1) // (gc.ny - 1)
    if (gc.nx - 1) * sx != gf.nx - 1 or (gc.ny - 1) * sy != gf.ny - 1:
        raise ValueError("grids are not nested")
    mismatch = max(np.abs(gf.x[::sx] - gc.x).max(), np.abs(gf.y[::sy] - gc.y).max())
    return fine["T"][::sy, ::sx], float(mismatch)


def distance_to_segment_end(grid) -> np.ndarray:
    X, Y = np.meshgrid(grid.x, grid.y)
    d = [np.hypot(X - fx * grid.Lx, Y - fy * grid.Ly) for fx, fy in SEGMENT_ENDS]
    return np.min(d, axis=0)


def temperature_differences(coarse: dict, fine: dict) -> dict:
    """T(coarse) - T(fine) at the coarse grid's nodes: where it is largest."""
    T_fine, mismatch = matched(coarse, fine)
    d = coarse["T"] - T_fine
    g = coarse["grid"]
    near = distance_to_segment_end(g) <= NEAR_END_M
    ties = np.argwhere(np.abs(d) >= (1.0 - TIE_RTOL) * np.abs(d).max())   # (j, i) of equivalent maxima
    j, i = min(ties.tolist(), key=lambda ji: (-ji[1], ji[0]))               # largest x, then smallest y
    return dict(diff=d, mismatch=mismatch, nodes=d.size, near_nodes=int(near.sum()),
                max_abs=float(np.abs(d).max()), x_at_max=float(g.x[i]), y_at_max=float(g.y[j]),
                diff_at_max=float(d[j, i]), max_abs_near=float(np.abs(d[near]).max()),
                max_abs_elsewhere=float(np.abs(d[~near]).max()),
                rms_elsewhere=float(h.rms(d[~near])), mean=float(d.mean()))


def segment_bins() -> np.ndarray:
    """Edges of the boundary intervals (fractions of Lx): one interval around each
    heated node of the 13 x 12 grid, clipped to the segment [L/4, 3L/4]."""
    c = SEGMENT_BIN_CENTRES
    return np.concatenate([[0.25], 0.5 * (c[:-1] + c[1:]), [0.75]])


def bin_by_location(x: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum node values into the intervals [edges[k], edges[k+1]]; a node lying on
    the edge between two intervals is shared equally between them."""
    out = np.zeros(len(edges) - 1)
    for xi, v in zip(x, values):
        inside = [k for k in range(len(out)) if edges[k] - MATCH_TOL <= xi <= edges[k + 1] + MATCH_TOL]
        for k in inside:
            out[k] += v / len(inside)
    return out


def segment_heat(run: dict, edges_m: np.ndarray) -> dict:
    """Boundary-flux heat input and half-cell energy per boundary interval, per segment (J/m)."""
    out = {}
    for name, on_top in (("bottom", False), ("top", True)):
        m = run["fixed_on_top"] == on_top
        x = run["fixed_x"][m]
        out[name] = dict(Q=bin_by_location(x, run["H_nodes"][m], edges_m),
                         E=bin_by_location(x, run["E_nodes"][m], edges_m),
                         Q_total=run["H_top"] if on_top else run["H_bottom"],
                         x_nodes=x, Q_nodes=run["H_nodes"][m])
    return out


# --------------------------------------------------------------------------
# D. Figures
# --------------------------------------------------------------------------
def figure_space_time(accounting_runs, spatial, spatial_coarse_dt, temporal, path: Path) -> None:
    fig, axes = rp.new_figure(3, 15.0, 4.6)
    ax = axes[0]
    labels = [grid_label((r["grid"].nx, r["grid"].ny)) for r in accounting_runs]
    pos = np.arange(len(labels))
    Q = np.array([r["Q"] for r in accounting_runs]) / 1e6
    E = np.array([r["E_seg"] for r in accounting_runs]) / 1e6
    ax.bar(pos, Q, color=rp.BLUE, width=0.62, label="boundary-flux heat input")
    ax.bar(pos, E, bottom=Q, color=rp.ORANGE, width=0.62, label="heated-segment half-cell energy")
    for k in pos:
        ax.annotate(f"{Q[k]:.0f}", (k, Q[k] / 2), ha="center", va="center", color=rp.SURFACE, fontsize=8.5)
        ax.annotate(f"{Q[k] + E[k]:.0f}", (k, Q[k] + E[k]), textcoords="offset points", xytext=(0, 3),
                    ha="center", color=rp.INK, fontsize=8.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 680)
    ax.set_xlabel("grid (nodes in x by y), Equation 3 time step")
    ax.set_ylabel("energy after 5 h (MJ per m of depth)")
    ax.set_title("Heat accounting: bar height = stored-energy increase", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=rp.INK, loc="upper right")

    ax = axes[1]
    for runs, colour, marker, label in (
            (spatial, rp.BLUE, "o", f"fixed dt = {spatial[0]['dt']:.2f} s"),
            (spatial_coarse_dt, rp.ORANGE, "s", f"fixed dt = {spatial_coarse_dt[0]['dt']:.0f} s"),
            (accounting_runs, rp.MUTED, "^", "Equation 3 dt (original study)")):
        ax.plot([r["grid"].nx for r in runs], [r["T_avg"] for r in runs], marker + "-", color=colour,
                linewidth=1.8, markersize=6, markeredgecolor=rp.SURFACE, label=label)
    rp.nx_ticks(ax, [r["grid"].nx for r in spatial])
    ax.set_xlabel("nodes in x (dx and dy halved at each step)")
    ax.set_ylabel("mean temperature after 5 h (K)")
    ax.set_title("Spatial study: grid refined, time step fixed", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=rp.INK)

    ax = axes[2]
    for (grid, runs), colour, marker in zip(temporal.items(), (rp.BLUE, rp.ORANGE), ("o", "s")):
        ax.plot([r["dt"] for r in runs], [r["T_avg"] for r in runs], marker + "-", color=colour,
                linewidth=1.8, markersize=6, markeredgecolor=rp.SURFACE, label=f"{grid_label(grid)} nodes")
    ax.set_xscale("log")
    ax.set_xticks([1000, 100, 10])
    ax.set_xticklabels(["1000", "100", "10"])
    ax.invert_xaxis()
    ax.set_xlabel("time step dt (s, log scale; finer to the right)")
    ax.set_ylabel("mean temperature after 5 h (K)")
    ax.set_title("Temporal study: time step refined, grid fixed", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=rp.INK, loc="center right")
    for ax in axes:
        rp.style_axes(ax)
    fig.savefig(path, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)


def figure_segment_ends(spatial, temperature, heat, edges_m, path: Path) -> None:
    fig, axes = rp.new_figure(3, 15.5, 4.6)
    # (a) map of the difference between the two finest grids
    ax = axes[0]
    run = spatial[-2]
    g = run["grid"]
    d = temperature[-1]["diff"]
    lim = float(np.abs(d).max())
    mesh = ax.pcolormesh(g.x, g.y, d, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), shading="nearest")
    for fx, fy in SEGMENT_ENDS:
        ax.plot(fx * g.Lx, fy * g.Ly, "o", markersize=7, markerfacecolor="none", markeredgecolor=rp.INK)
    ax.plot([0.25, 0.75], [g.Ly, g.Ly], color=rp.INK, linewidth=3, solid_capstyle="butt")
    ax.plot([0.25, 0.75], [0.0, 0.0], color=rp.INK, linewidth=3, solid_capstyle="butt")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"T({g.nx} x {g.ny}) - T({spatial[-1]['grid'].nx} x {spatial[-1]['grid'].ny}) "
                 f"at shared nodes", fontsize=10)
    bar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    bar.set_label("difference between grids (K)", color=rp.INK)
    ax.tick_params(colors=rp.INK2, labelsize=9)

    # (b) largest |difference| in bands of distance from the nearest segment end
    ax = axes[1]
    bands = np.arange(0.0, 0.6 + 1e-9, DISTANCE_BAND_M)
    centres = 0.5 * (bands[:-1] + bands[1:])
    for run, t, colour in zip(spatial[:-1], temperature, rp.ORDINAL[:4]):
        dist = distance_to_segment_end(run["grid"]).ravel()
        diff = np.abs(t["diff"]).ravel()
        keep = diff > 0.0                 # leave out the prescribed nodes, equal on every grid
        dist, diff = dist[keep], diff[keep]
        band_max = [diff[(dist >= lo) & (dist < hi)].max() if np.any((dist >= lo) & (dist < hi))
                    else np.nan for lo, hi in zip(bands[:-1], bands[1:])]
        ax.plot(centres, band_max, "o-", color=colour, linewidth=1.8, markersize=5,
                markeredgecolor=rp.SURFACE, label=f"{run['grid'].nx} x {run['grid'].ny}")
    ax.axvline(NEAR_END_M, color=rp.INK2, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(f"distance from the nearest segment end (m; {DISTANCE_BAND_M:g} m bands)")
    ax.set_ylabel("largest |T(grid) - T(193 x 177)| in the band (K)")
    ax.set_title("Temperature differences between grids", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=rp.INK, title="grid compared", title_fontsize=8.5,
              loc="upper right")

    # (c) boundary-flux heat input per unit length of edge, bottom segment
    ax = axes[2]
    length = np.diff(edges_m)
    for run, hseg, colour in zip(spatial, heat, rp.ORDINAL):
        per_length = hseg["bottom"]["Q"] / length / 1e6
        ax.stairs(per_length, edges_m, color=colour, linewidth=1.8,
                  label=f"{run['grid'].nx} x {run['grid'].ny}")
    ax.set_xlim(edges_m[0] - 0.02, edges_m[-1] + 0.02)
    ax.set_xlabel("x along the bottom segment (m)")
    ax.set_ylabel("heat input per m of edge (MJ per m depth per m)")
    ax.set_title("Boundary-flux heat input, bottom segment", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=rp.INK, title="grid", title_fontsize=8.5, ncol=2,
              loc="lower center")
    for ax in axes[1:]:
        rp.style_axes(ax)
    fig.suptitle(f"Segment ends: differences between grids at matching locations "
                 f"(all runs dt = {spatial[0]['dt']:.2f} s)", fontsize=11, color=rp.INK)
    fig.savefig(path, dpi=150, facecolor=rp.SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="October 2026 revision studies: heat-input "
                                     "accounting, separate space and time studies, segment ends.")
    parser.add_argument("--out", type=Path, default=rp.OUT,
                        help="folder for the results (default: results/ beside this script)")
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    log = rp.Log(out / "studies_run_log.txt")
    started = dt_.datetime.now(dt_.timezone.utc)
    clock = time.perf_counter()
    env = dict(
        started_utc=started.isoformat(timespec="seconds"),
        command="python " + " ".join(sys.argv),
        python=sys.version.split()[0], implementation=platform.python_implementation(),
        platform=platform.platform(), machine=platform.machine(),
        numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
    )
    log("Anisotropic plate (course Assignment (3)): October 2026 revision studies")
    log(f"started {env['started_utc']} on {env['platform']}")
    log(f"Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, "
        f"matplotlib {env['matplotlib']}")
    log("Same plate, boundary conditions and solver as run_project.py; only grids and time steps vary.")
    run = Runs(log)
    checks = []

    def check(name, value, tolerance, passed, establishes):
        checks.append(dict(check=name, value=value, tolerance=tolerance,
                           passed=bool(passed), establishes=establishes))
        log(f"   [{'PASS' if passed else 'FAIL'}] {name}: {value}  (tolerance {tolerance})")

    # A. heat-input accounting --------------------------------------------------
    log("\nA. HEAT-INPUT ACCOUNTING (five nested grids, Equation 3 time step; MJ per m of depth)")
    eq3 = [run(nx, ny, eq3_steps(nx, ny)) for nx, ny in GRIDS]
    assert eq3[0]["n_steps"] == N_COARSE and eq3[-1]["n_steps"] == N_FINE
    for r in eq3:
        log(f"   {r['grid'].nx:3d} x {r['grid'].ny:3d}: boundary-flux heat input Q = {r['Q'] / 1e6:8.3f}, "
            f"half-cell energy E = {r['E_seg'] / 1e6:7.3f}, stored-energy increase dE = "
            f"{r['dE'] / 1e6:8.3f} (= rho cp A (T_avg - 300 K), T_avg = {r['T_avg']:.4f} K)")
    a0, a1 = eq3[0], eq3[-1]
    heat_comparison = {}
    for key, name in (("Q", "boundary_flux_heat_input"), ("E_seg", "segment_half_cell_energy"),
                      ("dE", "stored_energy_increase")):
        heat_comparison[name] = dict(
            grid_13x12_MJ_per_m=a0[key] / 1e6, grid_193x177_MJ_per_m=a1[key] / 1e6,
            difference_MJ_per_m=(a0[key] - a1[key]) / 1e6,
            difference_percent_of_193x177=(a0[key] - a1[key]) / a1[key] * 100.0)
    for name, c in heat_comparison.items():
        log(f"   13 x 12 vs 193 x 177, {name.replace('_', ' ')}: {c['grid_13x12_MJ_per_m']:.1f} vs "
            f"{c['grid_193x177_MJ_per_m']:.1f} MJ/m, difference {c['difference_MJ_per_m']:+.1f} MJ/m "
            f"({c['difference_percent_of_193x177']:+.1f} %)")
    half_cell_ratios = [float(a["E_seg"] / b["E_seg"]) for a, b in zip(eq3[:-1], eq3[1:])]
    log(f"   half-cell energy ratio per halving of dx and dy: "
        f"{', '.join(f'{v:.2f}' for v in half_cell_ratios)} (it is proportional to dy)")
    rp.write_csv(out / "heat_accounting.csv", ACCOUNTING_HEADER, [accounting_row(r) for r in eq3])

    # B. space and time separated ---------------------------------------------
    log("\nB. SPACE AND TIME SEPARATED")
    log(f"   (a) spatial study: nested grids at fixed dt = {P['t_end'] / N_FINE:.4f} s "
        f"({N_FINE} steps), and at fixed dt = {P['t_end'] / N_COARSE:.2f} s ({N_COARSE} steps)")
    spatial = [run(nx, ny, N_FINE) for nx, ny in GRIDS]
    spatial_coarse_dt = [run(nx, ny, N_COARSE) for nx, ny in GRIDS]
    log(f"   (b) temporal study: dt halved from {P['t_end'] / N_COARSE:.2f} s on the "
        f"{grid_label(FINE)} grid (and on {grid_label(COARSE)} for contrast)")
    temporal = {g: [run(*g, n) for n in TEMPORAL_STEPS] for g in TEMPORAL_GRIDS}

    rows = []
    for n, runs in ((N_FINE, spatial), (N_COARSE, spatial_coarse_dt)):
        rows += study_rows(SPACE_LABELS[n], runs, [True] * len(runs))
    t_halvings = [False] + [b == 2 * a for a, b in zip(TEMPORAL_STEPS[:-1], TEMPORAL_STEPS[1:])]
    for g, runs in temporal.items():
        rows += study_rows(f"temporal, fixed grid {grid_label(g)}", runs, t_halvings)
    rp.write_csv(out / "space_time_study.csv", STUDY_HEADER, rows)

    def log_study(label, runs, halvings):
        change, ratio = changes_and_ratios([r["T_avg"] for r in runs], halvings)
        log(f"   {label}: T_avg changes (K) {', '.join(f'{c:+.4f}' for c in change[1:])}")
        log(f"      ratios of successive changes {', '.join(f'{q:.2f}' for q in ratio if q == q)}")
        return change, ratio

    space_change, space_ratio = log_study(SPACE_LABELS[N_FINE], spatial, [True] * len(spatial))
    log_study(SPACE_LABELS[N_COARSE], spatial_coarse_dt, [True] * len(spatial))
    time_change, time_ratio = log_study(f"temporal, {grid_label(FINE)}", temporal[FINE], t_halvings)
    log_study(f"temporal, {grid_label(COARSE)}", temporal[COARSE], t_halvings)
    time_effect_by_grid = {grid_label(g): run(*g, N_COARSE)["T_avg"] - run(*g, N_FINE)["T_avg"]
                           for g in GRIDS}
    log(f"   time-step effect T_avg(dt = {P['t_end'] / N_COARSE:.2f} s) - T_avg(dt = "
        f"{P['t_end'] / N_FINE:.4f} s) by grid: "
        f"{', '.join(f'{k} {v:+.4f} K' for k, v in time_effect_by_grid.items())}")

    split = split_difference(run)
    rp.write_csv(out / "coarse_grid_difference.csv",
                 ("quantity", "coarse_13x12_eq3_dt", "reference_193x177_eq3_dt", "total_difference",
                  "grid_effect_at_fine_dt", "time_step_effect_on_coarse_grid", "grid_effect_at_coarse_dt",
                  "time_step_effect_on_fine_grid", "interaction"),
                 [(name, *(f"{v[k]:.6f}" for k in ("coarse", "reference", "total", "grid_effect_at_fine_dt",
                                                    "time_step_effect_on_coarse_grid",
                                                    "grid_effect_at_coarse_dt", "time_step_effect_on_fine_grid",
                                                    "interaction")))
                  for name, v in split.items()])
    s = split["T_avg_5h_K"]
    log(f"   (c) 13 x 12 (dt {P['t_end'] / N_COARSE:.2f} s) minus 193 x 177 (dt {P['t_end'] / N_FINE:.4f} s), "
        f"mean temperature: {s['total']:+.4f} K")
    log(f"       = grid effect at the fine dt {s['grid_effect_at_fine_dt']:+.4f} K"
        f" + time-step effect on 13 x 12 {s['time_step_effect_on_coarse_grid']:+.4f} K")
    log(f"       = grid effect at the coarse dt {s['grid_effect_at_coarse_dt']:+.4f} K"
        f" + time-step effect on 193 x 177 {s['time_step_effect_on_fine_grid']:+.4f} K")
    log(f"       interaction (difference between the two orders) {s['interaction']:+.4f} K")
    for name in ("boundary_flux_heat_input_MJ_per_m", "segment_half_cell_energy_MJ_per_m",
                 "stored_energy_increase_MJ_per_m"):
        v = split[name]
        log(f"       {name}: total {v['total']:+.3f} = grid {v['grid_effect_at_fine_dt']:+.3f} "
            f"+ time step {v['time_step_effect_on_coarse_grid']:+.3f}")

    finest_pair = abs(spatial[-1]["T_avg"] - spatial[-2]["T_avg"])
    halved = temporal[FINE][-1]
    dt_change = abs(halved["T_avg"] - spatial[-1]["T_avg"])
    dt_change_Q = abs(halved["Q"] - spatial[-1]["Q"]) / abs(spatial[-1]["Q"] - spatial[-2]["Q"])
    check("S3 fixed time step of the spatial study",
          f"halving dt to {halved['dt']:.4f} s on {grid_label(FINE)} changes T_avg by {dt_change:.2e} K, "
          f"{dt_change / finest_pair * 100:.2f} % of the smallest grid-to-grid change ({finest_pair:.4f} K)",
          f"< {TIME_STEP_CRITERION * 100:g} %", dt_change < TIME_STEP_CRITERION * finest_pair,
          "the spatial study's differences are not set by its time step; the same halving changes the "
          f"boundary-flux heat input by {dt_change_Q * 100:.2f} % of its smallest grid-to-grid change")

    # C. segment ends ------------------------------------------------------------
    log(f"\nC. SEGMENT ENDS (dt = {P['t_end'] / N_FINE:.4f} s; differences between grids at matching "
        f"locations, not errors)")
    temperature = [temperature_differences(r, spatial[-1]) for r in spatial[:-1]]
    successive = [temperature_differences(a, b) for a, b in zip(spatial[:-1], spatial[1:])]
    t_rows = []
    for label, items, pairs in (("193 x 177", temperature, [(r, spatial[-1]) for r in spatial[:-1]]),
                                ("next finer grid", successive, list(zip(spatial[:-1], spatial[1:])))):
        for t, (a, b) in zip(items, pairs):
            t_rows.append((f"{a['grid'].nx} x {a['grid'].ny}", f"{b['grid'].nx} x {b['grid'].ny}", t["nodes"],
                           t["near_nodes"], f"{t['max_abs']:.6f}", f"{t['x_at_max']:.6f}", f"{t['y_at_max']:.6f}",
                           f"{t['diff_at_max']:.6f}", f"{t['max_abs_near']:.6f}",
                           f"{t['max_abs_elsewhere']:.6f}", f"{t['rms_elsewhere']:.6f}", f"{t['mean']:.6f}"))
            if label == "193 x 177":
                log(f"   T({a['grid'].nx} x {a['grid'].ny}) - T(193 x 177) at {t['nodes']} shared nodes: "
                    f"max |d| {t['max_abs']:.3f} K at (x, y) = ({t['x_at_max']:.3f}, {t['y_at_max']:.3f}) m; "
                    f"within {NEAR_END_M:g} m of an end {t['max_abs_near']:.3f} K, elsewhere "
                    f"{t['max_abs_elsewhere']:.3f} K (RMS {t['rms_elsewhere']:.3f} K)")
    for t, (a, b) in zip(successive, zip(spatial[:-1], spatial[1:])):
        log(f"   T({a['grid'].nx} x {a['grid'].ny}) - T({b['grid'].nx} x {b['grid'].ny}): max |d| within "
            f"{NEAR_END_M:g} m of an end {t['max_abs_near']:.3f} K, elsewhere {t['max_abs_elsewhere']:.3f} K")
    rp.write_csv(out / "segment_end_temperatures.csv",
                 ("grid", "compared_with", "nodes_compared", "nodes_within_0.1m_of_a_segment_end",
                  "max_abs_difference_K", "x_at_max_m", "y_at_max_m", "difference_at_max_K",
                  "max_abs_difference_within_0.1m_of_an_end_K", "max_abs_difference_elsewhere_K",
                  "rms_difference_elsewhere_K", "mean_difference_K"), t_rows)

    edges_m = segment_bins() * P["Lx"]
    heat = [segment_heat(r, edges_m) for r in spatial]
    h_rows = []
    for seg in ("bottom", "top"):
        ref = heat[-1][seg]["Q"]
        for r, hs in zip(spatial, heat):
            for k in range(len(edges_m) - 1):
                h_rows.append((seg, k, f"{edges_m[k]:.6f}", f"{edges_m[k + 1]:.6f}", r["grid"].nx, r["grid"].ny,
                               f"{hs[seg]['Q'][k] / 1e6:.6f}",
                               f"{hs[seg]['Q'][k] / (edges_m[k + 1] - edges_m[k]) / 1e6:.6f}",
                               f"{(hs[seg]['Q'][k] - ref[k]) / 1e6:.6f}", f"{hs[seg]['E'][k] / 1e6:.6f}"))
    rp.write_csv(out / "segment_heat_input.csv",
                 ("segment", "interval", "x_from_m", "x_to_m", "nx", "ny",
                  "boundary_flux_heat_input_MJ_per_m", "per_m_of_edge_MJ_per_m2",
                  "difference_from_193x177_MJ_per_m", "segment_half_cell_energy_MJ_per_m"), h_rows)
    end_bins = [0, len(edges_m) - 2]
    seg_summary = {}
    for seg in ("bottom", "top"):
        ref = heat[-1][seg]["Q"]
        rows_seg = []
        for r, hs in zip(spatial[:-1], heat[:-1]):
            d = (hs[seg]["Q"] - ref) / 1e6
            rows_seg.append(dict(grid=f"{r['grid'].nx} x {r['grid'].ny}",
                                 end_intervals_MJ_per_m=float(d[end_bins].sum()),
                                 other_intervals_MJ_per_m=float(np.delete(d, end_bins).sum()),
                                 end_interval_each_MJ_per_m=float(d[0]),
                                 largest_other_interval_MJ_per_m=float(np.delete(d, end_bins).min())))
            log(f"   {seg} segment, {r['grid'].nx} x {r['grid'].ny} minus 193 x 177: end intervals "
                f"{d[0]:+.3f} MJ/m each (length {edges_m[1] - edges_m[0]:.4f} m), other intervals "
                f"{np.delete(d, end_bins).min():+.3f} to {np.delete(d, end_bins).max():+.3f} MJ/m "
                f"(length {edges_m[2] - edges_m[1]:.4f} m)")
        seg_summary[seg] = rows_seg
    end_share = {f"{r['grid'].nx} x {r['grid'].ny}":
                 float(sum(hs[s]["Q_nodes"][np.isclose(hs[s]["x_nodes"], x)].sum()
                           for s in ("bottom", "top") for x in (0.25 * P["Lx"], 0.75 * P["Lx"])) / r["Q"])
                 for r, hs in zip(spatial, heat)}
    log(f"   share of the boundary-flux heat input supplied by the four end nodes: "
        f"{', '.join(f'{k} {v * 100:.1f} %' for k, v in end_share.items())}")

    # checks ------------------------------------------------------------------------
    log("\nD. STUDY CHECKS")
    all_runs = list(run.cache.values())
    worst = max(abs(r["accounting_closure"]) for r in all_runs)
    worst_energy = max(abs(r["energy_closure"]) for r in all_runs)
    check("S1 heat accounting (every run of this script)",
          f"worst |dE - (Q + E)| / dE = {worst:.1e}; worst V3-type closure {worst_energy:.1e}; "
          f"{len(all_runs)} runs",
          f"<= {ACCOUNTING_TOL:g} for both", worst <= ACCOUNTING_TOL and worst_energy <= ACCOUNTING_TOL,
          "the stored-energy increase measured by the mean temperature equals the boundary-flux heat input "
          "plus the heated-segment half-cell energy, so the two heat-input figures differ only by the half-cells")
    mismatch = max(t["mismatch"] for t in temperature + successive)
    bin_gap = max(abs(hs[s]["Q"].sum() - hs[s]["Q_total"]) / hs[s]["Q_total"]
                  for hs in heat for s in ("bottom", "top"))
    ends_on_nodes = all(np.isclose(hs[s]["x_nodes"].min(), 0.25 * P["Lx"], rtol=0, atol=MATCH_TOL)
                        and np.isclose(hs[s]["x_nodes"].max(), 0.75 * P["Lx"], rtol=0, atol=MATCH_TOL)
                        for hs in heat for s in ("bottom", "top"))
    check("S2 matching locations on the nested grids",
          f"largest node-coordinate mismatch {mismatch:.1e} m; interval sums vs segment totals "
          f"{bin_gap:.1e} relative; segment ends on nodes at x = L/4 and 3L/4: {ends_on_nodes}",
          f"<= {MATCH_TOL:g} m, <= {MATCH_TOL:g}, True",
          mismatch <= MATCH_TOL and bin_gap <= MATCH_TOL and ends_on_nodes,
          "grid comparisons are made at the same physical points and boundary intervals, and the interval "
          "values account for all of each segment's heat input")
    rp.write_csv(out / "study_checks.csv", ("check", "value", "tolerance", "passed", "establishes"),
                 [(c["check"], c["value"], c["tolerance"], c["passed"], c["establishes"])
                  for c in sorted(checks, key=lambda c: c["check"])])

    figure_space_time(eq3, spatial, spatial_coarse_dt, temporal, out / "fig5_space_time_sensitivity.png")
    figure_segment_ends(spatial, temperature, heat, edges_m, out / "fig6_segment_ends.png")

    wall = time.perf_counter() - clock
    env["finished_utc"] = dt_.datetime.now(dt_.timezone.utc).isoformat(timespec="seconds")
    env["wall_time_s"] = round(wall, 1)
    summary = dict(
        purpose=("October 2026 revision: heat-input accounting, spatial and temporal sensitivity "
                 "measured separately, and grid differences near the heated-segment ends"),
        units="temperatures in K; energies in MJ per metre of out-of-plane depth (MJ/m)",
        settings=dict(grids=[list(g) for g in GRIDS], fixed_fine_steps=N_FINE,
                      fixed_fine_dt_s=P["t_end"] / N_FINE, coarse_steps=N_COARSE,
                      coarse_dt_s=P["t_end"] / N_COARSE, temporal_steps=list(TEMPORAL_STEPS),
                      temporal_grids=[list(g) for g in TEMPORAL_GRIDS], near_end_m=NEAR_END_M,
                      segment_interval_edges_m=edges_m.tolist()),
        heat_accounting_13x12_vs_193x177=heat_comparison,
        half_cell_energy_ratio_per_halving=half_cell_ratios,
        spatial_study_T_avg_changes_K=space_change[1:], spatial_study_change_ratios=space_ratio[2:],
        temporal_study_193x177_T_avg_changes_K=time_change[1:],
        temporal_study_193x177_change_ratios=[q for q in time_ratio if q == q],
        time_step_effect_by_grid_K=time_effect_by_grid,
        coarse_grid_difference=split,
        segment_end_temperature_differences_vs_193x177=[
            {k: v for k, v in t.items() if k != "diff"} for t in temperature],
        segment_heat_input_differences_vs_193x177=seg_summary,
        end_node_share_of_boundary_flux_heat_input=end_share,
        checks=checks, environment=env)
    with open(out / "studies_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    n_fail = sum(not c["passed"] for c in checks)
    log(f"\nRESULTS written to {out.name}/: heat_accounting.csv, space_time_study.csv, "
        f"coarse_grid_difference.csv, segment_end_temperatures.csv, segment_heat_input.csv, "
        f"study_checks.csv, studies_summary.json, fig5, fig6")
    log(f"   {len(run.cache)} plate runs; study checks: {len(checks) - n_fail} of {len(checks)} passed; "
        f"wall time {wall:.1f} s")
    log.file.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
