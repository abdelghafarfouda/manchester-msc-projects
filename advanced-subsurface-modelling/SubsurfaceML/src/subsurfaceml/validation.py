"""Verification and validation suite for the numerical core.

Run *before* any training data is generated; results go to
``results/<config>/metrics/validation.json``.  Every reference solution is
derived from the course material:

V1  Steady-state radial single-phase pressure vs
    ``p(r) = p_e + q mu /(2 pi k h) ln(r_e/r)`` -- series radial Darcy flow
    (``2-Upscaling.pdf`` p.8, cylindrical form ``4-CO2 BL.pdf`` p.20).
V2  Bottom-hole pressure from the well equation (``3-IMPES.pdf`` p.16) vs the
    analytical ``p(r_w)`` -- validates the coarse well block.
V3  Closed-tank material balance ``p(t) = p_i + q t /(c_t V_p)`` from the
    compressibility definitions of ``1-Transmissibility.pdf`` p.6, 15-16.
V5  Buckley-Leverett: 1-D IMPES saturation profile vs the Welge /
    method-of-characteristics solution (``4-CO2 BL.pdf`` p.11-16).
V6  CO2 mass balance, saturation bounds, clipping count and near-well
    monotonicity for the radial two-phase model.
V7  Grid (n_r), near-well block (r_near) and time-step (max_dS) convergence.
V8  Strict-CFL (default) vs relaxed time stepping (``3-IMPES.pdf`` p.20).
V9  Upscaling: analytical bounds of ``2-Upscaling.pdf`` p.14, exact series and
    parallel limits (p.6-9), and the lecture's 2x2 flow-based layout (p.15).
V10 Single-phase upscaling applied to a *two-phase* problem
    (``2-Upscaling.pdf`` p.3-4 poses the question).
V11 Multi-layer well: prescribed total rate conserved, one common
    bottom-hole pressure, split between layers in the single-phase limits.
V12 The *two-phase* simulator's bottom-hole pressure, run in a
    single-phase-equivalent configuration in a sealed compartment, against
    the pseudo-steady-state solution of a bounded circular reservoir.
V13 The same for two commingled layers sharing one bottom-hole pressure
    (late-time limit: pore-volume rate split); also checks the analytical
    reduced-order model of :mod:`rom` in the same limit.

(A Theis line-source benchmark was removed: that solution is not in the
supplied material.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .fluids import (FluidProperties, RelPerm, RockProperties, welge_shock,
                     bl_profile_1d)
from .grid import CartesianGrid1D, RadialGrid
from .impes import InjectionSchedule, TwoPhaseModel
from .outputs import plume_radius_all_layers, extract_qois
from .petrophysics import make_layered_rock, dykstra_parsons
from .single_phase import (solve_single_phase, steady_state_radial,
                           tank_material_balance)
from .units import DAY, MPA, YEAR, md_to_m2, m2_to_md
from . import upscaling as ups


# --------------------------------------------------------------------------
def v1_v2_steady_radial() -> dict:
    k, mu, phi, c_t, h, q = md_to_m2(100.0), 6e-4, 0.20, 1e-9, 30.0, 0.02
    p_i, r_w, r_e = 15.0 * MPA, 0.15, 3000.0
    out = {}
    for r_near in (0.15, 5.0, 20.0):
        g = RadialGrid(n=60, r_w=r_w, r_e=r_e, h=h, r_near=r_near)
        res = solve_single_phase(g, k, phi, mu, c_t, p_i, q_well=q,
                                 t_end=200 * YEAR, n_steps=300)
        p_ana = steady_state_radial(g, k, mu, q, p_i)
        cell_err = float(np.max(np.abs(res.p[-1] - p_ana))
                         / np.max(np.abs(p_ana - p_i)))
        p_bh = res.p[-1, 0] + q / (g.well_index_geom() * k / mu)
        p_bh_ana = p_i + q * mu / (2 * np.pi * k * h) * np.log(r_e / r_w)
        out[f"r_near={r_near:g}"] = {
            "cell_pressure_rel_err": cell_err,
            "bhp_rel_err": float(abs(p_bh - p_bh_ana) / abs(p_bh_ana - p_i)),
            "volume_balance_err": res.mass_balance_error,
        }
    return out


def v3_tank() -> dict:
    k, mu, phi, c_t, h, q = md_to_m2(100.0), 6e-4, 0.20, 1e-9, 30.0, 0.02
    p_i = 15.0 * MPA
    g = RadialGrid(n=60, r_w=0.15, r_e=3000.0, h=h, r_near=5.0)
    res = solve_single_phase(g, k, phi, mu, c_t, p_i, q_well=q,
                             t_end=YEAR, n_steps=200, outer_bc="closed")
    Vp = float(np.sum(g.bulk_volume * phi))
    p_avg = float(np.sum(res.p[-1] * g.bulk_volume * phi) / Vp)
    p_ana = tank_material_balance(YEAR, q, Vp, c_t, p_i)
    return {"p_avg_MPa": p_avg / MPA, "p_analytic_MPa": p_ana / MPA,
            "rel_err": float(abs(p_avg - p_ana) / abs(p_ana - p_i)),
            "volume_balance_err": res.mass_balance_error}


def v5_buckley_leverett(n_cells=(100, 200, 400)) -> dict:
    fl = FluidProperties(c_a=0.0, c_g=0.0)
    rp = RelPerm()
    rk = RockProperties(c_r=0.0)
    L, A, phi, q_vol = 100.0, 10.0, 0.20, 1e-5
    T = 0.4 * L * A * phi / q_vol
    S_gf, f_gf, v_shock = welge_shock(rp, fl)
    out = {"welge_shock_Sg": S_gf, "welge_shock_fg": f_gf,
           "shock_speed_dfdS": v_shock,
           "endpoint_mobility_ratio": rp.endpoint_mobility_ratio(fl),
           "grids": {}}
    for n in n_cells:
        g = CartesianGrid1D(n=n, L=L, area=A)
        k = np.full(n, md_to_m2(100.0))
        m = TwoPhaseModel([g], k[None, :], np.full((1, n), phi), fl, rp, rk,
                          15.0 * MPA, cfl=0.3, max_dS=0.05, dt_init=100.0,
                          dt_max=T / 50)
        res = m.run(InjectionSchedule([0.0, T], [q_vol * fl.rho_g]),
                    np.array([T]))
        Sg = res.Sg[-1, 0]
        ana = bl_profile_1d(rp, fl, g.centres, T, q_vol, A, phi, L=L)
        x_shock = q_vol / (A * phi) * v_shock * T
        idx = np.where(Sg > 0.5 * S_gf)[0]
        x_sim = float(g.centres[idx[-1]]) if idx.size else 0.0
        out["grids"][str(n)] = {
            "L1_saturation_error": float(np.mean(np.abs(Sg - ana))),
            "front_position_sim_m": x_sim,
            "front_position_analytic_m": float(x_shock),
            "front_rel_err": float(abs(x_sim - x_shock) / x_shock),
            "mass_balance_err": res.mass_balance_error,
            "sg_min": float(Sg.min()), "sg_max": float(Sg.max()),
            "n_clipped": int(res.diagnostics["n_clipped"]),
            "wall_time_s": res.diagnostics["wall_time_s"],
        }
    out["profiles"] = None
    return out


def _radial_case(n_r=60, r_near=5.0, max_dS=0.05, enforce_cfl=True,
                 n_layers=4, V_DP=0.55, seed=1, q=30.0,
                 t_inj=5 * YEAR, t_tot=15 * YEAR, r_e=5000.0):
    # layers are homogeneous in r, so refining n_r or moving r_near changes
    # only the discretisation, not the rock
    rock = make_layered_rock(n_layers, n_r, 40.0, md_to_m2(120.0), V_DP, 0.20,
                             seed=seed)
    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=r_e, h=rock.h[l],
                        r_near=r_near) for l in range(n_layers)]
    fl, rp, rk = FluidProperties(), RelPerm(), RockProperties()
    m = TwoPhaseModel(grids, rock.k, rock.phi, fl, rp, rk, 15.0 * MPA,
                      max_dS=max_dS, dt_init=3600.0, dt_max=180 * DAY,
                      enforce_cfl=enforce_cfl,
                      max_overshoot=1e-3 if enforce_cfl else 0.02)
    sch = InjectionSchedule([0.0, t_inj, t_tot], [q, 0.0])
    res = m.run(sch, np.linspace(0.0, t_tot, 41))
    q_ = extract_qois(res, grids, m)
    q_.pop("_r_plume_history_m", None)
    # odd-even oscillation detector.  A smooth profile may rise or fall (the
    # CO2 is compressed where the pressure is higher, so S_g can increase
    # away from the well) but has only a few turning points; a spurious
    # odd-even oscillation creates a turning point at almost every cell.
    # Count sign changes of dS/dr (ignoring changes below 1e-5) per profile.
    inj = (res.t > 0) & (res.t <= t_inj)
    dS = np.diff(res.Sg[inj], axis=2)
    sgn = np.where(np.abs(dS) < 1e-5, 0, np.sign(dS))
    turns = []
    for prof in sgn.reshape(-1, sgn.shape[-1]):
        nz = prof[prof != 0]
        turns.append(int(np.sum(nz[1:] != nz[:-1])))
    q_["near_well_oscillation"] = int(max(turns))    # max turning points
    q_["max_bhp_spread_Pa"] = float(res.diagnostics["max_bhp_spread_Pa"])
    q_["max_rate_rel_err"] = float(res.diagnostics["max_rate_rel_err"])
    return res, q_, grids, m


def v6_v7_v8_radial(full: bool = False) -> dict:
    out = {"base": {}, "n_r": {}, "r_near": {}, "max_dS": {}, "cfl": {}}
    _, q0, _, _ = _radial_case()
    out["base"] = {k: v for k, v in q0.items() if not k.startswith("_")}

    # homogeneous-in-r layers so that refinement changes only the grid
    for n_r in ((30, 60, 120, 240) if full else (30, 60, 120)):
        _, q, _, _ = _radial_case(n_r=n_r)
        out["n_r"][str(n_r)] = {"dp_bh_max_MPa": q["dp_bh_max_Pa"] / MPA,
                                "r_plume_end_m": q["r_plume_end_m"],
                                "r_plume_m95_m": q["r_plume_m95_m"],
                                "sweep_efficiency": q["sweep_efficiency"],
                                "mass_retained_Mt": q["mass_retained_kg"] / 1e9,
                                "wall_time_s": q["wall_time_s"],
                                "n_steps": q["n_steps"]}
    for r_near in ((20.0, 10.0, 5.0, 2.5) if full else (20.0, 10.0, 5.0)):
        _, q, _, _ = _radial_case(r_near=r_near)
        out["r_near"][str(r_near)] = {"dp_bh_max_MPa": q["dp_bh_max_Pa"] / MPA,
                                      "r_plume_end_m": q["r_plume_end_m"],
                                "r_plume_m95_m": q["r_plume_m95_m"],
                                "sweep_efficiency": q["sweep_efficiency"],
                                      "mass_retained_Mt": q["mass_retained_kg"] / 1e9,
                                      "wall_time_s": q["wall_time_s"],
                                      "n_steps": q["n_steps"]}
    for mdS in ((0.10, 0.05, 0.02, 0.01) if full else (0.10, 0.05, 0.02)):
        _, q, _, _ = _radial_case(max_dS=mdS)
        out["max_dS"][str(mdS)] = {"dp_bh_max_MPa": q["dp_bh_max_Pa"] / MPA,
                                   "r_plume_end_m": q["r_plume_end_m"],
                                "r_plume_m95_m": q["r_plume_m95_m"],
                                "sweep_efficiency": q["sweep_efficiency"],
                                   "mass_retained_Mt": q["mass_retained_kg"] / 1e9,
                                   "wall_time_s": q["wall_time_s"],
                                   "n_steps": q["n_steps"]}
    _, qs, _, _ = _radial_case(enforce_cfl=False)      # the relaxed variant
    rel = lambda a, b: float(abs(a - b) / max(abs(b), 1e-30))
    out["cfl"] = {
        "relaxed": {"dp_bh_max_MPa": qs["dp_bh_max_Pa"] / MPA,
                   "r_plume_end_m": qs["r_plume_end_m"],
                   "mass_retained_Mt": qs["mass_retained_kg"] / 1e9,
                   "n_steps": qs["n_steps"], "wall_time_s": qs["wall_time_s"],
                   "near_well_oscillation": qs["near_well_oscillation"]},
        "strict_default": {"dp_bh_max_MPa": q0["dp_bh_max_Pa"] / MPA,
                    "r_plume_end_m": q0["r_plume_end_m"],
                    "mass_retained_Mt": q0["mass_retained_kg"] / 1e9,
                    "n_steps": q0["n_steps"], "wall_time_s": q0["wall_time_s"],
                    "near_well_oscillation": q0["near_well_oscillation"]},
        "rel_diff": {
            "dp_bh_max": rel(q0["dp_bh_max_Pa"], qs["dp_bh_max_Pa"]),
            "r_plume_end": rel(q0["r_plume_end_m"], qs["r_plume_end_m"]),
            "mass_retained": rel(q0["mass_retained_kg"], qs["mass_retained_kg"]),
        },
        "step_ratio_strict_over_relaxed": float(q0["n_steps"] / max(qs["n_steps"], 1)),
    }
    return out


def v9_upscaling() -> dict:
    case = ups.lecture_2x2_case()
    parallel = np.array([[1.0, 1.0], [5.0, 5.0], [20.0, 20.0]])
    series = np.array([[1.0, 5.0, 20.0], [1.0, 5.0, 20.0]])
    exact = {
        "parallel_flow_k_eff": ups.upscale_flow_based(parallel.T, 1, 1)["k_eff"],
        "parallel_flow_arithmetic": ups.k_arithmetic(parallel),
        "series_flow_k_eff": ups.upscale_flow_based(series.T, 1, 1)["k_eff"],
        "series_flow_harmonic": ups.k_harmonic(series),
    }
    rng = np.random.default_rng(7)
    rows = []
    for sigma in (0.5, 1.0, 1.75):
        kf = np.exp(rng.normal(0.0, sigma, (32, 32)))
        t0 = time.perf_counter()
        ke = ups.upscale_flow_based(kf, 1.0, 1.0)["k_eff"]
        t_num = time.perf_counter() - t0
        t1 = time.perf_counter()
        ka, kh = ups.k_arithmetic(kf), ups.k_harmonic(kf)
        kha, kah = ups.k_ha(kf), ups.k_ah(kf)
        t_ana = time.perf_counter() - t1
        rows.append({
            "sigma_lnk": sigma, "V_DP": dykstra_parsons(kf.ravel()),
            "K_H": kh, "K_HA": kha, "K_star": ke, "K_AH": kah, "K_A": ka,
            "err_arithmetic_pct": 100 * (ka - ke) / ke,
            "err_harmonic_pct": 100 * (kh - ke) / ke,
            "err_HA_pct": 100 * (kha - ke) / ke, "err_AH_pct": 100 * (kah - ke) / ke,
            "bounds_respected": bool(kh <= kha <= ke <= kah <= ka),
            "t_numerical_s": t_num, "t_analytical_s": t_ana,
        })
    return {"lecture_2x2_case": case, "exact_limits": exact,
            "random_fields": rows}


def v11_well_coupling() -> dict:
    """Two layers (500 mD / phi 0.25 vs 20 mD / phi 0.08) in a sealed
    compartment -- the case that exposed the old rate allocation."""
    n = 30
    grids = [RadialGrid(n=n, r_w=0.15, r_e=1500.0, h=10.0, r_near=5.0)
             for _ in range(2)]
    k = np.stack([np.full(n, md_to_m2(500.0)), np.full(n, md_to_m2(20.0))])
    phi = np.stack([np.full(n, 0.25), np.full(n, 0.08)])
    m = TwoPhaseModel(grids, k, phi, FluidProperties(), RelPerm(),
                      RockProperties(), 15 * MPA, outer_bc="closed",
                      max_dS=0.05, dt_max=30 * DAY)
    res = m.run(InjectionSchedule([0.0, 2 * YEAR, 3 * YEAR], [20.0, 0.0]),
                np.linspace(0, 3 * YEAR, 13))
    d = res.diagnostics
    inj = res.t <= 2 * YEAR
    q = res.q_layer[inj][1:]
    return {"max_bhp_spread_Pa": d["max_bhp_spread_Pa"],
            "max_rate_rel_err": d["max_rate_rel_err"],
            "p_bh_max_MPa": d["p_bh_max_Pa"] / MPA,
            "well_block_pressure_difference_MPa_end_injection":
                float(abs(res.p[inj][-1, 0, 0] - res.p[inj][-1, 1, 0]) / MPA),
            "layer_rate_fraction_high_k_end_injection": float(q[-1, 0] / q[-1].sum()),
            "shut_in_layer_rates_all_zero": bool(np.all(res.q_layer[~inj] == 0.0)),
            "mass_balance_error": res.mass_balance_error}


def v10_upscaling_two_phase() -> dict:
    """Does single-phase upscaling preserve the two-phase answer?

    Fine model: 4 layers with contrasted permeability.
    Coarse model: 1 layer whose permeability is the flow-based (parallel)
    upscale of the 4 layers, with volume-weighted porosity.
    """
    n_layers, n_r, h_tot = 4, 60, 40.0
    rock = make_layered_rock(n_layers, n_r, h_tot, md_to_m2(120.0), 0.65, 0.20,
                             seed=3)
    fl, rp, rk = FluidProperties(), RelPerm(), RockProperties()
    sch = InjectionSchedule([0.0, 5 * YEAR, 15 * YEAR], [30.0, 0.0])
    rep = np.linspace(0.0, 15 * YEAR, 41)

    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=5000.0, h=rock.h[l], r_near=5.0)
             for l in range(n_layers)]
    mf = TwoPhaseModel(grids, rock.k, rock.phi, fl, rp, rk, 15.0 * MPA,
                       max_dS=0.05, dt_init=3600.0, dt_max=180 * DAY)
    t0 = time.perf_counter()
    rf = mf.run(sch, rep)
    t_fine = time.perf_counter() - t0
    qf = extract_qois(rf, grids, mf); qf.pop("_r_plume_history_m")

    # layers are in parallel for radial flow -> thickness-weighted arithmetic
    k_up = np.average(rock.k, axis=0, weights=rock.h)
    phi_up = np.average(rock.phi, axis=0, weights=rock.h)
    gc = [RadialGrid(n=n_r, r_w=0.15, r_e=5000.0, h=h_tot, r_near=5.0)]
    mc = TwoPhaseModel(gc, k_up[None, :], phi_up[None, :], fl, rp, rk,
                       15.0 * MPA, max_dS=0.05, dt_init=3600.0, dt_max=180 * DAY)
    t0 = time.perf_counter()
    rc = mc.run(sch, rep)
    t_coarse = time.perf_counter() - t0
    qc = extract_qois(rc, gc, mc); qc.pop("_r_plume_history_m")

    rel = lambda a, b: float((a - b) / max(abs(b), 1e-30))
    return {
        "V_DP_layers": dykstra_parsons(rock.k_layer),
        "fine": {"dp_bh_max_MPa": qf["dp_bh_max_Pa"] / MPA,
                 "r_plume_end_m": qf["r_plume_end_m"],
                 "mass_retained_Mt": qf["mass_retained_kg"] / 1e9,
                 "sweep_efficiency": qf["sweep_efficiency"],
                 "wall_time_s": t_fine, "n_cells": n_layers * n_r},
        "upscaled": {"dp_bh_max_MPa": qc["dp_bh_max_Pa"] / MPA,
                     "r_plume_end_m": qc["r_plume_end_m"],
                     "mass_retained_Mt": qc["mass_retained_kg"] / 1e9,
                     "sweep_efficiency": qc["sweep_efficiency"],
                     "wall_time_s": t_coarse, "n_cells": n_r},
        "relative_error": {
            "dp_bh_max": rel(qc["dp_bh_max_Pa"], qf["dp_bh_max_Pa"]),
            "r_plume_end": rel(qc["r_plume_end_m"], qf["r_plume_end_m"]),
            "mass_retained": rel(qc["mass_retained_kg"], qf["mass_retained_kg"]),
            "sweep_efficiency": rel(qc["sweep_efficiency"], qf["sweep_efficiency"]),
        },
        "speedup": float(t_fine / max(t_coarse, 1e-9)),
        "interpretation": (
            "Single-phase (thickness-weighted arithmetic / parallel) upscaling "
            "of a no-crossflow layered unit preserves total flow capacity kh, "
            "so pressure is reproduced well, but it destroys the layer-by-layer "
            "velocity contrast that controls the leading edge of the plume. The "
            "plume-radius error is therefore much larger than the pressure "
            "error: single-phase upscaling does NOT preserve the two-phase "
            "quantity of interest."),
    }


def _pss_case(ks_mD, phis, h, r_e, q_mass, T, *, n_r=60, r_near=5.0):
    """Two-phase simulator run in a *single-phase-equivalent* configuration
    (equal phase viscosities and compressibilities, linear relative
    permeabilities with zero residuals, so ``lambda_t = 1/mu`` and
    ``c_t = c_r + c_a`` everywhere) in a sealed compartment, against the
    late-time pseudo-steady-state (PSS) solution of a bounded circular
    reservoir with a common bottom-hole pressure:

    ``p_w - p_i = Q t / (c_t V_T) + sum_l V_l (q_l / J_l) / V_T``,
    ``q_l = Q V_l / V_T``, ``J_l = 2 pi k_l h / (mu (ln(r_e/r_w) - 3/4))``.

    For one layer this is the classical PSS drawdown (here build-up)
    equation; for several layers it is its commingled late-time limit, in
    which every layer pressurises at the same rate, so the rate split follows
    pore volume.  This checks the quantity the surrogates learn -- the
    simulator's bottom-hole pressure -- including the well index, the units,
    the storage term and the sealed boundary, which V1-V3 check only for the
    separate single-phase solver.
    """
    from . import rom
    mu, c_a, c_r = rom.MU_A, rom.C_A, rom.C_R
    fl = FluidProperties(rho_g=1000.0, rho_a=1000.0, mu_g=mu, mu_a=mu,
                         c_g=c_a, c_a=c_a)
    rp = RelPerm(S_ar=0.0, S_gr=0.0, n_a=1.0, n_g=1.0, kra0=1.0, krg0=1.0)
    L = len(ks_mD)
    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=r_e, h=h, r_near=r_near)
             for _ in range(L)]
    k = np.stack([np.full(n_r, md_to_m2(x)) for x in ks_mD])
    phi = np.stack([np.full(n_r, x) for x in phis])
    m = TwoPhaseModel(grids, k, phi, fl, rp, RockProperties(c_r=c_r),
                      15.0 * MPA, outer_bc="closed", max_dS=1.0,
                      dt_max=10 * DAY)
    res = m.run(InjectionSchedule([0.0, T], [q_mass]), np.linspace(0.0, T, 5))
    Q, c_t = q_mass / fl.rho_g, c_r + c_a
    V = np.pi * (r_e ** 2 - 0.15 ** 2) * h * np.asarray(phis, float)
    J = np.array([2 * np.pi * md_to_m2(x) * h / mu / (np.log(r_e / 0.15) - 0.75)
                  for x in ks_mD])
    VT = V.sum()
    ana = Q * T / (c_t * VT) + float(np.sum(V * (Q * V / VT) / J)) / VT
    sim = float(res.p_bh[-1] - 15.0 * MPA)
    # the analytical ROM used as a surrogate feature, same configuration
    rom_dp = float(rom.bhp_buildup_layers(
        md_to_m2(np.asarray(ks_mD, float)), np.full(L, h), phis, r_e, 0.15,
        fl.rho_g, np.array([[q_mass]]), T, steps_per_period=400,
        co2_storage=False)[0])
    return {"layers_k_mD": list(map(float, ks_mD)), "phi": list(map(float, phis)),
            "t_years": T / YEAR, "q_kg_s": q_mass,
            "sim_dp_bh_MPa": sim / MPA, "pss_analytic_dp_MPa": ana / MPA,
            "rel_err_sim_vs_pss": abs(sim - ana) / ana,
            "rom_dp_MPa": rom_dp / MPA,
            "rel_err_rom_vs_pss": abs(rom_dp - ana) / ana,
            "late_layer_rate_fraction_sim": (res.q_layer[-1] / res.q_layer[-1].sum()).tolist(),
            "late_layer_rate_fraction_pss": (V / VT).tolist(),
            "mass_balance_error": res.mass_balance_error,
            "n_steps": int(res.diagnostics["n_steps"]),
            "wall_time_s": float(res.diagnostics["wall_time_s"])}


def v12_v13_pss_well_pressure() -> dict:
    """V12: one layer; V13: two commingled layers (the V11 contrast) sharing
    one bottom-hole pressure.  Both in a sealed compartment, compared at late
    time with the PSS solution (see :func:`_pss_case`)."""
    return {"V12_single_layer": _pss_case([100.0], [0.20], 20.0, 2000.0, 10.0,
                                          3 * YEAR),
            "V13_two_layers": _pss_case([500.0, 20.0], [0.25, 0.08], 10.0,
                                        1500.0, 20.0, 3 * YEAR)}


# --------------------------------------------------------------------------
def run_all(cfg: Config | None = None, *, save: bool = True,
            verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    res = {
        "V1_V2_steady_radial_and_well_index": v1_v2_steady_radial(),
        "V3_closed_tank_material_balance": v3_tank(),
        "V5_buckley_leverett": v5_buckley_leverett(),
        "V6_V7_V8_radial_two_phase": v6_v7_v8_radial(
            full=bool(getattr(getattr(cfg, "solver", None), "validation_full", False))),
        "V9_upscaling": v9_upscaling(),
        "V10_upscaling_two_phase": v10_upscaling_two_phase(),
        "V11_well_coupling": v11_well_coupling(),
        "V12_V13_pss_well_pressure": v12_v13_pss_well_pressure(),
    }
    res["wall_time_s"] = time.perf_counter() - t0
    res["verdict"] = _verdict(res)
    if save and cfg is not None:
        cfg.paths.mkdirs()
        p = Path(cfg.paths.metrics) / "validation.json"
        p.write_text(json.dumps(res, indent=2, default=str))
        if verbose:
            print(f"[validation] written to {p}")
    return res


def _verdict(res: dict) -> dict:
    """Pass/fail summary against pre-declared tolerances."""
    v = {}
    v["steady_radial_pressure"] = all(
        d["cell_pressure_rel_err"] < 1e-8
        for d in res["V1_V2_steady_radial_and_well_index"].values())
    v["well_index_bhp"] = all(
        d["bhp_rel_err"] < 1e-8
        for d in res["V1_V2_steady_radial_and_well_index"].values())
    v["closed_tank"] = res["V3_closed_tank_material_balance"]["rel_err"] < 1e-6
    bl = res["V5_buckley_leverett"]["grids"]
    v["buckley_leverett_front"] = bl[max(bl, key=int)]["front_rel_err"] < 0.05
    v["buckley_leverett_L1"] = bl[max(bl, key=int)]["L1_saturation_error"] < 0.02
    v["bl_no_clipping"] = all(d["n_clipped"] == 0 for d in bl.values())
    r = res["V6_V7_V8_radial_two_phase"]
    v["radial_mass_balance"] = r["base"]["mass_balance_error"] < 1e-6
    v["radial_no_clipping"] = r["base"]["n_clipped"] == 0
    v["radial_saturation_bounds"] = (r["base"]["sg_min"] >= -1e-12
                                     and r["base"]["sg_max"] <= 1.0)
    v["no_odd_even_oscillation"] = r["base"]["near_well_oscillation"] <= 3
    v["time_step_control_insensitive"] = max(r["cfl"]["rel_diff"].values()) < 0.02
    u = res["V9_upscaling"]
    c = u["lecture_2x2_case"]
    v["upscaling_2x2_bounds"] = c["K_H"] <= c["K_HA"] <= c["k_eff_D"] <= c["K_AH"] <= c["K_A"]
    e = u["exact_limits"]
    v["upscaling_parallel_exact"] = abs(e["parallel_flow_k_eff"] - e["parallel_flow_arithmetic"]) < 1e-9
    v["upscaling_series_exact"] = abs(e["series_flow_k_eff"] - e["series_flow_harmonic"]) < 1e-9
    v["upscaling_bounds_random"] = all(rw["bounds_respected"] for rw in u["random_fields"])
    w = res["V11_well_coupling"]
    v["well_common_bhp"] = w["max_bhp_spread_Pa"] < 1e-3
    v["well_total_rate_conserved"] = w["max_rate_rel_err"] < 1e-10
    v["well_shut_in_no_crossflow"] = w["shut_in_layer_rates_all_zero"]
    if "V12_V13_pss_well_pressure" in res:
        pss = res["V12_V13_pss_well_pressure"]
        v["pss_bhp_single_layer"] = pss["V12_single_layer"]["rel_err_sim_vs_pss"] < 1e-4
        v["pss_bhp_two_layers"] = pss["V13_two_layers"]["rel_err_sim_vs_pss"] < 1e-3
        v["rom_matches_pss_limit"] = max(
            pss["V12_single_layer"]["rel_err_rom_vs_pss"],
            pss["V13_two_layers"]["rel_err_rom_vs_pss"]) < 1e-3
    v["ALL_PASS"] = all(v.values())
    return v
