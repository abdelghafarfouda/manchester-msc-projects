"""Meaningful simulator tests: each one compares against an analytical result
or an exactly conserved quantity, not against a previously recorded number."""
from __future__ import annotations

import numpy as np
import pytest

from subsurfaceml.fluids import (FluidProperties, RelPerm, RockProperties,
                                 bl_profile_1d, welge_shock)
from subsurfaceml.grid import CartesianGrid1D, RadialGrid
from subsurfaceml.impes import InjectionSchedule, TwoPhaseModel
from subsurfaceml.outputs import plume_radius, plume_radius_mass_fraction
from subsurfaceml.single_phase import (solve_single_phase, steady_state_radial,
                                       tank_material_balance)
from subsurfaceml.units import DAY, MPA, YEAR, md_to_m2

K, MU, PHI, CT, H, Q = md_to_m2(100.0), 6e-4, 0.20, 1e-9, 30.0, 0.02
PI = 15.0 * MPA


# ---------------------------------------------------------------- geometry
def test_radial_grid_volume_is_exact():
    g = RadialGrid(n=40, r_w=0.15, r_e=3000.0, h=H, r_near=5.0)
    total = np.pi * (g.r_e ** 2 - g.r_w ** 2) * H
    assert g.bulk_volume.sum() == pytest.approx(total, rel=1e-12)
    assert np.all(np.diff(g.faces) > 0)
    assert np.all(g.centres > g.faces[:-1]) and np.all(g.centres < g.faces[1:])


def test_harmonic_k_reduces_to_the_value_when_homogeneous():
    g = RadialGrid(n=20, r_w=0.15, r_e=1000.0, h=H)
    k = np.full(20, md_to_m2(250.0))
    assert np.allclose(g.harmonic_k(k), k[0], rtol=1e-12)


# ------------------------------------------------------------ single phase
def test_steady_state_radial_matches_analytical():
    g = RadialGrid(n=60, r_w=0.15, r_e=3000.0, h=H, r_near=5.0)
    r = solve_single_phase(g, K, PHI, MU, CT, PI, q_well=Q,
                           t_end=200 * YEAR, n_steps=300)
    p_ana = steady_state_radial(g, K, MU, Q, PI)
    assert np.max(np.abs(r.p[-1] - p_ana)) / np.max(np.abs(p_ana - PI)) < 1e-9
    assert r.mass_balance_error < 1e-9


@pytest.mark.parametrize("r_near", [0.15, 5.0, 20.0])
def test_well_index_reproduces_analytical_bhp(r_near):
    """The coarse well block must not cost accuracy in the bottom-hole
    pressure -- that is the whole justification for introducing it."""
    g = RadialGrid(n=50, r_w=0.15, r_e=3000.0, h=H, r_near=r_near)
    r = solve_single_phase(g, K, PHI, MU, CT, PI, q_well=Q,
                           t_end=200 * YEAR, n_steps=300)
    p_bh = r.p[-1, 0] + Q / (g.well_index_geom() * K / MU)
    p_bh_ana = PI + Q * MU / (2 * np.pi * K * H) * np.log(g.r_e / g.r_w)
    assert abs(p_bh - p_bh_ana) / abs(p_bh_ana - PI) < 1e-9


def test_closed_domain_material_balance():
    g = RadialGrid(n=50, r_w=0.15, r_e=3000.0, h=H, r_near=5.0)
    r = solve_single_phase(g, K, PHI, MU, CT, PI, q_well=Q, t_end=YEAR,
                           n_steps=200, outer_bc="closed")
    Vp = float((g.bulk_volume * PHI).sum())
    p_avg = float((r.p[-1] * g.bulk_volume * PHI).sum() / Vp)
    assert p_avg == pytest.approx(tank_material_balance(YEAR, Q, Vp, CT, PI),
                                  rel=1e-9)


def test_closed_domain_without_compressibility_is_rejected():
    """An incompressible closed domain with injection has no solution; the
    solver must say so instead of returning a silently wrong answer."""
    g = RadialGrid(n=20, r_w=0.15, r_e=1000.0, h=H)
    with pytest.raises(ValueError):
        solve_single_phase(g, K, PHI, MU, 0.0, PI, q_well=Q, t_end=YEAR,
                           n_steps=10, outer_bc="closed")


def _bl_case(n):
    fl = FluidProperties(c_a=0.0, c_g=0.0)
    rp, rk = RelPerm(), RockProperties(c_r=0.0)
    L, A, phi, q_vol = 100.0, 10.0, 0.20, 1e-5
    T = 0.4 * L * A * phi / q_vol
    g = CartesianGrid1D(n=n, L=L, area=A)
    m = TwoPhaseModel([g], np.full((1, n), md_to_m2(100.0)),
                      np.full((1, n), phi), fl, rp, rk, PI,
                      cfl=0.3, max_dS=0.05, dt_init=100.0, dt_max=T / 50)
    res = m.run(InjectionSchedule([0.0, T], [q_vol * fl.rho_g]), np.array([T]))
    return g, res, fl, rp, T, q_vol, A, phi


def test_buckley_leverett_front_position():
    g, res, fl, rp, T, q_vol, A, phi = _bl_case(400)
    S_gf, _, v = welge_shock(rp, fl)
    x_ana = q_vol / (A * phi) * v * T
    Sg = res.Sg[-1, 0]
    idx = np.where(Sg > 0.5 * S_gf)[0]
    x_sim = g.centres[idx[-1]]
    assert abs(x_sim - x_ana) / x_ana < 0.05


def test_buckley_leverett_converges_first_order():
    errs = []
    for n in (100, 200, 400):
        g, res, fl, rp, T, q_vol, A, phi = _bl_case(n)
        ana = bl_profile_1d(rp, fl, g.centres, T, q_vol, A, phi, L=100.0)
        errs.append(float(np.mean(np.abs(res.Sg[-1, 0] - ana))))
    # first-order upwind: each doubling should roughly halve the L1 error
    assert errs[1] < 0.7 * errs[0]
    assert errs[2] < 0.7 * errs[1]


def test_two_phase_mass_balance_and_bounds_no_clipping():
    """CO2 mass balance must close, saturations must stay physical, and this
    must happen WITHOUT the solver clipping anything."""
    n_l, n_r = 3, 40
    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=2500.0, h=12.0, r_near=8.0)
             for _ in range(n_l)]
    k = np.full((n_l, n_r), md_to_m2(150.0))
    k[0] *= 4.0
    m = TwoPhaseModel(grids, k, np.full((n_l, n_r), 0.20), FluidProperties(),
                      RelPerm(), RockProperties(), PI, outer_bc="closed",
                      max_dS=0.05, dt_init=3600.0, dt_max=90 * DAY)
    sch = InjectionSchedule([0.0, 3 * YEAR, 6 * YEAR], [12.0, 0.0])
    res = m.run(sch, np.linspace(0.0, 6 * YEAR, 13))
    assert res.mass_balance_error < 1e-8
    assert res.diagnostics["n_clipped"] == 0
    assert res.Sg.min() >= -1e-12
    assert res.Sg.max() <= 1.0 - RelPerm().S_ar + 1e-9
    # a sealed compartment loses nothing: retained == injected
    assert res.mass_out[-1] == pytest.approx(0.0, abs=1e-9)
    assert res.mass_in_place[-1] == pytest.approx(res.mass_injected[-1],
                                                  rel=1e-8)


def test_open_boundary_can_lose_co2_and_still_balances():
    grids = [RadialGrid(n=40, r_w=0.15, r_e=400.0, h=20.0, r_near=6.0)]
    m = TwoPhaseModel(grids, np.full((1, 40), md_to_m2(300.0)),
                      np.full((1, 40), 0.20), FluidProperties(), RelPerm(),
                      RockProperties(), PI, outer_bc="constant_pressure",
                      max_dS=0.05, dt_init=3600.0, dt_max=30 * DAY)
    res = m.run(InjectionSchedule([0.0, 3 * YEAR], [20.0]),
                np.array([3 * YEAR]))
    assert res.mass_out[-1] > 0, "plume should reach the small outer boundary"
    assert res.mass_balance_error < 1e-8
    assert res.mass_in_place[-1] < res.mass_injected[-1]


def test_time_step_control_does_not_change_the_answer():
    """A 5x tighter saturation limiter must not move the QoIs."""
    out = []
    for max_dS in (0.10, 0.02):
        grids = [RadialGrid(n=40, r_w=0.15, r_e=2500.0, h=30.0, r_near=8.0)]
        m = TwoPhaseModel(grids, np.full((1, 40), md_to_m2(200.0)),
                          np.full((1, 40), 0.20), FluidProperties(), RelPerm(),
                          RockProperties(), PI, outer_bc="closed",
                          max_dS=max_dS, dt_init=3600.0, dt_max=90 * DAY)
        res = m.run(InjectionSchedule([0.0, 3 * YEAR, 5 * YEAR], [15.0, 0.0]),
                    np.array([5 * YEAR]))
        out.append((res.diagnostics["p_bh_max_Pa"], res.mass_in_place[-1]))
    assert abs(out[0][0] - out[1][0]) / abs(out[0][0] - PI) < 0.02
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-6)


def test_layer_rate_allocation_open_boundary_follows_flow_capacity():
    """Open (constant-pressure) boundary, two-phase: the high-kh layer must
    take most of the CO2.  (Exact kh proportionality is checked in the
    single-phase limit in tests/test_well_coupling.py.)"""
    n_r = 40
    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=2500.0, h=10.0, r_near=8.0)
             for _ in range(2)]
    k = np.stack([np.full(n_r, md_to_m2(400.0)), np.full(n_r, md_to_m2(100.0))])
    m = TwoPhaseModel(grids, k, np.full((2, n_r), 0.20), FluidProperties(),
                      RelPerm(), RockProperties(), PI,
                      outer_bc="constant_pressure",
                      max_dS=0.05, dt_init=3600.0, dt_max=60 * DAY)
    res = m.run(InjectionSchedule([0.0, 2 * YEAR], [10.0]),
                np.array([2 * YEAR]))
    mass = (m.Vp * res.Sg[-1]).sum(axis=1)
    assert 2.0 < mass[0] / mass[1] < 8.0
    assert res.diagnostics["max_bhp_spread_Pa"] < 1e-3


# ---------------------------------------------------------------- outputs
def test_plume_radius_definitions():
    g = RadialGrid(n=50, r_w=0.15, r_e=1000.0, h=10.0, r_near=5.0)
    Sg = np.where(g.centres < 300.0, 0.5, 0.0)
    r_thr = plume_radius(g.centres, Sg, threshold=0.05)
    assert 200.0 < r_thr < 360.0
    Vp = g.bulk_volume * 0.2
    r95 = plume_radius_mass_fraction([g], Vp[None, :], Sg[None, :], 0.95)
    assert 200.0 < r95 <= 320.0
    assert plume_radius(g.centres, np.zeros(50)) == 0.0


def test_no_dead_capillary_switch_remains():
    """Defect (c): a Pc parameter existed that the equations ignored.  The
    model now states Pc = 0 (4-CO2 BL.pdf p.19) and exposes no such switch."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(RelPerm)}
    assert not any(n.startswith("pc") for n in names)
    assert not hasattr(RelPerm(), "pc")
