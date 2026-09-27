"""Multi-layer well coupling: one rate-controlled injector, ONE bottom-hole
pressure.

The well model (``impes`` module docstring) is

    q_l = J_l (p_bh - p_l0),   sum_l q_l = Q,   J_l = WI_l lambda_t,l0

These tests check the three properties that define it -- the prescribed
total rate is conserved, every open completion sees the same p_bh, and the
split between layers has the physically correct limits -- plus the stated
shut-in and injector-only (no back-flow) behaviour.  Every expected value is
derived from the equations, not from a previous run.
"""
from __future__ import annotations

import numpy as np
import pytest

from subsurfaceml.fluids import FluidProperties, RelPerm, RockProperties
from subsurfaceml.grid import RadialGrid
from subsurfaceml.impes import InjectionSchedule, TwoPhaseModel
from subsurfaceml.units import DAY, MPA, YEAR, md_to_m2

PI = 15.0 * MPA


def _two_layer(k_md=(500.0, 20.0), phi=(0.25, 0.08), outer_bc="closed",
               fl=None, rp=None, n_r=30, r_e=1500.0, h=10.0):
    grids = [RadialGrid(n=n_r, r_w=0.15, r_e=r_e, h=h, r_near=5.0)
             for _ in range(2)]
    k = np.stack([np.full(n_r, md_to_m2(v)) for v in k_md])
    ph = np.stack([np.full(n_r, v) for v in phi])
    return TwoPhaseModel(grids, k, ph, fl or FluidProperties(), rp or RelPerm(),
                         RockProperties(), PI, outer_bc=outer_bc, max_dS=0.05,
                         dt_init=3600.0, dt_max=30 * DAY)


def _single_phase_like():
    """Identical phases, linear kr with no endpoints: lambda_t = 1/mu for
    every saturation, i.e. the single-phase limit of the two-phase model."""
    fl = FluidProperties(mu_g=6e-4, mu_a=6e-4, rho_g=1000.0)
    rp = RelPerm(S_ar=0.0, S_gr=0.0, n_a=1.0, n_g=1.0, kra0=1.0, krg0=1.0)
    return fl, rp


def test_total_rate_is_conserved_and_bhp_is_common_with_unequal_layers():
    """The case that exposed the old allocation: 500 mD / phi 0.25 against
    20 mD / phi 0.08 in a sealed compartment, so the layer pressures diverge."""
    m = _two_layer()
    res = m.run(InjectionSchedule([0.0, 2 * YEAR], [20.0]),
                np.linspace(0.0, 2 * YEAR, 9))
    d = res.diagnostics
    assert d["max_rate_rel_err"] < 1e-10
    # a common BHP to within round-off (Pa) while the BHP itself is ~1e8 Pa
    assert d["max_bhp_spread_Pa"] < 1e-3
    # the layer rates at every report time add up to the prescribed rate
    q_tot = res.q_layer[1:].sum(axis=1) * m.fl.rho_g
    assert np.allclose(q_tot, 20.0, rtol=1e-10)
    # the well-block pressures really are unequal -- otherwise this test
    # would not distinguish the corrected coupling from the old allocation
    assert abs(res.p[-1, 0, 0] - res.p[-1, 1, 0]) > 1.0 * MPA


def test_step_level_bhp_equality_from_the_well_equation():
    """Recompute p_bh_l = p_l0 + q_l / J_l from one step output."""
    m = _two_layer()
    sch = InjectionSchedule([0.0, 1 * YEAR], [15.0])
    res = m.run(sch, np.array([0.5 * YEAR]))
    p, Sg = res.p[-1], res.Sg[-1]
    out = m._step(p, Sg, 5 * DAY, 0.5 * YEAR, sch)
    J = m.WI_geom * m.rp.lam_t(Sg[:, 0], m.fl)
    implied = out["p"][:, 0] + out["q_layer"] / J
    assert np.ptp(implied) < 1e-6 * implied.mean()
    assert implied[0] == pytest.approx(out["p_bh"], rel=1e-12)
    assert out["q_layer"].sum() * m.fl.rho_g == pytest.approx(15.0, rel=1e-12)


def test_the_old_proportional_allocation_would_not_give_a_common_bhp():
    """Documents the defect: allocating q_l proportional to J_l at the NEW
    pressures gives per-layer implied BHPs that differ by the well-block
    pressure difference, which is large here."""
    m = _two_layer()
    res = m.run(InjectionSchedule([0.0, 1 * YEAR], [20.0]),
                np.array([1 * YEAR]))
    p, Sg = res.p[-1], res.Sg[-1]
    J = m.WI_geom * m.rp.lam_t(Sg[:, 0], m.fl)
    q_old = 20.0 / m.fl.rho_g * J / J.sum()
    implied_old = p[:, 0] + q_old / J
    assert np.ptp(implied_old) > 1.0 * MPA          # old rule: no common BHP


def test_open_boundary_single_phase_limit_splits_by_kh():
    """Steady radial flow to a constant-pressure boundary: q_l = kh_l *
    const, so the split equals the kh ratio (400 mD : 100 mD -> 4 : 1)."""
    fl, rp = _single_phase_like()
    m = _two_layer(k_md=(400.0, 100.0), phi=(0.2, 0.2),
                   outer_bc="constant_pressure", fl=fl, rp=rp)
    res = m.run(InjectionSchedule([0.0, 20 * YEAR], [5.0]),
                np.array([20 * YEAR]))
    q = res.q_layer[-1]
    assert q[0] / q[1] == pytest.approx(4.0, rel=1e-4)


def test_closed_boundary_single_phase_limit_splits_by_pore_volume():
    """Sealed layers at pseudo-steady state: every layer's mean pressure must
    rise at the same rate as the common BHP, so q_l / Q -> Vp_l / sum(Vp),
    independent of permeability (here 400 vs 100 mD, phi 0.3 vs 0.1)."""
    fl, rp = _single_phase_like()
    m = _two_layer(k_md=(400.0, 100.0), phi=(0.3, 0.1), outer_bc="closed",
                   fl=fl, rp=rp, r_e=800.0)
    res = m.run(InjectionSchedule([0.0, 10 * YEAR], [1.0]),
                np.array([10 * YEAR]))
    q = res.q_layer[-1]
    vp = m.Vp.sum(axis=1)
    assert q[0] / q.sum() == pytest.approx(vp[0] / vp.sum(), rel=5e-3)


def test_shut_in_closes_every_completion_no_crossflow():
    """After shut-in the layers stay at unequal pressures but exchange no
    fluid through the well: each layer's CO2 mass is frozen."""
    m = _two_layer()
    t = np.linspace(0.0, 3 * YEAR, 13)
    res = m.run(InjectionSchedule([0.0, 1 * YEAR, 3 * YEAR], [20.0, 0.0]), t)
    shut = t > 1 * YEAR + 1.0
    assert np.all(res.q_layer[shut] == 0.0)
    dp = res.p - PI
    mass_layer = (m.Vp[None] * (1 + m.rk.c_r * dp) * res.Sg * m.fl.rho_g
                  * (1 + m.fl.c_g * dp)).sum(axis=2)       # CO2 mass [kg]
    assert np.allclose(mass_layer[shut], mass_layer[shut][0], rtol=1e-9)
    # BHP maximum is taken over injecting steps only
    assert res.diagnostics["p_bh_max_Pa"] >= np.nanmax(res.p_bh[~shut]) - 1.0
    assert res.mass_balance_error < 1e-10


def test_over_pressured_layer_is_closed_rather_than_producing():
    """Inject, shut in (the tight, low-pore-volume layer stays at a much
    higher pressure), then restart at a small rate: the common BHP is below
    that layer's pressure, so an injector must put nothing into it -- never
    a negative (back-flow) rate."""
    m = _two_layer(k_md=(300.0, 300.0), phi=(0.30, 0.03))
    t = np.linspace(0.0, 2.2 * YEAR, 23)
    res = m.run(InjectionSchedule([0.0, 1 * YEAR, 2 * YEAR, 2.2 * YEAR],
                                  [20.0, 0.0, 0.5]), t)
    assert res.diagnostics["n_steps_layer_closed"] > 0
    assert np.all(res.q_layer >= 0.0)
    late = t > 2 * YEAR + 1.0
    assert np.all(res.q_layer[late, 1] == 0.0)
    assert np.allclose(res.q_layer[late, 0] * m.fl.rho_g, 0.5, rtol=1e-10)
    assert res.mass_balance_error < 1e-10
