"""The r-z reference model (gravity + vertical crossflow) used to measure the
model-form error of the layered simulator."""
from __future__ import annotations

import numpy as np
import pytest

from subsurfaceml.fluids import FluidProperties, RelPerm, RockProperties
from subsurfaceml.grid import RadialGrid
from subsurfaceml.impes import InjectionSchedule, TwoPhaseModel
from subsurfaceml.outputs import extract_qois
from subsurfaceml.rz import RZModel
from subsurfaceml.units import DAY, MPA, YEAR, md_to_m2

FL, RP, RK = FluidProperties(), RelPerm(), RockProperties()


def _layered_and_rz(n_sub, kv, g):
    """Two layers, sealed, compared with the r-z model built from the same
    layers (each split into ``n_sub`` rows)."""
    n, re = 20, 800.0
    k = [md_to_m2(300.0), md_to_m2(30.0)]
    phi = [0.22, 0.12]
    h = [6.0, 6.0]
    grids = [RadialGrid(n=n, r_w=0.15, r_e=re, h=hh, r_near=3.0) for hh in h]
    lay = TwoPhaseModel(grids, np.stack([np.full(n, kk) for kk in k]),
                        np.stack([np.full(n, p) for p in phi]), FL, RP, RK, 15 * MPA,
                        outer_bc="closed", dt_max=20 * DAY)
    sch = InjectionSchedule([0.0, 1.0 * YEAR, 1.5 * YEAR], [4.0, 0.0])
    r1 = lay.run(sch, np.linspace(0, 1.5 * YEAR, 5))
    q1 = extract_qois(r1, grids, lay)
    rad = RadialGrid(n=n, r_w=0.15, r_e=re, h=1.0, r_near=3.0)
    rz = RZModel(rad, np.repeat(np.array(h) / n_sub, n_sub), np.repeat(k, n_sub),
                 np.repeat(phi, n_sub), FL, RP, RK, 15 * MPA, kv_over_kh=kv, gravity=g,
                 dt_max=20 * DAY)
    r2 = rz.run(sch, np.linspace(0, 1.5 * YEAR, 5))
    return q1, r2, rz


def test_reduces_to_layered_model_without_gravity_and_crossflow():
    q1, r2, _ = _layered_and_rz(1, 0.0, 0.0)
    assert abs(r2["dp_bh_max_Pa"] - q1["dp_bh_max_Pa"]) / q1["dp_bh_max_Pa"] < 1e-4
    _, r3, _ = _layered_and_rz(3, 0.0, 0.0)          # splitting rows changes nothing
    assert abs(r3["dp_bh_max_Pa"] - r2["dp_bh_max_Pa"]) / r2["dp_bh_max_Pa"] < 1e-10
    assert r2["mass_balance_error"] < 1e-10


def test_gravity_and_crossflow_conserve_co2_and_lift_it():
    _, r, m = _layered_and_rz(4, 0.1, 9.80665)
    assert r["mass_balance_error"] < 1e-8
    S = r["Sg"][-1]
    top = np.sum((m.Vp * S)[:2]) / np.sum(m.Vp * S)
    assert top > 0.25                                   # buoyancy moves CO2 upwards


def test_hydrostatic_equilibrium_is_preserved():
    rad = RadialGrid(n=6, r_w=0.15, r_e=200.0, h=1.0, r_near=2.0)
    m = RZModel(rad, np.full(10, 2.0), np.full(10, md_to_m2(200.0)), np.full(10, 0.2),
                FL, RP, RK, 15 * MPA, kv_over_kh=0.5)
    res = m.run(InjectionSchedule([0.0, 5 * YEAR], [0.0]), np.array([5 * YEAR]))
    p0 = np.repeat(m.p0[:, None], m.nr, axis=1)
    assert np.max(np.abs(res["p"] - p0)) < 1e-6 and res["Sg"].max() == 0.0


def test_gravity_segregation_reaches_the_analytical_final_state():
    """Closed column, uniform S_g = 0.3: CO2 rises until the top holds
    1 - S_ar and the rest keeps the residual S_gr (immobile drainage end
    point); the top-zone height follows from mass balance."""
    nz, H = 20, 20.0
    rad = RadialGrid(n=3, r_w=0.15, r_e=20.0, h=1.0, r_near=1.0)
    m = RZModel(rad, np.full(nz, H / nz), np.full(nz, md_to_m2(500.0)), np.full(nz, 0.2),
                FL, RP, RK, 15 * MPA, kv_over_kh=1.0, dt_max=5 * DAY)
    n = nz * m.nr
    p = np.repeat(m.p0[:, None], m.nr, axis=1).ravel(); p0 = p.copy()
    Sg = np.full(n, 0.3); Vp = m.Vp.ravel(); m0 = np.sum(Vp * Sg)
    t, dt, T = 0.0, m.dt_init, 30 * YEAR
    while t < T - 1e-6:
        dtt = min(dt, T - t, m.dt_max)
        while True:
            r = m._step(p, p0, Sg, Vp, dtt, 0.0)
            if r is not None and r["max_dS"] <= m.max_dS and r["oor"] <= 1e-3:
                break
            dtt *= 0.5
        p, Sg = r["p"], np.clip(r["Sg"], 0, 1 - RP.S_ar)
        t += dtt
        dt = min(m.dt_max, max(m.dt_min, min(dtt * 1.5, r["dt_cfl"])))
    dp = p - p0
    m1 = np.sum(Vp * (1 + RK.c_r * dp) * Sg * (1 + FL.c_g * dp))
    assert abs(m1 - m0) / m0 < 1e-10
    prof = Sg.reshape(nz, m.nr)[:, 1]
    assert prof[0] == pytest.approx(1 - RP.S_ar, abs=0.01)
    assert prof[-1] == pytest.approx(RP.S_gr, abs=1e-3)
    h_top = np.sum(prof > 0.5 * (1 - RP.S_ar + RP.S_gr)) * H / nz
    h_ana = (0.3 - RP.S_gr) * H / ((1 - RP.S_ar) - RP.S_gr)
    assert abs(h_top - h_ana) <= H / nz + 1e-9
