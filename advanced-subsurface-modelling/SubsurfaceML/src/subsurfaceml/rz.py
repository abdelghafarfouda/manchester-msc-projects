"""Two-dimensional radial-vertical (r-z) IMPES model with gravity and
vertical crossflow -- a *reference* model for the model-form error of the
layered simulator.

Why this exists
---------------
The production simulator (:mod:`impes`) treats each layer as a horizontal,
1-D radial domain: no gravity, no flow between layers.  Gravity (buoyant
rise and override of CO2 under the caprock) and crossflow are the largest
omissions named in the project's review.  This module adds exactly those two
processes and nothing else -- the same fluids, relative permeabilities,
compressibilities, well model and sealed outer boundary -- so that the
change in each machine-learning target between the two models measures the
model-form error of the layered, no-gravity assumption for the cases of the
dataset (``scripts/run_model_form.py``).  It is not used to generate the
training data.

Discretisation
--------------
Cells ``(j, i)``: ``j`` rows from the top (depth ``z`` positive downwards),
``i`` radial cells of the same log-spaced grid with a well block as
:class:`grid.RadialGrid`.  Each layer of the layered model is split into
``n_sub`` rows of equal thickness with the layer's properties.

* radial transmissibility within a row: ``2 pi dz / ln(r_{i+1}/r_i)`` times
  the (homogeneous) row permeability;
* vertical transmissibility: ``A_i / (dz_j/2 k_v,j + dz_{j+1}/2 k_v,j+1)``,
  ``A_i = pi (r_{i+1/2}^2 - r_{i-1/2}^2)``, ``k_v = (k_v/k_h) k_h``;
* phase fluxes ``F_a = T lambda_a^up [(p_a - p_b) - rho_a g (z_a - z_b)]``,
  phase-potential upwinding (counter-current flow is allowed);
* pressure: the sum of the volumetric phase equations, implicit, with all
  saturation-dependent coefficients lagged (IMPES), solved together with the
  bottom-hole pressure (bordered sparse system);
* CO2 saturation: explicit and conservative in CO2 mass, as in :mod:`impes`;
* well: perforated over the full thickness, one bottom-hole pressure
  ``p_bh`` at the reference (mid-) depth, a CO2-filled wellbore
  ``p_w(z) = p_bh + rho_g g (z - z_ref)``, the radial well index per row,
  injector-only completions (active set) and all completions closed at
  shut-in, as in the layered model;
* initial state: brine at hydrostatic equilibrium with ``p_init`` at the
  reference depth; top, bottom and outer boundary sealed.

With ``kv_over_kh = 0`` and ``gravity = 0`` the rows decouple and the model
reduces to the layered model (verified to round-off in ``tests/test_rz.py``).
"""
from __future__ import annotations

import time as _time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fluids import FluidProperties, RelPerm, RockProperties
from .grid import RadialGrid
from .impes import InjectionSchedule, SimulationFailure

G_EARTH = 9.80665


class RZModel:
    def __init__(self, radial: RadialGrid, dz, k_h, phi, fluids: FluidProperties,
                 relperm: RelPerm, rock: RockProperties, p_init: float, *,
                 kv_over_kh: float = 0.1, gravity: float = G_EARTH,
                 cfl: float = 0.9, max_dS: float = 0.05, dt_init: float = 3600.0,
                 dt_max: float = 180 * 86400.0, dt_min: float = 1.0,
                 max_steps: int = 400_000):
        self.rg = radial
        self.dz = np.asarray(dz, float)
        self.nz, self.nr = self.dz.size, radial.n
        self.kh = np.asarray(k_h, float)
        self.phi = np.asarray(phi, float)
        self.fl, self.rp, self.rk = fluids, relperm, rock
        self.p_ref = float(p_init)
        self.g = float(gravity)
        self.kv_over_kh = float(kv_over_kh)
        self.cfl, self.max_dS = float(cfl), float(max_dS)
        self.dt_init, self.dt_max, self.dt_min = dt_init, dt_max, dt_min
        self.max_steps = int(max_steps)
        nz, nr = self.nz, self.nr
        zf = np.concatenate([[0.0], np.cumsum(self.dz)])
        self.zc = 0.5 * (zf[:-1] + zf[1:])                       # (nz,)
        self.z_ref = 0.5 * zf[-1]
        # per-row radial grids (same faces, thickness dz_j)
        self.grids = [RadialGrid(n=nr, r_w=radial.r_w, r_e=radial.r_e, h=float(h),
                                 r_near=radial.r_near) for h in self.dz]
        self.Vb = np.stack([g.bulk_volume for g in self.grids])  # (nz, nr)
        self.Vp = self.Vb * self.phi[:, None]
        # radial transmissibility (geometric x k), rows homogeneous in r
        geo_r = np.stack([g.geom_factor() for g in self.grids])  # (nz, nr-1)
        self.Tr = geo_r * self.kh[:, None]
        # vertical transmissibility between rows j and j+1
        area = np.pi * (radial.faces[1:] ** 2 - radial.faces[:-1] ** 2)  # (nr,)
        kv = self.kv_over_kh * self.kh
        if nz > 1 and self.kv_over_kh > 0:
            res = (0.5 * self.dz[:-1] / kv[:-1] + 0.5 * self.dz[1:] / kv[1:])
            self.Tz = area[None, :] / res[:, None]                # (nz-1, nr)
        else:
            self.Tz = np.zeros((max(nz - 1, 0), nr))
        self.WI = np.array([g.well_index_geom() for g in self.grids]) * self.kh
        # initial hydrostatic brine pressure
        self.p0 = self.p_ref + self.fl.rho_a * self.g * (self.zc - self.z_ref)  # (nz,)
        self._S_tab = np.linspace(0.0, 1.0 - self.rp.S_ar, 1001)
        self._dfg_tab = np.abs(self.rp.dfg_dSg(self._S_tab, self.fl))
        # face lists (flattened cell index c = j*nr + i)
        idx = np.arange(nz * nr).reshape(nz, nr)
        self.fa = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
        self.fb = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
        self.fT = np.concatenate([self.Tr.ravel(), self.Tz.ravel()])
        dzab = np.concatenate([np.zeros(nz * (nr - 1)),
                               np.repeat(self.zc[:-1] - self.zc[1:], nr)])
        self.dzab = dzab                                         # z_a - z_b
        keep = self.fT > 0
        self.fa, self.fb, self.fT, self.dzab = (x[keep] for x in
                                                (self.fa, self.fb, self.fT, self.dzab))
        self.well_cells = idx[:, 0]

    # ------------------------------------------------------------------
    def run(self, schedule: InjectionSchedule, report_times) -> dict:
        nz, nr, n = self.nz, self.nr, self.nz * self.nr
        fl, rp, rk = self.fl, self.rp, self.rk
        p = np.repeat(self.p0[:, None], nr, axis=1).ravel()
        p0 = p.copy()
        Sg = np.zeros(n)
        Vp = self.Vp.ravel()
        report_times = np.asarray(report_times, float)
        out_t, out_S, out_pbh = [0.0], [Sg.copy()], [self.p_ref]
        cum_inj = 0.0
        t, dt = 0.0, self.dt_init
        n_steps = n_rej = 0
        p_bh_max = self.p_ref
        breaks = np.unique(np.concatenate([schedule.t_edges, report_times]))
        t_final = float(max(schedule.t_end, report_times[-1]))
        ri = int(np.sum(report_times <= 0))
        t0 = _time.perf_counter()
        while t < t_final - 1e-9:
            nxt = breaks[breaks > t + 1e-9]
            dt_try = min(dt, float(nxt[0]) - t if nxt.size else t_final - t, self.dt_max)
            while True:
                dt_try = max(dt_try, self.dt_min)
                res = self._step(p, p0, Sg, Vp, dt_try, schedule.rate_at(t + 0.5 * dt_try))
                bad = (res is None or res["max_dS"] > self.max_dS or res["oor"] > 1e-3)
                if not bad:
                    break
                if dt_try <= self.dt_min * 1.0000001:
                    raise SimulationFailure(f"r-z step failed at t={t:.4g} s")
                dt_try *= 0.5
                n_rej += 1
            p, Sg = res["p"], np.clip(res["Sg"], 0.0, 1.0 - rp.S_ar)
            if res["injecting"]:
                p_bh_max = max(p_bh_max, res["p_bh"])
            cum_inj += res["q_mass"] * dt_try
            t += dt_try
            n_steps += 1
            if n_steps > self.max_steps:
                raise SimulationFailure("r-z model exceeded max_steps")
            growth = (float(np.clip(0.85 * self.max_dS / res["max_dS"], 0.5, 1.6))
                      if res["max_dS"] > 1e-12 else 1.6)
            dt = min(self.dt_max, max(self.dt_min, min(dt_try * growth, res["dt_cfl"])))
            while ri < len(report_times) and report_times[ri] <= t + 1e-6:
                out_t.append(report_times[ri]); out_S.append(Sg.copy())
                out_pbh.append(res["p_bh"]); ri += 1
        dp = p - p0
        mass = float(np.sum(Vp * (1 + rk.c_r * dp) * Sg * fl.rho_g * (1 + fl.c_g * dp)))
        return {"t": np.array(out_t), "Sg": np.array(out_S).reshape(-1, nz, nr),
                "p_bh": np.array(out_pbh), "p": p.reshape(nz, nr),
                "p_bh_max_Pa": p_bh_max, "dp_bh_max_Pa": p_bh_max - self.p_ref,
                "mass_injected_kg": cum_inj, "mass_in_place_kg": mass,
                "mass_balance_error": abs(mass - cum_inj) / max(cum_inj, 1e-30),
                "n_steps": n_steps, "n_rejects": n_rej,
                "wall_time_s": _time.perf_counter() - t0}

    # ------------------------------------------------------------------
    def _step(self, p, p0, Sg, Vp, dt, q_mass):
        fl, rp, rk, g = self.fl, self.rp, self.rk, self.g
        n = p.size
        a, b, T, dzab = self.fa, self.fb, self.fT, self.dzab
        lam_a, lam_g = rp.lam_a(Sg, fl), rp.lam_g(Sg, fl)
        dpab = p[a] - p[b]
        pot_a = dpab - fl.rho_a * g * dzab
        pot_g = dpab - fl.rho_g * g * dzab
        la = np.where(pot_a >= 0, lam_a[a], lam_a[b])
        rr = 1.0 + fl.c_g * (p - p0)                 # CO2 density ratio
        lg_vol = np.where(pot_g >= 0, lam_g[a], lam_g[b])
        lg_mass = np.where(pot_g >= 0, (lam_g * rr)[a], (lam_g * rr)[b])
        Tt = T * (la + lg_vol)
        Gab = T * (la * fl.rho_a + lg_vol * fl.rho_g) * g * dzab   # flux a->b at equal p
        c_t = rk.c_r + (1 - Sg) * fl.c_a + Sg * fl.c_g
        C = Vp * c_t / dt
        # A dp = rhs with dp = p_new - p ; flux a->b = Tt (p_a - p_b) - Gab
        diag = C.copy()
        np.add.at(diag, a, Tt)
        np.add.at(diag, b, Tt)
        rhs = np.zeros(n)
        f0 = Tt * dpab - Gab                          # current flux a->b
        np.add.at(rhs, a, -f0)
        np.add.at(rhs, b, f0)
        Q = q_mass / fl.rho_g
        wc = self.well_cells
        lam_t_w = (lam_a + lam_g)[wc]
        J_all = self.WI * lam_t_w
        hydro_w = fl.rho_g * g * (self.zc - self.z_ref)   # wellbore head, CO2 column
        injecting = Q > 0
        active = J_all > 0 if injecting else np.zeros_like(J_all, bool)
        rows = np.concatenate([a, b, a, b])
        cols = np.concatenate([b, a, a, b])
        vals = np.concatenate([-Tt, -Tt, np.zeros_like(Tt), np.zeros_like(Tt)])
        base = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
        for _ in range(self.nz + 1):
            J = np.where(active, J_all, 0.0)
            if injecting:
                # bordered system: unknowns dp (n) and d_bh = p_bh - p_ref
                Jd = np.zeros(n); Jd[wc] = J
                A = base + sp.diags(Jd)
                col = np.zeros(n); col[wc] = -J
                r_w = rhs.copy()
                r_w[wc] += J * (hydro_w + self.p_ref - p[wc])
                Ab = sp.bmat([[A, sp.csr_matrix(col[:, None])],
                              [sp.csr_matrix(-Jd[None, :]), sp.csr_matrix([[J.sum()]])]],
                             format="csc")
                rb = np.concatenate([r_w, [Q - float(np.sum(J * (hydro_w + self.p_ref - p[wc])))]])
                try:
                    sol = spla.spsolve(Ab, rb)
                except Exception:
                    return None
                dpn, d_bh = sol[:n], float(sol[n])
                p_new = p + dpn
                q_l = J * (self.p_ref + d_bh + hydro_w - p_new[wc])
                neg = active & (q_l < -1e-12 * Q)
                if not neg.any():
                    break
                active &= ~neg
            else:
                try:
                    dpn = spla.spsolve(base.tocsc(), rhs)
                except Exception:
                    return None
                p_new = p + dpn
                q_l = np.zeros_like(J_all)
                d_bh = float(np.sum(J_all * (p_new[wc] - hydro_w)) / max(J_all.sum(), 1e-300)) - self.p_ref
                break
        if not np.all(np.isfinite(p_new)):
            return None
        q_l = np.maximum(q_l, 0.0)
        # explicit CO2 update, conservative in CO2 mass
        dpn_ab = p_new[a] - p_new[b]
        Fg = T * lg_mass * (dpn_ab - fl.rho_g * g * dzab)          # a -> b, reference volume
        flux = np.zeros(n)
        np.add.at(flux, a, -Fg)
        np.add.at(flux, b, Fg)
        flux[wc] += q_l
        B_old = (1 + rk.c_r * (p - p0)) * (1 + fl.c_g * (p - p0))
        B_new = (1 + rk.c_r * (p_new - p0)) * (1 + fl.c_g * (p_new - p0))
        Sg_new = (Sg * B_old + dt * flux / Vp) / B_new
        hi = 1.0 - rp.S_ar
        oor = float(max(np.max(-Sg_new), np.max(Sg_new - hi), 0.0))
        # CFL from the throughput of both phases
        Fa = T * la * (dpn_ab - fl.rho_a * g * dzab)
        Fgv = T * lg_vol * (dpn_ab - fl.rho_g * g * dzab)
        thr = np.zeros(n)
        F = np.abs(Fa) + np.abs(Fgv)
        np.maximum.at(thr, a, F)
        np.maximum.at(thr, b, F)
        thr[wc] = np.maximum(thr[wc], q_l)
        dfg = np.maximum(np.interp(Sg, self._S_tab, self._dfg_tab), 1e-6)
        with np.errstate(divide="ignore"):
            dt_cfl = float(self.cfl * np.min(Vp / np.maximum(thr * dfg, 1e-30)))
        return {"p": p_new, "Sg": Sg_new, "max_dS": float(np.max(np.abs(Sg_new - Sg))),
                "oor": oor, "dt_cfl": dt_cfl, "p_bh": self.p_ref + d_bh,
                "q_mass": q_mass, "injecting": injecting}


# --------------------------------------------------------------------------
def build_rz(cfg, realisation, *, n_sub: int = 5, kv_over_kh: float = 0.1,
             gravity: float = G_EARTH):
    """The r-z counterpart of :func:`scenarios.build_model`: the same layered
    rock (same seed), each layer split into ``n_sub`` rows."""
    from .petrophysics import make_layered_rock
    from .scenarios import fluid_models
    from .units import MPA, md_to_m2
    r, g, s = realisation, cfg.grid, cfg.solver
    rock = make_layered_rock(g.n_layers, g.n_r, r.h_total_m, md_to_m2(r.k_median_mD),
                             r.V_DP, r.phi_mean, seed=r.seed)
    kh = np.repeat(rock.k[:, 0], n_sub)
    phi = np.repeat(rock.phi[:, 0], n_sub)
    dz = np.repeat(rock.h / n_sub, n_sub)
    rad = RadialGrid(n=g.n_r, r_w=g.r_w_m, r_e=r.r_e_m, h=1.0, r_near=g.r_near_m)
    fl, rp = fluid_models(r)
    return RZModel(rad, dz, kh, phi, fl, rp, RockProperties(), s.p_init_MPa * MPA,
                   kv_over_kh=kv_over_kh, gravity=gravity, cfl=s.cfl, max_dS=s.max_dS,
                   dt_init=s.dt_init_s, dt_max=s.dt_max_days * 86400.0, dt_min=s.dt_min_s)


def run_rz_scenario(cfg, realisation, rates_kg_s, **kw) -> dict:
    """Targets of one scenario computed with the r-z model."""
    from .outputs import plume_radius_mass_fraction, sweep_efficiency
    from .units import MPA, YEAR
    m = build_rz(cfg, realisation, **kw)
    sch = cfg.schedule
    edges = np.concatenate([np.linspace(0.0, sch.t_inject_years * YEAR, sch.n_periods + 1),
                            [sch.t_total_years * YEAR]])
    schedule = InjectionSchedule(edges, np.append(np.asarray(rates_kg_s, float), 0.0))
    res = m.run(schedule, np.linspace(0.0, schedule.t_end, 5))
    S_end = res["Sg"][-1]
    top = S_end[: max(1, m.nz // 4)]
    return {"dp_bh_max_MPa": res["dp_bh_max_Pa"] / MPA,
            "r_plume_m95_m": plume_radius_mass_fraction(m.grids, m.Vp, S_end, 0.95),
            "sweep_efficiency": sweep_efficiency(m.Vp, S_end),
            "co2_fraction_in_top_quarter": float(np.sum((m.Vp * S_end)[: max(1, m.nz // 4)])
                                                 / max(np.sum(m.Vp * S_end), 1e-30)),
            "mass_balance_error": res["mass_balance_error"],
            "n_steps": res["n_steps"], "wall_time_s": res["wall_time_s"],
            "top_rows_max_sg": float(top.max())}
