"""IMPES two-phase (CO2 / brine) finite-volume simulator.

IMplicit Pressure, Explicit Saturation, following ``3-IMPES.pdf`` p.10-18,
with the oil-water notation mapped to CO2 storage
(``oil -> a`` = brine/displaced, ``water -> g`` = CO2/injected).

Discrete equations
------------------
Primary unknowns are the pressure ``p`` and the CO2 saturation ``S_g``.
With ``B_a = B_g = 1`` (Assumption A3) the coefficient
``beta_i = -Cswo_i / Csww_i`` of ``3-IMPES.pdf`` p.14 reduces to 1, so the
pressure equation is the **sum** of the two phase equations, with every
saturation-dependent coefficient evaluated at the old time level
(``3-IMPES.pdf`` p.13):

.. math::

    \sum_j (T_{a,ij} + T_{g,ij})(p_j - p_i) + Q_i
        = \frac{V_{p,i} c_{t,i}}{\Delta t}(p_i - p_i^n)

with ``T_{l,ij} = G_ij k^harm_ij lambda_l^up`` (harmonic permeability,
``1-Transmissibility.pdf`` p.18-19; upstream mobility, ``3-IMPES.pdf``
p.7-8) and ``c_t = c_r + S_a c_a + S_g c_g``.  The saturation is then advanced
**explicitly** (``3-IMPES.pdf`` p.18; see ``_step`` for why the CO2 equation
is used).

Capillary pressure
------------------
``P_c = 0`` throughout.  This is the assumption of the CO2 Buckley-Leverett
lecture (``4-CO2 BL.pdf`` p.19, ``3-Advanced BL.pptx`` slide 10: "no
capillary and/or gravity forces").  The IMPES lecture carries ``P_cow`` in
the equations (``3-IMPES.pdf`` p.10-15) but the supplied material gives no
capillary-pressure curve for a CO2-brine system, so no ``P_c(S)`` model is
included.  An earlier version exposed a ``pc_entry`` parameter that the
equations never used; it has been removed rather than left as a dead switch.

Well model -- one rate-controlled injector, one bottom-hole pressure
---------------------------------------------------------------------
The layers are hydraulically isolated vertically (no crossflow) and
communicate only through the well.  Each completion obeys the radial well
equation of ``3-IMPES.pdf`` p.16,

.. math::

    q_l = J_l\,(p_{bh} - p_{a,l,0} - P_{c,l,0}), \qquad
    J_l = WI_l\,\lambda_{t,l,0}, \qquad
    WI_l = \frac{2\pi k_{l,0} h_l}{\ln(r_0/r_w)}

(total mobility of the well block, lagged -- a project choice that keeps the
injectivity finite before CO2 has arrived), and the single unknown ``p_bh``
is fixed by the rate constraint ``sum_l q_l = Q``.  ``p_bh`` is solved
**implicitly together with the pressure field** (one extra unknown, eliminated
exactly by superposition), so every open completion sees the same
bottom-hole pressure *at the new time level*.

An earlier version allocated ``q_l`` in proportion to ``J_l`` and reported the
maximum of the implied per-layer BHPs.  That enforces a common BHP only when
all well-block pressures are equal; with unequal layer pressures it did not
(a two-layer diagnostic gave 103 vs 38 MPa).  ``tests/test_well_coupling.py``
covers the corrected behaviour.

*Injector-only completions.*  If a layer's well-block gas pressure exceeds
the common BHP it would flow back into the wellbore.  An injector does not
produce, so that completion is closed for the step and the solve repeated
(active set); ``diagnostics['n_steps_layer_closed']`` counts such steps.

*Shut-in* (``Q = 0``): every completion is closed -- no fluid enters or
leaves any layer through the well and there is **no wellbore crossflow**.
The reported ``p_bh`` is then a diagnostic: the pressure a zero-net-rate open
wellbore would take, ``sum_l J_l (p_ws - p_l0) = 0``.  ``p_bh_max`` (and
therefore ``dp_bh_max``) is taken over injecting steps only.

Stability
---------
The explicit saturation update is CFL limited.  A step whose saturation
change exceeds ``max_dS``, leaves the physical range, or violates a discrete
local maximum principle is **rejected and retried with a halved step** --
saturations are never silently clipped.  Any residual clipping is counted and
reported in ``Result.diagnostics['n_clipped']``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time as _time

import numpy as np
from scipy.linalg import solve_banded

from .fluids import FluidProperties, RelPerm, RockProperties
from .grid import RadialGrid, CartesianGrid1D


class SimulationFailure(RuntimeError):
    """Raised when the IMPES time-step control cannot produce a valid step.

    Scenario generation catches this and records the run in
    ``failed_runs.csv`` rather than dropping it silently.
    """


# --------------------------------------------------------------------------
@dataclass
class InjectionSchedule:
    """Piecewise-constant CO2 **mass** injection schedule.

    ``t_edges`` has length ``n+1`` [s]; ``rates`` has length ``n`` [kg/s].
    A zero rate is a shut-in period.  ``t_edges[-1]`` is the end of the
    simulation (post-injection monitoring is expressed as a trailing
    zero-rate period).
    """

    t_edges: np.ndarray
    rates: np.ndarray

    def __post_init__(self) -> None:
        self.t_edges = np.asarray(self.t_edges, float)
        self.rates = np.asarray(self.rates, float)
        if self.t_edges.size != self.rates.size + 1:
            raise ValueError("len(t_edges) must be len(rates)+1")
        if np.any(np.diff(self.t_edges) <= 0):
            raise ValueError("t_edges must be strictly increasing")
        if np.any(self.rates < 0):
            raise ValueError("negative injection rates are not supported")

    @property
    def t_end(self) -> float:
        return float(self.t_edges[-1])

    def rate_at(self, t: float) -> float:
        i = int(np.searchsorted(self.t_edges, t, side="right") - 1)
        i = min(max(i, 0), self.rates.size - 1)
        return float(self.rates[i])

    @property
    def total_mass(self) -> float:
        """Planned injected mass [kg]."""
        return float(np.sum(self.rates * np.diff(self.t_edges)))

    def to_dict(self) -> dict:
        return {"t_edges_s": self.t_edges.tolist(),
                "rates_kg_s": self.rates.tolist(),
                "total_mass_kg": self.total_mass}


@dataclass
class ImpesResult:
    t: np.ndarray                 #: (nt,) report times [s]
    p: np.ndarray                 #: (nt, n_layers, n_r) pressure [Pa]
    Sg: np.ndarray                #: (nt, n_layers, n_r) CO2 saturation [-]
    p_bh: np.ndarray              #: (nt,) bottom-hole pressure [Pa]
    q_mass: np.ndarray            #: (nt,) instantaneous injection rate [kg/s]
    mass_injected: np.ndarray     #: (nt,) cumulative injected CO2 [kg]
    mass_out: np.ndarray          #: (nt,) cumulative CO2 leaving r_e [kg]
    mass_in_place: np.ndarray     #: (nt,) free-phase CO2 in the domain [kg]
    diagnostics: dict = field(default_factory=dict)
    q_layer: np.ndarray | None = None  #: (nt, n_layers) layer rates [ref m^3/s]

    @property
    def mass_balance_error(self) -> float:
        """Relative CO2 mass-balance error at the final time."""
        inj = self.mass_injected[-1]
        if inj <= 0:
            return 0.0
        return float(abs(self.mass_in_place[-1] + self.mass_out[-1] - inj) / inj)


# --------------------------------------------------------------------------
class TwoPhaseModel:
    """Layered (no-crossflow) IMPES CO2-brine model.

    Parameters
    ----------
    grids : list of grid objects, one per layer (``RadialGrid`` or a single
        ``CartesianGrid1D`` for the Buckley-Leverett benchmark)
    k : (n_layers, n_r) absolute permeability [m^2]
    phi : (n_layers, n_r) porosity [-]
    fluids, relperm, rock : property objects
    p_init : initial pressure [Pa]
    outer_bc : ``"constant_pressure"`` (aquifer / pressure relief, default) or
        ``"closed"`` (sealed compartment; storage term supplies the balance)
    Sg_init : initial CO2 saturation [-] (default 0)
    """

    def __init__(self, grids, k, phi, fluids: FluidProperties,
                 relperm: RelPerm, rock: RockProperties, p_init: float,
                 *, outer_bc: str = "constant_pressure",
                 Sg_init: float = 0.0, cfl: float = 0.9,
                 max_dS: float = 0.1, dt_init: float = 3600.0,
                 dt_max: float = 30 * 86400.0, dt_min: float = 1.0,
                 max_overshoot: float = 1e-3, enforce_cfl: bool = True,
                 max_steps: int = 200_000):
        self.grids = list(grids)
        self.n_layers = len(self.grids)
        self.n_r = self.grids[0].n
        self.k = np.asarray(k, float).reshape(self.n_layers, self.n_r)
        self.phi = np.asarray(phi, float).reshape(self.n_layers, self.n_r)
        self.fl, self.rp, self.rk = fluids, relperm, rock
        self.p_init = float(p_init)
        self.outer_bc = outer_bc
        self.Sg_init = float(Sg_init)
        self.cfl, self.max_dS = float(cfl), float(max_dS)
        self.dt_init, self.dt_max, self.dt_min = dt_init, dt_max, dt_min
        self.max_overshoot = float(max_overshoot)
        self.enforce_cfl = bool(enforce_cfl)
        self.max_steps = int(max_steps)
        if not (np.all(np.isfinite(self.k)) and np.all(self.k > 0)):
            raise ValueError("permeability must be finite and positive")
        if not (np.all(self.phi > 0) and np.all(self.phi < 1)):
            raise ValueError("porosity must lie in (0, 1)")
        if outer_bc == "closed" and (rock.c_r + fluids.c_a + fluids.c_g) <= 0:
            raise ValueError("a closed domain needs a positive compressibility; "
                             "an incompressible sealed box cannot accept fluid")
        if outer_bc not in ("constant_pressure", "closed"):
            raise ValueError(f"outer_bc must be 'constant_pressure' or 'closed', "
                             f"got {outer_bc!r}")
        if not (0.0 < cfl <= 1.0):
            raise ValueError("cfl must be in (0, 1]")
        if not (0.0 < max_dS <= 1.0):
            raise ValueError("max_dS must be in (0, 1]")

        # static geometry
        self.geom = np.stack([g.geom_factor() for g in self.grids])       # (L, n-1)
        self.kh_face = np.stack([g.harmonic_k(self.k[l])
                                 for l, g in enumerate(self.grids)])      # (L, n-1)
        self.Tgeom = self.geom * self.kh_face                             # (L, n-1)
        self.Vb = np.stack([g.bulk_volume for g in self.grids])           # (L, n)
        self.Vp = self.Vb * self.phi
        self.Tout_geom = np.array([g.outer_geom_factor() * self.k[l, -1]
                                   for l, g in enumerate(self.grids)])
        if isinstance(self.grids[0], RadialGrid):
            self.WI_geom = np.array([g.well_index_geom() * self.k[l, 0]
                                     for l, g in enumerate(self.grids)])
        else:
            self.WI_geom = np.full(self.n_layers, np.inf)  # not used

        # tabulated |df_g/dS_g| for the *local* CFL bound.  Using the global
        # maximum instead of the local value makes the bound ~2 orders of
        # magnitude too conservative in the near-well cells, where S_g is at
        # its plateau and df_g/dS_g -> 0.
        self._S_tab = np.linspace(0.0, 1.0 - self.rp.S_ar, 1001)
        self._dfg_tab = np.abs(self.rp.dfg_dSg(self._S_tab, self.fl))
        self.dfg_max = float(np.max(self._dfg_tab))

    def _dfg_local(self, Sg):
        return np.interp(Sg, self._S_tab, self._dfg_tab)

    # ---------------------------------------------------------------- utils
    def _upstream(self, lam, p):
        """Upstream mobility on each interior face, shape (L, n-1)."""
        return np.where(p[:, :-1] >= p[:, 1:], lam[:, :-1], lam[:, 1:])

    def _pore_volume_total(self) -> float:
        return float(np.sum(self.Vp))

    # ------------------------------------------------------------------ run
    def run(self, schedule: InjectionSchedule, report_times: np.ndarray,
            *, progress: bool = False) -> ImpesResult:
        L, n = self.n_layers, self.n_r
        fl, rp, rk = self.fl, self.rp, self.rk

        p = np.full((L, n), self.p_init)
        Sg = np.full((L, n), self.Sg_init)
        report_times = np.asarray(report_times, float)

        out_t, out_p, out_S, out_pbh, out_q, out_ql = [], [], [], [], [], []
        out_minj, out_mout, out_mip = [], [], []

        cum_inj = 0.0      # kg
        cum_out = 0.0      # kg CO2 across r_e
        t = 0.0
        dt = self.dt_init
        n_steps = 0
        n_rejects = 0
        n_clipped = 0        # steps where S_g had to be clipped into range
        n_dt_min = 0         # steps accepted at dt_min without meeting max_dS
        max_clip = 0.0
        max_brine_resid = 0.0
        max_bhp_spread = 0.0
        max_rate_err = 0.0
        n_steps_layer_closed = 0
        p_cell_max = self.p_init
        p_bh_max = self.p_init
        min_cfl_ratio = np.inf
        t0 = _time.perf_counter()

        # record initial state
        def _record(tt, pp, SS, pbh, qm, ql):
            out_t.append(tt); out_p.append(pp.copy()); out_S.append(SS.copy())
            out_pbh.append(pbh); out_q.append(qm); out_ql.append(np.array(ql))
            out_minj.append(cum_inj); out_mout.append(cum_out)
            # free-phase CO2 mass, with first-order pore- and fluid-volume
            # corrections consistent with the slightly-compressible closure
            dp = pp - self.p_init
            out_mip.append(float(np.sum(self.Vp * (1.0 + rk.c_r * dp) * SS
                                        * fl.rho_g * (1.0 + fl.c_g * dp))))

        _record(0.0, p, Sg, self.p_init, schedule.rate_at(0.0), np.zeros(L))

        ri = 0
        while ri < len(report_times) and report_times[ri] <= 0:
            ri += 1
        t_final = float(max(schedule.t_end, report_times[-1]))
        breaks = np.unique(np.concatenate([schedule.t_edges, report_times]))

        while t < t_final - 1e-9:
            # do not step across a schedule change or a report time
            nxt = breaks[breaks > t + 1e-9]
            t_target = float(nxt[0]) if nxt.size else t_final
            dt_try = min(dt, t_target - t, self.dt_max)
            accepted = False
            n_cuts = 0
            while not accepted:
                dt_try = max(dt_try, self.dt_min)
                res = self._step(p, Sg, dt_try, t, schedule)
                if res is None or res["max_dS"] > self.max_dS \
                        or res["out_of_range"] > 1e-3 \
                        or res["overshoot"] > self.max_overshoot:
                    if dt_try <= self.dt_min * 1.0000001:
                        if res is None:
                            raise SimulationFailure(
                                "linear solve failed at the minimum time step "
                                f"(t={t:.3g} s)")
                        # cannot reduce further: accept, but count it
                        accepted = True
                        n_dt_min += 1
                    else:
                        dt_try *= 0.5
                        n_rejects += 1
                        n_cuts += 1
                        if n_cuts > 60:
                            raise SimulationFailure(
                                "time-step control failed to converge at "
                                f"t={t:.4g} s after {n_cuts} halvings")
                        continue
                else:
                    accepted = True
            Sg_raw = res["Sg"]
            viol = float(max(np.max(-Sg_raw), np.max(Sg_raw - (1.0 - rp.S_ar)), 0.0))
            if viol > 1e-12:
                n_clipped += 1
                max_clip = max(max_clip, viol)
            p, Sg = res["p"], np.clip(Sg_raw, 0.0, 1.0 - rp.S_ar)
            max_brine_resid = max(max_brine_resid, res["brine_residual"])
            # running maxima over *every* step, not only report times
            p_cell_max = max(p_cell_max, float(np.max(p)))
            if res["injecting"]:
                # the BHP is a physical well pressure only while the well is
                # open; during shut-in it is a zero-net-rate diagnostic
                p_bh_max = max(p_bh_max, res["p_bh"])
            max_bhp_spread = max(max_bhp_spread, res["bhp_spread_Pa"])
            max_rate_err = max(max_rate_err, res["rate_rel_err"])
            n_steps_layer_closed += int(res["n_closed_layers"] > 0)
            cum_inj += res["q_mass"] * dt_try
            cum_out += res["q_g_out"] * fl.rho_g * dt_try
            t += dt_try
            n_steps += 1
            if n_steps > self.max_steps:
                raise SimulationFailure(
                    f"exceeded max_steps={self.max_steps} at "
                    f"t={t/3.15576e7:.3f} yr of {t_final/3.15576e7:.3f} yr")

            # step-size controller: target ~85% of the allowed saturation
            # change, but never exceed the local CFL bound
            if res["max_dS"] > 1e-12:
                growth = float(np.clip(0.85 * self.max_dS / res["max_dS"],
                                       0.5, 1.6))
            else:
                growth = 1.6
            dt_next = dt_try * growth
            if self.enforce_cfl:
                dt_next = min(dt_next, res["dt_cfl"])
            dt = min(self.dt_max, max(self.dt_min, dt_next))
            min_cfl_ratio = min(min_cfl_ratio, res["dt_cfl"] / dt_try)

            while ri < len(report_times) and report_times[ri] <= t + 1e-6:
                _record(report_times[ri], p, Sg, res["p_bh"], res["q_mass"],
                        res["q_layer"])
                ri += 1
            if progress and n_steps % 500 == 0:
                print(f"  t={t/3.15576e7:8.3f} yr  dt={dt/86400:8.3f} d")

        while ri < len(report_times):
            _record(report_times[ri], p, Sg, res["p_bh"], res["q_mass"],
                    res["q_layer"])
            ri += 1

        wall = _time.perf_counter() - t0
        return ImpesResult(
            t=np.array(out_t), p=np.array(out_p), Sg=np.array(out_S),
            p_bh=np.array(out_pbh), q_mass=np.array(out_q),
            q_layer=np.array(out_ql),
            mass_injected=np.array(out_minj), mass_out=np.array(out_mout),
            mass_in_place=np.array(out_mip),
            diagnostics={"n_steps": n_steps, "n_rejects": n_rejects,
                         "n_clipped": n_clipped, "max_clip": max_clip,
                         "n_dt_min": n_dt_min,
                         "max_brine_residual": max_brine_resid,
                         "p_cell_max_Pa": p_cell_max,
                         "p_bh_max_Pa": p_bh_max,
                         "max_bhp_spread_Pa": max_bhp_spread,
                         "max_rate_rel_err": max_rate_err,
                         "n_steps_layer_closed": n_steps_layer_closed,
                         "enforce_cfl": self.enforce_cfl,
                         "min_cfl_ratio": float(min_cfl_ratio),
                         "wall_time_s": wall, "dt_final_s": dt,
                         "outer_bc": self.outer_bc,
                         "pore_volume_m3": self._pore_volume_total()},
        )

    # ----------------------------------------------------------------- step
    def _banded(self, Tsum_l, C_l, Tout_l, J_l):
        n = self.n_r
        main = np.zeros(n)
        main[:-1] += Tsum_l
        main[1:] += Tsum_l
        main[-1] += Tout_l
        main += C_l
        main[0] += J_l
        ab = np.zeros((3, n))
        ab[0, 1:] = -Tsum_l
        ab[1, :] = main
        ab[2, :-1] = -Tsum_l
        return ab

    def _flux_residual(self, Tsum_l, Tout_l, p_l):
        """``sum_j T_ij (p_j - p_i) + T_out (p_init - p_i)`` at ``p_l``."""
        r = np.zeros(self.n_r)
        f = Tsum_l * (p_l[1:] - p_l[:-1])
        r[:-1] += f
        r[1:] -= f
        r[-1] += Tout_l * (self.p_init - p_l[-1])
        return r

    def _solve_pressure(self, Tsum, C, p_old, rhs_extra, J, q_vol_tot,
                        T_out_sum):
        """Implicit pressure solve with the well coupled through ONE BHP.

        Each layer ``l`` satisfies
        ``C (p - p^n) = F(p) + e_0 J_l (p_bh - p_0)``, where ``F`` is the
        linear inter-cell/boundary flux operator and ``J_l = WI_l
        lambda_t,l0`` the connection factor (zero for a closed completion;
        well equation of ``3-IMPES.pdf`` p.16).  ``p_bh`` is the single extra
        unknown, fixed by ``sum_l J_l (p_bh - p_l0) = Q``.

        The system is solved for the **increment** ``dp = p - p^n`` (better
        conditioned than solving for ~1e7 Pa absolute pressures when the
        storage term is small) and ``p_bh`` is eliminated exactly by
        superposition: ``dp_l = x_l + y_l (p_bh - p_init)`` with
        ``A_l x_l = r_l`` and ``A_l y_l = J_l e_0``.
        """
        L, n = self.n_layers, self.n_r
        x = np.empty((L, n))
        y = np.zeros((L, n))
        for l in range(L):
            ab = self._banded(Tsum[l], C[l], T_out_sum[l], J[l])
            r = self._flux_residual(Tsum[l], T_out_sum[l], p_old[l]) + rhs_extra[l]
            r[0] += J[l] * (self.p_init - p_old[l, 0])
            try:
                x[l] = solve_banded((1, 1), ab, r)
                if J[l] > 0:
                    e0 = np.zeros(n); e0[0] = J[l]
                    y[l] = solve_banded((1, 1), ab, e0)
            except Exception:
                return None
        denom = float(np.sum(J * (1.0 - y[:, 0])))
        if denom <= 0.0:
            return None
        d_bh = (q_vol_tot + float(np.sum(J * (x[:, 0] + p_old[:, 0]
                                              - self.p_init)))) / denom
        p_new = p_old + x + y * d_bh
        p_bh = self.p_init + d_bh
        q_l = J * (p_bh - p_new[:, 0])
        return p_new, p_bh, q_l

    def _step(self, p, Sg, dt, t, schedule):
        """One IMPES step.  Returns ``None`` if the linear solve fails.

        Primary variables: pressure ``p`` and CO2 saturation.  Every
        saturation-dependent coefficient is evaluated at the old time level,
        the IMPES approximation of ``3-IMPES.pdf`` p.13.
        """
        L, n = self.n_layers, self.n_r
        fl, rp, rk = self.fl, self.rp, self.rk

        lam_a = rp.lam_a(Sg, fl)
        lam_g = rp.lam_g(Sg, fl)
        lam_t = lam_a + lam_g
        # upstream weighting of the phase mobilities (3-IMPES.pdf p.8)
        Ta = self.Tgeom * self._upstream(lam_a, p)                 # (L, n-1)
        rho_ratio = 1.0 + fl.c_g * (p - self.p_init)               # rho_g/rho_g0
        Tg_vol = self.Tgeom * self._upstream(lam_g, p)             # volumetric
        Tg = self.Tgeom * self._upstream(lam_g * rho_ratio, p)     # mass-consistent
        Tsum = Ta + Tg_vol

        S_a = 1.0 - Sg
        c_t = rk.c_r + S_a * fl.c_a + Sg * fl.c_g
        C = self.Vp * c_t / dt

        # --- outer boundary ----------------------------------------------
        if self.outer_bc == "constant_pressure":
            out_a = p[:, -1] >= self.p_init
            out_g = out_a
            lam_a_out = np.where(out_a, lam_a[:, -1], rp.lam_a(np.zeros(L), fl))
            lam_g_out = np.where(out_g, lam_g[:, -1], 0.0)
            T_out_a = self.Tout_geom * lam_a_out
            T_out_g = self.Tout_geom * lam_g_out
        else:
            T_out_a = np.zeros(L); T_out_g = np.zeros(L)
        T_out_sum = T_out_a + T_out_g

        rhs_extra = np.zeros((L, n))

        # --- well: one rate-controlled injector, ONE bottom-hole pressure ---
        q_mass = schedule.rate_at(t + 0.5 * dt)
        q_vol_tot = q_mass / fl.rho_g                 # reference m^3/s
        radial = isinstance(self.grids[0], RadialGrid)
        n_closed = 0
        if radial:
            J_all = self.WI_geom * lam_t[:, 0]
            if q_vol_tot > 0.0:
                # Injector-only completions: a layer whose well-block gas
                # pressure exceeds the common BHP would *produce* into the
                # wellbore (crossflow).  An injector cannot do that, so such a
                # completion is closed and the solve repeated (active set).
                active = J_all > 0
                while True:
                    J = np.where(active, J_all, 0.0)
                    sol = self._solve_pressure(Tsum, C, p, rhs_extra, J,
                                               q_vol_tot, T_out_sum)
                    if sol is None:
                        return None
                    p_new, p_bh, q_layer = sol
                    neg = active & (q_layer < -1e-12 * q_vol_tot)
                    if not neg.any():
                        break
                    active &= ~neg
                    n_closed = int(np.sum(~active))
                q_layer = np.maximum(q_layer, 0.0)
                bhp_implied = np.where(active, p_new[:, 0]
                                       + q_layer / np.maximum(J_all, 1e-300),
                                       np.nan)
            else:
                # SHUT-IN: every completion is closed; no fluid enters or leaves
                # any layer through the well (no wellbore crossflow).
                p_new = self._solve_closed(Tsum, C, p, rhs_extra, T_out_sum)
                if p_new is None:
                    return None
                q_layer = np.zeros(L)
                # diagnostic only: the pressure a zero-net-rate open wellbore
                # would take, sum_l J_l (p_ws - p_l0) = 0
                p_bh = float(np.sum(J_all * p_new[:, 0])
                             / max(float(np.sum(J_all)), 1e-300))
                bhp_implied = np.full(L, np.nan)
        else:
            # 1-D Cartesian benchmark: prescribed rate into the first cell
            q_layer = np.full(L, q_vol_tot / L)
            rhs2 = rhs_extra.copy()
            rhs2[:, 0] += q_layer
            p_new = self._solve_closed(Tsum, C, p, rhs2, T_out_sum)
            if p_new is None:
                return None
            p_bh = float(p_new[0, 0])
            bhp_implied = np.array([p_bh])
        if not np.all(np.isfinite(p_new)):
            return None

        # --- explicit saturation from the *CO2* equation -------------------
        # 3-IMPES.pdf p.18 advances the saturation from the displaced-phase
        # (oil) equation.  We advance it from the **injected-phase** equation
        # instead, so that free-phase CO2 mass -- the quantity of interest and
        # the ML target -- is conserved to the accuracy of the linear solve.
        # The brine equation then carries the IMPES splitting error, which is
        # measured and reported (``brine_residual``).
        #
        # The accumulation term is written in **conservative** (telescoping)
        # form  Vp0 [Sg^{n+1} B(p^{n+1}) - Sg^n B(p^n)] / dt = flux,
        # B(p) = (1 + c_r dp)(1 + c_g dp), with the gas mobility scaled by the
        # gas density ratio, so that the conserved quantity is CO2 mass.
        dpn = p_new[:, 1:] - p_new[:, :-1]
        B_old = ((1.0 + rk.c_r * (p - self.p_init))
                 * (1.0 + fl.c_g * (p - self.p_init)))
        B_new = ((1.0 + rk.c_r * (p_new - self.p_init))
                 * (1.0 + fl.c_g * (p_new - self.p_init)))
        flux_g = np.zeros((L, n))
        flux_g[:, :-1] += Tg * dpn
        flux_g[:, 1:] += Tg * (-dpn)
        g_out_drive = self.p_init - p_new[:, -1]
        flux_g[:, -1] += T_out_g * rho_ratio[:, -1] * g_out_drive
        flux_g[:, 0] += q_layer                      # CO2 source (reference vol)
        Sg_new = (Sg * B_old + dt * flux_g / self.Vp) / B_new

        # brine-equation residual (the IMPES splitting error)
        flux_a = np.zeros((L, n))
        flux_a[:, :-1] += Ta * dpn
        flux_a[:, 1:] += Ta * (-dpn)
        flux_a[:, -1] += T_out_a * (self.p_init - p_new[:, -1])
        Ba_old = ((1.0 + rk.c_r * (p - self.p_init))
                  * (1.0 + fl.c_a * (p - self.p_init)))
        Ba_new = ((1.0 + rk.c_r * (p_new - self.p_init))
                  * (1.0 + fl.c_a * (p_new - self.p_init)))
        acc_a = self.Vp * ((1.0 - Sg_new) * Ba_new - (1.0 - Sg) * Ba_old) / dt
        resid_a = flux_a - acc_a
        scale = float(np.sum(np.abs(flux_a)) + np.sum(np.abs(acc_a))) + 1e-30
        brine_residual = float(np.sum(np.abs(resid_a)) / scale)

        # gas outflow across the outer boundary (positive = leaving domain),
        # expressed as a reference-density volumetric rate
        q_g_out = float(np.sum(-T_out_g * rho_ratio[:, -1] * g_out_drive))

        # --- diagnostics ---------------------------------------------------
        max_dS = float(np.max(np.abs(Sg_new - Sg)))
        lo, hi = -1e-8, 1.0 - rp.S_ar + 1e-8
        out_of_range = float(max(np.max(lo - Sg_new), np.max(Sg_new - hi), 0.0))

        # discrete local maximum principle (instability detector; see run())
        Sg_pad = np.pad(Sg, ((0, 0), (1, 1)), mode="edge")
        nbr_max = np.maximum.reduce([Sg_pad[:, :-2], Sg_pad[:, 1:-1],
                                     Sg_pad[:, 2:]])
        nbr_max[:, 0] = 1.0 - rp.S_ar          # the well cell may fill freely
        nbr_min = np.minimum.reduce([Sg_pad[:, :-2], Sg_pad[:, 1:-1],
                                     Sg_pad[:, 2:]])
        # two-sided: an explicit upstream update may create neither a new
        # local maximum nor a new local minimum (odd-even oscillation)
        overshoot = max(float(np.max(Sg_new - nbr_max - 1e-6)),
                        float(np.max(nbr_min - Sg_new - 1e-6)), 0.0)

        # CFL estimate for the next step (face throughput of both phases)
        F = np.abs(Ta * dpn) + np.abs(Tg_vol * dpn)        # (L, n-1)
        thr = np.zeros((L, n))
        thr[:, :-1] = np.maximum(thr[:, :-1], F)
        thr[:, 1:] = np.maximum(thr[:, 1:], F)
        thr[:, -1] = np.maximum(thr[:, -1], np.abs(T_out_sum * (p_new[:, -1]
                                                                - self.p_init)))
        thr[:, 0] = np.maximum(thr[:, 0], np.abs(q_layer))
        dfg_loc = np.maximum(self._dfg_local(Sg), 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            dt_cell = self.Vp / np.maximum(thr * dfg_loc, 1e-30)
        dt_cfl = float(self.cfl * np.min(dt_cell))

        act = np.isfinite(bhp_implied)
        bhp_spread = (float(np.max(bhp_implied[act]) - np.min(bhp_implied[act]))
                      if act.sum() > 1 else 0.0)
        rate_err = (abs(float(np.sum(q_layer)) - q_vol_tot) / q_vol_tot
                    if q_vol_tot > 0 else float(np.sum(np.abs(q_layer))))

        return {"p": p_new, "Sg": Sg_new, "max_dS": max_dS,
                "out_of_range": out_of_range, "dt_cfl": dt_cfl,
                "q_mass": q_mass, "q_g_out": q_g_out, "p_bh": float(p_bh),
                "injecting": bool(q_vol_tot > 0.0),
                "q_layer": q_layer, "n_closed_layers": n_closed,
                "bhp_spread_Pa": bhp_spread, "rate_rel_err": rate_err,
                "brine_residual": brine_residual, "overshoot": overshoot}

    def _solve_closed(self, Tsum, C, p_old, rhs_extra, T_out_sum):
        """Pressure solve with no well connection (shut-in or Cartesian),
        in increment form."""
        L, n = self.n_layers, self.n_r
        out = np.empty((L, n))
        for l in range(L):
            ab = self._banded(Tsum[l], C[l], T_out_sum[l], 0.0)
            r = self._flux_residual(Tsum[l], T_out_sum[l], p_old[l]) + rhs_extra[l]
            try:
                out[l] = p_old[l] + solve_banded((1, 1), ab, r)
            except Exception:
                return None
        return out
