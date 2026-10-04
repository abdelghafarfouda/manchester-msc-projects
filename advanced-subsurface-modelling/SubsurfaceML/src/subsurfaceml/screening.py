"""Verification-gated schedule screening.

The rule this module enforces
-----------------------------
**No schedule is recommended unless the simulator has run it and it met the
stated limits.**  Surrogates and the analytical ROM only *propose*
schedules; every proposal is a "candidate" until it is simulated.
:class:`Recommendation` refuses to hold a recommended schedule that is not
simulator-verified (its constructor raises), and the tests check the rule on
the failure paths as well as the success path.

Outcomes (``Recommendation.status``)
------------------------------------
``VERIFIED_FEASIBLE``
    a schedule was simulated and met both stated limits; it is the
    recommendation.
``NO_FEASIBLE_SCHEDULE_FOUND``
    every schedule simulated within the budget violated a limit (or the
    simulation failed); there is **no recommendation**.
``NO_CANDIDATE_PREDICTED_FEASIBLE``
    no candidate was predicted to be within the limits once the interval's
    upper edge was used, and no simulator fallback was requested; there is
    **no recommendation**.

Reliability flags (``Recommendation.flags``) record *why* the path taken
differs from the plain one, e.g. ``surrogate_out_of_domain`` (the reservoir
is outside the training population, so the surrogate is not used to rank
schedules), ``upper_bound_excludes_all`` (point predictions were feasible
but the interval upper edges were not -- the uncertainty is too large for
the surrogate to decide), ``repaired_by_simulator`` (the proposed schedule
violated and was scaled down along its own shape using simulator runs), and
``simulator_fallback``.

The limits are inputs.  The 9 MPa build-up limit of the study is a stated
modelling assumption, not a fracture or caprock criterion; meeting it says
only that the *modelled* bottom-hole pressure stays below that number.

Search with the simulator (:func:`proportional_search`)
-------------------------------------------------------
Along a fixed schedule shape ``s`` (rates ``c * s``) the build-up is close to
proportional to ``c`` (exactly so in the single-phase limit verified by V12)
and the plume radius close to proportional to ``sqrt(c)``.  The search
therefore updates ``c <- c * safety * min(dp_lim/dp, (r_lim/r)^2)`` after
each simulation, keeps the best *verified* schedule, and stops when the
budget is spent or the best verified schedule is within ``tol`` of the
binding limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import MPA, YEAR

VERIFIED = "VERIFIED_FEASIBLE"
NO_FEASIBLE = "NO_FEASIBLE_SCHEDULE_FOUND"
NO_CANDIDATE = "NO_CANDIDATE_PREDICTED_FEASIBLE"


@dataclass
class SimCheck:
    """The simulator's verdict on one schedule."""
    rates_kg_s: list
    status: str                      # "ok" or "failed"
    dp_MPa: float = np.nan
    r95_m: float = np.nan
    mass_Mt: float = np.nan
    feasible: bool = False
    label: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Recommendation:
    realisation_id: int
    method: str
    status: str
    recommended_rates_kg_s: list | None = None
    verified: SimCheck | None = None
    checks: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    predicted: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if self.status not in (VERIFIED, NO_FEASIBLE, NO_CANDIDATE):
            raise ValueError(f"unknown status {self.status!r}")
        if self.recommended_rates_kg_s is not None:
            if (self.verified is None or self.verified.status != "ok"
                    or not self.verified.feasible):
                raise ValueError("a schedule may only be recommended after the "
                                 "simulator has verified that it meets the limits")
            if not np.allclose(self.verified.rates_kg_s, self.recommended_rates_kg_s):
                raise ValueError("the recommended schedule is not the verified one")
            if self.status != VERIFIED:
                raise ValueError("a recommendation requires status VERIFIED_FEASIBLE")
        elif self.status == VERIFIED:
            raise ValueError("status VERIFIED_FEASIBLE requires a recommendation")

    @property
    def n_simulations(self) -> int:
        return len(self.checks)

    @property
    def mass_Mt(self) -> float:
        return self.verified.mass_Mt if self.recommended_rates_kg_s is not None else np.nan

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("checks", "verified")}
        d["verified"] = None if self.verified is None else self.verified.as_dict()
        d["checks"] = [c.as_dict() for c in self.checks]
        d["n_simulations"] = self.n_simulations
        d["mass_Mt"] = self.mass_Mt
        return d


# --------------------------------------------------------------------------
def make_simulator(cfg, realisation):
    """``simulate(rates, label) -> SimCheck`` using the validated simulator."""
    from .scenarios import make_schedule, run_scenario
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    r_lim = cfg.optim.r_plume_limit_m

    def simulate(rates, label=""):
        rates = np.asarray(rates, float)
        out = run_scenario(cfg, realisation,
                           make_schedule(cfg, realisation, rates, schedule_id=-2),
                           want_series=False)
        if out["status"] != "ok":
            return SimCheck(rates.tolist(), "failed", label=label)
        w = out["row"]
        dp, r95 = w["dp_bh_max_Pa"] / MPA, w["r_plume_m95_m"]
        return SimCheck(rates.tolist(), "ok", dp, r95, w["mass_retained_kg"] / 1e9,
                        bool(dp <= dp_lim and r95 <= r_lim), label)
    return simulate


def proportional_search(shape, c0, simulate, budget, dp_lim, r_lim, *,
                        safety=0.98, tol=0.02, c_max=None, label="search"):
    """Scale a schedule shape with simulator runs; returns ``(best, checks)``
    where ``best`` is the verified-feasible check with the largest mass (or
    ``None``)."""
    shape = np.asarray(shape, float)
    c = float(c0)
    checks, best = [], None
    for i in range(int(budget)):
        if c_max is not None:
            c = min(c, c_max)
        chk = simulate(c * shape, f"{label}_{i}")
        checks.append(chk)
        if chk.status != "ok":
            c *= 0.5                                   # back off after a failure
            continue
        if chk.feasible and (best is None or chk.mass_Mt > best.mass_Mt):
            best = chk
        ratio = min(dp_lim / max(chk.dp_MPa, 1e-12),
                    (r_lim / max(chk.r95_m, 1e-12)) ** 2)
        if chk.feasible and ratio <= 1.0 + tol:
            break                                      # within tol of the binding limit
        c *= safety * ratio
    return best, checks


def planned_mass_Mt(cfg, rates) -> np.ndarray:
    dt = cfg.schedule.t_inject_years * YEAR / cfg.schedule.n_periods
    return np.atleast_2d(rates).sum(axis=1) * dt / 1e9


# --------------------------------------------------------------------------
def recommend_constant_simulator(cfg, r, simulate, budget, *, c0=None) -> Recommendation:
    """Simulator only: constant rate, started at the reference rate."""
    from .scenarios import reference_rate
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    n = cfg.schedule.n_periods
    q0 = reference_rate(cfg, r) if c0 is None else c0
    best, checks = proportional_search(np.ones(n), q0, simulate, budget, dp_lim,
                                       cfg.optim.r_plume_limit_m,
                                       c_max=cfg.schedule.q_max_kg_s,
                                       label="constant")
    return _finish(r, "simulator_constant", best, checks, [], {})


def rom_max_constant_rate(cfg, r, *, n_bisect=40) -> float:
    """Largest constant rate whose ROM build-up is at the pressure limit."""
    from . import rom
    dp_lim = (cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa) * MPA
    n = cfg.schedule.n_periods
    lo, hi = 1e-6, cfg.schedule.q_max_kg_s
    if rom.bhp_buildup(cfg, r, np.full(n, hi)) <= dp_lim:
        return hi
    for _ in range(n_bisect):
        mid = np.sqrt(lo * hi)
        if rom.bhp_buildup(cfg, r, np.full(n, mid)) <= dp_lim:
            lo = mid
        else:
            hi = mid
    return lo


def recommend_constant_rom(cfg, r, simulate, budget) -> Recommendation:
    """ROM-guided: start the simulator search at the ROM's limiting constant
    rate."""
    q0 = rom_max_constant_rate(cfg, r)
    rec = recommend_constant_simulator(cfg, r, simulate, budget, c0=q0)
    rec.method = "rom_constant"
    rec.predicted = {"rom_constant_rate_kg_s": q0}
    return rec


def screen_candidates(cfg, r, predictor, candidates) -> dict:
    """Surrogate predictions for candidate schedules.  ``predictor(R)`` must
    return a dict with ``dp`` / ``dp_hi`` [MPa] and ``r95`` / ``r95_hi`` [m]."""
    pr = predictor(candidates)
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    r_lim = cfg.optim.r_plume_limit_m
    feas_hi = (pr["dp_hi"] <= dp_lim) & (pr["r95_hi"] <= r_lim)
    feas_pt = (pr["dp"] <= dp_lim) & (pr["r95"] <= r_lim)
    return {**pr, "feasible_upper": feas_hi, "feasible_point": feas_pt,
            "mass_Mt": planned_mass_Mt(cfg, candidates)}


def recommend_surrogate(cfg, r, simulate, budget, predictor, candidates, *,
                        in_domain=True, repair=True, n_verify=1,
                        fallback="rom_constant", method="surrogate_verified"
                        ) -> Recommendation:
    """Surrogate proposes, simulator decides.

    1. If the reservoir is outside the training domain the surrogate is not
       used (flag ``surrogate_out_of_domain``) and the ``fallback`` simulator
       search runs instead.
    2. Otherwise candidates are screened with the interval's *upper* edge; the
       ``n_verify`` best predicted-feasible candidates (by exact planned mass)
       are simulated.
    3. With ``repair``, the remaining budget scales the best proposal along
       its own shape with :func:`proportional_search`.
    4. Nothing predicted feasible: flag ``upper_bound_excludes_all`` if the
       point predictions were feasible, then the ``fallback`` search runs, or
       -- with ``fallback=None`` -- the outcome is ``NO_CANDIDATE_PREDICTED_FEASIBLE``.
    """
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    r_lim = cfg.optim.r_plume_limit_m
    flags = []
    if not in_domain:
        flags.append("surrogate_out_of_domain")
        return _fallback(cfg, r, simulate, budget, fallback, flags, method, {})
    sc = screen_candidates(cfg, r, predictor, candidates)
    feas = sc["feasible_upper"]
    pred_info = {"n_candidates": int(len(candidates)),
                 "n_predicted_feasible_upper": int(feas.sum()),
                 "n_predicted_feasible_point": int(sc["feasible_point"].sum())}
    if not feas.any():
        if sc["feasible_point"].any():
            flags.append("upper_bound_excludes_all")
        if fallback is None:
            return Recommendation(r.realisation_id, method, NO_CANDIDATE, flags=flags,
                                  predicted=pred_info,
                                  notes="no candidate predicted within the limits")
        return _fallback(cfg, r, simulate, budget, fallback, flags, method, pred_info)
    order = np.argsort(-np.where(feas, sc["mass_Mt"], -np.inf))
    top = [i for i in order[:n_verify] if feas[i]]
    pred_info["proposal"] = {"rates_kg_s": candidates[top[0]].tolist(),
                             "pred_dp_MPa": float(sc["dp"][top[0]]),
                             "pred_dp_upper_MPa": float(sc["dp_hi"][top[0]]),
                             "pred_r95_m": float(sc["r95"][top[0]]),
                             "pred_r95_upper_m": float(sc["r95_hi"][top[0]])}
    checks, best = [], None
    for j, i in enumerate(top[:budget]):
        chk = simulate(candidates[i], f"proposal_{j}")
        checks.append(chk)
        if chk.status == "ok" and chk.feasible and (best is None or chk.mass_Mt > best.mass_Mt):
            best = chk
    left = budget - len(checks)
    if repair and left > 0:
        first = checks[0]
        shape = np.asarray(first.rates_kg_s, float)
        if first.status == "ok":
            ratio = min(dp_lim / max(first.dp_MPa, 1e-12),
                        (r_lim / max(first.r95_m, 1e-12)) ** 2)
            needs = (not first.feasible) or ratio > 1.0 + 0.02
        else:
            ratio, needs = 0.5, True
        if needs:
            b2, ch2 = proportional_search(shape, 0.98 * ratio, simulate, left,
                                          dp_lim, r_lim, label="repair",
                                          c_max=cfg.schedule.q_max_kg_s / max(shape.max(), 1e-12))
            checks += ch2
            if b2 is not None and (best is None or b2.mass_Mt > best.mass_Mt):
                best = b2
                flags.append("repaired_by_simulator" if not first.feasible
                             else "scaled_up_by_simulator")
    return _finish(r, method, best, checks, flags, pred_info)


def _fallback(cfg, r, simulate, budget, fallback, flags, method, pred_info):
    flags = flags + ["simulator_fallback"]
    if fallback == "rom_constant":
        rec = recommend_constant_rom(cfg, r, simulate, budget)
    else:
        rec = recommend_constant_simulator(cfg, r, simulate, budget)
    return _finish(r, method, rec.verified, rec.checks, flags,
                   {**pred_info, **rec.predicted})


def _finish(r, method, best, checks, flags, pred_info) -> Recommendation:
    if best is not None:
        return Recommendation(r.realisation_id, method, VERIFIED,
                              recommended_rates_kg_s=list(best.rates_kg_s),
                              verified=best, checks=checks, flags=flags,
                              predicted=pred_info)
    return Recommendation(r.realisation_id, method, NO_FEASIBLE, checks=checks,
                          flags=flags, predicted=pred_info,
                          notes="no simulated schedule met the stated limits "
                                "within the simulation budget")
