"""Surrogate screening of injection schedules, checked by re-simulation.

Question
--------
For a reservoir the surrogate has never seen, which piecewise-constant
schedule ``q_1..q_N`` puts the most CO2 into the sealed compartment while the
modelled bottom-hole pressure buildup and plume radius stay below **stated**
limits?  Both limits are configuration inputs; meeting them says nothing
about fracture, caprock or leakage behaviour, none of which is modelled.

Method -- Monte Carlo screening (``5-Uncertainty.pdf`` p.44-45)
------------------------------------------------------------------
1. Draw ``n_candidates`` random schedules from the same level x shape space
   the training data were drawn from (no extrapolation in schedule space).
2. Build their inputs through :mod:`features` (the one feature path) and
   predict the two constraint quantities with the surrogates.  With
   ``use_error_band`` the **upper edge of the empirical error band** is
   compared with the limit, so a candidate is kept only if it is predicted
   feasible even at the pessimistic end of the surrogate's measured error.
3. Rank the feasible candidates by the planned injected mass, computed
   **exactly** from the schedule (under the sealed boundary it equals the
   retained mass by CO2 mass balance -- verified in the pipeline -- so no
   surrogate is needed or used for it).
4. **Re-simulate** the shortlisted schedules and the baseline with the
   simulator and report what the simulator says, including violations and
   failed runs.

Baseline: the best **constant-rate** schedule found by the same screening
(same limits, same surrogates, same error band), i.e. the only difference is
that the rate may vary between periods.

An earlier version added a Nelder-Mead refinement; that optimiser is not in
the supplied material and was removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .features import features_for_schedules
from .scenarios import Realisation, make_schedule, reference_rate, run_scenario
from .units import MPA, YEAR


@dataclass
class SurrogateBundle:
    """The two constraint surrogates and their error bands."""
    features: list
    pressure: object          # FittedSurrogate for dp_bh_max_MPa
    plume: object             # FittedSurrogate for r_plume_m95_m
    band_pressure: object = None
    band_plume: object = None


def exact_planned_mass(cfg: Config, rates_kg_s) -> float:
    """Planned injected mass [kg] -- exact, from the schedule alone."""
    dt = cfg.schedule.t_inject_years * YEAR / cfg.schedule.n_periods
    return float(np.sum(np.asarray(rates_kg_s, float)) * dt)


def sample_candidates(cfg: Config, r: Realisation, n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Random candidate rate vectors ``(n, n_periods)`` [kg/s] from the same
    level x shape design as :func:`scenarios.sample_schedules`."""
    from .scenarios import shape_vector
    n_p = cfg.schedule.n_periods
    q_ref = reference_rate(cfg, r)
    lo, hi = cfg.schedule.q_mult_min, cfg.schedule.q_mult_max
    m = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    tilt = rng.uniform(-1.0, 1.0, size=n)
    curve = rng.uniform(-0.5, 0.5, size=n)
    sh = np.stack([shape_vector(n_p, tilt[i], curve[i]) for i in range(n)])
    return np.clip(m[:, None] * q_ref * sh, cfg.schedule.q_min_kg_s,
                   cfg.schedule.q_max_kg_s)


def _screen(cfg, r, bundle, rates, use_band):
    X = features_for_schedules(cfg, r, rates, bundle.features)
    out = {}
    for key, surr, band in (("dp", bundle.pressure, bundle.band_pressure),
                            ("rp", bundle.plume, bundle.band_plume)):
        if use_band and band is not None:
            lo, p, hi = band.predict_interval(X)
        else:
            p = hi = np.asarray(surr.predict(X), float)
        out[key], out[key + "_hi"] = np.asarray(p), np.asarray(hi)
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    out["feasible"] = (out["dp_hi"] <= dp_lim) & (out["rp_hi"] <= cfg.optim.r_plume_limit_m)
    dt = cfg.schedule.t_inject_years * YEAR / cfg.schedule.n_periods
    out["mass_Mt"] = rates.sum(axis=1) * dt / 1e9
    return out


def optimise_schedule(cfg: Config, r: Realisation, bundle: SurrogateBundle,
                      *, rng=None) -> dict:
    """Screen candidates for one realisation.  Returns the shortlist and the
    baseline as *surrogate predictions*; :func:`resimulate` checks them."""
    rng = rng or np.random.default_rng(cfg.ml.random_state + r.realisation_id)
    o = cfg.optim
    cand = sample_candidates(cfg, r, o.n_candidates, rng)
    sc = _screen(cfg, r, bundle, cand, o.use_error_band)
    order = np.argsort(-np.where(sc["feasible"], sc["mass_Mt"], -np.inf))
    top = [i for i in order[:o.n_shortlist] if sc["feasible"][i]]

    q_ref = reference_rate(cfg, r)
    levels = np.exp(np.linspace(np.log(cfg.schedule.q_mult_min),
                                np.log(cfg.schedule.q_mult_max), 400)) * q_ref
    levels = np.clip(levels, cfg.schedule.q_min_kg_s, cfg.schedule.q_max_kg_s)
    flat = np.repeat(levels[:, None], cfg.schedule.n_periods, axis=1)
    sf = _screen(cfg, r, bundle, flat, o.use_error_band)
    base_idx = (int(np.argmax(np.where(sf["feasible"], sf["mass_Mt"], -np.inf)))
                if sf["feasible"].any() else None)

    def _entry(R, S, i):
        return {"rates_kg_s": R[i].tolist(), "pred_mass_Mt": float(S["mass_Mt"][i]),
                "pred_dp_MPa": float(S["dp"][i]), "pred_dp_upper_MPa": float(S["dp_hi"][i]),
                "pred_rplume_m": float(S["rp"][i]),
                "pred_rplume_upper_m": float(S["rp_hi"][i])}

    return {"realisation_id": r.realisation_id,
            "p_limit_MPa": o.p_limit_MPa,
            "dp_limit_MPa": o.p_limit_MPa - cfg.solver.p_init_MPa,
            "r_plume_limit_m": o.r_plume_limit_m,
            "use_error_band": bool(o.use_error_band),
            "n_candidates": int(o.n_candidates),
            "n_feasible": int(sc["feasible"].sum()),
            "feasible_fraction": float(sc["feasible"].mean()),
            "shortlist": [_entry(cand, sc, i) for i in top],
            "baseline": None if base_idx is None else _entry(flat, sf, base_idx)}
# --------------------------------------------------------------------------
def resimulate(cfg: Config, r: Realisation, rates_kg_s, label: str) -> dict:
    """Run the validated simulator on a proposed schedule and report the
    truth, including whether the stated limits are actually met."""
    sched = make_schedule(cfg, r, rates_kg_s, schedule_id=-2)
    out = run_scenario(cfg, r, sched, want_series=False)
    if out["status"] != "ok":
        return {"label": label, "realisation_id": r.realisation_id,
                "status": "failed", "error": out.get("error", "")}
    w = out["row"]
    dp = w["dp_bh_max_Pa"] / MPA
    rp = w["r_plume_m95_m"]
    return {"label": label, "realisation_id": r.realisation_id, "status": "ok",
            "rates_kg_s": list(map(float, rates_kg_s)),
            "sim_mass_retained_Mt": w["mass_retained_kg"] / 1e9,
            "sim_mass_injected_Mt": w["mass_injected_kg"] / 1e9,
            "sim_mass_lost_Mt": w["mass_lost_kg"] / 1e9,
            "sim_dp_bh_max_MPa": dp,
            "sim_p_bh_max_MPa": w["p_bh_max_Pa"] / MPA,
            "sim_r_plume_m95_m": rp,
            "dp_limit_MPa": cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa,
            "r_plume_limit_m": cfg.optim.r_plume_limit_m,
            "violates_pressure": bool(dp > cfg.optim.p_limit_MPa
                                      - cfg.solver.p_init_MPa),
            "violates_plume": bool(rp > cfg.optim.r_plume_limit_m),
            "sim_wall_time_s": w["wall_time_s"]}
