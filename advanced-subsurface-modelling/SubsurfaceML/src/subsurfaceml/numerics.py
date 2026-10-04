"""Discretisation error of the *dataset* targets, measured case by case.

The verification suite (``validation.py``, V7) measures grid, well-block and
time-step convergence on one verification case.  That does not say whether
the discretisation error of the cases the surrogates are trained on is small
compared with the surrogate error, or whether it can flip the pressure-limit
label of a case near the stated limit.  This module answers that directly:
a stratified sample of dataset scenarios is re-simulated with each
discretisation parameter refined in turn, and with all of them refined
together, and the change in each machine-learning target is reported.

Levels (``LEVELS``)
-------------------
``production``  the configuration used to generate the dataset
``n_r_x2``      twice the radial cells
``r_near_half`` half the well-block radius
``time_fine``   half ``max_dS`` and a 30-day maximum step
``fine``        all three refinements together
``finer``       twice ``fine`` again (a subset only; it is expensive)

The ``fine``-minus-``production`` difference is the estimate of the
discretisation error carried by the dataset; ``finer``-minus-``fine`` shows
whether that estimate is itself converged.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .scenarios import make_schedule, run_scenario, sample_realisations
from .units import MPA

TARGETS = ("dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency")


def level_config(cfg, level: str):
    """A copy of ``cfg`` with the discretisation of ``level``."""
    c = copy.deepcopy(cfg)
    g, s = c.grid, c.solver
    if level == "production":
        pass
    elif level == "n_r_x2":
        g.n_r *= 2
    elif level == "r_near_half":
        g.r_near_m /= 2
    elif level == "time_fine":
        s.max_dS /= 2
        s.dt_max_days = min(s.dt_max_days, 30.0)
    elif level == "fine":
        g.n_r *= 2
        g.r_near_m /= 2
        s.max_dS /= 2
        s.dt_max_days = min(s.dt_max_days, 30.0)
    elif level == "finer":
        g.n_r *= 4
        g.r_near_m /= 4
        s.max_dS /= 4
        s.dt_max_days = min(s.dt_max_days, 15.0)
    else:
        raise ValueError(f"unknown level {level!r}")
    return c


LEVELS = ("production", "n_r_x2", "r_near_half", "time_fine", "fine")


def select_cases(df: pd.DataFrame, n_per_cell: int = 4, seed: int = 0,
                 always=()) -> pd.DataFrame:
    """Stratified sample: permeability tercile x schedule-intensity quartile,
    ``n_per_cell`` cases each, plus the scenarios listed in ``always``."""
    d = df.copy()
    d["_kb"] = pd.qcut(d["k_median_mD"], 3, labels=False)
    d["_qb"] = pd.qcut(d["q_mult_mean"], 4, labels=False)
    rng = np.random.default_rng(seed)
    pick = []
    for _, g in d.groupby(["_kb", "_qb"]):
        idx = rng.choice(g.index.to_numpy(), size=min(n_per_cell, len(g)),
                         replace=False)
        pick.extend(idx.tolist())
    extra = d.index[d["scenario_id"].isin(list(always))].tolist()
    keep = sorted(set(pick) | set(extra))
    return d.loc[keep].drop(columns=["_kb", "_qb"])


def _run(cfg, r, rates, level):
    c = level_config(cfg, level)
    out = run_scenario(c, r, make_schedule(c, r, rates, schedule_id=0),
                       want_series=False)
    if out["status"] != "ok":
        return {"level": level, "status": "failed", "error": out.get("error")}
    w = out["row"]
    return {"level": level, "status": "ok",
            "dp_bh_max_MPa": w["dp_bh_max_Pa"] / MPA,
            "r_plume_m95_m": w["r_plume_m95_m"],
            "sweep_efficiency": w["sweep_efficiency"],
            "mass_balance_error": w["mass_balance_error"],
            "n_steps": w["n_steps"], "wall_time_s": w["wall_time_s"]}


def refinement_study(cfg, cases: pd.DataFrame, levels=LEVELS,
                     n_jobs: int = -1) -> pd.DataFrame:
    """Re-simulate ``cases`` (rows of the scenario table) at each level."""
    reals = {r.realisation_id: r for r in sample_realisations(cfg)}
    qcols = [f"q{i + 1}_kg_s" for i in range(cfg.schedule.n_periods)]
    jobs = []
    for _, row in cases.iterrows():
        r = reals[int(row["realisation_id"])]
        rates = row[qcols].to_numpy(float)
        for lv in levels:
            jobs.append((row["scenario_id"], r, rates, lv))
    res = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_run)(cfg, r, q, lv) for _, r, q, lv in jobs)
    out = pd.DataFrame(res)
    out.insert(0, "scenario_id", [j[0] for j in jobs])
    return out


def summarise(runs: pd.DataFrame, cases: pd.DataFrame, dp_limit_MPa: float,
              reference: str = "fine") -> dict:
    """Relative and absolute changes of each target between levels, and the
    number of pressure-limit labels that change."""
    ok = runs[runs.status == "ok"]
    wide = {t: ok.pivot(index="scenario_id", columns="level", values=t)
            for t in TARGETS}
    rep = {"n_cases": int(ok.scenario_id.nunique()),
           "n_failed_runs": int((runs.status != "ok").sum()),
           "reference_level": reference, "targets": {}}
    for t, w in wide.items():
        base = w["production"]
        tr = {}
        for lv in w.columns:
            if lv == "production":
                continue
            d = (base - w[lv]).dropna()
            rel = (d / w[lv].loc[d.index].abs()).abs()
            tr[f"production_minus_{lv}"] = {
                "median_rel": float(rel.median()), "p90_rel": float(rel.quantile(0.9)),
                "max_rel": float(rel.max()), "mean_signed": float(d.mean()),
                "max_abs": float(d.abs().max()), "n": int(len(d))}
        if "finer" in w.columns and reference in w.columns:
            d = (w[reference] - w["finer"]).dropna()
            rel = (d / w["finer"].loc[d.index].abs()).abs()
            tr[f"{reference}_minus_finer"] = {
                "median_rel": float(rel.median()), "max_rel": float(rel.max()),
                "max_abs": float(d.abs().max()), "n": int(len(d))}
        rep["targets"][t] = tr
    dp = wide["dp_bh_max_MPa"]
    if reference in dp.columns:
        both = dp[["production", reference]].dropna()
        flips = (both["production"] > dp_limit_MPa) != (both[reference] > dp_limit_MPa)
        rep["pressure_limit_label_flips"] = {
            "dp_limit_MPa": dp_limit_MPa, "n_compared": int(len(both)),
            "n_flipped": int(flips.sum()),
            "flipped_cases": both.index[flips].tolist(),
            "n_within_5pct_of_limit": int((np.abs(both["production"] - dp_limit_MPa)
                                           < 0.05 * dp_limit_MPa).sum())}
    return rep
