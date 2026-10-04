"""Independent evaluation sets and like-for-like model comparison.

Two sets of reservoirs are generated *after* every modelling choice has been
fixed on the development reservoirs (``docs/EVALUATION_PROTOCOL.md``):

``final_test``
    fresh reservoirs from the **same prior** as the development data, with a
    different seed -- the untouched in-distribution test set.
``shift``
    fresh reservoirs from a prior that differs in one respect only: median
    permeability 10-30 mD instead of 30-1000 mD (the regime in which the
    published surrogate failed) -- a distribution-shift test set.

Realisation ids are offset (10000+, 20000+) and the seeds differ from the
development seed, so no reservoir can appear in two sets
(:func:`check_disjoint`).

The comparison trains each *design* (feature set + target form) on the same
development reservoirs, calibrates its intervals on the same calibration
reservoirs, and scores it on the same independent sets.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .evaluation import add_subgroups, limit_decisions, point_metrics, subgroup_table
from .features import FEATURES, FEATURES_BASELINE, ROCK_FEATURES, engineer
from .scenarios import run_scenario, sample_realisations, sample_schedules

SETS = ("final_test", "shift")


def eval_config(cfg, which: str):
    """Config copy whose scenario prior and seed generate set ``which``."""
    if which not in SETS:
        raise ValueError(f"which must be one of {SETS}")
    e = cfg.evaluation
    c = copy.deepcopy(cfg)
    if which == "final_test":
        c.scenarios.seed = e.final_test_seed
        c.scenarios.n_realisations = e.final_test_realisations
    else:
        c.scenarios.seed = e.shift_seed
        c.scenarios.n_realisations = e.shift_realisations
        c.scenarios.k_median_mD = tuple(e.shift_k_median_mD)
    return c


def eval_realisations(cfg, which: str):
    e = cfg.evaluation
    off = e.final_test_id_offset if which == "final_test" else e.shift_id_offset
    c = eval_config(cfg, which)
    return c, [dataclasses.replace(r, realisation_id=off + r.realisation_id)
               for r in sample_realisations(c)]


def generate_eval_set(cfg, which: str, *, n_jobs=None, verbose=True):
    """Simulate every (reservoir, schedule) of set ``which``."""
    c, reals = eval_realisations(cfg, which)
    jobs = [(r, s) for r in reals for s in sample_schedules(c, r)]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs or cfg.n_jobs, batch_size=4)(
        delayed(run_scenario)(c, r, s, want_series=False) for r, s in jobs)
    wall = time.perf_counter() - t0
    rows = [r["row"] for r in res if r["status"] == "ok"]
    failed = pd.DataFrame([r for r in res if r["status"] != "ok"])
    df = engineer(pd.DataFrame(rows))
    df["set"] = which
    prov = {"set": which, "seed": c.scenarios.seed,
            "n_realisations": len(reals), "n_requested": len(jobs),
            "n_success": len(rows), "n_failed": int(len(failed)),
            "k_median_mD_range": list(c.scenarios.k_median_mD),
            "wall_time_s": wall}
    if verbose:
        print(f"  [{which}] {len(rows)} ok / {len(failed)} failed "
              f"({len(reals)} reservoirs, {wall:.0f} s)")
    return df, failed, prov


def load_or_generate(cfg, which: str, *, force=False):
    d = Path(cfg.paths.data)
    f = d / f"{which}_scenarios.csv"
    pf = d / f"{which}_provenance.json"
    if f.exists() and pf.exists() and not force:
        df = pd.read_csv(f)
        return df, json.loads(pf.read_text())
    df, failed, prov = generate_eval_set(cfg, which)
    df.to_csv(f, index=False)
    failed.to_csv(d / f"{which}_failed_runs.csv", index=False)
    pf.write_text(json.dumps(prov, indent=2))
    return df, prov


def check_disjoint(*frames, cols=("k_median_mD", "V_DP", "phi_mean", "h_total_m",
                                  "r_e_m", "krg0")) -> dict:
    """No realisation id and no reservoir description occurs in two sets."""
    ids = [set(f["realisation_id"]) for f in frames]
    desc = [set(map(tuple, f[list(cols)].round(10).to_numpy())) for f in frames]
    over_ids = sum(len(a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    over_desc = sum(len(a & b) for i, a in enumerate(desc) for b in desc[i + 1:])
    return {"overlapping_ids": int(over_ids), "overlapping_descriptions": int(over_desc)}


# --------------------------------------------------------------------------
def design_features(feature_set: str):
    return {"baseline": FEATURES_BASELINE,
            "rock": FEATURES_BASELINE + ROCK_FEATURES,
            "all": FEATURES}[feature_set]


def oof_abs_residuals(best, X, y, groups, n_splits=4, log_space=True):
    """Absolute out-of-fold residuals of a fitted design on its own training
    reservoirs (grouped CV) -- used only to fit the adaptive interval's
    difficulty model."""
    from sklearn.base import clone
    from .splits import grouped_cv
    pred = np.empty(len(y))
    for tr, te in grouped_cv(n_splits).split(X, y, groups):
        m = clone(best.estimator).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    if log_space:
        return np.abs(np.log(y) - np.log(pred))
    return np.abs(y - pred)


def evaluate_on(name, surrogate, intervals: dict, df_set: pd.DataFrame, target,
                *, k_edges, dp_limit, domain=None, selected_interval=None) -> dict:
    """Point, subgroup, limit-decision, interval and domain results on one set."""
    X = df_set
    y = df_set[target].to_numpy(float)
    p = np.asarray(surrogate.predict(X), float)
    out = {"model": name, "n_cases": int(len(y)),
           "n_reservoirs": int(df_set["realisation_id"].nunique()),
           "point": point_metrics(y, p),
           "mean_baseline_rmse": None}
    d = add_subgroups(df_set, k_edges=k_edges, dp_limit_MPa=dp_limit, target=target)
    d["_pred"] = p
    out["subgroups"] = subgroup_table(d, target, "_pred")
    worst = d.assign(err=p - y).sort_values("err").head(5)
    out["worst_underpredictions"] = [
        {"scenario_id": w["scenario_id"], "simulated": float(w[target]),
         "predicted": float(w["_pred"]), "k_median_mD": float(w["k_median_mD"]),
         "realised_vs_prior_k": float(w.get("realised_vs_prior_k", np.nan))}
        for _, w in worst.iterrows()]
    if target == "dp_bh_max_MPa":
        out["limit_point"] = limit_decisions(y, p, dp_limit)
    out["intervals"] = {}
    for meth, im in intervals.items():
        ev = im.evaluate(X, y, df_set["realisation_id"].to_numpy(),
                         subgroups=d["k_group"])
        if target == "dp_bh_max_MPa":
            ev["limit_upper"] = limit_decisions(y, im.predict_interval(X)[2], dp_limit)
        out["intervals"][meth] = ev
    out["selected_interval_method"] = selected_interval
    if domain is not None:
        chk = domain.check(df_set)
        out["domain"] = {"share_in_domain_cases": float(chk.in_domain.mean()),
                         "share_reservoirs_flagged": float(
                             (~chk.in_domain).groupby(df_set["realisation_id"]).any().mean())}
        if chk.in_domain.any() and (~chk.in_domain).any():
            m = chk.in_domain.to_numpy()
            out["domain"]["point_in_domain"] = point_metrics(y[m], p[m])
            out["domain"]["point_flagged"] = point_metrics(y[~m], p[~m])
    return out


def load_published(root: Path, targets):
    """The surrogates and bands published on 2026-09-20 (hash-checked)."""
    import hashlib
    import joblib
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    out = {}
    for t in targets:
        for kind in ("surrogate", "band"):
            f = f"{kind}_{t}.joblib"
            h = hashlib.sha256((root / f).read_bytes()).hexdigest()
            if h != man["files"][f]["sha256"]:
                raise RuntimeError(f"{f}: hash differs from the published manifest")
        out[t] = (joblib.load(root / f"surrogate_{t}.joblib"),
                  joblib.load(root / f"band_{t}.joblib"))
    return out


def realisation_lookup(cfg) -> dict:
    """``{realisation_id: Realisation}`` over the development, final-test and
    shift sets, with the config each was generated under."""
    out = {r.realisation_id: (cfg, r) for r in sample_realisations(cfg)}
    for which in SETS:
        c, reals = eval_realisations(cfg, which)
        out.update({r.realisation_id: (c, r) for r in reals})
    return out


def published_interval(surrogate, band):
    """Wrap a published ``uncertainty.ErrorBand`` as an
    :class:`intervals.IntervalModel` with method ``empirical``."""
    from .intervals import IntervalModel
    im = IntervalModel(surrogate, "empirical", alpha=0.10,
                       log_space=(band.space_ == "log"))
    im.lo_, im.hi_ = band.lo_, band.hi_
    im.n_cal_cases_ = band.n_cal_
    return im
