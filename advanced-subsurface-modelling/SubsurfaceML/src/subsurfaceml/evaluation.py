"""Evaluation that reports the tail, the subgroups and the uncertainty of the
comparison -- not only the average.

* :func:`point_metrics` -- RMSE, MAE, R^2 *and* the worst under- and
  over-prediction, the 95th-percentile absolute error and the share of cases
  with a relative error above 20 %.
* :func:`subgroup_table` -- the same metrics within physically meaningful
  subgroups (permeability tercile with **fixed** training-set edges, realised
  vs prior permeability mismatch, cases near the stated pressure limit).
* :func:`limit_decisions` -- missed exceedances and false alarms when a
  prediction (or an upper bound) is compared with the stated limit.
* :func:`paired_bootstrap` -- confidence interval of the difference between
  two methods, resampling **whole reservoirs** (the grouping unit), so the
  correlation between schedules of one reservoir is respected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(y, pred) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    e = p - y                                  # positive = over-prediction
    ae = np.abs(e)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or np.nan
    rel = ae / np.maximum(np.abs(y), 1e-12)
    return {"n": int(y.size),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAE": float(ae.mean()),
            "R2": float(1.0 - np.sum(e ** 2) / ss_tot),
            "max_abs_error": float(ae.max()),
            "p95_abs_error": float(np.percentile(ae, 95)),
            "worst_underprediction": float(max(0.0, -e.min())),
            "worst_overprediction": float(max(0.0, e.max())),
            "median_rel_error": float(np.median(rel)),
            "share_rel_error_above_20pct": float(np.mean(rel > 0.20)),
            "mean_signed_error": float(e.mean())}


def prior_mean_arith_k(k_median_mD, V_DP):
    """Expected arithmetic-mean layer permeability under the sampling prior:
    for lognormal layers with ``sigma = -ln(1 - V_DP)``, ``E[k] = k_50
    exp(sigma^2 / 2)``.  The ratio of the *realised* arithmetic mean to this
    expectation measures how unrepresentative a realisation's few layers are
    of the prior parameters ``(k_median, V_DP)``."""
    s = -np.log(1.0 - np.clip(np.asarray(V_DP, float), 0.0, 0.95))
    return np.asarray(k_median_mD, float) * np.exp(0.5 * s ** 2)


def add_subgroups(df: pd.DataFrame, k_edges=None, dp_limit_MPa=None,
                  target="dp_bh_max_MPa") -> pd.DataFrame:
    """Add subgroup labels.  ``k_edges`` (two numbers, mD) must come from the
    training data so every evaluation set is binned identically."""
    d = df.copy()
    k = d["k_median_mD"].to_numpy(float)
    if k_edges is None:
        k_edges = np.quantile(k, [1 / 3, 2 / 3])
    d["k_group"] = np.where(k < k_edges[0], "low k",
                            np.where(k < k_edges[1], "mid k", "high k"))
    if "k_arith_mean_m2" in d:
        ratio = (d["k_arith_mean_m2"] / 9.869233e-16) / prior_mean_arith_k(
            d["k_median_mD"], d["V_DP"])
        d["realised_vs_prior_k"] = ratio
        d["mismatch_group"] = np.where(ratio < 0.5, "realised k < 0.5x prior",
                                       np.where(ratio > 2.0, "realised k > 2x prior",
                                                "within 0.5-2x"))
    if dp_limit_MPa is not None and target in d:
        d["near_limit"] = np.where(np.abs(d[target] - dp_limit_MPa)
                                   <= 0.2 * dp_limit_MPa, "within 20% of limit",
                                   "far from limit")
    return d


def subgroup_table(d: pd.DataFrame, y_col: str, pred_col: str,
                   groups=("k_group", "mismatch_group", "near_limit")) -> dict:
    out = {}
    for g in groups:
        if g not in d:
            continue
        out[g] = {str(lv): point_metrics(sub[y_col], sub[pred_col])
                  for lv, sub in d.groupby(g, observed=True)}
    return out


def limit_decisions(y, pred_or_upper, limit) -> dict:
    """Exceedance decisions: a case is *declared safe* when the prediction
    (or its upper bound) is at or below ``limit``."""
    y = np.asarray(y, float)
    p = np.asarray(pred_or_upper, float)
    exceed = y > limit
    declared_safe = p <= limit
    return {"n": int(y.size), "n_exceeding": int(exceed.sum()),
            "missed_exceedances": int(np.sum(exceed & declared_safe)),
            "false_alarms": int(np.sum(~exceed & ~declared_safe)),
            "declared_safe": int(declared_safe.sum())}


def paired_bootstrap(groups, y, pred_a, pred_b, *, n_boot: int = 2000,
                     seed: int = 0, stats=("RMSE", "MAE", "max_abs_error")) -> dict:
    """Bootstrap over reservoirs of ``metric(b) - metric(a)``.

    Returns, for each statistic, the observed difference and the 2.5 / 97.5
    percentiles of its bootstrap distribution; a negative difference means
    ``b`` has the smaller error.
    """
    g = np.asarray(groups)
    y, a, b = (np.asarray(v, float) for v in (y, pred_a, pred_b))
    uniq = np.unique(g)
    idx_by = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)

    def _stat(ix, p, s):
        return point_metrics(y[ix], p[ix])[s]

    full = np.arange(y.size)
    obs = {s: _stat(full, b, s) - _stat(full, a, s) for s in stats}
    boots = {s: [] for s in stats}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        ix = np.concatenate([idx_by[u] for u in pick])
        for s in stats:
            boots[s].append(_stat(ix, b, s) - _stat(ix, a, s))
    return {s: {"difference_b_minus_a": float(obs[s]),
                "ci95": [float(np.percentile(boots[s], 2.5)),
                         float(np.percentile(boots[s], 97.5))],
                "share_boot_b_better": float(np.mean(np.asarray(boots[s]) < 0))}
            for s in stats} | {"n_reservoirs": int(uniq.size), "n_boot": n_boot}
