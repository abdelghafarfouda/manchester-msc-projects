"""Prediction intervals for the surrogates, calibrated and evaluated with the
reservoir realisation as the unit.

Four interval constructions are compared (all two-sided, nominal 90 %, on
the log scale for the strictly positive targets, i.e. relative intervals):

``empirical``  the original method -- P5 and P95 of the pooled calibration
    residuals.  No finite-sample statement.
``case_conformal``  split-conformal on the pooled calibration *cases*
    (score ``|ln y - ln yhat|``).  The usual guarantee assumes exchangeable
    cases; schedules of one reservoir are not independent, so for grouped
    data this is an approximation, reported as such.
``reservoir_conformal``  split-conformal on *reservoirs*: each calibration
    reservoir contributes one score, the largest of its schedules' scores.
    If a new reservoir and its schedules are drawn exactly like the
    calibration reservoirs (exchangeable reservoirs, same number of
    schedules), then with probability at least ``1 - alpha`` *all* of its
    schedules fall inside their intervals.  This is a statement about the
    sampling design of this study only: it does not hold for a reservoir
    from a different distribution (tested on the distribution-shift set), and
    it does not cover the thousands of candidate schedules screened per
    reservoir, which are many more than the four per reservoir it was
    calibrated on.
``adaptive_conformal``  as ``reservoir_conformal`` but with scores divided
    by a difficulty estimate ``sigma(x)`` -- a random forest fitted to the
    absolute out-of-fold residuals of the *training* reservoirs -- so the
    interval is wider where the surrogate has been less accurate.

References: split-conformal prediction -- Vovk, Gammerman & Shafer (2005),
*Algorithmic Learning in a Random World*, Springer; Lei, G'Sell, Rinaldo,
Tibshirani & Wasserman (2018), JASA 113(523), 1094-1111.  Conformal
prediction for grouped (two-layer hierarchical) data, where cases are not
exchangeable but groups are -- Dunn, Wasserman & Ramdas (2023), JASA
118(544), 2491-2502, doi:10.1080/01621459.2022.2060112.  The per-reservoir
maximum score used here is a simple, conservative construction for that
setting (simultaneous coverage of all schedules of a reservoir); it is not
one of that paper's pooling or subsampling estimators.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

METHODS = ("empirical", "case_conformal", "reservoir_conformal",
           "adaptive_conformal")


def conformal_quantile(scores, alpha: float) -> float:
    """``ceil((n + 1)(1 - alpha))``-th smallest score; ``inf`` if ``n`` is too
    small for the requested level."""
    s = np.sort(np.asarray(scores, float))
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


class IntervalModel:
    """A fitted surrogate plus one interval construction.

    Parameters
    ----------
    surrogate : object with ``predict(X)`` and attribute ``log_target`` (or
        pass ``log_space``)
    method : one of :data:`METHODS`
    alpha : miscoverage level (0.10 -> nominal 90 % two-sided)
    """

    def __init__(self, surrogate, method: str = "reservoir_conformal",
                 alpha: float = 0.10, log_space: bool | None = None,
                 difficulty=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.s = surrogate
        self.method = method
        self.alpha = float(alpha)
        self.log_space = (bool(getattr(surrogate, "log_target", False))
                          if log_space is None else bool(log_space))
        self.difficulty = difficulty          # fitted sigma(x) model or None
        self.q_ = self.lo_ = self.hi_ = None
        self.n_cal_cases_ = self.n_cal_groups_ = 0

    # -- residuals in the working space ------------------------------------
    def _r(self, y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        if self.log_space:
            return np.log(np.maximum(y, 1e-300)) - np.log(np.maximum(p, 1e-300))
        return y - p

    def _sigma(self, X):
        if self.difficulty is None:
            return np.ones(len(X))
        return np.maximum(np.asarray(self.difficulty.predict(X), float), 1e-6)

    def calibrate(self, X, y, groups):
        p = np.asarray(self.s.predict(X), float)
        r = self._r(y, p)
        g = np.asarray(groups)
        self.n_cal_cases_, self.n_cal_groups_ = int(r.size), int(np.unique(g).size)
        if self.method == "empirical":
            self.lo_ = float(np.percentile(r, 100 * self.alpha / 2))
            self.hi_ = float(np.percentile(r, 100 * (1 - self.alpha / 2)))
            return self
        score = np.abs(r)
        if self.method == "adaptive_conformal":
            score = score / self._sigma(X)
        if self.method == "case_conformal":
            self.q_ = conformal_quantile(score, self.alpha)
        else:
            per_group = pd.Series(score).groupby(g).max().to_numpy()
            self.q_ = conformal_quantile(per_group, self.alpha)
        return self

    def predict_interval(self, X):
        """``(lower, point, upper)`` in physical units."""
        p = np.asarray(self.s.predict(X), float)
        if self.method == "empirical":
            lo_r, hi_r = np.full(p.size, self.lo_), np.full(p.size, self.hi_)
        else:
            w = self.q_ * self._sigma(X)
            lo_r, hi_r = -w, w
        if self.log_space:
            return p * np.exp(lo_r), p, p * np.exp(hi_r)
        return p + lo_r, p, p + hi_r

    def evaluate(self, X, y, groups, subgroups: pd.Series | None = None) -> dict:
        lo, p, hi = self.predict_interval(X)
        y = np.asarray(y, float)
        g = np.asarray(groups)
        inside = (y >= lo) & (y <= hi)
        res_cov = pd.Series(inside).groupby(g).all()
        out = {"method": self.method, "nominal": 1 - self.alpha,
               "n_cases": int(y.size), "n_reservoirs": int(res_cov.size),
               "case_coverage": float(inside.mean()),
               "reservoir_all_covered": float(res_cov.mean()),
               "n_above_upper": int(np.sum(y > hi)),
               "n_below_lower": int(np.sum(y < lo)),
               "mean_width": float(np.mean(hi - lo)),
               "median_width": float(np.median(hi - lo)),
               "mean_relative_width": float(np.mean((hi - lo) / np.maximum(np.abs(p), 1e-12))),
               "n_calibration_cases": self.n_cal_cases_,
               "n_calibration_reservoirs": self.n_cal_groups_,
               "calibrated_quantile": self.q_,
               "empirical_residual_percentiles": [self.lo_, self.hi_]}
        if subgroups is not None:
            sg = pd.Series(np.asarray(subgroups))
            out["case_coverage_by_subgroup"] = {
                str(k): {"n": int(len(v)), "coverage": float(inside[v.index].mean())}
                for k, v in sg.groupby(sg)}
        return out


class _Difficulty:
    """``sigma(x)`` model that selects its own input columns."""

    def __init__(self, model, columns):
        self.model, self.columns = model, list(columns)

    def predict(self, X):
        return self.model.predict(X[self.columns])


def fit_difficulty_model(X_train, abs_oof_residual, random_state: int = 0):
    """``sigma(x)``: random forest of the absolute out-of-fold residuals of
    the training reservoirs (never of calibration or test data).  ``X_train``
    is a DataFrame whose columns are the inputs the model will use."""
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                              max_features=0.5, random_state=random_state,
                              n_jobs=1)
    m.fit(X_train, np.asarray(abs_oof_residual, float))
    return _Difficulty(m, X_train.columns)
