"""Surrogate error bands, Monte Carlo propagation and the error-source table.

Everything here uses ideas from ``5-Uncertainty.pdf``:

* **P10 / P50 / P90** (p.46) and Monte Carlo propagation (p.44-45);
* the three **sources of uncertainty** -- input data, model bias, numerical
  error (p.8-9, 27-31);
* combining independent uncertainties (p.32).

Empirical error band
--------------------
The surrogate's error is characterised by the distribution of its residuals
on the **calibration realisations** -- reservoirs held out from fitting and
tuning.  The band is ``prediction + [P_lo, P_hi]`` of those residuals (in log
space for targets modelled on a log scale, so the band is relative).  Its
**coverage is then measured** on the untouched test realisations and
reported as measured.  No distribution-free guarantee is claimed: the
residuals of schedules on the same reservoir are correlated, the calibration
set is small, and a reservoir unlike those seen could be missed entirely.

(An earlier version used split-conformal prediction.  That method is not in
the supplied course material and was replaced by this simpler, explicitly
empirical band.)
"""
from __future__ import annotations

import numpy as np


class ErrorBand:
    """Empirical residual-percentile band around a fitted surrogate."""

    def __init__(self, surrogate, lower_pct: float = 5.0,
                 upper_pct: float = 95.0):
        self.s = surrogate
        self.lo_pct, self.hi_pct = float(lower_pct), float(upper_pct)
        self.lo_ = self.hi_ = None
        self.space_ = None
        self.n_cal_ = 0

    def _res(self, y, pred):
        if getattr(self.s, "log_target", False):
            self.space_ = "log"
            return np.log(np.maximum(y, 1e-300)) - np.log(np.maximum(pred, 1e-300))
        self.space_ = "linear"
        return y - pred

    def calibrate(self, X_cal, y_cal):
        y = np.asarray(y_cal, float)
        r = self._res(y, self.s.predict(X_cal))
        self.lo_, self.hi_ = (float(np.percentile(r, self.lo_pct)),
                              float(np.percentile(r, self.hi_pct)))
        self.n_cal_ = int(len(y))
        return self

    def predict_interval(self, X):
        """``(lower, point, upper)`` in the target's physical units."""
        p = np.asarray(self.s.predict(X), float)
        if self.space_ == "log":
            return p * np.exp(self.lo_), p, p * np.exp(self.hi_)
        return p + self.lo_, p, p + self.hi_

    def evaluate(self, X, y) -> dict:
        lo, p, hi = self.predict_interval(X)
        y = np.asarray(y, float)
        inside = (y >= lo) & (y <= hi)
        return {"band_percentiles": [self.lo_pct, self.hi_pct],
                "nominal_fraction_on_calibration": (self.hi_pct - self.lo_pct) / 100,
                "measured_coverage_test": float(inside.mean()),
                "n_test": int(len(y)), "n_calibration": self.n_cal_,
                "residual_space": self.space_,
                "residual_P_lo": self.lo_, "residual_P_hi": self.hi_,
                "mean_width": float(np.mean(hi - lo)),
                "median_width": float(np.median(hi - lo)),
                "under_prediction_misses": int(np.sum(y > hi)),
                "over_prediction_misses": int(np.sum(y < lo)),
                "note": "empirical band; measured coverage, no guarantee"}


# --------------------------------------------------------------------------
def monte_carlo_propagate(surrogate, X_prior) -> dict:
    """Push a Monte Carlo sample of the (assumed) prior through the surrogate
    and report P10/P50/P90 (``5-Uncertainty.pdf`` p.44-46).  Following the
    lecture's definition, P10 is exceeded by 10% of outcomes, i.e. it is the
    90th percentile of the sample (the "high" case)."""
    pred = np.asarray(surrogate.predict(X_prior), float)
    return {"n_samples": int(len(pred)),
            "mean": float(np.mean(pred)), "std": float(np.std(pred, ddof=1)),
            "P90_low_case": float(np.percentile(pred, 10)),
            "P50": float(np.percentile(pred, 50)),
            "P10_high_case": float(np.percentile(pred, 90)),
            "convention": "5-Uncertainty.pdf p.46: P10 = value exceeded by "
                          "10% of estimates"}


def error_source_table(target: str, prior_std: float, surrogate_rmse: float,
                       numerical_rel_error: float, reference_value: float,
                       unmodelled: list[str]) -> dict:
    """The three sources of uncertainty of ``5-Uncertainty.pdf`` p.8-9 in one
    unit, side by side.  They are combined in quadrature (independent
    uncertainties) only for a rough total; unmodelled physics (model bias)
    is listed, not quantified."""
    numerical_abs = abs(numerical_rel_error) * abs(reference_value)
    total = float(np.sqrt(prior_std ** 2 + surrogate_rmse ** 2 + numerical_abs ** 2))
    return {
        "target": target,
        "input_uncertainty_std": float(prior_std),
        "surrogate_error_rmse": float(surrogate_rmse),
        "numerical_error_abs": float(numerical_abs),
        "numerical_error_rel": float(numerical_rel_error),
        "quadrature_total_excl_unmodelled": total,
        "share_input_pct": float(100 * prior_std ** 2 / total ** 2),
        "share_surrogate_pct": float(100 * surrogate_rmse ** 2 / total ** 2),
        "share_numerical_pct": float(100 * numerical_abs ** 2 / total ** 2),
        "model_bias_unmodelled_physics_not_quantified": unmodelled,
        "caveat": ("Input uncertainty is the spread of an ASSUMED prior, not "
                   "uncertainty constrained by measurements. Model bias from "
                   "omitted physics is listed, not quantified."),
    }


UNMODELLED_PHYSICS = [
    "gravity / buoyant override (1-D radial layers, no vertical flow)",
    "CO2 dissolution in brine and the two-shock structure (4-CO2 BL.pdf p.25-29)",
    "residual trapping / imbibition hysteresis (3-IMPES.pdf p.4)",
    "capillary pressure (Pc = 0, 4-CO2 BL.pdf p.19)",
    "vertical crossflow between layers",
    "non-isothermal effects, geomechanics, 3-D geological structure",
]
