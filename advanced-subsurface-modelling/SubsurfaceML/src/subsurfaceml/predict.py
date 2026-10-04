"""Reusable inference: reservoir description + schedule -> predictions.

The command line (``subsurfaceml predict``), the Streamlit interface and the
worked example all call :class:`Predictor`, which

1. loads and verifies the saved artifacts (:mod:`artifacts`),
2. builds inputs through the one feature path (:mod:`features`),
3. returns point predictions, the calibrated interval and the classifier
   decision, with units, plus the applicability-domain check.

Predictions are **not** verified results: any schedule that matters must be
run through the simulator (``subsurfaceml simulate``; the screening in
:mod:`screening` does this automatically).
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from .artifacts import load_bundle
from .features import FEATURES, features_for_schedules
from .scenarios import Realisation

UNITS = {"dp_bh_max_MPa": "MPa", "r_plume_m95_m": "m", "sweep_efficiency": "-"}


def realisation_from_dict(d: dict, *, realisation_id: int = -1,
                          seed: int = 12345) -> Realisation:
    """Build a :class:`Realisation` from user inputs, with validation."""
    names = [f.name for f in fields(Realisation)
             if f.name not in ("realisation_id", "seed")]
    missing = [n for n in names if n not in d]
    if missing:
        raise ValueError(f"missing reservoir inputs: {missing}")
    vals = {n: float(d[n]) for n in names}
    if vals["k_median_mD"] <= 0 or not (0 <= vals["V_DP"] < 1):
        raise ValueError("need k_median_mD > 0 and 0 <= V_DP < 1")
    if not (0 < vals["phi_mean"] < 1) or not (0 < vals["S_ar"] < 1):
        raise ValueError("porosity and S_ar must lie in (0, 1)")
    return Realisation(realisation_id=realisation_id, seed=int(seed), **vals)


class Predictor:
    """Load the verified surrogates of one configuration."""

    def __init__(self, cfg):
        self.cfg = cfg
        b = load_bundle(cfg)
        self.manifest = b["manifest"]
        self.surrogates = {k[len("surrogate_"):-len(".joblib")]: v
                           for k, v in b.items() if k.startswith("surrogate_")}
        self.bands = {k[len("band_"):-len(".joblib")]: v
                      for k, v in b.items() if k.startswith("band_")}
        self.classifier = b.get("pressure_classifier.joblib")
        self.domain = b.get("domain_check.joblib")

    def features(self, realisation, rates_matrix):
        return features_for_schedules(self.cfg, realisation, rates_matrix, FEATURES)

    def predict(self, realisation, rates_matrix) -> dict:
        X = self.features(realisation, rates_matrix)
        out = {"inputs": X}
        if self.domain is not None:
            chk = self.domain.check(X)
            out["in_training_domain"] = chk["in_domain"].to_numpy()
            out["domain_reasons"] = chk["reasons"].tolist()
        for t, s in self.surrogates.items():
            band = self.bands.get(t)
            if band is not None:
                lo, p, hi = band.predict_interval(X)
            else:
                p = s.predict(X); lo = hi = p
            out[t] = {"pred": np.asarray(p), "band_low": np.asarray(lo),
                      "band_high": np.asarray(hi), "unit": UNITS.get(t, "")}
        if self.classifier is not None:
            c = self.classifier
            Xc = X[c.get("features", list(X.columns))]
            score = (c["model"].decision_function(Xc) if c.get("uses_decision_function")
                     else c["model"].predict_proba(Xc)[:, 1])
            out["exceeds_pressure_limit"] = {
                "score": np.asarray(score), "threshold": c["threshold"],
                "flag": np.asarray(score) >= c["threshold"],
                "dp_limit_MPa": c["dp_limit_MPa"]}
        return out
