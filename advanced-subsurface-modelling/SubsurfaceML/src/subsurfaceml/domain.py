"""Applicability-domain check: is a reservoir like the training reservoirs?

The surrogates and their intervals were calibrated on reservoirs drawn from
one assumed prior.  For a reservoir outside that population neither the
point prediction nor the interval has any support from the experiments, so
the screening must not rely on them.  This check flags such reservoirs.

Two criteria, both computed from reservoir descriptors known before
simulating (the same descriptors the surrogate uses):

1. **range** -- a descriptor lies outside the range spanned by the training
   reservoirs widened by ``tolerance`` (2 %) of that range on each side.
   With no tolerance, 12.7 % of the development reservoirs fall outside the
   range of the others (leave-one-out), simply because each of 16
   descriptors has two extreme values; with 2 % the rate is 1.8 %, similar to
   the 1 % of the distance criterion.  (Measured on development data before
   the independent sets existed.)
2. **distance** -- the standardised Euclidean distance to the nearest
   training reservoir exceeds the 99th percentile of the same distance
   computed leave-one-reservoir-out within the training set.

The check can only recognise departures in the descriptors it sees.  A
reservoir that is unusual in a way the descriptors do not capture -- for
example in physics the simulator does not model -- is not detected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Reservoir-level descriptors (schedule-independent) used by the check.
DESCRIPTORS = ["log10_k_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m", "n_g",
               "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g", "V_DP_layers",
               "log10_k_arith_mD", "log10_k_harm_mD", "log10_k_min_layer_mD",
               "log10_k_max_layer_mD"]


class DomainCheck:
    def __init__(self, descriptors=None, quantile: float = 0.99,
                 tolerance: float = 0.02):
        self.descriptors = list(descriptors or DESCRIPTORS)
        self.quantile = float(quantile)
        self.tolerance = float(tolerance)

    def fit(self, df: pd.DataFrame, group_col: str = "realisation_id"):
        R = df.drop_duplicates(group_col)[self.descriptors].to_numpy(float)
        w = R.max(axis=0) - R.min(axis=0)
        self.lo_ = R.min(axis=0) - self.tolerance * w
        self.hi_ = R.max(axis=0) + self.tolerance * w
        self.mu_, self.sd_ = R.mean(axis=0), R.std(axis=0) + 1e-12
        Z = (R - self.mu_) / self.sd_
        self.Z_ = Z
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        self.nn_train_ = D.min(axis=1)
        self.threshold_ = float(np.quantile(self.nn_train_, self.quantile))
        self.n_train_reservoirs_ = int(R.shape[0])
        return self

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per input row: ``in_domain``, ``nn_distance`` and the
        descriptors outside the training range."""
        X = df[self.descriptors].to_numpy(float)
        out_lo, out_hi = X < self.lo_, X > self.hi_
        Z = (X - self.mu_) / self.sd_
        nn = np.sqrt(((Z[:, None, :] - self.Z_[None, :, :]) ** 2).sum(-1)).min(axis=1)
        reasons = []
        for i in range(len(X)):
            r = [f"{d} below training range" for d, b in zip(self.descriptors, out_lo[i]) if b]
            r += [f"{d} above training range" for d, b in zip(self.descriptors, out_hi[i]) if b]
            if nn[i] > self.threshold_:
                r.append("far from every training reservoir")
            reasons.append("; ".join(r))
        in_dom = ~(out_lo.any(axis=1) | out_hi.any(axis=1)) & (nn <= self.threshold_)
        return pd.DataFrame({"in_domain": in_dom, "nn_distance": nn,
                             "reasons": reasons}, index=df.index)
