"""Hybrid surrogate: the analytical ROM times a learned correction factor.

For the peak bottom-hole pressure build-up the analytical reduced-order model
of :mod:`rom` is already a good first estimate (it is exact in the
single-phase pseudo-steady-state limit, verification V12/V13).  Instead of
asking a regression model to learn the whole response from scratch, the
hybrid surrogate learns only the **multiplicative residual** the ROM leaves:

.. math::

    \\ln \\Delta p_{bh,max} = \\ln \\Delta p_{ROM}(x) + g(x),

with ``g`` fitted by any regression pipeline of :mod:`models` on
``ln(y) - ln(ROM)``.  The ROM estimate is read from a feature column, so the
same pipeline works unchanged in grouped cross-validation, randomised search,
the screening and the saved-model loader.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class ROMOffsetRegressor(RegressorMixin, BaseEstimator):
    """Fit ``regressor`` on ``ln(y) - ln(ROM)``; predict ``ROM * exp(g(x))``.

    Parameters
    ----------
    regressor : scikit-learn regressor (usually a preprocessing pipeline)
    offset_feature : column of ``X`` holding ``log10`` of the ROM estimate in
        the target's units (``log10_rom_dp_MPa`` for the pressure target)
    """

    def __init__(self, regressor=None, offset_feature: str = "log10_rom_dp_MPa"):
        self.regressor = regressor
        self.offset_feature = offset_feature

    def _offset(self, X):
        if not hasattr(X, "columns") or self.offset_feature not in X.columns:
            raise ValueError(f"X must be a DataFrame with column "
                             f"'{self.offset_feature}' (build it with features.py)")
        return np.log(10.0) * X[self.offset_feature].to_numpy(float)

    def fit(self, X, y):
        y = np.asarray(y, float)
        if np.any(y <= 0):
            raise ValueError("the hybrid surrogate needs a strictly positive target")
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.log(y) - self._offset(X))
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X) + self._offset(X))

    def predict_log_correction(self, X):
        """The learned ``g(x)`` alone (diagnostics)."""
        return self.regressor_.predict(X)
