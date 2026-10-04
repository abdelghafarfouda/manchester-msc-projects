"""Surrogate regression models -- only families taught in the supplied
Data Science and Machine Learning notebooks.

======================  ===========================================  ==========
name                    estimator                                    source
======================  ===========================================  ==========
``mean``                ``DummyRegressor`` (predict training mean)   Lecture01 (baseline)
``linear``              ``LinearRegression``                         Lecture01
``ridge``/``lasso``/    L2 / L1 / L1+L2 regularised linear models   Lecture02,
``elasticnet``                                                       E02_Regularization
``knn``                 ``KNeighborsRegressor``                      Lecture04
``svr``                 ``SVR`` (RBF kernel)                         Lecture05
``tree``                ``DecisionTreeRegressor``                    Lecture05
``random_forest``       ``RandomForestRegressor``                    Lecture06
``gradient_boosting``   ``GradientBoostingRegressor``                Lecture06
``adaboost``            ``AdaBoostRegressor``                        Lecture06
``xgboost``             ``XGBRegressor`` (if installed)              Lecture06, E02_Ensembles
======================  ===========================================  ==========

Every model is a :class:`~sklearn.pipeline.Pipeline` (``E01_Pipeline``,
``Lecture04``): imputation and scaling are fitted inside each CV fold and
never see validation rows.  Hyper-parameters are searched with
``RandomizedSearchCV`` (``Lecture06``) over bounded ranges, using
``GroupKFold`` on ``realisation_id`` -- K-fold CV (``Lecture05``) with the
realisation as the block, as in the geographical splitting of
``E03_geographicalspliting.ipynb``.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .splits import grouped_cv

try:                                             # optional
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:                                # pragma: no cover
    HAS_XGB = False


def _pre():
    """Median imputation + standard scaling, fitted inside every fold."""
    return [("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]


def build_models(random_state: int = 0, families=None) -> dict:
    """``{name: (pipeline, param_distributions)}`` for the course families."""
    rs = random_state
    m = {
        "mean": (DummyRegressor(strategy="mean"), {}),
        "linear": (LinearRegression(), {}),
        "ridge": (Ridge(), {"model__alpha": loguniform(1e-3, 1e3)}),
        "lasso": (Lasso(max_iter=50000), {"model__alpha": loguniform(1e-5, 1e0)}),
        "elasticnet": (ElasticNet(max_iter=50000),
                       {"model__alpha": loguniform(1e-5, 1e0),
                        "model__l1_ratio": uniform(0.05, 0.9)}),
        "knn": (KNeighborsRegressor(),
                {"model__n_neighbors": randint(2, 25),
                 "model__weights": ["uniform", "distance"]}),
        "svr": (SVR(kernel="rbf"),
                {"model__C": loguniform(1e-1, 1e3),
                 "model__gamma": loguniform(1e-4, 1e0),
                 "model__epsilon": loguniform(1e-4, 1e-1)}),
        "tree": (DecisionTreeRegressor(random_state=rs),
                 {"model__max_depth": randint(2, 14),
                  "model__min_samples_leaf": randint(1, 15)}),
        "random_forest": (RandomForestRegressor(n_estimators=300,
                                                random_state=rs, n_jobs=1),
                          {"model__max_depth": randint(3, 20),
                           "model__min_samples_leaf": randint(1, 10),
                           "model__max_features": uniform(0.2, 0.8)}),
        "gradient_boosting": (GradientBoostingRegressor(random_state=rs),
                              {"model__n_estimators": randint(100, 600),
                               "model__learning_rate": loguniform(0.01, 0.3),
                               "model__max_depth": randint(2, 6),
                               "model__subsample": uniform(0.6, 0.4)}),
        "adaboost": (AdaBoostRegressor(random_state=rs),
                     {"model__n_estimators": randint(50, 400),
                      "model__learning_rate": loguniform(0.01, 1.0)}),
    }
    if HAS_XGB:
        m["xgboost"] = (XGBRegressor(n_estimators=500, random_state=rs,
                                     n_jobs=1, tree_method="hist",
                                     verbosity=0),
                        {"model__max_depth": randint(2, 8),
                         "model__learning_rate": loguniform(0.01, 0.3),
                         "model__subsample": uniform(0.6, 0.4),
                         "model__colsample_bytree": uniform(0.5, 0.5),
                         "model__reg_lambda": loguniform(1e-2, 1e2)})
    out = {}
    for name, (est, dist) in m.items():
        if families is not None and name not in families:
            continue
        out[name] = (Pipeline(_pre() + [("model", est)]), dist)
    return out


# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, unit: str = "") -> dict:
    """MAE, MSE, RMSE, R^2 and max error (``E03_Evaluationmetrics``), plus
    relative versions that make targets of different units comparable."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    scale = float(np.mean(np.abs(y_true))) or 1.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse,
            "R2": float(r2_score(y_true, y_pred)),
            "max_error": float(max_error(y_true, y_pred)),
            "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
            "nMAE": mae / scale, "nRMSE": rmse / scale,
            "n": int(len(y_true)), "unit": unit}


@dataclass
class FittedSurrogate:
    """A fitted pipeline plus everything needed to use it correctly."""
    name: str
    target: str
    unit: str
    estimator: object
    features: list
    cv_score_rmse: float
    best_params: dict
    fit_seconds: float
    log_target: bool = False
    meta: dict = field(default_factory=dict)

    def predict(self, X):
        missing = [f for f in self.features if f not in getattr(X, "columns", [])]
        if missing:
            raise ValueError(f"surrogate '{self.target}' needs features "
                             f"{missing}; build inputs with features.py")
        return self.estimator.predict(X[self.features])


def _wrap(pipe, dist, log_target, hybrid=False,
          offset_feature="log10_rom_dp_MPa"):
    if hybrid:
        # learn ln(y) - ln(ROM): the analytical ROM times a learned
        # correction factor (hybrid.py)
        from .hybrid import ROMOffsetRegressor
        est = ROMOffsetRegressor(regressor=pipe, offset_feature=offset_feature)
        return est, {f"regressor__{k}": v for k, v in dist.items()}
    if not log_target:
        return pipe, dist
    # log transform of a strictly positive, multiplicative target -- the log
    # transform of E01_FeatureEngineering_Encoding applied to the target
    est = TransformedTargetRegressor(regressor=pipe, func=np.log,
                                     inverse_func=np.exp)
    return est, {f"regressor__{k}": v for k, v in dist.items()}


def fit_and_select(X, y, groups, features, target, unit="", *, n_splits=4,
                   n_iter=16, random_state=0, log_target=False,
                   hybrid=False, offset_feature="log10_rom_dp_MPa",
                   families=None, verbose=True, n_jobs=-1):
    """Tune every family with a bounded randomised search inside grouped
    K-fold CV and select the lowest CV RMSE.  Returns ``(best, results,
    fitted)`` where ``results[name]`` holds the CV RMSE, parameters and fit
    time.  ``hybrid=True`` fits every family as the multiplicative
    correction of the analytical ROM (:mod:`hybrid`); only the training data
    passed in are used for tuning and selection."""
    cv = grouped_cv(n_splits)
    results, fitted = {}, {}
    best = None
    X = X[list(features)]
    for name, (pipe, dist) in build_models(random_state, families).items():
        est, dist = _wrap(pipe, dist, log_target, hybrid, offset_feature)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dist:
                search = RandomizedSearchCV(
                    est, dist, n_iter=n_iter, cv=cv,
                    scoring="neg_root_mean_squared_error",
                    random_state=random_state, n_jobs=n_jobs, refit=True)
                search.fit(X, y, groups=groups)
                model, score, params = (search.best_estimator_,
                                        -search.best_score_,
                                        search.best_params_)
            else:
                sc = cross_val_score(est, X, y, groups=groups, cv=cv,
                                     scoring="neg_root_mean_squared_error",
                                     n_jobs=n_jobs)
                model, score, params = est.fit(X, y), float(-sc.mean()), {}
        dt = time.perf_counter() - t0
        results[name] = {"cv_rmse": float(score), "best_params":
                         {k: (v.item() if hasattr(v, "item") else v)
                          for k, v in params.items()}, "fit_seconds": dt}
        if verbose:
            print(f"    {name:18s} CV RMSE = {score:10.4g}  ({dt:5.1f} s)")
        cand = FittedSurrogate(name=name, target=target, unit=unit,
                               estimator=model, features=list(features),
                               cv_score_rmse=float(score),
                               best_params=results[name]["best_params"],
                               fit_seconds=dt, log_target=log_target or hybrid,
                               meta={"kind": "hybrid" if hybrid else
                                     ("log" if log_target else "linear")})
        fitted[name] = cand
        if best is None or cand.cv_score_rmse < best.cv_score_rmse:
            best = cand
    return best, results, fitted
