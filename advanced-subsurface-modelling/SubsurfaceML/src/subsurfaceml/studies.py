"""ML-method studies: each answers one question about the surrogate using a
method taught in the supplied ``Data Science and machine learning`` folder.

All studies use the **training realisations only** with ``GroupKFold`` on
``realisation_id`` (or, where stated, the calibration realisations, which are
also held out from fitting).  None of them touches the test realisations, so
none of them can leak into the reported test scores.

==============================  =====================================  ========
study                           question                               topic
==============================  =====================================  ========
``learning_curve``              how many simulated reservoirs are      E03
                                enough?
``scaler_comparison``           does the scaler matter for             P06
                                distance/kernel/linear models?
``missing_descriptors``         what if a reservoir descriptor is      P04
                                unmeasured at prediction time?
``outliers``                    are extreme scenarios outliers to      P05
                                remove, or cases to keep?
``collinearity``                do correlated engineered features      P10
                                destabilise the linear surrogate?
``feature_selection``           how few inputs are enough? (filter /   P12
                                wrapper / embedded)
``polynomial``                  does a quadratic expansion rescue      P08
                                the linear model?
``column_transformer``          log-transform only the rate columns    P09
``tuning_strategies``           grid vs random vs Bayesian search      E08, E09
``normal_equation``             closed-form OLS/ridge vs scikit-learn  A01, A02
                                and gradient descent
``split_comparison``            how optimistic is a random row split?  P14
``boosting_early_stopping``     XGBoost train/validation curves        A09, E03
``ensembles``                   voting / stacking / bagging vs best    A08, A10
                                single model
==============================  =====================================  ========
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (BaggingRegressor, GradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       f_regression)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     cross_val_score, learning_curve)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .splits import grouped_cv


def _rmse_cv(est, X, y, g, cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cross_val_score(est, X, y, groups=g, cv=cv,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
    return float(-s.mean()), float(s.std())


def _p(*steps):
    return Pipeline(list(steps))


# --------------------------------------------------------------------------
def learning_curve_study(X, y, g, cv, est):
    """E03-LearningCurve / Lecture02: train vs validation error as the number
    of training rows (whole realisations per fold) grows."""
    sizes = np.linspace(0.15, 1.0, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, tr, va = learning_curve(est, X, y, groups=g, cv=cv, train_sizes=sizes,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
    return {"n_train_rows": n.tolist(),
            "train_rmse": (-tr.mean(axis=1)).tolist(),
            "cv_rmse": (-va.mean(axis=1)).tolist(),
            "cv_rmse_std": va.std(axis=1).tolist()}


def scaler_comparison(X, y, g, cv):
    """Lecture01 / E01_Preprocessing: Standard vs MinMax vs Robust vs none."""
    scalers = {"none": "passthrough", "standard": StandardScaler(),
               "minmax": MinMaxScaler(), "robust": RobustScaler()}
    models = {"knn": KNeighborsRegressor(n_neighbors=7, weights="distance"),
              "svr": SVR(C=30.0, gamma="scale"), "ridge": Ridge(alpha=1.0)}
    out = {}
    for mn, m in models.items():
        out[mn] = {}
        for sn, sc in scalers.items():
            est = _p(("impute", SimpleImputer(strategy="median")), ("scale", sc),
                     ("model", clone(m)))
            out[mn][sn] = _rmse_cv(est, X, y, g, cv)[0]
    return out


def missing_descriptors(X, y, g, cv, est, groups: dict, frac=0.5, seed=0):
    """E01_Preprocessing / Extra_transformer: a reservoir descriptor is not
    measured for ``frac`` of the validation reservoirs.  Every feature derived
    from it is then unknown too (a group of columns).  Compare SimpleImputer
    strategies (fitted on the training fold only) with a model trained
    without the group."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, cols in groups.items():
        res = {"mean": [], "median": [], "drop_feature": [], "complete": []}
        for tr, va in cv.split(X, y, g):
            Xtr, Xva = X.iloc[tr], X.iloc[va].copy()
            vg = np.unique(g[va])
            miss_g = vg[rng.random(len(vg)) < frac]          # whole reservoirs
            mask = np.isin(g[va], miss_g)
            base = clone(est).fit(Xtr, y[tr])
            res["complete"].append(np.sqrt(mean_squared_error(y[va], base.predict(Xva))))
            Xm = Xva.copy()
            Xm.loc[Xm.index[mask], cols] = np.nan
            for strat in ("mean", "median"):
                e = clone(est)
                e.set_params(**{_imp_param(e): strat})
                e.fit(Xtr, y[tr])
                res[strat].append(np.sqrt(mean_squared_error(y[va], e.predict(Xm))))
            keep = [c for c in X.columns if c not in cols]
            e = clone(est).fit(Xtr[keep], y[tr])
            pred = base.predict(Xva)
            if mask.any():
                pred[mask] = e.predict(Xva[keep])[mask]
            res["drop_feature"].append(np.sqrt(mean_squared_error(y[va], pred)))
        out[name] = {"columns": list(cols), **{k: float(np.mean(v)) for k, v in res.items()}}
    return out


def _imp_param(est):
    for k in est.get_params():
        if k.endswith("impute__strategy"):
            return k
    raise KeyError("no imputer in pipeline")


def outlier_study(df, target, feats, est, masks):
    """Lecture02 / E01_EDA: z-score and IQR flags on the target; are flagged
    scenarios errors to remove, or the physically extreme cases the model must
    handle?  Refit with and without them and score on the calibration
    realisations (never the test set)."""
    tr, ca = masks["train"], masks["calib"]
    y = df[target].to_numpy(float)
    q1, q3 = np.percentile(y[tr], [25, 75])
    iqr = q3 - q1
    iqr_flag = (y > q3 + 1.5 * iqr) | (y < q1 - 1.5 * iqr)
    z = (y - y[tr].mean()) / y[tr].std()
    z_flag = np.abs(z) > 3.0
    keep = tr & ~iqr_flag
    e_all = clone(est).fit(df.loc[tr, feats], y[tr])
    e_cut = clone(est).fit(df.loc[keep, feats], y[keep])
    ycal = y[ca]
    p_all, p_cut = e_all.predict(df.loc[ca, feats]), e_cut.predict(df.loc[ca, feats])
    fl = iqr_flag[ca]
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else None
    return {"iqr_bounds": [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)],
            "n_iqr_flagged_train": int(iqr_flag[tr].sum()),
            "n_z3_flagged_train": int(z_flag[tr].sum()),
            "flagged_train_median_k_mD": float(np.median(df.loc[tr & iqr_flag, "k_median_mD"]))
            if (tr & iqr_flag).any() else None,
            "unflagged_train_median_k_mD": float(np.median(df.loc[tr & ~iqr_flag, "k_median_mD"])),
            "calib_rmse_keep_all": rm(ycal, p_all),
            "calib_rmse_outliers_removed": rm(ycal, p_cut),
            "calib_rmse_on_flagged_rows_keep_all": rm(ycal[fl], p_all[fl]),
            "calib_rmse_on_flagged_rows_removed": rm(ycal[fl], p_cut[fl]),
            "n_calib_flagged": int(fl.sum()),
            "flagged_train_rows": df.loc[tr & iqr_flag, ["scenario_id", target, "k_median_mD",
                                                        "h_total_m", "q_mult_mean"]].to_dict("records")}


def collinearity_study(X, y, g, cv, thr=0.75):
    """Lecture05 / E01_FeatureSelection rule of thumb: drop one feature of
    every pair with |r| > 0.75; compare CV error and the fold-to-fold
    stability of the linear coefficients."""
    corr = X.corr().abs()
    cols = list(X.columns)
    dropped = []
    for i, a in enumerate(cols):
        if a in dropped:
            continue
        for b in cols[i + 1:]:
            if b not in dropped and corr.loc[a, b] > thr:
                dropped.append(b)
    kept = [c for c in cols if c not in dropped]
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(cols)
             for b in cols[i + 1:] if corr.loc[a, b] > thr]
    out = {"threshold": thr, "n_features": len(cols), "n_pairs_above": len(pairs),
           "dropped": dropped, "kept": kept,
           "top_pairs": sorted(pairs, key=lambda t: -t[2])[:10]}
    for name, cset in (("all", cols), ("filtered", kept)):
        lin = _p(("impute", SimpleImputer()), ("scale", StandardScaler()),
                 ("model", LinearRegression()))
        rmse, _ = _rmse_cv(lin, X[cset], y, g, cv)
        coefs = []
        for tr, _va in cv.split(X, y, g):
            coefs.append(clone(lin).fit(X[cset].iloc[tr], y[tr])[-1].coef_)
        coefs = np.array(coefs)
        out[name] = {"cv_rmse_linear": rmse,
                     "median_coef_cv_across_folds": float(np.median(
                         coefs.std(axis=0) / (np.abs(coefs.mean(axis=0)) + 1e-12)))}
    return out


def feature_selection_study(X, y, g, cv, ks=(3, 5, 8, 12)):
    """Lecture05 FAQ: filter (SelectKBest, f_regression), wrapper (RFE) and
    embedded (Lasso / SelectFromModel) selection, all inside the folds."""
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                   learning_rate=0.05, random_state=0)
    out = {"all_features": _rmse_cv(_p(("impute", SimpleImputer()),
                                       ("scale", StandardScaler()),
                                       ("model", clone(gb))), X, y, g, cv)[0]}
    for k in ks:
        out[f"filter_kbest_{k}"] = _rmse_cv(_p(
            ("impute", SimpleImputer()), ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)), ("model", clone(gb))),
            X, y, g, cv)[0]
        out[f"wrapper_rfe_{k}"] = _rmse_cv(_p(
            ("impute", SimpleImputer()), ("scale", StandardScaler()),
            ("select", RFE(Ridge(alpha=1.0), n_features_to_select=k)),
            ("model", clone(gb))), X, y, g, cv)[0]
    out["embedded_lasso"] = _rmse_cv(_p(
        ("impute", SimpleImputer()), ("scale", StandardScaler()),
        ("select", SelectFromModel(Lasso(alpha=0.01, max_iter=50000))),
        ("model", clone(gb))), X, y, g, cv)[0]
    # which features does each selector keep on the whole training set?
    pre = _p(("impute", SimpleImputer()), ("scale", StandardScaler()))
    Z = pre.fit_transform(X)
    kb = SelectKBest(f_regression, k=5).fit(Z, y)
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=5).fit(Z, y)
    las = SelectFromModel(Lasso(alpha=0.01, max_iter=50000)).fit(Z, y)
    cols = np.array(X.columns)
    out["kept_filter_5"] = cols[kb.get_support()].tolist()
    out["kept_rfe_5"] = cols[rfe.get_support()].tolist()
    out["kept_lasso"] = cols[las.get_support()].tolist()
    return out


def polynomial_study(X, y, g, cv, top_feats):
    """Lecture02 / Lecture08: polynomial and interaction features for a
    linear model, on the most informative inputs."""
    base = _p(("impute", SimpleImputer()), ("scale", StandardScaler()),
              ("model", Ridge(alpha=1.0)))
    poly = _p(("impute", SimpleImputer()), ("scale", StandardScaler()),
              ("poly", PolynomialFeatures(degree=2, include_bias=False)),
              ("scale2", StandardScaler()), ("model", Ridge(alpha=1.0)))
    return {"features": list(top_feats),
            "ridge_linear_all": _rmse_cv(base, X, y, g, cv)[0],
            "ridge_linear_top": _rmse_cv(base, X[top_feats], y, g, cv)[0],
            "ridge_quadratic_top": _rmse_cv(poly, X[top_feats], y, g, cv)[0]}


def column_transformer_study(X, y, g, cv):
    """Lecture04 / E03_geographicalspliting: treat column groups differently
    -- log-transform the (right-skewed) rate columns with a
    FunctionTransformer, scale everything."""
    rate_cols = [c for c in X.columns if c in ("q_mean_kg_s", "q_max_kg_s",
                                                "planned_mass_kg", "r_fill_est_m")]
    other = [c for c in X.columns if c not in rate_cols]
    ct = ColumnTransformer([
        ("rates", _p(("log", FunctionTransformer(np.log)), ("sc", StandardScaler())), rate_cols),
        ("rest", _p(("imp", SimpleImputer()), ("sc", StandardScaler())), other)])
    out = {"log_columns": rate_cols}
    for mn, m in (("knn", KNeighborsRegressor(n_neighbors=7, weights="distance")),
                  ("ridge", Ridge(alpha=1.0))):
        plain = _p(("imp", SimpleImputer()), ("sc", StandardScaler()), ("model", clone(m)))
        out[mn] = {"standard_only": _rmse_cv(plain, X, y, g, cv)[0],
                   "column_transformer_log_rates": _rmse_cv(
                       Pipeline([("ct", ct), ("model", clone(m))]), X, y, g, cv)[0]}
    return out


def tuning_strategies(X, y, g, cv, budget=12, seed=0):
    """Lecture06 / E03_TuningRandomforest: grid vs randomised vs Bayesian
    search for gradient boosting with (about) the same number of fits."""
    from scipy.stats import loguniform, randint
    pipe = _p(("impute", SimpleImputer()), ("scale", StandardScaler()),
              ("model", GradientBoostingRegressor(random_state=seed)))
    grid = {"model__n_estimators": [150, 400], "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1]}                 # 12 points
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error",
                          n_jobs=-1).fit(X, y, groups=g)
        out["grid"] = {"cv_rmse": float(-gs.best_score_), "n_candidates": 12,
                       "seconds": time.perf_counter() - t,
                       "best": {k: v.item() if hasattr(v, "item") else v
                                for k, v in gs.best_params_.items()}}
        t = time.perf_counter()
        rs = RandomizedSearchCV(pipe, {"model__n_estimators": randint(100, 500),
                                       "model__max_depth": randint(2, 6),
                                       "model__learning_rate": loguniform(0.01, 0.3)},
                                n_iter=budget, cv=cv, random_state=seed,
                                scoring="neg_root_mean_squared_error",
                                n_jobs=-1).fit(X, y, groups=g)
        out["random"] = {"cv_rmse": float(-rs.best_score_), "n_candidates": budget,
                         "seconds": time.perf_counter() - t,
                         "best": {k: v.item() if hasattr(v, "item") else v
                                  for k, v in rs.best_params_.items()}}
        try:
            from skopt import BayesSearchCV
            from skopt.space import Integer, Real
            t = time.perf_counter()
            bs = BayesSearchCV(pipe, {"model__n_estimators": Integer(100, 500),
                                      "model__max_depth": Integer(2, 5),
                                      "model__learning_rate": Real(0.01, 0.3, prior="log-uniform")},
                               n_iter=budget, cv=list(cv.split(X, y, g)),
                               random_state=seed,
                               scoring="neg_root_mean_squared_error", n_jobs=1).fit(X, y)
            out["bayesian"] = {"cv_rmse": float(-bs.best_score_), "n_candidates": budget,
                               "seconds": time.perf_counter() - t,
                               "best": {k: (v.item() if hasattr(v, "item") else v)
                                        for k, v in bs.best_params_.items()}}
        except Exception as exc:                          # pragma: no cover
            out["bayesian"] = {"status": "unavailable", "reason": repr(exc)}
    return out


def normal_equation_study(X, y):
    """Lecture01 / mathematics_regression.ipynb: OLS by the normal equation
    theta = (X^T X)^-1 X^T y, ridge by (X^T X + alpha I)^-1 X^T y, and batch
    gradient descent on the MSE cost -- compared with scikit-learn."""
    Z = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))
    A = np.column_stack([np.ones(len(Z)), Z])
    theta = np.linalg.lstsq(A.T @ A, A.T @ y, rcond=None)[0]
    sk = LinearRegression().fit(Z, y)
    alpha = 1.0
    I = np.eye(A.shape[1]); I[0, 0] = 0.0          # intercept not penalised
    theta_r = np.linalg.solve(A.T @ A + alpha * I, A.T @ y)
    skr = Ridge(alpha=alpha).fit(Z, y)
    # gradient descent on J = 1/(2m) ||A theta - y||^2
    th = np.zeros(A.shape[1]); m = len(y)
    lr = 1.0 / np.linalg.eigvalsh(A.T @ A / m).max()
    hist = []
    for it in range(20000):
        grad = A.T @ (A @ th - y) / m
        th -= lr * grad
        if it % 500 == 0:
            hist.append(float(0.5 * np.mean((A @ th - y) ** 2)))
    return {"max_abs_diff_normal_eq_vs_sklearn_ols": float(np.max(np.abs(
                theta - np.r_[sk.intercept_, sk.coef_]))),
            "max_abs_diff_ridge_closed_form_vs_sklearn": float(np.max(np.abs(
                theta_r - np.r_[skr.intercept_, skr.coef_]))),
            "gd_cost_history_every_500": hist,
            "gd_final_cost": hist[-1],
            "normal_eq_cost": float(0.5 * np.mean((A @ theta - y) ** 2)),
            "condition_number_XtX": float(np.linalg.cond(A.T @ A))}


def split_comparison(df, feats, target, est, masks, seed=0):
    """Lecture08 / E03_geographicalspliting: score the same model after a
    random ROW split and after a grouped (realisation) split of the same
    non-test data.  The gap is the optimism a leaky split would report."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    d = df[masks["train_fit"]]
    X, y, g = d[feats], d[target].to_numpy(), d["realisation_id"].to_numpy()
    rows = []
    for s in range(5):
        tr, va = train_test_split(np.arange(len(d)), test_size=0.25, random_state=seed + s)
        e = clone(est).fit(X.iloc[tr], y[tr])
        r_row = np.sqrt(mean_squared_error(y[va], e.predict(X.iloc[va])))
        tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed + s).split(X, y, g))
        e = clone(est).fit(X.iloc[tr], y[tr])
        r_grp = np.sqrt(mean_squared_error(y[va], e.predict(X.iloc[va])))
        rows.append((r_row, r_grp))
    a = np.array(rows)
    return {"random_row_split_rmse": float(a[:, 0].mean()),
            "grouped_split_rmse": float(a[:, 1].mean()),
            "optimism_pct": float(100 * (1 - a[:, 0].mean() / a[:, 1].mean())),
            "repeats": 5}


def boosting_early_stopping(X, y, g, seed=0):
    """E02_Ensembles / Lecture06: XGBoost training and validation error at
    every boosting round, with early stopping on a grouped validation split."""
    try:
        from xgboost import XGBRegressor
    except Exception:                                   # pragma: no cover
        return {"status": "xgboost unavailable"}
    from sklearn.model_selection import GroupShuffleSplit
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, g))
    pre = _p(("impute", SimpleImputer()), ("scale", StandardScaler())).fit(X.iloc[tr])
    Xt, Xv = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
    m = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=4,
                     early_stopping_rounds=50, eval_metric="rmse",
                     random_state=seed, n_jobs=1, verbosity=0)
    m.fit(Xt, y[tr], eval_set=[(Xt, y[tr]), (Xv, y[va])], verbose=False)
    ev = m.evals_result()
    return {"best_iteration": int(m.best_iteration),
            "train_rmse_curve": ev["validation_0"]["rmse"],
            "valid_rmse_curve": ev["validation_1"]["rmse"],
            "valid_rmse_at_best": float(ev["validation_1"]["rmse"][m.best_iteration])}


def ensemble_study(X, y, g, cv, fitted: dict, top=3):
    """Lecture06 / E02_Ensembles: combine the best tuned families by voting
    (averaging) and stacking (ridge meta-model), and bag a decision tree.
    Every candidate is re-scored here with the same folds and on the same
    target scale, so the comparison is like for like."""
    ranked = sorted((k for k in fitted if k != "mean"),
                    key=lambda k: fitted[k].cv_score_rmse)[:top]
    base = [(k, _unwrap(fitted[k].estimator)) for k in ranked]
    inner = KFold(n_splits=4, shuffle=True, random_state=0)
    out = {"members": ranked,
           "members_cv_rmse": {k: _rmse_cv(e, X, y, g, cv)[0] for k, e in base}}
    out["voting_cv_rmse"] = _rmse_cv(VotingRegressor(base), X, y, g, cv)[0]
    out["stacking_cv_rmse"] = _rmse_cv(StackingRegressor(base, final_estimator=Ridge(1.0),
                                                         cv=inner), X, y, g, cv)[0]
    out["single_tree_cv_rmse"] = _rmse_cv(_p(("imp", SimpleImputer()),
                                             ("model", DecisionTreeRegressor(max_depth=8, random_state=0))),
                                          X, y, g, cv)[0]
    out["bagged_tree_cv_rmse"] = _rmse_cv(_p(("imp", SimpleImputer()),
                                             ("model", BaggingRegressor(DecisionTreeRegressor(max_depth=8),
                                                                        n_estimators=100, random_state=0))),
                                          X, y, g, cv)[0]
    # a diverse ensemble: best linear + best tree ensemble + SVR
    div = [k for k in ("elasticnet", "gradient_boosting", "svr") if k in fitted]
    if len(div) >= 2:
        dbase = [(k, _unwrap(fitted[k].estimator)) for k in div]
        out["diverse_members"] = div
        out["diverse_members_cv_rmse"] = {k: _rmse_cv(e, X, y, g, cv)[0] for k, e in dbase}
        out["diverse_voting_cv_rmse"] = _rmse_cv(VotingRegressor(dbase), X, y, g, cv)[0]
        out["diverse_stacking_cv_rmse"] = _rmse_cv(StackingRegressor(
            dbase, final_estimator=Ridge(1.0), cv=inner), X, y, g, cv)[0]
    return out


def _unwrap(est):
    """Plain pipeline of a fitted surrogate (drops the log-target wrapper so
    ensembles combine models on the same scale)."""
    from sklearn.compose import TransformedTargetRegressor
    if isinstance(est, TransformedTargetRegressor):
        return clone(est.regressor)
    return clone(est)


# --------------------------------------------------------------------------
def run_all(df, feats, target, masks, fitted, best, n_splits=4, seed=0,
            log=print) -> dict:
    """Run every study for one target (the pressure-buildup surrogate)."""
    tr = masks["train"]
    X = df.loc[tr, feats].reset_index(drop=True)
    y = df.loc[tr, target].to_numpy(float)
    g = df.loc[tr, "realisation_id"].to_numpy()
    ly = np.log(y) if best.log_target else y
    cv = grouped_cv(n_splits)
    gb = _p(("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                learning_rate=0.05, random_state=seed)))
    out = {"target": target,
           "units_of_all_rmse_values": "RMSE of log(target)" if best.log_target else "target units",
           "n_train_rows": int(len(y)), "n_train_realisations": int(len(np.unique(g)))}
    steps = [
        ("learning_curve", lambda: learning_curve_study(X, ly, g, cv, gb)),
        ("scaler_comparison", lambda: scaler_comparison(X, ly, g, cv)),
        ("missing_descriptors", lambda: missing_descriptors(X, ly, g, cv, gb, {
            "permeability": ["log10_k_mD", "log10_kh", "log10_injectivity",
                             "log10_q_ref", "log10_q_mult_mean", "q_mult_max"],
            "Dykstra-Parsons": ["V_DP", "V_DP_layers"],
            "CO2 end-point kr": ["krg0", "log10_mobility_ratio"],
            "Corey exponents": ["n_g", "n_a"]})),
        ("outliers", lambda: outlier_study(df, target, feats, clone(best.estimator), masks)),
        ("collinearity", lambda: collinearity_study(X, ly, g, cv)),
        ("feature_selection", lambda: feature_selection_study(X, ly, g, cv)),
        ("polynomial", lambda: polynomial_study(X, ly, g, cv, [
            "log10_q_mult_mean", "log10_kh", "log10_pv", "q_front_load",
            "r_fill_over_re", "V_DP_layers", "log10_mobility_ratio"])),
        ("column_transformer", lambda: column_transformer_study(X, ly, g, cv)),
        ("tuning_strategies", lambda: tuning_strategies(X, ly, g, cv, seed=seed)),
        ("normal_equation", lambda: normal_equation_study(X, ly)),
        ("split_comparison", lambda: split_comparison(df, feats, target,
                                                      clone(best.estimator), masks, seed)),
        ("boosting_early_stopping", lambda: boosting_early_stopping(X, ly, g, seed)),
        ("ensembles", lambda: ensemble_study(X, ly, g, cv, fitted)),
    ]
    for name, fn in steps:
        t = time.perf_counter()
        try:
            out[name] = fn()
            out[name + "_seconds"] = time.perf_counter() - t
            log(f"    study {name:24s} done ({out[name + '_seconds']:5.1f} s)")
        except Exception as exc:                             # pragma: no cover
            out[name] = {"status": "failed", "error": repr(exc)}
            log(f"    study {name:24s} FAILED: {exc!r}")
    return out
