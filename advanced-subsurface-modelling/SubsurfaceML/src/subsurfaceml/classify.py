"""Pressure-limit screening as a classification problem.

Engineering question: *before simulating, will this schedule on this reservoir
exceed the stated bottom-hole pressure limit?*  A missed exceedance is worse
than a false alarm, so the decision threshold is chosen for high recall.

Course methods used (``Data Science and machine learning`` folder):

* logistic regression, KNN, SVM, decision tree, random forest, XGBoost
  (``Lecture03-06``, ``E02_logisticRegression``, ``E03_ROC_AUC``,
  ``E02_Ensembles``);
* confusion matrix, precision, recall, F1 (``Lecture03``, ``E02-Threshold``);
* ROC-AUC and the precision-recall trade-off with threshold adjustment
  (``Lecture04``, ``E02-Threshold``, ``E03_ROC_AUC``);
* class weights, random over/under-sampling and SMOTE applied **inside the
  training folds only** (``Lecture06`` "Use pipelines for handling
  imbalanced data", ``E04_Imbalance``);
* multiclass classification with a multiclass confusion matrix
  (``E02_Multiclassification``, ``multiclassification.ipynb``) for a
  green / amber / red operating-margin screen.

Every score used for model choice or threshold choice is an **out-of-fold**
prediction from ``GroupKFold`` on the training realisations; the test
realisations are scored once, at the end.  (Probability *calibration* is not
taught in the supplied material and is not used; the scores are used for
ranking and thresholding only.)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .splits import grouped_cv

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                                  # pragma: no cover
    HAS_XGB = False


def _pipe(model, sampler=None):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps + [("resample", sampler), ("model", model)])
    return Pipeline(steps + [("model", model)])


def classifier_families(rs=0) -> dict:
    m = {"logistic": LogisticRegression(max_iter=5000, C=1.0),
         "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
         "svc": SVC(kernel="rbf", C=10.0, gamma="scale"),
         "tree": DecisionTreeClassifier(max_depth=5, random_state=rs),
         "random_forest": RandomForestClassifier(n_estimators=300,
                                                 random_state=rs, n_jobs=1)}
    if HAS_XGB:
        m["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.1, random_state=rs,
                                     n_jobs=1, verbosity=0)
    return m


def _scores(est, X, y, groups, cv):
    """Out-of-fold ranking scores (probability or decision function)."""
    method = ("predict_proba" if hasattr(est, "predict_proba")
              and not isinstance(est[-1], SVC) else "decision_function")
    s = cross_val_predict(est, X, y, groups=groups, cv=cv, method=method,
                          n_jobs=-1)
    return s[:, 1] if s.ndim == 2 else s


def _score_test(est, X):
    if isinstance(est[-1], SVC):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def _cls_metrics(y, pred, score=None) -> dict:
    out = {"accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if score is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["average_precision"] = float(average_precision_score(y, score))
    return out


def threshold_for_recall(y, score, target_recall=0.95) -> float:
    """Highest threshold whose out-of-fold recall reaches ``target_recall``
    (``E02-Threshold``: move the threshold along the precision-recall
    curve instead of accepting 0.5)."""
    p, r, t = precision_recall_curve(y, score)
    ok = np.where(r[:-1] >= target_recall)[0]
    return float(t[ok[-1]]) if ok.size else float(np.min(score))


def run_binary(df, feats, y, masks, *, rs=0, n_splits=4, target_recall=0.95,
               regression_pred=None, dp_limit=None) -> dict:
    """Model comparison, imbalance study, threshold choice and final test."""
    tr, te = masks["train_fit"], masks["test"]
    Xtr, ytr = df.loc[tr, feats], y[tr]
    gtr = df.loc[tr, "realisation_id"].to_numpy()
    Xte, yte = df.loc[te, feats], y[te]
    cv = grouped_cv(n_splits)
    out = {"positive_rate_train": float(ytr.mean()),
           "positive_rate_test": float(yte.mean()),
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "target_recall": target_recall, "families": {}}

    # ---- 1. family comparison on out-of-fold scores -----------------------
    oof = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in classifier_families(rs).items():
            s = _scores(_pipe(model), Xtr, ytr, gtr, cv)
            oof[name] = s
            out["families"][name] = {
                "oof_roc_auc": float(roc_auc_score(ytr, s)),
                "oof_average_precision": float(average_precision_score(ytr, s))}
    best = max(out["families"], key=lambda k: out["families"][k]["oof_average_precision"])
    out["selected"] = best

    # ---- 2. imbalance handling, inside the folds ----------------------------
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    variants = {
        "none": (LogisticRegression(max_iter=5000), None),
        "class_weight_balanced": (LogisticRegression(max_iter=5000,
                                                     class_weight="balanced"), None),
        "random_oversampling": (LogisticRegression(max_iter=5000),
                                RandomOverSampler(random_state=rs)),
        "random_undersampling": (LogisticRegression(max_iter=5000),
                                 RandomUnderSampler(random_state=rs)),
        "smote": (LogisticRegression(max_iter=5000), SMOTE(random_state=rs)),
    }
    imb = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (model, sampler) in variants.items():
            est = _pipe(model, sampler)
            s = _scores(est, Xtr, ytr, gtr, cv)
            pred = (s >= 0.5).astype(int)
            m = _cls_metrics(ytr, pred, s)
            imb[name] = {k: m[k] for k in ("precision", "recall", "f1",
                                           "roc_auc", "average_precision")}
    out["imbalance_study_logistic_oof"] = imb

    # ---- 3. threshold from out-of-fold scores, then ONE test evaluation ----
    s_best = oof[best]
    thr = threshold_for_recall(ytr, s_best, target_recall)
    default_thr = 0.0 if best == "svc" else 0.5
    final = _pipe(classifier_families(rs)[best]).fit(Xtr, ytr)
    s_te = _score_test(final, Xte)
    out["threshold_default"] = default_thr
    out["threshold_selected"] = thr
    out["oof_at_selected_threshold"] = _cls_metrics(ytr, (s_best >= thr).astype(int))
    out["test_default_threshold"] = _cls_metrics(yte, (s_te >= default_thr).astype(int), s_te)
    out["test_selected_threshold"] = _cls_metrics(yte, (s_te >= thr).astype(int), s_te)
    fpr, tpr, _ = roc_curve(yte, s_te)
    pr, rc, _ = precision_recall_curve(yte, s_te)
    out["_curves"] = {"fpr": fpr, "tpr": tpr, "precision": pr, "recall": rc,
                      "oof_scores": s_best, "y_train": ytr}
    out["_model"] = final
    # ---- 4. the alternative: threshold the regression surrogate ------------
    if regression_pred is not None and dp_limit is not None:
        pred = (np.asarray(regression_pred) > dp_limit).astype(int)
        out["test_regression_surrogate_thresholded"] = _cls_metrics(yte, pred)
    return out


def traffic_light(dp, dp_limit, warn_frac):
    """0 = green (below ``warn_frac`` of the allowed buildup), 1 = amber,
    2 = red (limit exceeded)."""
    dp = np.asarray(dp, float)
    return np.where(dp > dp_limit, 2, np.where(dp > warn_frac * dp_limit, 1, 0))


def run_multiclass(df, feats, y3, masks, *, rs=0, n_splits=4,
                   regression_pred=None, dp_limit=None, warn_frac=None) -> dict:
    """Green/amber/red screen: one-vs-rest vs multinomial logistic, KNN and
    random forest; macro-F1 on out-of-fold predictions, one test score."""
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    tr, te = masks["train_fit"], masks["test"]
    Xtr, ytr = df.loc[tr, feats], y3[tr]
    gtr = df.loc[tr, "realisation_id"].to_numpy()
    Xte, yte = df.loc[te, feats], y3[te]
    cv = grouped_cv(n_splits)
    models = {
        "logistic_multinomial": LogisticRegression(max_iter=5000),
        "logistic_ovr": OneVsRestClassifier(LogisticRegression(max_iter=5000)),
        "logistic_ovo": OneVsOneClassifier(LogisticRegression(max_iter=5000)),
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=rs,
                                                n_jobs=1, class_weight="balanced"),
    }
    res = {"class_counts_train": np.bincount(ytr, minlength=3).tolist(),
           "class_counts_test": np.bincount(yte, minlength=3).tolist(),
           "labels": ["green", "amber", "red"], "families": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, m in models.items():
            pred = cross_val_predict(_pipe(m), Xtr, ytr, groups=gtr, cv=cv, n_jobs=-1)
            res["families"][name] = {"oof_macro_f1": float(f1_score(ytr, pred, average="macro")),
                                     "oof_accuracy": float(accuracy_score(ytr, pred))}
    best = max(res["families"], key=lambda k: res["families"][k]["oof_macro_f1"])
    final = _pipe(models[best]).fit(Xtr, ytr)
    p = final.predict(Xte)
    res["selected"] = best
    res["test"] = {"macro_f1": float(f1_score(yte, p, average="macro")),
                   "accuracy": float(accuracy_score(yte, p)),
                   "per_class_recall": recall_score(yte, p, average=None,
                                                    labels=[0, 1, 2], zero_division=0).tolist(),
                   "confusion_matrix": confusion_matrix(yte, p, labels=[0, 1, 2]).tolist()}
    if regression_pred is not None:
        pr = traffic_light(regression_pred, dp_limit, warn_frac)
        res["test_regression_surrogate_banded"] = {
            "macro_f1": float(f1_score(yte, pr, average="macro")),
            "accuracy": float(accuracy_score(yte, pr)),
            "confusion_matrix": confusion_matrix(yte, pr, labels=[0, 1, 2]).tolist()}
    res["_model"] = final
    return res
