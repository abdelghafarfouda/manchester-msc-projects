"""Defect (d): probability calibration allowed rows of one realisation on
both sides of its internal folds.  Calibration (not in the course material)
was removed; every score used for model choice or threshold choice is now an
out-of-fold prediction from GroupKFold on realisation_id.  This test builds a
task that can only be solved by memorising realisations: grouped out-of-fold
scores must then be uninformative, whereas ungrouped folds would look
excellent."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

from subsurfaceml.classify import _pipe, classifier_families, run_binary
from subsurfaceml.splits import grouped_cv


def _memorisation_task(n_groups=40, per=6, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    label_of_group = rng.integers(0, 2, n_groups)
    code = rng.normal(size=(n_groups, 3))          # a per-realisation signature
    X = code[g] + 0.01 * rng.normal(size=(len(g), 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["realisation_id"] = g
    return df, label_of_group[g]


def test_grouped_out_of_fold_scores_cannot_memorise_realisations():
    df, y = _memorisation_task()
    est = _pipe(classifier_families()["knn"])
    s_grp = cross_val_predict(est, df[["a", "b", "c"]], y, groups=df.realisation_id,
                              cv=grouped_cv(4), method="predict_proba")[:, 1]
    s_leak = cross_val_predict(est, df[["a", "b", "c"]], y,
                               cv=KFold(4, shuffle=True, random_state=0),
                               method="predict_proba")[:, 1]
    assert roc_auc_score(y, s_leak) > 0.95          # leaky folds look perfect
    assert roc_auc_score(y, s_grp) < 0.75           # grouped folds do not


def test_run_binary_touches_the_test_set_only_once():
    df, y = _memorisation_task(n_groups=48)
    g = df.realisation_id.to_numpy()
    masks = {"train_fit": g < 36, "test": g >= 36}
    out = run_binary(df, ["a", "b", "c"], y, masks, n_splits=4)
    fam = out["families"]
    # grouped out-of-fold ranking scores of a memorisation task are weak
    assert max(v["oof_roc_auc"] for v in fam.values()) < 0.8
    assert out["n_test"] == int((g >= 36).sum())
    assert "calibrat" not in " ".join(out.keys())
