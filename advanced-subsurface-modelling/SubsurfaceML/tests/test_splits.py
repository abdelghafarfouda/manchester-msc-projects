"""Data-splitting tests.

These are the tests that stop the headline numbers from being a lie: if a
reservoir realisation can appear in both training and test, every held-out
metric in the report is optimistic and nothing else in the project matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subsurfaceml.splits import (check_no_leakage, grouped_cv,
                                 split_by_realisation)


def _frame(n_real=40, n_sched=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_real):
        base = rng.normal()
        for s in range(n_sched):
            rows.append({"realisation_id": r, "schedule_id": s,
                         "scenario_id": f"R{r:04d}_S{s:+03d}",
                         "x1": base + 0.01 * rng.normal(),
                         "x2": rng.normal(),
                         "y": base * 2 + 0.05 * rng.normal()})
    return pd.DataFrame(rows)


def test_no_realisation_appears_in_two_partitions():
    df = _frame()
    sp = split_by_realisation(df, test_fraction=0.25, calib_fraction=0.2,
                              random_state=0)
    chk = check_no_leakage(df, sp["masks"])
    assert chk["overlaps"] == {"train_calib": 0, "train_test": 0, "calib_test": 0}


def test_partitions_cover_every_row_exactly_once():
    df = _frame()
    m = split_by_realisation(df, random_state=1)["masks"]
    total = m["train"].astype(int) + m["calib"].astype(int) + m["test"].astype(int)
    assert np.all(total == 1)


def test_all_schedules_of_a_realisation_stay_together():
    df = _frame(n_sched=4)
    m = split_by_realisation(df, random_state=2)["masks"]
    for _, g in df.groupby("realisation_id"):
        idx = g.index.to_numpy()
        for name in ("train", "calib", "test"):
            v = m[name][idx]
            assert v.all() or (~v).all(), "a realisation was split"


@pytest.mark.parametrize("seed", range(5))
def test_split_is_deterministic_given_the_seed(seed):
    df = _frame()
    a = split_by_realisation(df, random_state=seed)
    b = split_by_realisation(df, random_state=seed)
    assert np.array_equal(a["test_ids"], b["test_ids"])


def test_grouped_cv_folds_never_split_a_group():
    df = _frame(n_real=24)
    cv = grouped_cv(4)
    g = df["realisation_id"].to_numpy()
    for tr, te in cv.split(df[["x1", "x2"]], df["y"], groups=g):
        assert not set(g[tr]) & set(g[te])


def test_a_naive_random_split_would_look_better_than_the_grouped_one():
    """Guards the reason the grouped split exists.

    Schedules of the same realisation share almost all of their features, so a
    row-wise random split lets the model see near-copies of its test rows.  The
    grouped split must score WORSE -- if it does not, the leakage this project
    is guarding against is not being exercised and the test is not doing its
    job.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    df = _frame(n_real=60, n_sched=4, seed=7)
    X, y = df[["x1", "x2"]], df["y"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    naive = RandomForestRegressor(n_estimators=120, random_state=0).fit(Xtr, ytr)
    naive_rmse = float(np.sqrt(np.mean((yte - naive.predict(Xte)) ** 2)))

    m = split_by_realisation(df, test_fraction=0.25, calib_fraction=0.0,
                             random_state=0)["masks"]
    grouped = RandomForestRegressor(n_estimators=120, random_state=0).fit(
        X[m["train"]], y[m["train"]])
    grouped_rmse = float(np.sqrt(np.mean(
        (y[m["test"]] - grouped.predict(X[m["test"]])) ** 2)))
    assert grouped_rmse > naive_rmse


def test_duplicate_inputs_are_dropped_before_learning():
    """Data cleaning (Lecture01 / E01_EDA drop_duplicates): two schedules
    with identical inputs on one reservoir must not both enter training."""
    from subsurfaceml.features import FEATURES
    from subsurfaceml.pipeline import deduplicate
    df = pd.DataFrame({f: [1.0, 1.0, 2.0] for f in FEATURES})
    df["q1_kg_s"] = [1.0, 1.0, 1.0]
    df["realisation_id"] = [0, 0, 0]
    df["scenario_id"] = ["a", "b", "c"]
    out, n = deduplicate(df)
    assert n == 1 and list(out.scenario_id) == ["a", "c"]
