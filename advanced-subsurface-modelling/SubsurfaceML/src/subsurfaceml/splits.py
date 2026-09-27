"""Grouped and chronological splitting.

The grouping unit is the **reservoir realisation**.  Every row that derives
from a realisation -- all of its injection schedules, all of its report times,
and any coarse/fine variant of it -- must live in the same partition, or the
model is scored on a reservoir it has effectively already seen.

This is the geographical-splitting idea of ``Lecture08.ipynb`` and
``E03_geographicalspliting.ipynb``: that notebook splits a geochemical survey
by location so that nearby, correlated samples cannot straddle the
train/test boundary.  The realisation ID plays the role of the spatial block
here, and ``GroupKFold`` applies the same idea inside K-fold cross-validation
(``Lecture05.ipynb``).

Three partitions are produced:

``train``  model fitting and hyper-parameter search (with GroupKFold inside)
``calib``  held out from fitting; used *only* to calibrate the empirical error bands
``test``   touched once, at the very end, for the reported numbers

Because whole realisations are held out, the test set contains **both** unseen
reservoirs and unseen injection schedules.  A secondary evaluation -- unseen
schedule on a *seen* reservoir -- is provided by
:func:`seen_reservoir_new_schedule_mask` for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


GROUP_COL = "realisation_id"


def split_by_realisation(df: pd.DataFrame, *, test_fraction: float = 0.25,
                         calib_fraction: float = 0.20,
                         random_state: int = 0,
                         group_col: str = GROUP_COL) -> dict:
    """Partition rows into train / calib / test by whole realisations.

    ``calib_fraction`` is a fraction of the **non-test** realisations.
    Returns a dict of boolean masks plus the realisation ids in each part.
    """
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(uniq)

    n_test = max(1, int(round(test_fraction * len(uniq))))
    test_ids = np.sort(perm[:n_test])
    rest = perm[n_test:]
    n_cal = max(1, int(round(calib_fraction * len(rest))))
    calib_ids = np.sort(rest[:n_cal])
    train_ids = np.sort(rest[n_cal:])

    masks = {
        "train": np.isin(groups, train_ids),
        "calib": np.isin(groups, calib_ids),
        "test": np.isin(groups, test_ids),
    }
    assert masks["train"].sum() + masks["calib"].sum() + masks["test"].sum() == len(df)
    assert not (set(train_ids) & set(calib_ids)) and \
        not (set(train_ids) & set(test_ids)) and not (set(calib_ids) & set(test_ids))
    return {"masks": masks,
            "train_ids": train_ids, "calib_ids": calib_ids,
            "test_ids": test_ids,
            "n_realisations": int(len(uniq)),
            "group_col": group_col}


def grouped_cv(n_splits: int = 4):
    """GroupKFold for hyper-parameter search inside the training partition."""
    return GroupKFold(n_splits=n_splits)


def seen_reservoir_new_schedule_mask(df: pd.DataFrame, train_mask: np.ndarray,
                                     schedule_col: str = "schedule_id",
                                     group_col: str = GROUP_COL) -> np.ndarray:
    """Rows whose realisation is in ``train`` but whose schedule is the
    highest-numbered one for that realisation -- i.e. a *new schedule on a
    seen reservoir*.  Used only as a contrast to the main (new reservoir)
    test set; these rows are excluded from fitting by the caller."""
    out = np.zeros(len(df), bool)
    for g, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        rows = df.loc[idx]
        if not train_mask[df.index.get_indexer(idx)].all():
            continue
        keep = idx[rows[schedule_col].to_numpy().argmax()]
        out[df.index.get_loc(keep)] = True
    return out


def check_no_leakage(df: pd.DataFrame, masks: dict,
                     group_col: str = GROUP_COL) -> dict:
    """Assert the partitions share no realisation.  Returns the overlap
    counts so a test can assert they are zero."""
    ids = {k: set(df.loc[m, group_col].unique()) for k, m in masks.items()}
    overlaps = {
        "train_calib": len(ids["train"] & ids["calib"]),
        "train_test": len(ids["train"] & ids["test"]),
        "calib_test": len(ids["calib"] & ids["test"]),
    }
    return {"overlaps": overlaps,
            "sizes": {k: len(v) for k, v in ids.items()},
            "rows": {k: int(m.sum()) for k, m in masks.items()}}
