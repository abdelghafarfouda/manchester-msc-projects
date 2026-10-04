"""Controlled experiments on the *development* reservoirs.

Everything in this module uses only the development data set (the 220
reservoirs of the study run).  The final, untouched test reservoirs and the
distribution-shift reservoirs are generated and scored separately
(:mod:`final_eval`), after every choice made here has been fixed.

Ablation (:func:`nested_cv`)
----------------------------
Nested, reservoir-grouped cross-validation: an outer ``GroupKFold`` (shuffled,
fixed seed) holds out whole reservoirs; inside each outer training part the
usual randomised search with an inner ``GroupKFold`` tunes and selects the
model family, exactly as the production pipeline does.  Every development
case therefore receives one out-of-fold prediction from a model that never
saw its reservoir, for every variant, with identical folds, identical search
budgets and identical seeds -- so the variants can be compared case by case
and with a reservoir-level paired bootstrap.

Interval selection (:func:`select_interval_method`)
---------------------------------------------------
The interval constructions of :mod:`intervals` are compared on the
out-of-fold residuals by repeatedly splitting the development reservoirs
into a part that fits the difficulty model, a calibration part of the same
size as the production calibration set, and an evaluation part.  The method
chosen here is the one used in the final evaluation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluation import paired_bootstrap, point_metrics
from .models import fit_and_select

GROUP = "realisation_id"


def outer_folds(groups, n_splits: int = 5, seed: int = 0):
    """Shuffled GroupKFold over reservoirs (deterministic for a seed)."""
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    fold_of = {u: i % n_splits for i, u in enumerate(perm)}
    f = np.array([fold_of[u] for u in g])
    for k in range(n_splits):
        yield np.flatnonzero(f != k), np.flatnonzero(f == k)


def nested_cv(df: pd.DataFrame, target: str, *, features=None, mode="log",
              outer_splits=5, inner_splits=4, n_iter=20, families=None,
              seed=0, n_jobs=-1, verbose=False) -> pd.DataFrame:
    """Out-of-fold predictions of one variant.

    ``mode``: ``"log"`` (log target), ``"linear"``, ``"hybrid"`` (ROM times a
    learned correction) or ``"rom"`` (the analytical ROM alone, no learning).
    """
    y = df[target].to_numpy(float)
    g = df[GROUP].to_numpy()
    pred = np.full(len(df), np.nan)
    fam = np.empty(len(df), dtype=object)
    t0 = time.perf_counter()
    for k, (tr, te) in enumerate(outer_folds(g, outer_splits, seed)):
        if mode == "rom":
            pred[te] = df["rom_dp_bh_max_Pa"].to_numpy(float)[te] / 1e6
            fam[te] = "rom"
            continue
        best, _, _ = fit_and_select(
            df.iloc[tr], y[tr], g[tr], list(features), target,
            n_splits=inner_splits, n_iter=n_iter, random_state=seed,
            log_target=(mode == "log"), hybrid=(mode == "hybrid"),
            families=families, verbose=False, n_jobs=n_jobs)
        pred[te] = best.predict(df.iloc[te])
        fam[te] = best.name
        if verbose:
            print(f"      outer fold {k}: selected {best.name} "
                  f"({time.perf_counter() - t0:.0f} s)")
    return pd.DataFrame({"scenario_id": df["scenario_id"].to_numpy(),
                         GROUP: g, "y": y, "pred": pred, "family": fam})


def compare(oof: dict, reference: str, *, n_boot=2000, seed=0) -> dict:
    """Overall metrics of every variant and the reservoir-bootstrap
    difference of each variant against ``reference``."""
    ref = oof[reference]
    out = {}
    for name, o in oof.items():
        m = point_metrics(o["y"], o["pred"])
        m["selected_families"] = o["family"].value_counts().to_dict()
        if name != reference:
            m["vs_" + reference] = paired_bootstrap(
                o[GROUP], o["y"], ref["pred"], o["pred"], n_boot=n_boot, seed=seed)
        out[name] = m
    return out


# --------------------------------------------------------------------------
def select_interval_method(oof: pd.DataFrame, X: pd.DataFrame, *, alpha=0.10,
                           n_cal_reservoirs=41, n_repeats=200, seed=0,
                           log_space=True) -> dict:
    """Compare interval constructions on out-of-fold residuals.

    In each repeat the development reservoirs are split at random into a
    difficulty-model part (half of the rest), a calibration part of
    ``n_cal_reservoirs`` reservoirs, and an evaluation part.  Coverage (case
    and whole-reservoir) and relative width are averaged over repeats.
    """
    from .intervals import METHODS, IntervalModel, fit_difficulty_model

    class _Fixed:                       # surrogate whose predictions are the OOF ones
        log_target = log_space

        def __init__(self, pred):
            self.pred = pd.Series(pred, index=X.index)

        def predict(self, Xs):
            return self.pred.loc[Xs.index].to_numpy(float)

    rng = np.random.default_rng(seed)
    g = oof[GROUP].to_numpy()
    uniq = np.unique(g)
    sur = _Fixed(oof["pred"].to_numpy(float))
    y = oof["y"].to_numpy(float)
    if log_space:
        abs_r = np.abs(np.log(y) - np.log(oof["pred"].to_numpy(float)))
    else:
        abs_r = np.abs(y - oof["pred"].to_numpy(float))
    rows = []
    for rep in range(n_repeats):
        perm = rng.permutation(uniq)
        cal = perm[:n_cal_reservoirs]
        rest = perm[n_cal_reservoirs:]
        dif, ev = rest[: len(rest) // 2], rest[len(rest) // 2:]
        m_cal, m_dif, m_ev = (np.isin(g, s) for s in (cal, dif, ev))
        diff_model = fit_difficulty_model(X.loc[m_dif], abs_r[m_dif],
                                          random_state=rep)
        for meth in METHODS:
            im = IntervalModel(sur, meth, alpha, log_space=log_space,
                               difficulty=diff_model if meth == "adaptive_conformal" else None)
            im.calibrate(X.loc[m_cal], y[m_cal], g[m_cal])
            ev_ = im.evaluate(X.loc[m_ev], y[m_ev], g[m_ev])
            rows.append({"repeat": rep, "method": meth,
                         "case_coverage": ev_["case_coverage"],
                         "reservoir_all_covered": ev_["reservoir_all_covered"],
                         "mean_relative_width": ev_["mean_relative_width"]})
    d = pd.DataFrame(rows)
    agg = d.groupby("method").agg(
        case_coverage_mean=("case_coverage", "mean"),
        case_coverage_p05=("case_coverage", lambda s: s.quantile(0.05)),
        reservoir_all_covered_mean=("reservoir_all_covered", "mean"),
        reservoir_all_covered_p05=("reservoir_all_covered", lambda s: s.quantile(0.05)),
        mean_relative_width=("mean_relative_width", "mean")).reset_index()
    return {"alpha": alpha, "n_repeats": n_repeats,
            "n_calibration_reservoirs": n_cal_reservoirs,
            "table": agg.to_dict("records")}


# --------------------------------------------------------------------------
def rebuild_inputs(cfg, scenarios: pd.DataFrame, realisations=None) -> pd.DataFrame:
    """Recompute every pre-simulation input column of a scenario table through
    the one feature path (:func:`features.raw_inputs`) and engineer the
    features.  Simulator outputs (the targets) are kept from the table.  Used
    to add the realised-rock and ROM columns to a table written before they
    existed; the simulations themselves are not repeated."""
    from .features import engineer, raw_inputs
    from .scenarios import sample_realisations
    reals = realisations or {r.realisation_id: r for r in sample_realisations(cfg)}
    qcols = [f"q{i + 1}_kg_s" for i in range(cfg.schedule.n_periods)]
    rows = []
    for _, row in scenarios.iterrows():
        r = reals[int(row["realisation_id"])]
        rows.append(raw_inputs(cfg, r, row[qcols].to_numpy(float)))
    raw = pd.DataFrame(rows, index=scenarios.index)
    keep = scenarios.drop(columns=[c for c in raw.columns if c in scenarios.columns])
    return engineer(pd.concat([keep, raw], axis=1))


#: pandas' default ("high") float parser is not round-trip exact (about one
#: unit in the last place); it is named explicitly so the experiment inputs
#: are always parsed the same way
CSV_FLOAT_PRECISION = "high"


def load_dev_table(cfg) -> pd.DataFrame:
    """The development table every experiment uses: the inputs are always
    rebuilt from ``scenarios.csv`` (the simulator outputs) through the one
    feature path, never read from ``scenarios_features.csv``.  The two
    sources agree only to about 2e-13 relative, which is enough to tip a
    near-tied model choice (``docs/TECHNICAL_REPORT.md`` §8)."""
    from pathlib import Path
    from .pipeline import deduplicate
    sc = pd.read_csv(Path(cfg.paths.data) / "scenarios.csv",
                     float_precision=CSV_FLOAT_PRECISION)
    df, _ = deduplicate(rebuild_inputs(cfg, sc))
    return df.reset_index(drop=True)


def input_table_sha256(df: pd.DataFrame, columns) -> str:
    """Hash of the exact numbers an experiment trains on (inputs, targets,
    groups), so a re-run can show it used identical inputs."""
    import hashlib
    a = np.ascontiguousarray(df[list(columns)].to_numpy(float))
    return hashlib.sha256(a.tobytes()).hexdigest()
