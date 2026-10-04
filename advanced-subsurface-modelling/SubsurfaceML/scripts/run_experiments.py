#!/usr/bin/env python
"""Controlled experiments on the development reservoirs (ablation, interval
method selection, numerical refinement of dataset cases).

    python scripts/run_experiments.py --config config/study.yaml
    python scripts/run_experiments.py --config config/study.yaml --only ablation
    python scripts/run_experiments.py --config config/study.yaml --only numerics

Reads ``results/<name>/data/scenarios.csv`` (written by run_pipeline.py) and
writes everything to ``results/<name>/experiments/``.  Only the development
reservoirs are used; the untouched final-test and distribution-shift
reservoirs are generated and scored by ``scripts/run_final_evaluation.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

#: Ablation variants for the pressure target.  Same folds, same search
#: budget per family, same seeds; only the inputs / target form differ
#: (V5 differs only in its search budget).
PRESSURE_VARIANTS = {
    "V0_published_features": {"features": "baseline", "mode": "log"},
    "V1_plus_realised_rock": {"features": "rock", "mode": "log"},
    "V2_plus_rom_feature": {"features": "all", "mode": "log"},
    "V3_hybrid_rom_residual": {"features": "all", "mode": "hybrid"},
    "V4_rom_only": {"features": None, "mode": "rom"},
    "V5_published_features_3x_search": {"features": "baseline", "mode": "log",
                                        "n_iter_mult": 3},
}
OTHER_VARIANTS = {
    "V0_published_features": {"features": "baseline"},
    "V2_plus_rock_and_rom_features": {"features": "all"},
}


def _jdump(obj, path):
    from subsurfaceml.pipeline import _json_default
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def load_dev_table(cfg):
    from subsurfaceml.experiments import rebuild_inputs
    from subsurfaceml.features import FEATURES
    from subsurfaceml.pipeline import deduplicate
    f = Path(cfg.paths.data) / "scenarios_features.csv"
    df = pd.read_csv(f) if f.exists() else None
    if df is None or any(c not in df.columns for c in FEATURES):
        df = rebuild_inputs(cfg, pd.read_csv(Path(cfg.paths.data) / "scenarios.csv"))
    df, _ = deduplicate(df)
    return df.reset_index(drop=True)


def feature_set(name):
    from subsurfaceml.features import FEATURES, FEATURES_BASELINE, ROCK_FEATURES
    return {"baseline": FEATURES_BASELINE, "rock": FEATURES_BASELINE + ROCK_FEATURES,
            "all": FEATURES, None: None}[name]


def run_ablation(cfg, df, out_dir, n_iter, outer, seed, targets):
    from subsurfaceml.evaluation import add_subgroups, subgroup_table
    from subsurfaceml.experiments import compare, nested_cv, select_interval_method
    from subsurfaceml.pipeline import LOG_TARGETS
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    res = {"protocol": {
        "data": "development reservoirs only (the study run's 220 realisations)",
        "outer_cv": f"{outer}-fold GroupKFold over reservoirs, shuffled with seed {seed}",
        "inner_cv": f"{cfg.ml.n_splits}-fold GroupKFold inside each outer training part",
        "search": f"RandomizedSearchCV, n_iter={n_iter} per family (V5: x3), "
                  "same families as the published pipeline",
        "selection_rule": "family with the lowest inner grouped-CV RMSE",
        "comparison": "paired reservoir-level bootstrap (2000 resamples) vs V0",
        "n_cases": int(len(df)), "n_reservoirs": int(df.realisation_id.nunique())},
        "targets": {}}
    k_edges = list(np.quantile(df.drop_duplicates("realisation_id")["k_median_mD"],
                               [1 / 3, 2 / 3]))
    for target in targets:
        variants = PRESSURE_VARIANTS if target == "dp_bh_max_MPa" else OTHER_VARIANTS
        oof, timing = {}, {}
        for name, v in variants.items():
            t0 = time.perf_counter()
            mode = v.get("mode") or ("log" if target in LOG_TARGETS else "linear")
            print(f"  [{target}] {name} ...", flush=True)
            o = nested_cv(df, target, features=feature_set(v["features"]), mode=mode,
                          outer_splits=outer, inner_splits=cfg.ml.n_splits,
                          n_iter=n_iter * v.get("n_iter_mult", 1),
                          families=cfg.ml.families, seed=seed, verbose=True)
            timing[name] = time.perf_counter() - t0
            oof[name] = o
            o.to_csv(Path(out_dir) / f"ablation_oof_{target}_{name}.csv", index=False)
            print(f"      RMSE {np.sqrt(np.mean((o.pred - o.y) ** 2)):.4g}  "
                  f"({timing[name]:.0f} s)", flush=True)
        ref = "V0_published_features"
        cmp_ = compare(oof, ref)
        sub = {}
        d = add_subgroups(df, k_edges=k_edges, dp_limit_MPa=dp_lim, target=target)
        for name, o in oof.items():
            d["_pred"] = o["pred"].to_numpy()
            sub[name] = subgroup_table(d, target, "_pred")
        res["targets"][target] = {"metrics": cmp_, "subgroups": sub,
                                  "seconds": timing, "k_tercile_edges_mD": k_edges}
    # interval-method selection on the best learned pressure variant
    t = "dp_bh_max_MPa"
    if t in res["targets"]:
        learned = {k: v for k, v in res["targets"][t]["metrics"].items()
                   if k not in ("V4_rom_only",)}
        best = min(learned, key=lambda k: learned[k]["RMSE"])
        res["targets"][t]["selected_variant"] = best
        o = pd.read_csv(Path(out_dir) / f"ablation_oof_{t}_{best}.csv")
        from subsurfaceml.features import FEATURES
        sel = select_interval_method(o, df[FEATURES], alpha=0.10,
                                     n_cal_reservoirs=41, n_repeats=200, seed=seed)
        res["interval_selection"] = {"variant": best, **sel}
        o0 = pd.read_csv(Path(out_dir) / f"ablation_oof_{t}_V0_published_features.csv")
        res["interval_selection_V0"] = select_interval_method(
            o0, df[FEATURES], alpha=0.10, n_cal_reservoirs=41, n_repeats=200, seed=seed)
    res["decisions"] = apply_decision_rules(res)
    _jdump(res, Path(out_dir) / "ablation_summary.json")
    return res


def apply_decision_rules(res: dict) -> dict:
    """The pre-declared rules of docs/EVALUATION_PROTOCOL.md (1-3)."""
    dec = {}
    t = res["targets"].get("dp_bh_max_MPa")
    if t:
        learned = {k: v for k, v in t["metrics"].items() if k != "V4_rom_only"}
        best = min(learned, key=lambda k: learned[k]["RMSE"])
        spec = PRESSURE_VARIANTS[best]
        dec["pressure_variant"] = best
        dec["ml.feature_set"] = spec["features"]
        dec["ml.pressure_model"] = spec["mode"]
    # rule 2, applied per target
    for k, key in (("r_plume_m95_m", "ml.feature_set_plume"),
                   ("sweep_efficiency", "ml.feature_set_sweep")):
        if k in res["targets"]:
            m = res["targets"][k]["metrics"]
            better = (m["V2_plus_rock_and_rom_features"]["RMSE"]
                      < m["V0_published_features"]["RMSE"])
            dec[key] = "all" if better else "baseline"
            dec.setdefault("other_targets_rmse", {})[k] = {n: m[n]["RMSE"] for n in m}
    sel = res.get("interval_selection")
    if sel:
        tab = sel["table"]
        ok = [r for r in tab if r["reservoir_all_covered_mean"] >= 0.88]
        pick = (min(ok, key=lambda r: r["mean_relative_width"]) if ok
                else max(tab, key=lambda r: r["reservoir_all_covered_mean"]))
        dec["ml.interval_method"] = pick["method"]
        dec["interval_rule"] = ("smallest mean relative width with mean whole-reservoir "
                                "coverage >= 0.88" if ok else
                                "no method reached 0.88: highest whole-reservoir coverage")
    return dec


def run_numerics(cfg, df, out_dir, n_per_cell, finer_n, seed):
    from subsurfaceml.numerics import refinement_study, select_cases, summarise
    always = ["R0172_S+00", "R0036_S+02", "R0039_S+03", "R0150_S+03", "R0149_S+00"]
    cases = select_cases(df, n_per_cell=n_per_cell, seed=seed, always=always)
    t0 = time.perf_counter()
    runs = refinement_study(cfg, cases)
    # the 'finer' level on a subset (it is the expensive one)
    sub = cases.sort_values("k_median_mD").iloc[
        np.linspace(0, len(cases) - 1, finer_n).astype(int)]
    runs_f = refinement_study(cfg, sub, levels=("finer",))
    runs = pd.concat([runs, runs_f], ignore_index=True)
    runs.to_csv(Path(out_dir) / "numerics_refinement_runs.csv", index=False)
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    rep = summarise(runs, cases, dp_lim)
    rep["seconds"] = time.perf_counter() - t0
    rep["case_selection"] = {"strata": "k_median tercile x schedule-intensity quartile",
                             "n_per_cell": n_per_cell, "seed": seed,
                             "always_included": always, "finer_subset_n": int(len(sub))}
    # the same cases' dataset values must be reproduced by the production level
    prod = runs[(runs.level == "production") & (runs.status == "ok")].set_index("scenario_id")
    ref = df.set_index("scenario_id").loc[prod.index]
    rep["production_level_reproduces_dataset_max_rel"] = float(
        (np.abs(prod["dp_bh_max_MPa"] - ref["dp_bh_max_MPa"]) / ref["dp_bh_max_MPa"]).max())
    _jdump(rep, Path(out_dir) / "numerics_summary.json")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "study.yaml"))
    ap.add_argument("--only", choices=["ablation", "numerics", "decide"], default=None)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="+",
                    default=["dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency"])
    ap.add_argument("--numerics-per-cell", type=int, default=3)
    ap.add_argument("--numerics-finer", type=int, default=10)
    a = ap.parse_args()
    from subsurfaceml.config import load_config, provenance
    cfg = load_config(a.config)
    out = Path(cfg.paths.results) / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    df = load_dev_table(cfg)
    meta = {"command": " ".join(sys.argv), "environment": provenance(),
            "started": time.ctime()}
    if a.only in (None, "ablation"):
        print("=== ablation (nested grouped CV, development reservoirs) ===")
        t0 = time.perf_counter()
        res = run_ablation(cfg, df, out, a.n_iter, a.outer, a.seed, a.targets)
        meta["ablation_seconds"] = time.perf_counter() - t0
        if "dp_bh_max_MPa" in res["targets"]:
            from subsurfaceml import figures as F
            F.ablation(res, cfg.paths.figures)
    if a.only == "decide":
        f = out / "ablation_summary.json"
        res = json.loads(f.read_text())
        res["decisions"] = apply_decision_rules(res)
        _jdump(res, f)
        from subsurfaceml import figures as F
        F.ablation(res, cfg.paths.figures)
        print(json.dumps(res["decisions"], indent=2))
        return 0
    if a.only in (None, "numerics"):
        print("=== discretisation error of dataset cases ===")
        t0 = time.perf_counter()
        run_numerics(cfg, df, out, a.numerics_per_cell, a.numerics_finer, a.seed)
        meta["numerics_seconds"] = time.perf_counter() - t0
    meta["finished"] = time.ctime()
    prev = out / "experiments_run.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old[a.only or "all"] = meta
    _jdump(old, prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
