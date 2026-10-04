#!/usr/bin/env python
"""Controlled experiments (ablation, interval method selection, numerical
refinement of dataset cases, screen for transient peaks).

    python scripts/run_experiments.py --config config/study.yaml --only ablation
    python scripts/run_experiments.py --config config/study.yaml --only decide
    python scripts/run_experiments.py --config config/study.yaml --only numerics
    python scripts/run_experiments.py --config config/study.yaml --only peak_screen

Reads ``results/<name>/data/`` (written by run_pipeline.py) and writes to
``results/<name>/experiments/``.  The ablation, the decision rules and the
refinement study use the development reservoirs only; the final-test and
distribution-shift reservoirs are generated and scored once by
``run_pipeline.py``.  ``peak_screen`` is a numerical check of the targets of
every set (it re-simulates the test and shift cases to record their pressure
series and refines the cases it flags); it changes no model, decision or
reported evaluation.
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
    # the 'finer' level on a subset (it is the expensive one); same rule as
    # pipeline.stage_numerics
    elig = cases[cases["n_steps"] <= cfg.numerics.finer_max_production_steps]
    sub = elig.sort_values("k_median_mD").iloc[
        np.unique(np.linspace(0, len(elig) - 1, finer_n).astype(int))]
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


def run_peak_screen(cfg, out_dir, tol=0.01):
    """Screen every development, final-test and shift case for a peak build-up
    set by a transient between reporting times (``numerics.startup_peak_screen``),
    refine the flagged cases, and recompute the revised pressure surrogate's
    test and shift errors with the refined targets."""
    from joblib import Parallel, delayed
    from subsurfaceml.artifacts import load_bundle
    from subsurfaceml.evaluation import point_metrics
    from subsurfaceml.final_eval import eval_realisations
    from subsurfaceml.numerics import refinement_study, startup_peak_screen
    from subsurfaceml.scenarios import run_scenario, sample_realisations, sample_schedules
    from subsurfaceml.units import MPA
    d = Path(cfg.paths.data)
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    levels = ("production", "r_near_half", "n_r_x2", "fine")
    sur = load_bundle(cfg, ["surrogate_dp_bh_max_MPa.joblib"])["surrogate_dp_bh_max_MPa.joblib"]
    rep = {"rule": "flag a case when its time-step peak build-up exceeds the largest "
                   f"reported (quarter-yearly) value during injection by more than {tol:.0%}",
           "tolerance": tol, "levels": list(levels), "dp_limit_MPa": dp_lim, "sets": {}}
    screens, all_runs = [], []
    t0 = time.perf_counter()
    for which in ("development", "final_test", "shift"):
        if which == "development":
            c, rl = cfg, sample_realisations(cfg)
            sc = pd.read_csv(d / "scenarios.csv")
            ser = pd.read_csv(d / "timeseries.csv")
            reproduced = None
        else:
            c, rl = eval_realisations(cfg, which)
            sc = pd.read_csv(d / f"{which}_scenarios.csv")
            jobs = [(r, s) for r in rl for s in sample_schedules(c, r)]
            res = Parallel(n_jobs=cfg.n_jobs, batch_size=4)(
                delayed(run_scenario)(c, r, s, want_series=True) for r, s in jobs)
            ok = [x for x in res if x["status"] == "ok"]
            ser = pd.concat([x["series"] for x in ok], ignore_index=True)
            new = pd.DataFrame([x["row"] for x in ok]).set_index("scenario_id")
            reproduced = float(np.max(np.abs(
                new.loc[sc["scenario_id"], "dp_bh_max_Pa"].to_numpy()
                / sc["dp_bh_max_Pa"].to_numpy() - 1.0)))
        reals = {r.realisation_id: r for r in rl}
        scr = startup_peak_screen(ser, sc, tol=tol)
        scr.insert(0, "set", which)
        screens.append(scr)
        flagged = sc[sc["scenario_id"].isin(scr.loc[scr["flagged"], "scenario_id"])]
        cases = []
        if len(flagged):
            runs = refinement_study(c, flagged, levels=levels, n_jobs=cfg.n_jobs,
                                    realisations=reals)
            runs.insert(0, "set", which)
            all_runs.append(runs)
            w = runs[runs["status"] == "ok"].pivot(index="scenario_id", columns="level",
                                                   values="dp_bh_max_MPa")
            for sid, v in w.iterrows():
                cases.append({"scenario_id": sid,
                              "excess": float(scr.set_index("scenario_id").loc[sid, "excess"]),
                              **{f"dp_{lv}_MPa": float(v[lv]) for lv in levels if lv in v},
                              "rel_change_fine": float(v["fine"] / v["production"] - 1.0),
                              "limit_label_changes": bool((v["production"] > dp_lim)
                                                          != (v["fine"] > dp_lim))})
        ex = scr["excess"]
        srep = {"n_cases": int(len(scr)), "n_flagged": int(scr["flagged"].sum()),
                "share_flagged": float(scr["flagged"].mean()),
                "excess_median": float(ex.median()), "excess_p99": float(ex.quantile(0.99)),
                "excess_max": float(ex.max()),
                "n_excess_between_0.1pct_and_tol": int(((ex > 1e-3) & (ex <= tol)).sum()),
                "flagged_cases": cases,
                "n_limit_label_changes": int(sum(x["limit_label_changes"] for x in cases))}
        if reproduced is not None:
            srep["resimulation_reproduces_stored_targets_max_rel"] = reproduced
            # the revised pressure surrogate, scored against the stored and the
            # refined targets (refined value for the flagged cases only)
            y = sc["dp_bh_max_MPa"].to_numpy(float)
            y_ref = y.copy()
            fine = {x["scenario_id"]: x["dp_fine_MPa"] for x in cases}
            m = sc["scenario_id"].isin(list(fine)).to_numpy()
            y_ref[m] = sc.loc[m, "scenario_id"].map(fine).to_numpy(float)
            p = np.asarray(sur.predict(sc), float)
            srep["revised_surrogate"] = {"stored_targets": point_metrics(y, p),
                                         "flagged_targets_refined": point_metrics(y_ref, p)}
        rep["sets"][which] = srep
        print(f"  [{which}] {srep['n_flagged']} of {srep['n_cases']} flagged; "
              f"label changes {srep['n_limit_label_changes']}")
    rep["seconds"] = time.perf_counter() - t0
    pd.concat(screens, ignore_index=True).to_csv(Path(out_dir) / "peak_screen_cases.csv",
                                                 index=False)
    if all_runs:
        pd.concat(all_runs, ignore_index=True).to_csv(
            Path(out_dir) / "peak_screen_refinement_runs.csv", index=False)
    _jdump(rep, Path(out_dir) / "peak_screen.json")
    return rep


def run_calibration_check(cfg, out_dir):
    """Protocol addendum: re-calibrate the fixed, saved design's intervals on
    fresh reservoirs from the development prior and score them once on the
    final-test and shift sets, next to the original calibration.  Nothing is
    refitted or re-selected."""
    from subsurfaceml.artifacts import load_bundle
    from subsurfaceml.evaluation import limit_decisions
    from subsurfaceml.final_eval import check_disjoint, load_or_generate
    from subsurfaceml.intervals import METHODS, IntervalModel
    d = Path(cfg.paths.data)
    sets = {"final_test": pd.read_csv(d / "final_test_scenarios.csv"),
            "shift": pd.read_csv(d / "shift_scenarios.csv")}
    dev = pd.read_csv(d / "scenarios.csv")
    cal, prov = load_or_generate(cfg, "calibration_check")
    disj = check_disjoint(dev, *sets.values(), cal)
    assert disj["overlapping_ids"] == 0 and disj["overlapping_descriptions"] == 0, disj
    summary = json.loads((Path(cfg.paths.metrics) / "summary.json").read_text())
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    b = load_bundle(cfg)
    rep = {"addendum": "docs/EVALUATION_PROTOCOL.md, 'Addendum ... independence of the interval calibration'",
           "calibration_check_set": prov, "disjoint": disj,
           "calibration_check_realisation_ids": sorted(cal["realisation_id"].unique().tolist()),
           "original_calibration_realisation_ids": b["manifest"].get("calib_realisation_ids"),
           "targets": {}}

    def score(im, df_set, t):
        y = df_set[t].to_numpy(float)
        ev = im.evaluate(df_set, y, df_set["realisation_id"].to_numpy())
        if t == "dp_bh_max_MPa":
            ev["limit_upper"] = limit_decisions(y, im.predict_interval(df_set)[2], dp_lim)
        return ev

    for t in cfg.ml.targets:
        band = b[f"band_{t}.joblib"]
        rt = {"selected_method": band.method, "original": {}, "fresh": {}}
        for name, df_set in sets.items():
            ev = score(band, df_set, t)
            rep_ = summary["ml"]["targets"][t]["designs"]["revised"]["evaluation"][
                "test" if name == "final_test" else name]
            ref = rep_["intervals"][band.method]
            ev["matches_pipeline_record"] = bool(
                abs(ev["case_coverage"] - ref["case_coverage"]) < 1e-12
                and abs(ev["reservoir_all_covered"] - ref["reservoir_all_covered"]) < 1e-12
                and abs(ev["mean_width"] - ref["mean_width"]) < 1e-9 * max(1.0, ref["mean_width"]))
            rt["original"][name] = ev
        for meth in METHODS:
            im = IntervalModel(band.s, meth, band.alpha, log_space=band.log_space,
                               difficulty=band.difficulty if meth == "adaptive_conformal" else None)
            im.calibrate(cal, cal[t].to_numpy(float), cal["realisation_id"].to_numpy())
            rt["fresh"][meth] = {name: score(im, df_set, t) for name, df_set in sets.items()}
            rt["fresh"][meth]["calibrated_quantile"] = im.q_
        rt["original_calibrated_quantile"] = band.q_
        rep["targets"][t] = rt
        f, o = rt["fresh"][band.method], rt["original"]
        print(f"  [{t}] {band.method}: test cases/reservoirs covered "
              f"original {o['final_test']['case_coverage']:.3f}/{o['final_test']['reservoir_all_covered']:.3f}, "
              f"fresh {f['final_test']['case_coverage']:.3f}/{f['final_test']['reservoir_all_covered']:.3f}; "
              f"shift original {o['shift']['reservoir_all_covered']:.3f}, fresh {f['shift']['reservoir_all_covered']:.3f}")
    _jdump(rep, Path(out_dir) / "calibration_check.json")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "study.yaml"))
    ap.add_argument("--only", choices=["ablation", "numerics", "decide", "peak_screen",
                                       "calibration_check"], default=None)
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
    from subsurfaceml.experiments import input_table_sha256, load_dev_table
    from subsurfaceml.features import FEATURES
    df = load_dev_table(cfg)
    meta = {"command": " ".join(sys.argv), "environment": provenance(),
            "started": time.ctime(),
            "input_table": "inputs rebuilt from data/scenarios.csv (experiments.load_dev_table)",
            "input_table_sha256": input_table_sha256(
                df, FEATURES + list(a.targets) + ["realisation_id"])}
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
    if a.only == "calibration_check":
        print("=== calibration check: fixed design re-calibrated on fresh reservoirs ===")
        t0 = time.perf_counter()
        run_calibration_check(cfg, out)
        meta["calibration_check_seconds"] = time.perf_counter() - t0
    if a.only == "peak_screen":
        print("=== screen for transient peaks (all sets) ===")
        t0 = time.perf_counter()
        run_peak_screen(cfg, out)
        meta["peak_screen_seconds"] = time.perf_counter() - t0
    meta["finished"] = time.ctime()
    prev = out / "experiments_run.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old[a.only or "all"] = meta
    _jdump(old, prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
