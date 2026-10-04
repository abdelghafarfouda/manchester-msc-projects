"""End-to-end pipeline: verify -> simulate -> independent test sets -> learn
-> evaluate -> screen.

Run it with ``python scripts/run_pipeline.py --config config/study.yaml``.
Every stage writes machine-readable results under ``paths.metrics`` and
figures under ``paths.figures``; the report is assembled from those files,
never from numbers typed by hand.

Data roles (``docs/EVALUATION_PROTOCOL.md``)
--------------------------------------------
``dev``         the development reservoirs (the study's 220 realisations);
                split by reservoir into ``train`` (fitting, tuning, model
                selection) and ``calib`` (interval calibration only)
``final_test``  fresh reservoirs from the same prior, generated with their
                own seed -- untouched until the final scoring
``shift``       fresh reservoirs from a lower-permeability prior -- the
                distribution-shift test

Two designs are trained on the same ``train`` reservoirs, calibrated on the
same ``calib`` reservoirs and scored on the same independent sets: the
**published approach** (inputs, target forms and empirical band of the
2026-09-20 release) and the **revised** design (realised-layer and ROM
inputs, hybrid pressure surrogate, reservoir-level conformal intervals).
The models published on 2026-09-20 are also scored as they are.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as F
from . import validation as V
from .config import Config
from .dataset import generate_dataset
from .evaluation import paired_bootstrap, point_metrics
from .features import FEATURES, FEATURES_BASELINE, engineer, features_for_schedules
from .models import fit_and_select, regression_metrics
from .splits import split_by_realisation
from .uncertainty import UNMODELLED_PHYSICS, error_source_table, monte_carlo_propagate
from .units import MPA, YEAR

TARGET_UNITS = {"dp_bh_max_MPa": "MPa", "r_plume_m95_m": "m",
                "sweep_efficiency": "-"}
#: Strictly positive targets whose drivers act multiplicatively (rate,
#: permeability, thickness): modelled as log(target), so errors and the
#: intervals are relative.  Metrics are always reported in physical units.
LOG_TARGETS = {"dp_bh_max_MPa", "r_plume_m95_m"}

#: The design of the 2026-09-20 release, re-run like for like.
PUBLISHED_DESIGN = {"name": "published_approach", "feature_set": "baseline",
                    "feature_set_plume": "baseline", "feature_set_sweep": "baseline",
                    "pressure_model": "log", "interval_method": "empirical"}
PUBLISHED_MODELS_DIR = "results/published_2026-09-20/study_models"


def revised_design(cfg) -> dict:
    return {"name": "revised", "feature_set": cfg.ml.feature_set,
            "feature_set_plume": cfg.ml.feature_set_plume,
            "feature_set_sweep": cfg.ml.feature_set_sweep,
            "pressure_model": cfg.ml.pressure_model,
            "interval_method": cfg.ml.interval_method}


def target_mode(design: dict, target: str) -> str:
    if target == "dp_bh_max_MPa":
        return design["pressure_model"]
    return "log" if target in LOG_TARGETS else "linear"


def target_features(design: dict, target: str) -> list:
    from .final_eval import design_features
    key = {"dp_bh_max_MPa": "feature_set", "r_plume_m95_m": "feature_set_plume",
           "sweep_efficiency": "feature_set_sweep"}[target]
    return list(design_features(design[key]))


def _jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))
    return str(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(d):
    """Drop private (underscore) keys before writing JSON."""
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return d


# ==========================================================================
# 1. VERIFICATION
# ==========================================================================
def stage_validate(cfg: Config) -> dict:
    print("\n=== STAGE 1  numerical verification ===")
    t0 = time.perf_counter()
    res = V.run_all(cfg, save=True)
    n = sum(1 for k, v in res["verdict"].items() if k != "ALL_PASS" and v)
    print(f"  verdict: {'ALL PASS' if res['verdict']['ALL_PASS'] else 'FAILURES'}"
          f" ({n}/{len(res['verdict']) - 1} checks, {time.perf_counter() - t0:.0f} s)")
    for k, v in res["verdict"].items():
        if not v:
            print(f"    FAILED: {k}")
    F.validation(cfg, res, cfg.paths.figures)
    return res


# ==========================================================================
# 2. DEVELOPMENT DATASET, DATA QUALITY, EDA
# ==========================================================================
def _input_columns(df):
    return [c for c in df.columns if c in FEATURES or re.fullmatch(r"q\d+_kg_s", c)]


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop scenarios whose inputs exactly repeat an earlier one
    (``drop_duplicates``, ``Lecture01`` / ``E01_EDA``).  Duplicates can only
    arise within one realisation (e.g. clipped rates), so they never straddle
    a grouped split, but they would double-weight a case."""
    dup = df.duplicated(subset=_input_columns(df), keep="first")
    return df.loc[~dup].reset_index(drop=True), int(dup.sum())


def data_quality(df: pd.DataFrame, failed: pd.DataFrame, cfg=None) -> dict:
    """Data cleaning checks (``Lecture01``, ``E01_EDA``): duplicates, missing
    values, physical ranges, constant (uninformative) columns, failed runs,
    and rates sitting on the configured floor/ceiling."""
    inp = _input_columns(df)
    num = df.select_dtypes("number")
    const = [c for c in num.columns if num[c].nunique() <= 1]
    rate_cols = [c for c in inp if re.fullmatch(r"q\d+_kg_s", c)]
    lo = cfg.schedule.q_min_kg_s * 1.0001 if cfg else 0
    hi = cfg.schedule.q_max_kg_s * 0.9999 if cfg else np.inf
    feats = [f for f in FEATURES if f in df]
    return {"n_rows": int(len(df)),
            "n_failed_runs": int(len(failed)),
            "duplicate_scenario_ids": int(df["scenario_id"].duplicated().sum()),
            "duplicate_input_rows": int(df[inp].duplicated().sum()),
            "missing_values_total": int(df[feats].isna().sum().sum()),
            "constant_columns_dropped_from_analysis": const,
            "scenarios_with_a_rate_on_the_floor_or_ceiling": int(
                ((df[rate_cols] <= lo) | (df[rate_cols] >= hi)).any(axis=1).sum()),
            "checks": {
                "saturation_in_bounds": bool((df["sg_min"] >= -1e-9).all()
                                             and (df["sg_max"] <= 1).all()),
                "mass_balance_below_1e-9": bool((df["mass_balance_error"] < 1e-9).all()),
                "no_clipping": bool((df["n_clipped"] == 0).all()),
                "positive_buildup": bool((df["dp_bh_max_Pa"] > 0).all()),
                "common_bhp_below_1Pa": bool((df["max_bhp_spread_Pa"] < 1.0).all())
                if "max_bhp_spread_Pa" in df else None}}


def stage_dataset(cfg: Config) -> dict:
    print("\n=== STAGE 2  development dataset (SYNTHETIC, from the simulator) ===")
    out = generate_dataset(cfg)
    df = engineer(out["scenarios"])
    dq = data_quality(df, out["failed"], cfg)
    df, n_dup = deduplicate(df)
    df["set"] = "dev"
    dq["duplicate_rows_dropped"] = n_dup
    df.to_csv(Path(cfg.paths.data) / "scenarios_features.csv", index=False)
    print(f"  data quality: {dq['duplicate_scenario_ids']} duplicate ids, "
          f"{dq['missing_values_total']} missing, checks {dq['checks']}")
    _jdump(dq, Path(cfg.paths.metrics) / "data_quality.json")
    F.eda(cfg, df, cfg.paths.figures, FEATURES)
    return {"df": df, "ts": out["timeseries"], "failed": out["failed"],
            "provenance": out["provenance"], "data_quality": dq}


# ==========================================================================
# 3. INDEPENDENT EVALUATION SETS
# ==========================================================================
def stage_eval_sets(cfg: Config, dev: pd.DataFrame, *, force=False) -> dict:
    print("\n=== STAGE 3  independent evaluation sets (generated after the "
          "design was fixed) ===")
    from .final_eval import SETS, check_disjoint, load_or_generate
    frames, prov = {}, {}
    for which in SETS:
        d, p = load_or_generate(cfg, which, force=force)
        d, ndup = deduplicate(d)
        d["set"] = which
        frames[which], prov[which] = d, {**p, "duplicate_rows_dropped": ndup}
        print(f"  {which}: {d.realisation_id.nunique()} reservoirs, {len(d)} cases")
    disj = check_disjoint(dev, *frames.values())
    assert disj["overlapping_ids"] == 0 and disj["overlapping_descriptions"] == 0, disj
    print(f"  disjoint from development and from each other: {disj}")
    return {"frames": frames, "provenance": prov, "disjoint": disj}


# ==========================================================================
# 4. NUMERICAL ERROR OF THE DATASET TARGETS
# ==========================================================================
def stage_numerics(cfg: Config, dev: pd.DataFrame) -> dict:
    print("\n=== STAGE 4  discretisation error of the dataset targets ===")
    from .numerics import refinement_study, select_cases, summarise
    always = [s for s in ("R0172_S+00", "R0036_S+02", "R0039_S+03", "R0150_S+03",
                          "R0149_S+00") if s in set(dev.scenario_id)]
    cases = select_cases(dev, n_per_cell=cfg.numerics.cases_per_cell,
                         seed=cfg.ml.random_state, always=always)
    t0 = time.perf_counter()
    runs = refinement_study(cfg, cases, n_jobs=cfg.n_jobs)
    elig = cases[cases["n_steps"] <= cfg.numerics.finer_max_production_steps]
    sub = elig.sort_values("k_median_mD").iloc[
        np.unique(np.linspace(0, len(elig) - 1, cfg.numerics.finer_cases).astype(int))]
    runs = pd.concat([runs, refinement_study(cfg, sub, levels=("finer",),
                                             n_jobs=cfg.n_jobs)], ignore_index=True)
    runs.to_csv(Path(cfg.paths.metrics) / "numerics_refinement_runs.csv", index=False)
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    rep = summarise(runs, cases, dp_lim)
    prod = runs[(runs.level == "production") & (runs.status == "ok")].set_index("scenario_id")
    ref = dev.set_index("scenario_id").loc[prod.index]
    rep["production_level_reproduces_dataset_max_rel"] = float(
        (np.abs(prod["dp_bh_max_MPa"] - ref["dp_bh_max_MPa"]) / ref["dp_bh_max_MPa"]).max())
    rep["case_selection"] = {"strata": "k_median tercile x schedule-intensity quartile",
                             "cases_per_cell": cfg.numerics.cases_per_cell,
                             "always_included": always, "finer_subset": int(len(sub)),
                             "finer_subset_rule": (f"cases with at most "
                                                   f"{cfg.numerics.finer_max_production_steps} "
                                                   "production time steps, spread by permeability")}
    rep["seconds"] = time.perf_counter() - t0
    _jdump(rep, Path(cfg.paths.metrics) / "numerics_summary.json")
    F.numerics(runs, cases, cfg.paths.figures, dp_lim)
    t = rep["targets"]["dp_bh_max_MPa"]["production_minus_fine"]
    print(f"  {rep['n_cases']} cases: build-up production vs fine median "
          f"{100 * t['median_rel']:.2f} %, max {100 * t['max_rel']:.2f} %; label flips "
          f"{rep['pressure_limit_label_flips']['n_flipped']}  ({rep['seconds']:.0f} s)")
    return rep


# ==========================================================================
# 5. SURROGATES: two designs, three test populations
# ==========================================================================
def make_split(cfg, df):
    """Masks over the combined table (``df`` has a RangeIndex and a ``set``
    column).  The development reservoirs are split exactly as in the
    published run (same seed and fractions); its former test reservoirs have
    been inspected during development, so they now join the training part,
    and the untouched independent sets take their place."""
    dev = (df["set"] == "dev").to_numpy()
    sp = split_by_realisation(df.loc[dev], test_fraction=cfg.ml.test_fraction,
                              calib_fraction=cfg.ml.calib_fraction,
                              random_state=cfg.ml.random_state)
    idx = np.flatnonzero(dev)
    n = len(df)
    m = {k: np.zeros(n, bool) for k in ("train", "calib", "dev_former_test")}
    m["train"][idx[sp["masks"]["train"] | sp["masks"]["test"]]] = True
    m["calib"][idx[sp["masks"]["calib"]]] = True
    m["dev_former_test"][idx[sp["masks"]["test"]]] = True
    m["train_fit"] = m["train"] | m["calib"]
    m["test"] = (df["set"] == "final_test").to_numpy()
    m["shift"] = (df["set"] == "shift").to_numpy()
    ids = {k: set(df.loc[m[k], "realisation_id"]) for k in ("train", "calib", "test", "shift")}
    keys = list(ids)
    overlaps = {f"{a}_{b}": len(ids[a] & ids[b])
                for i, a in enumerate(keys) for b in keys[i + 1:]}
    leak = {"overlaps": overlaps, "reservoirs": {k: len(v) for k, v in ids.items()},
            "rows": {k: int(m[k].sum()) for k in keys},
            "former_test_reservoirs_now_in_training": int(len(sp["test_ids"]))}
    assert max(overlaps.values()) == 0, leak
    return sp, leak, m


def _fit_design(cfg, df, m, target, design):
    from .final_eval import oof_abs_residuals
    from .intervals import METHODS, IntervalModel, fit_difficulty_model
    feats = target_features(design, target)
    mode = target_mode(design, target)
    tr, ca = m["train"], m["calib"]
    y = df[target].to_numpy(float)
    g = df["realisation_id"].to_numpy()
    best, allres, fitted = fit_and_select(
        df.loc[tr], y[tr], g[tr], feats, target, TARGET_UNITS.get(target, ""),
        n_splits=cfg.ml.n_splits, n_iter=cfg.ml.n_iter_search,
        random_state=cfg.ml.random_state, log_target=(mode == "log"),
        hybrid=(mode == "hybrid"), families=cfg.ml.families)
    log_space = mode in ("log", "hybrid")
    abs_oof = oof_abs_residuals(best, df.loc[tr, feats], y[tr], g[tr],
                                cfg.ml.n_splits, log_space)
    diff = fit_difficulty_model(df.loc[tr, feats], abs_oof,
                                random_state=cfg.ml.random_state)
    ims = {}
    for meth in METHODS:
        ims[meth] = IntervalModel(best, meth, cfg.ml.interval_alpha, log_space=log_space,
                                  difficulty=diff if meth == "adaptive_conformal" else None
                                  ).calibrate(df.loc[ca], y[ca], g[ca])
    return {"best": best, "allres": allres, "fitted": fitted, "intervals": ims,
            "features": feats, "mode": mode,
            "train_oof_abs_residual_median": float(np.median(abs_oof))}


def stage_ml(cfg: Config, df: pd.DataFrame) -> dict:
    print("\n=== STAGE 5  surrogates: published approach vs revised design ===")
    from .domain import DomainCheck
    from .final_eval import evaluate_on, load_published, published_interval
    sp, leak, m = make_split(cfg, df)
    print("  reservoirs:", leak["reservoirs"], " overlaps:", leak["overlaps"])
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    k_edges = list(np.quantile(df.loc[m["train_fit"]].drop_duplicates("realisation_id")
                               ["k_median_mD"], [1 / 3, 2 / 3]))
    domain = DomainCheck().fit(df.loc[m["train_fit"]])
    pub_dir = Path(cfg.paths.root) / PUBLISHED_MODELS_DIR
    published = (load_published(pub_dir, cfg.ml.targets)
                 if cfg.name == "study" and pub_dir.exists() else {})
    designs = [revised_design(cfg), PUBLISHED_DESIGN]
    out = {"split": {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in sp.items() if k != "masks"},
           "leakage": leak, "designs": designs,
           "features": {d["name"]: {t: target_features(d, t) for t in cfg.ml.targets}
                        for d in designs},
           "k_tercile_edges_mD": k_edges,
           "domain_check": {"descriptors": domain.descriptors,
                            "distance_threshold": domain.threshold_,
                            "n_training_reservoirs": domain.n_train_reservoirs_},
           "targets": {}}
    surrogates, bands, fitted_rev, base_models = {}, {}, {}, {}
    for target in cfg.ml.targets:
        unit = TARGET_UNITS.get(target, "")
        y = df[target].to_numpy(float)
        res_t = {"unit": unit, "designs": {}}
        fits = {}
        for design in designs:
            print(f"\n  --- {target} [{unit}] -- {design['name']} ---")
            f = _fit_design(cfg, df, m, target, design)
            fits[design["name"]] = f
            best = f["best"]
            print(f"    selected: {best.name} (grouped-CV RMSE {best.cv_score_rmse:.4g} {unit}"
                  f"; {f['mode']} target; {len(f['features'])} inputs)")
            ev = {}
            for s in ("test", "shift"):
                ev[s] = evaluate_on(design["name"], best, f["intervals"], df.loc[m[s]],
                                    target, k_edges=k_edges, dp_limit=dp_lim,
                                    domain=domain, selected_interval=design["interval_method"])
                ev[s]["mean_baseline_rmse"] = float(np.sqrt(np.mean(
                    (y[m[s]] - y[m["train"]].mean()) ** 2)))
                sel = ev[s]["intervals"][design["interval_method"]]
                print(f"    {s:5s}: RMSE {ev[s]['point']['RMSE']:.4g} | MAE "
                      f"{ev[s]['point']['MAE']:.4g} | R2 {ev[s]['point']['R2']:.4f} | worst "
                      f"under {ev[s]['point']['worst_underprediction']:.4g} {unit} | "
                      f"{design['interval_method']} coverage {sel['case_coverage']:.3f} "
                      f"(reservoirs {sel['reservoir_all_covered']:.3f})")
            ev["calib"] = point_metrics(y[m["calib"]], best.predict(df.loc[m["calib"]]))
            res_t["designs"][design["name"]] = {
                "selected_model": best.name, "target_form": f["mode"],
                "n_inputs": len(f["features"]), "best_params": best.best_params,
                "cv_rmse_selected": best.cv_score_rmse,
                "model_comparison_cv_rmse": {k: v["cv_rmse"] for k, v in f["allres"].items()},
                "model_fit_seconds": {k: v["fit_seconds"] for k, v in f["allres"].items()},
                "evaluation": ev}
        if target in published:
            sur, band = published[target]
            im = published_interval(sur, band)
            ev = {s: evaluate_on("published_2026_09_20", sur, {"empirical": im},
                                 df.loc[m[s]], target, k_edges=k_edges, dp_limit=dp_lim,
                                 selected_interval="empirical")
                  for s in ("test", "shift")}
            res_t["designs"]["published_models_as_released"] = {
                "selected_model": sur.name, "target_form": "log" if sur.log_target else "linear",
                "trained_on": "124 reservoirs of the 2026-09-20 split", "evaluation": ev}
        rev, base = fits["revised"]["best"], fits["published_approach"]["best"]
        boots = {}
        for s in ("test", "shift"):
            ys, g = y[m[s]], df.loc[m[s], "realisation_id"].to_numpy()
            pr = rev.predict(df.loc[m[s]])
            boots[f"revised_vs_published_approach_{s}"] = paired_bootstrap(
                g, ys, base.predict(df.loc[m[s]]), pr, seed=cfg.ml.random_state)
            if target in published:
                boots[f"revised_vs_published_models_{s}"] = paired_bootstrap(
                    g, ys, published[target][0].predict(df.loc[m[s]]), pr,
                    seed=cfg.ml.random_state)
        res_t["bootstrap"] = boots
        # convenience keys used by the report and the notebooks
        r_ev = res_t["designs"]["revised"]
        res_t["selected_model"] = r_ev["selected_model"]
        res_t["test"] = r_ev["evaluation"]["test"]["point"]
        res_t["shift"] = r_ev["evaluation"]["shift"]["point"]
        res_t["interval_test"] = r_ev["evaluation"]["test"]["intervals"][cfg.ml.interval_method]
        res_t["interval_shift"] = r_ev["evaluation"]["shift"]["intervals"][cfg.ml.interval_method]
        res_t["mean_baseline_test_rmse"] = r_ev["evaluation"]["test"]["mean_baseline_rmse"]
        res_t["model_comparison_cv_rmse"] = r_ev["model_comparison_cv_rmse"]
        res_t["difficult_cases"] = difficult_cases(df, m, target, rev.predict(df.loc[m["test"]]))
        out["targets"][target] = res_t
        surrogates[target] = rev
        bands[target] = fits["revised"]["intervals"][cfg.ml.interval_method]
        fitted_rev[target] = fits["revised"]["fitted"]
        base_models[target] = (base, fits["published_approach"]["intervals"]["empirical"])
        from .evaluation import add_subgroups
        d = add_subgroups(df.loc[m["test"]], k_edges=k_edges)
        F.errors_vs_inputs(d, target, rev.predict(d), base.predict(d), cfg.paths.figures)
    # figures
    preds, ivals = {}, {}
    for t, s in surrogates.items():
        X = df.loc[m["test"]]
        lo, p, hi = bands[t].predict_interval(X)
        preds[t], ivals[t] = (X[t].to_numpy(float), p), (lo, hi)
    F.parity(df.loc[m["test"]], preds, ivals, cfg.paths.figures)
    F.model_comparison(out["targets"], cfg.paths.figures)
    if "dp_bh_max_MPa" in out["targets"]:
        F.interval_coverage(out["targets"]["dp_bh_max_MPa"]["designs"]["revised"]["evaluation"],
                            cfg.paths.figures)
    out["_surrogates"], out["_bands"], out["_fitted"], out["_masks"] = \
        surrogates, bands, fitted_rev, m
    out["_baseline"], out["_domain"] = base_models, domain
    out["features_revised_pressure"] = target_features(designs[0], "dp_bh_max_MPa")
    return out


def difficult_cases(df, m, target, pred, n=5) -> dict:
    """Where does the surrogate fail?  Worst test cases, and error by
    permeability and by how hard the schedule pushes (q_mult)."""
    d = df.loc[m["test"]].copy()
    d["pred"], d["abs_err"] = pred, np.abs(pred - d[target].to_numpy())
    d["rel_err"] = d["abs_err"] / np.maximum(np.abs(d[target]), 1e-12)
    cols = ["scenario_id", target, "pred", "abs_err", "rel_err", "k_median_mD",
            "V_DP", "h_total_m", "r_e_m", "q_mult_mean", "r_fill_over_re"]
    worst = d.sort_values("abs_err", ascending=False).head(n)[cols]
    d["k_bin"] = pd.qcut(d["k_median_mD"], 3, labels=["low k", "mid k", "high k"])
    d["q_bin"] = pd.qcut(d["q_mult_mean"], 3, labels=["gentle", "moderate", "aggressive"])
    by = lambda c: d.groupby(c, observed=True).agg(
        n=("abs_err", "size"), MAE=("abs_err", "mean"),
        median_rel_err=("rel_err", "median")).reset_index().astype({c: str}).to_dict("records")
    share = float(np.sort(d["abs_err"])[::-1][:max(1, len(d) // 10)].sum() / d["abs_err"].sum())
    return {"worst_cases": worst.to_dict("records"),
            "by_permeability_tercile": by("k_bin"),
            "by_schedule_intensity_tercile": by("q_bin"),
            "share_of_total_abs_error_from_worst_10pct": share}


# ==========================================================================
# 6. UNCERTAINTY
# ==========================================================================
def stage_uncertainty(cfg, df, ml, validation, numerics=None) -> dict:
    print("\n=== STAGE 6  Monte Carlo propagation and error sources ===")
    surr, m = ml["_surrogates"], ml["_masks"]
    v7 = validation["V6_V7_V8_radial_two_phase"]

    def _num(section, key, used):
        ks = sorted(v7[section], key=float)
        fine = ks[0] if section == "r_near" else ks[-1]
        near = min(ks, key=lambda k: abs(float(k) - used))
        a, b = v7[section][near][key], v7[section][fine][key]
        return abs(a - b) / max(abs(b), 1e-30), (f"V7 {section}: {near} (closest to the "
                                                 f"dataset's {used:g}) vs finest {fine}")
    num = {}
    for t in surr:
        if numerics is not None and t in numerics.get("targets", {}):
            r = numerics["targets"][t]["production_minus_fine"]
            num[t] = (r["median_rel"], f"dataset cases re-simulated with every "
                      f"discretisation refined (median of {r['n']} cases; max "
                      f"{100 * r['max_rel']:.2f} %)")
        else:
            num[t] = {"dp_bh_max_MPa": _num("r_near", "dp_bh_max_MPa", cfg.grid.r_near_m),
                      "r_plume_m95_m": _num("n_r", "r_plume_m95_m", cfg.grid.n_r),
                      "sweep_efficiency": _num("n_r", "sweep_efficiency", cfg.grid.n_r)
                      }.get(t, (0.0, "n/a"))
    out = {"targets": {}, "prior_note": "Monte Carlo over the ASSUMED prior "
           "(development reservoirs); P10/P90 follow 5-Uncertainty.pdf p.46"}
    X_all = df.loc[m["train_fit"]]
    for t, s in surr.items():
        mc = monte_carlo_propagate(s, X_all)
        rel, src = num[t]
        tab = error_source_table(t, prior_std=float(X_all[t].std()),
                                 surrogate_rmse=ml["targets"][t]["test"]["RMSE"],
                                 numerical_rel_error=rel,
                                 reference_value=float(X_all[t].median()),
                                 unmodelled=UNMODELLED_PHYSICS)
        tab["numerical_error_source"] = src
        out["targets"][t] = {"monte_carlo_prior": mc, "error_sources": tab}
        print(f"  {t}: shares input/surrogate/numerical {tab['share_input_pct']:.0f}/"
              f"{tab['share_surrogate_pct']:.0f}/{tab['share_numerical_pct']:.1f} %")
    F.uncertainty_sources(out, cfg.paths.figures)
    return out


# ==========================================================================
# 7. CLASSIFICATION
# ==========================================================================
def stage_classifier(cfg, df, ml) -> dict:
    print("\n=== STAGE 7  pressure-limit screening (classification) ===")
    from .classify import run_binary, run_multiclass, traffic_light
    m, feats = ml["_masks"], ml["features_revised_pressure"]
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    dp = df["dp_bh_max_MPa"].to_numpy(float)
    y = (dp > dp_lim).astype(int)
    reg_pred = ml["_surrogates"]["dp_bh_max_MPa"].predict(df.loc[m["test"]])
    b = run_binary(df, feats, y, m, rs=cfg.ml.random_state, n_splits=cfg.ml.n_splits,
                   regression_pred=reg_pred, dp_limit=dp_lim)
    b["dp_limit_MPa"] = dp_lim
    b["features"] = feats
    up = ml["_bands"]["dp_bh_max_MPa"].predict_interval(df.loc[m["test"]])[2]
    from .evaluation import limit_decisions
    b["test_regression_upper_bound"] = limit_decisions(dp[m["test"]], up, dp_lim)
    ts, td = b["test_selected_threshold"], b["test_default_threshold"]
    print(f"  positives: train {b['positive_rate_train']*100:.0f}%, "
          f"test {b['positive_rate_test']*100:.0f}% | selected {b['selected']}")
    print(f"  test ROC-AUC {ts.get('roc_auc', float('nan')):.3f} | high-recall threshold: "
          f"recall {ts['recall']:.2f} precision {ts['precision']:.2f}")
    y3 = traffic_light(dp, dp_lim, cfg.ml.margin_warning_fraction)
    mc = run_multiclass(df, feats, y3, m, rs=cfg.ml.random_state,
                        n_splits=cfg.ml.n_splits, regression_pred=reg_pred,
                        dp_limit=dp_lim, warn_frac=cfg.ml.margin_warning_fraction)
    print(f"  traffic light ({mc['selected']}): test macro-F1 {mc['test']['macro_f1']:.3f}"
          f" | regression-surrogate banding {mc['test_regression_surrogate_banded']['macro_f1']:.3f}")
    F.classifier(b, mc, cfg.paths.figures)
    model = b.pop("_model"); b.pop("_curves"); mc_model = mc.pop("_model")
    from sklearn.svm import SVC
    ml["_classifier"] = {"model": model, "threshold": b["threshold_selected"],
                         "uses_decision_function": isinstance(model[-1], SVC),
                         "dp_limit_MPa": dp_lim, "family": b["selected"],
                         "features": feats}
    ml["_traffic_light"] = mc_model
    return {"binary": b, "traffic_light": mc}


# ==========================================================================
# 8. COST AND SPEED
# ==========================================================================
def stage_speed(cfg, df, ml, n_sim: int = 5, repeats: int = 30) -> dict:
    """Inference cost vs simulator cost for the same outputs, measured in
    this process on this machine (``time.perf_counter``)."""
    print("\n=== STAGE 8  inference cost vs simulation cost ===")
    from .final_eval import realisation_lookup
    from .scenarios import make_schedule, run_scenario
    surr, m = ml["_surrogates"], ml["_masks"]
    X = df.loc[m["test"]]
    n = len(X)
    t0 = time.perf_counter()
    for s in surr.values():
        s.predict(X)
    t_batch = time.perf_counter() - t0
    x1 = X.iloc[[0]]
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in surr.values():
            s.predict(x1)
        ts.append(time.perf_counter() - t0)
    t_single = float(np.median(ts))
    look = realisation_lookup(cfg)
    rows = df.loc[m["test"]].head(n_sim)
    qcols = [f"q{i+1}_kg_s" for i in range(cfg.schedule.n_periods)]
    te2e, tsim = [], []
    for _, row in rows.iterrows():
        c, r = look[int(row["realisation_id"])]
        rates = row[qcols].to_numpy(float)
        t0 = time.perf_counter()
        Xe = features_for_schedules(c, r, rates[None, :])
        for s in surr.values():
            s.predict(Xe)
        te2e.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_scenario(c, r, make_schedule(c, r, rates), want_series=False)
        tsim.append(time.perf_counter() - t0)
    sim = float(np.mean(tsim))
    res = {"hardware_note": "same machine, same process, 1 scenario at a time for "
                            "the simulator; scikit-learn/XGBoost default threading",
           "n_test_rows_batch": int(n),
           "surrogate_batch_seconds_total": t_batch,
           "surrogate_seconds_per_case_batched": t_batch / n,
           "surrogate_seconds_single_call_median": t_single,
           "surrogate_seconds_single_end_to_end_median": float(np.median(te2e)),
           "simulator_seconds_per_case_mean_serial": sim,
           "simulator_cases_timed": len(tsim),
           "speedup_batched": sim / (t_batch / n),
           "speedup_single_call": sim / t_single,
           "speedup_single_end_to_end": sim / float(np.median(te2e)),
           "outputs_compared": "the three scalar QoIs; the end-to-end time includes "
                               "building the inputs and the analytical ROM; the "
                               "simulator also yields fields and time series"}
    print(f"  simulator {sim:.2f} s/case | surrogate batched {t_batch/n:.2e} s/case | "
          f"end-to-end single {np.median(te2e):.2e} s ({res['speedup_single_end_to_end']:.0f}x)")
    return res


# ==========================================================================
# 9. INTERPRETATION AND UNSUPERVISED STRUCTURE
# ==========================================================================
def stage_interpret(cfg, df, ml) -> dict:
    print("\n=== STAGE 9  interpretation (model behaviour, not causality) ===")
    from .classify import traffic_light
    from .final_eval import realisation_lookup
    from .interpret import (cluster_regimes, describe_clusters, lime_explanation,
                            parse_profiles, pca_regimes, permutation_importances,
                            profile_structure, shap_summary, tree_impurity_importance,
                            what_if_rate_scaling)
    surr, m = ml["_surrogates"], ml["_masks"]
    out = {"disclaimer": "Permutation importance, SHAP and local explanations "
                         "describe how the fitted model uses its inputs; they "
                         "are not causal effects."}
    Xte = df.loc[m["test"]]
    for t, s in surr.items():
        imp = permutation_importances(s.estimator, Xte[s.features], Xte[t].to_numpy(),
                                      s.features, random_state=cfg.ml.random_state)
        imp.to_csv(Path(cfg.paths.metrics) / f"permutation_importance_{t}.csv", index=False)
        out[t] = {"permutation_top8": imp.head(8).to_dict("records")}
        ti = tree_impurity_importance(s.estimator, s.features)
        if ti is not None:
            out[t]["impurity_top8"] = ti.head(8).to_dict("records")
    t = "dp_bh_max_MPa"
    fam = ml["_fitted"][t]
    tree_name = next((k for k in sorted(fam, key=lambda k: fam[k].cv_score_rmse)
                      if k in ("xgboost", "gradient_boosting", "random_forest")), None)
    if tree_name:
        feats = fam[tree_name].features
        sv = shap_summary(fam[tree_name].estimator, Xte[feats], feats)
        if sv is not None:
            vals, _ = sv
            mean_abs = np.abs(vals).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:12]
            kind = (fam[tree_name].meta or {}).get("kind", "log")
            out["shap"] = {"model": tree_name,
                           "space": "ln(dp / ROM)" if kind == "hybrid" else "log(dp)",
                           "top": [{"feature": feats[i], "mean_abs_shap": float(mean_abs[i])}
                                   for i in order]}
            F.shap_bar(feats, mean_abs, order, tree_name, cfg.paths.figures)
    s = surr[t]
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    d = df.loc[m["test"]]
    i0 = int(np.argmin(np.abs(d[t].to_numpy() - dp_lim)))
    case = d.iloc[i0]
    out["local_case"] = {"scenario_id": case["scenario_id"], "simulated_dp_MPa": float(case[t]),
                         "predicted_dp_MPa": float(s.predict(d.iloc[[i0]])[0])}
    out["local_case"]["lime_top6"] = lime_explanation(
        lambda Z: s.predict(Z), case[s.features].astype(float), df.loc[m["train"], s.features],
        random_state=cfg.ml.random_state).head(6).to_dict("records")
    c, r = realisation_lookup(cfg)[int(case["realisation_id"])]
    q0 = case[[f"q{i+1}_kg_s" for i in range(cfg.schedule.n_periods)]].to_numpy(float)
    wi = what_if_rate_scaling(lambda R: s.predict(features_for_schedules(c, r, R)), q0, dp_lim)
    out["local_case"]["what_if"] = {"largest_rate_factor_predicted_below_limit":
                                    wi["largest_factor_below_limit"], "limit_MPa": dp_lim}
    F.what_if(wi, dp_lim, case["scenario_id"], cfg.paths.figures)
    res_feats = ["log10_k_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m", "n_g",
                 "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g", "log10_kh", "log10_pv"]
    dev = df.loc[m["train_fit"]]
    uniq = dev.drop_duplicates("realisation_id")
    pc = pca_regimes(uniq[res_feats].to_numpy(), res_feats)
    cl = cluster_regimes(pc["scores"][:, :3], random_state=cfg.ml.random_state)
    desc = describe_clusters(uniq, cl["labels"], res_feats + ["dp_bh_max_MPa", "r_plume_m95_m"])
    desc.to_csv(Path(cfg.paths.metrics) / "reservoir_regimes.csv", index=False)
    out["reservoir_pca"] = {"explained_variance_ratio": pc["explained_variance_ratio"],
                            "top_loadings": pc["top_loadings"],
                            "note": "the reservoir inputs are sampled independently, "
                                    "so PCA finds little shared structure -- a "
                                    "property of the synthetic design"}
    out["reservoir_kmeans"] = {"k": cl["k"], "silhouette": cl["silhouette"],
                               "table": cl["table"].to_dict("records")}
    if "final_Sg_profile" in dev.columns:
        P = parse_profiles(dev["final_Sg_profile"])
        y3 = traffic_light(dev["dp_bh_max_MPa"].to_numpy(), dp_lim,
                           cfg.ml.margin_warning_fraction)
        ps = profile_structure(P, labels_ref=y3, random_state=cfg.ml.random_state)
        out["plume_shapes"] = {k: v for k, v in ps.items() if not k.startswith("_")}
        F.profiles(P, ps, cfg.paths.figures)
    return out


# ==========================================================================
# 10. ML-METHOD STUDIES
# ==========================================================================
def stage_studies(cfg, df, ml) -> dict:
    print("\n=== STAGE 10  ML-method studies (development reservoirs only) ===")
    from .studies import run_all
    t = "dp_bh_max_MPa"
    s = ml["_surrogates"][t]
    res = run_all(df, s.features, t, ml["_masks"], ml["_fitted"][t], s,
                  n_splits=cfg.ml.n_splits, seed=cfg.ml.random_state)
    F.studies(res, cfg.paths.figures)
    return res


# ==========================================================================
# 11. CONSERVATION CHECK, OPEN BOUNDARY
# ==========================================================================
def stage_conservation_check(cfg, df) -> dict:
    print("\n=== STAGE 11a  conservation identity check ===")
    planned = df["planned_mass_kg"].to_numpy(float)
    retained = df["mass_retained_kg"].to_numpy(float)
    rel = np.abs(retained - planned) / np.maximum(planned, 1e-30)
    print(f"  |retained - planned| / planned <= {rel.max():.2e} over {len(df)} scenarios")
    return {"outer_bc": cfg.solver.outer_bc, "n": int(len(df)),
            "max_rel_error_retained_vs_planned": float(rel.max()),
            "conclusion": "Under the sealed boundary retained mass equals the "
                          "planned injected mass (CO2 mass balance), so it is "
                          "computed exactly and is not an ML target."}


def stage_boundary_comparison(cfg, n: int = 10) -> dict:
    """Injected vs retained mass with a constant-pressure outer boundary
    (``1-Transmissibility.pdf`` p.11 Dirichlet condition) and a compartment
    shrunk so that the plume reaches it."""
    print("\n=== STAGE 11b  open-boundary variant: injected vs retained ===")
    import copy, dataclasses
    from .scenarios import run_scenario, sample_realisations, sample_schedules
    c2 = copy.deepcopy(cfg)
    c2.solver.outer_bc = "constant_pressure"
    rows = []
    for r in sample_realisations(c2)[:n]:
        sc = sample_schedules(c2, r)[0]
        dt = c2.schedule.t_inject_years * YEAR / c2.schedule.n_periods
        mass = float(sc["rates_kg_s"].sum() * dt)
        r_fill = float(np.sqrt(mass / (r.rho_g * np.pi * r.h_total_m * r.phi_mean
                                       * (1.0 - r.S_ar))))
        r2 = dataclasses.replace(r, r_e_m=float(np.clip(0.8 * r_fill, 200.0, 4000.0)))
        o = run_scenario(c2, r2, sc, want_series=False)
        if o["status"] != "ok":
            rows.append({"realisation_id": r.realisation_id, "status": "failed",
                         "error": o.get("error")})
            continue
        w = o["row"]
        rows.append({"realisation_id": r.realisation_id, "status": "ok",
                     "injected_Mt": w["mass_injected_kg"] / 1e9,
                     "retained_Mt": w["mass_retained_kg"] / 1e9,
                     "lost_Mt": w["mass_lost_kg"] / 1e9,
                     "retention_fraction": w["retention_fraction"],
                     "r_e_m": r2.r_e_m, "mass_balance_error": w["mass_balance_error"]})
    d = pd.DataFrame(rows)
    d.to_csv(Path(cfg.paths.metrics) / "open_boundary_comparison.csv", index=False)
    ok = d[d.status == "ok"]
    res = {"n_runs": int(len(d)), "n_failed": int((d.status != "ok").sum()),
           "n_with_loss": int((ok.lost_Mt > 0).sum()),
           "median_retention_fraction": float(ok.retention_fraction.median()),
           "min_retention_fraction": float(ok.retention_fraction.min()),
           "max_mass_balance_error": float(ok.mass_balance_error.max())}
    print(f"  {res['n_with_loss']}/{len(ok)} runs lost CO2 across r_e; median "
          f"retention {res['median_retention_fraction']:.3f}")
    return res


# ==========================================================================
# 12. VERIFICATION-GATED SCHEDULE SCREENING
# ==========================================================================
def _band_predictor(cfg, r, pressure, plume):
    """``predictor(R)`` for :mod:`screening` from (surrogate, interval) pairs."""
    def pred(R):
        X = features_for_schedules(cfg, r, R)
        lo, p, hi = pressure[1].predict_interval(X)
        lo2, p2, hi2 = plume[1].predict_interval(X)
        return {"dp": p, "dp_hi": hi, "r95": p2, "r95_hi": hi2}
    return pred


def _rom_predictor(cfg, r, plume):
    """Pressure from the analytical ROM alone (no learned correction, no
    interval), plume radius from the revised surrogate and its interval --
    the control that isolates the learned pressure correction."""
    def pred(R):
        X = features_for_schedules(cfg, r, R)
        dp = 10.0 ** X["log10_rom_dp_MPa"].to_numpy(float)
        lo2, p2, hi2 = plume[1].predict_interval(X)
        return {"dp": dp, "dp_hi": dp, "r95": p2, "r95_hi": hi2}
    return pred


def _screen_one(cfg, c, r, which, rev, base, in_domain, budget, n_cand, seed):
    from . import screening as S
    from .optimise import sample_candidates
    sim = S.make_simulator(c, r)
    rng = np.random.default_rng(seed + int(r.realisation_id))
    cands = sample_candidates(c, r, n_cand, rng)
    recs = [S.recommend_constant_simulator(c, r, sim, budget),
            S.recommend_constant_rom(c, r, sim, budget),
            S.recommend_surrogate(c, r, sim, min(3, budget), _band_predictor(c, r, *base),
                                  cands, repair=False, n_verify=3, fallback=None,
                                  method="surrogate_published"),
            S.recommend_surrogate(c, r, sim, budget, _rom_predictor(c, r, rev[1]), cands,
                                  in_domain=True, repair=True, n_verify=1,
                                  fallback="rom_constant", method="rom_shaped"),
            S.recommend_surrogate(c, r, sim, budget, _band_predictor(c, r, *rev), cands,
                                  in_domain=in_domain, repair=True, n_verify=1,
                                  fallback="rom_constant", method="surrogate_verified")]
    rows = []
    for rec in recs:
        d = rec.as_dict()
        first = rec.checks[0] if rec.checks else None
        rows.append({"realisation_id": r.realisation_id, "set": which,
                     "method": rec.method, "status": rec.status,
                     "mass_Mt": rec.mass_Mt, "n_simulations": rec.n_simulations,
                     "flags": ";".join(rec.flags),
                     "first_simulated_violates": (None if first is None or first.status != "ok"
                                                  else not first.feasible),
                     "verified_dp_MPa": d["verified"]["dp_MPa"] if d["verified"] else np.nan,
                     "verified_r95_m": d["verified"]["r95_m"] if d["verified"] else np.nan,
                     "recommended_rates_kg_s": rec.recommended_rates_kg_s,
                     "in_domain": bool(in_domain), "k_median_mD": r.k_median_mD,
                     "detail": d})
    return rows


def stage_screening(cfg, df, ml) -> dict:
    print("\n=== STAGE 12  verification-gated schedule screening ===")
    from joblib import Parallel, delayed
    from .final_eval import realisation_lookup
    m, e = ml["_masks"], cfg.evaluation
    look = realisation_lookup(cfg)
    dom = ml["_domain"]
    rev_p = (ml["_surrogates"]["dp_bh_max_MPa"], ml["_bands"]["dp_bh_max_MPa"])
    rev_r = (ml["_surrogates"]["r_plume_m95_m"], ml["_bands"]["r_plume_m95_m"])
    base_p, base_r = ml["_baseline"]["dp_bh_max_MPa"], ml["_baseline"]["r_plume_m95_m"]
    jobs = []
    for which, mask, n in (("final_test", m["test"], e.screening_reservoirs_final_test),
                           ("shift", m["shift"], e.screening_reservoirs_shift)):
        ids = sorted(df.loc[mask, "realisation_id"].unique())[:n]
        for rid in ids:
            c, r = look[int(rid)]
            row = df.loc[mask & (df.realisation_id == rid)].iloc[[0]]
            in_dom = bool(dom.check(row)["in_domain"].iloc[0])
            jobs.append((c, r, which, in_dom))
    print(f"  {len(jobs)} reservoirs x 5 methods, budget {e.screening_budget} "
          f"simulations per method")
    t0 = time.perf_counter()
    out_rows = Parallel(n_jobs=cfg.n_jobs, batch_size=1)(
        delayed(_screen_one)(cfg, c, r, which, (rev_p, rev_r), (base_p, base_r), in_dom,
                             e.screening_budget, e.screening_candidates, cfg.ml.random_state)
        for c, r, which, in_dom in jobs)
    rows = [x for rs in out_rows for x in rs]
    rec = pd.DataFrame(rows)
    base = rec[rec.method == "simulator_constant"].set_index("realisation_id")["mass_Mt"]
    rec["constant_mass_Mt"] = rec["realisation_id"].map(base)
    rec["mass_vs_constant_pct"] = 100 * (rec["mass_Mt"] - rec["constant_mass_Mt"]) / rec["constant_mass_Mt"]
    rec.drop(columns=["detail"]).to_csv(Path(cfg.paths.metrics) / "screening_recommendations.csv",
                                        index=False)
    _jdump([r["detail"] for r in rows], Path(cfg.paths.metrics) / "screening_details.json")
    summ = {"budget_simulations_per_method": e.screening_budget,
            "n_candidates": e.screening_candidates,
            "dp_limit_MPa": cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa,
            "r_plume_limit_m": cfg.optim.r_plume_limit_m,
            "limit_status": "stated modelling assumptions, not fracture or caprock criteria",
            "seconds": time.perf_counter() - t0, "by_set": {}}
    for which in ("final_test", "shift"):
        d = rec[rec.set == which]
        if d.empty:
            continue
        s = {"n_reservoirs": int(d.realisation_id.nunique()),
             "n_reservoirs_out_of_domain": int((~d.drop_duplicates("realisation_id").in_domain).sum()),
             "methods": {}}
        for meth, g in d.groupby("method"):
            gain = g.mass_vs_constant_pct.dropna()
            s["methods"][meth] = {
                "n_verified": int((g.status == "VERIFIED_FEASIBLE").sum()),
                "n_no_feasible_found": int((g.status == "NO_FEASIBLE_SCHEDULE_FOUND").sum()),
                "n_no_candidate_predicted": int((g.status == "NO_CANDIDATE_PREDICTED_FEASIBLE").sum()),
                "mean_simulations": float(g.n_simulations.mean()),
                "first_simulated_schedule_violates": int(g.first_simulated_violates.fillna(False).sum()),
                "median_mass_vs_constant_pct": float(gain.median()) if len(gain) else None,
                "min_mass_vs_constant_pct": float(gain.min()) if len(gain) else None,
                "max_mass_vs_constant_pct": float(gain.max()) if len(gain) else None,
                "n_compared": int(len(gain)),
                "flags": g["flags"][g["flags"] != ""].value_counts().to_dict()}
        summ["by_set"][which] = s
        for meth, v in s["methods"].items():
            print(f"  {which:10s} {meth:22s} verified {v['n_verified']}/{s['n_reservoirs']} | "
                  f"first proposal violated {v['first_simulated_schedule_violates']} | "
                  f"median vs constant {v['median_mass_vs_constant_pct']}")
    F.screening(cfg, rec, cfg.paths.figures)
    return summ


# ==========================================================================
def save_models(cfg, ml) -> dict:
    from .artifacts import save_bundle
    objs = {f"surrogate_{t}.joblib": s for t, s in ml["_surrogates"].items()}
    objs.update({f"band_{t}.joblib": b for t, b in ml["_bands"].items()})
    objs["domain_check.joblib"] = ml["_domain"]
    if "_classifier" in ml:
        objs["pressure_classifier.joblib"] = ml["_classifier"]
    man = save_bundle(cfg, objs, dataset_file=Path(cfg.paths.data) / "scenarios.csv",
                      extra={"calib_realisation_ids": ml["split"]["calib_ids"],
                             "selected_models": {t: s.name for t, s in ml["_surrogates"].items()},
                             "design": revised_design(cfg),
                             "features_by_model": {t: s.features for t, s in ml["_surrogates"].items()}})
    print(f"\n  saved {len(objs)} model files + manifest to {cfg.paths.models}")
    return man
