"""End-to-end pipeline: validate -> simulate -> learn -> evaluate -> screen.

Run it with ``python scripts/run_pipeline.py --config config/demo.yaml``.
Every stage writes machine-readable results under ``paths.metrics`` and
figures under ``paths.figures``; the report is assembled from those files,
never from numbers typed by hand.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import validation as V
from .config import Config
from .dataset import generate_dataset
from .features import FEATURES, engineer, features_for_schedules
from .models import fit_and_select, regression_metrics
from .splits import check_no_leakage, split_by_realisation
from .uncertainty import (ErrorBand, UNMODELLED_PHYSICS, error_source_table,
                          monte_carlo_propagate)
from .units import MPA, YEAR

TARGET_UNITS = {"dp_bh_max_MPa": "MPa", "r_plume_m95_m": "m",
                "sweep_efficiency": "-"}
#: Strictly positive targets whose drivers act multiplicatively (rate,
#: permeability, thickness): modelled as log(target), so errors and the
#: error band are relative.  Metrics are always reported in physical units.
LOG_TARGETS = {"dp_bh_max_MPa", "r_plume_m95_m"}

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))
    return str(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
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
# 1. VALIDATION
# ==========================================================================
def stage_validate(cfg: Config) -> dict:
    print("\n=== STAGE 1  numerical verification & validation ===")
    t0 = time.perf_counter()
    res = V.run_all(cfg, save=True)
    print(f"  verdict: {'ALL PASS' if res['verdict']['ALL_PASS'] else 'FAILURES'}"
          f"  ({time.perf_counter() - t0:.0f} s)")
    for k, v in res["verdict"].items():
        if not v:
            print(f"    FAILED: {k}")
    _figure_validation(cfg, res)
    return res


def _figure_validation(cfg, res):
    from .fluids import FluidProperties, RelPerm, RockProperties, bl_profile_1d
    from .grid import CartesianGrid1D
    from .impes import InjectionSchedule, TwoPhaseModel
    from .units import md_to_m2
    fl = FluidProperties(c_a=0.0, c_g=0.0)
    rp, rk = RelPerm(), RockProperties(c_r=0.0)
    L, A, phi, q_vol = 100.0, 10.0, 0.20, 1e-5
    T = 0.4 * L * A * phi / q_vol
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for n, c in zip((100, 400), ("tab:orange", "tab:blue")):
        g = CartesianGrid1D(n=n, L=L, area=A)
        m = TwoPhaseModel([g], np.full((1, n), md_to_m2(100.0)),
                          np.full((1, n), phi), fl, rp, rk, 15 * MPA,
                          cfl=0.3, max_dS=0.05, dt_init=100.0, dt_max=T / 50)
        r = m.run(InjectionSchedule([0.0, T], [q_vol * fl.rho_g]), np.array([T]))
        ax[0].plot(g.centres, r.Sg[-1, 0], c, lw=1.2, label=f"IMPES n={n}")
    xg = np.linspace(0, L, 2000)
    ax[0].plot(xg, bl_profile_1d(rp, fl, xg, T, q_vol, A, phi, L=L), "k--",
               lw=1.4, label="Buckley-Leverett (Welge)")
    ax[0].set_xlabel("distance [m]"); ax[0].set_ylabel(r"$S_g$ [-]")
    ax[0].set_title("V5 Buckley-Leverett, 0.4 PV injected"); ax[0].legend(fontsize=7)
    bl = res["V5_buckley_leverett"]["grids"]
    ns = np.array(sorted(int(k) for k in bl))
    l1 = np.array([bl[str(n)]["L1_saturation_error"] for n in ns])
    ax[1].loglog(ns, l1, "o-", label="IMPES $L_1$ error")
    ax[1].loglog(ns, l1[0] * ns[0] / ns, "k--", lw=1, label="first order")
    ax[1].set_xlabel("cells"); ax[1].set_ylabel(r"$L_1$ error in $S_g$")
    ax[1].set_title("V5 convergence"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "01_validation_buckley_leverett.png")

    r7 = res["V6_V7_V8_radial_two_phase"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, key, xl in zip(ax, ("n_r", "r_near", "max_dS"),
                          ("radial cells $n_r$", "well block $r_{near}$ [m]",
                           r"max $\Delta S_g$ per step")):
        ks = sorted(r7[key], key=float)
        x = [float(k) for k in ks]
        a.plot(x, [r7[key][k]["dp_bh_max_MPa"] for k in ks], "o-")
        a2 = a.twinx()
        a2.plot(x, [r7[key][k]["r_plume_end_m"] for k in ks], "s--", color="tab:red")
        a2.grid(False)
        a.set_xlabel(xl); a.set_ylabel(r"$\Delta p_{bh,max}$ [MPa]")
        a2.set_ylabel("threshold plume radius [m]", color="tab:red")
        if key != "n_r":
            a.set_xscale("log")
    ax[0].set_title("V7 discretisation convergence")
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "02_validation_convergence.png")


# ==========================================================================
# 2. DATASET, DATA QUALITY, EDA
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
    rep = {"n_rows": int(len(df)),
           "n_failed_runs": int(len(failed)),
           "duplicate_scenario_ids": int(df["scenario_id"].duplicated().sum()),
           "duplicate_input_rows": int(df[inp].duplicated().sum()),
           "missing_values_total": int(df[FEATURES].isna().sum().sum()),
           "constant_columns_dropped_from_analysis": const,
           "scenarios_with_a_rate_on_the_floor_or_ceiling": int(
               ((df[[c for c in inp if c.endswith("_kg_s") and c[1].isdigit()]]
                 <= (cfg.schedule.q_min_kg_s * 1.0001 if cfg else 0))
                | (df[[c for c in inp if c.endswith("_kg_s") and c[1].isdigit()]]
                   >= (cfg.schedule.q_max_kg_s * 0.9999 if cfg else np.inf))).any(axis=1).sum()),
           "checks": {
               "saturation_in_bounds": bool((df["sg_min"] >= -1e-9).all()
                                            and (df["sg_max"] <= 1).all()),
               "mass_balance_below_1e-9": bool((df["mass_balance_error"] < 1e-9).all()),
               "no_clipping": bool((df["n_clipped"] == 0).all()),
               "positive_buildup": bool((df["dp_bh_max_Pa"] > 0).all()),
               "common_bhp_below_1Pa": bool((df["max_bhp_spread_Pa"] < 1.0).all())
               if "max_bhp_spread_Pa" in df else None}}
    return rep


def stage_dataset(cfg: Config) -> dict:
    print("\n=== STAGE 2  scenario dataset (SYNTHETIC, from the simulator) ===")
    out = generate_dataset(cfg)
    df = engineer(out["scenarios"])
    dq = data_quality(df, out["failed"], cfg)
    df, n_dup = deduplicate(df)
    dq["duplicate_rows_dropped"] = n_dup
    df.to_csv(Path(cfg.paths.data) / "scenarios_features.csv", index=False)
    print(f"  data quality: {dq['duplicate_scenario_ids']} duplicate ids, "
          f"{dq['missing_values_total']} missing, checks {dq['checks']}")
    _jdump(dq, Path(cfg.paths.metrics) / "data_quality.json")
    _figure_eda(cfg, df)
    return {"df": df, "ts": out["timeseries"], "failed": out["failed"],
            "provenance": out["provenance"], "data_quality": dq}


def _figure_eda(cfg, df):
    tg = ["dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency", "mass_retained_Mt"]
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.9))
    for a, t in zip(ax, tg):
        a.hist(df[t].dropna(), bins=30, color="tab:blue", alpha=.85)
        a.set_xlabel(t); a.set_ylabel("count")
    ax[0].axvline(cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa, color="r",
                  ls="--", lw=1.2, label="stated limit")
    ax[0].legend(fontsize=7)
    ax[1].axvline(cfg.optim.r_plume_limit_m, color="r", ls="--", lw=1.2)
    fig.suptitle("EDA - simulated QoI distributions (synthetic data)", y=1.04)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "03_eda_targets.png")

    feats = ["log10_k_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m",
             "log10_q_mult_mean", "q_front_load", "r_fill_over_re"]
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.4))
    for a, f in zip(ax.ravel(), feats):
        a.scatter(df[f], df["dp_bh_max_MPa"], s=8, alpha=.6)
        a.set_xlabel(f); a.set_ylabel(r"$\Delta p_{bh,max}$ [MPa]")
    fig.suptitle("EDA - pressure buildup vs inputs", y=1.01)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "04_eda_scatter.png")

    num = df[FEATURES + tg[:3]].corr()
    fig, a = plt.subplots(figsize=(9, 8))
    im = a.imshow(num.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(num))); a.set_xticklabels(num.columns, rotation=90, fontsize=6)
    a.set_yticks(range(len(num))); a.set_yticklabels(num.columns, fontsize=6)
    a.grid(False); fig.colorbar(im, shrink=.7)
    a.set_title("Feature / target correlation (Pearson)")
    _save(fig, Path(cfg.paths.figures) / "05_eda_correlation.png")


# ==========================================================================
# 3. SURROGATES
# ==========================================================================
def make_split(cfg, df):
    sp = split_by_realisation(df, test_fraction=cfg.ml.test_fraction,
                              calib_fraction=cfg.ml.calib_fraction,
                              random_state=cfg.ml.random_state)
    leak = check_no_leakage(df, sp["masks"])
    assert max(leak["overlaps"].values()) == 0, leak
    m = dict(sp["masks"])
    m["train_fit"] = m["train"] | m["calib"]
    return sp, leak, m


def stage_ml(cfg: Config, df: pd.DataFrame) -> dict:
    print("\n=== STAGE 3  surrogates (grouped by reservoir realisation) ===")
    sp, leak, m = make_split(cfg, df)
    print("  rows:", leak["rows"], " realisations:", leak["sizes"],
          " overlaps:", leak["overlaps"])
    feats = list(FEATURES)
    Xtr, Xca, Xte = (df.loc[m[k], feats] for k in ("train", "calib", "test"))
    gtr = df.loc[m["train"], "realisation_id"].to_numpy()
    out = {"split": {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in sp.items() if k != "masks"},
           "leakage": leak, "features": feats, "targets": {}}
    surrogates, bands, fitted_all = {}, {}, {}
    for target in cfg.ml.targets:
        unit = TARGET_UNITS.get(target, "")
        print(f"\n  --- target: {target} [{unit}] ---")
        ytr, yca, yte = (df.loc[m[k], target].to_numpy(float)
                         for k in ("train", "calib", "test"))
        best, allres, fitted = fit_and_select(
            Xtr, ytr, gtr, feats, target, unit, n_splits=cfg.ml.n_splits,
            n_iter=cfg.ml.n_iter_search, random_state=cfg.ml.random_state,
            log_target=target in LOG_TARGETS, families=cfg.ml.families)
        print(f"    selected: {best.name}  (grouped-CV RMSE {best.cv_score_rmse:.4g} {unit}"
              f"{'; fitted on log(target)' if best.log_target else ''})")
        pte = best.predict(Xte)
        te = regression_metrics(yte, pte, unit)
        base = regression_metrics(yte, np.full(len(yte), ytr.mean()), unit)
        # every family on the test set too -- reported, never used to choose
        fam_test = {k: regression_metrics(yte, f.predict(Xte), unit)["RMSE"]
                    for k, f in fitted.items()}
        band = ErrorBand(best, cfg.ml.band_lower_pct, cfg.ml.band_upper_pct
                         ).calibrate(Xca, yca)
        cov = band.evaluate(Xte, yte)
        print(f"    test MAE {te['MAE']:.4g} {unit} | RMSE {te['RMSE']:.4g} | "
              f"R2 {te['R2']:.4f} | mean-baseline RMSE {base['RMSE']:.4g}")
        print(f"    P{cfg.ml.band_lower_pct:g}-P{cfg.ml.band_upper_pct:g} error "
              f"band: measured test coverage {cov['measured_coverage_test']*100:.1f}%"
              f", mean width {cov['mean_width']:.4g} {unit}")
        hard = difficult_cases(df, m, target, pte)
        surrogates[target], bands[target], fitted_all[target] = best, band, fitted
        out["targets"][target] = {
            "unit": unit, "selected_model": best.name,
            "log_target": best.log_target,
            "best_params": best.best_params,
            "cv_rmse_selected": best.cv_score_rmse,
            "cv_rmse_units": unit,
            "fitted_on": "log(target)" if best.log_target else "target",
            "model_comparison_cv_rmse": {k: v["cv_rmse"] for k, v in allres.items()},
            "model_comparison_test_rmse_physical_units": fam_test,
            "model_fit_seconds": {k: v["fit_seconds"] for k, v in allres.items()},
            "test": te, "mean_baseline_test": base,
            "calib": regression_metrics(yca, best.predict(Xca), unit),
            "error_band": cov, "difficult_cases": hard}
        _figure_difficult(cfg, df, m, target, pte, unit)
    _figure_parity(cfg, df, m, surrogates, bands)
    _figure_model_comparison(cfg, out)
    out["_surrogates"], out["_bands"], out["_fitted"], out["_masks"] = \
        surrogates, bands, fitted_all, m
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


def _figure_parity(cfg, df, m, surrogates, bands):
    n = len(surrogates)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 3.6))
    ax = np.atleast_1d(ax)
    for a, (t, s) in zip(ax, surrogates.items()):
        X = df.loc[m["test"], s.features]
        y = df.loc[m["test"], t].to_numpy()
        lo, p, hi = bands[t].predict_interval(X)
        a.vlines(y, lo, hi, color="tab:blue", lw=.6, alpha=.5)
        a.plot(y, p, "o", ms=3, color="tab:blue")
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel(f"simulator {t}"); a.set_ylabel(f"surrogate {t}")
        a.set_title(f"{s.name} (unseen reservoirs)", fontsize=9)
    fig.suptitle("Surrogate parity with empirical P5-P95 error bands", y=1.04)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "06_surrogate_parity.png")


def _figure_model_comparison(cfg, out):
    tg = list(out["targets"])
    fig, ax = plt.subplots(1, len(tg), figsize=(4.4 * len(tg), 3.6))
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, tg):
        c = out["targets"][t]["model_comparison_test_rmse_physical_units"]
        names = sorted(c, key=c.get)
        a.barh(names, [c[k] for k in names], color="tab:blue")
        a.set_xscale("log")
        a.set_xlabel(f"test RMSE [{out['targets'][t]['unit']}] (log scale)")
        a.set_title(f"{t} (selected: {out['targets'][t]['selected_model']})", fontsize=8)
    fig.suptitle("Model families: test RMSE on unseen reservoirs "
                 "(selection used grouped CV only)", y=1.03)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "07_model_comparison.png")


def _figure_difficult(cfg, df, m, target, pred, unit):
    d = df.loc[m["test"]]
    err = pred - d[target].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].scatter(d["k_median_mD"], err, s=10); ax[0].set_xscale("log")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xlabel("median permeability [mD]"); ax[0].set_ylabel(f"error [{unit}]")
    ax[1].scatter(d["q_mult_mean"], err, s=10, color="tab:orange"); ax[1].set_xscale("log")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("mean rate / reference rate [-]"); ax[1].set_ylabel(f"error [{unit}]")
    fig.suptitle(f"Where the {target} surrogate errs (test realisations)", y=1.03)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / f"08_errors_{target}.png")


# ==========================================================================
# 4. UNCERTAINTY
# ==========================================================================
def stage_uncertainty(cfg, df, ml, validation) -> dict:
    print("\n=== STAGE 4  Monte Carlo propagation and error sources ===")
    surr, m = ml["_surrogates"], ml["_masks"]
    v7 = validation["V6_V7_V8_radial_two_phase"]

    def _num(section, key, used):
        """Relative difference between the level closest to the one used for
        the dataset and the finest level of the V7 study."""
        ks = sorted(v7[section], key=float)
        fine = ks[0] if section == "r_near" else ks[-1]
        near = min(ks, key=lambda k: abs(float(k) - used))
        a, b = v7[section][near][key], v7[section][fine][key]
        return abs(a - b) / max(abs(b), 1e-30), f"V7 {section}: {near} (closest to the dataset's {used:g}) vs finest {fine}"

    num = {"dp_bh_max_MPa": _num("r_near", "dp_bh_max_MPa", cfg.grid.r_near_m),
           "r_plume_m95_m": _num("n_r", "r_plume_m95_m", cfg.grid.n_r),
           "sweep_efficiency": _num("n_r", "sweep_efficiency", cfg.grid.n_r)}
    out = {"targets": {}, "prior_note": "Monte Carlo over the ASSUMED prior "
           "(config scenarios ranges); P10/P90 follow 5-Uncertainty.pdf p.46"}
    X_all = df[FEATURES]
    for t, s in surr.items():
        mc = monte_carlo_propagate(s, X_all)
        rel, src = num.get(t, (0.0, "n/a"))
        tab = error_source_table(t, prior_std=float(df[t].std()),
                                 surrogate_rmse=ml["targets"][t]["test"]["RMSE"],
                                 numerical_rel_error=rel,
                                 reference_value=float(df[t].median()),
                                 unmodelled=UNMODELLED_PHYSICS)
        tab["numerical_error_source"] = src
        out["targets"][t] = {"monte_carlo_prior": mc, "error_sources": tab}
        print(f"  {t}: P90(low) {mc['P90_low_case']:.4g} P50 {mc['P50']:.4g} "
              f"P10(high) {mc['P10_high_case']:.4g} | shares input/surrogate/numerical "
              f"{tab['share_input_pct']:.0f}/{tab['share_surrogate_pct']:.0f}/"
              f"{tab['share_numerical_pct']:.0f}%")
    fig, ax = plt.subplots(1, len(surr), figsize=(4.2 * len(surr), 3.2))
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, surr):
        d = out["targets"][t]["error_sources"]
        a.bar(["input\n(assumed prior)", "surrogate", "numerical"],
              [d["input_uncertainty_std"], d["surrogate_error_rmse"],
               d["numerical_error_abs"]],
              color=["tab:blue", "tab:orange", "tab:green"])
        a.set_ylabel(TARGET_UNITS.get(t, "")); a.set_title(t, fontsize=9)
    fig.suptitle("Sources of uncertainty (5-Uncertainty.pdf); model bias from "
                 "omitted physics listed, not quantified", y=1.04)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "09_uncertainty_sources.png")
    return out


# ==========================================================================
# 5. CLASSIFICATION
# ==========================================================================
def stage_classifier(cfg, df, ml) -> dict:
    print("\n=== STAGE 5  pressure-limit screening (classification) ===")
    from .classify import run_binary, run_multiclass, traffic_light
    m, feats = ml["_masks"], ml["features"]
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    dp = df["dp_bh_max_MPa"].to_numpy(float)
    y = (dp > dp_lim).astype(int)
    reg_pred = ml["_surrogates"]["dp_bh_max_MPa"].predict(df.loc[m["test"], feats])
    b = run_binary(df, feats, y, m, rs=cfg.ml.random_state, n_splits=cfg.ml.n_splits,
                   regression_pred=reg_pred, dp_limit=dp_lim)
    b["dp_limit_MPa"] = dp_lim
    ts, td = b["test_selected_threshold"], b["test_default_threshold"]
    print(f"  positives: train {b['positive_rate_train']*100:.0f}%, "
          f"test {b['positive_rate_test']*100:.0f}% | selected {b['selected']}")
    print(f"  test ROC-AUC {ts.get('roc_auc', float('nan')):.3f} | default threshold: "
          f"recall {td['recall']:.2f} precision {td['precision']:.2f} | "
          f"high-recall threshold: recall {ts['recall']:.2f} precision {ts['precision']:.2f}")
    y3 = traffic_light(dp, dp_lim, cfg.ml.margin_warning_fraction)
    mc = run_multiclass(df, feats, y3, m, rs=cfg.ml.random_state,
                        n_splits=cfg.ml.n_splits, regression_pred=reg_pred,
                        dp_limit=dp_lim, warn_frac=cfg.ml.margin_warning_fraction)
    print(f"  traffic light ({mc['selected']}): test macro-F1 {mc['test']['macro_f1']:.3f}"
          f" | regression-surrogate banding {mc['test_regression_surrogate_banded']['macro_f1']:.3f}")
    _figure_classifier(cfg, b, mc)
    model = b.pop("_model"); curves = b.pop("_curves"); mc_model = mc.pop("_model")
    from sklearn.svm import SVC
    ml["_classifier"] = {"model": model, "threshold": b["threshold_selected"],
                         "uses_decision_function": isinstance(model[-1], SVC),
                         "dp_limit_MPa": dp_lim, "family": b["selected"],
                         "features": feats}
    ml["_traffic_light"] = mc_model
    return {"binary": b, "traffic_light": mc}


def _figure_classifier(cfg, b, mc):
    c = b["_curves"]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ax[0].plot(c["fpr"], c["tpr"]); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title(f"ROC, test (AUC {b['test_selected_threshold'].get('roc_auc', 0):.3f})")
    ax[1].plot(c["recall"], c["precision"])
    ax[1].axvline(b["target_recall"], color="r", ls="--", lw=1, label="target recall")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall, test"); ax[1].legend(fontsize=7)
    cm = np.array(mc["test"]["confusion_matrix"])
    ax[2].imshow(cm, cmap="Blues"); ax[2].grid(False)
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")
    lab = ["green", "amber", "red"]
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(lab)
    ax[2].set_yticks(range(3)); ax[2].set_yticklabels(lab)
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("simulated")
    ax[2].set_title(f"Traffic light ({mc['selected']}), test")
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "10_classifier.png")


# ==========================================================================
# 6. COST AND SPEED
# ==========================================================================
def stage_speed(cfg, df, ml, n_sim: int = 5, repeats: int = 30) -> dict:
    """Inference cost vs simulator cost for the same outputs, measured in
    this process on this machine (``time.perf_counter``).

    * batch: all test rows in one call per surrogate, features pre-built;
    * single: one row per call (median of ``repeats``), features pre-built;
    * single end-to-end: raw reservoir + schedule -> features -> 3 surrogates;
    * simulator: ``n_sim`` test scenarios re-run serially in this process.
    Data generation and training are reported separately, not amortised.
    """
    print("\n=== STAGE 6  inference cost vs simulation cost ===")
    from .scenarios import run_scenario, sample_realisations, make_schedule
    surr, m = ml["_surrogates"], ml["_masks"]
    X = df.loc[m["test"], FEATURES]
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
    reals = {r.realisation_id: r for r in sample_realisations(cfg)}
    rows = df.loc[m["test"]].head(n_sim)
    qcols = [f"q{i+1}_kg_s" for i in range(cfg.schedule.n_periods)]
    te2e, tsim = [], []
    for _, row in rows.iterrows():
        r = reals[int(row["realisation_id"])]
        rates = row[qcols].to_numpy(float)
        t0 = time.perf_counter()
        Xe = features_for_schedules(cfg, r, rates[None, :])
        for s in surr.values():
            s.predict(Xe)
        te2e.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_scenario(cfg, r, make_schedule(cfg, r, rates), want_series=False)
        tsim.append(time.perf_counter() - t0)
    sim = float(np.mean(tsim))
    res = {
        "hardware_note": "same machine, same process, 1 scenario at a time for "
                         "the simulator; scikit-learn/XGBoost default threading",
        "n_test_rows_batch": int(n),
        "surrogate_batch_seconds_total": t_batch,
        "surrogate_seconds_per_case_batched": t_batch / n,
        "surrogate_seconds_single_call_median": t_single,
        "surrogate_seconds_single_end_to_end_median": float(np.median(te2e)),
        "simulator_seconds_per_case_mean_serial": sim,
        "simulator_cases_timed": len(tsim),
        "simulator_seconds_per_case_dataset_mean": float(df.loc[m["test"], "wall_time_s"].mean()),
        "speedup_batched": sim / (t_batch / n),
        "speedup_single_call": sim / t_single,
        "speedup_single_end_to_end": sim / float(np.median(te2e)),
        "outputs_compared": "the three scalar QoIs (dp_bh_max, r_plume_m95, "
                            "sweep efficiency); the simulator also yields full "
                            "fields and time series the surrogate does not",
    }
    print(f"  simulator {sim:.2f} s/case | surrogate batched {t_batch/n:.2e} s/case "
          f"({res['speedup_batched']:.0f}x) | single call {t_single:.2e} s "
          f"({res['speedup_single_call']:.0f}x) | end-to-end single "
          f"{np.median(te2e):.2e} s ({res['speedup_single_end_to_end']:.0f}x)")
    return res


# ==========================================================================
# 7. INTERPRETATION AND UNSUPERVISED STRUCTURE
# ==========================================================================
def stage_interpret(cfg, df, ml) -> dict:
    print("\n=== STAGE 7  interpretation (model behaviour, not causality) ===")
    from .interpret import (cluster_regimes, describe_clusters, lime_explanation,
                            parse_profiles, pca_regimes, permutation_importances,
                            profile_structure, shap_summary,
                            tree_impurity_importance, what_if_rate_scaling)
    from .classify import traffic_light
    surr, m = ml["_surrogates"], ml["_masks"]
    feats = ml["features"]
    Xte = df.loc[m["test"], feats]
    out = {"disclaimer": "Permutation importance, SHAP and local explanations "
                         "describe how the fitted model uses its inputs; they "
                         "are not causal effects."}
    for t, s in surr.items():
        imp = permutation_importances(s.estimator, Xte, df.loc[m["test"], t].to_numpy(),
                                      feats, random_state=cfg.ml.random_state)
        imp.to_csv(Path(cfg.paths.metrics) / f"permutation_importance_{t}.csv", index=False)
        out[t] = {"permutation_top8": imp.head(8).to_dict("records")}
        ti = tree_impurity_importance(s.estimator, feats)
        if ti is not None:
            out[t]["impurity_top8"] = ti.head(8).to_dict("records")
    # SHAP on a tree ensemble for the pressure target (the selected model if
    # it is a tree ensemble, otherwise the best tree family of the comparison)
    t = "dp_bh_max_MPa"
    fam = ml["_fitted"][t]
    tree_name = next((k for k in sorted(fam, key=lambda k: fam[k].cv_score_rmse)
                      if k in ("xgboost", "gradient_boosting", "random_forest")), None)
    if tree_name:
        sv = shap_summary(fam[tree_name].estimator, Xte, feats)
        if sv is not None:
            vals, _ = sv
            mean_abs = np.abs(vals).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:12]
            out["shap"] = {"model": tree_name, "space": "log(dp)" if fam[tree_name].log_target else "dp",
                           "top": [{"feature": feats[i], "mean_abs_shap": float(mean_abs[i])}
                                   for i in order]}
            fig, a = plt.subplots(figsize=(5.4, 3.6))
            a.barh([feats[i] for i in order][::-1], mean_abs[order][::-1], color="tab:green")
            a.set_xlabel("mean |SHAP| (log dp space)"); a.set_title(f"SHAP - {tree_name}", fontsize=9)
            fig.tight_layout(); _save(fig, Path(cfg.paths.figures) / "11_shap_dp.png")
    # local explanation + what-if for one held-out case near the limit
    s = surr[t]
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    d = df.loc[m["test"]]
    i0 = int(np.argmin(np.abs(d[t].to_numpy() - dp_lim)))
    case = d.iloc[i0]
    out["local_case"] = {"scenario_id": case["scenario_id"], "simulated_dp_MPa": float(case[t]),
                         "predicted_dp_MPa": float(s.predict(d.iloc[[i0]][feats])[0])}
    out["local_case"]["lime_top6"] = lime_explanation(
        s.predict, case[feats].astype(float), df.loc[m["train"], feats],
        random_state=cfg.ml.random_state).head(6).to_dict("records")
    from .scenarios import sample_realisations
    r = {x.realisation_id: x for x in sample_realisations(cfg)}[int(case["realisation_id"])]
    q0 = case[[f"q{i+1}_kg_s" for i in range(cfg.schedule.n_periods)]].to_numpy(float)
    wi = what_if_rate_scaling(lambda R: s.predict(features_for_schedules(cfg, r, R)),
                              q0, dp_lim)
    out["local_case"]["what_if"] = {"largest_rate_factor_predicted_below_limit":
                                    wi["largest_factor_below_limit"],
                                    "limit_MPa": dp_lim}
    fig, a = plt.subplots(figsize=(5, 3.2))
    a.plot(wi["factors"], wi["predictions"]); a.axhline(dp_lim, color="r", ls="--")
    a.axvline(1.0, color="grey", lw=1)
    a.set_xlabel("schedule scaled by factor [-]"); a.set_ylabel("predicted dp_bh_max [MPa]")
    a.set_title(f"What-if for {case['scenario_id']}", fontsize=9)
    fig.tight_layout(); _save(fig, Path(cfg.paths.figures) / "12_what_if.png")

    # reservoir-description regimes: PCA + k-means
    res_feats = ["log10_k_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m", "n_g",
                 "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g", "log10_kh", "log10_pv"]
    uniq = df.drop_duplicates("realisation_id")
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
    # plume-shape analysis on the final saturation profiles
    if "final_Sg_profile" in df.columns:
        P = parse_profiles(df["final_Sg_profile"])
        y3 = traffic_light(df["dp_bh_max_MPa"].to_numpy(), dp_lim,
                           cfg.ml.margin_warning_fraction)
        ps = profile_structure(P, labels_ref=y3, random_state=cfg.ml.random_state)
        out["plume_shapes"] = {k: v for k, v in ps.items() if not k.startswith("_")}
        _figure_profiles(cfg, P, ps)
        print(f"  plume shapes: {ps['n_components_95pct']} PCs for 95% variance; "
              f"k-means k={ps['kmeans_on_pca']['k']} (silhouette "
              f"{ps['kmeans_on_pca']['silhouette']:.2f}); DBSCAN noise "
              f"{ps['dbscan']['n_noise']}")
    return out


def _figure_profiles(cfg, P, ps):
    rr = np.geomspace(1.0, 2000.0, P.shape[1])
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    lab = ps["_labels"]
    for k in np.unique(lab):
        ax[0].plot(rr, P[lab == k].mean(axis=0), label=f"cluster {k} (n={int((lab == k).sum())})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("radius [m]")
    ax[0].set_ylabel("mean final $S_g$"); ax[0].legend(fontsize=7)
    ax[0].set_title("Plume-shape clusters (k-means on PCA)")
    ax[1].scatter(ps["_kpca"][:, 0], ps["_kpca"][:, 1], c=lab, s=6, cmap="tab10")
    ax[1].set_title("Kernel PCA (RBF)")
    ax[2].scatter(ps["_tsne"][:, 0], ps["_tsne"][:, 1], c=lab, s=6, cmap="tab10")
    ax[2].set_title("t-SNE (colour = k-means cluster)")
    if "_umap" in ps:
        ax[3].scatter(ps["_umap"][:, 0], ps["_umap"][:, 1], c=lab, s=6, cmap="tab10")
        noise = ps["_dbscan_labels"] == -1
        ax[3].scatter(ps["_umap"][noise, 0], ps["_umap"][noise, 1], s=18,
                      facecolors="none", edgecolors="k", label="DBSCAN noise")
        ax[3].legend(fontsize=7)
    ax[3].set_title("UMAP (circles: DBSCAN noise)")
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "13_plume_shapes.png")


# ==========================================================================
# 8. ML-METHOD STUDIES
# ==========================================================================
def stage_studies(cfg, df, ml) -> dict:
    print("\n=== STAGE 8  ML-method studies (training realisations only) ===")
    from .studies import run_all
    t = "dp_bh_max_MPa"
    res = run_all(df, ml["features"], t, ml["_masks"], ml["_fitted"][t],
                  ml["_surrogates"][t], n_splits=cfg.ml.n_splits,
                  seed=cfg.ml.random_state)
    _figure_studies(cfg, res)
    return res


def _figure_studies(cfg, r):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    lc = r.get("learning_curve", {})
    if "n_train_rows" in lc:
        ax[0].plot(lc["n_train_rows"], lc["train_rmse"], "o-", label="training")
        ax[0].plot(lc["n_train_rows"], lc["cv_rmse"], "s-", label="grouped CV")
        ax[0].set_xlabel("training rows"); ax[0].set_ylabel("RMSE of log(dp)")
        ax[0].set_title("Learning curve (gradient boosting)"); ax[0].legend(fontsize=7)
    es = r.get("boosting_early_stopping", {})
    if "train_rmse_curve" in es:
        ax[1].plot(es["train_rmse_curve"], label="training")
        ax[1].plot(es["valid_rmse_curve"], label="grouped validation")
        ax[1].axvline(es["best_iteration"], color="r", ls="--", lw=1, label="early stop")
        ax[1].set_yscale("log"); ax[1].set_xlabel("boosting round")
        ax[1].set_ylabel("RMSE of log(dp)"); ax[1].set_title("XGBoost early stopping")
        ax[1].legend(fontsize=7)
    sc = r.get("scaler_comparison", {})
    if sc and "knn" in sc:
        names = list(sc["knn"])
        w = 0.25
        for i, mdl in enumerate(sc):
            ax[2].bar(np.arange(len(names)) + i * w, [sc[mdl][n] for n in names], w, label=mdl)
        ax[2].set_xticks(np.arange(len(names)) + w); ax[2].set_xticklabels(names)
        ax[2].set_ylabel("grouped-CV RMSE of log(dp)"); ax[2].set_title("Scaler comparison")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "14_ml_studies.png")


# ==========================================================================
# 9. CONSERVATION CHECK, OPEN BOUNDARY
# ==========================================================================
def stage_conservation_check(cfg, df) -> dict:
    print("\n=== STAGE 9a  conservation identity check ===")
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
    print("\n=== STAGE 9b  open-boundary variant: injected vs retained ===")
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
# 10. SCHEDULE SCREENING
# ==========================================================================
def stage_optimise(cfg, df, ml) -> dict:
    print("\n=== STAGE 10  schedule screening on unseen reservoirs ===")
    from .optimise import SurrogateBundle, optimise_schedule, resimulate
    from .scenarios import sample_realisations
    surr, bands = ml["_surrogates"], ml["_bands"]
    bundle = SurrogateBundle(features=ml["features"], pressure=surr["dp_bh_max_MPa"],
                             plume=surr["r_plume_m95_m"],
                             band_pressure=bands["dp_bh_max_MPa"],
                             band_plume=bands["r_plume_m95_m"])
    reals = {r.realisation_id: r for r in sample_realisations(cfg)}
    test_ids = [int(i) for i in ml["split"]["test_ids"]][:cfg.optim.n_test_realisations]
    print(f"  {len(test_ids)} test realisations: {test_ids}")
    rows, searches, t_search = [], [], 0.0
    for rid in test_ids:
        r = reals[rid]
        t0 = time.perf_counter()
        s = optimise_schedule(cfg, r, bundle)
        t_search += time.perf_counter() - t0
        searches.append(s)
        if s["baseline"] is not None:
            rows.append(resimulate(cfg, r, s["baseline"]["rates_kg_s"], "baseline_constant"))
        for j, c in enumerate(s["shortlist"][:3]):
            rows.append(resimulate(cfg, r, c["rates_kg_s"], f"screened_{j}"))
    sim = pd.DataFrame(rows)
    sim.to_csv(Path(cfg.paths.metrics) / "optimisation_resimulation.csv", index=False)
    ok = sim[sim.status == "ok"]
    summary = []
    for rid in test_ids:
        g = ok[ok.realisation_id == rid]
        b, o = g[g.label == "baseline_constant"], g[g.label != "baseline_constant"]
        of = o[~o.violates_pressure & ~o.violates_plume]
        bf = bool(len(b) and not b.violates_pressure.iloc[0] and not b.violates_plume.iloc[0])
        best = of.loc[of.sim_mass_retained_Mt.idxmax()] if len(of) else None
        row = {"realisation_id": rid,
               "baseline_mass_Mt": float(b.sim_mass_retained_Mt.iloc[0]) if len(b) else np.nan,
               "baseline_feasible_after_resimulation": bf,
               "n_screened_resimulated": int(len(o)),
               "n_screened_feasible": int(len(of)),
               "best_screened_mass_Mt": float(best.sim_mass_retained_Mt) if best is not None else np.nan,
               "best_dp_MPa": float(best.sim_dp_bh_max_MPa) if best is not None else np.nan,
               "best_rplume_m": float(best.sim_r_plume_m95_m) if best is not None else np.nan}
        row["improvement_pct"] = (100 * (row["best_screened_mass_Mt"] - row["baseline_mass_Mt"])
                                  / row["baseline_mass_Mt"]
                                  if bf and best is not None else np.nan)
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(Path(cfg.paths.metrics) / "optimisation_summary.csv", index=False)
    prop = ok[ok.label != "baseline_constant"]
    base = ok[ok.label == "baseline_constant"]
    n_viol = int((prop.violates_pressure | prop.violates_plume).sum())
    res = {"n_test_realisations": len(test_ids), "test_realisation_ids": test_ids,
           "use_error_band": bool(cfg.optim.use_error_band),
           "n_realisations_no_feasible_candidate": sum(1 for s in searches if s["n_feasible"] == 0),
           "mean_feasible_fraction": float(np.mean([s["feasible_fraction"] for s in searches])),
           "surrogate_screening_seconds_per_realisation": t_search / max(len(test_ids), 1),
           "n_resimulations": int(len(sim)),
           "n_failed_resimulations": int((sim.status != "ok").sum()),
           "n_screened_resimulated": int(len(prop)),
           "n_screened_violating": n_viol,
           "screened_violation_rate": n_viol / max(len(prop), 1),
           "n_baselines_violating": int((base.violates_pressure | base.violates_plume).sum()),
           "median_improvement_pct": float(np.nanmedian(sdf["improvement_pct"]))
           if sdf["improvement_pct"].notna().any() else None,
           "n_realisations_improved": int(np.nansum(sdf["improvement_pct"] > 0)),
           "n_realisations_compared": int(sdf["improvement_pct"].notna().sum()),
           "per_realisation": sdf.to_dict("records"),
           "searches": searches}
    print(f"  re-simulated {len(prop)} screened schedules: {n_viol} violated a stated "
          f"limit; baselines violating: {res['n_baselines_violating']}; failed runs: "
          f"{res['n_failed_resimulations']}")
    print(f"  median improvement over constant rate: {res['median_improvement_pct']}")
    _figure_optimisation(cfg, ok, sdf)
    return res


def _figure_optimisation(cfg, sim, sdf):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    for lab, mk, c in (("baseline_constant", "s", "tab:grey"), ("screened", "o", "tab:blue")):
        s = sim[sim.label == "baseline_constant"] if lab == "baseline_constant" \
            else sim[sim.label != "baseline_constant"]
        ax[0].scatter(s.sim_dp_bh_max_MPa, s.sim_mass_retained_Mt, marker=mk, s=26,
                      alpha=.8, color=c, label=lab)
    ax[0].axvline(dp_lim, color="r", ls="--", lw=1.2, label="stated dp limit")
    ax[0].set_xlabel(r"simulated $\Delta p_{bh,max}$ [MPa]")
    ax[0].set_ylabel("simulated retained CO$_2$ [Mt]")
    ax[0].set_title("Re-simulated schedules vs the stated limit"); ax[0].legend(fontsize=7)
    d = sdf.dropna(subset=["improvement_pct"])
    ax[1].bar(d.realisation_id.astype(str), d.improvement_pct, color="tab:blue")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("unseen realisation"); ax[1].set_ylabel("mass vs constant rate [%]")
    ax[1].set_title("Screened schedule vs constant-rate baseline")
    fig.tight_layout()
    _save(fig, Path(cfg.paths.figures) / "15_schedule_screening.png")


# ==========================================================================
def save_models(cfg, ml) -> dict:
    from .artifacts import save_bundle
    objs = {f"surrogate_{t}.joblib": s for t, s in ml["_surrogates"].items()}
    objs.update({f"band_{t}.joblib": b for t, b in ml["_bands"].items()})
    if "_classifier" in ml:
        objs["pressure_classifier.joblib"] = ml["_classifier"]
    man = save_bundle(cfg, objs, dataset_file=Path(cfg.paths.data) / "scenarios.csv",
                      extra={"test_realisation_ids": ml["split"]["test_ids"],
                             "calib_realisation_ids": ml["split"]["calib_ids"],
                             "selected_models": {t: s.name for t, s in ml["_surrogates"].items()}})
    print(f"\n  saved {len(objs)} model files + manifest to {cfg.paths.models}")
    return man
