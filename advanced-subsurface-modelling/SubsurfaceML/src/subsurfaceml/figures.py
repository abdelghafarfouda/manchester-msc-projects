"""Every figure of the pipeline and the experiments, with readable labels.

Plotting only: each function receives results that were computed elsewhere
and writes one PNG.  Labels come from :mod:`labels`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .labels import MODEL_NAMES, UNITS, VARIANT_NAMES, label, short

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

C_REV, C_BASE, C_PUB, C_ROM = "#1f6f8b", "#d1495b", "#8d8d8d", "#edae49"


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------- verification
def validation(cfg, res, fig_dir):
    from .fluids import FluidProperties, RelPerm, RockProperties, bl_profile_1d
    from .grid import CartesianGrid1D
    from .impes import InjectionSchedule, TwoPhaseModel
    from .units import MPA, md_to_m2
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
        ax[0].plot(g.centres, r.Sg[-1, 0], c, lw=1.2, label=f"IMPES, {n} cells")
    xg = np.linspace(0, L, 2000)
    ax[0].plot(xg, bl_profile_1d(rp, fl, xg, T, q_vol, A, phi, L=L), "k--",
               lw=1.4, label="Buckley-Leverett (Welge)")
    ax[0].set_xlabel("Distance from the inlet [m]")
    ax[0].set_ylabel("CO$_2$ saturation [-]")
    ax[0].set_title("V5 Buckley-Leverett profile, 0.4 pore volumes injected")
    ax[0].legend(fontsize=7)
    bl = res["V5_buckley_leverett"]["grids"]
    ns = np.array(sorted(int(k) for k in bl))
    l1 = np.array([bl[str(n)]["L1_saturation_error"] for n in ns])
    ax[1].loglog(ns, l1, "o-", label="IMPES L$_1$ error")
    ax[1].loglog(ns, l1[0] * ns[0] / ns, "k--", lw=1, label="first order")
    ax[1].set_xlabel("Number of cells")
    ax[1].set_ylabel("L$_1$ error in CO$_2$ saturation [-]")
    ax[1].set_title("V5 convergence")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "01_validation_buckley_leverett.png")

    r7 = res["V6_V7_V8_radial_two_phase"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for a, key, xl in zip(ax, ("n_r", "r_near", "max_dS"),
                          ("Radial cells per layer", "Well-block outer radius [m]",
                           "Largest saturation change per step [-]")):
        ks = sorted(r7[key], key=float)
        x = [float(k) for k in ks]
        a.plot(x, [r7[key][k]["dp_bh_max_MPa"] for k in ks], "o-")
        a2 = a.twinx()
        a2.plot(x, [r7[key][k]["r_plume_end_m"] for k in ks], "s--", color="tab:red")
        a2.grid(False)
        a.set_xlabel(xl)
        a.set_ylabel("Peak build-up [MPa]")
        a2.set_ylabel("Threshold plume radius [m]", color="tab:red")
        if key != "n_r":
            a.set_xscale("log")
    ax[0].set_title("V7 discretisation convergence (verification case)")
    fig.tight_layout()
    save(fig, Path(fig_dir) / "02_validation_convergence.png")


# ---------------------------------------------------------------------- EDA
def eda(cfg, df, fig_dir, features):
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    tg = ["dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency", "mass_retained_Mt"]
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.0))
    for a, t in zip(ax, tg):
        a.hist(df[t].dropna(), bins=30, color=C_REV, alpha=.85)
        a.set_xlabel(label(t)); a.set_ylabel("Number of cases")
    ax[0].axvline(dp_lim, color="r", ls="--", lw=1.2, label="assumed limit")
    ax[0].legend(fontsize=7)
    ax[1].axvline(cfg.optim.r_plume_limit_m, color="r", ls="--", lw=1.2)
    fig.suptitle("Simulated outcomes of the development cases (synthetic data)", y=1.04)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "03_eda_targets.png")

    feats = ["log10_k_mD", "V_DP", "log10_k_arith_mD", "h_total_m", "r_e_m",
             "log10_q_mult_mean", "q_front_load", "log10_rom_dp_MPa"]
    feats = [f for f in feats if f in df]
    fig, ax = plt.subplots(2, 4, figsize=(13.5, 5.6))
    for a, f in zip(ax.ravel(), feats):
        a.scatter(df[f], df["dp_bh_max_MPa"], s=7, alpha=.6, color=C_REV)
        a.set_xlabel(label(f), fontsize=7.5); a.set_ylabel("Peak build-up [MPa]")
        if f == "log10_rom_dp_MPa":
            a.set_yscale("log")
    fig.suptitle("Peak pressure build-up against the model inputs (development cases)",
                 y=1.01)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "04_eda_scatter.png")

    cols = list(features) + tg[:3]
    num = df[cols].corr()
    fig, a = plt.subplots(figsize=(10, 9))
    im = a.imshow(num.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(num))); a.set_xticklabels(num.columns, rotation=90, fontsize=6)
    a.set_yticks(range(len(num))); a.set_yticklabels(num.columns, fontsize=6)
    a.grid(False); fig.colorbar(im, shrink=.7, label="Pearson correlation")
    a.set_title("Correlation between inputs and outcomes (column names as in the data)")
    save(fig, Path(fig_dir) / "05_eda_correlation.png")


# ------------------------------------------------------------- surrogates
def parity(df_test, preds: dict, intervals: dict, fig_dir, title_suffix=""):
    """``preds[target] = (y, p)``; ``intervals[target] = (lo, hi)``."""
    n = len(preds)
    fig, ax = plt.subplots(1, n, figsize=(4.3 * n, 3.9))
    ax = np.atleast_1d(ax)
    for a, (t, (y, p)) in zip(ax, preds.items()):
        lo, hi = intervals[t]
        a.vlines(y, lo, hi, color=C_REV, lw=.6, alpha=.45)
        a.plot(y, p, "o", ms=2.5, color=C_REV)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel(f"Simulated: {short(t).lower()} [{UNITS[t]}]")
        a.set_ylabel(f"Surrogate [{UNITS[t]}]")
        a.set_title(short(t), fontsize=9)
    fig.suptitle("Revised surrogates on the untouched test reservoirs, with "
                 "90 % reservoir-calibrated intervals" + title_suffix, y=1.04)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "06_surrogate_parity.png")


def model_comparison(out_targets, fig_dir):
    tg = list(out_targets)
    fig, ax = plt.subplots(1, len(tg), figsize=(4.7 * len(tg), 3.8))
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, tg):
        c = out_targets[t]["model_comparison_cv_rmse"]
        names = sorted(c, key=c.get)
        a.barh([MODEL_NAMES.get(k, k) for k in names], [c[k] for k in names], color=C_REV)
        a.set_xscale("log")
        a.set_xlabel(f"Grouped-CV RMSE on training reservoirs [{UNITS[t]}]")
        a.set_title(f"{short(t)} (selected: "
                    f"{MODEL_NAMES.get(out_targets[t]['selected_model'])})", fontsize=8)
    fig.suptitle("Model families compared by grouped cross-validation "
                 "(the only basis of selection)", y=1.03)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "07_model_comparison.png")


def errors_vs_inputs(d, target, pred_rev, pred_base, fig_dir):
    y = d[target].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for p, c, lab in ((pred_base, C_BASE, "published approach (retrained)"),
                      (pred_rev, C_REV, "revised")):
        ax[0].scatter(d["k_median_mD"], p - y, s=9, color=c, alpha=.7, label=lab)
        if "realised_vs_prior_k" in d:
            ax[1].scatter(d["realised_vs_prior_k"], p - y, s=9, color=c, alpha=.7)
    for a in ax:
        a.set_xscale("log"); a.axhline(0, color="k", lw=1)
        a.set_ylabel(f"Prediction error [{UNITS[target]}]")
    ax[0].set_xlabel(label("k_median_mD"))
    ax[1].set_xlabel(label("realised_vs_prior_k"))
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Where the {short(target).lower()} surrogate errs "
                 "(untouched test reservoirs)", y=1.03)
    fig.tight_layout()
    save(fig, Path(fig_dir) / f"08_errors_{target}.png")


def uncertainty_sources(unc, fig_dir):
    surr = list(unc["targets"])
    fig, ax = plt.subplots(1, len(surr), figsize=(4.4 * len(surr), 3.3))
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, surr):
        d = unc["targets"][t]["error_sources"]
        a.bar(["input spread\n(assumed prior)", "surrogate\n(test RMSE)", "discretisation"],
              [d["input_uncertainty_std"], d["surrogate_error_rmse"],
               d["numerical_error_abs"]], color=[C_PUB, C_REV, C_ROM])
        a.set_ylabel(f"[{UNITS.get(t, '')}]"); a.set_title(short(t), fontsize=9)
    fig.suptitle("Sources of uncertainty in one unit; the bias from omitted "
                 "physics is listed in the report, not quantified", y=1.05)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "09_uncertainty_sources.png")


def classifier(b, mc, fig_dir):
    c = b["_curves"]
    fig, ax = plt.subplots(1, 3, figsize=(12.8, 3.5))
    ax[0].plot(c["fpr"], c["tpr"], color=C_REV); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("False-alarm rate"); ax[0].set_ylabel("Detection rate")
    ax[0].set_title(f"ROC on test reservoirs (AUC "
                    f"{b['test_selected_threshold'].get('roc_auc', 0):.3f})")
    ax[1].plot(c["recall"], c["precision"], color=C_REV)
    ax[1].axvline(b["target_recall"], color="r", ls="--", lw=1, label="target recall")
    ax[1].set_xlabel("Recall (exceedances detected)"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-recall on test reservoirs"); ax[1].legend(fontsize=7)
    cm = np.array(mc["test"]["confusion_matrix"])
    ax[2].imshow(cm, cmap="Blues"); ax[2].grid(False)
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")
    lab = ["green", "amber", "red"]
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(lab)
    ax[2].set_yticks(range(3)); ax[2].set_yticklabels(lab)
    ax[2].set_xlabel("Predicted class"); ax[2].set_ylabel("Simulated class")
    ax[2].set_title(f"Traffic light ({MODEL_NAMES.get(mc['selected'], mc['selected'])})")
    fig.tight_layout()
    save(fig, Path(fig_dir) / "10_classifier.png")


def shap_bar(feats, mean_abs, order, model_name, fig_dir):
    fig, a = plt.subplots(figsize=(6.2, 3.8))
    a.barh([feats[i] for i in order][::-1], mean_abs[order][::-1], color=C_REV)
    a.set_xlabel("Mean |SHAP value| (log build-up space)")
    a.set_title(f"SHAP importance, {MODEL_NAMES.get(model_name, model_name)} "
                "(model behaviour, not causality)", fontsize=9)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "11_shap_dp.png")


def what_if(wi, dp_lim, case_id, fig_dir):
    fig, a = plt.subplots(figsize=(5.2, 3.3))
    a.plot(wi["factors"], wi["predictions"], color=C_REV)
    a.axhline(dp_lim, color="r", ls="--", label="assumed limit")
    a.axvline(1.0, color="grey", lw=1)
    a.set_xlabel("Whole schedule scaled by factor [-]")
    a.set_ylabel("Predicted peak build-up [MPa]")
    a.set_title(f"What-if scaling for {case_id}", fontsize=9); a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "12_what_if.png")


def profiles(P, ps, fig_dir):
    rr = np.geomspace(1.0, 2000.0, P.shape[1])
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.5))
    lab = ps["_labels"]
    for k in np.unique(lab):
        ax[0].plot(rr, P[lab == k].mean(axis=0), label=f"cluster {k} (n={int((lab == k).sum())})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Radius [m]")
    ax[0].set_ylabel("Mean final CO$_2$ saturation [-]"); ax[0].legend(fontsize=7)
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
    for a in ax[1:]:
        a.set_xlabel("component 1"); a.set_ylabel("component 2")
    fig.tight_layout()
    save(fig, Path(fig_dir) / "13_plume_shapes.png")


def studies(r, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))
    lc = r.get("learning_curve", {})
    if "n_train_rows" in lc:
        ax[0].plot(lc["n_train_rows"], lc["train_rmse"], "o-", label="training")
        ax[0].plot(lc["n_train_rows"], lc["cv_rmse"], "s-", label="grouped CV")
        ax[0].set_xlabel("Training cases"); ax[0].set_ylabel("RMSE of log build-up [-]")
        ax[0].set_title("Learning curve (gradient boosting)"); ax[0].legend(fontsize=7)
    es = r.get("boosting_early_stopping", {})
    if "train_rmse_curve" in es:
        ax[1].plot(es["train_rmse_curve"], label="training")
        ax[1].plot(es["valid_rmse_curve"], label="grouped validation")
        ax[1].axvline(es["best_iteration"], color="r", ls="--", lw=1, label="early stop")
        ax[1].set_yscale("log"); ax[1].set_xlabel("Boosting round")
        ax[1].set_ylabel("RMSE of log build-up [-]"); ax[1].set_title("XGBoost early stopping")
        ax[1].legend(fontsize=7)
    sc = r.get("scaler_comparison", {})
    if sc and "knn" in sc:
        names = list(sc["knn"])
        w = 0.25
        for i, mdl in enumerate(sc):
            ax[2].bar(np.arange(len(names)) + i * w, [sc[mdl][n] for n in names], w,
                      label=MODEL_NAMES.get(mdl, mdl))
        ax[2].set_xticks(np.arange(len(names)) + w); ax[2].set_xticklabels(names)
        ax[2].set_ylabel("Grouped-CV RMSE of log build-up [-]")
        ax[2].set_title("Scaler comparison"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "14_ml_studies.png")


# ----------------------------------------------------------------- screening
def screening(cfg, recs, fig_dir):
    """``recs``: DataFrame, one row per (reservoir, method) with columns
    method, set, status, mass_Mt, constant_mass_Mt, n_simulations."""
    import pandas as pd
    order = ["simulator_constant", "rom_constant", "surrogate_published",
             "rom_shaped", "surrogate_verified"]
    names = {"simulator_constant": "constant rate,\nsimulator only",
             "rom_constant": "constant rate,\nROM start",
             "surrogate_published": "published\nscreening",
             "rom_shaped": "ROM-ranked\nschedules",
             "surrogate_verified": "revised:\nverified screening"}
    cols = {"simulator_constant": C_PUB, "rom_constant": C_ROM,
            "surrogate_published": C_BASE, "rom_shaped": "#9bc1bc",
            "surrogate_verified": C_REV}
    k = len(order)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
    for j, s in enumerate(("final_test", "shift")):
        d = recs[recs.set == s]
        if d.empty:
            continue
        ok_share = [float((d[d.method == m].status == "VERIFIED_FEASIBLE").mean())
                    for m in order]
        ax[0].bar(np.arange(k) + 0.38 * j, ok_share, 0.36,
                  color=[cols[m] for m in order], alpha=1.0 if j == 0 else 0.45,
                  edgecolor="k", lw=0.5, hatch=None if j == 0 else "//")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(facecolor="#555555", edgecolor="k", label="test reservoirs"),
                          Patch(facecolor="#555555", alpha=0.45, hatch="//", edgecolor="k",
                                label="distribution-shift reservoirs (10-30 mD)")],
                 fontsize=7, loc="lower left")
    ax[0].set_xticks(np.arange(k) + 0.19); ax[0].set_xticklabels([names[m] for m in order], fontsize=7)
    ax[0].set_ylabel("Share of reservoirs with a\nsimulator-verified schedule")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"Verified recommendations (budget {cfg.evaluation.screening_budget} "
                    "simulations per reservoir)", fontsize=9)
    d = recs[(recs.set == "final_test") & recs.mass_vs_constant_pct.notna()]
    data = [d[d.method == m].mass_vs_constant_pct.to_numpy() for m in order[1:]]
    ax[1].boxplot(data, showfliers=True)
    ax[1].set_xticks(range(1, k)); ax[1].set_xticklabels([names[m] for m in order[1:]], fontsize=7)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylabel("Verified mass vs simulator-only\nconstant rate [%]")
    ax[1].set_title("Gain over the simulator-only baseline (test reservoirs, "
                    "both verified)", fontsize=9)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "15_schedule_screening.png")


# ------------------------------------------------------------- experiments
def ablation(summary, fig_dir):
    m = summary["targets"]["dp_bh_max_MPa"]["metrics"]
    names = list(m)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for a, stat, lab in zip(ax, ("RMSE", "p95_abs_error", "worst_underprediction"),
                            ("RMSE [MPa]", "95th-percentile absolute error [MPa]",
                             "Worst under-prediction [MPa]")):
        vals = [m[n][stat] for n in names]
        cols = [C_REV if "V3" in n else C_ROM if "V4" in n else C_BASE if n.startswith("V0") else C_PUB
                for n in names]
        a.barh([VARIANT_NAMES.get(n, n) for n in names][::-1], vals[::-1], color=cols[::-1])
        a.set_xlabel(lab)
    ax[0].set_title("Out-of-fold, development reservoirs", fontsize=9)
    fig.suptitle("Ablation of the peak-pressure surrogate (nested reservoir-grouped "
                 "cross-validation, identical folds and search budget)", y=1.03)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "16_ablation_pressure.png")


def numerics(runs, df_cases, fig_dir, dp_limit):
    import pandas as pd
    ok = runs[runs.status == "ok"]
    w = ok.pivot(index="scenario_id", columns="level", values="dp_bh_max_MPa")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    levels = [c for c in ("n_r_x2", "r_near_half", "time_fine", "fine") if c in w]
    names = {"n_r_x2": "2x radial cells", "r_near_half": "well block / 2",
             "time_fine": "smaller time steps", "fine": "all refined"}
    rel = [100 * (w["production"] - w[lv]) / w[lv] for lv in levels]
    ax[0].boxplot([r.dropna().to_numpy() for r in rel])
    ax[0].set_xticks(range(1, len(levels) + 1)); ax[0].set_xticklabels([names[l] for l in levels])
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_ylabel("Production minus refined build-up [%]")
    ax[0].set_title("Discretisation effect on the training target", fontsize=9)
    k = df_cases.set_index("scenario_id").loc[w.index, "k_median_mD"]
    ax[1].scatter(k, 100 * (w["production"] - w["fine"]) / w["fine"], s=14, color=C_REV)
    ax[1].set_xscale("log"); ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel(label("k_median_mD"))
    ax[1].set_ylabel("Production minus fully refined [%]")
    ax[1].set_title("Is the error larger in tight reservoirs?", fontsize=9)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "17_numerics_dataset_cases.png")


def interval_coverage(comp, fig_dir, target="dp_bh_max_MPa"):
    """Case and whole-reservoir coverage of every interval method, revised
    design, test vs shift."""
    methods = ["empirical", "case_conformal", "reservoir_conformal", "adaptive_conformal"]
    names = ["empirical\nP5-P95", "case\nconformal", "reservoir\nconformal", "adaptive\nconformal"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, key, ttl in zip(ax, ("case_coverage", "reservoir_all_covered"),
                           ("Share of cases inside their interval",
                            "Share of reservoirs with every case inside")):
        for j, (s, alpha) in enumerate((("test", 1.0), ("shift", 0.5))):
            if s not in comp:
                continue
            v = [comp[s]["intervals"][m][key] for m in methods]
            a.bar(np.arange(4) + 0.38 * j, v, 0.36, color=C_REV, alpha=alpha,
                  edgecolor="k", lw=.5,
                  label="test reservoirs" if s == "test" else "distribution shift (10-30 mD)")
        a.axhline(0.9, color="r", ls="--", lw=1, label="nominal 90 %")
        a.set_xticks(np.arange(4) + 0.19); a.set_xticklabels(names, fontsize=7.5)
        a.set_ylim(0, 1.05); a.set_title(ttl, fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Interval coverage, {short(target).lower()}, revised surrogate", y=1.03)
    fig.tight_layout()
    save(fig, Path(fig_dir) / "18_interval_coverage.png")
