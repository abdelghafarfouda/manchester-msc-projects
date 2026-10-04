"""Where do the six saved networks fail?  Error across the two-phase pressure window.

Run:  python scripts/analyse_errors.py [--out-dir DIR] [--fig-dir DIR]
Writes results/analysis/*.json, *.csv and results/figures/fig6-fig8.  No training.

Window position.  For every row the Wilson bubble- and dew-point pressures of
its own mixture at its own temperature are computed with the same sums the
notes use (pp. 12-14; ``sfp.flash.wilson_saturation_pressures``), and

    xi = ln(p / p_dew) / ln(p_bubble / p_dew)

``xi = 0`` at the dew point (all vapour, ``F_V = 1``) and ``xi = 1`` at the
bubble point (all liquid, ``F_V = 0``); it rises with pressure.

Bands.  The test rows are ranked by ``xi``: the 5 % with the smallest ``xi``
are the *dew band*, the 5 % with the largest the *bubble band*, and the rest
the *middle*.  The ``xi`` limits found that way are then applied unchanged to
the extrapolation set, whose rows all lie in the upper part of their windows
(2000-4000 psia), so that "near the dew point" means the same thing in both
sets.  The same rows are used for every model.  The profile figures use twenty
equal-count groups (5 % of each set's rows) along ``xi``.

Uncertainty.  Two different spreads are reported and never mixed:
* the population standard deviation over seeds 0, 1, 2 -- variability between
  training runs on the same data;
* a mixture bootstrap -- for a fixed pair of trained models, whole test
  mixtures (all their rows together) are resampled with replacement 2,000
  times and the paired RMSE difference recomputed.  Its 2.5-97.5 % range says
  how much the comparison depends on which 600 (or 401) mixtures happen to be
  in the evaluation set.  It is not a confidence interval for training a new
  model, and it does not include training-run variability.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components as C  # noqa: E402
from sfp import data as sdata  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import nn as snn  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BAND = 0.05                      # fraction of rows in each boundary band
N_GROUPS = 20                    # equal-count groups along the window
WORST = 0.01                     # "worst errors": the largest 1 % of |error|
N_BOOT = 2000
BOOT_SEED = 20261004
THREADS = 2                      # the recorded run's torch thread count


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = snn.simpleFFN(ck["n_inputs"], num_hidden=tuple(ck["hidden"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, sdata.Standardiser().load_state_dict(ck["scaler"]), ck


def overall(err, h):
    a = np.abs(err)
    q = np.quantile(a, [0.5, 0.9, 0.95, 0.99, 0.999])
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(a.mean()),
            "abs_err_q50": float(q[0]), "abs_err_q90": float(q[1]), "abs_err_q95": float(q[2]),
            "abs_err_q99": float(q[3]), "abs_err_q999": float(q[4]), "max_abs_err": float(a.max()),
            "mean_h2": float((h ** 2).mean()), "median_abs_h": float(np.median(np.abs(h))),
            "abs_h_q99": float(np.quantile(np.abs(h), 0.99))}


def band_limits(xi_test):
    """xi limits of the 5 % of test rows nearest each boundary (by rank)."""
    s = np.sort(xi_test, kind="stable")
    k = int(round(BAND * s.size))
    return float(s[k - 1]), float(s[s.size - k])


def bands_of(xi, limits):
    dew_max, bubble_min = limits
    band = np.full(xi.size, "middle", dtype=object)
    band[xi <= dew_max] = "dew"
    band[xi >= bubble_min] = "bubble"
    order = np.argsort(xi, kind="stable")
    groups = np.empty(xi.size, dtype=int)
    groups[order] = np.minimum((np.arange(xi.size) * N_GROUPS) // xi.size, N_GROUPS - 1)
    return band, groups


def rmse(e):
    return float(np.sqrt((e ** 2).mean())) if e.size else None


def boot_rmse_diff(err_a, err_b, mixture, rng):
    """Paired RMSE(b) - RMSE(a), whole mixtures resampled with replacement."""
    ids, inv = np.unique(mixture, return_inverse=True)
    sa = np.bincount(inv, weights=err_a ** 2)
    sb = np.bincount(inv, weights=err_b ** 2)
    cnt = np.bincount(inv)
    draws = rng.integers(0, ids.size, size=(N_BOOT, ids.size))
    na = cnt[draws].sum(1)
    d = np.sqrt(sb[draws].sum(1) / na) - np.sqrt(sa[draws].sum(1) / na)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "analysis"))
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "results", "figures"))
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    os.makedirs(args.out_dir, exist_ok=True)

    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    ite = blob["idx_test"]
    sets = {
        "test": dict(X=blob["X"][ite], FV=blob["FV"][ite], z=blob["z"][ite], K=blob["K"][ite],
                     p=blob["p_psia"][ite], T=blob["T_R"][ite], mix=blob["realisation"][ite]),
        "extrapolation": dict(X=blob["X_ood"], FV=blob["FV_ood"], z=blob["z_ood"], K=blob["K_ood"],
                              p=blob["p_ood_psia"], T=blob["T_ood_R"], mix=blob["realisation_ood"]),
    }
    with open(os.path.join(HERE, "configs", "prediction_domain.json"), encoding="utf-8") as fh:
        lim = json.load(fh)["limits"]
    for s in sets.values():
        pb, pd = flash.wilson_saturation_pressures(s["z"], s["T"], C.TC_RANKINE, C.PC_PSIA, C.OMEGA)
        s["xi"] = flash.window_position(s["p"], pb, pd)
    limits = band_limits(sets["test"]["xi"])
    for s in sets.values():
        s["band"], s["group"] = bands_of(s["xi"], limits)
        s["z_outside"] = ((s["z"] < np.array(lim["z_min"])) | (s["z"] > np.array(lim["z_max"]))).any(1)
        assert (s["xi"] > 0).all() and (s["xi"] < 1).all()      # two-phase rows only

    # ---- predictions of the six saved models --------------------------------------
    preds, meta = {}, {}
    for path in sorted(glob.glob(os.path.join(HERE, "results", "checkpoints", "ffn_phys*_s*.pt"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        model, scaler, ck = load_model(path)
        meta[tag] = {"physics": bool(ck["args"]["physics"]), "seed": int(ck["args"]["seed"])}
        for name, s in sets.items():
            preds[(tag, name)] = snn.predict(model, scaler.transform(s["X"])).astype(float)
    tags = sorted(meta)
    seeds = sorted({m["seed"] for m in meta.values()})

    report = {
        "definitions": {
            "xi": "ln(p/p_dew)/ln(p_bubble/p_dew) with Wilson saturation pressures of the row's own "
                  "mixture and temperature; 0 = dew point (F_V = 1), 1 = bubble point (F_V = 0)",
            "bands": (f"the {BAND:.0%} of test rows with the smallest xi (dew band) and the {BAND:.0%} "
                      "with the largest (bubble band), the rest the middle; the same xi limits are "
                      "applied to the extrapolation set"),
            "band_limits_xi": {"dew_band_xi_max": limits[0], "bubble_band_xi_min": limits[1]},
            "groups": f"{N_GROUPS} equal-count groups of each set's rows ranked by xi",
            "worst_errors": f"the {WORST:.0%} of rows with the largest absolute error, per model",
            "bootstrap": (f"{N_BOOT} resamples of whole mixtures with replacement (seed {BOOT_SEED}); "
                          "paired RMSE difference of fixed trained models; represents dependence on "
                          "which mixtures are in the evaluation set, not training variability"),
            "seed_spread": "population std (ddof=0) over seeds 0, 1, 2: training-run variability",
        },
        "sets": {}, "models": {}, "physics_vs_data_only": {}, "summary": {},
    }

    band_rows, group_rows, model_rows, pvd_rows = [], [], [], []
    rng = np.random.default_rng(BOOT_SEED)
    for name, s in sets.items():
        xi, band, group = s["xi"], s["band"], s["group"]
        report["sets"][name] = {
            "rows": int(xi.size), "mixtures": int(np.unique(s["mix"]).size),
            "xi_min": float(xi.min()), "xi_max": float(xi.max()), "xi_median": float(np.median(xi)),
            "rows_per_band": {b: int((band == b).sum()) for b in ("dew", "middle", "bubble")},
            "FV_mean_per_band": {b: (float(s["FV"][band == b].mean()) if (band == b).any() else None)
                                 for b in ("dew", "middle", "bubble")},
            "rows_with_a_mole_fraction_outside_training_ranges": int(s["z_outside"].sum()),
        }
        for tag in tags:
            err = preds[(tag, name)] - s["FV"]
            h = flash.rachford_rice(preds[(tag, name)], s["z"], s["K"])
            sse = float((err ** 2).sum())
            worst = np.argsort(np.abs(err), kind="stable")[::-1][: int(round(WORST * err.size))]
            entry = {"overall": overall(err, h), "bands": {}, "groups": []}
            for b in ("dew", "middle", "bubble"):
                m = band == b
                if not m.any():
                    entry["bands"][b] = {"rows": 0, "rmse": None, "mae": None, "abs_err_q99": None,
                                         "share_of_squared_error": 0.0,
                                         "share_of_worst_1pct_errors": 0.0, "mean_h2": None}
                else:
                    entry["bands"][b] = {
                        "rows": int(m.sum()),
                        "rmse": rmse(err[m]),
                        "mae": float(np.abs(err[m]).mean()),
                        "abs_err_q99": float(np.quantile(np.abs(err[m]), 0.99)),
                        "share_of_squared_error": float((err[m] ** 2).sum() / sse),
                        "share_of_worst_1pct_errors": float((band[worst] == b).mean()),
                        "mean_h2": float((h[m] ** 2).mean()),
                    }
                e = entry["bands"][b]
                band_rows.append([tag, name, b, e["rows"],
                                  "" if e["rmse"] is None else f"{e['rmse']:.6e}",
                                  "" if e["mae"] is None else f"{e['mae']:.6e}",
                                  f"{e['share_of_squared_error']:.6f}",
                                  f"{e['share_of_worst_1pct_errors']:.6f}",
                                  "" if e["mean_h2"] is None else f"{e['mean_h2']:.6e}"])
            for g in range(N_GROUPS):
                m = group == g
                entry["groups"].append({"group": g, "xi_lo": float(xi[m].min()), "xi_hi": float(xi[m].max()),
                                        "xi_median": float(np.median(xi[m])), "rows": int(m.sum()),
                                        "rmse": float(np.sqrt((err[m] ** 2).mean())),
                                        "share_of_squared_error": float((err[m] ** 2).sum() / sse)})
                group_rows.append([tag, name, g, f"{xi[m].min():.6f}", f"{xi[m].max():.6f}", int(m.sum()),
                                   f"{entry['groups'][-1]['rmse']:.6e}",
                                   f"{entry['groups'][-1]['share_of_squared_error']:.6f}"])
            inside = ~s["z_outside"]
            entry["composition_inside_vs_outside_training_ranges"] = {
                "rows_inside": int(inside.sum()), "rows_outside": int((~inside).sum()),
                "rmse_inside": float(np.sqrt((err[inside] ** 2).mean())),
                "rmse_outside": (float(np.sqrt((err[~inside] ** 2).mean()))
                                 if (~inside).any() else None),
            }
            o = entry["overall"]
            model_rows.append([tag, meta[tag]["physics"], meta[tag]["seed"], name, f"{o['rmse']:.6e}",
                               f"{o['mae']:.6e}", f"{o['abs_err_q95']:.6e}", f"{o['abs_err_q99']:.6e}",
                               f"{o['max_abs_err']:.6e}", f"{o['mean_h2']:.6e}",
                               f"{o['median_abs_h']:.6e}"])
            report["models"].setdefault(tag, {"physics": meta[tag]["physics"], "seed": meta[tag]["seed"]})[name] = entry

        # ---- physics loss against data loss, seed by seed --------------------------
        pvd = {}
        for seed in seeds:
            a, b = f"ffn_phys0_s{seed}", f"ffn_phys1_s{seed}"
            ea, eb = preds[(a, name)] - s["FV"], preds[(b, name)] - s["FV"]
            ra, rb = np.sqrt((ea ** 2).mean()), np.sqrt((eb ** 2).mean())
            d_sse = (ea ** 2) - (eb ** 2)                      # > 0 where the physics model is better
            total = d_sse.sum()
            boot = boot_rmse_diff(ea, eb, s["mix"], rng)
            pvd[str(seed)] = {
                "rmse_data_only": float(ra), "rmse_physics": float(rb),
                "rmse_change_percent": float(100 * (rb - ra) / ra),
                "squared_error_reduction_total": float(total),
                "share_of_squared_error_reduction_by_band": {
                    bnd: float(d_sse[band == bnd].sum() / total) + 0.0 for bnd in ("dew", "middle", "bubble")},
                "rmse_change_percent_by_band": {
                    bnd: (float(100 * (rmse(eb[band == bnd]) / rmse(ea[band == bnd]) - 1))
                          if (band == bnd).any() else None)
                    for bnd in ("dew", "middle", "bubble")},
                "bootstrap_rmse_difference_physics_minus_data": {
                    "q025": float(np.quantile(boot, 0.025)), "q500": float(np.quantile(boot, 0.5)),
                    "q975": float(np.quantile(boot, 0.975)),
                    "fraction_physics_better": float((boot < 0).mean())},
            }
            q = pvd[str(seed)]
            pvd_rows.append([name, seed, f"{ra:.6e}", f"{rb:.6e}", f"{q['rmse_change_percent']:.4f}",
                             f"{q['bootstrap_rmse_difference_physics_minus_data']['q025']:.6e}",
                             f"{q['bootstrap_rmse_difference_physics_minus_data']['q975']:.6e}",
                             f"{q['bootstrap_rmse_difference_physics_minus_data']['fraction_physics_better']:.4f}",
                             *(f"{q['share_of_squared_error_reduction_by_band'][b]:.6f}"
                               for b in ("dew", "middle", "bubble"))])
        report["physics_vs_data_only"][name] = pvd

    # ---- headline numbers ------------------------------------------------------------
    summ = {}
    for name in sets:
        for phys, label in ((False, "data_only"), (True, "physics")):
            ts = [t for t in tags if meta[t]["physics"] == phys]
            def per(b, key):
                return [report["models"][t][name]["bands"][b][key] for t in ts]
            summ[f"{name}_{label}"] = {
                "rmse_mean": float(np.mean([report["models"][t][name]["overall"]["rmse"] for t in ts])),
                "dew_band_share_of_squared_error_per_seed": per("dew", "share_of_squared_error"),
                "dew_band_share_of_squared_error_mean": float(np.mean(per("dew", "share_of_squared_error"))),
                "bubble_band_share_of_squared_error_per_seed": per("bubble", "share_of_squared_error"),
                "bubble_band_share_of_squared_error_mean": float(np.mean(per("bubble", "share_of_squared_error"))),
                "dew_band_share_of_worst_1pct_per_seed": per("dew", "share_of_worst_1pct_errors"),
                "bubble_band_share_of_worst_1pct_per_seed": per("bubble", "share_of_worst_1pct_errors"),
            }
        pv = report["physics_vs_data_only"][name]
        summ[f"{name}_physics_vs_data"] = {
            "seeds_where_physics_better": [int(k) for k, v in pv.items() if v["rmse_change_percent"] < 0],
            "rmse_change_percent_per_seed": [pv[str(k)]["rmse_change_percent"] for k in seeds],
            "dew_band_share_of_squared_error_reduction_per_seed":
                [pv[str(k)]["share_of_squared_error_reduction_by_band"]["dew"] for k in seeds],
        }
    report["summary"] = summ

    with open(os.path.join(args.out_dir, "error_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    for fname, header, rows in (
        ("metrics_by_model.csv", ["model", "physics", "seed", "set", "rmse", "mae", "abs_err_q95",
                                  "abs_err_q99", "max_abs_err", "mean_h2", "median_abs_h"], model_rows),
        ("boundary_bands.csv", ["model", "set", "band", "rows", "rmse", "mae", "share_of_squared_error",
                                "share_of_worst_1pct_errors", "mean_h2"], band_rows),
        ("window_profile.csv", ["model", "set", "group", "xi_lo", "xi_hi", "rows", "rmse",
                                "share_of_squared_error"], group_rows),
        ("physics_vs_data_only.csv", ["set", "seed", "rmse_data_only", "rmse_physics",
                                      "rmse_change_percent", "bootstrap_diff_q025", "bootstrap_diff_q975",
                                      "bootstrap_fraction_physics_better", "reduction_share_dew",
                                      "reduction_share_middle", "reduction_share_bubble"], pvd_rows),
    ):
        with open(os.path.join(args.out_dir, fname), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    figures(report, sets, preds, tags, meta, seeds, args.fig_dir)
    print(json.dumps(summ, indent=2))


def figures(report, sets, preds, tags, meta, seeds, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    colours = {False: "C0", True: "C1"}
    labels = {False: "data loss only", True: "data + Rachford-Rice loss"}

    # fig 6 -- RMSE along the window, every seed
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, name in zip(axes, sets):
        for tag in tags:
            g = report["models"][tag][name]["groups"]
            x = [r["xi_median"] for r in g]
            y = [r["rmse"] for r in g]
            ax.plot(x, y, "o-", color=colours[meta[tag]["physics"]], ms=3, lw=1.2, alpha=0.8,
                    label=f"{labels[meta[tag]['physics']]}" if meta[tag]["seed"] == 0 else None)
        ax.set_yscale("log")
        ax.set_xlim(0, 1) if name == "test" else ax.set_xlim(0.8, 1.0)
        ax.set_xlabel("position in the two-phase window, xi  [-]\n(0 = dew point, F_V = 1;  "
                      "1 = bubble point, F_V = 0)")
        ax.set_title(f"{name}: RMSE in each 5 % of rows (3 seeds per loss)" +
                     ("" if name == "test" else "\nall rows lie at xi > 0.8 (note the axis)"), fontsize=10)
        for lim_ in report["definitions"]["band_limits_xi"].values():
            ax.axvline(lim_, color="0.4", ls=":", lw=1)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMSE in F_V  [-]")
    axes[0].legend(fontsize=8, title="dotted: limits of the 5 % boundary bands", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_error_across_window.png"), dpi=150)
    plt.close(fig)

    # fig 7 -- where the squared error and the worst errors sit
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, name in zip(axes, sets):
        x = np.arange(len(tags))
        bottom = np.zeros(len(tags))
        for b, col in (("dew", "C3"), ("middle", "0.75"), ("bubble", "C2")):
            v = np.array([report["models"][t][name]["bands"][b]["share_of_squared_error"] for t in tags])
            ax.bar(x, v, bottom=bottom, color=col, width=0.6,
                   label={"dew": "dew band (5 % of test rows nearest the dew point)",
                          "middle": "middle",
                          "bubble": "bubble band (5 % of test rows nearest the bubble point)"}[b])
            bottom += v
        wd = [report["models"][t][name]["bands"]["dew"]["share_of_worst_1pct_errors"] for t in tags]
        wb = [report["models"][t][name]["bands"]["bubble"]["share_of_worst_1pct_errors"] for t in tags]
        ax.plot(x, wd, "kD", ms=6, label="worst 1 % of errors: share in the dew band")
        ax.plot(x, wb, "k^", ms=6, mfc="white", label="worst 1 % of errors: share in the bubble band")
        ax.set_xticks(x)
        ax.set_xticklabels([("physics" if meta[t]["physics"] else "data") + f"\nseed {meta[t]['seed']}"
                            for t in tags], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}: share of squared error by band", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("fraction  [-]")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(fig_dir, "fig7_boundary_share.png"), dpi=150)
    plt.close(fig)

    # fig 8 -- physics against data only, seed by seed, with the mixture bootstrap
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, name in zip(axes, sets):
        pv = report["physics_vs_data_only"][name]
        for i, seed in enumerate(seeds):
            q = pv[str(seed)]
            bt = q["bootstrap_rmse_difference_physics_minus_data"]
            d = q["rmse_physics"] - q["rmse_data_only"]
            ax.errorbar([i], [d], yerr=[[d - bt["q025"]], [bt["q975"] - d]], fmt="o", color="C1",
                        capsize=5, ms=7)
            ax.annotate(f"{q['rmse_change_percent']:+.1f} %", (i, d), textcoords="offset points",
                        xytext=(10, -3), fontsize=9)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_xlim(-0.5, len(seeds) - 0.3)
        ax.set_ylabel("RMSE(physics) - RMSE(data only)  [-]")
        ax.set_title(f"{name}: below 0 = physics loss better\n(bars: 2.5-97.5 % of mixture "
                     "resampling)", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig8_physics_vs_data.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
