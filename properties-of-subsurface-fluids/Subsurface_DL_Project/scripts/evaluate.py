"""Evaluate the trained surrogates and draw the figures.

Run:  python scripts/evaluate.py

Reads every results/checkpoints/ffn_phys*_s*.pt, scores them on the held-out
test mixtures and on the pressure-extrapolation set, and writes
results/metrics/evaluation.json plus results/figures/*.png
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import data as sdata  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import nn as snn  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(HERE, "results", "figures")
MET = os.path.join(HERE, "results", "metrics")


def scores(true, pred, z, K):
    err = pred - true
    sse = float((err ** 2).sum())
    sst = float(((true - true.mean()) ** 2).sum())
    h = flash.rachford_rice(pred, z, K)
    return {
        "n": int(true.size),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "max_abs_err": float(np.abs(err).max()),
        "r2": float(1.0 - sse / sst),
        "mean_h2": float((h ** 2).mean()),
        "median_abs_h": float(np.median(np.abs(h))),
    }


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = snn.simpleFFN(ck["n_inputs"], num_hidden=tuple(ck["hidden"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    scaler = sdata.Standardiser().load_state_dict(ck["scaler"])
    return model, scaler, ck


def main():
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    ite = blob["idx_test"]
    itr = blob["idx_train"]

    sets = {
        "test": (blob["X"][ite], blob["FV"][ite], blob["z"][ite], blob["K"][ite]),
        "extrapolation": (blob["X_ood"], blob["FV_ood"], blob["z_ood"], blob["K_ood"]),
    }

    report = {"models": {}, "baselines": {}, "timing": {}, "sets": {}}
    for name, (X, y, z, K) in sets.items():
        report["sets"][name] = {"rows": int(y.size),
                                "realisations": int(np.unique(
                                    blob["realisation"][ite] if name == "test"
                                    else blob["realisation_ood"]).size)}

    # --- baseline: the training-mean vapour fraction -----------------------
    mean_FV = float(blob["FV"][itr].mean())
    for name, (X, y, z, K) in sets.items():
        report["baselines"].setdefault("training_mean_FV", {})[name] = scores(
            y, np.full_like(y, mean_FV), z, K
        )
    report["baselines"]["training_mean_FV"]["value"] = mean_FV

    # --- the models --------------------------------------------------------
    preds = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results", "checkpoints",
                                              "ffn_phys*_s*.pt"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        model, scaler, ck = load_model(path)
        entry = {"lambda": ck["lambda"], "physics": bool(ck["args"]["physics"]),
                 "seed": ck["args"]["seed"], "hidden": ck["hidden"],
                 "n_parameters": int(sum(p.numel() for p in model.parameters()))}
        for name, (X, y, z, K) in sets.items():
            p = snn.predict(model, scaler.transform(X)).astype(float)
            preds[(tag, name)] = p
            entry[name] = scores(y, p, z, K)
        report["models"][tag] = entry

    # --- aggregate over seeds ---------------------------------------------
    report["summary"] = {}
    for phys in (0, 1):
        tags = [t for t in report["models"] if report["models"][t]["physics"] == bool(phys)]
        if not tags:
            continue
        agg = {}
        for name in sets:
            for metric in ("rmse", "mae", "r2", "mean_h2", "median_abs_h"):
                vals = [report["models"][t][name][metric] for t in tags]
                agg[f"{name}_{metric}_mean"] = float(np.mean(vals))
                agg[f"{name}_{metric}_std"] = float(np.std(vals))
        agg["seeds"] = sorted(report["models"][t]["seed"] for t in tags)
        report["summary"]["physics" if phys else "data_only"] = agg

    # --- timing: the iteration the surrogate replaces ----------------------
    Xte, yte, zte, Kte = sets["test"]
    tag0 = sorted(report["models"])[0]
    model, scaler, _ = load_model(os.path.join(HERE, "results", "checkpoints",
                                               tag0 + ".pt"))
    Xs = scaler.transform(Xte)

    def timeit(fn, warmup=3, reps=15):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))

    t_solver, s_lo, s_hi = timeit(lambda: flash.solve_fv(zte, Kte))
    t_net, n_lo, n_hi = timeit(lambda: snn.predict(model, Xs))
    report["timing"] = {
        "rows": int(yte.size),
        "bisection_solver_seconds_median": t_solver,
        "bisection_solver_seconds_min_max": [s_lo, s_hi],
        "network_forward_seconds_median": t_net,
        "network_forward_seconds_min_max": [n_lo, n_hi],
        "speedup_median": t_solver / t_net if t_net > 0 else None,
        "note": ("median of 15 timed repetitions after 3 warm-up calls; both "
                 "vectorised over all test rows on the same CPU; the solver is "
                 "numpy bisection to 1e-12 (40 iterations), the network is a "
                 "single float32 forward pass on "
                 f"{torch.get_num_threads()} torch threads"),
        "device": snn.device,
        "torch_version": torch.__version__,
    }

    # ---------------------------------------------------------------- figures
    os.makedirs(FIG, exist_ok=True)
    hist = {}
    for phys in (0, 1):
        f = os.path.join(MET, f"train_ffn_phys{phys}_s0.json")
        if os.path.exists(f):
            hist[phys] = json.load(open(f))["history"]

    # fig 1 -- training curves
    if hist:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for phys, style in ((0, "-"), (1, "--")):
            if phys not in hist:
                continue
            e = [h["epoch"] for h in hist[phys]]
            lab = "data + physics loss" if phys else "data loss only"
            ax.semilogy(e, [h["train_mse"] for h in hist[phys]], style,
                        color="C0", alpha=0.8, label=f"train MSE, {lab}")
            ax.semilogy(e, [h["val_mse"] for h in hist[phys]], style,
                        color="C3", alpha=0.8, label=f"validation MSE, {lab}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"MSE in $F_V$  [$-$]")
        ax.set_title("Training history (seed 0)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig1_training_curves.png"), dpi=150)
        plt.close(fig)

    # fig 2 -- parity on the test mixtures
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for ax, phys in zip(axes, (0, 1)):
        tag = f"ffn_phys{phys}_s0"
        if (tag, "test") not in preds:
            continue
        p = preds[(tag, "test")]
        ax.plot([0, 1], [0, 1], "k-", lw=1)
        ax.plot(yte, p, ".", ms=2, alpha=0.3)
        s = report["models"][tag]["test"]
        ax.set_title(("data + physics loss" if phys else "data loss only")
                     + f"\nRMSE = {s['rmse']:.4f}, $R^2$ = {s['r2']:.3f}")
        ax.set_xlabel(r"$F_V$ from Rachford-Rice solver  [$-$]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$F_V$ predicted by network  [$-$]")
    fig.suptitle("Held-out test mixtures (seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_parity_test.png"), dpi=150)
    plt.close(fig)

    # fig 3 -- where the error sits
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    edges = np.linspace(0, 1, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for phys, style in ((0, "o-"), (1, "s--")):
        tag = f"ffn_phys{phys}_s0"
        if (tag, "test") not in preds:
            continue
        e = np.abs(preds[(tag, "test")] - yte)
        m = [np.sqrt((e[(yte >= a) & (yte < b)] ** 2).mean())
             if ((yte >= a) & (yte < b)).any() else np.nan
             for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(centres, m, style,
                label="data + physics loss" if phys else "data loss only")
    ax.set_xlabel(r"true $F_V$  [$-$]")
    ax.set_ylabel(r"RMSE in $F_V$ within bin  [$-$]")
    ax.set_title("Error against vapour fraction (test mixtures)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_error_vs_FV.png"), dpi=150)
    plt.close(fig)

    # fig 4 -- Rachford-Rice residual of the predictions
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, name in zip(axes, ("test", "extrapolation")):
        z, K = sets[name][2], sets[name][3]
        for phys, colour in ((0, "C0"), (1, "C1")):
            tag = f"ffn_phys{phys}_s0"
            if (tag, name) not in preds:
                continue
            h = np.abs(flash.rachford_rice(preds[(tag, name)], z, K))
            ax.hist(np.log10(np.maximum(h, 1e-8)), bins=50, alpha=0.55,
                    color=colour, density=True,
                    label="data + physics loss" if phys else "data loss only")
        ax.set_xlabel(r"$\log_{10}|h(\hat{F}_V)|$  [$-$]")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rachford-Rice residual of the predictions (seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_rr_residual.png"), dpi=150)
    plt.close(fig)

    # fig 5 -- learning curve, if the subset runs exist
    lc_path = os.path.join(MET, "learning_curve.json")
    if os.path.exists(lc_path):
        lc = json.load(open(lc_path))
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(lc["n_train_realisations"], lc["test_rmse"], "o-")
        ax.set_xscale("log")
        ax.set_xlabel("training mixtures (realisations)")
        ax.set_ylabel(r"test RMSE in $F_V$  [$-$]")
        ax.set_title("Is the surrogate limited by the network or by the data?")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig5_learning_curve.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(MET, "evaluation.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps({"summary": report["summary"],
                      "baseline": {k: v for k, v in
                                   report["baselines"]["training_mean_FV"].items()
                                   if k != "value"},
                      "timing": report["timing"]}, indent=2))


if __name__ == "__main__":
    main()
