"""Bounded network-capacity comparison: 3 x 64, 3 x 128 and 3 x 192 hidden units.

The configuration is frozen in configs/capacity.json (committed before any of
its models was scored on the test or extrapolation sets).  Steps:

    python scripts/capacity.py provenance     # may the original 3 x 128 runs be reused?
    # the six new training runs, listed under "new_runs" in configs/capacity.json
    python scripts/capacity.py epoch-timing   # seconds per epoch, every width, one machine
    python scripts/capacity.py score          # selection (validation), then test and extrapolation

``score`` reads saved checkpoints only -- it never trains -- and writes
results/capacity/capacity_results.json, results/capacity/capacity_table.csv and
results/figures/fig9_capacity.png (or, with --out-dir DIR, the same names in DIR).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import data as sdata  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import nn as snn  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_PATH = os.path.join(HERE, "configs", "capacity.json")
CAP = os.path.join(HERE, "results", "capacity")


def load_cfg():
    with open(CFG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_paths(cfg, width, seed):
    """(checkpoint, training record) of one run: reused for 128, new otherwise."""
    if width == cfg["reused_runs"]["width"]:
        i = cfg["seeds"].index(seed)
        return (os.path.join(HERE, cfg["reused_runs"]["checkpoints"][i]),
                os.path.join(HERE, cfg["reused_runs"]["records"][i]))
    tag = f"ffn_h{width}_s{seed}"
    return (os.path.join(CAP, "checkpoints", tag + ".pt"),
            os.path.join(CAP, "metrics", f"train_{tag}.json"))


# --------------------------------------------------------------------------
def provenance(args):
    """Check that the original 3 x 128 data-only runs match the frozen configuration."""
    cfg = load_cfg()
    tr = cfg["training"]
    with open(os.path.join(HERE, "results", "metrics", "environment.json"), encoding="utf-8") as fh:
        env = json.load(fh)
    checks = []
    for seed in cfg["seeds"]:
        ck_path, rec_path = run_paths(cfg, 128, seed)
        with open(rec_path, encoding="utf-8") as fh:
            rec = json.load(fh)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        items = {
            "epochs": (rec["epochs"], tr["epochs"]),
            "batch": (rec["batch"], tr["batch"]),
            "lr": (rec["lr"], tr["lr"]),
            "hidden": (rec["hidden"], [128] * cfg["depth"]),
            "checkpoint hidden": (ck["hidden"], [128] * cfg["depth"]),
            "physics": (rec["physics"], False),
            "lambda": (rec["lambda_used"], 0.0),
            "seed": (rec["seed"], seed),
            "training rows": (rec["n_train_rows"], 36000),
            "validation rows": (rec["n_val_rows"], 12000),
            "torch version": (rec["torch_version"], torch.__version__),
            "torch threads (recorded run, environment.json)": (env["torch_threads"],
                                                                 tr["torch_threads"]),
        }
        for name, (got, want) in items.items():
            checks.append({"run": os.path.basename(ck_path), "item": name, "recorded": got,
                           "required": want, "match": got == want})
    ok = all(c["match"] for c in checks)
    out = {"reusable": ok, "checks": checks,
           "dataset_sha256": sha256(os.path.join(HERE, "data", "flash_dataset.npz")),
           "note": ("the original runs were selected on the same validation data MSE and saved "
                    "the scaler fitted on the training rows; the new runs use the same script")}
    os.makedirs(CAP, exist_ok=True)
    with open(os.path.join(CAP, "provenance_128.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for c in checks:
        if not c["match"]:
            print("MISMATCH", c)
    print("original 3 x 128 runs reusable:", ok)
    return 0 if ok else 1


# --------------------------------------------------------------------------
def epoch_timing(args):
    """Seconds per training epoch for every width, measured back to back on one machine.

    A measurement only: one warm-up epoch, then three timed epochs, with the
    same data, batch size, optimiser and thread count as the real runs.  The
    trained parameters are discarded.
    """
    cfg = load_cfg()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    itr = blob["idx_train"]
    scaler = sdata.Standardiser().fit(blob["X"][itr])
    tr = tuple(torch.as_tensor(a, dtype=torch.float32) for a in
               (scaler.transform(blob["X"][itr]), blob["FV"][itr], blob["z"][itr], blob["K"][itr]))
    out = {}
    for width in cfg["widths"]:
        snn.set_seed(0)
        model = snn.simpleFFN(tr[0].shape[1], num_hidden=(width,) * cfg["depth"])
        opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
        dl = DataLoader(TensorDataset(*tr), batch_size=cfg["training"]["batch"], shuffle=True)
        snn.train_one_epoch(model, opt, nn.MSELoss(), dl, 0.0)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            snn.train_one_epoch(model, opt, nn.MSELoss(), dl, 0.0)
            ts.append(time.perf_counter() - t)
        out[str(width)] = {"seconds_per_epoch_median": float(np.median(ts)),
                           "seconds_per_epoch_all": ts}
        print(f"width {width:4d}: {np.median(ts):.2f} s per epoch")
    os.makedirs(CAP, exist_ok=True)
    with open(os.path.join(CAP, "epoch_timing.json"), "w", encoding="utf-8") as fh:
        json.dump({"timing": out, "torch_threads": torch.get_num_threads(),
                   "torch_version": torch.__version__,
                   "note": "1 warm-up + 3 timed epochs per width, run back to back"}, fh, indent=2)
    return 0


# --------------------------------------------------------------------------
def scores(true, pred, z, K):
    err = pred - true
    h = flash.rachford_rice(pred, z, K)
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "r2": float(1.0 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()),
            "mean_h2": float((h ** 2).mean())}


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = snn.simpleFFN(ck["n_inputs"], num_hidden=tuple(ck["hidden"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, sdata.Standardiser().load_state_dict(ck["scaler"]), ck


def score(args):
    cfg = load_cfg()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    out_dir = args.out_dir or CAP
    fig_dir = args.fig_dir or os.path.join(HERE, "results", "figures")
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    iva, ite = blob["idx_val"], blob["idx_test"]
    sets = {
        "validation": (blob["X"][iva], blob["FV"][iva], blob["z"][iva], blob["K"][iva]),
        "test": (blob["X"][ite], blob["FV"][ite], blob["z"][ite], blob["K"][ite]),
        "extrapolation": (blob["X_ood"], blob["FV_ood"], blob["z_ood"], blob["K_ood"]),
    }
    with open(os.path.join(CAP, "epoch_timing.json"), encoding="utf-8") as fh:
        epoch_t = json.load(fh)["timing"]

    # ---- 1. selection, from the training records' validation MSE only ------------
    runs, by_width = {}, {}
    for width in cfg["widths"]:
        vals = []
        for seed in cfg["seeds"]:
            ck_path, rec_path = run_paths(cfg, width, seed)
            with open(rec_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            hist = rec["history"]
            best = rec["best_epoch"]
            runs[f"h{width}_s{seed}"] = {
                "width": width, "seed": seed,
                "checkpoint": os.path.relpath(ck_path, HERE),
                "reused_original_run": width == cfg["reused_runs"]["width"],
                "n_parameters": rec["n_parameters"],
                "best_epoch": best,
                "best_val_mse": rec["best_val_mse"],
                "train_mse_at_best_epoch": hist[best]["train_mse"],
                "val_mse_last_epoch": hist[-1]["val_mse"],
                "train_mse_last_epoch": hist[-1]["train_mse"],
                "timing": {"training_wall_seconds": rec["wall_seconds"],
                           "training_threads": rec.get("torch_threads", 2)},
            }
            vals.append(rec["best_val_mse"])
        by_width[width] = {"mean_best_val_mse": float(np.mean(vals)),
                           "std_best_val_mse": float(np.std(vals)),
                           "best_val_mse_per_seed": vals}
    tol = cfg["selection_rule"]["tolerance_relative"]
    lowest = min(v["mean_best_val_mse"] for v in by_width.values())
    eligible = [w for w in cfg["widths"] if by_width[w]["mean_best_val_mse"] <= (1.0 + tol) * lowest]
    selected = min(eligible)
    selection = {
        "rule": cfg["selection_rule"]["rule"],
        "statistic": cfg["selection_rule"]["statistic"],
        "lowest_mean_best_val_mse": lowest,
        "threshold": (1.0 + tol) * lowest,
        "eligible_widths": eligible,
        "selected_width": selected,
        "original_width": cfg["reused_runs"]["width"],
        "selected_is_original": selected == cfg["reused_runs"]["width"],
    }

    # ---- 2. every model on validation (recomputed), test and extrapolation ------------
    for key, r in runs.items():
        model, scaler, ck = load_model(os.path.join(HERE, r["checkpoint"]))
        assert ck["hidden"] == [r["width"]] * cfg["depth"] and not ck["args"]["physics"]
        for name, (X, y, z, K) in sets.items():
            r[name] = scores(y, snn.predict(model, scaler.transform(X)).astype(float), z, K)
        # the recomputed validation MSE must equal the training record's checkpoint value
        r["validation_mse_recomputed"] = r["validation"]["rmse"] ** 2

    summary = {}
    for width in cfg["widths"]:
        keys = [k for k in runs if runs[k]["width"] == width]
        s = {"n_parameters": runs[keys[0]]["n_parameters"], **by_width[width]}
        for name in ("test", "extrapolation"):
            for m in ("rmse", "mae", "r2", "mean_h2"):
                v = [runs[k][name][m] for k in keys]
                s[f"{name}_{m}_mean"] = float(np.mean(v))
                s[f"{name}_{m}_std"] = float(np.std(v))
        s["timing"] = {
            "training_wall_seconds_per_run": [runs[k]["timing"]["training_wall_seconds"] for k in keys],
            "seconds_per_epoch_one_machine": epoch_t[str(width)]["seconds_per_epoch_median"],
        }
        summary[str(width)] = s

    # ---- 3. inference time, whole test set, same machine and session ------------
    Xte = sets["test"][0]
    for width in cfg["widths"]:
        model, scaler, _ = load_model(os.path.join(HERE, runs[f"h{width}_s0"]["checkpoint"]))
        Xs = scaler.transform(Xte)
        for _ in range(3):
            snn.predict(model, Xs)
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            snn.predict(model, Xs)
            ts.append(time.perf_counter() - t)
        summary[str(width)]["timing"]["inference_seconds_test_set_median"] = float(np.median(ts))

    report = {
        "config": "configs/capacity.json",
        "selection": selection,
        "summary_by_width": summary,
        "runs": runs,
        "notes": {
            "statistics": "mean and population std (ddof=0) over seeds 0, 1, 2; the std measures training-run variability only",
            "test_set": cfg["test_set_status"],
            "physics_loss": "not part of this comparison; its effect was measured only at 3 x 128",
        },
        "run": {"torch_version": torch.__version__, "torch_threads": torch.get_num_threads()},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "capacity_results.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "capacity_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["width", "n_parameters", "mean_best_val_mse", "std_best_val_mse",
                    "test_rmse_mean", "test_rmse_std", "extrapolation_rmse_mean",
                    "extrapolation_rmse_std", "test_mean_h2_mean", "selected"])
        for width in cfg["widths"]:
            s = summary[str(width)]
            w.writerow([width, s["n_parameters"], f"{s['mean_best_val_mse']:.6e}",
                        f"{s['std_best_val_mse']:.6e}", f"{s['test_rmse_mean']:.6e}",
                        f"{s['test_rmse_std']:.6e}", f"{s['extrapolation_rmse_mean']:.6e}",
                        f"{s['extrapolation_rmse_std']:.6e}", f"{s['test_mean_h2_mean']:.6e}",
                        width == selected])
    figure(cfg, runs, summary, selected, os.path.join(fig_dir, "fig9_capacity.png"))
    print(json.dumps(selection, indent=2))
    for width in cfg["widths"]:
        s = summary[str(width)]
        print(f"3 x {width:3d}: {s['n_parameters']:6d} parameters | val MSE {s['mean_best_val_mse']:.3e} "
              f"+/- {s['std_best_val_mse']:.1e} | test RMSE {s['test_rmse_mean']:.5f} +/- "
              f"{s['test_rmse_std']:.5f} | extrapolation RMSE {s['extrapolation_rmse_mean']:.5f} "
              f"+/- {s['extrapolation_rmse_std']:.5f} | {s['timing']['seconds_per_epoch_one_machine']:.2f} s/epoch, "
              f"inference {1e3 * s['timing']['inference_seconds_test_set_median']:.1f} ms")
    return 0


def figure(cfg, runs, summary, selected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = cfg["widths"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (key, label) in zip(axes, (("best_val_mse", "best validation MSE  [-]"),
                                       ("test", "test RMSE  [-]"),
                                       ("extrapolation", "extrapolation RMSE  [-]"))):
        for seed, marker in zip(cfg["seeds"], ("o", "s", "^")):
            ys = [runs[f"h{w}_s{seed}"][key] if key == "best_val_mse"
                  else runs[f"h{w}_s{seed}"][key]["rmse"] for w in widths]
            ax.plot(widths, ys, marker + "-", color="0.55", lw=1, ms=5, label=f"seed {seed}")
        means = [summary[str(w)]["mean_best_val_mse"] if key == "best_val_mse"
                 else summary[str(w)][f"{key}_rmse_mean"] for w in widths]
        ax.plot(widths, means, "D-", color="C0", lw=2, ms=7, label="mean of 3 seeds")
        ax.set_xticks(widths)
        ax.set_xticklabels([f"3 x {w}\n{summary[str(w)]['n_parameters']:,} params" for w in widths])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].axhline((1 + cfg["selection_rule"]["tolerance_relative"]) *
                    min(summary[str(w)]["mean_best_val_mse"] for w in widths),
                    color="C3", ls="--", lw=1, label="selection threshold (+10 %)")
    axes[0].set_title(f"Selection (validation): 3 x {selected} selected", fontsize=10)
    axes[1].set_title("Held-out test mixtures (established benchmark)", fontsize=10)
    axes[2].set_title("Pressure extrapolation, 2000-4000 psia", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Data-only network: hidden-layer width (3 hidden layers, same training rules)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("provenance")
    sub.add_parser("epoch-timing")
    sp = sub.add_parser("score")
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    return {"provenance": provenance, "epoch-timing": epoch_timing, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
