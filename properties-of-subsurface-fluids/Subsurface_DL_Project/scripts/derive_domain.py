"""Derive the network's training domain from the saved data, for the guarded predictor.

Run:  python scripts/derive_domain.py [--out PATH]
Writes configs/prediction_domain.json, read by sfp.predict.FlashSurrogate.

What is measured, from the training split of data/flash_dataset.npz and from
configs/experiment.json:

* pressure and temperature: the sampling window of the training band
  (2-2000 psia, 610-680 R).  They are drawn continuously across it
  (log-uniform pressure, uniform temperature), and the training rows fill it
  (their own minima and maxima are recorded alongside);
* each mole fraction: the range actually present in the training rows.  The
  generator could in principle produce 1/241 = 0.00415 to 40/46 = 0.870
  (integer mole charges 1-40), but the extremes need one charge at 40 and the
  other six at 1 and practically never occur, so the realised range is much
  narrower and is the one used.

It also names the model the guarded predictor loads by default, chosen by a
validation rule (lowest validation data MSE among the six reported models),
without test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components  # noqa: E402
from sfp import data as sdata  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REVIEW_LIMITS = {"p_max_psia": 2000.0, "T_R": [610.0, 680.0], "z": [0.004, 0.87]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "configs", "prediction_domain.json"))
    args = ap.parse_args()
    path = os.path.join(HERE, "data", "flash_dataset.npz")
    blob = np.load(path)
    with open(os.path.join(HERE, "configs", "experiment.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    dom = sdata.Domain()
    itr = blob["idx_train"]
    z, p, T, n = blob["z"][itr], blob["p_psia"][itr], blob["T_R"][itr], blob["moles"][itr]

    # the configuration and the saved data must agree before anything is derived
    assert cfg["dataset"]["split_fractions_by_mixture"] == [0.6, 0.2, 0.2]
    assert dom.p_min_psia <= p.min() and p.max() <= dom.p_max_train_psia
    assert dom.T_min_R <= T.min() and T.max() <= dom.T_max_R
    assert dom.moles_min <= n.min() and n.max() <= dom.moles_max

    n_other = components.N_COMPONENTS - 1
    z_lo_cfg = dom.moles_min / (dom.moles_min + n_other * dom.moles_max)
    z_hi_cfg = dom.moles_max / (dom.moles_max + n_other * dom.moles_min)
    z_min, z_max = z.min(axis=0), z.max(axis=0)

    def outside(zz):
        return int(((zz < z_min) | (zz > z_max)).any(axis=1).sum())

    models = {}
    for phys in (0, 1):
        for seed in (0, 1, 2):
            tag = f"ffn_phys{phys}_s{seed}"
            with open(os.path.join(HERE, "results", "metrics", f"train_{tag}.json"),
                      encoding="utf-8") as fh:
                models[tag] = json.load(fh)["best_val_mse"]
    default = min(models, key=models.get)

    out = {
        "_purpose": ("Training domain of the network and the default model of the guarded "
                     "predictor (src/sfp/predict.py). Written by scripts/derive_domain.py "
                     "from the training split; do not edit by hand."),
        "units": {"p": "psia", "T": "degrees Rankine (T[F] + 460)", "z": "mole fraction"},
        "components": list(components.NAMES),
        "limits": {
            "p_psia": [dom.p_min_psia, dom.p_max_train_psia],
            "T_R": [dom.T_min_R, dom.T_max_R],
            "z_min": [float(v) for v in z_min],
            "z_max": [float(v) for v in z_max],
            "p_max_tested_psia": dom.p_max_psia,
        },
        "basis": {
            "p_psia": ("training band of the sampling window (log-uniform draw); the "
                       "2000-4000 psia band was used only for the extrapolation test"),
            "T_R": "sampling window (uniform draw), 150-220 F",
            "z": ("minimum and maximum of each mole fraction over the training rows. The "
                  "checks are marginal: a state can pass every range and still be a "
                  "combination the training data never covered"),
        },
        "measured_on_training_rows": {
            "rows": int(itr.size),
            "mixtures": int(np.unique(blob["realisation"][itr]).size),
            "p_psia_min_max": [float(p.min()), float(p.max())],
            "T_R_min_max": [float(T.min()), float(T.max())],
            "z_min": [float(v) for v in z_min],
            "z_max": [float(v) for v in z_max],
            "moles_min_max": [int(n.min()), int(n.max())],
        },
        "configuration_implied_z_range": {
            "z_min": z_lo_cfg, "z_max": z_hi_cfg,
            "rule": (f"integer mole charges {dom.moles_min}-{dom.moles_max}: one component at "
                     "the minimum with the other six at the maximum, and the reverse"),
        },
        "review_limits_checked": {
            "stated": REVIEW_LIMITS,
            "pressure": "confirmed: training band ends at 2000 psia; the extrapolation test covers 2000-4000 psia",
            "temperature": "confirmed: 610-680 R",
            "composition": (f"the stated 0.004-0.87 is the range the configuration allows "
                            f"({z_lo_cfg:.5f}-{z_hi_cfg:.5f}); the training rows contain only "
                            f"{z_min.min():.4f}-{z_max.max():.4f} (per component "
                            f"{z_max.min():.3f}-{z_max.max():.3f} at the top), so the realised "
                            f"ranges are used"),
        },
        "rows_with_a_mole_fraction_outside_the_training_ranges": {
            "validation": outside(blob["z"][blob["idx_val"]]),
            "test": outside(blob["z"][blob["idx_test"]]),
            "extrapolation": outside(blob["z_ood"]),
        },
        "default_checkpoint": {
            "path": f"results/checkpoints/{default}.pt",
            "rule": ("lowest best validation data MSE among the six reported models "
                     "(results/metrics/train_*.json); no test data used"),
            "validation_mse": models,
        },
        "dataset_sha256": sha256(path),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("limits", "review_limits_checked",
                                          "rows_with_a_mole_fraction_outside_the_training_ranges")},
                     indent=2))
    print("default checkpoint:", out["default_checkpoint"]["path"])


if __name__ == "__main__":
    main()
