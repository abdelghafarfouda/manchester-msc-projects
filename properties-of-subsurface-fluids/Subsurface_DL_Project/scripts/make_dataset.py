"""Generate the flash dataset and the grouped train / validation / test split.

Run:  python scripts/make_dataset.py
Writes data/flash_dataset.npz and results/metrics/dataset.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import data  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-realisations", type=int, default=3000)
    ap.add_argument("--states-per-realisation", type=int, default=20)
    ap.add_argument("--n-extrapolation", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20240920)
    args = ap.parse_args()

    domain = data.Domain()

    ds = data.build_dataset(
        args.n_realisations,
        domain,
        seed=args.seed,
        states_per_realisation=args.states_per_realisation,
    )
    # Pressure-extrapolation set: same mixture sampling, but pressures above
    # anything in training.  Mixtures with no two-phase state in that band are
    # simply absent -- at high pressure many mixtures are single phase.
    ood = data.build_dataset(
        args.n_extrapolation,
        domain,
        seed=args.seed + 1,
        states_per_realisation=args.states_per_realisation,
        p_range_psia=(domain.p_max_train_psia, domain.p_max_psia),
        max_attempts=400,
    )

    idx_tr, idx_va, idx_te = data.group_split(
        ds["realisation"], (0.6, 0.2, 0.2), seed=args.seed
    )

    X = data.features(ds)
    X_ood = data.features(ood)

    np.savez_compressed(
        os.path.join(HERE, "data", "flash_dataset.npz"),
        X=X, FV=ds["FV"], z=ds["z"], K=ds["K"], moles=ds["moles"],
        p_psia=ds["p_psia"], T_R=ds["T_R"], realisation=ds["realisation"],
        idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te,
        X_ood=X_ood, FV_ood=ood["FV"], z_ood=ood["z"], K_ood=ood["K"],
        p_ood_psia=ood["p_psia"], T_ood_R=ood["T_R"],
        realisation_ood=ood["realisation"],
    )

    def part(name, idx):
        return {
            "rows": int(idx.size),
            "realisations": int(np.unique(ds["realisation"][idx]).size),
            "FV_min": float(ds["FV"][idx].min()),
            "FV_mean": float(ds["FV"][idx].mean()),
            "FV_max": float(ds["FV"][idx].max()),
            "p_psia_min": float(ds["p_psia"][idx].min()),
            "p_psia_max": float(ds["p_psia"][idx].max()),
            "T_R_min": float(ds["T_R"][idx].min()),
            "T_R_max": float(ds["T_R"][idx].max()),
        }

    report = {
        "domain": domain.as_dict(),
        "feature_names": data.feature_names(),
        "seed": args.seed,
        "states_requested_per_realisation": args.states_per_realisation,
        "realisations_requested": args.n_realisations,
        "realisations_with_at_least_one_state": int(
            (ds["kept_per_realisation"] > 0).sum()
        ),
        "rows_total": int(ds["FV"].size),
        "split": "by realisation (60 / 20 / 20 of mixtures)",
        "composition_rule": domain.as_dict()["composition_rule"],
        "units": {"p": "psia", "T": "degrees Rankine (T[F] + 460)", "FV": "dimensionless"},
        "train": part("train", idx_tr),
        "val": part("val", idx_va),
        "test": part("test", idx_te),
        "extrapolation": {
            "p_range_psia": list(ood["p_range_psia"]),
            "realisations_requested": args.n_extrapolation,
            "realisations_with_at_least_one_state": int(
                (ood["kept_per_realisation"] > 0).sum()
            ),
            "rows": int(ood["FV"].size),
            "FV_mean": float(ood["FV"].mean()),
        },
        "leakage_check_train_test_realisation_overlap": int(
            np.intersect1d(
                ds["realisation"][idx_tr], ds["realisation"][idx_te]
            ).size
        ),
    }
    out = os.path.join(HERE, "results", "metrics", "dataset.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
