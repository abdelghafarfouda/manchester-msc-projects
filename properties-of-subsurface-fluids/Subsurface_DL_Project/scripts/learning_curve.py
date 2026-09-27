"""How much of the test error is the network and how much is the data?

Retrains the data-only network on nested subsets of the *training mixtures*
(never on subsets of rows, so a mixture is still never split), with everything
else held fixed, and records the test error.

Run:  python scripts/learning_curve.py
Writes results/metrics/learning_curve.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import data as sdata  # noqa: E402
from sfp import nn as snn  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FRACTIONS = (0.125, 0.25, 0.5, 1.0)
EPOCHS = 200
HIDDEN = (128, 128, 128)
SEED = 0


def main():
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    itr, iva, ite = blob["idx_train"], blob["idx_val"], blob["idx_test"]
    rid = blob["realisation"]
    train_groups = np.unique(rid[itr])
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_groups)

    out = {"n_train_realisations": [], "n_train_rows": [], "val_mse": [],
           "test_rmse": [], "epochs": EPOCHS, "hidden": list(HIDDEN),
           "seed": SEED, "torch_version": torch.__version__,
           "device": snn.device}

    for frac in FRACTIONS:
        keep = set(train_groups[: max(1, int(round(frac * train_groups.size)))].tolist())
        sub = itr[np.isin(rid[itr], list(keep))]

        scaler = sdata.Standardiser().fit(blob["X"][sub])
        def T(idx):
            return (torch.as_tensor(scaler.transform(blob["X"][idx]), dtype=torch.float32),
                    torch.as_tensor(blob["FV"][idx], dtype=torch.float32),
                    torch.as_tensor(blob["z"][idx], dtype=torch.float32),
                    torch.as_tensor(blob["K"][idx], dtype=torch.float32))

        tr, va = T(sub), T(iva)
        snn.set_seed(SEED)
        model = snn.simpleFFN(tr[0].shape[1], num_hidden=HIDDEN).to(snn.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        dl = DataLoader(TensorDataset(*tr), batch_size=64, shuffle=True)
        vl = DataLoader(TensorDataset(*va), batch_size=512, shuffle=False)

        best, best_state = float("inf"), None
        t0 = time.time()
        for _ in range(EPOCHS):
            snn.train_one_epoch(model, opt, crit, dl, 0.0)
            _, vd, _ = snn.validate(model, crit, vl, 0.0)
            if vd < best:
                best = vd
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        pred = snn.predict(model, scaler.transform(blob["X"][ite]))
        rmse = float(np.sqrt(((pred - blob["FV"][ite]) ** 2).mean()))
        out["n_train_realisations"].append(int(len(keep)))
        out["n_train_rows"].append(int(sub.size))
        out["val_mse"].append(best)
        out["test_rmse"].append(rmse)
        print(f"{len(keep):5d} mixtures ({sub.size:6d} rows): "
              f"val MSE {best:.4e}, test RMSE {rmse:.4f}  [{time.time() - t0:.0f} s]")

    with open(os.path.join(HERE, "results", "metrics", "learning_curve.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
