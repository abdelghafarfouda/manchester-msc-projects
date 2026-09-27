"""Train one surrogate: data-only loss, or data + Rachford-Rice physics loss.

Run:  python scripts/train.py --physics 0 --seed 0
      python scripts/train.py --physics 1 --seed 0

Writes results/checkpoints/<tag>.pt and results/metrics/train_<tag>.json
"""

from __future__ import annotations

import argparse
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


def tensors(X, FV, z, K):
    return (
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(FV, dtype=torch.float32),
        torch.as_tensor(z, dtype=torch.float32),
        torch.as_tensor(K, dtype=torch.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", type=int, default=0, choices=(0, 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    args = ap.parse_args()

    tag = f"ffn_phys{args.physics}_s{args.seed}"
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    itr, iva = blob["idx_train"], blob["idx_val"]

    # Standardisation fitted on the training rows only.
    scaler = sdata.Standardiser().fit(blob["X"][itr])
    Xtr = scaler.transform(blob["X"][itr])
    Xva = scaler.transform(blob["X"][iva])

    tr = tensors(Xtr, blob["FV"][itr], blob["z"][itr], blob["K"][itr])
    va = tensors(Xva, blob["FV"][iva], blob["z"][iva], blob["K"][iva])

    lam_star, data_ref, phys_ref = snn.calibrate_lambda(tr[1], tr[2], tr[3])
    lam = lam_star if args.physics else 0.0

    snn.set_seed(args.seed)
    model = snn.simpleFFN(Xtr.shape[1], num_hidden=tuple(args.hidden)).to(snn.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(*tr), batch_size=args.batch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*va), batch_size=512,
                            shuffle=False, num_workers=0)

    history = []
    best = {"val_data": float("inf"), "epoch": -1, "state": None}
    t0 = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_data, tr_phys = snn.train_one_epoch(
            model, optimizer, criterion, train_loader, lam
        )
        va_loss, va_data, va_phys = snn.validate(model, criterion, val_loader, lam)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_mse": tr_data,
             "train_rr2": tr_phys, "val_loss": va_loss, "val_mse": va_data,
             "val_rr2": va_phys}
        )
        # Model selection on the validation data term only, so the two
        # variants are selected on the same quantity.
        if va_data < best["val_data"]:
            best = {"val_data": va_data, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in
                              model.state_dict().items()}}
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train mse {tr_data:.3e}  "
                  f"val mse {va_data:.3e}  val h^2 {va_phys:.3e}")
    wall = time.time() - t0

    model.load_state_dict(best["state"])
    ck = os.path.join(HERE, "results", "checkpoints")
    os.makedirs(ck, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "scaler": scaler.state_dict(),
         "args": vars(args),
         "lambda": lam,
         "n_inputs": int(Xtr.shape[1]),
         "hidden": list(args.hidden),
         "torch_version": torch.__version__},
        os.path.join(ck, tag + ".pt"),
    )

    report = {
        "tag": tag,
        "physics": bool(args.physics),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": list(args.hidden),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_train_rows": int(itr.size),
        "n_val_rows": int(iva.size),
        "lambda_used": lam,
        "lambda_calibration": {
            "lambda_star": lam_star,
            "trivial_predictor_mse": data_ref,
            "trivial_predictor_mean_h2": phys_ref,
            "note": "lambda* makes the two terms equal for the training-mean predictor",
        },
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_data"],
        "wall_seconds": wall,
        "device": snn.device,
        "torch_version": torch.__version__,
        "history": history,
    }
    out = os.path.join(HERE, "results", "metrics", f"train_{tag}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"{tag}: best val MSE {best['val_data']:.4e} at epoch {best['epoch']}, "
          f"{wall:.1f} s on {snn.device}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
