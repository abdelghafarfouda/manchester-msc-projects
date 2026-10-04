"""How much can rounding of the network's float32 output move each recorded number?

Run:  python scripts/ulp_sensitivity.py
Writes results/reproducibility/ulp_sensitivity.json.  No training.

The networks compute in float32, so their outputs are known only to one unit in
the last place (ULP): up to 6e-8 near F_V = 1.  A different CPU, instruction set
or PyTorch build can move a prediction by about that much.  This script moves
*every* prediction of every saved model by k ULPs (k = 1 and 3, random sign,
fixed seed), recomputes each output that CI compares, and records how far each
number moved.  The tolerances of scripts/compare_results.py are set from these
measurements, with a margin, so that CI fails on a real change but not on
float32 rounding.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from sfp import nn as snn  # noqa: E402

import compare_results as cr  # noqa: E402

_ORIGINAL_PREDICT = snn.predict


def perturbing_predict(k, rng):
    def predict(model, X):
        p = _ORIGINAL_PREDICT(model, X)
        for _ in range(k):
            direction = np.where(rng.random(p.size) < 0.5, -np.inf, np.inf).astype(np.float32)
            p = np.nextafter(p, direction.reshape(p.shape))
        return p
    return predict


def leaves(obj, path=()):
    if isinstance(obj, dict):
        for key, v in obj.items():
            if key not in cr.SKIP_KEYS:
                yield from leaves(v, path + (key,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, path + (i,))
    else:
        yield path, obj


def csv_leaves(path):
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    for r, row in enumerate(rows[1:]):
        for col, cell in zip(rows[0], row):
            yield (r, col), cr.parse_cell(cell)


def profile_of(rec_path):
    for prefix, prof in cr.FILE_PROFILES.items():
        if rec_path == prefix or rec_path.startswith(prefix + "/"):
            return prof
    return "metrics"


def worst(recorded, new, profile):
    rtol, atol = cr.PROFILES[profile]
    a, b = dict(recorded), dict(new)
    out = {"profile": profile, "rtol": rtol, "atol": atol, "max_abs": 0.0, "max_rel": 0.0,
           "where_abs": None, "where_rel": None, "changed": 0,
           "largest_fraction_of_tolerance_used": 0.0}
    for k, va in a.items():
        vb = b.get(k)
        if not (cr.is_number(va) and cr.is_number(vb)) or va == vb:
            continue
        out["changed"] += 1
        d = abs(vb - va)
        if d > out["max_abs"]:
            out["max_abs"], out["where_abs"] = d, ".".join(map(str, k))
        if va != 0 and d / abs(va) > out["max_rel"]:
            out["max_rel"], out["where_rel"] = d / abs(va), ".".join(map(str, k))
        out["largest_fraction_of_tolerance_used"] = max(
            out["largest_fraction_of_tolerance_used"], d / (atol + rtol * abs(va)))
    out["within_tolerance"] = out["largest_fraction_of_tolerance_used"] <= 1.0
    return out


def run_outputs(tmp):
    """Run every compared output into tmp with the current (patched) snn.predict."""
    import analyse_errors
    import capacity
    import evaluate
    import guarded_demo
    with contextlib.redirect_stdout(io.StringIO()):
        sys.argv = ["evaluate", "--out-dir", os.path.join(tmp, "metrics"),
                    "--fig-dir", os.path.join(tmp, "fig")]
        evaluate.main()
        sys.argv = ["analyse_errors", "--out-dir", os.path.join(tmp, "analysis"),
                    "--fig-dir", os.path.join(tmp, "fig")]
        analyse_errors.main()
        sys.argv = ["guarded_demo", "--out", os.path.join(tmp, "guarded", "demo_cases.json")]
        guarded_demo.main()
        args = type("A", (), {"out_dir": os.path.join(tmp, "capacity"),
                              "fig_dir": os.path.join(tmp, "fig")})()
        capacity.score(args)


def recorded_and_new(tmp):
    pairs = {
        "results/metrics/evaluation.json": os.path.join(tmp, "metrics", "evaluation.json"),
        "results/guarded/demo_cases.json": os.path.join(tmp, "guarded", "demo_cases.json"),
        "results/capacity/capacity_results.json": os.path.join(tmp, "capacity", "capacity_results.json"),
        "results/capacity/capacity_table.csv": os.path.join(tmp, "capacity", "capacity_table.csv"),
    }
    for name in sorted(os.listdir(os.path.join(HERE, "results", "analysis"))):
        pairs[f"results/analysis/{name}"] = os.path.join(tmp, "analysis", name)
    out = {}
    for rec, new in pairs.items():
        reader = csv_leaves if rec.endswith(".csv") else (
            lambda p: leaves(json.load(open(p, encoding="utf-8"))))
        out[rec] = worst(reader(os.path.join(HERE, rec)), reader(new), profile_of(rec))
    return out


def main():
    import torch
    torch.set_num_threads(2)
    report = {"what": __doc__.split("\n\n")[1].replace("\n", " "), "perturbations": {}}
    for k in (1, 3):
        snn.predict = perturbing_predict(k, np.random.default_rng(1000 + k))
        with tempfile.TemporaryDirectory() as tmp:
            run_outputs(tmp)
            report["perturbations"][f"{k}_ulp"] = recorded_and_new(tmp)
        snn.predict = _ORIGINAL_PREDICT
    os.makedirs(os.path.join(HERE, "results", "reproducibility"), exist_ok=True)
    with open(os.path.join(HERE, "results", "reproducibility", "ulp_sensitivity.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    for k, files in report["perturbations"].items():
        print(f"--- every prediction moved by {k}")
        for f, w in files.items():
            print(f"  {f:46s} max abs {w['max_abs']:.2e}  max rel {w['max_rel']:.2e}  "
                  f"{w['profile']:>8}: {100 * w['largest_fraction_of_tolerance_used']:5.1f} % of "
                  f"tolerance used")


if __name__ == "__main__":
    main()
