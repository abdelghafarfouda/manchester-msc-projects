"""The guarded predictor on a fixed set of cases, next to what the raw network says.

Run:  python scripts/guarded_demo.py [--out PATH]
Writes results/guarded/demo_cases.json (statuses are compared exactly in CI).

Each case records the inputs, the route the guarded predictor took
(``status``, ``phase``, ``method``), its ``F_V``, the reference solver's ``F_V``
where the state is two-phase, and -- for every input the network can be fed at
all -- what the bare network returns.  The bare network answers every one of
them with a plausible-looking fraction, including the single-phase states;
that is the hazard the guard removes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components as C  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import predict as P  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cases():
    z_ex = C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum()
    pb, pd = (float(v[0]) for v in flash.wilson_saturation_pressures(
        z_ex[None], 610.0, C.TC_RANKINE, C.PC_PSIA, C.OMEGA))
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    z_ood, p_ood, T_ood = blob["z_ood"][0], float(blob["p_ood_psia"][0]), float(blob["T_ood_R"][0])
    z_pure = np.zeros(7)
    z_pure[1] = 1.0
    p_sat_c1 = float(C.PC_PSIA[1] * np.exp(5.37 * (1 + C.OMEGA[1]) * (1 - C.TC_RANKINE[1] / 640.0)))
    z_co2 = np.array([0.60, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10])
    return [
        ("notes p. 15 flash: 796.43821 psia, 180 F", z_ex, 796.43821, 640.0, {}),
        ("notes p. 14 mixture at its bubble point, 150 F", z_ex, pb, 610.0, {}),
        ("notes p. 14 mixture at its dew point, 150 F", z_ex, pd, 610.0, {}),
        ("p. 14 mixture at 2500 psia, above its bubble point (liquid)", z_ex, 2500.0, 610.0, {}),
        ("p. 14 mixture at 1 psia, below its dew point (vapour)", z_ex, 1.0, 610.0, {}),
        ("two-phase at 2667 psia (first extrapolation-set row), default", z_ood, p_ood, T_ood, {}),
        ("the same state, on_unsupported='status'", z_ood, p_ood, T_ood, {"on_unsupported": "status"}),
        ("the same state, on_unsupported='extrapolate'", z_ood, p_ood, T_ood,
         {"on_unsupported": "extrapolate"}),
        ("p. 15 state at 700 R, above the training temperatures", z_ex, 796.43821, 700.0, {}),
        ("CO2-rich mixture, z_CO2 = 0.60 above the training range", z_co2, 300.0, 640.0, {}),
        ("pure C1 at its Wilson vapour pressure, 640 R (degenerate)", z_pure, p_sat_c1, 640.0, {}),
        ("C1/nC10 binary of notes pp. 16-18", np.array([0.6, 0.4]), 1000.0, 620.0, {}),
        ("composition summing to 0.98", z_ex * 0.98, 796.43821, 640.0, {}),
        ("temperature given in degrees F", z_ex, 796.43821, 180.0, {"temperature_unit": "F"}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "guarded", "demo_cases.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    s = P.FlashSurrogate()
    out = []
    for name, z, p, T, kw in cases():
        rec = {"case": name, "inputs": {"z": [float(v) for v in z], "p_psia": p, "T_R": T,
                                        "options": kw}}
        try:
            r = s.predict(z, p, T, **kw)
        except P.InputError as exc:
            rec.update(status="rejected", message=str(exc))
            out.append(rec)
            continue
        rec.update({k: v for k, v in r.as_dict().items()
                    if k in ("status", "phase", "method", "FV", "in_training_domain",
                             "domain_violations", "model", "message")})
        rec["p_bubble_psia"], rec["p_dew_psia"] = r.p_bubble_psia, r.p_dew_psia
        K = flash.wilson_k(p, T, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)[None, :]
        ref = flash.solve_fv(z[None, :], K)[0][0]
        rec["reference_solver_FV"] = None if np.isnan(ref) else float(ref)
        rec["bare_network_FV"] = float(s._network(z[None, :], np.array([p]), np.array([T]))[0])
        out.append(rec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"default_model": s.model_name, "cases": out}, fh, indent=2)
    w = max(len(r["case"]) for r in out)
    for r in out:
        fv = r.get("FV")
        print(f"{r['case']:<{w}}  {r['status']:<11} {r.get('method', ''):<26} "
              f"F_V = {'-' if fv is None else f'{fv:.4f}':>6}   bare network "
              f"{r.get('bare_network_FV', float('nan')):.4f}")


if __name__ == "__main__":
    main()
