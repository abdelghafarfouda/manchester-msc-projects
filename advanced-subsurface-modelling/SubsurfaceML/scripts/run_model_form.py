#!/usr/bin/env python
"""Model-form error of the layered, no-gravity, no-crossflow simulator.

    python scripts/run_model_form.py --config config/study.yaml

A stratified sample of development cases (permeability tercile x schedule
intensity quartile) is re-run with the r-z reference model of
``src/subsurfaceml/rz.py``:

``gravity_crossflow``  buoyancy + vertical crossflow, k_v/k_h = 0.1, 5 rows per layer
``crossflow_only``     the same without gravity (attributes the change)
``gravity_fine``       gravity + crossflow with 10 rows per layer (a subset;
                       checks the vertical resolution)

and the change of each machine-learning target relative to the dataset value
(the layered model) is reported, together with the number of cases whose
pressure-limit label would change.  The r-z model is verified in
``tests/test_rz.py`` (reduction to the layered model, hydrostatic
equilibrium, gravity segregation to the analytical final state, CO2 mass
conservation).  k_v/k_h = 0.1 is an assumed value, not a measured one.
Writes ``results/<name>/experiments/model_form_*.{csv,json}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

VARIANTS = {"gravity_crossflow": dict(n_sub=5, kv_over_kh=0.1, gravity=9.80665),
            "crossflow_only": dict(n_sub=5, kv_over_kh=0.1, gravity=0.0),
            "gravity_fine": dict(n_sub=10, kv_over_kh=0.1, gravity=9.80665)}
TARGETS = ("dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency")


def _run(cfg, r, rates, sid, name, kw):
    from subsurfaceml.rz import run_rz_scenario
    try:
        o = run_rz_scenario(cfg, r, rates, **kw)
        return {"scenario_id": sid, "variant": name, "status": "ok", **o}
    except Exception as exc:                                   # pragma: no cover
        return {"scenario_id": sid, "variant": name, "status": "failed", "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "study.yaml"))
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--fine-cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    from subsurfaceml.config import load_config, provenance
    from subsurfaceml.numerics import select_cases
    from subsurfaceml.pipeline import _json_default
    from subsurfaceml.scenarios import sample_realisations
    cfg = load_config(a.config)
    out = Path(cfg.paths.results) / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(Path(cfg.paths.data) / "scenarios_features.csv")
    cases = select_cases(dev, n_per_cell=a.per_cell, seed=a.seed,
                         always=["R0036_S+02", "R0172_S+00"])
    fine = cases.sort_values("k_median_mD").iloc[
        np.unique(np.linspace(0, len(cases) - 1, a.fine_cases).astype(int))]
    reals = {r.realisation_id: r for r in sample_realisations(cfg)}
    qcols = [f"q{i + 1}_kg_s" for i in range(cfg.schedule.n_periods)]
    jobs = []
    for name, kw in VARIANTS.items():
        for _, row in (fine if name == "gravity_fine" else cases).iterrows():
            jobs.append((reals[int(row.realisation_id)], row[qcols].to_numpy(float),
                         row.scenario_id, name, kw))
    t0 = time.perf_counter()
    res = Parallel(n_jobs=cfg.n_jobs, batch_size=1)(
        delayed(_run)(cfg, r, q, sid, name, kw) for r, q, sid, name, kw in jobs)
    runs = pd.DataFrame(res)
    runs.to_csv(out / "model_form_runs.csv", index=False)
    ok = runs[runs.status == "ok"]
    ref = cases.set_index("scenario_id")
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    rep = {"command": " ".join(sys.argv), "environment": provenance(),
           "n_cases": int(len(cases)), "n_failed": int((runs.status != "ok").sum()),
           "variants": VARIANTS, "kv_over_kh_status": "assumed value (0.1)",
           "seconds": time.perf_counter() - t0, "targets": {}, "labels": {}}
    for name in VARIANTS:
        g = ok[ok.variant == name].set_index("scenario_id")
        if g.empty:
            continue
        rep["targets"][name] = {}
        for t in TARGETS:
            rel = (g[t] - ref.loc[g.index, t]) / ref.loc[g.index, t]
            rep["targets"][name][t] = {
                "median_rel_change": float(rel.median()),
                "p10_rel_change": float(rel.quantile(0.1)),
                "p90_rel_change": float(rel.quantile(0.9)),
                "min_rel_change": float(rel.min()), "max_rel_change": float(rel.max()),
                "n": int(len(rel))}
        lab0 = ref.loc[g.index, "dp_bh_max_MPa"] > dp_lim
        lab1 = g["dp_bh_max_MPa"] > dp_lim
        rep["labels"][name] = {"n_compared": int(len(g)),
                               "n_exceeding_layered": int(lab0.sum()),
                               "n_exceeding_rz": int(lab1.sum()),
                               "n_label_changes": int((lab0 != lab1).sum())}
        rep["targets"][name]["co2_fraction_in_top_quarter_median"] = float(
            g["co2_fraction_in_top_quarter"].median())
        rep["targets"][name]["max_mass_balance_error"] = float(g["mass_balance_error"].max())
    # relation of the plume-radius change to permeability
    gc = ok[ok.variant == "gravity_crossflow"].set_index("scenario_id")
    if len(gc):
        k = ref.loc[gc.index, "k_median_mD"]
        rel = (gc["r_plume_m95_m"] - ref.loc[gc.index, "r_plume_m95_m"]) / ref.loc[gc.index, "r_plume_m95_m"]
        rep["plume_change_by_k_tercile"] = (
            pd.DataFrame({"k": k, "rel": rel})
            .assign(k_group=lambda d: pd.qcut(d.k, 3, labels=["low k", "mid k", "high k"]))
            .groupby("k_group", observed=True).rel.agg(["median", "min", "max", "size"])
            .reset_index().astype({"k_group": str}).to_dict("records"))
    (out / "model_form_summary.json").write_text(json.dumps(rep, indent=2, default=_json_default))
    _figure(cfg, cases, ok, out)
    print(json.dumps({k: rep[k] for k in ("targets", "labels")}, indent=1, default=_json_default))
    return 0


def _figure(cfg, cases, ok, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from subsurfaceml.labels import label
    ref = cases.set_index("scenario_id")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    cols = {"gravity_crossflow": "#1f6f8b", "crossflow_only": "#edae49",
            "gravity_fine": "#d1495b"}
    names = {"gravity_crossflow": "gravity + crossflow", "crossflow_only": "crossflow only",
             "gravity_fine": "gravity + crossflow, 2x vertical cells"}
    for a, t in zip(ax, TARGETS):
        for v, g in ok.groupby("variant"):
            g = g.set_index("scenario_id")
            a.scatter(ref.loc[g.index, "k_median_mD"],
                      100 * (g[t] - ref.loc[g.index, t]) / ref.loc[g.index, t],
                      s=16, color=cols[v], label=names[v], alpha=.85)
        a.set_xscale("log"); a.axhline(0, color="k", lw=1)
        a.set_xlabel(label("k_median_mD"))
        a.set_ylabel(f"r-z minus layered model [%]")
        a.set_title(label(t).split(" [")[0], fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("Model-form error: change of each target when buoyancy and vertical "
                 "crossflow are added (k$_v$/k$_h$ = 0.1, assumed)", y=1.03)
    fig.tight_layout()
    fig.savefig(Path(cfg.paths.figures) / "19_model_form_error.png", bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
