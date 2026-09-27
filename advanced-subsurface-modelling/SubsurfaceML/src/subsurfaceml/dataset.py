"""Scenario dataset generation: run the simulator over the sampled prior and
write a fully traceable tabular + time-series dataset.

Outputs (all under ``results/data``)
------------------------------------
``scenarios.csv``      one row per successful scenario (tabular ML targets)
``timeseries.csv``     one row per (scenario, report time) -- trajectories for plots
``failed_runs.csv``    every scenario that failed, with the error text
``provenance.json``    config, seeds, package versions, timings, sampling ranges
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config, provenance
from .scenarios import sample_realisations, sample_schedules, run_scenario
from .units import MPA, YEAR


def _one(cfg, r, s):
    return run_scenario(cfg, r, s)


def generate_dataset(cfg: Config, *, verbose: bool = True) -> dict:
    """Run every (realisation, schedule) pair and persist the dataset."""
    cfg.paths.mkdirs()
    reals = sample_realisations(cfg)
    jobs = []
    for r in reals:
        for s in sample_schedules(cfg, r):
            jobs.append((r, s))
    if verbose:
        print(f"[dataset] {len(reals)} realisations x "
              f"{cfg.scenarios.n_schedules_per_realisation} schedules "
              f"= {len(jobs)} simulations")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5 if verbose else 0,
                       batch_size=4)(
        delayed(_one)(cfg, r, s) for r, s in jobs)
    wall = time.perf_counter() - t0

    rows, series, failed = [], [], []
    for res in results:
        if res["status"] == "ok":
            rows.append(res["row"])
            series.append(res["series"])
        else:
            failed.append(res)

    df = pd.DataFrame(rows)
    ts = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
    fdf = pd.DataFrame(failed)

    d = cfg.paths.data
    df.to_csv(Path(d) / "scenarios.csv", index=False)
    ts.to_csv(Path(d) / "timeseries.csv", index=False)
    fdf.to_csv(Path(d) / "failed_runs.csv", index=False)

    prov = {
        "generated_by": "subsurfaceml.dataset.generate_dataset",
        "data_nature": "SYNTHETIC - produced by the in-repo IMPES simulator; "
                       "no field data, no field validation",
        "config_name": cfg.name,
        "config": cfg.to_dict(),
        "environment": provenance(),
        "master_seed": cfg.scenarios.seed,
        "n_requested": len(jobs),
        "n_success": int(len(rows)),
        "n_failed": int(len(failed)),
        "wall_time_s": wall,
        "cpu_seconds_simulation": float(df["wall_time_s"].sum()) if len(df) else 0.0,
        "mean_sim_seconds": float(df["wall_time_s"].mean()) if len(df) else 0.0,
        "sampling_ranges": {k: list(v) for k, v in
                            cfg.scenarios.__dict__.items()
                            if isinstance(v, (tuple, list))},
        "realisation_seeds": {int(r.realisation_id): int(r.seed) for r in reals},
        "units": {
            "dp_bh_max_Pa": "Pa", "r_plume_end_m": "m",
            "mass_retained_kg": "kg", "mass_injected_kg": "kg",
            "k_median_mD": "mD", "h_total_m": "m", "r_e_m": "m",
            "q*_kg_s": "kg/s", "t_s": "s",
        },
    }
    (Path(d) / "provenance.json").write_text(json.dumps(prov, indent=2,
                                                        default=str))
    if verbose:
        print(f"[dataset] {len(rows)} ok, {len(failed)} failed, "
              f"{wall:.1f} s wall, {prov['cpu_seconds_simulation']:.1f} s CPU")
    return {"scenarios": df, "timeseries": ts, "failed": fdf,
            "provenance": prov}


# --------------------------------------------------------------------------
# Feature engineering lives in ONE place (features.py); these names are kept
# for backwards compatibility with the notebooks and older scripts.
from .features import FEATURES, engineer as add_engineered_features  # noqa: E402,F401
