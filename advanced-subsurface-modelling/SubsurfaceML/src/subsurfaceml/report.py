"""Assemble RESULTS.md from ``summary.json`` -- formatting only.

Nothing here computes a result.  If a stage did not run, its rows say so.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "n/a"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{nd}g}"
    return str(x)


def build_rows(s: dict) -> list:
    rows = []
    add = lambda sec, m, v, n="": rows.append((sec, m, v, n))
    v = s.get("validation_verdict", {})
    n_ok = sum(1 for k, x in v.items() if k != "ALL_PASS" and x)
    add("Simulator verification", "All checks", _f(v.get("ALL_PASS")),
        f"{n_ok}/{len(v) - 1} checks pass (validation.json)")
    for k, x in v.items():
        if k != "ALL_PASS":
            add("Simulator verification", k.replace("_", " "), _f(x))
    d = s.get("dataset", {})
    add("Dataset", "Scenarios", f"{_f(d.get('n_success'))} ok / {_f(d.get('n_failed'))} failed",
        d.get("data_nature", ""))
    add("Dataset", "Generation cost", f"{_f(d.get('wall_time_s'))} s wall, "
        f"{_f(d.get('cpu_seconds_simulation'))} s CPU", f"mean {_f(d.get('mean_sim_seconds'))} s per simulation")
    for t, r in _g(s, "ml", "targets", default={}).items():
        u = r["unit"]
        te, b, eb = r["test"], r["mean_baseline_test"], r["error_band"]
        add("Surrogates (unseen reservoirs)", f"{t}: selected model", r["selected_model"],
            "lowest grouped-CV RMSE among the course model families")
        add("Surrogates (unseen reservoirs)", f"{t}: MAE / RMSE / R2",
            f"{_f(te['MAE'])} / {_f(te['RMSE'])} {u} / {_f(te['R2'], 4)}",
            f"mean-baseline RMSE {_f(b['RMSE'])} {u}; n = {te['n']}")
        add("Surrogates (unseen reservoirs)", f"{t}: P5-P95 error band",
            f"{_f(100 * eb['measured_coverage_test'])}% measured coverage",
            f"mean width {_f(eb['mean_width'])} {u}; calibrated on {eb['n_calibration']} rows; no guarantee")
    b = _g(s, "classifier", "binary", default={})
    if b:
        ts = b["test_selected_threshold"]
        add("Pressure-limit screen", "selected / test ROC-AUC",
            f"{b['selected']} / {_f(ts.get('roc_auc'))}", f"limit dp > {_f(b['dp_limit_MPa'])} MPa")
        add("Pressure-limit screen", "recall / precision at chosen threshold",
            f"{_f(ts['recall'])} / {_f(ts['precision'])}",
            f"threshold chosen on out-of-fold scores for recall >= {b['target_recall']}")
        tl = _g(s, "classifier", "traffic_light", default={})
        add("Pressure-limit screen", "traffic light macro-F1 (test)",
            _f(_g(tl, "test", "macro_f1")),
            f"regression surrogate banded: {_f(_g(tl, 'test_regression_surrogate_banded', 'macro_f1'))}")
    sp = s.get("speed", {})
    if sp:
        add("Cost", "simulator per case (serial)", f"{_f(sp['simulator_seconds_per_case_mean_serial'])} s",
            f"{sp['simulator_cases_timed']} test scenarios re-run in-process")
        add("Cost", "surrogate batched / single call / single end-to-end",
            f"{_f(sp['surrogate_seconds_per_case_batched'])} / {_f(sp['surrogate_seconds_single_call_median'])} / "
            f"{_f(sp['surrogate_seconds_single_end_to_end_median'])} s",
            "3 QoIs; end-to-end includes feature building")
        add("Cost", "speed-up batched / single / end-to-end",
            f"{_f(sp['speedup_batched'])} / {_f(sp['speedup_single_call'])} / {_f(sp['speedup_single_end_to_end'])}",
            f"excludes {_f(sp['dataset_generation_wall_seconds'])} s data generation and "
            f"{_f(sp['surrogate_training_seconds'])} s training")
    o = s.get("optimisation", {})
    if o and not o.get("skipped"):
        add("Schedule screening", "median improvement vs constant rate",
            f"{_f(o.get('median_improvement_pct'))}%",
            f"{o['n_realisations_improved']}/{o['n_realisations_compared']} unseen reservoirs improved (re-simulated)")
        add("Schedule screening", "screened schedules violating a limit",
            f"{o['n_screened_violating']}/{o['n_screened_resimulated']}",
            f"baselines violating: {o['n_baselines_violating']}; failed re-simulations: {o['n_failed_resimulations']}")
    c = s.get("conservation_check", {})
    if c:
        add("Physics checks", "retained vs planned mass", _f(c.get("max_rel_error_retained_vs_planned")),
            "max relative difference, sealed boundary")
    bc = s.get("boundary_comparison", {})
    if bc:
        add("Physics checks", "open boundary retention (median)", _f(bc.get("median_retention_fraction")),
            f"{bc.get('n_with_loss')}/{bc.get('n_runs')} runs lost CO2 across r_e")
    return rows


def write_results_table(cfg, summary: dict) -> str:
    rows = build_rows(summary)
    lines = [f"# SubsurfaceML results - configuration `{cfg.name}`", "",
             f"Generated {time.ctime()} by `scripts/run_pipeline.py` from "
             f"`metrics/summary.json`. Total wall time "
             f"{_f(summary.get('total_wall_seconds', 0) / 60)} min.", "",
             "All data are **synthetic**, produced by the simulator in this "
             "repository. No field data are used and no field validation is claimed. "
             "Pressure and plume limits are stated assumptions, not safety limits.", "",
             "| Section | Metric | Value | Notes |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | **{c}** | {d} |" for a, b, c, d in rows]
    txt = "\n".join(lines) + "\n"
    p = Path(cfg.paths.reports) / "RESULTS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return txt
