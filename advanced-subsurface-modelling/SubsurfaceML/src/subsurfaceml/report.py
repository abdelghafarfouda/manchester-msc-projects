"""Assemble RESULTS.md from ``summary.json`` -- formatting only.

Nothing here computes a result.  If a stage did not run, its rows say so.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .labels import MODEL_NAMES, SHORT


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


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def build_rows(s: dict) -> list:
    rows = []
    add = lambda sec, m, v, n="": rows.append((sec, m, v, n))
    v = s.get("validation_verdict", {})
    n_ok = sum(1 for k, x in v.items() if k != "ALL_PASS" and x)
    add("Verification", "All checks", _f(v.get("ALL_PASS")),
        f"{n_ok}/{len(v) - 1} checks pass (validation.json)")
    for k, x in v.items():
        if k != "ALL_PASS":
            add("Verification", k.replace("_", " "), _f(x))
    d = s.get("dataset", {})
    add("Data", "Development scenarios", f"{_f(d.get('n_success'))} ok / {_f(d.get('n_failed'))} failed",
        d.get("data_nature", ""))
    for which, p in _g(s, "evaluation_sets", "provenance", default={}).items():
        add("Data", f"{which.replace('_', ' ')} scenarios",
            f"{_f(p.get('n_success'))} ok / {_f(p.get('n_failed'))} failed",
            f"{p.get('n_realisations')} fresh reservoirs, seed {p.get('seed')}, "
            f"median k {p.get('k_median_mD_range')} mD")
    dj = _g(s, "evaluation_sets", "disjoint", default={})
    if dj:
        add("Data", "overlap between sets", f"{dj.get('overlapping_ids')} ids / "
            f"{dj.get('overlapping_descriptions')} descriptions", "must be 0 / 0")
    nm = s.get("numerics", {})
    if nm and not nm.get("skipped"):
        t = _g(nm, "targets", "dp_bh_max_MPa", "production_minus_fine", default={})
        add("Discretisation error of dataset cases", "peak build-up, production vs refined",
            f"median {_pct(t.get('median_rel'))}, max {_pct(t.get('max_rel'))}",
            f"{nm.get('n_cases')} development cases re-simulated")
        fl = nm.get("pressure_limit_label_flips", {})
        add("Discretisation error of dataset cases", "pressure-limit labels changed",
            f"{fl.get('n_flipped')}/{fl.get('n_compared')}",
            f"{fl.get('n_within_5pct_of_limit')} cases within 5 % of the limit")
        for tt in ("r_plume_m95_m", "sweep_efficiency"):
            t = _g(nm, "targets", tt, "production_minus_fine", default={})
            add("Discretisation error of dataset cases", SHORT[tt],
                f"median {_pct(t.get('median_rel'))}, max {_pct(t.get('max_rel'))}")
    for t, r in _g(s, "ml", "targets", default={}).items():
        u = r["unit"]
        for dname, lab in (("revised", "revised"), ("published_approach", "published approach, retrained"),
                           ("published_models_as_released", "published models as released")):
            dd = _g(r, "designs", dname)
            if not dd:
                continue
            for sset in ("test", "shift"):
                e = _g(dd, "evaluation", sset)
                if not e:
                    continue
                p = e["point"]
                meth = e.get("selected_interval_method")
                iv = _g(e, "intervals", meth, default={})
                add(f"{SHORT.get(t, t)} - {'test' if sset == 'test' else 'shift (10-30 mD)'}",
                    f"{lab} ({MODEL_NAMES.get(dd['selected_model'], dd['selected_model'])})",
                    f"RMSE {_f(p['RMSE'])} {u}",
                    f"MAE {_f(p['MAE'])}, R2 {_f(p['R2'], 4)}, worst under-prediction "
                    f"{_f(p['worst_underprediction'])} {u}, P95 |error| {_f(p['p95_abs_error'])}; "
                    f"{meth} interval: case coverage {_pct(iv.get('case_coverage'))}, "
                    f"reservoirs fully covered {_pct(iv.get('reservoir_all_covered'))}, "
                    f"mean width {_f(iv.get('mean_width'))} {u}")
        for key, bt in r.get("bootstrap", {}).items():
            rm = bt.get("RMSE", {})
            add(f"{SHORT.get(t, t)} - paired bootstrap", key.replace("_", " "),
                f"dRMSE {_f(rm.get('difference_b_minus_a'))} {u}",
                f"95 % CI {[round(x, 4) for x in rm.get('ci95', [])]}, resampling "
                f"{bt.get('n_reservoirs')} reservoirs")
    b = _g(s, "classifier", "binary", default={})
    if b:
        ts = b["test_selected_threshold"]
        add("Pressure-limit screen (test)", "classifier / ROC-AUC",
            f"{b['selected']} / {_f(ts.get('roc_auc'))}", f"limit: build-up > {_f(b['dp_limit_MPa'])} MPa")
        add("Pressure-limit screen (test)", "recall / precision at chosen threshold",
            f"{_f(ts['recall'])} / {_f(ts['precision'])}",
            f"threshold from out-of-fold scores for recall >= {b['target_recall']}")
        rb = b.get("test_regression_upper_bound", {})
        add("Pressure-limit screen (test)", "interval upper edge: missed exceedances",
            f"{rb.get('missed_exceedances')}/{rb.get('n_exceeding')}",
            f"false alarms {rb.get('false_alarms')}")
    sp = s.get("speed", {})
    if sp:
        add("Cost", "simulator per case (serial)", f"{_f(sp['simulator_seconds_per_case_mean_serial'])} s",
            f"{sp['simulator_cases_timed']} test scenarios re-run in-process")
        add("Cost", "surrogates, single case end to end",
            f"{_f(sp['surrogate_seconds_single_end_to_end_median'])} s",
            f"speed-up {_f(sp['speedup_single_end_to_end'])}x; batched "
            f"{_f(sp['surrogate_seconds_per_case_batched'])} s/case")
    sc = s.get("screening", {})
    for which, d in (sc.get("by_set") or {}).items():
        for meth, m in d["methods"].items():
            add(f"Screening - {which.replace('_', ' ')}", meth.replace("_", " "),
                f"{m['n_verified']}/{d['n_reservoirs']} verified",
                f"first simulated proposal violated: {m['first_simulated_schedule_violates']}; "
                f"no feasible found: {m['n_no_feasible_found']}; median verified mass vs "
                f"simulator-only constant rate {_f(m['median_mass_vs_constant_pct'])} %; "
                f"mean simulations {_f(m['mean_simulations'])}")
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
             "Pressure and plume limits are stated modelling assumptions, not safety limits. "
             "'test' = fresh reservoirs from the training prior, untouched until this run; "
             "'shift' = fresh reservoirs with median permeability 10-30 mD, below the "
             "training range.", "",
             "| Section | Metric | Value | Notes |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | **{c}** | {d} |" for a, b, c, d in rows]
    txt = "\n".join(lines) + "\n"
    p = Path(cfg.paths.reports) / "RESULTS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return txt
