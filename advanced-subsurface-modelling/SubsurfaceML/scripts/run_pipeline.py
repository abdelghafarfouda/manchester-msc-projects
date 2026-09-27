#!/usr/bin/env python
"""SubsurfaceML - one command for the complete workflow.

    python scripts/run_pipeline.py --config config/demo.yaml
    python scripts/run_pipeline.py --config config/study.yaml

Stages can be skipped while iterating, e.g. ``--skip-validation`` (reuses
validation.json) or ``--skip-dataset`` (reuses the saved CSVs).  Every stage
writes its results to ``paths.metrics``; the report is built from those files.
A log of the run is written to ``paths.reports/run.log``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="reuse results/<name>/data/scenarios_features.csv")
    ap.add_argument("--skip-studies", action="store_true")
    ap.add_argument("--skip-optimisation", action="store_true")
    args = ap.parse_args()

    from subsurfaceml import pipeline as P
    from subsurfaceml.config import ConfigError, load_config, provenance
    from subsurfaceml.features import engineer

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2
    cfg.paths.mkdirs()
    log = open(Path(cfg.paths.reports) / "run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log)
    stage_t = {}
    t_start = time.perf_counter()
    print("=" * 74)
    print(f"SubsurfaceML pipeline - config '{cfg.name}'  ->  {cfg.paths.results}")
    print("=" * 74)
    summary = {"config_name": cfg.name, "config_file": str(args.config),
               "config": cfg.to_dict(), "environment": provenance(),
               "started": time.ctime()}

    def timed(name, fn, *a):
        t0 = time.perf_counter()
        r = fn(*a)
        stage_t[name] = time.perf_counter() - t0
        return r

    # 1 validation ------------------------------------------------------------
    vpath = Path(cfg.paths.metrics) / "validation.json"
    if args.skip_validation and vpath.exists():
        validation = json.loads(vpath.read_text())
        print("\n=== STAGE 1  skipped (reusing validation.json) ===")
    else:
        validation = timed("validation", P.stage_validate, cfg)
    summary["validation_verdict"] = validation["verdict"]
    summary["validation_wall_s"] = validation.get("wall_time_s")
    if not validation["verdict"]["ALL_PASS"]:
        print("\n!! numerical validation FAILED - refusing to generate training "
              "data from an unvalidated simulator")
        (Path(cfg.paths.metrics) / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        return 2

    # 2 dataset ---------------------------------------------------------------
    dpath = Path(cfg.paths.data) / "scenarios_features.csv"
    if args.skip_dataset and dpath.exists():
        df, _ = P.deduplicate(engineer(pd.read_csv(Path(cfg.paths.data) / "scenarios.csv")))
        prov = json.loads((Path(cfg.paths.data) / "provenance.json").read_text())
        dq = json.loads((Path(cfg.paths.metrics) / "data_quality.json").read_text())
        print(f"\n=== STAGE 2  skipped (reusing {len(df)} scenarios) ===")
    else:
        d = timed("dataset", P.stage_dataset, cfg)
        df, prov, dq = d["df"], d["provenance"], d["data_quality"]
    summary["dataset"] = {k: prov[k] for k in
                          ("n_requested", "n_success", "n_failed", "wall_time_s",
                           "cpu_seconds_simulation", "mean_sim_seconds", "data_nature")}
    summary["data_quality"] = dq

    # 3-5 learning --------------------------------------------------------------
    ml = timed("surrogates", P.stage_ml, cfg, df)
    summary["ml"] = P._clean(ml)
    summary["uncertainty"] = timed("uncertainty", P.stage_uncertainty, cfg, df, ml, validation)
    summary["classifier"] = P._clean(timed("classifier", P.stage_classifier, cfg, df, ml))
    summary["models_manifest"] = P.save_models(cfg, ml)

    # 6 cost ------------------------------------------------------------------
    summary["speed"] = timed("speed", P.stage_speed, cfg, df, ml)
    summary["speed"]["dataset_generation_wall_seconds"] = prov["wall_time_s"]
    summary["speed"]["dataset_generation_cpu_seconds"] = prov["cpu_seconds_simulation"]
    summary["speed"]["surrogate_training_seconds"] = stage_t["surrogates"]

    # 7-8 interpretation and studies -----------------------------------------------
    summary["interpretation"] = P._clean(timed("interpretation", P.stage_interpret, cfg, df, ml))
    if cfg.ml.run_studies and not args.skip_studies:
        summary["studies"] = timed("studies", P.stage_studies, cfg, df, ml)
    else:
        summary["studies"] = {"skipped": True}

    # 9-10 physics checks and screening ----------------------------------------
    summary["conservation_check"] = P.stage_conservation_check(cfg, df)
    if args.skip_optimisation:
        summary["optimisation"] = {"skipped": True}
    else:
        summary["boundary_comparison"] = timed("open_boundary", P.stage_boundary_comparison, cfg)
        summary["optimisation"] = timed("screening", P.stage_optimise, cfg, df, ml)

    summary["stage_seconds"] = stage_t
    summary["total_wall_seconds"] = time.perf_counter() - t_start
    out = Path(cfg.paths.metrics) / "summary.json"
    out.write_text(json.dumps(P._clean(summary), indent=2, default=P._json_default))

    from subsurfaceml.report import write_results_table
    tbl = write_results_table(cfg, json.loads(out.read_text()))
    print("\n" + tbl)
    print(f"\nTotal wall time {summary['total_wall_seconds']/60:.1f} min")
    print(f"Summary: {out}\nFigures: {cfg.paths.figures}\nModels:  {cfg.paths.models}")
    sys.stdout = sys.__stdout__
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
